# ROI-aware training for Post-Contrast prediction

import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from accelerate import Accelerator
from diffusers import DDPMScheduler
from torch_ema import ExponentialMovingAverage

# --- project imports ---
from data_loading import config, train_loader, test_loader, test_dataset  # loaders must yield {"pre_img","post_img","tumor_mask"}
from models.conditional_unet import ConditionalUNet
from losses import PerceptualLoss, total_variation_loss
from metrics import evaluate_batch  # returns dict with "MAE" and "MSE"

# ============== TRAINING SETUP ==============
model = ConditionalUNet(in_channels=2).to("cuda")
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
noise_scheduler = DDPMScheduler(num_train_timesteps=config.num_train_timesteps, beta_schedule="squaredcos_cap_v2")
perceptual_loss_fn = PerceptualLoss().to("cuda")  # must expose extract_features(x)
ema = ExponentialMovingAverage(model.parameters(), decay=0.999)
accelerator = Accelerator(mixed_precision=config.mixed_precision)

model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
torch.manual_seed(config.seed)
os.makedirs(config.best_model_dir, exist_ok=True)

best_loss = float("inf")
all_metrics = []
loss_components = {"Epoch": [], "MAE": [], "Percep": [], "TV": [], "MSE": [], "Total": []}

# ============== TRAINING LOOP ==============
for epoch in range(config.num_epochs):
    model.train()
    total_loss = total_mae = total_percep = total_tv = total_mse = 0.0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}"):
        pre_img = batch["pre_img"].to(accelerator.device)
        post_img = batch["post_img"].to(accelerator.device)
        tumor_mask = batch["tumor_mask"].to(accelerator.device)

        # DDPM forward
        noise = torch.randn_like(post_img)
        t = torch.randint(0, config.num_train_timesteps, (post_img.size(0),), device=post_img.device).long()
        noisy_post = noise_scheduler.add_noise(post_img, noise, t)
        input_concat = torch.cat([noisy_post.clamp(-1, 1), pre_img], dim=1)

        with accelerator.accumulate(model):
            x0_pred = model(input_concat).clamp(-1, 1)

            tumor_mask_bin = tumor_mask.bool()
            x0_roi = x0_pred * tumor_mask
            post_roi = post_img * tumor_mask
            pre_roi = pre_img * tumor_mask

            # --- GLOBAL LOSSES ---
            loss_mae_global = F.l1_loss(x0_pred, post_img)
            loss_mse_global = F.mse_loss(x0_pred, post_img)
            loss_tv_global = total_variation_loss(x0_pred)

            # --- ROI LOSSES ---
            loss_mae_roi = F.l1_loss(x0_roi, post_roi)
            loss_mse_roi = F.mse_loss(x0_roi, post_roi)
            loss_tv_roi = total_variation_loss(x0_roi)

            # --- Perceptual features ---
            feat_pred = perceptual_loss_fn.extract_features(x0_pred)
            feat_target = perceptual_loss_fn.extract_features(post_img)
            loss_percep_global = F.mse_loss(feat_pred, feat_target)

            # --- Perceptual ROI loss (safe) ---
            mask_resized = F.interpolate(
                tumor_mask, size=feat_pred.shape[-2:], mode='bilinear', align_corners=False
            ).clamp(0, 1)
            if torch.isnan(feat_pred).any() or torch.isnan(feat_target).any():
                loss_percep_roi = torch.tensor(0.0, device=accelerator.device)
            else:
                loss_percep_roi = F.mse_loss(feat_pred * mask_resized, feat_target * mask_resized)

            # --- Contrast difference loss ---
            contrast_pred = F.relu(x0_roi - pre_roi)
            contrast_target = F.relu(post_roi - pre_roi)
            loss_contrast_roi = F.l1_loss(contrast_pred, contrast_target)

            # --- Intensity loss inside ROI (safe mean) ---
            if tumor_mask_bin.sum() > 0:
                mean_intensity_pred = x0_pred[tumor_mask_bin].mean()
                mean_intensity_target = post_img[tumor_mask_bin].mean()
                loss_intensity_roi = F.l1_loss(mean_intensity_pred, mean_intensity_target)
            else:
                loss_intensity_roi = torch.tensor(0.0, device=accelerator.device)

            # === Weighted combination ===
            loss_global = (
                0.3 * loss_mae_global +
                0.6 * loss_percep_global +
                0.15 * loss_tv_global +
                0.05 * loss_mse_global
            )
            loss_roi = (
                0.3 * loss_mae_roi +
                0.6 * loss_percep_roi +
                0.15 * loss_tv_roi +
                0.05 * loss_mse_roi
            )
            total = (
                0.3 * loss_global +
                0.6 * loss_roi +
                0.05 * loss_contrast_roi +
                0.05 * loss_intensity_roi
            )

            # NaN protection
            if torch.isnan(total).any():
                accelerator.print("Skipping batch due to NaN loss.")
                continue

            accelerator.backward(total)
            optimizer.step()
            optimizer.zero_grad()
            ema.update()

        total_loss += total.item()
        total_mae += loss_mae_global.item()
        total_percep += loss_percep_global.item()
        total_tv += loss_tv_global.item()
        total_mse += loss_mse_global.item()

    # --- Log epoch losses ---
    loss_components["Epoch"].append(epoch + 1)
    loss_components["MAE"].append(total_mae / len(train_loader))
    loss_components["Percep"].append(total_percep / len(train_loader))
    loss_components["TV"].append(total_tv / len(train_loader))
    loss_components["MSE"].append(total_mse / len(train_loader))
    loss_components["Total"].append(total_loss / len(train_loader))

    # ============== EVALUATION (EMA) ==============
    model.eval()
    metrics_list = []

    # Inline sampler
    def generate_sample(model_ref, title, epoch_idx):
        model_ref.eval()
        sample = test_dataset[0]
        pre_s = sample["pre_img"].unsqueeze(0).to(accelerator.device)
        post_s = sample["post_img"].unsqueeze(0).to(accelerator.device)

        x_t = torch.randn_like(pre_s)
        with torch.no_grad():
            for tt in reversed(range(config.num_train_timesteps)):
                model_input = torch.cat([x_t, pre_s], dim=1)  # no mask here
                x0_hat = model_ref(model_input).clamp(-1, 1)

                alpha_cumprod_t = noise_scheduler.alphas_cumprod[tt].to(x_t.device)
                sqrt_alpha = torch.sqrt(alpha_cumprod_t).view(1, 1, 1, 1)
                sqrt_beta = torch.sqrt(1 - alpha_cumprod_t).view(1, 1, 1, 1)
                noise = torch.randn_like(x_t) if tt > 0 else torch.zeros_like(x_t)

                x_t = (sqrt_alpha * x0_hat + sqrt_beta * noise).clamp(-1, 1)

        gen_img = (x_t + 1) / 2
        pre_plot = (pre_s.squeeze().cpu().numpy() + 1) / 2
        post_plot = (post_s.squeeze().cpu().numpy() + 1) / 2
        gen_plot = gen_img.squeeze().cpu().numpy()

        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(pre_plot, cmap="gray"); axs[0].set_title("Pre-contrast"); axs[0].axis("off")
        axs[1].imshow(post_plot, cmap="gray"); axs[1].set_title("Ground Truth"); axs[1].axis("off")
        axs[2].imshow(gen_plot, cmap="gray"); axs[2].set_title(f"{title} (Epoch {epoch_idx+1})"); axs[2].axis("off")
        plt.tight_layout(); plt.show()

    with ema.average_parameters():
        with torch.no_grad():
            for batch in test_loader:
                pre_img = batch["pre_img"].to(accelerator.device)
                post_img = batch["post_img"].to(accelerator.device)

                noise = torch.randn_like(post_img)
                t = torch.randint(0, config.num_train_timesteps, (post_img.size(0),), device=post_img.device).long()
                noisy_post = noise_scheduler.add_noise(post_img, noise, t)

                model_input = torch.cat([noisy_post.clamp(-1, 1), pre_img], dim=1)  # no mask in input
                x0_pred = model(model_input).clamp(-1, 1)

                metrics_list.append(evaluate_batch(x0_pred, post_img))

        # optional sample each epoch
        generate_sample(model, title="Generated Image", epoch_idx=epoch)

    # --- Aggregate metrics ---
    avg_metrics = {k: float(np.mean([m[k] for m in metrics_list])) for k in metrics_list[0]}
    avg_metrics["Epoch"] = epoch + 1
    all_metrics.append(avg_metrics)

    accelerator.print(f"[Epoch {epoch+1}] Loss: {total_loss / len(train_loader):.6f} | Metrics: {avg_metrics}")

    # --- Save best by MAE (EMA) ---
    if avg_metrics["MAE"] < best_loss:
        best_loss = avg_metrics["MAE"]
        with ema.average_parameters():
            torch.save(
                accelerator.unwrap_model(model).state_dict(),
                os.path.join(config.best_model_dir, "pytorch_model.bin")
            )
        accelerator.print(f"✅ Saved best EMA model at epoch {epoch+1} (MAE={best_loss:.6f})")

# ============== PLOTTING ==============
metrics_df = pd.DataFrame(all_metrics)

# Normalized trend
normalized = metrics_df.copy()
for col in ['MAE', 'MSE']:
    min_val = metrics_df[col].min()
    max_val = metrics_df[col].max()
    normalized[col] = (metrics_df[col] - min_val) / (max_val - min_val + 1e-8)

plt.figure(figsize=(10, 5))
for col in ['MAE', 'MSE']:
    sns.lineplot(data=normalized, x="Epoch", y=col, label=col)
plt.title("Normalized Metric Trends Over Epochs")
plt.xlabel("Epoch"); plt.ylabel("Normalized Value (0–1)")
plt.grid(True); plt.legend(); plt.tight_layout()
plt.show()

# Per-metric raw
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
cols = ['MAE', 'MSE']
for i, col in enumerate(cols):
    sns.lineplot(data=metrics_df, x="Epoch", y=col, ax=axs[i])
    axs[i].set_title(col); axs[i].set_xlabel("Epoch"); axs[i].set_ylabel(col); axs[i].grid(True)
plt.suptitle("Metric Evolution Per Epoch (Raw Scale)", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# Loss components
loss_df = pd.DataFrame(loss_components)
plt.figure(figsize=(12, 6))
for col in ["MAE", "Percep", "TV", "MSE", "Total"]:
    sns.lineplot(data=loss_df, x="Epoch", y=col, label=col)
plt.title("Loss Components Over Epochs")
plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.grid(True); plt.legend(); plt.tight_layout()
plt.show()
