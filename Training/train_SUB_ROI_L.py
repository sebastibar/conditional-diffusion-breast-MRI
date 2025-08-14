# ROI-aware training for Subtraction prediction (delta = post - pre)

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
from data_loading import config, train_loader, test_loader  # loaders must yield {"pre_img","post_img","tumor_mask"}
from models.conditional_unet import ConditionalUNet
from losses import PerceptualLoss, total_variation_loss
from metrics import evaluate_batch  # returns dict with "MAE" and "MSE"

# ============== TRAINING SETUP ==============
model = ConditionalUNet(in_channels=2).to("cuda")
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
noise_scheduler = DDPMScheduler(num_train_timesteps=config.num_train_timesteps, beta_schedule="squaredcos_cap_v2")
perceptual_loss_fn = PerceptualLoss().to("cuda")
ema = ExponentialMovingAverage(model.parameters(), decay=0.999)
accelerator = Accelerator(mixed_precision=config.mixed_precision)

model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
torch.manual_seed(config.seed)

best_loss = float("inf")
loss_components = {"Epoch": [], "MAE": [], "Percep": [], "TV": [], "MSE": [], "Total": []}
all_metrics = []

# ============== TRAINING LOOP ==============
for epoch in range(config.num_epochs):
    model.train()
    total_loss = total_mae = total_percep = total_tv = total_mse = 0.0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}"):
        pre = batch["pre_img"].to(accelerator.device)
        post = batch["post_img"].to(accelerator.device)
        mask = batch["tumor_mask"].to(accelerator.device)
        mask_bin = mask.bool()

        # Subtraction target
        delta = (post - pre) / 0.5

        # DDPM forward
        noise = torch.randn_like(delta)
        t = torch.randint(0, config.num_train_timesteps, (delta.size(0),), device=delta.device).long()
        noisy_delta = noise_scheduler.add_noise(delta, noise, t)

        model_input = torch.cat([noisy_delta, pre], dim=1)

        with accelerator.accumulate(model):
            pred_delta = model(model_input).clamp(-1, 1)
            recon_post = (pred_delta * 0.5 + pre).clamp(-1, 1)

            # ROI views
            recon_roi = recon_post * mask
            post_roi = post * mask
            pre_roi = pre * mask

            # --- Global losses ---
            loss_mae_global = F.l1_loss(recon_post, post)
            loss_mse_global = F.mse_loss(recon_post, post)
            loss_tv_global = total_variation_loss(recon_post)
            loss_percep_global = perceptual_loss_fn(recon_post, post)

            # --- ROI losses ---
            loss_mae_roi = F.l1_loss(recon_roi, post_roi)
            loss_mse_roi = F.mse_loss(recon_roi, post_roi)
            loss_tv_roi = total_variation_loss(recon_roi)

            # Perceptual ROI via features
            feat_pred = perceptual_loss_fn.extract_features(recon_post)
            feat_target = perceptual_loss_fn.extract_features(post)
            mask_resized = F.interpolate(mask, size=feat_pred.shape[-2:], mode='bilinear', align_corners=False)
            loss_percep_roi = F.mse_loss(feat_pred * mask_resized, feat_target * mask_resized)

            # Contrast alignment inside ROI
            contrast_pred = F.relu(recon_roi - pre_roi)
            contrast_target = F.relu(post_roi - pre_roi)
            loss_contrast = F.l1_loss(contrast_pred, contrast_target)

            # Intensity inside ROI (safe mean)
            if mask_bin.sum() > 0:
                mean_intensity_pred = recon_post[mask_bin].mean()
                mean_intensity_target = post[mask_bin].mean()
                loss_intensity = F.l1_loss(mean_intensity_pred, mean_intensity_target)
            else:
                loss_intensity = torch.tensor(0.0, device=accelerator.device)

            # === Weighted combination  ===
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
            total = 0.3 * loss_global + 0.6 * loss_roi + 0.05 * loss_contrast + 0.05 * loss_intensity

            # NaN protection
            if torch.isnan(total).any():
                accelerator.print("⚠️ Skipping due to NaN loss.")
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
    with ema.average_parameters():
        with torch.no_grad():
            for batch in test_loader:
                pre = batch["pre_img"].to(accelerator.device)
                post = batch["post_img"].to(accelerator.device)

                delta = (post - pre) / 0.5
                noise = torch.randn_like(delta)
                t = torch.randint(0, config.num_train_timesteps, (post.size(0),), device=post.device).long()
                noisy_delta = noise_scheduler.add_noise(delta, noise, t)

                input_model = torch.cat([noisy_delta, pre], dim=1)
                pred_delta = model(input_model).clamp(-1, 1)
                recon_post = (pred_delta * 0.5 + pre).clamp(-1, 1)
                metrics_list.append(evaluate_batch(recon_post, post))

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
        accelerator.print(f"✅ Best model saved at epoch {epoch+1} (MAE={best_loss:.6f})")

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
