# ROI-aware training for Subtraction prediction (delta = post - pre)
import os
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from accelerate import Accelerator
from diffusers import DDPMScheduler
from torch_ema import ExponentialMovingAverage

# Import from our modules
from models.conditional_unet import ConditionalUNet
from utils.masked_data_loading import TrainingConfig, get_masked_dataloaders
from utils.losses_metrics import PerceptualLoss, total_variation_loss, evaluate_batch, clamp_and_safe_pred
from utils.plot_utils import plot_training_curves
from training.train_SUB import generate_sample  # Reuse the sample generation from SUB

def main():
    # === CONFIG & SETUP ===
    config = TrainingConfig()
    train_loader, test_loader = get_masked_dataloaders(config) # This now provides masks
    
    model = ConditionalUNet(in_channels=2) # Input: [noisy_delta, pre_img]
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    noise_scheduler = DDPMScheduler(num_train_timesteps=config.num_train_timesteps, beta_schedule="squaredcos_cap_v2")
    perceptual_loss_fn = PerceptualLoss()
    ema = ExponentialMovingAverage(model.parameters(), decay=0.999)
    accelerator = Accelerator(mixed_precision=config.mixed_precision)

    # Let accelerator handle device placement
    model, optimizer, train_loader, perceptual_loss_fn = accelerator.prepare(
        model, optimizer, train_loader, perceptual_loss_fn
    )

    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.best_model_dir, exist_ok=True)
    torch.manual_seed(config.seed)

    # Get a sample batch for visualization (contains mask, but generator won't use it)
    sample_batch = next(iter(test_loader))

    # === TRAINING STATE ===
    best_loss = float("inf")
    all_metrics = []
    loss_components = {"Epoch": [], "MAE": [], "Percep": [], "TV": [], "MSE": [], "Total": []}

    # === TRAINING LOOP ===
    for epoch in range(config.num_epochs):
        model.train()
        total_loss = total_mae = total_percep = total_tv = total_mse = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}", disable=not accelerator.is_local_main_process)
        for batch in progress_bar:
            pre_img = batch["pre_img"]
            post_img = batch["post_img"]
            tumor_mask = batch["tumor_mask"] # Get the mask

            # Create subtraction target (delta)
            delta = (post_img - pre_img) / 0.5
            noise = torch.randn_like(delta)
            t = torch.randint(0, config.num_train_timesteps, (delta.size(0),), device=delta.device).long()
            noisy_delta = noise_scheduler.add_noise(delta, noise, t)
            
            # Model input is [noisy_delta, pre_img]
            model_input = torch.cat([noisy_delta, pre_img], dim=1)

            with accelerator.accumulate(model):
                # Model predicts the clean subtraction image (delta)
                pred_x0 = clamp_and_safe_pred(model(model_input))
                # Reconstruct the post-contrast image for loss calculation
                recon_post = (pred_x0 * 0.5 + pre_img).clamp(-1, 1)

                # Create masked views for ROI losses
                tumor_mask_bin = tumor_mask.bool()
                recon_roi = recon_post * tumor_mask
                post_roi = post_img * tumor_mask
                pre_roi = pre_img * tumor_mask

                # --- GLOBAL LOSSES (on the whole image) ---
                loss_mae_global = F.l1_loss(recon_post, post_img)
                loss_mse_global = F.mse_loss(recon_post, post_img)
                loss_tv_global = total_variation_loss(recon_post)
                feat_pred_global = perceptual_loss_fn.extract_features(recon_post)
                feat_target_global = perceptual_loss_fn.extract_features(post_img)
                loss_percep_global = F.mse_loss(feat_pred_global, feat_target_global)

                # --- ROI LOSSES (only inside the tumor mask) ---
                loss_mae_roi = F.l1_loss(recon_roi, post_roi)
                loss_mse_roi = F.mse_loss(recon_roi, post_roi)
                loss_tv_roi = total_variation_loss(recon_roi)
                # Perceptual ROI Loss
                feat_pred = perceptual_loss_fn.extract_features(recon_post)
                feat_target = perceptual_loss_fn.extract_features(post_img)
                mask_resized = F.interpolate(tumor_mask, size=feat_pred.shape[-2:], mode='bilinear', align_corners=False)
                loss_percep_roi = F.mse_loss(feat_pred * mask_resized, feat_target * mask_resized)

                # --- SPECIALIZED ROI LOSSES ---
                # Contrast Difference Loss
                contrast_pred = F.relu(recon_roi - pre_roi)
                contrast_target = F.relu(post_roi - pre_roi)
                loss_contrast = F.l1_loss(contrast_pred, contrast_target)

                # Intensity Loss
                if tumor_mask_bin.any():
                    mean_intensity_pred = recon_post[tumor_mask_bin].mean()
                    mean_intensity_target = post_img[tumor_mask_bin].mean()
                    loss_intensity = F.l1_loss(mean_intensity_pred, mean_intensity_target)
                else:
                    loss_intensity = torch.tensor(0.0, device=accelerator.device)

                # === COMBINED LOSS ===
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
                loss = (
                    0.3 * loss_global +
                    0.6 * loss_roi +
                    0.05 * loss_contrast +
                    0.05 * loss_intensity
                )

                # NaN check
                if torch.isnan(loss):
                    accelerator.print("⚠️ Skipping batch due to NaN loss.")
                    continue

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                ema.update()

            # Accumulate losses for logging
            total_loss += loss.item()
            total_mae += loss_mae_global.item()
            total_percep += loss_percep_global.item()
            total_tv += loss_tv_global.item()
            total_mse += loss_mse_global.item()

            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

        # === EPOCH LOGGING ===
        avg_total_loss = total_loss / len(train_loader)
        loss_components["Epoch"].append(epoch + 1)
        loss_components["MAE"].append(total_mae / len(train_loader))
        loss_components["Percep"].append(total_percep / len(train_loader))
        loss_components["TV"].append(total_tv / len(train_loader))
        loss_components["MSE"].append(total_mse / len(train_loader))
        loss_components["Total"].append(avg_total_loss)

        # === EVALUATION (using EMA weights) ===
        model.eval()
        metrics_list = []
        with ema.average_parameters():
            with torch.no_grad():
                for batch in test_loader:
                    pre_img = batch["pre_img"]
                    post_img = batch["post_img"]
                    
                    delta = (post_img - pre_img) / 0.5
                    noise = torch.randn_like(delta)
                    t = torch.randint(0, config.num_train_timesteps, (post_img.size(0),), device=post_img.device).long()
                    noisy_delta = noise_scheduler.add_noise(delta, noise, t)
                    
                    model_input = torch.cat([noisy_delta, pre_img], dim=1)
                    pred_x0 = clamp_and_safe_pred(model(model_input))
                    recon_post = (pred_x0 * 0.5 + pre_img).clamp(-1, 1)
                    
                    metrics_list.append(evaluate_batch(recon_post, post_img))

        avg_metrics = {k: np.mean([m[k] for m in metrics_list]) for k in metrics_list[0]}
        avg_metrics["Epoch"] = epoch + 1
        all_metrics.append(avg_metrics)

        accelerator.print(f"[Epoch {epoch+1}] Loss: {avg_total_loss:.6f} | Test MAE: {avg_metrics['MAE']:.6f}, MSE: {avg_metrics['MSE']:.6f}")

        # === SAVE BEST MODEL ===
        if avg_metrics["MAE"] < best_loss:
            best_loss = avg_metrics["MAE"]
            unwrapped_model = accelerator.unwrap_model(model)
            with ema.average_parameters():
                torch.save(unwrapped_model.state_dict(), os.path.join(config.best_model_dir, "pytorch_model.bin"))
            accelerator.print(f"✅ Saved best EMA model at epoch {epoch+1} (MAE={best_loss:.6f})")

        # === GENERATE SAMPLE IMAGE ===
        if accelerator.is_local_main_process:
            with ema.average_parameters():
                # We use the generate_sample function from train_SUB
                generate_sample(model, sample_batch, noise_scheduler, accelerator, epoch, "Generated Post [EMA]")

    # === AFTER TRAINING: PLOT RESULTS ===
    if accelerator.is_local_main_process:
        plot_training_curves(all_metrics, loss_components)
        accelerator.print("ROI Loss (SUB) Training complete!")

if __name__ == "__main__":
    main()
