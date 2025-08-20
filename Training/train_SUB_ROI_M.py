import os
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from accelerate import Accelerator
from diffusers import DDPMScheduler
from torch_ema import ExponentialMovingAverage

# Import from our modules
from models.masked_conditional_unet import MaskConditionedUNet # Uses in_channels=3
from utils.masked_data_loading import TrainingConfig, get_masked_dataloaders
from utils.losses_metrics import PerceptualLoss, total_variation_loss, evaluate_batch, clamp_and_safe_pred
from utils.plot_utils import plot_training_curves

def generate_sample(model, sample_batch, noise_scheduler, accelerator, epoch, title="Generated Post"):
    """Helper function to generate and plot a sample during training for MASK-CONDITIONED SUB model."""
    model.eval()
    with torch.no_grad():
        sample_pre_img = sample_batch["pre_img"][0].unsqueeze(0)
        sample_post_img = sample_batch["post_img"][0].unsqueeze(0)
        sample_tumor_mask = sample_batch["tumor_mask"][0].unsqueeze(0) # Get the mask

        sample_pre_img = sample_pre_img.to(accelerator.device)
        sample_tumor_mask = sample_tumor_mask.to(accelerator.device)
        
        xt = torch.randn_like(sample_pre_img)
        
        # Reverse diffusion process
        for t in reversed(range(noise_scheduler.config.num_train_timesteps)):
            # Input: [x_t, pre_img, tumor_mask]
            model_input = torch.cat([xt, sample_pre_img, sample_tumor_mask], dim=1)
            pred_x0 = clamp_and_safe_pred(model(model_input))

            alpha = noise_scheduler.alphas_cumprod[t].view(1, 1, 1, 1).to(xt.device)
            beta = 1 - alpha
            noise = torch.randn_like(xt) if t > 0 else torch.zeros_like(xt)

            xt = torch.sqrt(alpha) * pred_x0 + torch.sqrt(beta) * noise
            xt = xt.clamp(-1, 1)

        # Reconstruct final post-contrast image from predicted subtraction
        gen_post = (xt * 0.5 + sample_pre_img).clamp(-1, 1)

        # Plotting
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow((sample_pre_img.squeeze().cpu().numpy() + 1) / 2, cmap="gray")
        axs[0].set_title("Pre-contrast"); axs[0].axis("off")
        axs[1].imshow((sample_post_img.squeeze().cpu().numpy() + 1) / 2, cmap="gray")
        axs[1].set_title("GT Post"); axs[1].axis("off")
        axs[2].imshow((gen_post.squeeze().cpu().numpy() + 1) / 2, cmap="gray")
        axs[2].set_title(f"{title} (Epoch {epoch+1})"); axs[2].axis("off")
        plt.tight_layout()
        plt.savefig(f"{config.output_dir}/sample_epoch_{epoch+1}.png")
        plt.close()

def main():
    # === CONFIG & SETUP ===
    config = TrainingConfig()
    train_loader, test_loader = get_masked_dataloaders(config) # Provides masks
    
    model = MaskConditionedUNet(in_channels=3) # Input: [noisy_delta, pre_img, tumor_mask]
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

    # Get a sample batch for visualization (contains mask)
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
            
            # Model input is [noisy_delta, pre_img, tumor_mask]
            model_input = torch.cat([noisy_delta, pre_img, tumor_mask], dim=1)

            with accelerator.accumulate(model):
                # Model predicts the clean subtraction image (delta)
                pred_x0 = clamp_and_safe_pred(model(model_input))
                # Reconstruct the post-contrast image for loss calculation
                recon_post = (pred_x0 * 0.5 + pre_img).clamp(-1, 1)

                # Calculate STANDARD losses on the whole image.
                loss_mae = F.l1_loss(recon_post, post_img)
                loss_percep = perceptual_loss_fn(recon_post, post_img)
                loss_tv = total_variation_loss(recon_post)
                loss_mse = F.mse_loss(recon_post, post_img)

                # Combined loss (standard weighting)
                loss = 0.3 * loss_mae + 0.6 * loss_percep + 0.15 * loss_tv + 0.05 * loss_mse

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                ema.update()

            # Accumulate losses for logging
            total_loss += loss.item()
            total_mae += loss_mae.item()
            total_percep += loss_percep.item()
            total_tv += loss_tv.item()
            total_mse += loss_mse.item()

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
                    tumor_mask = batch["tumor_mask"] # Mask is needed for input
                    
                    delta = (post_img - pre_img) / 0.5
                    noise = torch.randn_like(delta)
                    t = torch.randint(0, config.num_train_timesteps, (post_img.size(0),), device=post_img.device).long()
                    noisy_delta = noise_scheduler.add_noise(delta, noise, t)
                    
                    # Input: [noisy_delta, pre_img, tumor_mask]
                    model_input = torch.cat([noisy_delta, pre_img, tumor_mask], dim=1)
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
                generate_sample(model, sample_batch, noise_scheduler, accelerator, epoch, "Generated Post [EMA]")

    # === AFTER TRAINING: PLOT RESULTS ===
    if accelerator.is_local_main_process:
        plot_training_curves(all_metrics, loss_components)
        accelerator.print("Mask-Conditioned (SUB) Training complete!")

if __name__ == "__main__":
    main()
