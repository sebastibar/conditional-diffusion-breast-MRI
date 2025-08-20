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
from utils.losses_metrics import PerceptualLoss, total_variation_loss, evaluate_batch
from utils.plot_utils import plot_training_curves

def generate_sample(model, test_dataset, noise_scheduler, accelerator, epoch, title="Generated Post"):
    """Helper function to generate and plot a sample during training for MASK-CONDITIONED model."""
    model.eval()
    sample = test_dataset[0]
    
    pre_img = sample["pre_img"].unsqueeze(0)
    post_img = sample["post_img"].unsqueeze(0)
    tumor_mask = sample["tumor_mask"].unsqueeze(0) # Get the mask for conditioning

    # Move to device
    pre_img = pre_img.to(accelerator.device)
    tumor_mask = tumor_mask.to(accelerator.device)
    noisy = torch.randn_like(pre_img)

    with torch.no_grad():
        for t in reversed(range(noise_scheduler.config.num_train_timesteps)):
            # Concatenate the MASK as the third input channel
            model_input = torch.cat([noisy, pre_img, tumor_mask], dim=1)
            x0_pred = model(model_input).clamp(-1, 1)
            
            alpha = noise_scheduler.alphas_cumprod[t].to(pre_img.device).view(1, 1, 1, 1)
            beta = 1 - alpha
            noise = torch.randn_like(noisy) if t > 0 else torch.zeros_like(noisy)
            noisy = torch.sqrt(alpha) * x0_pred + torch.sqrt(beta) * noise
            noisy = noisy.clamp(-1, 1)

    gen_img = (noisy + 1) / 2
    gen_img_np = gen_img.squeeze().cpu().numpy()
    pre_img_plot = (pre_img.squeeze().cpu().numpy() + 1) / 2
    target_img_plot = (post_img.squeeze().numpy() + 1) / 2

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(pre_img_plot, cmap="gray"); axs[0].set_title("Pre-contrast"); axs[0].axis("off")
    axs[1].imshow(target_img_plot, cmap="gray"); axs[1].set_title("Ground Truth"); axs[1].axis("off")
    axs[2].imshow(gen_img_np, cmap="gray"); axs[2].set_title(f"{title} (Epoch {epoch+1})"); axs[2].axis("off")
    plt.tight_layout(); 
    plt.savefig(f"{config.output_dir}/sample_epoch_{epoch+1}.png")
    plt.close()

def main():
    # === CONFIG & SETUP ===
    config = TrainingConfig()
    train_loader, test_loader = get_masked_dataloaders(config) # Provides masks
    
    model = MaskConditionedUNet(in_channels=3) # Input: [noisy_img, pre_img, tumor_mask]
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

            # DDPM forward process
            noise = torch.randn_like(post_img)
            t = torch.randint(0, config.num_train_timesteps, (post_img.size(0),), device=post_img.device).long()
            noisy_post = noise_scheduler.add_noise(post_img, noise, t)
            
            # Model input is [noisy_image, pre_image, tumor_mask]
            input_concat = torch.cat([noisy_post, pre_img, tumor_mask], dim=1)

            with accelerator.accumulate(model):
                x0_pred = model(input_concat) # Model predicts the clean post-contrast image
                target = post_img

                # Calculate STANDARD losses on the whole image.
                # The model is conditioned on the mask, so we don't need complex ROI losses here.
                loss_mae = F.l1_loss(x0_pred, target)
                loss_percep = perceptual_loss_fn(x0_pred, target)
                loss_tv = total_variation_loss(x0_pred)
                loss_mse = F.mse_loss(x0_pred, target)

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
                    tumor_mask = batch["tumor_mask"] # Mask is needed for input during evaluation too
                    
                    # Create noisy input for evaluation
                    noise = torch.randn_like(post_img)
                    t = torch.randint(0, config.num_train_timesteps, (post_img.size(0),), device=post_img.device).long()
                    noisy_post = noise_scheduler.add_noise(post_img, noise, t)
                    
                    # Input: [noisy_image, pre_image, tumor_mask]
                    input_concat = torch.cat([noisy_post, pre_img, tumor_mask], dim=1)
                    
                    x0_pred = model(input_concat)
                    metrics_list.append(evaluate_batch(x0_pred, post_img))

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
                generate_sample(model, test_loader.dataset, noise_scheduler, accelerator, epoch, "Generated Post [EMA]")

    # === AFTER TRAINING: PLOT RESULTS ===
    if accelerator.is_local_main_process:
        plot_training_curves(all_metrics, loss_components)
        accelerator.print("Mask-Conditioned (PC) Training complete!")

if __name__ == "__main__":
    main()
