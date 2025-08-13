import os
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from accelerate import Accelerator
from diffusers import DDPMScheduler
from torch_ema import ExponentialMovingAverage
from torchvision.models import vgg16, VGG16_Weights

from models.conditional_unet import ConditionalUNet
from utils.data_loading import TrainingConfig, get_dataloaders


# === LOSSES ===
class PerceptualLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg16(weights=VGG16_Weights.DEFAULT).features[:16].eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg
        self.criterion = torch.nn.MSELoss()

    def forward(self, pred, target):
        pred_features = self.vgg(pred.repeat(1, 3, 1, 1))
        target_features = self.vgg(target.repeat(1, 3, 1, 1))
        return self.criterion(pred_features, target_features)


def total_variation_loss(x):
    diff_h = torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])
    diff_v = torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])
    return torch.mean(diff_h) + torch.mean(diff_v)


def clamp_and_safe_pred(x):
    return torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1, 1)


# === METRICS ===
def evaluate_batch(recon, target):
    with torch.no_grad():
        mae = F.l1_loss(recon, target).item()
        mse = F.mse_loss(recon, target).item()
    return {"MAE": mae, "MSE": mse}


# === SAMPLE GENERATION ===
def generate_sample(model, sample_batch, noise_scheduler, accelerator, epoch):
    model.eval()
    with torch.no_grad():
        sample_pre_img = sample_batch["pre_img"][0].unsqueeze(0).to(accelerator.device)
        sample_post_img = sample_batch["post_img"][0].unsqueeze(0).to(accelerator.device)

        xt = torch.randn_like(sample_pre_img)
        for t in reversed(range(noise_scheduler.config.num_train_timesteps)):
            model_input = torch.cat([xt, sample_pre_img], dim=1)
            pred_x0 = clamp_and_safe_pred(model(model_input))

            alpha = noise_scheduler.alphas_cumprod[t].view(1, 1, 1, 1).to(xt.device)
            beta = 1 - alpha
            noise = torch.randn_like(xt) if t > 0 else torch.zeros_like(xt)

            xt = torch.sqrt(alpha) * pred_x0 + torch.sqrt(beta) * noise
            xt = xt.clamp(-1, 1)

        gen_post = (xt * 0.5 + sample_pre_img).clamp(-1, 1)

        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow((sample_pre_img.squeeze().cpu().numpy() + 1) / 2, cmap="gray"); axs[0].set_title("Pre-contrast"); axs[0].axis("off")
        axs[1].imshow((sample_post_img.squeeze().cpu().numpy() + 1) / 2, cmap="gray"); axs[1].set_title("GT Post"); axs[1].axis("off")
        axs[2].imshow((gen_post.squeeze().cpu().numpy() + 1) / 2, cmap="gray"); axs[2].set_title(f"Generated Post (Epoch {epoch+1})"); axs[2].axis("off")
        plt.tight_layout(); plt.show()


# === MAIN TRAINING ===
def main():
    config = TrainingConfig()
    train_loader, test_loader = get_dataloaders(config)

    model = ConditionalUNet(in_channels=2).to("cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    perceptual_loss_fn = PerceptualLoss().to("cuda")
    noise_scheduler = DDPMScheduler(num_train_timesteps=config.num_train_timesteps, beta_schedule="squaredcos_cap_v2")
    ema = ExponentialMovingAverage(model.parameters(), decay=0.999)
    accelerator = Accelerator(mixed_precision=config.mixed_precision)

    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    torch.manual_seed(config.seed)
    os.makedirs(config.best_model_dir, exist_ok=True)

    sample_batch = next(iter(test_loader))
    best_loss = float("inf")
    all_metrics, loss_components = [], {"Epoch": [], "MAE": [], "Percep": [], "TV": [], "MSE": [], "Total": []}

    for epoch in range(config.num_epochs):
        model.train()
        total_loss = total_mae = total_percep = total_tv = total_mse = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}"):
            pre_img = batch["pre_img"].to(accelerator.device)
            post_img = batch["post_img"].to(accelerator.device)

            delta = (post_img - pre_img) / 0.5
            noise = torch.randn_like(delta)
            t = torch.randint(0, config.num_train_timesteps, (delta.size(0),), device=delta.device).long()
            noisy_delta = noise_scheduler.add_noise(delta, noise, t)
            model_input = torch.cat([noisy_delta, pre_img], dim=1)

            with accelerator.accumulate(model):
                pred_x0 = clamp_and_safe_pred(model(model_input))
                recon_post = (pred_x0 * 0.5 + pre_img).clamp(-1, 1)

                loss_mae = F.l1_loss(recon_post, post_img)
                loss_percep = perceptual_loss_fn(recon_post, post_img)
                loss_tv = total_variation_loss(recon_post)
                loss_mse = F.mse_loss(recon_post, post_img)

                loss = 0.3 * loss_mae + 0.6 * loss_percep + 0.15 * loss_tv + 0.05 * loss_mse
                accelerator.backward(loss)
                optimizer.step(); optimizer.zero_grad()
                ema.update()

            total_loss += loss.item()
            total_mae += loss_mae.item()
            total_percep += loss_percep.item()
            total_tv += loss_tv.item()
            total_mse += loss_mse.item()

        # Logging
        loss_components["Epoch"].append(epoch + 1)
        loss_components["MAE"].append(total_mae / len(train_loader))
        loss_components["Percep"].append(total_percep / len(train_loader))
        loss_components["TV"].append(total_tv / len(train_loader))
        loss_components["MSE"].append(total_mse / len(train_loader))
        loss_components["Total"].append(total_loss / len(train_loader))

        # Evaluation
        model.eval()
        metrics_list = []
        with ema.average_parameters():
            with torch.no_grad():
                for batch in test_loader:
                    pre_img = batch["pre_img"].to(accelerator.device)
                    post_img = batch["post_img"].to(accelerator.device)
                    delta = (post_img - pre_img) / 0.5
                    noise = torch.randn_like(delta)
                    t = torch.randint(0, config.num_train_timesteps, (post_img.size(0),), device=post_img.device).long()
                    noisy_delta = noise_scheduler.add_noise(delta, noise, t)
                    model_input = torch.cat([noisy_delta, pre_img], dim=1)
                    pred_x0 = clamp_and_safe_pred(model(model_input))
                    recon_post = (pred_x0 * 0.5 + pre_img).clamp(-1, 1)
                    metrics_list.append(evaluate_batch(recon_post, post_img))

        avg_metrics = {k: float(np.mean([m[k] for m in metrics_list])) for k in metrics_list[0]}
        avg_metrics["Epoch"] = epoch + 1
        all_metrics.append(avg_metrics)

        accelerator.print(f"[Epoch {epoch+1}] Loss: {total_loss/len(train_loader):.6f} | Metrics: {avg_metrics}")

        if avg_metrics["MAE"] < best_loss:
            best_loss = avg_metrics["MAE"]
            with ema.average_parameters():
                torch.save(accelerator.unwrap_model(model).state_dict(),
                           os.path.join(config.best_model_dir, "pytorch_model.bin"))
            accelerator.print(f"✅ Saved best EMA model at epoch {epoch+1} (MAE={best_loss:.6f})")

        with ema.average_parameters():
            generate_sample(model, sample_batch, noise_scheduler, accelerator, epoch)


if __name__ == "__main__":
    main()
