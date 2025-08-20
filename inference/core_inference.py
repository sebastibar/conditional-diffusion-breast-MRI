import os
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips
from torchmetrics.functional import structural_similarity_index_measure as ssim_torch
from torchmetrics.functional import peak_signal_noise_ratio as psnr_torch
from .distribution_metrics import evaluate_distribution_metrics, compute_baseline_fids


def get_masked_dataloaders_for_inference(pre_folder, post_folder, mask_folder, img_size=256, batch_size=1):
    """A simplified version to load data for inference."""
    from torch.utils.data import Dataset, DataLoader

    class InferenceDataset(Dataset):
        def __init__(self, pre_folder, post_folder, mask_folder, transform=None, img_size=256):
            self.pre_folder = pre_folder
            self.post_folder = post_folder
            self.mask_folder = mask_folder
            self.transform = transform
            self.img_size = img_size

            pre_files = {f for f in os.listdir(pre_folder) if f.lower().endswith(".png")}
            post_files = {f for f in os.listdir(post_folder) if f.lower().endswith(".png")}
            mask_files = {f for f in os.listdir(mask_folder) if f.lower().endswith(".png")}

            self.common_files = sorted(list(pre_files & post_files & mask_files))
            if not self.common_files:
                raise ValueError(f"No matching PNG files across directories")

        def __len__(self):
            return len(self.common_files)

        def __getitem__(self, idx):
            filename = self.common_files[idx]
            
            pre_img = Image.open(os.path.join(self.pre_folder, filename)).convert("L")
            post_img = Image.open(os.path.join(self.post_folder, filename)).convert("L")
            mask_img = Image.open(os.path.join(self.mask_folder, filename)).convert("L")

            # Convert to tensor and normalize
            pre_tensor = transforms.functional.to_tensor(pre_img)
            pre_tensor = (pre_tensor - 0.5) / 0.5

            post_tensor = transforms.functional.to_tensor(post_img)
            post_tensor = (post_tensor - 0.5) / 0.5

            # Convert mask to binary
            tumor_mask = transforms.functional.to_tensor(mask_img)
            tumor_mask = (tumor_mask > 0).float()

            # Resize if needed
            if self.transform:
                pre_tensor = self.transform(pre_tensor)
                post_tensor = self.transform(post_tensor)
                tumor_mask = self.transform(tumor_mask)
            else:
                if tumor_mask.shape[-2:] != (self.img_size, self.img_size):
                    tumor_mask = F.interpolate(tumor_mask.unsqueeze(0), size=(self.img_size, self.img_size), mode='nearest').squeeze(0)

            return {
                "pre_img": pre_tensor,
                "post_img": post_tensor,
                "tumor_mask": tumor_mask,
                "filename": filename
            }

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size))
    ])

    dataset = InferenceDataset(pre_folder, post_folder, mask_folder, transform, img_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return dataloader

def run_inference(model, dataloader, config, device, mode="PC"):
    """
    Core inference function that runs generation and calculates metrics.
    
    Args:
        model: Loaded trained model
        dataloader: DataLoader with test data
        config: Dictionary with inference parameters
        device: torch device
        mode: Either "PC" (post-contrast) or "SUB" (subtraction)
    
    Returns:
        results: List of dictionaries with metrics per sample
    """
    from diffusers import DDPMScheduler
    import lpips
    
    # Initialize scheduler
    scheduler = DDPMScheduler(num_train_timesteps=config.get("num_inference_timesteps", 50), 
                             beta_schedule="squaredcos_cap_v2")
    
    # Initialize metrics
    lpips_fn = lpips.LPIPS(net='vgg').to(device).eval()
    
    results = []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Running inference"):
            pre_img = batch["pre_img"].to(device)
            post_img = batch["post_img"].to(device)
            tumor_mask = batch["tumor_mask"].to(device) if "tumor_mask" in batch else None
            filenames = batch["filename"]
            
            # Prepare model input based on mode
            if mode == "PC":
                # Start with random noise
                noisy_input = torch.randn_like(pre_img)
                
                # DDPM sampling loop
                for t in reversed(range(scheduler.config.num_train_timesteps)):
                    # Prepare model input
                    if tumor_mask is not None and config.get("use_mask_input", False):
                        model_input = torch.cat([noisy_input, pre_img, tumor_mask], dim=1)
                    else:
                        model_input = torch.cat([noisy_input, pre_img], dim=1)
                    
                    # Predict x0
                    x0_pred = model(model_input).clamp(-1, 1)
                    
                    # Update noisy input
                    alpha = scheduler.alphas_cumprod[t].to(device).view(-1, 1, 1, 1)
                    beta = 1 - alpha
                    noise = torch.randn_like(noisy_input) if t > 0 else torch.zeros_like(noisy_input)
                    
                    noisy_input = torch.sqrt(alpha) * x0_pred + torch.sqrt(beta) * noise
                    noisy_input = noisy_input.clamp(-1, 1)
                
                generated = noisy_input
                
            elif mode == "SUB":
                # For subtraction models, we predict the enhancement map
                delta_target = (post_img - pre_img) / 0.5
                noisy_delta = torch.randn_like(delta_target)
                
                # DDPM sampling loop
                for t in reversed(range(scheduler.config.num_train_timesteps)):
                    # Prepare model input
                    if tumor_mask is not None and config.get("use_mask_input", False):
                        model_input = torch.cat([noisy_delta, pre_img, tumor_mask], dim=1)
                    else:
                        model_input = torch.cat([noisy_delta, pre_img], dim=1)
                    
                    # Predict clean delta
                    pred_delta = model(model_input).clamp(-1, 1)
                    
                    # Update noisy delta
                    alpha = scheduler.alphas_cumprod[t].to(device).view(-1, 1, 1, 1)
                    beta = 1 - alpha
                    noise = torch.randn_like(noisy_delta) if t > 0 else torch.zeros_like(noisy_delta)
                    
                    noisy_delta = torch.sqrt(alpha) * pred_delta + torch.sqrt(beta) * noise
                    noisy_delta = noisy_delta.clamp(-1, 1)
                
                # Reconstruct post-contrast image
                generated = (noisy_delta * 0.5 + pre_img).clamp(-1, 1)
            
            # Calculate metrics
            mae = F.l1_loss(generated, post_img).item()
            ssim_val = ssim_torch(generated, post_img).item()
            psnr_val = psnr_torch(generated, post_img).item()
            lpips_val = lpips_fn(generated.repeat(1, 3, 1, 1), 
                                post_img.repeat(1, 3, 1, 1)).item()
            
            results.append({
                "filename": filenames[0],
                "MAE": mae,
                "SSIM": ssim_val,
                "PSNR": psnr_val,
                "LPIPS": lpips_val,
                "generated": generated.cpu(),
                "ground_truth": post_img.cpu(),
                "pre_contrast": pre_img.cpu()
            })
    
    return results

def save_results(results, output_dir, config):
    """Save inference results, metrics, and visualizations."""
    os.makedirs(output_dir, exist_ok=True)
    gen_dir = os.path.join(output_dir, "generated_images")
    os.makedirs(gen_dir, exist_ok=True)
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'generated' and k != 'ground_truth' and k != 'pre_contrast'} 
                              for r in results])
    metrics_df.to_csv(os.path.join(output_dir, "inference_metrics.csv"), index=False)
    
    # Save sample visualizations
    n_samples = min(config.get("num_sample_visualizations", 5), len(results))
    sampled_results = results[:n_samples]
    
    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for idx, result in enumerate(sampled_results):
        # Convert tensors to numpy for plotting
        pre_np = (result["pre_contrast"].squeeze().numpy() + 1) / 2
        gt_np = (result["ground_truth"].squeeze().numpy() + 1) / 2
        gen_np = (result["generated"].squeeze().numpy() + 1) / 2
        
        axes[idx, 0].imshow(pre_np, cmap="gray")
        axes[idx, 0].set_title(f"Pre-contrast\n{result['filename']}")
        axes[idx, 0].axis("off")
        
        axes[idx, 1].imshow(gt_np, cmap="gray")
        axes[idx, 1].set_title("Ground Truth")
        axes[idx, 1].axis("off")
        
        axes[idx, 2].imshow(gen_np, cmap="gray")
        axes[idx, 2].set_title(f"Generated\nMAE: {result['MAE']:.3f}, SSIM: {result['SSIM']:.3f}")
        axes[idx, 2].axis("off")
        
        # Save individual generated image
        gen_img = Image.fromarray((gen_np * 255).astype(np.uint8))
        gen_img.save(os.path.join(gen_dir, result["filename"]))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sample_comparisons.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create summary plots
    create_summary_plots(metrics_df, output_dir)
    
    return metrics_df

def create_summary_plots(metrics_df, output_dir):
    """Create summary plots of the metrics."""
    # Violin plot of metrics
    plt.figure(figsize=(12, 8))
    melted_df = metrics_df.melt(value_vars=["MAE", "SSIM", "PSNR", "LPIPS"], 
                               var_name="Metric", value_name="Value")
    
    sns.violinplot(x="Metric", y="Value", data=melted_df, inner="box")
    plt.title("Distribution of Evaluation Metrics")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "metrics_distribution.png"), dpi=300)
    plt.close()
    
    # Print summary statistics
    summary = metrics_df[["MAE", "SSIM", "PSNR", "LPIPS"]].agg(["mean", "std", "min", "max"])
    print("=== Inference Results Summary ===")
    print(summary)
    
    summary.to_csv(os.path.join(output_dir, "metrics_summary.csv"))

def compute_distribution_metrics(real_results, generated_results, output_dir, config, device='cuda'):
    """
    Compute distribution-level metrics (FID, FRD) for the entire dataset.
    
    Args:
        real_results: List of results from real data (from run_inference)
        generated_results: List of results from generated data
        output_dir: Directory to save results
        config: Configuration dictionary
        device: Device to use for computation
        
    Returns:
        Dictionary with distribution metrics
    """
    # Convert tensors back to PIL Images for FID calculation
    real_images = []
    generated_images = []
    
    for real_res, gen_res in zip(real_results, generated_results):
        # Convert tensors to PIL Images
        real_img = tensor_to_pil(real_res['ground_truth'])
        gen_img = tensor_to_pil(gen_res['generated'])
        
        real_images.append(real_img)
        generated_images.append(gen_img)
    
    # Compute distribution metrics
    dist_metrics = evaluate_distribution_metrics(
        real_images, 
        generated_images, 
        device=device
    )
    
    # Save distribution metrics
    dist_df = pd.DataFrame([dist_metrics])
    dist_df.to_csv(os.path.join(output_dir, "distribution_metrics.csv"), index=False)
    
    # Compute baselines for reference
    baselines = compute_baseline_fids(
        config["pre_folder"],
        config["post_folder"],
        device=device
    )
    
    # Add baselines to results
    dist_metrics['Baseline_Post_vs_Pre'] = baselines['Post_vs_Pre']
    dist_metrics['Baseline_Subtraction_vs_Pre'] = baselines['Subtraction_vs_Pre']
    
    # Create comparison plot
    create_distribution_comparison_plot(dist_metrics, output_dir)
    
    return dist_metrics

def tensor_to_pil(tensor):
    """Convert a tensor back to PIL Image."""
    import torch
    from PIL import Image
    
    # Convert from [-1, 1] to [0, 1]
    img_np = (tensor.squeeze().numpy() + 1) / 2.0
    # Convert to [0, 255] and uint8
    img_np = (img_np * 255).astype(np.uint8)
    
    if len(img_np.shape) == 2:  # Grayscale
        return Image.fromarray(img_np, mode='L')
    else:  # RGB
        return Image.fromarray(img_np, mode='RGB')

def create_distribution_comparison_plot(metrics, output_dir):
    """Create a bar plot comparing distribution metrics."""
    import matplotlib.pyplot as plt
    
    # Extract metrics for plotting
    fid_values = {
        'Generated vs Real': metrics['FID'],
        'Baseline: Post vs Pre': metrics.get('Baseline_Post_vs_Pre', 0),
        'Baseline: Subtraction vs Pre': metrics.get('Baseline_Subtraction_vs_Pre', 0)
    }
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(fid_values)), list(fid_values.values()), 
                   color=['skyblue', 'lightcoral', 'lightgreen'])
    
    plt.title('FID Comparison', fontsize=14)
    plt.ylabel('FID Score', fontsize=12)
    plt.xticks(range(len(fid_values)), list(fid_values.keys()), rotation=45, ha='right')
    
    # Add values on top of bars
    for i, (name, value) in enumerate(fid_values.items()):
        plt.text(i, value + 5, f'{value:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fid_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
