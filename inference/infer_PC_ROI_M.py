#!/usr/bin/env python3
"""
Inference script for Mask-Conditioned Post-Contrast model with distribution metrics.
"""

import torch
import os
from models.masked_conditional_unet import MaskConditionedUNet
from inference.core_inference import (
    get_masked_dataloaders_for_inference, 
    run_inference, 
    save_results,
    compute_distribution_metrics
)

# Configuration
config = {
    "model_path": "best-model/pytorch_model.bin",
    "pre_folder": "bilateral_slices/test/precontrast",
    "post_folder": "bilateral_slices/test/postcontrast", 
    "mask_folder": "bilateral_slices/test/masks",
    "output_dir": "inference_results/PC_ROI_M",
    "img_size": 256,
    "num_inference_timesteps": 50,
    "use_mask_input": True,  # ROI-M uses mask as input!
    "num_sample_visualizations": 5,
    "mode": "PC"
}

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model - this one has 3 input channels
    model = MaskConditionedUNet(in_channels=3).to(device)
    model.load_state_dict(torch.load(config["model_path"], map_location=device))
    print(" Model loaded successfully")
    
    # Load data
    dataloader = get_masked_dataloaders_for_inference(
        config["pre_folder"], 
        config["post_folder"],
        config["mask_folder"],
        config["img_size"],
        batch_size=1
    )
    print(f" Loaded {len(dataloader.dataset)} test samples")
    
    # Run inference on generated data
    print(" Running inference on generated data...")
    generated_results = run_inference(model, dataloader, config, device, mode=config["mode"])
    
    # Prepare real results for distribution metrics
    print(" Preparing real data for comparison...")
    real_results = []
    with torch.no_grad():
        for batch in dataloader:
            real_results.append({
                "filename": batch["filename"][0],
                "ground_truth": batch["post_img"].cpu(),
                "pre_contrast": batch["pre_img"].cpu()
            })
    
    # Save sample-level results
    print(" Saving sample-level results...")
    metrics_df = save_results(generated_results, config["output_dir"], config)
    
    # Compute distribution metrics
    print(" Computing distribution metrics...")
    dist_metrics = compute_distribution_metrics(
        real_results, 
        generated_results, 
        config["output_dir"], 
        config, 
        device
    )
    
    print(f"\n Distribution Metrics:")
    for metric, value in dist_metrics.items():
        if 'Baseline' not in metric:  # Don't print baselines here
            print(f"{metric}: {value:.4f}")
    
    print(f"\n Inference complete! Results saved to {config['output_dir']}")

if __name__ == "__main__":
    main()
