#!/usr/bin/env python3
"""
Inference script for Vanilla Post-Contrast model.
"""

import torch
from models.conditional_unet import ConditionalUNet
from inference.core_inference import get_masked_dataloaders_for_inference, run_inference, save_results

# Configuration
config = {
    "model_path": "best-model/pytorch_model.bin",
    "pre_folder": "bilateral_slices/test/precontrast",
    "post_folder": "bilateral_slices/test/postcontrast", 
    "mask_folder": "bilateral_slices/test/masks",
    "output_dir": "inference_results/PC_vanilla",
    "img_size": 256,
    "num_inference_timesteps": 50,
    "use_mask_input": False,
    "num_sample_visualizations": 5,
    "mode": "PC"
}

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = ConditionalUNet(in_channels=2).to(device)
    model.load_state_dict(torch.load(config["model_path"], map_location=device))
    
    # Load data
    dataloader = get_masked_dataloaders_for_inference(
        config["pre_folder"], 
        config["post_folder"],
        config["mask_folder"],
        config["img_size"],
        batch_size=1
    )
    
    # Run inference
    results = run_inference(model, dataloader, config, device, mode=config["mode"])
    
    # Save results
    metrics_df = save_results(results, config["output_dir"], config)
    
    print(f"Inference complete! Results saved to {config['output_dir']}")

if __name__ == "__main__":
    main()
