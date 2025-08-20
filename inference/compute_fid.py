#!/usr/bin/env python3
"""
Standalone script to compute FID between two directories of images.
Useful for quick comparisons and baseline calculations.
"""

import argparse
from PIL import Image
import os
from inference.distribution_metrics import compute_fid, compute_baseline_fids

def main():
    parser = argparse.ArgumentParser(description='Compute Fréchet Inception Distance (FID) between image directories')
    parser.add_argument('--real_dir', type=str, required=True, help='Directory with real images')
    parser.add_argument('--fake_dir', type=str, required=True, help='Directory with generated/fake images')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--baseline', action='store_true', help='Compute baseline FIDs for breast MRI')
    
    args = parser.parse_args()
    
    if args.baseline:
        # Compute baseline FIDs for breast MRI
        print(" Computing baseline FIDs for breast MRI...")
        baselines = compute_baseline_fids(args.real_dir, args.fake_dir, device=args.device)
        
        print("\n Baseline FID Results:")
        print(f"Post vs Pre: {baselines['Post_vs_Pre']:.4f}")
        print(f"Subtraction vs Pre: {baselines['Subtraction_vs_Pre']:.4f}")
        
    else:
        # Load images from directories
        real_images = []
        fake_images = []
        
        # Get common files
        real_files = [f for f in os.listdir(args.real_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        fake_files = [f for f in os.listdir(args.fake_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        common_files = sorted(list(set(real_files) & set(fake_files)))
        
        print(f" Found {len(common_files)} common images")
        
        # Load images
        for fname in common_files:
            real_img = Image.open(os.path.join(args.real_dir, fname)).convert("RGB")
            fake_img = Image.open(os.path.join(args.fake_dir, fname)).convert("RGB")
            
            real_images.append(real_img)
            fake_images.append(fake_img)
        
        # Compute FID
        print("Computing FID...")
        fid_score = compute_fid(real_images, fake_images, device=args.device)
        
        print(f"\n FID Score: {fid_score:.4f}")

if __name__ == "__main__":
    main()
