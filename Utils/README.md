utils/ - Utility Modules
This directory contains core utility modules for data handling and configuration used across the project for training conditional diffusion models on breast MRI data.

Modules Overview
1. data_loading.py
This module provides the foundational classes and functions for loading and preprocessing standard paired image datasets (pre-contrast and post-contrast images).

Key Components:
TrainingConfig (dataclass):
A centralized configuration class that holds all critical hyperparameters and paths for training. This promotes reproducibility and easy experimentation.

image_size: Target size for input images (e.g., 256x256).

train_batch_size, eval_batch_size: Batch sizes for training and evaluation.

num_epochs: Number of training epochs.

learning_rate, lr_warmup_steps: Optimizer settings.

output_dir, best_model_dir: Paths for saving outputs and the best model.

mixed_precision: Whether to use FP16 training for speed and memory efficiency.

num_train_timesteps: The number of diffusion timesteps (T).

SafePairedPNGSliceDataset (Dataset):
A PyTorch Dataset class for loading pairs of pre-contrast and post-contrast 2D MRI slices.

Safety Feature: Intelligently checks for and only loads PNG files that exist in both the pre-contrast and post-contrast directories, preventing errors from mismatched data.

Preprocessing: Automatically converts images to grayscale, transforms them into PyTorch tensors, and normalizes pixel values from [0, 255] to the range [-1, 1] expected by the diffusion model.

Transforms: Applies a consistent transformation (e.g., resizing) to both images in the pair.

get_dataloaders(config) (Function):
A convenience function that initializes the SafePairedPNGSliceDataset for both training and testing splits based on the provided TrainingConfig. It returns the corresponding PyTorch DataLoader objects, which handle batching and shuffling.

Usage:
This module is used for training vanilla models (PC_Vanilla, SUB_Vanilla) that do not require tumor mask information.

2. masked_data_loading.py
This module extends the functionality of data_loading.py to handle datasets that include expert tumor segmentation masks. It is essential for training the ROI-aware and ROI-mask conditioned model variants described in the paper.

Key Components:
FastSliceDataset (Dataset):
A PyTorch Dataset class for loading triplets of data: (pre-contrast image, post-contrast image, tumor segmentation mask).

Safety Feature: Ensures that a common PNG file exists across all three directories (pre, post, and mask) before loading.

Preprocessing: Processes the images identically to SafePairedPNGSliceDataset (conversion, normalization to [-1, 1]).

Mask Processing: Loads the mask and binarizes it (values > 0 become 1.0, else 0.0). The mask is also resized to match the image dimensions.

Configuration and Instantiation:
The code at the bottom of the file demonstrates how to instantiate the dataset and dataloaders. It uses the same TrainingConfig dataclass for consistency and creates DataLoader objects for both training and testing.

Usage:
This module is used for training model variants such as:

PC-ROI_(M): Post-contrast model with the mask concatenated as an input channel.

SUB-ROI_(L): Subtraction-based model trained with a tumor-aware loss function (which requires the mask for calculating the loss within the ROI).

How They Fit Into the Project
These data loaders are imported by the training scripts in the root training/ directory (e.g., train_PC_Vanilla.py, train_SUB_ROI_L.py). The choice of which data loader to use is determined by the specific model being trained:

Vanilla Models -> Import and use get_dataloaders from data_loading.py.

ROI-aware/Mask-conditioned Models -> Import and use FastSliceDataset and DataLoader from masked_data_loading.py.
