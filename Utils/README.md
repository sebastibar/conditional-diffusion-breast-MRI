# `utils/` - Utility Modules

This directory contains core utility modules for data handling and configuration used across the project for training conditional diffusion models on breast MRI data.

## Modules Overview

### 1. `data_loading.py`

This module provides the foundational classes and functions for loading and preprocessing **standard paired image datasets** (pre-contrast and post-contrast images).

#### Key Components:

*   **`TrainingConfig` (dataclass):**
    A centralized configuration class that holds all critical hyperparameters and paths for training.
    *   `image_size`: Target size for input images (e.g., 256x256).
    *   `train_batch_size`, `eval_batch_size`: Batch sizes for training and evaluation.
    *   `num_epochs`: Number of training epochs.
    *   `learning_rate`, `lr_warmup_steps`: Optimizer settings.
    *   `output_dir`, `best_model_dir`: Paths for saving outputs and the best model.
    *   `mixed_precision`: Whether to use FP16 training for speed and memory efficiency.
    *   `num_train_timesteps`: The number of diffusion timesteps (T).

*   **`SafePairedPNGSliceDataset` (Dataset):**
    A PyTorch `Dataset` class for loading pairs of pre-contrast and post-contrast 2D MRI slices.
    *   **Safety Feature:** Only loads PNG files that exist in *both* the pre-contrast and post-contrast directories.
    *   **Preprocessing:** Converts images to grayscale tensors and normalizes pixel values to the range [-1, 1].
    *   **Transforms:** Applies a consistent transformation (e.g., resizing) to both images in the pair.

*   **`get_dataloaders(config)` (Function):**
    A convenience function that initializes the `SafePairedPNGSliceDataset` for training and testing splits based on a `TrainingConfig`. Returns the corresponding PyTorch `DataLoader` objects.

#### Usage:
This module is used for training **vanilla models** (`PC_training`, `SUB_training`) that do not require tumor mask information.

---

### 2. `masked_data_loading.py`

This module extends the functionality to handle datasets that **include expert tumor segmentation masks**. It is essential for training **ROI-aware and ROI-mask conditioned model variants**.

#### Key Components:

*   **`FastSliceDataset` (Dataset):**
    A PyTorch `Dataset` class for loading triplets of data: **(pre-contrast image, post-contrast image, tumor segmentation mask)**.
    *   **Safety Feature:** Ensures a common PNG file exists across all three directories (pre, post, and mask).
    *   **Preprocessing:** Processes images identically to `SafePairedPNGSliceDataset`.
    *   **Mask Processing:** Loads the mask, binarizes it, and resizes it to match the image dimensions.

*   **Configuration and Instantiation:**
    The code demonstrates how to instantiate the dataset and dataloaders using the `TrainingConfig` dataclass.

#### Usage:
This module is used for training model variants such as:
*   `PC-ROI_(M)`: Post-contrast model with the mask concatenated as an input channel.
*   `SUB-ROI_(L)`: Subtraction-based model trained with a tumor-aware loss function.

---

## How They Fit Into the Project

These data loaders are imported by the training scripts in the root `training/` directory. The choice of which data loader to use is determined by the specific model being trained:

*   **Vanilla Models** -> Import and use `get_dataloaders` from `data_loading.py`.
*   **ROI-aware/Mask-conditioned Models** -> Import and use `FastSliceDataset` from `masked_data_loading.py`.

## Expected Data Structure

The loaders expect your preprocessed data to be organized in the following structure. Corresponding slices across folders **must have identical filenames**.


-   **`bilateral_slices/`** (For full-breast models)
    -   `train/` & `test/` splits
        -   `precontrast/`: Pre-contrast input images (e.g., `patient1_slice0.png`)
        -   `postcontrast/`: Target post-contrast images
        -   `masks/`: Tumor segmentation masks (for ROI-aware models)
-   **`unilateral_slices/`** (For single-breast models)
    -   `train/` & `test/` splits
        -   `precontrast/`: Pre-contrast input images
        -   `postcontrast/`: Target post-contrast images
-   **`metadata/`**
    -   `tumor_slices.csv`: CSV file mapping filenames to tumor labels.

**The Golden Rule:** For any given slice (e.g., `patient1_slice0.png`), an identically named file **must** exist in the corresponding `precontrast`, `postcontrast`, and (if used) `masks` folders. This is how the data loader pairs them correctly.
