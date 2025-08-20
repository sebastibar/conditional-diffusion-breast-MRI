# `models/` - Neural Network Architectures and Losses

This directory contains the core neural network model definitions and custom loss functions used for training conditional diffusion models to synthesize contrast-enhanced breast MRI.

## Modules Overview

### 1. `conditional_unet.py`

This module defines the primary U-Net architecture used as the denoiser in the Denoising Diffusion Probabilistic Model (DDPM). It is a conditional U-Net, meaning it takes both a noisy image and a conditioning image (the pre-contrast scan) as input to predict the target.

#### Key Components:

*   **`SelfAttention` (Module):**
    Implements a self-attention mechanism, allowing the model to focus on global contextual relationships within the feature maps. This is typically applied in the bottleneck of the U-Net to handle long-range dependencies.

*   **`ResidualConvBlock` (Module):**
    A fundamental building block consisting of two convolutional layers with Group Normalization, Dropout, and a residual skip connection. This promotes stable gradient flow and enables the training of very deep networks.

*   **`ConditionalUNet` (Module):**
    The main U-Net architecture.
    *   **Encoder:** A downsampling path that captures context from the input (composed of `ResidualConvBlock`s and max-pooling).
    *   **Bottleneck:** Processes the most compressed features using a `ResidualConvBlock` and a `SelfAttention` layer.
    *   **Decoder:** An upsampling path that reconstructs the image detail (composed of transposed convolutions and `ResidualConvBlock`s). Features from the encoder are concatenated with the decoder via skip connections to preserve spatial information.
    *   **Final Layer:** A 1x1 convolution followed by a `Tanh` activation, outputting values in the range [-1, 1] to match the normalized target data.

**Input:** A 2-channel tensor `[noisy_image, pre_contrast_condition]`
**Output:** A 1-channel tensor `predicted_image` in the range [-1, 1]

---

### 2. `masked_conditional_unet.py`

**Note:** This file appears to be functionally similar to `conditional_unet.py` based on the code. In a practical implementation, this would be the version designed to accept an *additional input channel* for a tumor segmentation mask.

**Intended Purpose:** This model would be used for **ROI-mask conditioned** variants (e.g., `PC-ROI_(M)`). The input would be a 3-channel tensor: `[noisy_image, pre_contrast_condition, tumor_mask]`, allowing the model to explicitly focus its learning and generation on the anatomically relevant regions.

---

### 3. `losses_metrics.py`

This module implements a suite of loss functions and metrics used to train and evaluate the diffusion models. The training objective is a weighted combination of these losses.

#### Loss Functions:

*   **`PerceptualLoss` (Module):**
    A feature-based loss using a pre-trained VGG-16 network. It compares high-level feature representations of the predicted and target images, encouraging them to be perceptually similar rather than just pixel-wise accurate. This improves the visual realism of the generated images.

*   **`total_variation_loss` (Function):**
    Encourages spatial smoothness in the generated images by penalizing large differences between neighboring pixels. This helps reduce high-frequency noise and artifacts.

*   **`clamp_and_safe_pred` (Function):**
    A utility function to ensure predictions are within the valid range [-1, 1] and to handle any numerical instability (NaN or infinity values) that may occur during training, replacing them with zeros.

#### Training Metrics:

*   **`mse_metric`, `mae_metric`:**
    Instances of TorchMetrics for calculating Mean Squared Error and Mean Absolute Error. These are used for monitoring basic pixel-level reconstruction accuracy during training.
*   **`evaluate_batch` (Function):**
    A convenience function that computes MSE and MAE for a batch of predictions and targets in evaluation mode (with `torch.no_grad()`).

---

## How They Fit Into the Project

*   The **`ConditionalUNet`** from `conditional_unet.py` is the core denoising model imported by training scripts for vanilla, non-mask-conditioned tasks.
*   The **loss functions** in `losses_metrics.py` are imported by training scripts to compute the complex, multi-component loss used to optimize the U-Net.
*   The **`MaskedConditionalUNet`** (to be implemented by modifying the provided class) would be used for experiments involving explicit tumor mask conditioning.

## Model Architecture Summary

| Model File | Input Channels | Output Channels | Primary Use Case |
| :--- | :--- | :--- | :--- |
| `conditional_unet.py` | 2 (`noise + pre_contrast`) | 1 | Vanilla PC and SUB models |
| `masked_conditional_unet.py` | 3 (`noise + pre_contrast + mask`) | 1 | ROI-mask conditioned models |
