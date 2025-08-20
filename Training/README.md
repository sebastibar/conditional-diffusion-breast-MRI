# `training/` - Model Training Scripts

This directory contains the core scripts for training the various Denoising Diffusion Probabilistic Models (DDPMs) described in the paper. Each script is designed to train a specific model variant, differing in their **learning target** (Post-Contrast image or Subtraction map) and **training strategy** (Vanilla, ROI-aware Loss, or Mask-conditioned Input).

## 🧠 Script Overview & Purpose

The training scripts are named using a clear convention: `train_<TARGET>_<STRATEGY>.py`

| Script | Target (TARGET) | Strategy (STRATEGY) | Description |
| :--- | :--- | :--- | :--- |
| `train_PC.py` | **P**ost-**C**ontrast Image | Vanilla | Predicts the full post-contrast image directly. |
| `train_SUB.py` | **SUB**traction Map | Vanilla | Predicts the contrast enhancement map `(post - pre) / 0.5`. |
| `train_PC_ROI_L.py` | Post-Contrast Image | **R**egion-**O**f-**I**nterest **L**oss | Uses a tumor-aware loss function to improve lesion fidelity. |
| `train_SUB_ROI_L.py` | Subtraction Map | **R**egion-**O**f-**I**nterest **L**oss | Uses a tumor-aware loss function on the SUB target. |
| `train_PC_ROI_M.py` | Post-Contrast Image | **M**ask-conditioned Input | Uses the tumor mask as a direct input channel to the model. |
| `train_SUB_ROI_M.py` | Subtraction Map | **M**ask-conditioned Input | Uses the tumor mask as a direct input channel for SUB prediction. |

### Key Concepts:

*   **Vanilla:** Standard training with a loss computed over the entire image.
*   **ROI-aware Loss (L):** The model receives only the pre-contrast image but the loss function is weighted to prioritize accuracy within the tumor region (requires mask for loss calculation, **not** for model input).
*   **Mask-conditioned Input (M):** The tumor segmentation mask is concatenated as an input channel to the model, explicitly guiding it to focus on the relevant area (requires mask for both training and inference).

## 📁 File Descriptions

### 1. Core Training Scripts

#### **`train_PC.py` & `train_SUB.py`**
These are the foundational scripts for training the vanilla models.
*   **Input:** `[noisy_image, pre_contrast_image]`
*   **Output:** `predicted_image` (PC) or `predicted_subtraction_map` (SUB)
*   **Loss:** Weighted combination of L1, Perceptual (VGG), Total Variation, and MSE losses computed on the entire image.
*   **Usage:** Baseline models. Use these to establish a performance benchmark.

#### **`train_PC_ROI_L.py` & `train_SUB_ROI_L.py`**
These scripts implement tumor-aware training through a specialized loss function.
*   **Input:** `[noisy_image, pre_contrast_image]` (same as vanilla)
*   **Output:** `predicted_image` (PC) or `predicted_subtraction_map` (SUB)
*   **Loss:** A complex, multi-component loss:
    *   **Global Loss:** Standard loss over the full image.
    *   **ROI Loss:** The same losses computed *only within the tumor region*.
    *   **Specialized ROI Losses:** Contrast difference loss and mean intensity loss within the tumor.
*   **Usage:** Best for scenarios where you want improved tumor enhancement but **do not have** tumor masks available at inference time (e.g., screening).

#### **`train_PC_ROI_M.py` & `train_SUB_ROI_M.py`**
These scripts train models that use the tumor mask as a direct input for conditioning.
*   **Input:** `[noisy_image, pre_contrast_image, tumor_mask]`
*   **Output:** `predicted_image` (PC) or `predicted_subtraction_map` (SUB)
*   **Loss:** Standard global loss (L1, Perceptual, TV, MSE). The model learns to use the mask input to focus its reconstruction.
*   **Usage:** Ideal for applications where tumor location is known *a priori*, such as treatment monitoring, response assessment, or longitudinal studies. **Requires a mask during inference.**

### 2. Support Modules

#### **`utils/plot_utils.py`**
Contains the `plot_training_curves()` function used by all scripts to generate consistent plots of metrics and loss components after training completes.

## 🚀 How to Use

1.  **Ensure Data is Prepared:** Your data must be preprocessed and organized according to the structure specified in `utils/README.md`. For ROI scripts, ensure the `masks/` folders are populated.

2.  **Install Dependencies:** All required packages are listed in the main project `README.md` and imported at the top of each script.

3.  **Run a Training Script:** Execute the desired script from the project root directory.

    ```bash
    # Example: Train the vanilla Post-Contrast model
    python training/train_PC.py

    # Example: Train the Mask-conditioned Subtraction model
    python training/train_SUB_ROI_M.py
    ```

4.  **Monitor Output:** The scripts will:
    *   Print progress logs to the console.
    *   Save the best model (using EMA weights) to the `best-model/` directory.
    *   Save sample generated images every epoch to the `output_dir/`.
    *   Display evaluation metrics (MAE, MSE) at the end of each epoch.
    *   Generate comprehensive plots of training curves and metrics upon completion.

## ⚙️ Configuration

All training parameters are controlled by the `TrainingConfig` dataclass imported from `utils.data_loading` or `utils.masked_data_loading`. Key parameters include:
*   `image_size`: Input image resolution (e.g., 256).
*   `train_batch_size`, `eval_batch_size`: Batch sizes.
*   `num_epochs`: Number of training epochs.
*   `learning_rate`: Optimizer learning rate.
*   `num_train_timesteps`: Number of diffusion steps (T).
*   `output_dir`: Directory for saved samples and logs.
*   `best_model_dir`: Directory where the best model checkpoint is saved.

To modify these, edit the `config` object instantiated at the beginning of each script's `main()` function.

## 🔧 Key Functions

*   **`main()`:** The primary function that sets up the config, data, model, and runs the training loop.
*   **`generate_sample()`:** Function used to visualize a generated sample from the validation set during training. Different for PC and SUB targets.
*   **`evaluate_batch()`:** Computes metrics (MAE, MSE) for a batch of predictions (imported from `utils.losses_metrics`).

## 📈 Outputs

Each training run produces:
1.  **Console Logs:** Epoch-wise loss and metrics.
2.  **Model Checkpoint:** The best model weights (`pytorch_model.bin`) saved in `best_model_dir`.
3.  **Sample Images:** PNG files showing generated samples, saved every epoch in `output_dir`.
4.  **Training Plots:** Displayed at the end of training:
    *   Normalized metric trends.
    *   Raw metric evolution.
    *   Loss components over time.

## 🧪 Choosing the Right Script

| Your Application | Recommended Script |
| :--- | :--- |
| General contrast synthesis benchmark | `train_PC.py` or `train_SUB.py` |
| Improved lesion synthesis without inference masks | `train_PC_ROI_L.py` or `train_SUB_ROI_L.py` |
| Lesion-focused synthesis **with** known inference masks | `train_PC_ROI_M.py` or `train_SUB_ROI_M.py` |

**Note:** The paper found that **subtraction-based (SUB) models consistently outperformed** their post-contrast (PC) counterparts.
