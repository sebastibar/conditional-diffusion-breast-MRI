# `inference/` - Model Inference and Evaluation

This directory contains scripts for generating synthetic images using trained models and evaluating their performance against ground truth data.

## 📁 Script Overview

| Script | Model Type | Input Channels | Description |
| :--- | :--- | :--- | :--- |
| `infer_PC.py` | Vanilla Post-Contrast | 2 (`[noise, pre_img]`) | Inference for models predicting post-contrast images directly. |
| `infer_SUB.py` | Vanilla Subtraction | 2 (`[noise, pre_img]`) | Inference for models predicting subtraction maps. |
| `infer_PC_ROI_L.py` | ROI-Loss Post-Contrast | 2 (`[noise, pre_img]`) | Inference for ROI-aware loss trained PC models. |
| `infer_SUB_ROI_L.py` | ROI-Loss Subtraction | 2 (`[noise, pre_img]`) | Inference for ROI-aware loss trained SUB models. |
| `infer_PC_ROI_M.py` | Mask-Conditioned Post-Contrast | 3 (`[noise, pre_img, mask]`) | Inference for mask-conditioned PC models. |
| `infer_SUB_ROI_M.py` | Mask-Conditioned Subtraction | 3 (`[noise, pre_img, mask]`) | Inference for mask-conditioned SUB models. |

## 🚀 How to Use

1.  **Ensure Models are Trained:** Each script expects a trained model checkpoint at the path specified in its config.

2.  **Run Inference:** Execute the desired script from the project root.

    ```bash
    # Example: Run inference for vanilla Post-Contrast model
    python inference/infer_PC.py

    # Example: Run inference for mask-conditioned Subtraction model  
    python inference/infer_SUB_ROI_M.py
    ```

3.  **View Results:** Each script creates an output directory with:
    -   Generated images in `generated_images/`
    -   Sample comparison visualizations (`sample_comparisons.png`)
    -   Detailed metrics in CSV format (`inference_metrics.csv`)
    -   Summary statistics and distribution plots

## ⚙️ Configuration

Each script has a `config` dictionary at the top that controls:
-   `model_path`: Path to the trained model checkpoint.
-   `pre_folder`, `post_folder`, `mask_folder`: Paths to test data.
-   `output_dir`: Where to save inference results.
-   `img_size`: Image size for inference.
-   `num_inference_timesteps`: Number of diffusion steps for sampling.
-   `use_mask_input`: Whether the model expects a mask as input (for ROI-M models).
-   `num_sample_visualizations`: How many sample images to visualize.

## 📊 Output Metrics

All inference scripts calculate and save the following metrics:
-   **MAE**: Mean Absolute Error
-   **SSIM**: Structural Similarity Index Measure
-   **PSNR**: Peak Signal-to-Noise Ratio
-   **LPIPS**: Learned Perceptual Image Patch Similarity

## 🔧 Core Module

The `core_inference.py` module contains shared functionality:
-   `get_masked_dataloaders_for_inference()`: Loads test data with masks.
-   `run_inference()`: Main function that runs the diffusion sampling process.
-   `save_results()`: Saves generated images, metrics, and visualizations.
-   `create_summary_plots()`: Creates violin plots and summary statistics.

## 📝 Usage Example

```python
# Minimal example of using the core inference function
from inference.core_inference import run_inference, save_results

config = {
    "num_inference_timesteps": 50,
    "use_mask_input": False,
    "num_sample_visualizations": 5
}

results = run_inference(model, dataloader, config, device, mode="PC")
save_results(results, "my_results", config)
