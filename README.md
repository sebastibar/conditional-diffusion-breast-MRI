# Conditional Diffusion Models for Breast MRI Synthesis

[![arXiv](https://img.shields.io/badge/arXiv-2508.13776-b31b1b.svg)](https://arxiv.org/abs/2508.13776)
[![MICCAI 2025](https://img.shields.io/badge/MICCAI-2025-8A2BE2.svg)](https://deep-breath-miccai.github.io/deepbreath-2025/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

This repository contains the official implementation of our **MICCAI 2025 DeepBreath Workshop** paper:

> **Comparing Conditional Diffusion Models for Synthesizing Contrast-Enhanced Breast MRI from Pre-Contrast Images**  
> Sebastian Ibarra, Javier del Riego, Alessandro Catanese, Julian Cuba, Julian Cardona, Nataly Leon, Jonathan Infante, Karim Lekadir, Oliver Diaz, Richard Osuala  
> *MICCAI 2025 DeepBreath Workshop*  
> [arXiv:2508.13776](https://arxiv.org/abs/2508.13776)

---

## 📖 Abstract

Dynamic contrast-enhanced (DCE) MRI is essential for breast cancer diagnosis but relies on gadolinium-based contrast agents (GBCAs), which pose safety risks, contraindications, and increased costs. This work explores **denoising diffusion probabilistic models (DDPMs)** conditioned on pre-contrast breast MRI to synthesize realistic contrast-enhanced images **without using contrast agents**.

We implement and compare **22 model variants**, including:
- **Post-contrast** and **subtraction-based** diffusion models
- **Tumor-aware loss functions** and **segmentation mask conditioning**
- **Single-breast** and **full-breast** synthesis strategies

Our results show that **subtraction-based models outperform post-contrast models** across multiple metrics. A reader study with radiologists and MRI technologists confirms the **clinical realism** of synthetic images.

---

## 🚀 Features

- ✅ Pre-contrast to post-contrast DCE-MRI synthesis using DDPMs
- ✅ Two conditioning strategies: post-contrast vs. subtraction target
- ✅ Tumor-aware training: loss weighting and mask conditioning
- ✅ Support for both single-breast and full-breast MRI synthesis
- ✅ Comprehensive quantitative evaluation (MAE, SSIM, PSNR, LPIPS, FID, FRD)
- ✅ Expert reader study for clinical validation

---

## 📦 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/sebastibar/conditional-diffusion-breast-MRI.git
cd conditional-diffusion-breast-MRI
pip install -r requirements.txt
```

## 🗂️ Dataset

We use the **MAMA-MIA dataset** ([Garrucho et al., 2025](https://doi.org/10.1038/s41597-025-04515-5)), a public multicenter breast DCE-MRI benchmark.

### Preprocessing

1. Extract axial slices from 3D NIfTI volumes
2. Min-max normalize each slice to [0, 1] and scale to [0, 255]
3. Export as 8-bit PNGs
4. Include slices with tumors and 20% of adjacent non-tumor slices
5. Split into train/test sets for unilateral and bilateral cases

Final dataset: **92,838 paired 2D images** (pre- and post-contrast)

### Data Structure

- **data/raw/nifti/**: Contains original NIfTI files (patient_001.nii.gz, patient_002.nii.gz, etc.)
- **data/raw/masks/**: Contains segmentation masks (patient_001_mask.nii.gz, patient_002_mask.nii.gz, etc.)
- **data/processed/images/**: Contains processed PNG images (patient_001_slice_000_pre.png, patient_001_slice_000_post.png, etc.)
- **data/processed/masks/**: Contains processed mask PNGs (patient_001_slice_000_mask.png, etc.)
- **data/processed/metadata.csv**: Contains patient metadata and file paths
- **data/splits/**: Contains train.csv, val.csv, test.csv split files

### Usage

Preprocess the data:
```bash
python scripts/preprocess.py --input_dir data/raw --output_dir data/processed
```
Create dataset splits:
```bash
python scripts/create_splits.py --data_dir data/processed --output_dir data/splits
```


### Metadata CSV Format

The metadata.csv file contains the following columns:
- patient_id: Unique patient identifier
- slice_index: Slice number
- tumor_label: Binary indicator (1=tumor present, 0=no tumor)
- laterality: Unilateral/bilateral classification
- file_path_pre: Path to pre-contrast image
- file_path_post: Path to post-contrast image
- file_path_mask: Path to segmentation mask

## 🧠 Model Architectures

### 1. Post-Contrast DDPM
- **Input**: Pre-contrast image + noisy post-contrast image at timestep t
- **Target**: Full post-contrast image (x₀)
- **Architecture**: U-Net with residual blocks and bottleneck self-attention layer
- **Conditioning**: 2-channel input (pre-contrast + noisy target)

### 2. Subtraction-Based DDPM
- **Input**: Pre-contrast image + noisy subtraction image at timestep t
- **Target**: Scaled subtraction image (x_post - x_pre)/0.5
- **Reconstruction**: x_post_gen = 0.5 × predicted_subtraction + x_pre
- **Architecture**: Same U-Net architecture as post-contrast model

### 3. Tumor-Aware Variants

#### ROI-Mask Conditioning
- **Input**: Pre-contrast + noisy target + segmentation mask (3 channels)
- **Implementation**: Concatenate expert segmentation mask as additional input channel
- **Use Case**: When tumor location is known a priori (e.g., treatment monitoring)

#### ROI-Aware Loss
- **Focus**: Weighted loss components within tumor region only
- **Components**: 
  - Pixel-wise MAE and MSE within tumor region
  - VGG-based perceptual loss over tumor region
  - Total variation regularization
  - Contrast-specific MAE (penalizes under-enhancement)
  - Intensity loss (preserves mean enhancement)

## 🧪 Training

### Loss Function
Weighted combination of complementary objectives:
- **MAE (30%)**: Pixel-level accuracy
- **Perceptual Loss (60%)**: Structural similarity
- **Total Variation (15%)**: Spatial smoothness
- **MSE (5%)**: Additional pixel consistency

### Optimization
- **Optimizer**: AdamW with decoupled weight decay
- **EMA**: Exponential Moving Average (λ = 0.999)
- **Precision**: Mixed-precision training for efficiency
- **Schedule**: Cosine noise schedule for diffusion steps

### Tumor-Aware Training Strategies

#### ROI-Loss Weighting
```python
# Final loss composition
L_total = 0.3 * L_global + 0.6 * L_roi + 0.05 * L_contrast_mae + 0.05 * L_intensity
```
Contrast-Specific Penalty
Only penalizes under-estimated enhancement signals

Uses ReLU-masked residuals: MAE([pred - pre]+, [target - pre]+)

Training Commands

### Vanilla post-contrast model
python train.py --model_type post_contrast --data_dir data/processed --output_dir results/pc_vanilla

### Vanilla subtraction model  
python train.py --model_type subtraction --data_dir data/processed --output_dir results/sub_vanilla

### With ROI-aware loss
python train.py --model_type subtraction --use_roi_loss --data_dir data/processed --output_dir results/sub_roi_loss

### With mask conditioning
python train.py --model_type post_contrast --use_mask_conditioning --data_dir data/processed --output_dir results/pc_mask_cond

### Full breast training
python train.py --data_dir data/processed --breast_type full --output_dir results/full_breast

### Single breast training
python train.py --data_dir data/processed --breast_type single --output_dir results/single_breast

Training Configuration:
- Batch Size: 16-32 depending on GPU memory
- Epochs: 50-100 with early stopping
- Learning Rate: 1e-4 with cosine decay
- Diffusion Steps: 1000 timesteps
- Gradient Accumulation: Used for larger effective batch sizes

### Monitoring
TensorBoard logging of losses and metrics

Validation set evaluation every epoch

EMA model checkpointing

Synthetic sample generation during training for qualitative monitoring

## 📊 Evaluation

### Quantitative Metrics

We evaluate synthetic image quality using six complementary metrics:

- **MAE (↓)**: Mean Absolute Error - pixel-level accuracy
- **SSIM (↑)**: Structural Similarity Index - structural preservation  
- **PSNR (↑)**: Peak Signal-to-Noise Ratio - signal fidelity
- **LPIPS (↓)**: Learned Perceptual Image Patch Similarity - perceptual quality
- **FID (↓)**: Fréchet Inception Distance - distribution similarity
- **FRD (↓)**: Fréchet Radiomics Distance - radiomic feature distribution

### Evaluation Scripts

```bash
# Evaluate all models on test set
python evaluate.py --test_dir data/splits/test.csv --model_dir results/ --output_dir evaluations/

# Evaluate specific model
python evaluate.py --test_dir data/splits/test.csv --model_path results/subtraction_vanilla/model.pt --output_dir evaluations/sub_vanilla

# Compute metrics for ROI regions only
python evaluate.py --test_dir data/splits/test.csv --model_path results/subtraction_roi/model.pt --roi_only --output_dir evaluations/sub_roi_roi

# Generate qualitative samples
python generate_samples.py --test_samples 50 --model_path results/subtraction_vanilla/model.pt --output_dir samples/sub_vanilla
```
### Reader Study Design

**Participants**: 6 domain experts
- 2 radiologists (11+ and 9+ years experience)
- 4 MRI technologists (10-15+ years experience)

**Three-Part Visual Assessment**:

**Task 1 - Discrimination**:
```python
# 15 mixed images presented individually
task1_images = {
    'synthetic': 10,  # 10 synthetic images
    'real': 5         # 5 real images
}
# Assessment: Distinguish real vs synthetic
```

**Task 2 - Comparison**:
```python
# Randomly selected triplets per case
task2_triplets = {
    'pre_contrast': 1,      # Pre-contrast reference
    'real_post': 1,         # Real post-contrast
    'synthetic_post': 1     # Synthetic post-contrast
}
# Assessment: Identify real image between two post-contrast options
```

**Task 3 - Annotation**:
```python
# Labeled triplets (known real/synthetic)
task3_annotation = {
    'differences': 'annotate visual differences',
    'realism_score': 'score 1-10 scale',
    'diagnostic_relevance': 'qualitative assessment'
}
# Semi-structured follow-up discussion for clinical insights
```

**Evaluation Protocol**:
```python
# Study Setup
study_design = {
    'participants': {
        'radiologists': 2,
        'mri_technologists': 4,
        'experience_range': '9-15+ years'
    },
    'tasks': {
        'task_1': {
            'name': 'Discrimination',
            'images': '15 mixed (10 synthetic, 5 real)',
            'instruction': 'Assess distinguishability real vs synthetic'
        },
        'task_2': {
            'name': 'Comparative',
            'images': 'Triplets (pre-contrast, real post, synthetic post)',
            'instruction': 'Identify real image between two post-contrast options'
        },
        'task_3': {
            'name': 'Annotation',
            'images': 'Labeled triplets',
            'instruction': 'Annotate differences and score realism 1-10'
        }
    },
    'conditions': {
        'blinding': 'Full',
        'randomization': 'Image presentation order',
        'scoring': 'Standardized sheets',
        'session_duration': '30-45 minutes per reader',
        'feedback': 'Semi-structured debriefing'
    }
}
```

