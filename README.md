# Conditional Diffusion Models for Breast MRI Synthesis

[![arXiv](https://img.shields.io/badge/arXiv-2508.13776-b31b1b.svg)](https://arxiv.org/abs/2508.13776)
[![MICCAI 2025](https://img.shields.io/badge/MICCAI-2025-8A2BE2.svg)](https://miccai2025.org/)
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
