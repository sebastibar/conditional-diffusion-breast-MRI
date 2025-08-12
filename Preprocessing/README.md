# PreProcessing

This folder contains scripts to prepare the [MAMA-MIA](https://doi.org/10.1038/s41597-025-04707-4) breast MRI dataset for training the conditional diffusion models described in our work.  
The preprocessing includes downloading selected patient data, extracting relevant contrast phases, and slicing the volumes into tumor and non-tumor 2D images.

---

## 📂 Contents

- **`mamamia_dataset.py`**  
  Downloads the MAMA-MIA dataset from Synapse, including:
  - Pre-contrast (Phase 0) images
  - Early post-contrast (Phase 1) images
  - Expert segmentation masks
  - Metadata files  
  Only patients with expert-verified tumor segmentations are included.  

- **`slicing_preprocessing.py`**  
  Processes the downloaded 3D NIfTI volumes by:
  - Identifying slices containing tumors (based on expert masks)
  - Selecting 20% of adjacent non-tumor slices for diversity
  - Normalizing, resizing (256×256), and saving slices as PNG
  - Splitting into `train`/`test` and `unilateral`/`bilateral` folders
  - Creating `tumor_slices.csv` with patient ID, slice index, and tumor label

- **`tumor_slices.csv`**  
  Metadata table mapping each extracted slice to:
  - Patient ID
  - Slice index
  - Tumor presence (1 or 0)

---

## ⚙️ Usage

### 1. Download the dataset

```bash
python mamamia_dataset.py

