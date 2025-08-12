# PreProcessing

This folder contains scripts to prepare the [MAMA-MIA](https://doi.org/10.1038/s41597-025-04707-4) breast MRI dataset for training the conditional diffusion models described in our work.  
The preprocessing includes downloading selected patient data, extracting relevant contrast phases, and slicing the volumes into tumor and non-tumor 2D images.


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


## ⚙️ Usage

### 1. Download the dataset

- **`mamamia_dataset.py`**  

Requirements:
- A valid Synapse account and access to the MAMA-MIA dataset
- Set your Synapse personal access token in mamamia_dataset.py

This will create:

MAMA-MIA/
├── images/
│   ├── precontrast/
│   └── postcontrast/
├── segmentations_expert/
└── metadata/


### 2. Slice and preprocess images

python slicing_preprocessing.py

bilateral_slices/
├── train/

│   ├── precontrast/

│   └── postcontrast/

└── test/
    ├── precontrast/
    └── postcontrast/


unilateral_slices/

├── train/

│   ├── precontrast/

│   └── postcontrast/

└── test/
    ├── precontrast/
    └── postcontrast/

metadata/
└── tumor_slices.csv

📝 Notes
- The preprocessing assumes data is in axial orientation and matches the MAMA-MIA naming convention.
- Only patients with expert segmentations are processed.
- The proportion of non-tumor slices can be adjusted in slicing_preprocessing.py (0.2 = 20%).
- All output images are normalized per slice and saved as 8-bit PNG.

