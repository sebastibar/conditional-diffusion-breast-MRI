import os
import glob
import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

# ==== CONFIGURATION ====
IMG_SIZE = (256, 256)
OUTPUT_BASE = "MAMA-MIA"
PRECONTRAST_DIR = os.path.join(OUTPUT_BASE, "images/precontrast")
POSTCONTRAST_DIR = os.path.join(OUTPUT_BASE, "images/postcontrast")
SEGMENTATIONS_DIR = os.path.join(OUTPUT_BASE, "segmentations_expert")
METADATA_DIR = os.path.join(OUTPUT_BASE, "metadata")
SLICE_CSV = os.path.join(METADATA_DIR, "tumor_slices.csv")

# ==== Load Metadata ====
meta_df = pd.read_excel(os.path.join(METADATA_DIR, "clinical_and_imaging_info.xlsx"))
split_df = pd.read_csv(os.path.join(METADATA_DIR, "train_test_splits.csv"))

# Process split CSV columns (corrected)
train_ids = split_df["train_split"].dropna().astype(str).tolist()
test_ids = split_df["test_split"].dropna().astype(str).tolist()

meta_df["patient_id"] = meta_df["patient_id"].astype(str)

# Add split column based on membership
meta_df["split"] = meta_df["patient_id"].apply(lambda x: "train" if x in train_ids else ("test" if x in test_ids else None))
merged_df = meta_df.dropna(subset=["split"])

# ==== Organize patients ====
def get_patient_groups(df):
    return {
        "bilateral": {
            "train": df[(df["bilateral_mri"] == 1) & (df["split"] == "train")]["patient_id"].tolist(),
            "test": df[(df["bilateral_mri"] == 1) & (df["split"] == "test")]["patient_id"].tolist()
        },
        "unilateral": {
            "train": df[(df["bilateral_mri"] == 0) & (df["split"] == "train")]["patient_id"].tolist(),
            "test": df[(df["bilateral_mri"] == 0) & (df["split"] == "test")]["patient_id"].tolist()
        }
    }

groups = get_patient_groups(merged_df)

# ==== Create folders ====
def ensure_dirs(base_path):
    for subset in ["train", "test"]:
        for contrast in ["precontrast", "postcontrast"]:
            os.makedirs(os.path.join(base_path, subset, contrast), exist_ok=True)

ensure_dirs("bilateral_slices")
ensure_dirs("unilateral_slices")

# ==== Image Preprocessing ====
def normalize_and_resize(slice_2d):
    scaled = (slice_2d - np.min(slice_2d)) / (np.ptp(slice_2d) + 1e-5)
    image = Image.fromarray((scaled * 255).astype(np.uint8))
    return image.resize(IMG_SIZE, resample=Image.BILINEAR)

# ==== Slice Extraction ====
tumor_slice_data = []

def process_patient(patient_id, group_type, split):
    try:
        seg_path = os.path.join(SEGMENTATIONS_DIR, f"{patient_id}.nii.gz")
        pre_path = os.path.join(PRECONTRAST_DIR, f"{patient_id}_0000.nii.gz")
        post_path = os.path.join(POSTCONTRAST_DIR, f"{patient_id}_0001.nii.gz")

        if not (os.path.exists(seg_path) and os.path.exists(pre_path) and os.path.exists(post_path)):
            print(f"Skipping {patient_id}: Missing required files.")
            return

        seg_data = nib.load(seg_path).get_fdata()
        pre_data = nib.load(pre_path).get_fdata()
        post_data = nib.load(post_path).get_fdata()

        assert seg_data.shape == pre_data.shape == post_data.shape

        num_slices = seg_data.shape[2]
        tumor_slices = [i for i in range(num_slices) if np.any(seg_data[:, :, i] > 0)]
        non_tumor_slices = [i for i in range(num_slices) if i not in tumor_slices]

        non_tumor_selected = np.random.choice(
            non_tumor_slices, size=max(1, int(0.2 * len(tumor_slices))),
            replace=False
        ) if non_tumor_slices else []

        selected_slices = [(i, 1) for i in tumor_slices] + [(i, 0) for i in non_tumor_selected]
        selected_slices.sort()

        for slice_idx, has_tumor in selected_slices:
            slice_pre = normalize_and_resize(pre_data[:, :, slice_idx])
            slice_post = normalize_and_resize(post_data[:, :, slice_idx])

            pre_out = os.path.join(f"{group_type}_slices", split, "precontrast", f"{patient_id}_slice{slice_idx:03}.png")
            post_out = os.path.join(f"{group_type}_slices", split, "postcontrast", f"{patient_id}_slice{slice_idx:03}.png")

            slice_pre.save(pre_out)
            slice_post.save(post_out)

            tumor_slice_data.append([patient_id, slice_idx, has_tumor])

    except Exception as e:
        print(f" Error processing {patient_id}: {e}")

# ==== Run all patients ====
for group_type in ["bilateral", "unilateral"]:
    for split in ["train", "test"]:
        print(f"\n Processing {group_type} - {split}")
        for patient_id in tqdm(groups[group_type][split]):
            process_patient(patient_id, group_type, split)

# ==== Save CSV ====
df_out = pd.DataFrame(tumor_slice_data, columns=["Patient_ID", "Slice_Index", "Tumor_Present"])
df_out.to_csv(SLICE_CSV, index=False)
print(f"\n Done! Tumor slice metadata saved to {SLICE_CSV}")
