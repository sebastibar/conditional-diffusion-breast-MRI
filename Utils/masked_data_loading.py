import os
from dataclasses import dataclass
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F

@dataclass
class TrainingConfig:
    image_size: int = 256
    train_batch_size: int = 8
    eval_batch_size: int = 8
    num_epochs: int = 50
    learning_rate: float = 5e-6
    lr_warmup_steps: int = 500
    output_dir: str = "ddpm-mri"
    best_model_dir: str = "best-model"
    mixed_precision: str = "fp16"
    seed: int = 42
    num_train_timesteps: int = 250

class FastSliceDataset(Dataset):
    def __init__(self, pre_folder, post_folder, mask_folder, transform=None, img_size=256):
        self.pre_folder = pre_folder
        self.post_folder = post_folder
        self.mask_folder = mask_folder
        self.transform = transform
        self.img_size = img_size

        pre_files = {f for f in os.listdir(pre_folder) if f.lower().endswith(".png")}
        post_files = {f for f in os.listdir(post_folder) if f.lower().endswith(".png")}
        mask_files = {f for f in os.listdir(mask_folder) if f.lower().endswith(".png")}

        self.common_files = sorted(list(pre_files & post_files & mask_files))
        if not self.common_files:
            raise ValueError(f"No matching PNG files across {pre_folder}, {post_folder}, and {mask_folder}")

    def __len__(self):
        return len(self.common_files)

    def __getitem__(self, idx):
        filename = self.common_files[idx]
        
        # Load images
        pre_img = Image.open(os.path.join(self.pre_folder, filename)).convert("L")
        post_img = Image.open(os.path.join(self.post_folder, filename)).convert("L")
        mask_img = Image.open(os.path.join(self.mask_folder, filename)).convert("L")

        # Convert to tensor and normalize pre/post
        pre_tensor = transforms.functional.to_tensor(pre_img)
        pre_tensor = (pre_tensor - 0.5) / 0.5 # [-1, 1]

        post_tensor = transforms.functional.to_tensor(post_img)
        post_tensor = (post_tensor - 0.5) / 0.5 # [-1, 1]

        # Convert mask to binary tensor
        tumor_mask = transforms.functional.to_tensor(mask_img)
        tumor_mask = (tumor_mask > 0).float()  # 0.0 or 1.0

        # Apply transforms (e.g., resize) if provided
        if self.transform:
            pre_tensor = self.transform(pre_tensor)
            post_tensor = self.transform(post_tensor)
            tumor_mask = self.transform(tumor_mask)
        else:
            # Ensure mask is resized to match the expected image size if no transform is applied
            if tumor_mask.shape[-2:] != (self.img_size, self.img_size):
                tumor_mask = F.interpolate(tumor_mask.unsqueeze(0), size=(self.img_size, self.img_size), mode='nearest').squeeze(0)

        return {"pre_img": pre_tensor, "post_img": post_tensor, "tumor_mask": tumor_mask, "filename": filename}


def get_masked_dataloaders(config: TrainingConfig):
    """Creates DataLoaders for the dataset with masks."""
    preprocess = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size))
    ])

    train_dataset = FastSliceDataset(
        pre_folder="bilateral_slices/train/precontrast",
        post_folder="bilateral_slices/train/postcontrast",
        mask_folder="bilateral_slices/train/masks",
        transform=preprocess,
        img_size=config.image_size
    )

    test_dataset = FastSliceDataset(
        pre_folder="bilateral_slices/test/precontrast",
        post_folder="bilateral_slices/test/postcontrast",
        mask_folder="bilateral_slices/test/masks",
        transform=preprocess,
        img_size=config.image_size
    )

    train_loader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config.eval_batch_size, shuffle=False, num_workers=2, pin_memory=True)

    print(f"Loaded {len(train_dataset)} training samples (with masks).")
    print(f"Loaded {len(test_dataset)} testing samples (with masks).")

    return train_loader, test_loader
