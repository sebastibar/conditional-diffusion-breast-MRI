# utils/data_loading.py

import os
from dataclasses import dataclass
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

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
    alpha: float = 0.7
    beta: float = 0.5


class SafePairedPNGSliceDataset(Dataset):

    def __init__(self, pre_folder, post_folder, transform=None):
        pre_files = {f for f in os.listdir(pre_folder) if f.lower().endswith(".png")}
        post_files = {f for f in os.listdir(post_folder) if f.lower().endswith(".png")}

        self.common_files = sorted(list(pre_files & post_files))
        if not self.common_files:
            raise ValueError(f"No matching filenames in {pre_folder} and {post_folder}")

        self.pre_folder = pre_folder
        self.post_folder = post_folder
        self.transform = transform

    def __len__(self):
        return len(self.common_files)

    def __getitem__(self, idx):
        filename = self.common_files[idx]
        
        # Load grayscale
        pre_img = Image.open(os.path.join(self.pre_folder, filename)).convert("L")
        post_img = Image.open(os.path.join(self.post_folder, filename)).convert("L")

        # Convert to tensor [0,1]
        pre_img = transforms.functional.to_tensor(pre_img)
        post_img = transforms.functional.to_tensor(post_img)

        # Normalize to [-1, 1]
        pre_img = (pre_img - 0.5) / 0.5
        post_img = (post_img - 0.5) / 0.5

        # Apply optional transform (e.g., resize)
        if self.transform:
            pre_img = self.transform(pre_img)
            post_img = self.transform(post_img)

        return {"pre_img": pre_img, "post_img": post_img}


def get_dataloaders(config: TrainingConfig):

    preprocess = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size))
    ])

    train_dataset = SafePairedPNGSliceDataset(
        pre_folder="bilateral_slices/train/precontrast",
        post_folder="bilateral_slices/train/postcontrast",
        transform=preprocess
    )

    test_dataset = SafePairedPNGSliceDataset(
        pre_folder="bilateral_slices/test/precontrast",
        post_folder="bilateral_slices/test/postcontrast",
        transform=preprocess
    )

    train_loader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=config.eval_batch_size, shuffle=False, num_workers=2)

    print(f"Loaded {len(train_dataset)} training pairs.")
    print(f"Loaded {len(test_dataset)} testing pairs.")

    return train_loader, test_loader
