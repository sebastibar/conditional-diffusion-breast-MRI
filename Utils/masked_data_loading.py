class FastSliceDataset(Dataset):
    def __init__(self, pre_folder, post_folder, mask_folder, transform=None):
        self.pre_folder = pre_folder
        self.post_folder = post_folder
        self.mask_folder = mask_folder
        self.transform = transform

        self.common_files = sorted(list(
            set(os.listdir(pre_folder)) &
            set(os.listdir(post_folder)) &
            set(os.listdir(mask_folder))
        ))

        assert len(self.common_files) > 0, "No matching .png filenames across pre, post, and masks."

    def __len__(self):
        return len(self.common_files)

    def __getitem__(self, idx):
        fname = self.common_files[idx]
        pre_img = Image.open(os.path.join(self.pre_folder, fname)).convert("L")
        post_img = Image.open(os.path.join(self.post_folder, fname)).convert("L")
        mask_img = Image.open(os.path.join(self.mask_folder, fname)).convert("L")

        pre_tensor = (transforms.functional.to_tensor(pre_img) - 0.5) / 0.5
        post_tensor = (transforms.functional.to_tensor(post_img) - 0.5) / 0.5
        tumor_mask = (transforms.functional.to_tensor(mask_img) > 0).float()

        if self.transform:
            pre_tensor = self.transform(pre_tensor)
            post_tensor = self.transform(post_tensor)
            tumor_mask = self.transform(tumor_mask)

        return {
            "pre_img": pre_tensor,
            "post_img": post_tensor,
            "tumor_mask": tumor_mask,
            "filename": fname
        }

# === CONFIG ===
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

config = TrainingConfig()

# === TRANSFORM ===
preprocess = transforms.Compose([
    transforms.Resize((config.image_size, config.image_size))
])

# === LOAD DATA ===
train_dataset = FastSliceDataset(
    pre_folder="bilateral_slices/train/precontrast",
    post_folder="bilateral_slices/train/postcontrast",
    mask_folder="bilateral_slices/train/masks",
    transform=preprocess
)

test_dataset = FastSliceDataset(
    pre_folder="bilateral_slices/test/precontrast",
    post_folder="bilateral_slices/test/postcontrast",
    mask_folder="bilateral_slices/test/masks",
    transform=preprocess
)

train_loader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=config.eval_batch_size, shuffle=False, num_workers=2)

print(f"Loaded {len(train_dataset)} training samples.")
print(f"Loaded {len(test_dataset)} testing samples.")