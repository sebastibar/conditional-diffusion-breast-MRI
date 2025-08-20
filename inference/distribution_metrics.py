import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import inception_v3
from PIL import Image
import numpy as np
from scipy.linalg import sqrtm
import os
from tqdm import tqdm
import pandas as pd

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Calculate the Fréchet distance between two multivariate Gaussians.
    
    Args:
        mu1: Mean of first distribution
        sigma1: Covariance matrix of first distribution
        mu2: Mean of second distribution
        sigma2: Covariance matrix of second distribution
        eps: Small regularization term for numerical stability
    
    Returns:
        Frechet distance
    """
    covmean = sqrtm(sigma1 @ sigma2)
    if not np.isfinite(covmean).all():
        print("⚠ sqrtm produced NaNs — adding regularization")
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = sqrtm((sigma1 + offset) @ (sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    diff = mu1 - mu2
    return np.sum(diff**2) + np.trace(sigma1 + sigma2 - 2 * covmean)

class InceptionFeatureExtractor:
    """Class to extract features using Inception v3 for FID calculation."""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.preprocess = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.model = self._load_inception()
        
    def _load_inception(self):
        """Load and configure Inception v3 model for feature extraction."""
        model = inception_v3(pretrained=True).to(self.device)
        model.fc = torch.nn.Identity()  # Remove final classification layer
        model.AuxLogits = None  # Disable auxiliary outputs
        model.eval()
        return model
    
    def get_features(self, images):
        """
        Extract features from a list of PIL Images.
        
        Args:
            images: List of PIL Images
            
        Returns:
            numpy array of features
        """
        features = []
        for img in tqdm(images, desc="Extracting Inception features"):
            x = self.preprocess(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feat = self.model(x).cpu().numpy().reshape(-1)
                features.append(feat)
        return np.array(features)

def compute_fid(real_imgs, fake_imgs, device='cuda'):
    """
    Compute Fréchet Inception Distance (FID) between real and generated images.
    
    Args:
        real_imgs: List of PIL Images (real data)
        fake_imgs: List of PIL Images (generated data)
        device: Device to run inference on
        
    Returns:
        FID score
    """
    extractor = InceptionFeatureExtractor(device)
    
    real_feats = extractor.get_features(real_imgs)
    fake_feats = extractor.get_features(fake_imgs)
    
    mu_real = np.mean(real_feats, axis=0)
    sigma_real = np.cov(real_feats, rowvar=False)
    mu_fake = np.mean(fake_feats, axis=0)
    sigma_fake = np.cov(fake_feats, rowvar=False)
    
    return calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)

def compute_frd(real_features, fake_features):
    """
    Compute Fréchet Radiomics Distance (FRD) between real and generated radiomic features.
    
    Args:
        real_features: DataFrame or array of radiomic features from real data
        fake_features: DataFrame or array of radiomic features from generated data
        
    Returns:
        FRD score
    """
    if isinstance(real_features, pd.DataFrame):
        real_features = real_features.values
    if isinstance(fake_features, pd.DataFrame):
        fake_features = fake_features.values
    
    mu_real = np.mean(real_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    mu_fake = np.mean(fake_features, axis=0)
    sigma_fake = np.cov(fake_features, rowvar=False)
    
    return calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)

def compute_baseline_fids(pre_folder, post_folder, device='cuda'):
    """
    Compute baseline FID scores for reference.
    
    Args:
        pre_folder: Path to pre-contrast images
        post_folder: Path to post-contrast images
        device: Device to run inference on
        
    Returns:
        Dictionary with baseline FID scores
    """
    # Get common files
    common_files = sorted(list(
        set(os.listdir(pre_folder)) & 
        set(os.listdir(post_folder)) & 
        {f for f in os.listdir(pre_folder) if f.lower().endswith(".png")}
    ))
    
    # Load images
    pre_imgs = [Image.open(os.path.join(pre_folder, f)).convert("RGB") for f in common_files]
    post_imgs = [Image.open(os.path.join(post_folder, f)).convert("RGB") for f in common_files]
    
    # Compute subtraction images
    subtr_imgs = []
    for pre, post in zip(pre_imgs, post_imgs):
        pre_arr = np.array(pre.convert("L"), dtype=np.float32)
        post_arr = np.array(post.convert("L"), dtype=np.float32)
        subtr_arr = np.clip(post_arr - pre_arr, 0, 255).astype(np.uint8)
        subtr_rgb = Image.fromarray(subtr_arr).convert("RGB")
        subtr_imgs.append(subtr_rgb)
    
    # Compute baseline FIDs
    print("\n🔬 Computing baseline FIDs...")
    
    print("FID Baseline A: Post vs Pre")
    fid_A = compute_fid(post_imgs, pre_imgs, device=device)
    
    print("FID Baseline B: Subtraction vs Pre")
    fid_B = compute_fid(subtr_imgs, pre_imgs, device=device)
    
    return {
        "Post_vs_Pre": fid_A,
        "Subtraction_vs_Pre": fid_B
    }

def evaluate_distribution_metrics(real_images, generated_images, real_radiomics=None, generated_radiomics=None, device='cuda'):
    """
    Comprehensive distribution-level evaluation.
    
    Args:
        real_images: List of PIL Images (real data)
        generated_images: List of PIL Images (generated data)
        real_radiomics: Radiomic features for real data (optional)
        generated_radiomics: Radiomic features for generated data (optional)
        device: Device to run inference on
        
    Returns:
        Dictionary with all distribution metrics
    """
    results = {}
    
    # Compute FID
    print(" Computing FID...")
    results['FID'] = compute_fid(real_images, generated_images, device=device)
    
    # Compute FRD if radiomic features are provided
    if real_radiomics is not None and generated_radiomics is not None:
        print(" Computing FRD...")
        results['FRD'] = compute_frd(real_radiomics, generated_radiomics)
    
    return results
