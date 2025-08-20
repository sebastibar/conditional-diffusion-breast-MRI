import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights
from torchmetrics import MeanSquaredError, MeanAbsoluteError

# === LOSSES ===

class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg16(weights=VGG16_Weights.DEFAULT).features[:16].eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.criterion = nn.MSELoss()

    def forward(self, pred, target):
        pred_features = self.vgg(pred.repeat(1, 3, 1, 1))
        target_features = self.vgg(target.repeat(1, 3, 1, 1))
        return self.criterion(pred_features, target_features)


def total_variation_loss(x):
    diff_h = torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])
    diff_v = torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])
    return torch.mean(diff_h) + torch.mean(diff_v)


def clamp_and_safe_pred(x):
    return torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1, 1)


# === METRICS (for training) ===

mse_metric = MeanSquaredError().to("cuda")
mae_metric = MeanAbsoluteError().to("cuda")

def evaluate_batch(recon, target):
    with torch.no_grad():
        return {
            'MSE': mse_metric(recon, target).item(),
            'MAE': mae_metric(recon, target).item()
        }
