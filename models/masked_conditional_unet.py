import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.query(x).view(B, -1, H * W).permute(0, 2, 1)
        k = self.key(x).view(B, -1, H * W)
        v = self.value(x).view(B, -1, H * W)
        attn = F.softmax(torch.bmm(q, k), dim=-1)
        out = torch.bmm(v, attn.permute(0, 2, 1))
        out = out.view(B, C, H, W)
        return self.gamma * out + x

class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.dropout2 = nn.Dropout(dropout)

        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        x = F.relu(self.norm1(self.conv1(x)))
        x = self.dropout1(x)
        x = self.norm2(self.conv2(x))
        x = self.dropout2(x)
        return F.relu(x + identity)

# This is the same class as conditional_unet.py, but its purpose is clear from the filename.
# It is intended to be used with in_channels=3 for mask conditioning.
class MaskConditionedUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_channels=64, dropout=0.1):
        super().__init__()
        self.enc1 = ResidualConvBlock(in_channels, base_channels, dropout)
        self.enc2 = ResidualConvBlock(base_channels, base_channels * 2, dropout)
        self.enc3 = ResidualConvBlock(base_channels * 2, base_channels * 4, dropout)

        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ResidualConvBlock(base_channels * 4, base_channels * 8, dropout)
        self.attn = SelfAttention(base_channels * 8)

        self.upconv3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.dec3 = ResidualConvBlock(base_channels * 8, base_channels * 4, dropout)

        self.upconv2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec2 = ResidualConvBlock(base_channels * 4, base_channels * 2, dropout)

        self.upconv1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.dec1 = ResidualConvBlock(base_channels * 2, base_channels, dropout)

        self.final = nn.Sequential(
            nn.Conv2d(base_channels, out_channels, kernel_size=1),
            nn.Tanh()
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        b = self.pool(e3)
        b = self.bottleneck(b)
        b = self.attn(b)

        d3 = self.upconv3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.upconv2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.upconv1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.final(d1)
