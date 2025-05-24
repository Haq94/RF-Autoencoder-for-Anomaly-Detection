# models/Autoencoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # [B, 1, H, W] → [B, 16, H, W]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # → [B, 16, H/2, W/2]
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # → [B, 32, H/4, W/4]
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),  # → [B, 16, H/2, W/2]
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2),   # → [B, 1, H, W]
            nn.Sigmoid()
        )

    def forward(self, x):
        original_size = x.shape[-2:]  # (H, W)
        x = self.encoder(x)
        x = self.decoder(x)
        x = F.interpolate(x, size=original_size, mode='bilinear', align_corners=False)
        return x
