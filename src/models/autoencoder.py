from __future__ import annotations

import torch
import torch.nn as nn


class ConvAutoencoder(nn.Module):
    """Small conv AE for 64x64 single-channel wafer maps.

    Trained on 'none' only so reconstruction error is high for unseen defects.
    """

    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.ReLU(inplace=True),  # 32x32
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),  # 16x16
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(inplace=True),  # 8x8
        )
        self.to_latent = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, latent_dim),
        )
        self.from_latent = nn.Sequential(
            nn.Linear(latent_dim, 128 * 8 * 8),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.ReLU(inplace=True),  # 16
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(inplace=True),   # 32
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1), nn.Sigmoid(),             # 64
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.to_latent(self.encoder(x))
        h = self.from_latent(z).view(-1, 128, 8, 8)
        return self.decoder(h)
