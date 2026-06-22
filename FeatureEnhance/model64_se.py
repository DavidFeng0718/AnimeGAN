import torch
from torch import nn

from feature_modules import SEBlock
from model64 import Discriminator


class Generator(nn.Module):
    """Base generator with SE channel attention at 16 x 16 and 32 x 32."""

    def __init__(self, noise_dim=100, se_reduction=16):
        super().__init__()
        # Register every shared Base layer before SE so paired initialization keeps
        # the same parameter order as model64.Generator.
        self.block_4 = nn.Sequential(
            nn.ConvTranspose2d(noise_dim, 256, kernel_size=4),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.block_8 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.block_16 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.block_32 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.to_rgb = nn.Sequential(
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

        rng_state = torch.get_rng_state()
        self.se_16 = SEBlock(64, reduction=se_reduction)
        self.se_32 = SEBlock(32, reduction=se_reduction)
        torch.set_rng_state(rng_state)

    def forward(self, inputs):
        features = self.block_4(inputs)
        features = self.block_8(features)
        features = self.se_16(self.block_16(features))
        features = self.se_32(self.block_32(features))
        return self.to_rgb(features)


__all__ = ["Generator", "Discriminator"]
