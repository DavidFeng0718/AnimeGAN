import torch
from torch import nn

from feature_modules import HaarResidualFusion
from model64 import Discriminator


class Generator(nn.Module):
    """Base generator with one gated Haar fusion at the 32 x 32 feature stage."""

    def __init__(self, noise_dim=100, frequency_channels=32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.ConvTranspose2d(noise_dim, 256, kernel_size=4),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.to_rgb = nn.Sequential(
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )
        rng_state = torch.get_rng_state()
        self.frequency_fusion = HaarResidualFusion(32, frequency_channels)
        torch.set_rng_state(rng_state)

    def forward(self, inputs):
        features = self.stem(inputs)
        return self.to_rgb(self.frequency_fusion(features))


__all__ = ["Generator", "Discriminator"]
