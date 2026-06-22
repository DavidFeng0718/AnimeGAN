import torch
from torch import nn
from torch.nn.utils import spectral_norm

from model64 import Generator


def spectral_norm_without_rng_drift(module):
    rng_state = torch.get_rng_state()
    wrapped = spectral_norm(module)
    torch.set_rng_state(rng_state)
    return wrapped


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            spectral_norm_without_rng_drift(
                nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
            ),
            nn.LeakyReLU(0.2),
            spectral_norm_without_rng_drift(
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
            ),
            nn.LeakyReLU(0.2),
            spectral_norm_without_rng_drift(
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
            ),
            nn.LeakyReLU(0.2),
            spectral_norm_without_rng_drift(
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
            ),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            spectral_norm_without_rng_drift(nn.Linear(4 * 4 * 256, 256)),
            nn.LeakyReLU(0.2),
            spectral_norm_without_rng_drift(nn.Linear(256, 1)),
        )

    def forward(self, inputs):
        return self.net(inputs).view(-1)


__all__ = ["Generator", "Discriminator"]
