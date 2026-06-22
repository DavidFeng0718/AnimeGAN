import torch
import torch.nn.functional as F
from torch import nn


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention with a safe hidden width."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive.")
        if reduction <= 0:
            raise ValueError("reduction must be positive.")
        hidden_channels = max(channels // reduction, 1)
        self.channels = channels
        self.reduction = reduction
        self.hidden_channels = hidden_channels
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, inputs):
        if inputs.ndim != 4 or inputs.shape[1] != self.channels:
            raise ValueError(
                f"SEBlock expected NCHW input with {self.channels} channels, "
                f"got shape {tuple(inputs.shape)}."
            )
        scale = self.excitation(self.pool(inputs))
        return inputs * scale


class SobelFeatureExtractor(nn.Module):
    """Fixed depthwise Sobel X/Y filters with reflected boundary context."""

    def __init__(self, channels):
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive.")
        kernels = torch.tensor([
            [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]],
            [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]],
        ]) / 4.0
        self.channels = channels
        self.register_buffer("kernels", kernels.repeat(channels, 1, 1, 1))

    def forward(self, inputs):
        if inputs.ndim != 4 or inputs.shape[1] != self.channels:
            raise ValueError(
                f"SobelFeatureExtractor expected NCHW input with {self.channels} channels, "
                f"got shape {tuple(inputs.shape)}."
            )
        padded = F.pad(inputs, (1, 1, 1, 1), mode="reflect")
        return F.conv2d(padded, self.kernels, groups=self.channels)


class SobelResidualFusion(nn.Module):
    """Encode fixed Sobel features, concatenate them, and learn a gated residual."""

    def __init__(self, channels, edge_channels):
        super().__init__()
        if channels <= 0 or edge_channels <= 0:
            raise ValueError("channels and edge_channels must be positive.")
        self.channels = channels
        self.sobel = SobelFeatureExtractor(channels)
        self.edge_encoder = nn.Sequential(
            nn.Conv2d(channels * 2, edge_channels, kernel_size=1),
            nn.ReLU(),
        )
        self.fusion = nn.Conv2d(channels + edge_channels, channels, kernel_size=1)
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, inputs):
        edges = self.edge_encoder(self.sobel(inputs))
        residual = self.fusion(torch.cat((inputs, edges), dim=1))
        return inputs + torch.tanh(self.gate) * residual


class HaarDWT(nn.Module):
    """Fixed, orthonormal single-level Haar LL/LH/HL/HH decomposition."""

    def __init__(self, channels):
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive.")
        kernels = torch.tensor([
            [[[1.0, 1.0], [1.0, 1.0]]],
            [[[-1.0, -1.0], [1.0, 1.0]]],
            [[[-1.0, 1.0], [-1.0, 1.0]]],
            [[[1.0, -1.0], [-1.0, 1.0]]],
        ]) / 2.0
        self.channels = channels
        self.register_buffer("kernels", kernels.repeat(channels, 1, 1, 1))

    def forward(self, inputs):
        if inputs.ndim != 4 or inputs.shape[1] != self.channels:
            raise ValueError(
                f"HaarDWT expected NCHW input with {self.channels} channels, "
                f"got shape {tuple(inputs.shape)}."
            )
        if inputs.shape[-2] % 2 or inputs.shape[-1] % 2:
            raise ValueError("HaarDWT requires even spatial dimensions.")
        return F.conv2d(inputs, self.kernels, stride=2, groups=self.channels)


class HaarResidualFusion(nn.Module):
    """Encode Haar subbands and fuse them into a gated spatial residual."""

    def __init__(self, channels, frequency_channels):
        super().__init__()
        if channels <= 0 or frequency_channels <= 0:
            raise ValueError("channels and frequency_channels must be positive.")
        self.channels = channels
        self.dwt = HaarDWT(channels)
        self.frequency_encoder = nn.Sequential(
            nn.Conv2d(channels * 4, frequency_channels, kernel_size=1),
            nn.ReLU(),
        )
        self.fusion = nn.Conv2d(channels + frequency_channels, channels, kernel_size=1)
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, inputs):
        frequency = self.frequency_encoder(self.dwt(inputs))
        frequency = F.interpolate(
            frequency,
            size=inputs.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        residual = self.fusion(torch.cat((inputs, frequency), dim=1))
        return inputs + torch.tanh(self.gate) * residual
