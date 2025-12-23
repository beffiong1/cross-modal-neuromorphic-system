"""
Shared SNN backbones for visual (N-MNIST-like) and audio (SHD-like) inputs.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate


spike_grad = surrogate.fast_sigmoid(slope=25)


class VisualBackbone(nn.Module):
    """Conv-based backbone for event vision inputs (e.g., N-MNIST)."""

    def __init__(self, input_channels: int = 2, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, 3, padding=1)
        self.lif1 = snn.Leaky(beta=0.9, spike_grad=spike_grad, init_hidden=True)

        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.lif2 = snn.Leaky(beta=0.9, spike_grad=spike_grad, init_hidden=True)

        self.fc1 = nn.Linear(128 * 34 * 34, 512)
        self.lif3 = snn.Leaky(beta=0.9, spike_grad=spike_grad, init_hidden=True)

        self.fc_out = nn.Linear(512, num_classes)
        self.lif_out = snn.Leaky(beta=0.9, spike_grad=spike_grad, init_hidden=True, output=True)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.lif1.reset_hidden()
        self.lif2.reset_hidden()
        self.lif3.reset_hidden()
        self.lif_out.reset_hidden()

        x = x.permute(1, 0, 2, 3, 4)
        spk_out_rec = []
        features_rec = []

        for step in range(x.shape[0]):
            spk1 = self.lif1(self.conv1(x[step]))
            spk2 = self.lif2(self.conv2(spk1))
            spk3 = self.lif3(self.fc1(spk2.flatten(1)))
            spk_out, _ = self.lif_out(self.fc_out(spk3))

            spk_out_rec.append(spk_out)
            features_rec.append(spk3)

        spk_sum = torch.stack(spk_out_rec, dim=0).sum(dim=0)
        features = torch.stack(features_rec, dim=0).mean(dim=0)
        return spk_sum, features


class AudioBackbone(nn.Module):
    """FC-based backbone for audio inputs (e.g., SHD)."""

    def __init__(self, input_size: int = 700, num_classes: int = 10):
        super().__init__()
        self.fc_input = nn.Linear(input_size, 1024)
        self.lif_input = snn.Leaky(beta=0.9, spike_grad=spike_grad, init_hidden=True)

        self.fc1 = nn.Linear(1024, 1024)
        self.lif1 = snn.Leaky(beta=0.9, spike_grad=spike_grad, init_hidden=True)

        self.fc_hidden = nn.Linear(1024, 512)
        self.lif_hidden = snn.Leaky(beta=0.9, spike_grad=spike_grad, init_hidden=True)

        self.fc_out = nn.Linear(512, num_classes)
        self.lif_out = snn.Leaky(beta=0.9, spike_grad=spike_grad, init_hidden=True, output=True)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(x.shape) == 5:
            x = x.squeeze(2).squeeze(2)

        self.lif_input.reset_hidden()
        self.lif1.reset_hidden()
        self.lif_hidden.reset_hidden()
        self.lif_out.reset_hidden()

        x = x.permute(1, 0, 2)
        spk_out_rec = []
        features_rec = []

        for step in range(x.shape[0]):
            spk_in = self.lif_input(self.fc_input(x[step]))
            spk1 = self.lif1(self.fc1(spk_in))
            spk_hidden = self.lif_hidden(self.fc_hidden(spk1))
            spk_out, _ = self.lif_out(self.fc_out(spk_hidden))

            spk_out_rec.append(spk_out)
            features_rec.append(spk_hidden)

        spk_sum = torch.stack(spk_out_rec, dim=0).sum(dim=0)
        features = torch.stack(features_rec, dim=0).mean(dim=0)
        return spk_sum, features


def build_backbone(
    input_type: str,
    input_channels: int = 2,
    input_size: int = 700,
    num_classes: int = 10,
) -> nn.Module:
    if input_type == "nmnist":
        return VisualBackbone(input_channels=input_channels, num_classes=num_classes)
    if input_type == "shd":
        return AudioBackbone(input_size=input_size, num_classes=num_classes)
    raise ValueError(f"Unknown input_type: {input_type}")
