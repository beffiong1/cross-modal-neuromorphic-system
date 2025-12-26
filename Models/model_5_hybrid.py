"""
Model 5: Full Hybrid
Combines Hopfield + HGRN + SCL
"""

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

from .backbones import build_backbone
from .hopfield_layer import ModernHopfieldLayer
from .hgrn_layer import ImprovedHGRNGate

spike_grad = surrogate.fast_sigmoid(slope=25)


class Model_5_Hybrid(nn.Module):
    """Model 5: Full hybrid (Hopfield + HGRN)."""

    def __init__(
        self,
        input_type="nmnist",
        input_channels=2,
        input_size=700,
        spatial_size=(34, 34),
        num_classes=10,
    ):
        super().__init__()
        self.input_type = input_type
        self.backbone = build_backbone(
            input_type=input_type,
            input_channels=input_channels,
            input_size=input_size,
            spatial_size=spatial_size,
            num_classes=num_classes,
        )
        self.hopfield = ModernHopfieldLayer(input_size=512, memory_size=256, temperature=0.1)
        self.hgrn = ImprovedHGRNGate(input_size=512, hidden_size=512)
        self.fc_out = nn.Linear(512, num_classes)
        self.lif_out = snn.Leaky(beta=0.9, spike_grad=spike_grad, init_hidden=True, output=True)
        self.name = f"Full_Hybrid_{input_type}"

    def forward(self, x):
        _, features = self.backbone(x)
        retrieved = self.hopfield(features)
        batch_size = retrieved.shape[0]
        h = torch.zeros(batch_size, 512).to(retrieved.device)
        h = self.hgrn(retrieved, h)
        spk_out, _ = self.lif_out(self.fc_out(h))
        return spk_out, h
