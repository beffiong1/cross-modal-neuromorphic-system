"""
Model 4: Baseline + HGRN
Balanced performance: 97.48% visual, 80.08% audio
"""

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

from .backbones import build_backbone
from .hgrn_layer import ImprovedHGRNGate

spike_grad = surrogate.fast_sigmoid(slope=25)


class Model_4_HGRN(nn.Module):
    """Model 4: Baseline + HGRN gate."""

    def __init__(self, input_type='nmnist', input_channels=2, input_size=700, num_classes=10):
        super().__init__()
        self.input_type = input_type
        self.backbone = build_backbone(
            input_type=input_type,
            input_channels=input_channels,
            input_size=input_size,
            num_classes=num_classes,
        )
        self.hgrn = ImprovedHGRNGate(input_size=512, hidden_size=512)
        self.fc_out = nn.Linear(512, num_classes)
        self.lif_out = snn.Leaky(beta=0.9, spike_grad=spike_grad, init_hidden=True, output=True)
        self.name = f"SNN_HGRN_{input_type}"

    def forward(self, x):
        _, features = self.backbone(x)
        batch_size = features.shape[0]
        h = torch.zeros(batch_size, 512).to(features.device)
        h = self.hgrn(features, h)
        spk_out, _ = self.lif_out(self.fc_out(h))
        return spk_out, h
