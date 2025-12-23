"""
Model 2: Baseline + Supervised Contrastive Learning
Best average cross-modal performance: 89.44%
N-MNIST: 96.72%, SHD: 82.16% (best audio)
"""

import torch.nn as nn

from .backbones import build_backbone


class Model_2_SCL(nn.Module):
    """Model 2: Adds Supervised Contrastive Learning."""

    def __init__(self, input_type='nmnist', input_channels=2, input_size=700, num_classes=10):
        super().__init__()
        self.input_type = input_type
        self.backbone = build_backbone(
            input_type=input_type,
            input_channels=input_channels,
            input_size=input_size,
            num_classes=num_classes,
        )
        self.name = f"SCL_{input_type}"

    def forward(self, x):
        return self.backbone(x)
