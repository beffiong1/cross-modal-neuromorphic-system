"""
Model 1: Baseline SNN
Standard spiking neural network with LIF neurons
No memory augmentation - TRUE baseline for comparison
"""

import torch.nn as nn

from .backbones import build_backbone


class Model_1_Baseline(nn.Module):
    """
    Model 1: TRUE BASELINE (NO Contrastive Loss)
    
    Just the SNN backbone trained with CrossEntropy only.
    This establishes baseline performance without any improvements.
    
    Performance:
    - N-MNIST: 96.77%
    - SHD: 80.04%
    - Average: 88.40%
    """
    
    def __init__(self, input_type='nmnist', input_channels=2, input_size=700, num_classes=10):
        super().__init__()
        self.input_type = input_type
        self.backbone = build_backbone(
            input_type=input_type,
            input_channels=input_channels,
            input_size=input_size,
            num_classes=num_classes,
        )
        
        self.name = f"Baseline_NoSCL_{input_type}"
    
    def forward(self, x):
        return self.backbone(x)
