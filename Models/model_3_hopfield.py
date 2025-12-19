"""
Model 3: Baseline + Hopfield Networks
Best visual: 97.68%, Worst audio: 76.15%
21.53% modality gap
"""

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from .hopfield_layer import ModernHopfieldLayer

spike_grad = surrogate.fast_sigmoid(slope=25)

class Model_3_SHD(nn.Module):
    """SNN + Hopfield"""
    def __init__(self, input_size=700, num_classes=20):
        super().__init__()
        self.backbone = SNN_Backbone_SHD(input_size, num_classes)
        self.hopfield = ModernHopfieldLayer(input_size=512, memory_size=256, temperature=0.1)
        self.fc_out = nn.Linear(512, num_classes)
        self.lif_out = snn.Leaky(beta=0.9, spike_grad=spike_grad, init_hidden=True, output=True)
        self.name = "SNN_Hopfield_SHD"
    
    def forward(self, x):
        _, features = self.backbone(x)
        retrieved_features = self.hopfield(features)
        spk_out, _ = self.lif_out(self.fc_out(retrieved_features))
        return spk_out, retrieved_features

