"""
Model 2: Baseline + Supervised Contrastive Learning
Best average cross-modal performance: 89.44%
N-MNIST: 96.72%, SHD: 82.16% (best audio)
"""

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

spike_grad = surrogate.fast_sigmoid(slope=25)

# Visual backbone (Conv2D)
class SNN_Backbone(nn.Module):
    """SNN backbone for N-MNIST"""
    def __init__(self, input_channels=2, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, 3, padding=1)
        self.lif1 = snn.Leaky(beta=0.9, spike_grad=spike_grad, init_hidden=True)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.lif2 = snn.Leaky(beta=0.9, spike_grad=spike_grad, init_hidden=True)
        self.fc1 = nn.Linear(128*34*34, 512)
        self.lif3 = snn.Leaky(beta=0.9, spike_grad=spike_grad, init_hidden=True)
        self.fc_out = nn.Linear(512, num_classes)
        self.lif_out = snn.Leaky(beta=0.9, spike_grad=spike_grad, init_hidden=True, output=True)

    def forward(self, x):
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
        return torch.stack(spk_out_rec).sum(0), torch.stack(features_rec).mean(0)


# Auditory backbone (3 Linear layers)
class SNN_Backbone_SHD(nn.Module):
    """
    Native SNN backbone for SHD (1D audio input)
    Uses Linear (FC) layers instead of Conv2D
    """
    def __init__(self, input_size=700, num_classes=20):
        super().__init__()
        
        self.fc_input = nn.Linear(input_size, 1024)
        self.lif_input = snn.Leaky(beta=0.9, spike_grad=spike_grad, init_hidden=True)
        
        # Larger Hidden Layers: 1024 -> 512
        self.fc1 = nn.Linear(1024, 1024)
        self.lif3 = snn.Leaky(beta=0.9, spike_grad=spike_grad, init_hidden=True)
        
        self.fc_hidden = nn.Linear(1024, 512)  # 512 feature/memory size
        self.lif_hidden = snn.Leaky(beta=0.9, spike_grad=spike_grad, init_hidden=True)
        
        # Output layer
        self.fc_out = nn.Linear(512, num_classes)
        self.lif_out = snn.Leaky(beta=0.9, spike_grad=spike_grad, init_hidden=True, output=True)

    def forward(self, x):
        # x: (batch, time, 1, 1, 700) -> squeeze to (batch, time, 700)
        if len(x.shape) == 5:
            x = x.squeeze(2).squeeze(2)  # Remove spatial dimensions
        
        # Reset hidden states
        self.lif_input.reset_hidden()
        self.lif3.reset_hidden()
        self.lif_hidden.reset_hidden()
        self.lif_out.reset_hidden()

        # [Batch, Time, Features] -> [Time, Batch, Features]
        x = x.permute(1, 0, 2)
        
        spk_out_rec = []
        features_rec = []

        for step in range(x.shape[0]):
            spk_in = self.lif_input(self.fc_input(x[step]))
            spk3 = self.lif3(self.fc1(spk_in))
            spk_hidden = self.lif_hidden(self.fc_hidden(spk3))
            spk_out, _ = self.lif_out(self.fc_out(spk_hidden))
            
            spk_out_rec.append(spk_out)
            features_rec.append(spk_hidden)

        # Sum spikes over time for classification
        spk_sum = torch.stack(spk_out_rec, dim=0).sum(dim=0)
        # Average features over time for contrastive loss
        features = torch.stack(features_rec, dim=0).mean(dim=0)
        
        return spk_sum, features


# Define all 5 model variants for SHD



class Model_2_SCL(nn.Module):
    """Model 2: Adds Supervised Contrastive Learning"""
    def __init__(self, input_type='nmnist', num_classes=10):
        super().__init__()
        self.input_type = input_type
        if input_type == 'nmnist':
            self.backbone = SNN_Backbone(2, num_classes)
        else:
            self.backbone = SNN_Backbone_SHD(700, num_classes)
        self.name = f"SCL_{input_type}"
    
    def forward(self, x):
        return self.backbone(x)
