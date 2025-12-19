"""
Model 1: Baseline SNN
Standard spiking neural network with LIF neurons
No memory augmentation - TRUE baseline for comparison
"""

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

# Surrogate gradient
spike_grad = surrogate.fast_sigmoid(slope=25)


class SNN_Backbone(nn.Module):
    """
    Base SNN architecture for N-MNIST (visual input)
    Uses Conv2D layers for spatial processing
    """
    def __init__(self, input_channels=2, num_classes=10):
        super().__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 64, 3, padding=1)
        self.lif1 = snn.Leaky(beta=0.9, spike_grad=spike_grad, init_hidden=True)
        
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.lif2 = snn.Leaky(beta=0.9, spike_grad=spike_grad, init_hidden=True)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128*34*34, 512)  # 512 feature/memory size
        self.lif3 = snn.Leaky(beta=0.9, spike_grad=spike_grad, init_hidden=True)
        
        self.fc_out = nn.Linear(512, num_classes)
        self.lif_out = snn.Leaky(beta=0.9, spike_grad=spike_grad, init_hidden=True, output=True)

    def forward(self, x):
        # Reset hidden states
        self.lif1.reset_hidden()
        self.lif2.reset_hidden()
        self.lif3.reset_hidden()
        self.lif_out.reset_hidden()

        # x: [Batch, Time, Channels, H, W] -> [Time, Batch, Channels, H, W]
        x = x.permute(1, 0, 2, 3, 4)
        
        spk_out_rec = []
        features_rec = []

        for step in range(x.shape[0]):
            # Conv layers
            spk1 = self.lif1(self.conv1(x[step]))
            spk2 = self.lif2(self.conv2(spk1))
            
            # Flatten and FC
            spk2_flat = spk2.flatten(1)
            spk3 = self.lif3(self.fc1(spk2_flat))
            
            # Output
            spk_out, _ = self.lif_out(self.fc_out(spk3))
            
            spk_out_rec.append(spk_out)
            features_rec.append(spk3)

        # Sum spikes over time for classification
        spk_sum = torch.stack(spk_out_rec, dim=0).sum(dim=0)
        # Average features over time
        features = torch.stack(features_rec, dim=0).mean(dim=0)
        
        return spk_sum, features


class SNN_Backbone_SHD(nn.Module):
    """
    Native SNN backbone for SHD (auditory input)
    Uses Linear (FC) layers for 1D temporal processing
    """
    def __init__(self, input_size=700, num_classes=10):
        super().__init__()
        
        # Three linear layers as specified in paper
        self.fc_input = nn.Linear(input_size, 1024)
        self.lif_input = snn.Leaky(beta=0.9, spike_grad=spike_grad, init_hidden=True)
        
        self.fc1 = nn.Linear(1024, 1024)
        self.lif1 = snn.Leaky(beta=0.9, spike_grad=spike_grad, init_hidden=True)
        
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
        self.lif1.reset_hidden()
        self.lif_hidden.reset_hidden()
        self.lif_out.reset_hidden()

        # [Batch, Time, Features] -> [Time, Batch, Features]
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

        # Sum spikes over time for classification
        spk_sum = torch.stack(spk_out_rec, dim=0).sum(dim=0)
        # Average features over time for contrastive loss
        features = torch.stack(features_rec, dim=0).mean(dim=0)
        
        return spk_sum, features


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
        
        if input_type == 'nmnist':
            self.backbone = SNN_Backbone(input_channels, num_classes)
        elif input_type == 'shd':
            self.backbone = SNN_Backbone_SHD(input_size, num_classes)
        else:
            raise ValueError(f"Unknown input_type: {input_type}")
        
        self.name = f"Baseline_NoSCL_{input_type}"
    
    def forward(self, x):
        return self.backbone(x)
