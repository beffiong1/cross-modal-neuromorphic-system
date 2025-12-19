"""
Baseline SNN Model (Model 1)
Standard spiking neural network with LIF neurons
"""

import torch
import torch.nn as nn
import snntorch as snn

class BaselineSNN(nn.Module):
    """
    Baseline Spiking Neural Network
    
    Architecture for N-MNIST:
        Conv2d(2, 64) → LIF → Conv2d(64, 128) → LIF → FC(512) → LIF → FC(10)
    
    Architecture for SHD:
        Linear(700, 1024) → LIF → Linear(1024, 1024) → LIF → 
        Linear(1024, 512) → LIF → FC(10)
    """
    
    def __init__(self, input_type='nmnist', num_classes=10, beta=0.9):
        super().__init__()
        
        self.input_type = input_type
        self.beta = beta
        
        if input_type == 'nmnist':
            # Visual pathway (N-MNIST)
            self.conv1 = nn.Conv2d(2, 64, kernel_size=3, stride=1, padding=1)
            self.lif1 = snn.Leaky(beta=beta)
            
            self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
            self.lif2 = snn.Leaky(beta=beta)
            
            self.fc1 = nn.Linear(128 * 34 * 34, 512)
            self.lif3 = snn.Leaky(beta=beta)
            
        elif input_type == 'shd':
            # Auditory pathway (SHD) - 3 linear layers
            self.fc_input = nn.Linear(700, 1024)
            self.lif1 = snn.Leaky(beta=beta)
            
            self.fc_hidden = nn.Linear(1024, 1024)
            self.lif2 = snn.Leaky(beta=beta)
            
            self.fc1 = nn.Linear(1024, 512)
            self.lif3 = snn.Leaky(beta=beta)
        
        # Output layer (common)
        self.fc_out = nn.Linear(512, num_classes)
        self.lif_out = snn.Leaky(beta=beta)
    
    def forward(self, x):
        """
        Forward pass through network
        
        Args:
            x: Input tensor [batch, time, channels, height, width] for nmnist
               or [batch, time, channels] for shd
               
        Returns:
            spk_out: Output spikes [time, batch, num_classes]
            features: Feature representations [time, batch, 512]
        """
        
        batch_size = x.size(0)
        num_steps = x.size(1)
        
        # Initialize membrane potentials
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem_out = self.lif_out.init_leaky()
        
        spk_out_rec = []
        features_rec = []
        
        for step in range(num_steps):
            x_step = x[:, step]
            
            if self.input_type == 'nmnist':
                # Visual processing
                cur1 = self.conv1(x_step)
                spk1, mem1 = self.lif1(cur1, mem1)
                
                cur2 = self.conv2(spk1)
                spk2, mem2 = self.lif2(cur2, mem2)
                
                # Flatten
                cur3 = self.fc1(spk2.flatten(1))
                spk3, mem3 = self.lif3(cur3, mem3)
                
            elif self.input_type == 'shd':
                # Auditory processing
                cur1 = self.fc_input(x_step)
                spk1, mem1 = self.lif1(cur1, mem1)
                
                cur2 = self.fc_hidden(spk1)
                spk2, mem2 = self.lif2(cur2, mem2)
                
                cur3 = self.fc1(spk2)
                spk3, mem3 = self.lif3(cur3, mem3)
            
            # Output layer
            cur_out = self.fc_out(spk3)
            spk_out, mem_out = self.lif_out(cur_out, mem_out)
            
            spk_out_rec.append(spk_out)
            features_rec.append(spk3)  # 512-dim features
        
        return torch.stack(spk_out_rec), torch.stack(features_rec)
