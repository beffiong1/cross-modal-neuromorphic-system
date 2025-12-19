"""
Modern Hopfield Network Layer
Associative memory for pattern completion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ModernHopfieldLayer(nn.Module):
    """
    Modern Hopfield Layer with continuous states
    
    Improvements:
    ✅ Better Hopfield: Scaled dot-product attention + temperature
    ✅ Skip Connection: Residual connection for gradient flow
    ✅ LayerNorm: Stabilizes training
    
    This acts as associative memory - given a noisy/partial pattern,
    it retrieves the closest stored pattern from memory.
    """
    
    def __init__(self, input_size=512, memory_size=256, temperature=0.1):
        super().__init__()
        
        self.input_size = input_size
        self.memory_size = memory_size
        self.temperature = temperature
        
        # Learnable memory patterns
        self.memory = nn.Parameter(torch.randn(memory_size, input_size))
        nn.init.xavier_uniform_(self.memory)
        
        # Layer normalization
        self.ln = nn.LayerNorm(input_size)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, input_size) - Input features from SNN
        Returns:
            retrieved: (batch_size, input_size) - Retrieved patterns with residual
        """
        # Normalize features and memory
        x_norm = F.normalize(x, p=2, dim=1)
        mem_norm = F.normalize(self.memory, p=2, dim=1)
        
        # ✅ Better Hopfield: Scaled dot-product attention
        similarity = torch.matmul(x_norm, mem_norm.T) / np.sqrt(self.input_size)
        
        # ✅ Better Hopfield: Temperature scaling
        attention_weights = F.softmax(similarity / self.temperature, dim=1)
        
        # Retrieve from memory (weighted sum)
        retrieved = torch.matmul(attention_weights, mem_norm)
        
        # ✅ Skip Connection: Residual + LayerNorm
        retrieved = self.ln(retrieved + x)
        
        return retrieved


# Test Hopfield layer