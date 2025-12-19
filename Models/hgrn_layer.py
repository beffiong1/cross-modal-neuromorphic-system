"""
Hierarchical Gated Recurrent Network (HGRN)
Temporal gating for sequential processing
"""

import torch
import torch.nn as nn

class ImprovedHGRNGate(nn.Module):
    """
    Hierarchical Gated Recurrent Network Gate
    
    Improvements:
    ✅ LayerNorm: Stabilizes recurrent dynamics
    ✅ Xavier Initialization: Better gradient flow
    
    This provides temporal gating (like GRU) to decide what information
    to keep from previous timesteps and what to update.
    """
    
    def __init__(self, input_size=512, hidden_size=512):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Reset gate
        self.W_r = nn.Linear(input_size + hidden_size, hidden_size)
        nn.init.xavier_uniform_(self.W_r.weight)
        
        # Update gate
        self.W_z = nn.Linear(input_size + hidden_size, hidden_size)
        nn.init.xavier_uniform_(self.W_z.weight)
        
        # Candidate hidden state
        self.W_h = nn.Linear(input_size + hidden_size, hidden_size)
        nn.init.xavier_uniform_(self.W_h.weight)
        
        # ✅ LayerNorm for stability
        self.ln_r = nn.LayerNorm(hidden_size)
        self.ln_z = nn.LayerNorm(hidden_size)
        self.ln_h = nn.LayerNorm(hidden_size)
    
    def forward(self, x, h_prev):
        """
        Args:
            x: (batch_size, input_size) - Current input
            h_prev: (batch_size, hidden_size) - Previous hidden state
        Returns:
            h_new: (batch_size, hidden_size) - Updated hidden state
        """
        # Concatenate input and previous hidden state
        combined = torch.cat([x, h_prev], dim=1)
        
        # Reset gate (what to forget)
        r = torch.sigmoid(self.ln_r(self.W_r(combined)))
        
        # Update gate (how much to update)
        z = torch.sigmoid(self.ln_z(self.W_z(combined)))
        
        # Candidate hidden state
        combined_reset = torch.cat([x, r * h_prev], dim=1)
        h_tilde = torch.tanh(self.ln_h(self.W_h(combined_reset)))
        
        # New hidden state (weighted combination)
        h_new = (1 - z) * h_prev + z * h_tilde
        
        return h_new


# Test HGRN gate