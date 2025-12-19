"""
SHD (Spiking Heidelberg Digits) Dataset Loader
Spoken digits encoded through artificial cochlea
"""

import torch
import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader

class SHDDataset(Dataset):
    """Spiking Heidelberg Digits Dataset"""
    
    def __init__(self, file_path, train=True):
        self.file_path = file_path
        self.train = train
        
        with h5py.File(file_path, 'r') as f:
            if train:
                self.data = f['train']['spikes'][()]
                self.labels = f['train']['labels'][()]
            else:
                self.data = f['test']['spikes'][()]
                self.labels = f['test']['labels'][()]
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # Convert sparse spike data to dense format
        spike_data = self.data[idx]
        label = self.labels[idx]
        
        # Process spike data (100 time steps, 700 channels)
        # Implementation depends on your specific format
        # This is a placeholder - adjust based on actual data format
        dense_spikes = torch.tensor(spike_data, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        
        return dense_spikes, label

def get_shd_loaders(batch_size=32, file_path='./data/shd_dataset.h5', num_workers=2):
    """
    Get SHD train and test data loaders
    
    Args:
        batch_size: Batch size for training
        file_path: Path to SHD HDF5 file
        num_workers: Number of worker processes
        
    Returns:
        train_loader, test_loader: PyTorch DataLoader objects
    """
    
    train_dataset = SHDDataset(file_path, train=True)
    test_dataset = SHDDataset(file_path, train=False)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False
    )
    
    print(f"✓ SHD loaded: {len(train_dataset)} train, {len(test_dataset)} test")
    return train_loader, test_loader
