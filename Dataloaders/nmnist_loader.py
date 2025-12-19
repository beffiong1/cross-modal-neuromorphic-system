"""
N-MNIST Dataset Loader
Event-camera recordings of handwritten digits
"""

import torch
import tonic
from torch.utils.data import DataLoader
import tonic.transforms as transforms

def get_nmnist_loaders(batch_size=32, time_steps=25, num_workers=2):
    """
    Get N-MNIST train and test data loaders
    
    Args:
        batch_size: Batch size for training
        time_steps: Number of time steps for temporal encoding
        num_workers: Number of worker processes for data loading
        
    Returns:
        train_loader, test_loader: PyTorch DataLoader objects
    """
    
    # Define transforms
    sensor_size = tonic.datasets.NMNIST.sensor_size
    frame_transform = transforms.Compose([
        transforms.Denoise(filter_time=10000),
        transforms.ToFrame(
            sensor_size=sensor_size,
            time_window=1000
        ),
    ])
    
    # Load datasets
    train_dataset = tonic.datasets.NMNIST(
        save_to='./data',
        transform=frame_transform,
        train=True
    )
    
    test_dataset = tonic.datasets.NMNIST(
        save_to='./data',
        transform=frame_transform,
        train=False
    )
    
    # Create data loaders
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
    
    print(f"✓ N-MNIST loaded: {len(train_dataset)} train, {len(test_dataset)} test")
    return train_loader, test_loader
