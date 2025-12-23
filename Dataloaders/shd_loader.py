"""
SHD (Spiking Heidelberg Digits) Dataset Loader
Spoken digits encoded through artificial cochlea.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import tonic
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class DenseSHDDataset(Dataset):
    """Pre-converted SHD dataset with dense spike tensors."""

    def __init__(self, data: List[Tuple[torch.Tensor, int]]):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        spikes, label = self.data[idx]
        return spikes, torch.tensor(label, dtype=torch.long)


def _events_to_dense(events) -> torch.Tensor:
    time_bins, channels = 100, 700
    dense = torch.zeros(time_bins, channels)

    if len(events) > 0:
        max_time = events["t"].max() if len(events) > 0 else 1
        time_indices = (events["t"] / max_time * (time_bins - 1)).astype(int)
        channel_indices = events["x"].astype(int)

        for t, c in zip(time_indices, channel_indices):
            if 0 <= t < time_bins and 0 <= c < channels:
                dense[t, c] = 1.0

    return dense.unsqueeze(1).unsqueeze(1)


def get_shd_loaders(batch_size: int = 32, num_workers: int = 2, save_to: str = "./data"):
    """
    Get SHD train and test data loaders.

    Returns:
        train_loader, test_loader: PyTorch DataLoader objects
    """
    train_dataset = tonic.datasets.SHD(save_to=save_to, train=True)
    test_dataset = tonic.datasets.SHD(save_to=save_to, train=False)

    train_data = []
    for events, label in tqdm(train_dataset, desc="SHD train -> dense"):
        train_data.append((_events_to_dense(events), label))

    test_data = []
    for events, label in tqdm(test_dataset, desc="SHD test -> dense"):
        test_data.append((_events_to_dense(events), label))

    train_loader = DataLoader(
        DenseSHDDataset(train_data),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    test_loader = DataLoader(
        DenseSHDDataset(test_data),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    print(f"✓ SHD loaded: {len(train_data)} train, {len(test_data)} test")
    return train_loader, test_loader
