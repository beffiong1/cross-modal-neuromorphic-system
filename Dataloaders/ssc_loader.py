"""
SSC (Spiking Speech Commands) Dataset Loader.
Spoken commands encoded through artificial cochlea.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import tonic
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class DenseSSCDataset(Dataset):
    """Pre-converted SSC dataset with dense spike tensors."""

    def __init__(self, data: List[Tuple[torch.Tensor, int]], sensor_size, num_classes: int):
        self.data = data
        self.sensor_size = sensor_size
        self.num_classes = num_classes

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        spikes, label = self.data[idx]
        return spikes, torch.tensor(label, dtype=torch.long)


def _events_to_dense(events, channels: int, time_bins: int) -> torch.Tensor:
    dense = torch.zeros(time_bins, channels)

    if len(events) > 0:
        max_time = events["t"].max() if len(events) > 0 else 1
        time_indices = (events["t"] / max_time * (time_bins - 1)).astype(int)
        channel_indices = events["x"].astype(int)

        for t, c in zip(time_indices, channel_indices):
            if 0 <= t < time_bins and 0 <= c < channels:
                dense[t, c] = 1.0

    return dense.unsqueeze(1).unsqueeze(1)


def get_ssc_loaders(
    batch_size: int = 32,
    num_workers: int = 2,
    save_to: str = "./data",
    time_bins: int = 100,
):
    """
    Get SSC train and test data loaders.

    Returns:
        train_loader, test_loader: PyTorch DataLoader objects
    """
    if not hasattr(tonic.datasets, "SSC"):
        raise AttributeError("tonic.datasets missing SSC dataset class")

    def _build_dataset(split: str):
        try:
            return tonic.datasets.SSC(save_to=save_to, train=(split == "train"))
        except TypeError:
            return tonic.datasets.SSC(save_to=save_to, split=split)

    train_dataset = _build_dataset("train")
    test_dataset = _build_dataset("test")

    sensor_size = train_dataset.sensor_size
    channels = sensor_size[0] if isinstance(sensor_size, tuple) else sensor_size

    train_data = []
    for events, label in tqdm(train_dataset, desc="SSC train -> dense"):
        train_data.append((_events_to_dense(events, channels, time_bins), label))

    test_data = []
    for events, label in tqdm(test_dataset, desc="SSC test -> dense"):
        test_data.append((_events_to_dense(events, channels, time_bins), label))

    if hasattr(train_dataset, "classes"):
        num_classes = len(train_dataset.classes)
    else:
        num_classes = len({label for _, label in train_data})

    train_loader = DataLoader(
        DenseSSCDataset(train_data, sensor_size, num_classes),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    test_loader = DataLoader(
        DenseSSCDataset(test_data, sensor_size, num_classes),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    print(f"✓ SSC loaded: {len(train_data)} train, {len(test_data)} test")
    return train_loader, test_loader
