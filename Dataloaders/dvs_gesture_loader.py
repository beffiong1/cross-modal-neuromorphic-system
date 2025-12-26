"""
DVS-Gesture Dataset Loader
Event-based recordings of hand/arm gestures (DVS128 Gesture).
"""

from __future__ import annotations

from typing import Callable, Tuple

import torch
import tonic
import tonic.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F


def _get_dvs_gesture_dataset_class() -> Callable:
    """
    Resolve the DVS-Gesture dataset class across tonic versions.
    """
    if hasattr(tonic.datasets, "DVS128Gesture"):
        return tonic.datasets.DVS128Gesture
    if hasattr(tonic.datasets, "DVSGesture"):
        return tonic.datasets.DVSGesture
    raise AttributeError("tonic.datasets missing DVS-Gesture dataset class")


def get_dvs_gesture_loaders(
    batch_size: int = 16,
    time_steps: int = 25,
    num_workers: int = 2,
    save_to: str = "./data",
    time_window: int = 1000,
    target_spatial_size: Tuple[int, int] = (128, 128),
) -> Tuple[DataLoader, DataLoader]:
    """
    Get DVS-Gesture train and test data loaders.

    Args:
        batch_size: Batch size for training
        time_steps: Number of time steps for temporal encoding
        num_workers: Number of worker processes for data loading
        save_to: Directory to download/store the dataset
        time_window: Window size (in microseconds) for event framing
        target_spatial_size: Optional spatial resize to match model input

    Returns:
        train_loader, test_loader: PyTorch DataLoader objects
    """

    dataset_cls = _get_dvs_gesture_dataset_class()
    sensor_size = dataset_cls.sensor_size

    frame_transform = transforms.Compose(
        [
            transforms.Denoise(filter_time=10000),
            transforms.ToFrame(sensor_size=sensor_size, time_window=time_window),
        ]
    )

    train_dataset = dataset_cls(save_to=save_to, transform=frame_transform, train=True)
    test_dataset = dataset_cls(save_to=save_to, transform=frame_transform, train=False)

    def collate_fn(batch):
        frames, labels = zip(*batch)
        batch_size_local = len(frames)
        first_tensor = torch.as_tensor(frames[0], dtype=torch.float32)
        if first_tensor.dim() == 4 and first_tensor.shape[1] not in (1, 2):
            if first_tensor.shape[-1] in (1, 2):
                first_tensor = first_tensor.permute(0, 3, 1, 2)
        if target_spatial_size is not None:
            first_tensor = F.interpolate(
                first_tensor, size=target_spatial_size, mode="nearest"
            )

        channels = first_tensor.shape[1]
        height = first_tensor.shape[2]
        width = first_tensor.shape[3]

        padded = torch.zeros(
            batch_size_local, time_steps, channels, height, width, dtype=torch.float32
        )

        length = min(first_tensor.shape[0], time_steps)
        padded[0, :length] = first_tensor[:length]

        for idx, frame in enumerate(frames[1:], start=1):
            frame_tensor = torch.as_tensor(frame, dtype=torch.float32)
            if frame_tensor.dim() == 4 and frame_tensor.shape[1] not in (1, 2):
                if frame_tensor.shape[-1] in (1, 2):
                    frame_tensor = frame_tensor.permute(0, 3, 1, 2)
            if target_spatial_size is not None:
                frame_tensor = F.interpolate(
                    frame_tensor,
                    size=target_spatial_size,
                    mode="nearest",
                )
            length = min(frame_tensor.shape[0], time_steps)
            padded[idx, :length] = frame_tensor[:length]
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        return padded, labels_tensor

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        collate_fn=collate_fn,
    )

    print(
        f"✓ DVS-Gesture loaded: {len(train_dataset)} train, {len(test_dataset)} test"
    )
    return train_loader, test_loader
