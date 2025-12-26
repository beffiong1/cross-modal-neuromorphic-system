"""
Dataset loader registry with tonic integration.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

import tonic
from torch.utils.data import DataLoader

from Dataloaders.nmnist_loader import get_nmnist_loaders
from Dataloaders.shd_loader import get_shd_loaders


def get_tonic_loaders(
    dataset_name: str,
    batch_size: int = 32,
    num_workers: int = 2,
    save_to: str = "./data",
    transform: Optional[Callable] = None,
    **dataset_kwargs: Any,
) -> Tuple[DataLoader, DataLoader]:
    dataset_cls = getattr(tonic.datasets, dataset_name, None)
    if dataset_cls is None:
        raise ValueError(f"Unknown tonic dataset: {dataset_name}")

    train_dataset = dataset_cls(save_to=save_to, train=True, transform=transform, **dataset_kwargs)
    test_dataset = dataset_cls(save_to=save_to, train=False, transform=transform, **dataset_kwargs)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False
    )
    return train_loader, test_loader


def get_loaders(
    name: str,
    batch_size: int = 32,
    num_workers: int = 2,
    **kwargs: Any,
) -> Tuple[DataLoader, DataLoader]:
    key = name.lower()
    if key == "nmnist":
        return get_nmnist_loaders(batch_size=batch_size, num_workers=num_workers)
    if key == "shd":
        save_to = kwargs.get("save_to", "./data")
        return get_shd_loaders(batch_size=batch_size, num_workers=num_workers, save_to=save_to)
    if key.startswith("tonic:"):
        dataset_name = key.split("tonic:", 1)[1]
        return get_tonic_loaders(
            dataset_name=dataset_name,
            batch_size=batch_size,
            num_workers=num_workers,
            save_to=kwargs.get("save_to", "./data"),
            transform=kwargs.get("transform"),
            **kwargs.get("dataset_kwargs", {}),
        )
    raise ValueError(f"Unknown dataset name: {name}")
