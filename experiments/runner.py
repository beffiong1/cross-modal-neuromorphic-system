"""
Experiment runner for dataset/model sweeps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

from Models.registry import build_model
from experiments.datasets import get_loaders
from training.pipeline import train_model
from training.settings import TrainingConfig


@dataclass(frozen=True)
class ExperimentSpec:
    dataset: str
    model_variant: str
    input_type: str
    num_classes: int = 10
    batch_size: int = 32
    num_workers: int = 2


def uses_contrastive(variant: str) -> bool:
    return variant in {"scl", "hybrid"}


def run_experiment(
    spec: ExperimentSpec,
    device: torch.device,
    config: TrainingConfig,
    dataset_kwargs: Optional[Dict] = None,
) -> Tuple[torch.nn.Module, Dict, float]:
    dataset_kwargs = dataset_kwargs or {}
    train_loader, test_loader = get_loaders(
        name=spec.dataset,
        batch_size=spec.batch_size,
        num_workers=spec.num_workers,
        **dataset_kwargs,
    )
    model = build_model(
        variant=spec.model_variant,
        input_type=spec.input_type,
        num_classes=spec.num_classes,
    ).to(device)
    model_name = f"{spec.model_variant}_{spec.input_type}_{spec.dataset}"
    return train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        model_name=model_name,
        dataset_name=spec.dataset,
        use_contrastive=uses_contrastive(spec.model_variant),
        device=device,
        config=config,
    )
