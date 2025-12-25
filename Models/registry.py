"""
Model registry for consistent variant selection.
"""

from __future__ import annotations

from typing import Callable, Dict, Type

import torch.nn as nn

from .model_1_baseline import Model_1_Baseline
from .model_2_scl import Model_2_SCL
from .model_3_hopfield import Model_3_Hopfield
from .model_4_hgrn import Model_4_HGRN
from .model_5_hybrid import Model_5_Hybrid


MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {
    "baseline": Model_1_Baseline,
    "scl": Model_2_SCL,
    "hopfield": Model_3_Hopfield,
    "hgrn": Model_4_HGRN,
    "hybrid": Model_5_Hybrid,
}


def build_model(
    variant: str,
    input_type: str,
    input_channels: int = 2,
    input_size: int = 700,
    spatial_size: tuple[int, int] = (34, 34),
    num_classes: int = 10,
) -> nn.Module:
    if variant not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model variant: {variant}")
    model_cls = MODEL_REGISTRY[variant]
    return model_cls(
        input_type=input_type,
        input_channels=input_channels,
        input_size=input_size,
        spatial_size=spatial_size,
        num_classes=num_classes,
    )
