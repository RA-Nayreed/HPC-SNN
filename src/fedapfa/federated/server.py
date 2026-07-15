"""Global-model validation and update operations."""

from __future__ import annotations

import torch
from torch import nn

from .aggregation import state_l2_norm
from .client import evaluate_model, reset_snn_state
from .round_state import EvaluationResult


def validate_global_model(
    model: nn.Module,
    dataset,
    device: torch.device,
    batch_size: int,
    seed: int,
    workers: int,
    persistent_workers: bool,
) -> EvaluationResult:
    reset_snn_state(model)
    result = evaluate_model(model, dataset, device, batch_size, seed, workers, persistent_workers)
    reset_snn_state(model)
    return result


def global_model_norm(model: nn.Module) -> float:
    return state_l2_norm(model.state_dict())
