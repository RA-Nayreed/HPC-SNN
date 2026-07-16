"""Federated client optimizer construction."""

from __future__ import annotations

import torch


def learning_rate_for_round(federated: dict, round_number: int) -> float:
    """Resolve the configured learning rate at one communication round."""

    if not isinstance(round_number, int) or round_number <= 0:
        raise ValueError("round_number must be a positive integer")
    learning_rate = float(federated["learning_rate"])
    reduction_rounds = federated.get("learning_rate_reduction_rounds", [])
    factor = float(federated.get("learning_rate_reduction_factor", 1.0))
    reductions = sum(round_number > boundary for boundary in reduction_rounds)
    return learning_rate / (factor**reductions)


def make_federated_optimizer(parameters, federated: dict, round_number: int) -> torch.optim.Optimizer:
    """Construct a new local optimizer with the round-resolved learning rate."""

    learning_rate = learning_rate_for_round(federated, round_number)
    if federated["optimizer"] == "adam":
        return torch.optim.Adam(
            parameters,
            lr=learning_rate,
            weight_decay=float(federated["weight_decay"]),
        )
    if federated["optimizer"] == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=learning_rate,
            momentum=float(federated["momentum"]),
            weight_decay=float(federated["weight_decay"]),
        )
    raise ValueError(f"unsupported optimizer: {federated['optimizer']}")
