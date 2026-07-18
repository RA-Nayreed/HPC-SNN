"""Common data, model, and client-training boundary for federated execution."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from torch import nn

from fedapfa.federated.client import train_client
from fedapfa.federated.data_protocol import FederatedWorkload, prepare_federated_workload
from fedapfa.federated.round_state import ClientResult
from fedapfa.training.federated import make_initialized_federated_model


@dataclass(frozen=True)
class FederatedExecutionWorkload:
    """Resolved workload services consumed by process coordination."""

    data: FederatedWorkload
    model_factory: Callable[[dict], nn.Module]
    client_training: Callable[..., ClientResult]


def prepare_federated_execution_workload(
    config: dict, *, coordinator: bool
) -> FederatedExecutionWorkload:
    return FederatedExecutionWorkload(
        data=prepare_federated_workload(config, coordinator=coordinator),
        model_factory=make_initialized_federated_model,
        client_training=train_client,
    )
