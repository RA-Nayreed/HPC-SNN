"""Typed records exchanged by federated client and server components."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class AggregationInput:
    client_id: str
    example_count: int
    state_dict: dict[str, torch.Tensor]


@dataclass(frozen=True)
class ClientResult:
    round_number: int
    client_id: str
    example_count: int
    batch_count: int
    starting_training_loss: float
    starting_training_accuracy: float
    ending_training_loss: float
    ending_training_accuracy: float
    spike_rates: dict[str, float]
    execution_time_seconds: float
    update_l2_norm: float
    peak_cuda_memory_bytes: int | None
    logical_download_bytes: int
    logical_upload_bytes: int
    resolved_training_seed: int
    state_dict: dict[str, torch.Tensor]

    def record(self, aggregation_weight: float) -> dict:
        value = asdict(self)
        value.pop("state_dict")
        value["aggregation_weight"] = aggregation_weight
        value["logical_total_bytes"] = self.logical_download_bytes + self.logical_upload_bytes
        return value


@dataclass(frozen=True)
class EvaluationResult:
    loss: float
    accuracy: float
    examples: int
    batches: int
    spike_rates: dict[str, float]


@dataclass(frozen=True)
class RoundResult:
    round_number: int
    selected_client_ids: list[str]
    client_example_counts: list[int]
    aggregation_weights: list[float]
    total_selected_examples: int
    validation_loss: float
    validation_accuracy: float
    validation_spike_rates: dict[str, float]
    global_model_l2_norm: float
    aggregated_update_l2_norm: float
    client_training_time_seconds: float
    aggregation_time_seconds: float
    validation_time_seconds: float
    total_round_time_seconds: float
    logical_download_bytes: int
    logical_upload_bytes: int
    logical_communication_bytes: int
    cumulative_logical_download_bytes: int
    cumulative_logical_upload_bytes: int
    cumulative_logical_communication_bytes: int
    current_best_validation_round: int
    selected_checkpoint: bool

    def record(self) -> dict:
        return asdict(self)
