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
    client_population_examples: int
    presented_examples_per_local_epoch: int
    local_training_examples_presented: int
    batch_count: int
    starting_training_loss: float
    starting_training_accuracy: float
    ending_training_loss: float
    ending_training_accuracy: float
    spike_rates: dict[str, float]
    execution_time_seconds: float
    data_wait_time_seconds: float
    update_l2_norm: float
    peak_cuda_memory_bytes: int | None
    peak_cuda_reserved_bytes: int | None
    logical_download_bytes: int
    logical_upload_bytes: int
    resolved_training_seed: int
    resolved_learning_rate: float
    state_dict: dict[str, torch.Tensor]

    def record(self, aggregation_weight: float, update_cosine_similarity: float) -> dict:
        value = asdict(self)
        value.pop("state_dict")
        value["aggregation_weight"] = aggregation_weight
        value["logical_total_bytes"] = self.logical_download_bytes + self.logical_upload_bytes
        value["initial_local_loss"] = self.starting_training_loss
        value["initial_local_accuracy"] = self.starting_training_accuracy
        value["final_local_loss"] = self.ending_training_loss
        value["final_local_accuracy"] = self.ending_training_accuracy
        value["client_spike_rates"] = dict(self.spike_rates)
        value["training_duration_seconds"] = self.execution_time_seconds
        value["update_cosine_similarity"] = update_cosine_similarity
        return value


@dataclass(frozen=True)
class EvaluationResult:
    loss: float
    accuracy: float
    examples: int
    batches: int
    spike_rates: dict[str, float]
    confusion_matrix: list[list[int]]
    per_class_accuracy: list[float]
    macro_f1: float
    peak_cuda_memory_bytes: int | None


@dataclass(frozen=True)
class RoundResult:
    round_number: int
    selected_client_ids: list[str]
    aggregation_weighting: str
    client_example_counts: list[int]
    client_training_examples_presented: list[int]
    aggregation_weights: list[float]
    total_selected_examples: int
    total_training_examples_presented: int
    validation_loss: float | None
    validation_accuracy: float | None
    validation_macro_f1: float | None
    validation_per_class_accuracy: list[float] | None
    validation_confusion_matrix: list[list[int]] | None
    validation_spike_rates: dict[str, float] | None
    global_model_l2_norm: float
    aggregated_update_l2_norm: float
    mean_client_update_l2_norm: float
    standard_deviation_client_update_l2_norm: float
    mean_client_to_aggregate_cosine_similarity: float
    minimum_client_to_aggregate_cosine_similarity: float
    maximum_client_to_aggregate_cosine_similarity: float
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
    peak_cuda_memory_bytes: int | None
    current_best_validation_round: int | None
    selected_checkpoint: bool

    def record(self) -> dict:
        return asdict(self)
