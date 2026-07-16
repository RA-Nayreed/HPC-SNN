"""Isolated local federated training for one selected client."""

from __future__ import annotations

import copy
import math
import random
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from fedapfa.datasets.sequence_collation import EventBatch, collate_event_sequences
from fedapfa.metrics.classification import (
    confusion_matrix,
    macro_f1_from_confusion_matrix,
    per_class_accuracy,
)
from fedapfa.training.centralized import seed_worker
from fedapfa.training.optimization import learning_rate_for_round, make_federated_optimizer

from .aggregation import clone_state_dict, state_difference_l2_norm
from .randomness import global_rng_state, restore_global_rng_state
from .round_state import ClientResult, EvaluationResult


@dataclass(frozen=True)
class ModelBatch:
    inputs: torch.Tensor
    labels: torch.Tensor
    lengths: torch.Tensor | None
    valid_mask: torch.Tensor | None

    @property
    def rate_weight(self) -> int:
        return int(self.valid_mask.sum()) if self.valid_mask is not None else len(self.labels)


def synchronize_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def reset_snn_state(model: nn.Module) -> None:
    for module in model.modules():
        reset = getattr(module, "reset_state", None)
        if callable(reset):
            reset()


def _move(batch, device: torch.device) -> ModelBatch:
    non_blocking = device.type == "cuda"
    if isinstance(batch, EventBatch):
        moved = EventBatch(*(value.to(device, non_blocking=non_blocking) for value in batch))
        return ModelBatch(moved.inputs, moved.labels, moved.lengths, moved.valid_mask)
    inputs, labels = batch
    return ModelBatch(
        inputs.to(device, non_blocking=non_blocking),
        labels.to(device, non_blocking=non_blocking),
        None,
        None,
    )


def _batch_kind(dataset) -> str:
    declared = getattr(dataset, "fedapfa_batch_kind", None)
    if declared in {"event_sequence", "image"}:
        return declared
    sample = dataset[0]
    if not isinstance(sample, tuple) or len(sample) != 2 or not isinstance(sample[0], torch.Tensor):
        raise TypeError("federated datasets must return (tensor, label) samples")
    return "event_sequence" if sample[0].ndim == 2 else "image"


def _loader(dataset, batch_size: int, shuffle: bool, seed: int, workers: int, persistent_workers: bool):
    collate = collate_event_sequences if _batch_kind(dataset) == "event_sequence" else default_collate
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        persistent_workers=persistent_workers and workers > 0,
        pin_memory=False,
        collate_fn=collate,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(seed),
    )


def _generator(device: torch.device, seed: int) -> torch.Generator:
    target = device if device.type == "cuda" else torch.device("cpu")
    return torch.Generator(device=target).manual_seed(seed)


def _forward(model: nn.Module, batch: ModelBatch, generator: torch.Generator):
    if batch.lengths is None:
        return model(batch.inputs, generator=generator)
    return model(batch.inputs, batch.lengths)


def _finite_metrics(metrics: dict) -> None:
    values = []
    for value in metrics.values():
        if isinstance(value, dict):
            values.extend(item for item in value.values() if isinstance(item, (int, float)))
        elif isinstance(value, (int, float)):
            values.append(value)
    if any(not math.isfinite(float(value)) for value in values):
        raise FloatingPointError("training or evaluation produced NaN or infinity")


def evaluate_model(
    model: nn.Module,
    dataset,
    device: torch.device,
    batch_size: int,
    seed: int,
    workers: int = 0,
    persistent_workers: bool = False,
) -> EvaluationResult:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    loader = _loader(dataset, batch_size, False, seed, workers, persistent_workers)
    total = correct = batches = 0
    loss_sum = 0.0
    rate_sums: dict[str, float] = {}
    rate_weights: dict[str, int] = {}
    predictions: list[int] = []
    targets: list[int] = []
    classes: int | None = None
    poisson_generator = _generator(device, seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    reset_snn_state(model)
    with torch.no_grad():
        for batch in loader:
            reset_snn_state(model)
            moved = _move(batch, device)
            logits, rates = _forward(model, moved, poisson_generator)
            if logits.ndim != 2:
                raise ValueError("model logits must have shape [batch, classes]")
            classes = logits.shape[1] if classes is None else classes
            if logits.shape[1] != classes:
                raise ValueError("model class dimension changed during evaluation")
            loss = criterion(logits, moved.labels)
            predicted = logits.argmax(1)
            count = len(moved.labels)
            total += count
            correct += int((predicted == moved.labels).sum())
            loss_sum += float(loss) * count
            batches += 1
            predictions.extend(int(value) for value in predicted.detach().cpu())
            targets.extend(int(value) for value in moved.labels.detach().cpu())
            for name, value in rates.items():
                rate_sums[name] = rate_sums.get(name, 0.0) + float(value) * moved.rate_weight
                rate_weights[name] = rate_weights.get(name, 0) + moved.rate_weight
            reset_snn_state(model)
    if total == 0 or classes is None:
        raise RuntimeError("evaluation dataset produced no batches")
    matrix = confusion_matrix(predictions, targets, classes)
    result = EvaluationResult(
        loss=loss_sum / total,
        accuracy=correct / total,
        examples=total,
        batches=batches,
        spike_rates={name: value / rate_weights[name] for name, value in rate_sums.items()},
        confusion_matrix=matrix,
        per_class_accuracy=per_class_accuracy(matrix),
        macro_f1=macro_f1_from_confusion_matrix(matrix),
        peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
    )
    _finite_metrics(result.__dict__)
    return result


def train_client(
    server_model: nn.Module,
    dataset,
    client_id: str,
    round_number: int,
    config: dict,
    device: torch.device,
    training_seed: int,
    model_payload: int,
) -> ClientResult:
    """Train a detached model copy with a newly constructed local optimizer."""

    server_before = clone_state_dict(server_model.state_dict())
    local_model = copy.deepcopy(server_model).to(device)
    local_model.load_state_dict(server_before)
    federation = config["federated"]
    loader = _loader(
        dataset,
        federation["local_batch_size"],
        True,
        training_seed,
        federation["data_loader_workers"],
        federation["persistent_workers"],
    )
    optimizer = make_federated_optimizer(local_model.parameters(), federation, round_number)
    criterion = nn.CrossEntropyLoss()
    saved_rng = global_rng_state()
    random.seed(training_seed)
    np.random.seed(training_seed % (2**32))
    torch.manual_seed(training_seed)
    peak_memory_values: list[int] = []
    if device.type == "cuda":
        torch.cuda.manual_seed_all(training_seed)
        torch.cuda.reset_peak_memory_stats(device)
    extended = bool(federation.get("record_extended_diagnostics", False))
    first_loss = first_accuracy = None
    last_loss = last_accuracy = None
    batch_count = 0
    rate_sums: dict[str, float] = {}
    rate_weights: dict[str, int] = {}
    try:
        if extended:
            initial = evaluate_model(
                local_model,
                dataset,
                device,
                federation["local_batch_size"],
                training_seed + 1,
                0,
                False,
            )
            first_loss = initial.loss
            first_accuracy = initial.accuracy
            if initial.peak_cuda_memory_bytes is not None:
                peak_memory_values.append(initial.peak_cuda_memory_bytes)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        local_model.train()
        poisson_generator = _generator(device, training_seed)
        synchronize_cuda(device)
        started = time.monotonic()
        for _ in range(federation["local_epochs"]):
            for batch in loader:
                reset_snn_state(local_model)
                moved = _move(batch, device)
                optimizer.zero_grad(set_to_none=True)
                logits, rates = _forward(local_model, moved, poisson_generator)
                loss = criterion(logits, moved.labels)
                if not torch.isfinite(loss):
                    raise FloatingPointError("client loss contains NaN or infinity")
                loss.backward()
                for parameter in local_model.parameters():
                    if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                        raise FloatingPointError("client gradient contains NaN or infinity")
                gradient_clip = federation.get("gradient_clip")
                if gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(local_model.parameters(), gradient_clip)
                optimizer.step()
                if any(not torch.isfinite(parameter).all() for parameter in local_model.parameters()):
                    raise FloatingPointError("client model contains NaN or infinity")
                loss_value = float(loss.detach())
                accuracy_value = float((logits.argmax(1) == moved.labels).float().mean())
                if first_loss is None:
                    first_loss, first_accuracy = loss_value, accuracy_value
                last_loss, last_accuracy = loss_value, accuracy_value
                batch_count += 1
                for name, value in rates.items():
                    rate_sums[name] = rate_sums.get(name, 0.0) + float(value.detach()) * moved.rate_weight
                    rate_weights[name] = rate_weights.get(name, 0) + moved.rate_weight
                reset_snn_state(local_model)
        synchronize_cuda(device)
        elapsed = time.monotonic() - started
        if device.type == "cuda":
            peak_memory_values.append(torch.cuda.max_memory_allocated(device))
        if extended:
            final_evaluation = evaluate_model(
                local_model,
                dataset,
                device,
                federation["local_batch_size"],
                training_seed + 2,
                0,
                False,
            )
            last_loss = final_evaluation.loss
            last_accuracy = final_evaluation.accuracy
            if final_evaluation.peak_cuda_memory_bytes is not None:
                peak_memory_values.append(final_evaluation.peak_cuda_memory_bytes)
        local_state = clone_state_dict(local_model.state_dict())
        peak_memory = max(peak_memory_values) if device.type == "cuda" else None
    finally:
        restore_global_rng_state(saved_rng)
    if batch_count == 0 or first_loss is None or last_loss is None:
        raise RuntimeError(f"client {client_id} produced no training batches")
    if any(not torch.equal(server_before[name], server_model.state_dict()[name]) for name in server_before):
        raise RuntimeError("client training mutated the server model before aggregation")
    result = ClientResult(
        round_number=round_number,
        client_id=client_id,
        example_count=len(dataset),
        batch_count=batch_count,
        starting_training_loss=first_loss,
        starting_training_accuracy=float(first_accuracy),
        ending_training_loss=last_loss,
        ending_training_accuracy=float(last_accuracy),
        spike_rates={name: value / rate_weights[name] for name, value in rate_sums.items()},
        execution_time_seconds=elapsed,
        update_l2_norm=state_difference_l2_norm(local_state, server_before),
        peak_cuda_memory_bytes=peak_memory,
        logical_download_bytes=model_payload,
        logical_upload_bytes=model_payload,
        resolved_training_seed=training_seed,
        resolved_learning_rate=learning_rate_for_round(federation, round_number),
        state_dict=local_state,
    )
    _finite_metrics(result.record(1.0, 0.0))
    return result
