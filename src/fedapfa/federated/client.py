"""Isolated local LIF training for one selected client."""

from __future__ import annotations

import copy
import math
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from fedapfa.datasets.sequence_collation import EventBatch, collate_event_sequences
from fedapfa.training.centralized import seed_worker

from .aggregation import clone_state_dict, state_difference_l2_norm
from .randomness import global_rng_state, restore_global_rng_state
from .round_state import ClientResult, EvaluationResult


def synchronize_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def reset_snn_state(model: nn.Module) -> None:
    for module in model.modules():
        reset = getattr(module, "reset_state", None)
        if callable(reset):
            reset()


def _move(batch: EventBatch, device: torch.device) -> EventBatch:
    non_blocking = device.type == "cuda"
    return EventBatch(*(value.to(device, non_blocking=non_blocking) for value in batch))


def _loader(dataset, batch_size: int, shuffle: bool, seed: int, workers: int, persistent_workers: bool):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        persistent_workers=persistent_workers and workers > 0,
        pin_memory=False,
        collate_fn=collate_event_sequences,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(seed),
    )


def _finite_metrics(metrics: dict) -> None:
    values = []
    for value in metrics.values():
        if isinstance(value, dict):
            values.extend(value.values())
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
    reset_snn_state(model)
    with torch.no_grad():
        for batch in loader:
            reset_snn_state(model)
            moved = _move(batch, device)
            logits, rates = model(moved.inputs, moved.lengths)
            loss = criterion(logits, moved.labels)
            count = len(moved.labels)
            total += count
            correct += int((logits.argmax(1) == moved.labels).sum())
            loss_sum += float(loss) * count
            batches += 1
            valid_steps = int(moved.valid_mask.sum())
            for name, value in rates.items():
                rate_sums[name] = rate_sums.get(name, 0.0) + float(value) * valid_steps
                rate_weights[name] = rate_weights.get(name, 0) + valid_steps
            reset_snn_state(model)
    if total == 0:
        raise RuntimeError("evaluation dataset produced no batches")
    result = EvaluationResult(
        loss=loss_sum / total,
        accuracy=correct / total,
        examples=total,
        batches=batches,
        spike_rates={name: value / rate_weights[name] for name, value in rate_sums.items()},
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
    """Train a detached model copy with a newly constructed Adam optimizer."""

    server_before = clone_state_dict(server_model.state_dict())
    local_model = copy.deepcopy(server_model).to(device)
    local_model.load_state_dict(server_before)
    local_model.train()
    federation = config["federated"]
    loader = _loader(
        dataset,
        federation["local_batch_size"],
        True,
        training_seed,
        federation["data_loader_workers"],
        federation["persistent_workers"],
    )
    optimizer = torch.optim.Adam(
        local_model.parameters(),
        lr=federation["learning_rate"],
        weight_decay=federation["weight_decay"],
    )
    criterion = nn.CrossEntropyLoss()
    saved_rng = global_rng_state()
    random.seed(training_seed)
    np.random.seed(training_seed % (2**32))
    torch.manual_seed(training_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(training_seed)
        torch.cuda.reset_peak_memory_stats(device)
    first_loss = first_accuracy = None
    last_loss = last_accuracy = None
    batch_count = 0
    rate_sums: dict[str, float] = {}
    rate_weights: dict[str, int] = {}
    synchronize_cuda(device)
    started = time.monotonic()
    try:
        for _ in range(federation["local_epochs"]):
            for batch in loader:
                reset_snn_state(local_model)
                moved = _move(batch, device)
                optimizer.zero_grad(set_to_none=True)
                logits, rates = local_model(moved.inputs, moved.lengths)
                loss = criterion(logits, moved.labels)
                if not torch.isfinite(loss):
                    raise FloatingPointError("client loss contains NaN or infinity")
                loss.backward()
                for parameter in local_model.parameters():
                    if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                        raise FloatingPointError("client gradient contains NaN or infinity")
                torch.nn.utils.clip_grad_norm_(local_model.parameters(), federation["gradient_clip"])
                optimizer.step()
                if any(not torch.isfinite(parameter).all() for parameter in local_model.parameters()):
                    raise FloatingPointError("client model contains NaN or infinity")
                loss_value = float(loss.detach())
                accuracy_value = float((logits.argmax(1) == moved.labels).float().mean())
                if first_loss is None:
                    first_loss, first_accuracy = loss_value, accuracy_value
                last_loss, last_accuracy = loss_value, accuracy_value
                batch_count += 1
                valid_steps = int(moved.valid_mask.sum())
                for name, value in rates.items():
                    rate_sums[name] = rate_sums.get(name, 0.0) + float(value.detach()) * valid_steps
                    rate_weights[name] = rate_weights.get(name, 0) + valid_steps
                reset_snn_state(local_model)
        synchronize_cuda(device)
        elapsed = time.monotonic() - started
        local_state = clone_state_dict(local_model.state_dict())
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
        peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        logical_download_bytes=model_payload,
        logical_upload_bytes=model_payload,
        resolved_training_seed=training_seed,
        state_dict=local_state,
    )
    _finite_metrics(result.record(1.0))
    return result
