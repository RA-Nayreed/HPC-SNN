"""Rank-local client training with ordered CPU result envelopes."""

from __future__ import annotations

import gc
import math
import time
from dataclasses import dataclass, replace

import torch
from torch import nn

from fedapfa.federated.checkpointing import state_identity
from fedapfa.federated.client import train_client
from fedapfa.federated.randomness import derive_seed
from fedapfa.federated.round_state import ClientResult

from .assignment_broadcast import ClientAssignment, assignments_for_rank
from .process_context import ProcessContext


@dataclass(frozen=True)
class ClientResultEnvelope:
    selected_position: int
    process_rank: int
    incoming_global_model_id: str
    update_identity: str
    result: ClientResult
    completed_at_unix_nanoseconds: int


@dataclass(frozen=True)
class RankClientPayload:
    process_rank: int
    device_index: int
    device_slot: int
    node_rank: int
    local_rank: int
    assigned_client_ids: list[str]
    assigned_example_count: int
    process_busy_time_seconds: float
    peak_cuda_memory_bytes: int | None
    peak_cuda_reserved_bytes: int | None
    results: list[ClientResultEnvelope]


def _finite(value) -> bool:
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def _cpu_result(result: ClientResult) -> ClientResult:
    state = {name: value.detach().cpu().clone() for name, value in result.state_dict.items()}
    return replace(result, state_dict=state)


def _validate_result(
    envelope: ClientResultEnvelope,
    assignment: ClientAssignment,
    round_number: int,
    expected_seed: int,
    incoming_state: dict[str, torch.Tensor],
) -> None:
    result = envelope.result
    if (
        envelope.selected_position != assignment.selected_position
        or envelope.process_rank != assignment.process_rank
        or result.client_id != assignment.client_id
        or result.round_number != round_number
    ):
        raise RuntimeError("distributed client result identity is incompatible with its assignment")
    if result.resolved_training_seed != expected_seed:
        raise RuntimeError("distributed client result has an incompatible training seed")
    if result.example_count <= 0 or result.local_training_examples_presented <= 0:
        raise RuntimeError("distributed client result has invalid example counts")
    if set(result.state_dict) != set(incoming_state):
        raise RuntimeError("distributed client result has incompatible model-state keys")
    for name, value in result.state_dict.items():
        expected = incoming_state[name]
        if value.device.type != "cpu" or value.requires_grad:
            raise RuntimeError("collected client state tensors must be detached on CPU")
        if value.shape != expected.shape or value.dtype != expected.dtype:
            raise RuntimeError(f"distributed client tensor {name} has incompatible shape or dtype")
        if (value.is_floating_point() or value.is_complex()) and not bool(torch.isfinite(value).all()):
            raise RuntimeError(f"distributed client tensor {name} contains NaN or infinity")
    if not _finite(result.record(1.0, 0.0)):
        raise RuntimeError("distributed client result contains non-finite metrics")
    if envelope.update_identity != state_identity(result.state_dict):
        raise RuntimeError("distributed client update identity does not match its state")


def train_rank_clients(
    model: nn.Module,
    bundle,
    config: dict,
    context: ProcessContext,
    assignments: list[ClientAssignment],
    round_number: int,
    incoming_global_model_id: str,
    model_payload_bytes: int,
    client_training=None,
) -> RankClientPayload:
    """Train the rank's clients sequentially without rank-derived randomness."""

    local_assignments = assignments_for_rank(assignments, context.rank)
    incoming_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    if context.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(context.device)
    started = time.monotonic()
    envelopes: list[ClientResultEnvelope] = []
    training_function = train_client if client_training is None else client_training
    for assignment in local_assignments:
        training_seed = derive_seed(
            config["seed"],
            config["seed_streams"]["client_training"],
            round_number,
            assignment.client_id,
        )
        with torch.profiler.record_function("client_model_construction_and_training"):
            client_dataset = bundle.client_dataset(assignment.client_id)
            if hasattr(training_function, "train_assigned_client"):
                result = training_function.train_assigned_client(
                    model,
                    client_dataset,
                    assignment.client_id,
                    round_number,
                    config,
                    context.device,
                    training_seed,
                    model_payload_bytes,
                    selected_position=assignment.selected_position,
                    process_rank=context.rank,
                )
            else:
                result = training_function(
                    model,
                    client_dataset,
                    assignment.client_id,
                    round_number,
                    config,
                    context.device,
                    training_seed,
                    model_payload_bytes,
                )
        cpu_result = _cpu_result(result)
        envelope = ClientResultEnvelope(
            selected_position=assignment.selected_position,
            process_rank=context.rank,
            incoming_global_model_id=incoming_global_model_id,
            update_identity=state_identity(cpu_result.state_dict),
            result=cpu_result,
            completed_at_unix_nanoseconds=time.time_ns(),
        )
        _validate_result(envelope, assignment, round_number, training_seed, incoming_state)
        envelopes.append(envelope)
        del result
        gc.collect()
        if context.device.type == "cuda":
            torch.cuda.empty_cache()
    busy_time = time.monotonic() - started
    peak_values = [
        value.result.peak_cuda_memory_bytes for value in envelopes if value.result.peak_cuda_memory_bytes is not None
    ]
    peak = max(peak_values) if peak_values else None
    reserved_values = [
        value.result.peak_cuda_reserved_bytes
        for value in envelopes
        if value.result.peak_cuda_reserved_bytes is not None
    ]
    return RankClientPayload(
        process_rank=context.rank,
        device_index=context.device_index,
        device_slot=context.device_slot,
        node_rank=context.node_rank,
        local_rank=context.local_rank,
        assigned_client_ids=[value.client_id for value in local_assignments],
        assigned_example_count=sum(value.result.example_count for value in envelopes),
        process_busy_time_seconds=busy_time,
        peak_cuda_memory_bytes=peak,
        peak_cuda_reserved_bytes=max(reserved_values) if reserved_values else None,
        results=envelopes,
    )


def order_client_results(
    payloads: list[RankClientPayload],
    assignments: list[ClientAssignment],
    round_number: int,
    config: dict,
    incoming_state: dict[str, torch.Tensor],
    incoming_global_model_id: str,
) -> tuple[list[ClientResult], list[ClientResultEnvelope]]:
    """Validate and restore results to the original selected-client order."""

    payload_ranks = [payload.process_rank for payload in payloads]
    if len(payload_ranks) != len(set(payload_ranks)):
        raise RuntimeError("distributed result payloads contain a duplicate process rank")
    expected_ranks = sorted({value.process_rank for value in assignments})
    if not set(expected_ranks).issubset(payload_ranks):
        raise RuntimeError("distributed result payloads are missing an assigned process rank")
    for payload in payloads:
        expected_clients = [value.client_id for value in assignments if value.process_rank == payload.process_rank]
        if payload.assigned_client_ids != expected_clients:
            raise RuntimeError("distributed result payload has an incompatible assigned-client list")
    envelopes = [envelope for payload in payloads for envelope in payload.results]
    if len(envelopes) != len(assignments):
        raise RuntimeError("distributed result count does not match selected clients")
    by_position = {value.selected_position: value for value in envelopes}
    expected_positions = {value.selected_position for value in assignments}
    if len(by_position) != len(envelopes) or set(by_position) != expected_positions:
        raise RuntimeError("distributed results contain missing or duplicate selected positions")
    ordered = [by_position[value.selected_position] for value in assignments]
    for assignment, envelope in zip(assignments, ordered, strict=True):
        expected_seed = derive_seed(
            config["seed"],
            config["seed_streams"]["client_training"],
            round_number,
            assignment.client_id,
        )
        _validate_result(envelope, assignment, round_number, expected_seed, incoming_state)
        if envelope.incoming_global_model_id != incoming_global_model_id:
            raise RuntimeError("a client received an incompatible incoming global model")
    return [value.result for value in ordered], ordered
