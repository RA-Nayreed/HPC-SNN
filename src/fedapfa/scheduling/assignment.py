"""Deterministic round-robin and longest-processing-time assignment."""

from __future__ import annotations

import math

from .base import SCHEDULING_STRATEGIES, ScheduledClient


def _validate_selected(selected_client_ids: list[str], process_count: int) -> None:
    if not isinstance(process_count, int) or isinstance(process_count, bool) or process_count <= 0:
        raise ValueError("process count must be a positive integer")
    if not selected_client_ids:
        raise ValueError("selected-client order cannot be empty")
    if any(not isinstance(value, str) or not value for value in selected_client_ids):
        raise ValueError("selected client IDs must be nonempty strings")
    if len(selected_client_ids) != len(set(selected_client_ids)):
        raise ValueError("every selected client must appear exactly once")


def assign_selected_clients(
    selected_client_ids: list[str],
    process_count: int,
    strategy: str,
    costs: dict[str, float] | None = None,
    *,
    cost_source: str,
    features: dict[str, dict[str, float]] | None = None,
) -> tuple[list[ScheduledClient], dict[str, float]]:
    """Assign selected clients without altering their scientific order."""

    _validate_selected(selected_client_ids, process_count)
    if strategy not in SCHEDULING_STRATEGIES:
        raise ValueError(f"unsupported scheduling strategy: {strategy}")
    if not isinstance(cost_source, str) or not cost_source:
        raise ValueError("cost source must be a nonempty string")
    if costs is None:
        costs = {client_id: 1.0 for client_id in selected_client_ids}
    if set(costs) != set(selected_client_ids):
        raise ValueError("scheduling costs must cover exactly the selected clients")
    numeric_costs: dict[str, float] = {}
    for client_id, value in costs.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"client {client_id} cost must be numeric")
        cost = float(value)
        if not math.isfinite(cost) or cost < 0:
            raise ValueError(f"client {client_id} cost must be finite and nonnegative")
        numeric_costs[client_id] = cost
    if features is not None and set(features) != set(selected_client_ids):
        raise ValueError("feature records must cover exactly the selected clients")

    loads = [0.0 for _ in range(process_count)]
    counts = [0 for _ in range(process_count)]
    rank_by_position: dict[int, int] = {}
    if strategy == "round_robin":
        for position, client_id in enumerate(selected_client_ids):
            rank = position % process_count
            rank_by_position[position] = rank
            loads[rank] += numeric_costs[client_id]
            counts[rank] += 1
    else:
        ordered = sorted(
            enumerate(selected_client_ids),
            key=lambda item: (-numeric_costs[item[1]], item[0], item[1]),
        )
        for position, client_id in ordered:
            rank = min(range(process_count), key=lambda value: (loads[value], counts[value], value))
            rank_by_position[position] = rank
            loads[rank] += numeric_costs[client_id]
            counts[rank] += 1

    assignments = [
        ScheduledClient(
            selected_position=position,
            client_id=client_id,
            process_rank=rank_by_position[position],
            cost_source=cost_source,
            cost=numeric_costs[client_id],
            features=None if features is None else dict(features[client_id]),
        )
        for position, client_id in enumerate(selected_client_ids)
    ]
    validate_assignments(assignments, selected_client_ids, process_count)
    return assignments, {str(rank): loads[rank] for rank in range(process_count)}


def validate_assignments(
    assignments: list[ScheduledClient], selected_client_ids: list[str], process_count: int
) -> None:
    """Reject missing, duplicate, unexpected, or out-of-order assignments."""

    _validate_selected(selected_client_ids, process_count)
    if len(assignments) != len(selected_client_ids):
        raise ValueError("assignment count differs from selected-client count")
    positions = [value.selected_position for value in assignments]
    clients = [value.client_id for value in assignments]
    if positions != list(range(len(selected_client_ids))):
        raise ValueError("assignments are not restored to original selected-client order")
    if clients != selected_client_ids:
        raise ValueError("assignments contain missing, duplicate, or unexpected clients")
    if any(not 0 <= value.process_rank < process_count for value in assignments):
        raise ValueError("assignment contains an out-of-range process rank")


def assignments_for_rank(assignments: list[ScheduledClient], process_rank: int) -> list[ScheduledClient]:
    return [value for value in assignments if value.process_rank == process_rank]
