"""Selected-order client assignment and coordinator-owned broadcast."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch.distributed as dist

from fedapfa.scheduling.assignment import validate_assignments
from fedapfa.scheduling.base import ScheduledClient

from .process_context import ProcessContext


@dataclass(frozen=True)
class ClientAssignment:
    selected_position: int
    client_id: str
    process_rank: int

    def record(self) -> dict:
        return asdict(self)


def assign_clients(selected_client_ids: list[str], world_size: int) -> list[ClientAssignment]:
    if not isinstance(world_size, int) or isinstance(world_size, bool) or world_size <= 0:
        raise ValueError("client assignment process count must be positive")
    if not selected_client_ids or len(selected_client_ids) != len(set(selected_client_ids)):
        raise ValueError("selected clients must be a nonempty unique ordered collection")
    return [
        ClientAssignment(position, client_id, position % world_size)
        for position, client_id in enumerate(selected_client_ids)
    ]


def assignments_for_rank(assignments: list[ClientAssignment], process_rank: int) -> list[ClientAssignment]:
    return [value for value in assignments if value.process_rank == process_rank]


def broadcast_selected_clients(
    context: ProcessContext,
    selected_client_ids: list[str] | None,
) -> list[str]:
    if context.is_coordinator:
        if selected_client_ids is None:
            raise ValueError("rank 0 must provide the selected-client order")
        payload = [list(selected_client_ids)]
    else:
        if selected_client_ids is not None:
            raise ValueError("nonzero ranks cannot provide a selected-client order")
        payload = [None]
    dist.broadcast_object_list(payload, src=0, device=context.control_device)
    selected = payload[0]
    if not isinstance(selected, list) or any(not isinstance(value, str) or not value for value in selected):
        raise RuntimeError("selected-client broadcast is incompatible")
    if len(selected) != len(set(selected)):
        raise RuntimeError("selected-client broadcast contains duplicates")
    return selected


def broadcast_assignments(
    context: ProcessContext,
    assignments: list[ScheduledClient] | None,
    selected_client_ids: list[str],
) -> list[ScheduledClient]:
    """Broadcast coordinator-created assignments before any client starts training."""

    if context.is_coordinator:
        if assignments is None:
            raise ValueError("rank 0 must provide client assignments")
        validate_assignments(assignments, selected_client_ids, context.world_size)
        payload = [[assignment.record() for assignment in assignments]]
    else:
        if assignments is not None:
            raise ValueError("nonzero ranks cannot provide client assignments")
        payload = [None]
    dist.broadcast_object_list(payload, src=0, device=context.control_device)
    records = payload[0]
    if not isinstance(records, list):
        raise RuntimeError("assignment broadcast is incompatible")
    try:
        received = [ScheduledClient(**record) for record in records]
        validate_assignments(received, selected_client_ids, context.world_size)
    except (TypeError, ValueError) as error:
        raise RuntimeError("assignment broadcast is incompatible") from error
    return received
