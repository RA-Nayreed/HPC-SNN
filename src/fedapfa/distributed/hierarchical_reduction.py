"""Node-local FedAvg sufficient-statistics collection and global combination."""

from __future__ import annotations

import io
from dataclasses import dataclass

import torch
import torch.distributed as dist

from fedapfa.distributed.client_worker import RankClientPayload, order_client_results
from fedapfa.distributed.process_context import (
    ProcessContext,
    node_leader_process_group,
    node_process_group,
)
from fedapfa.federated.aggregation import (
    SufficientStatistics,
    build_sufficient_statistics,
    combine_sufficient_statistics,
    state_difference_cosine_similarity,
    sufficient_statistics_payload_bytes,
)
from fedapfa.federated.round_state import AggregationInput


@dataclass(frozen=True)
class NodeContribution:
    node_rank: int
    leader_global_rank: int
    statistics: SufficientStatistics
    client_metadata: tuple[dict, ...]
    logical_payload_bytes: int


def serialized_payload_bytes(value) -> int:
    stream = io.BytesIO()
    torch.save(value, stream)
    return len(stream.getbuffer())


def gather_node_payloads(
    payload: RankClientPayload, context: ProcessContext
) -> tuple[list[RankClientPayload] | None, list[int]]:
    """Move individual updates only within the process's node."""

    sizes: list[int | None] = [None for _ in range(context.local_world_size)]
    dist.all_gather_object(sizes, serialized_payload_bytes(payload), group=node_process_group(context))
    gathered = [None for _ in range(context.local_world_size)] if context.rank == context.node_leader_rank else None
    dist.gather_object(
        payload,
        object_gather_list=gathered,
        dst=context.node_leader_rank,
        group=node_process_group(context),
    )
    return (
        None if gathered is None else [value for value in gathered if value is not None],
        [int(value) for value in sizes if value is not None],
    )


def build_node_contribution(
    payloads: list[RankClientPayload],
    assignments,
    round_number: int,
    config: dict,
    incoming_state: dict[str, torch.Tensor],
    incoming_model_id: str,
    context: ProcessContext,
    normalized_weights_by_client: dict[str, float],
) -> tuple[NodeContribution, list]:
    """Validate node results and replace their tensors by one sufficient statistic."""

    node_assignments = [assignment for assignment in assignments if assignment.process_rank in context.node_group_ranks]
    results, envelopes = order_client_results(
        payloads,
        node_assignments,
        round_number,
        config,
        incoming_state,
        incoming_model_id,
    )
    node_inputs = [AggregationInput(result.client_id, result.example_count, result.state_dict) for result in results]
    statistics = build_sufficient_statistics(
        node_inputs,
        policy=config["federated"]["aggregation_weighting"],
        normalized_weights={
            item.client_id: normalized_weights_by_client[item.client_id]
            for item in node_inputs
        },
    )
    metadata = tuple(
        {
            **envelope.result.record(0.0, 0.0),
            "selected_position": envelope.selected_position,
            "process_rank": envelope.process_rank,
            "incoming_global_model_id": envelope.incoming_global_model_id,
            "update_identity": envelope.update_identity,
        }
        for envelope in envelopes
    )
    contribution = NodeContribution(
        node_rank=context.node_rank,
        leader_global_rank=context.node_leader_rank,
        statistics=statistics,
        client_metadata=metadata,
        logical_payload_bytes=sufficient_statistics_payload_bytes(statistics),
    )
    return contribution, envelopes


def gather_node_contributions(
    contribution: NodeContribution | None, context: ProcessContext
) -> list[NodeContribution] | None:
    """Move one tensor-bearing contribution per node to global rank zero."""

    if context.rank != context.node_leader_rank:
        if contribution is not None:
            raise ValueError("only a node leader may provide a node contribution")
        return None
    gathered = [None for _ in range(context.node_count)] if context.is_coordinator else None
    dist.gather_object(
        contribution,
        object_gather_list=gathered,
        dst=0,
        group=node_leader_process_group(context),
    )
    if gathered is None:
        return None
    values = [value for value in gathered if value is not None]
    if [value.node_rank for value in values] != list(range(context.node_count)):
        raise RuntimeError("node contributions are missing or out of node-rank order")
    return values


def combine_node_contributions(
    contributions: list[NodeContribution],
    selected_client_ids: list[str],
    weighting_policy: str,
) -> tuple[dict[str, torch.Tensor], list[float]]:
    return combine_sufficient_statistics(
        [value.statistics for value in contributions],
        policy=weighting_policy,
        expected_client_ids=selected_client_ids,
    )


def node_client_records(
    envelopes,
    incoming_state: dict[str, torch.Tensor],
    aggregated_state: dict[str, torch.Tensor],
    weights_by_client: dict[str, float],
) -> list[dict]:
    """Discard retained client tensors after deriving scalar comparison records."""

    records = []
    for envelope in envelopes:
        cosine = state_difference_cosine_similarity(
            envelope.result.state_dict,
            incoming_state,
            aggregated_state,
        )
        record = envelope.result.record(weights_by_client[envelope.result.client_id], cosine)
        record.update(
            {
                "selected_position": envelope.selected_position,
                "process_rank": envelope.process_rank,
                "incoming_global_model_id": envelope.incoming_global_model_id,
                "update_identity": envelope.update_identity,
                "completed_at_unix_nanoseconds": envelope.completed_at_unix_nanoseconds,
            }
        )
        records.append(record)
    return records


def gather_node_client_records(records: list[dict] | None, context: ProcessContext) -> list[dict] | None:
    """Gather scalar client records from node leaders without client tensors."""

    if context.rank != context.node_leader_rank:
        if records is not None:
            raise ValueError("only a node leader may provide client records")
        return None
    gathered = [None for _ in range(context.node_count)] if context.is_coordinator else None
    dist.gather_object(
        records,
        object_gather_list=gathered,
        dst=0,
        group=node_leader_process_group(context),
    )
    if gathered is None:
        return None
    flattened = [record for node_records in gathered if node_records for record in node_records]
    by_position = {record["selected_position"]: record for record in flattened}
    if len(by_position) != len(flattened):
        raise RuntimeError("hierarchical client records contain duplicate selected positions")
    return [by_position[position] for position in range(len(flattened))]
