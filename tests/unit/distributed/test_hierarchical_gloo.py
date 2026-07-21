import os
import socket

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from fedapfa.distributed.hierarchical_reduction import (
    NodeContribution,
    combine_node_contributions,
    gather_node_contributions,
)
from fedapfa.distributed.process_context import (
    close_process_context,
    initialize_process_context,
    node_process_group,
)
from fedapfa.federated.aggregation import (
    aggregation_weights,
    build_sufficient_statistics,
    sufficient_statistics_payload_bytes,
)
from fedapfa.federated.round_state import AggregationInput


def _port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind(("127.0.0.1", 0))
        return handle.getsockname()[1]


def _worker(rank, port, queue):
    os.environ.update(
        {
            "RANK": str(rank),
            "LOCAL_RANK": str(rank % 2),
            "WORLD_SIZE": "4",
            "LOCAL_WORLD_SIZE": "2",
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port),
        }
    )
    parallel = {
        "node_count": 2,
        "devices_per_node": 2,
        "device_count": 4,
        "client_processes_per_device": 1,
        "process_count": 4,
        "control_backend": "gloo",
        "cuda_process_service": "none",
    }
    context = initialize_process_context(parallel, allow_gloo=True)
    try:
        update = AggregationInput(
            client_id=f"client_{rank}",
            example_count=rank + 1,
            state_dict={"weight": torch.tensor([float(rank + 1)], dtype=torch.float32)},
        )
        node_updates = [None, None]
        dist.all_gather_object(node_updates, update, group=node_process_group(context))
        if context.rank == context.node_leader_rank:
            global_weight_inputs = [
                AggregationInput(f"client_{client_rank}", client_rank + 1, {})
                for client_rank in range(4)
            ]
            global_weights = dict(
                zip(
                    [item.client_id for item in global_weight_inputs],
                    aggregation_weights(global_weight_inputs, "example_count"),
                    strict=True,
                )
            )
            statistics = build_sufficient_statistics(
                node_updates,
                policy="example_count",
                normalized_weights={item.client_id: global_weights[item.client_id] for item in node_updates},
            )
            contribution = NodeContribution(
                context.node_rank,
                context.node_leader_rank,
                statistics,
                (),
                sufficient_statistics_payload_bytes(statistics),
            )
        else:
            contribution = None
        contributions = gather_node_contributions(contribution, context)
        if context.is_coordinator:
            state, weights = combine_node_contributions(
                contributions,
                ["client_0", "client_1", "client_2", "client_3"],
                "example_count",
            )
            queue.put(
                {
                    "value": float(state["weight"].item()),
                    "weights": weights,
                    "node_groups": [list(value.statistics.client_ids) for value in contributions],
                    "payload_count": len(contributions) - 1,
                }
            )
        dist.barrier()
    finally:
        close_process_context()


def test_four_process_gloo_uses_two_synthetic_node_groups_and_one_remote_contribution():
    context = mp.get_context("spawn")
    queue = context.SimpleQueue()
    mp.spawn(_worker, args=(_port(), queue), nprocs=4, join=True)
    result = queue.get()
    assert result["node_groups"] == [["client_0", "client_1"], ["client_2", "client_3"]]
    assert result["weights"] == [0.1, 0.2, 0.3, 0.4]
    assert result["value"] == 3.0
    assert result["payload_count"] == 1
