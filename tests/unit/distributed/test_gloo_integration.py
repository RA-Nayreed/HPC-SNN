import copy
import json
import os
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.utils.data import TensorDataset

from fedapfa.distributed import client_worker
from fedapfa.distributed.process_context import (
    close_process_context,
    initialize_process_context,
    process_resident_memory_bytes,
    verify_identity_consensus,
)
from fedapfa.federated.checkpointing import configuration_identity, state_identity
from fedapfa.federated.numerical_equivalence import classify_model_states
from fedapfa.scheduling.assignment import assign_selected_clients
from fedapfa.scheduling.base import EVENT_STRUCTURE_FEATURES, SchedulingPlan
from fedapfa.training import distributed_federated as distributed_training
from fedapfa.training.distributed_federated import train_distributed_federated
from fedapfa.utilities.run_records import initialize_run, run_directory
from fedapfa.utilities.serialization import sha256_json


class IntegrationFederatedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(2, 6)
        self.dropout = nn.Dropout(0.25)
        self.output = nn.Linear(6, 2)

    def forward(self, inputs, generator=None):
        hidden = torch.relu(self.hidden(inputs))
        logits = self.output(self.dropout(hidden))
        return logits, {"activation": (hidden > 0).float().mean()}


class CountingTensorDataset(TensorDataset):
    def __init__(self, *tensors):
        super().__init__(*tensors)
        self.access_count = 0

    def __getitem__(self, index):
        self.access_count += 1
        return super().__getitem__(index)


@dataclass
class IntegrationPartition:
    client_indices: dict[str, list[int]]
    partition_id: str
    artifact: dict


class IntegrationBundle:
    def __init__(self, config, coordinator):
        self.config = config
        sizes = [8, 10, 10, 12]
        self.datasets = {}
        client_indices = {}
        clients = []
        cursor = 0
        for client_number, size in enumerate(sizes):
            client_id = f"client_{client_number:02d}"
            values = torch.arange(size * 2, dtype=torch.float32).reshape(size, 2) / (size * 2)
            values[:, 0] += client_number * 0.05
            labels = torch.tensor([(index + client_number) % 2 for index in range(size)])
            self.datasets[client_id] = TensorDataset(values, labels)
            indices = list(range(cursor, cursor + size))
            client_indices[client_id] = indices
            clients.append(
                {
                    "client_id": client_id,
                    "indices": indices,
                    "size": size,
                    "class_counts": {
                        "0": int((labels == 0).sum()),
                        "1": int((labels == 1).sum()),
                    },
                }
            )
            cursor += size
        partition_core = {
            "schema_version": 1,
            "client_count": 4,
            "clients": clients,
            "integrity_checks": {
                "all_indices_assigned": True,
                "no_overlap": True,
                "minimum_size_satisfied": True,
            },
        }
        partition_artifact = dict(partition_core)
        partition_artifact["partition_id"] = sha256_json(partition_core)
        self.partition = IntegrationPartition(
            client_indices=client_indices,
            partition_id=partition_artifact["partition_id"],
            artifact=partition_artifact,
        )
        training_indices = list(range(cursor))
        validation_indices = list(range(cursor, cursor + 4))
        split_core = {
            "schema_version": 1,
            "split_seed": config["seed"],
            "validation_fraction": 0.1,
            "dataset_identity": {"name": "integration_events", "sha256": "integration"},
            "training_indices": training_indices,
            "validation_indices": validation_indices,
        }
        self.split_artifact = dict(split_core)
        self.split_artifact["split_id"] = sha256_json(split_core)
        self.train_indices = np.asarray(training_indices)
        self.validation_indices = np.asarray(validation_indices)
        self.validation_dataset = (
            CountingTensorDataset(
                torch.tensor([[0.0, 0.1], [0.8, 0.9], [0.2, 0.3], [0.7, 0.6]]),
                torch.tensor([0, 1, 0, 1]),
            )
            if coordinator
            else None
        )
        self.resolved_seed_values = {
            "split": config["seed"],
            "partition": 101,
            "model_initialization": 202,
            "client_selection": 303,
            "client_training": 404,
            "validation": 505,
            "final_test": 606,
        }
        self.official_test_access_count = 0
        self.official_test_identity = {"name": "integration_test", "examples": 6}

    @property
    def client_ids(self):
        return sorted(self.partition.client_indices)

    @property
    def aggregation_weighting(self):
        return self.config["federated"]["aggregation_weighting"]

    @property
    def checkpoint_selection(self):
        return self.config["federated"]["checkpoint_selection"]

    @property
    def evaluation_protocol(self):
        return {
            "validation_collection": "derived_training_validation",
            "internal_validation_available": True,
            "official_test_publication_collection_name": None,
            "external_implementation_monitors_official_test": False,
            "complete_standard_training_collection": False,
        }

    def client_dataset(self, client_id):
        return self.datasets[client_id]

    def official_test_dataset(self, model_selected):
        if not model_selected or self.official_test_access_count != 0:
            raise RuntimeError("official test access is incompatible")
        self.official_test_access_count += 1
        return TensorDataset(
            torch.tensor([[0.0, 0.2], [0.9, 0.7], [0.1, 0.4], [0.8, 0.5], [0.3, 0.2], [0.6, 0.9]]),
            torch.tensor([0, 1, 0, 1, 0, 1]),
        )


def _config(
    output_root: str,
    world_size: int,
    topology: str = "flat_ordered",
    scheduler_strategy: str | None = None,
) -> dict:
    count_name = {1: "one", 2: "two", 4: "four"}[world_size]
    node_count = 2 if world_size == 4 else 1
    devices_per_node = world_size // node_count
    strategy_suffix = "" if scheduler_strategy is None else f"_{scheduler_strategy}"
    config = {
        "name": f"integration_federated_{count_name}_gpu_{topology}{strategy_suffix}",
        "seed": 7,
        "mode": "scientific_evaluation",
        "execution": "federated",
        "protocol": "independent_evaluation",
        "device": "cpu",
        "output_root": output_root,
        "pairing_group": "integration_federated_execution",
        "dataset": {"name": "shd", "validation_fraction": 0.1},
        "model": {"name": "lif_2layer", "timesteps": None, "input_encoding": None},
        "subset": {
            "train_examples": 0,
            "validation_examples": 0,
            "test_examples": 0,
        },
        "training": {
            "max_train_batches": None,
            "max_validation_batches": None,
            "max_test_batches": None,
        },
        "federated": {
            "algorithm": "fedavg",
            "rounds": 2,
            "clients": 4,
            "clients_per_round": 4,
            "participation_fraction": 1.0,
            "local_epochs": 1,
            "local_batch_size": 2,
            "optimizer": "adam",
            "learning_rate": 0.01,
            "weight_decay": 0.0,
            "gradient_clip": 1.0,
            "learning_rate_reduction_rounds": [],
            "learning_rate_reduction_factor": 1.0,
            "aggregation_weighting": "example_count",
            "drop_last_local_batch": False,
            "checkpoint_selection": "best_validation",
            "data_loader_workers": 0,
            "persistent_workers": False,
            "pin_memory": False,
            "prefetch_factor": None,
            "non_blocking_transfer": True,
            "record_extended_diagnostics": False,
            "partition": {"method": "label_dirichlet", "alpha": 0.5, "minimum_examples_per_client": 1},
        },
        "seed_streams": {
            "split": "split",
            "partition": "partition",
            "model_initialization": "model_initialization",
            "client_selection": "client_selection",
            "client_training": "client_training",
            "validation": "validation",
            "final_test": "final_test",
        },
        "acceptance": {
            "expected_model_class": "IntegrationFederatedModel",
            "reference_test_accuracy": None,
            "absolute_tolerance": None,
            "descriptive_reference_accuracy": None,
        },
        "parallel_execution": {
            "node_count": node_count,
            "devices_per_node": devices_per_node,
            "device_count": world_size,
            "client_processes_per_device": 1,
            "process_count": world_size,
            "control_backend": "gloo",
            "cuda_process_service": "none",
            "client_assignment": scheduler_strategy or "selected_order_round_robin",
            "aggregation_order": "selected_client_order",
            "synchronize_each_round": True,
            "aggregation_topology": topology,
            "rank_mapping": "node_major_local_device_index",
        },
        "aggregation_execution": {
            "topology": topology,
            "absolute_tolerance": 1e-6,
            "relative_tolerance": 1e-5,
        },
        "execution_measurement": {
            "profiler_enabled": False,
            "profiled_rounds": [],
            "record_cuda_memory": True,
            "record_device_utilization": False,
            "utilization_interval_seconds": 2,
        },
    }
    if scheduler_strategy is not None:
        config["scheduler"] = {
            "strategy": scheduler_strategy,
            "cost_model": {"fixture": True},
        }
        config["evaluation"] = {"collection": "synthetic_scheduler_equivalence"}
    return config


def _model():
    state = torch.get_rng_state()
    torch.manual_seed(8181)
    try:
        return IntegrationFederatedModel()
    finally:
        torch.set_rng_state(state)


def _worker(
    rank,
    world_size,
    port,
    output_root,
    topology,
    scheduler_strategy,
    scheduler_ready,
    start_events,
    result_queue,
):
    local_world_size = 2 if world_size == 4 else world_size
    os.environ.update(
        {
            "RANK": str(rank),
            "LOCAL_RANK": str(rank % local_world_size),
            "WORLD_SIZE": str(world_size),
            "LOCAL_WORLD_SIZE": str(local_world_size),
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port),
        }
    )
    config = _config(output_root, world_size, topology, scheduler_strategy)
    context = initialize_process_context(config["parallel_execution"], allow_gloo=True)
    base_train_client = client_worker.train_client
    base_aggregate = distributed_training.aggregate_client_results
    base_hierarchical_aggregate = distributed_training.combine_node_contributions
    base_scheduler_runtime = distributed_training.SchedulerRuntime
    aggregation_call_count = 0
    hierarchical_aggregation_call_count = 0

    def counted_aggregation(*args, **kwargs):
        nonlocal aggregation_call_count
        aggregation_call_count += 1
        return base_aggregate(*args, **kwargs)

    def synchronized_train_client(*args, **kwargs):
        if scheduler_strategy is not None and not scheduler_ready.is_set():
            raise RuntimeError("client training started before scheduling completed")
        start_events[rank].set()
        if not all(value.wait(timeout=30) for value in start_events):
            raise RuntimeError("client processes did not enter local training concurrently")
        if world_size == 4:
            time.sleep((world_size - rank) * 0.01)
        return base_train_client(*args, **kwargs)

    def counted_hierarchical_aggregation(*args, **kwargs):
        nonlocal hierarchical_aggregation_call_count
        hierarchical_aggregation_call_count += 1
        return base_hierarchical_aggregate(*args, **kwargs)

    class IntegrationSchedulerRuntime:
        def __init__(self, runtime_config, bundle):
            self.strategy = runtime_config["scheduler"]["strategy"]
            self.bundle = bundle
            self.model = SimpleNamespace(
                artifact_path=Path("synthetic-cost-model.json"),
                artifact_sha256="2" * 64,
                provenance_identity="synthetic-model-provenance",
                row_provenance={"source": "synthetic_gloo_integration_fixture"},
            )
            self.model_load_seconds = 0.0
            self.model_load_count = 1

        def schedule(self, selected_client_ids, round_number, process_count):
            if self.strategy == "round_robin":
                costs = {client_id: 1.0 for client_id in selected_client_ids}
                source = "selected_position"
                features = None
                privacy = []
            else:
                costs = {
                    client_id: float(len(self.bundle.client_dataset(client_id))) for client_id in selected_client_ids
                }
                source = (
                    "training_example_count"
                    if self.strategy == "example_count_longest_processing_time"
                    else "frozen_event_structure_wall_time_prediction"
                )
                features = (
                    None
                    if self.strategy == "example_count_longest_processing_time"
                    else {
                        client_id: {name: costs[client_id] for name in EVENT_STRUCTURE_FEATURES}
                        for client_id in selected_client_ids
                    }
                )
                fields = (
                    ("example_count",)
                    if self.strategy == "example_count_longest_processing_time"
                    else EVENT_STRUCTURE_FEATURES
                )
                privacy = [
                    {
                        "client_id": client_id,
                        "field": name,
                        "contains_label_information": False,
                        "raw_events_leave_client": False,
                    }
                    for client_id in selected_client_ids
                    for name in fields
                ]
            assignments, loads = assign_selected_clients(
                selected_client_ids,
                process_count,
                self.strategy,
                costs,
                cost_source=source,
                features=features,
            )
            scheduler_ready.set()
            return SchedulingPlan(
                strategy=self.strategy,
                assignments=assignments,
                predicted_process_loads=loads,
                feature_availability={
                    name: self.strategy == "event_structure_longest_processing_time"
                    for name in EVENT_STRUCTURE_FEATURES
                },
                model_artifact_path=str(self.model.artifact_path),
                model_sha256=self.model.artifact_sha256,
                model_provenance_identity=self.model.provenance_identity,
                model_name="event_structure",
                static_feature_lookup_seconds=0.0,
                invariant_feature_extraction_seconds=0.0,
                seed_dependent_feature_seconds=0.0,
                model_prediction_seconds=0.0,
                sorting_and_assignment_seconds=0.0,
                scheduler_seconds_before_broadcast=0.0,
                metadata_serialized_bytes=0,
                privacy_metadata=privacy,
            )

    client_worker.train_client = synchronized_train_client
    distributed_training.aggregate_client_results = counted_aggregation
    distributed_training.combine_node_contributions = counted_hierarchical_aggregation
    if scheduler_strategy is not None:
        distributed_training.SchedulerRuntime = IntegrationSchedulerRuntime
    try:
        resident_memory_before_workload = process_resident_memory_bytes()
        bundle = IntegrationBundle(config, context.is_coordinator)
        model = _model()
        run_dir = run_directory(config)
        if context.is_coordinator:
            initialize_run(
                config,
                {"train": bundle.train_indices.tolist(), "validation": bundle.validation_indices.tolist()},
                "gloo integration execution",
            )
        dist.barrier()
        identity = {
            "configuration_id": configuration_identity(config),
            "split_id": bundle.split_artifact["split_id"],
            "partition_id": bundle.partition.partition_id,
            "model_initialization_id": state_identity(model.state_dict()),
            "resolved_seeds": bundle.resolved_seed_values,
            "world_size": world_size,
        }
        process_records = verify_identity_consensus(
            context,
            identity,
            process_resident_memory_before_workload_bytes=resident_memory_before_workload,
        )
        first = train_distributed_federated(
            model,
            bundle,
            config,
            run_dir,
            context,
            process_records,
            stop_after_round=1,
        )
        if context.is_coordinator and (first["completed"] or first["completed_rounds"] != 1):
            raise RuntimeError("interrupted execution did not stop after the first round")
        if bundle.official_test_access_count != 0:
            raise RuntimeError("official test was accessed before model selection")
        resumed_process_records = copy.deepcopy(process_records)
        for record in resumed_process_records:
            record["host"] = f"resumed-allocation-{world_size}"
            record["process_resident_memory_after_workload_bytes"] += 4096
            record["workload_resident_memory_delta_bytes"] += 4096
        if world_size == 1:
            checkpoint_path = Path(run_dir) / "checkpoints" / "last.pt"

            def failing_client(*args, **kwargs):
                raise RuntimeError("controlled client failure")

            client_worker.train_client = failing_client
            try:
                train_distributed_federated(
                    _model(),
                    bundle,
                    config,
                    run_dir,
                    context,
                    resumed_process_records,
                    resume_checkpoint=checkpoint_path,
                )
            except RuntimeError as error:
                if "controlled client failure" not in str(error):
                    raise
            else:
                raise RuntimeError("controlled client failure was not propagated")
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if checkpoint["next_round"] != 2 or len(checkpoint["round_records"]) != 1:
                raise RuntimeError("client failure advanced the durable checkpoint")
            if bundle.official_test_access_count != 0:
                raise RuntimeError("official test was accessed after a failed round")
            client_worker.train_client = synchronized_train_client
        resumed_model = _model()
        result = train_distributed_federated(
            resumed_model,
            bundle,
            config,
            run_dir,
            context,
            resumed_process_records,
            resume_checkpoint=Path(run_dir) / "checkpoints" / "last.pt",
        )
        if context.is_coordinator:
            rounds = [
                json.loads(line)
                for line in (Path(run_dir) / "round_metrics.jsonl").read_text().splitlines()
                if line.strip()
            ]
            final = json.loads((Path(run_dir) / "final_metrics.json").read_text())
            official = json.loads((Path(run_dir) / "official_test_metrics.json").read_text())
            execution_measurements = json.loads((Path(run_dir) / "execution_measurements.json").read_text())
            stable_process_mapping = json.loads((Path(run_dir) / "process_mapping.json").read_text())
            client_metrics = [
                json.loads(line)
                for line in (Path(run_dir) / "client_metrics.jsonl").read_text().splitlines()
                if line.strip()
            ]
            result_queue.put(
                {
                    "rank": rank,
                    "official_accesses": bundle.official_test_access_count,
                    "validation_accesses": bundle.validation_dataset.access_count,
                    "completed": result["completed"],
                    "model_identity": state_identity(resumed_model.state_dict()),
                    "selected_orders": [value["selected_client_ids"] for value in rounds],
                    "assignments": [value["client_assignments"] for value in rounds],
                    "ordered_update_identities": [value["ordered_update_identities"] for value in rounds],
                    "completion_orders": [
                        [item["completion_order"] for item in value["client_assignments"]] for value in rounds
                    ],
                    "aggregation_topologies": [value["aggregation_topology"] for value in rounds],
                    "logical_intra_node_bytes": [value["logical_intra_node_bytes"] for value in rounds],
                    "logical_inter_node_bytes": [value["logical_inter_node_bytes"] for value in rounds],
                    "model_sized_payloads_crossing_node_boundaries": [
                        value["model_sized_payloads_crossing_node_boundaries"] for value in rounds
                    ],
                    "incoming_identity_counts": [
                        len({item["incoming_global_model_id"] for item in value})
                        for value in (
                            [
                                record
                                for record in (
                                    json.loads(line)
                                    for line in (Path(run_dir) / "client_metrics.jsonl").read_text().splitlines()
                                    if line.strip()
                                )
                                if record["round_number"] == round_number
                            ]
                            for round_number in (1, 2)
                        )
                    ],
                    "client_training_seed_records": [
                        (
                            record["round_number"],
                            record["client_id"],
                            record["resolved_training_seed"],
                        )
                        for record in client_metrics
                    ],
                    "logical_communication": final["logical_communication"]["cumulative_total_bytes"],
                    "selected_round": final["selected_round"],
                    "official_record_accesses": official["access_count"],
                    "resume_count": execution_measurements["resume_count"],
                    "process_mapping_attempt_count": len(execution_measurements["process_mapping_attempts"]),
                    "stable_mapping_excludes_allocation_measurements": all(
                        "host" not in value
                        and "process_resident_memory_bytes" not in value
                        and "workload_resident_memory_delta_bytes" not in value
                        for value in stable_process_mapping["processes"]
                    ),
                    "workload_memory_recorded": all(
                        value["process_resident_memory_before_workload_bytes"] is not None
                        and value["process_resident_memory_after_workload_bytes"] is not None
                        and value["workload_resident_memory_delta_bytes"] is not None
                        for value in process_records
                    ),
                    "aggregation_call_count": aggregation_call_count,
                    "hierarchical_aggregation_call_count": hierarchical_aggregation_call_count,
                    "scheduler_model": final["scheduler_model"],
                }
            )
        else:
            result_queue.put(
                {
                    "rank": rank,
                    "official_accesses": bundle.official_test_access_count,
                    "validation_accesses": 0,
                    "aggregation_call_count": aggregation_call_count,
                    "hierarchical_aggregation_call_count": hierarchical_aggregation_call_count,
                }
            )
    finally:
        client_worker.train_client = base_train_client
        distributed_training.aggregate_client_results = base_aggregate
        distributed_training.combine_node_contributions = base_hierarchical_aggregate
        distributed_training.SchedulerRuntime = base_scheduler_runtime
        close_process_context()


def _available_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind(("127.0.0.1", 0))
        return handle.getsockname()[1]


def _execute(
    output_root,
    world_size,
    topology="flat_ordered",
    scheduler_strategy=None,
):
    context = mp.get_context("spawn")
    start_events = [context.Event() for _ in range(world_size)]
    scheduler_ready = context.Event()
    result_queue = context.SimpleQueue()
    mp.spawn(
        _worker,
        args=(
            world_size,
            _available_port(),
            str(output_root),
            topology,
            scheduler_strategy,
            scheduler_ready,
            start_events,
            result_queue,
        ),
        nprocs=world_size,
        join=True,
    )
    records = [result_queue.get() for _ in range(world_size)]
    coordinator = next(value for value in records if value["rank"] == 0)
    return (
        coordinator,
        records,
        all(value.is_set() for value in start_events),
        scheduler_ready.is_set(),
    )


def test_one_and_two_process_gloo_execution_are_numerically_equivalent(tmp_path):
    one, one_process_records, one_overlap, _ = _execute(tmp_path / "one_gpu", 1)
    two, two_process_records, two_overlap, _ = _execute(tmp_path / "two_gpu", 2)
    assert one["completed"] and two["completed"]
    assert one["model_identity"] == two["model_identity"]
    assert one["selected_orders"] == two["selected_orders"]
    assert one["ordered_update_identities"] == two["ordered_update_identities"]
    assert one["logical_communication"] == two["logical_communication"]
    assert one["incoming_identity_counts"] == two["incoming_identity_counts"] == [1, 1]
    assert one["client_training_seed_records"] == two["client_training_seed_records"]
    assert one["selected_round"] == two["selected_round"]
    assert one["resume_count"] == 2
    assert two["resume_count"] == 1
    assert one["process_mapping_attempt_count"] == 3
    assert two["process_mapping_attempt_count"] == 2
    assert one["stable_mapping_excludes_allocation_measurements"]
    assert two["stable_mapping_excludes_allocation_measurements"]
    assert one["official_record_accesses"] == two["official_record_accesses"] == 1
    assert one["workload_memory_recorded"] and two["workload_memory_recorded"]
    assert one["aggregation_call_count"] == two["aggregation_call_count"] == 2
    assert [value["aggregation_call_count"] for value in one_process_records] == [2]
    assert sorted(value["aggregation_call_count"] for value in two_process_records) == [0, 2]
    assert one_overlap and two_overlap
    assert [value["official_accesses"] for value in one_process_records] == [1]
    assert sorted(value["official_accesses"] for value in two_process_records) == [0, 1]
    assert next(value for value in two_process_records if value["rank"] == 0)["validation_accesses"] > 0
    assert next(value for value in two_process_records if value["rank"] == 1)["validation_accesses"] == 0
    assert [[item["process_rank"] for item in value] for value in two["assignments"]] == [
        [0, 1, 0, 1],
        [0, 1, 0, 1],
    ]


def test_four_process_flat_and_hierarchical_training_are_structurally_and_mathematically_equivalent(
    tmp_path,
):
    flat_root = tmp_path / "flat"
    hierarchy_root = tmp_path / "hierarchy"
    flat, flat_processes, flat_overlap, _ = _execute(flat_root, 4, "flat_ordered")
    hierarchy, hierarchy_processes, hierarchy_overlap, _ = _execute(hierarchy_root, 4, "node_hierarchical")
    assert flat["completed"] and hierarchy["completed"] and flat_overlap and hierarchy_overlap
    assert flat["selected_orders"] == hierarchy["selected_orders"]
    assert flat["client_training_seed_records"] == hierarchy["client_training_seed_records"]
    assert flat["ordered_update_identities"] == hierarchy["ordered_update_identities"]
    assert flat["logical_communication"] == hierarchy["logical_communication"]
    assert flat["selected_round"] == hierarchy["selected_round"]
    assert flat["official_record_accesses"] == hierarchy["official_record_accesses"] == 1
    flat_by_rank = {value["rank"]: value for value in flat_processes}
    hierarchy_by_rank = {value["rank"]: value for value in hierarchy_processes}
    assert [flat_by_rank[rank]["official_accesses"] for rank in range(4)] == [1, 0, 0, 0]
    assert [hierarchy_by_rank[rank]["official_accesses"] for rank in range(4)] == [1, 0, 0, 0]
    assert all(flat_by_rank[rank]["validation_accesses"] == 0 for rank in range(1, 4))
    assert all(hierarchy_by_rank[rank]["validation_accesses"] == 0 for rank in range(1, 4))
    assert flat["aggregation_call_count"] == 2
    assert flat["hierarchical_aggregation_call_count"] == 0
    assert hierarchy["aggregation_call_count"] == 0
    assert hierarchy["hierarchical_aggregation_call_count"] == 2
    assert hierarchy["aggregation_topologies"] == ["node_hierarchical", "node_hierarchical"]
    assert all(value > 0 for value in hierarchy["logical_intra_node_bytes"])
    assert all(value > 0 for value in hierarchy["logical_inter_node_bytes"])
    assert hierarchy["model_sized_payloads_crossing_node_boundaries"] == [1, 1]
    assert any(value != [1, 2, 3, 4] for value in flat["completion_orders"])

    flat_checkpoint = torch.load(
        run_directory(_config(str(flat_root), 4, "flat_ordered")) / "checkpoints" / "best.pt",
        map_location="cpu",
        weights_only=False,
    )["global_model_state"]
    hierarchy_checkpoint = torch.load(
        run_directory(_config(str(hierarchy_root), 4, "node_hierarchical")) / "checkpoints" / "best.pt",
        map_location="cpu",
        weights_only=False,
    )["global_model_state"]
    classification = classify_model_states(
        flat_checkpoint,
        hierarchy_checkpoint,
        absolute_tolerance=1e-6,
        relative_tolerance=1e-5,
    )
    assert classification.structural_identity
    assert classification.mathematical_equivalence


def test_scheduler_strategies_change_only_process_placement(tmp_path):
    records = {}
    for strategy in (
        "round_robin",
        "example_count_longest_processing_time",
        "event_structure_longest_processing_time",
    ):
        coordinator, processes, overlap, scheduler_ready = _execute(
            tmp_path / strategy,
            2,
            scheduler_strategy=strategy,
        )
        assert coordinator["completed"] and overlap and scheduler_ready
        assert coordinator["scheduler_model"]["model_load_count"] == 1
        assert sorted(value["official_accesses"] for value in processes) == [0, 1]
        records[strategy] = coordinator

    reference = records["round_robin"]
    for candidate in records.values():
        assert candidate["selected_orders"] == reference["selected_orders"]
        assert candidate["client_training_seed_records"] == reference["client_training_seed_records"]
        assert candidate["ordered_update_identities"] == reference["ordered_update_identities"]
        assert candidate["model_identity"] == reference["model_identity"]
        assert candidate["selected_round"] == reference["selected_round"]
        assert candidate["official_record_accesses"] == 1
        assert candidate["aggregation_call_count"] == 2
    assert [
        [value["process_rank"] for value in assignments] for assignments in records["round_robin"]["assignments"]
    ] != [
        [value["process_rank"] for value in assignments]
        for assignments in records["example_count_longest_processing_time"]["assignments"]
    ]
