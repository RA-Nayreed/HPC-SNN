"""Coordinator-owned pre-execution scheduler runtime."""

from __future__ import annotations

import json
import time

from fedapfa.federated.randomness import derive_seed

from .assignment import assign_selected_clients
from .base import EVENT_STRUCTURE_FEATURES, SCHEDULING_STRATEGIES, SchedulingPlan
from .client_features import EventStructureFeatureCache, privacy_metadata_record
from .runtime_cost_model import FrozenEventStructureModel


class SchedulerRuntime:
    """Load the frozen model once and schedule every round before client training."""

    def __init__(self, config: dict, bundle, repository_root=None) -> None:
        scheduler = config.get("scheduler")
        if not isinstance(scheduler, dict):
            raise ValueError("scheduler configuration is missing")
        if scheduler.get("strategy") not in SCHEDULING_STRATEGIES:
            raise ValueError("scheduler strategy is unsupported")
        self.config = config
        self.bundle = bundle
        self.strategy = scheduler["strategy"]
        model_load_started = time.perf_counter()
        self.model = FrozenEventStructureModel.load(
            scheduler["cost_model"],
            dataset_name=config["dataset"]["name"],
            model_name=config["model"]["name"],
            repository_root=repository_root,
        )
        self.model_load_seconds = time.perf_counter() - model_load_started
        self.feature_cache = EventStructureFeatureCache()
        self.model_load_count = 1

    def schedule(self, selected_client_ids: list[str], round_number: int, process_count: int) -> SchedulingPlan:
        started = time.perf_counter()
        federation = self.config["federated"]
        features: dict[str, dict[str, float]] = {}
        static_lookup = invariant_extraction = seed_dependent = 0.0
        privacy_records: list[dict] = []
        metadata_bytes = 0

        if self.strategy == "round_robin":
            costs = {client_id: 1.0 for client_id in selected_client_ids}
            cost_source = "selected_position"
        elif self.strategy == "example_count_longest_processing_time":
            lookup_started = time.perf_counter()
            costs = {client_id: float(len(self.bundle.client_dataset(client_id))) for client_id in selected_client_ids}
            static_lookup = time.perf_counter() - lookup_started
            cost_source = "training_example_count"
        else:
            for client_id in selected_client_ids:
                training_seed = derive_seed(
                    self.config["seed"],
                    self.config["seed_streams"]["client_training"],
                    round_number,
                    client_id,
                )
                extracted = self.feature_cache.features(
                    client_id,
                    self.bundle.client_dataset(client_id),
                    training_seed=training_seed,
                    batch_size=federation["local_batch_size"],
                    input_features=self.config["dataset"]["input_features"],
                    local_epochs=federation["local_epochs"],
                    drop_last=federation["drop_last_local_batch"],
                )
                features[client_id] = extracted.values
                static_lookup += extracted.static_lookup_seconds
                invariant_extraction += extracted.invariant_extraction_seconds
                seed_dependent += extracted.seed_dependent_seconds
                metadata, serialized_bytes = privacy_metadata_record(extracted.values)
                privacy_records.extend({"client_id": client_id, **record} for record in metadata)
                metadata_bytes += serialized_bytes
            prediction_started = time.perf_counter()
            predictions = self.model.predict([features[client_id] for client_id in selected_client_ids])
            prediction_duration = time.perf_counter() - prediction_started
            costs = dict(zip(selected_client_ids, predictions, strict=True))
            cost_source = "frozen_event_structure_wall_time_prediction"

        if self.strategy != "event_structure_longest_processing_time":
            prediction_duration = 0.0
            if self.strategy == "example_count_longest_processing_time":
                metadata_payload = json.dumps(costs, sort_keys=True, separators=(",", ":")).encode("utf-8")
                metadata_bytes = len(metadata_payload)
                privacy_records = [
                    {
                        "client_id": client_id,
                        "field": "example_count",
                        "value": costs[client_id],
                        "contains_label_information": False,
                        "may_reveal_workload_or_behavior": True,
                        "stability": "static",
                        "cacheable": True,
                        "raw_events_leave_client": False,
                    }
                    for client_id in selected_client_ids
                ]

        assignment_started = time.perf_counter()
        assignments, loads = assign_selected_clients(
            selected_client_ids,
            process_count,
            self.strategy,
            costs,
            cost_source=cost_source,
            features=features or None,
        )
        assignment_duration = time.perf_counter() - assignment_started
        elapsed = time.perf_counter() - started
        return SchedulingPlan(
            strategy=self.strategy,
            assignments=assignments,
            predicted_process_loads=loads,
            feature_availability={
                name: self.strategy == "event_structure_longest_processing_time" for name in EVENT_STRUCTURE_FEATURES
            },
            model_artifact_path=str(self.model.artifact_path),
            model_sha256=self.model.artifact_sha256,
            model_provenance_identity=self.model.provenance_identity,
            model_name="event_structure",
            static_feature_lookup_seconds=static_lookup,
            invariant_feature_extraction_seconds=invariant_extraction,
            seed_dependent_feature_seconds=seed_dependent,
            model_prediction_seconds=prediction_duration,
            sorting_and_assignment_seconds=assignment_duration,
            scheduler_seconds_before_broadcast=elapsed,
            metadata_serialized_bytes=metadata_bytes,
            privacy_metadata=privacy_records,
        )
