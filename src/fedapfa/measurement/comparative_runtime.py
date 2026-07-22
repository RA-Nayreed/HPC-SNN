"""Distributed measurement lifecycle for comparative energy evaluations."""

from __future__ import annotations

import hashlib
import json
import os
import statistics
import time
from contextlib import nullcontext
from pathlib import Path

import torch.distributed as dist

from fedapfa.distributed.process_context import canonical_gpu_uuid
from fedapfa.federated.checkpointing import state_identity
from fedapfa.federated.client import train_client
from fedapfa.federated.randomness import derive_seed
from fedapfa.utilities.git_metadata import git_metadata
from fedapfa.utilities.serialization import atomic_write_json, atomic_write_text

from .client_interval import ClientIntervalIdentity, IntervalRecorder
from .clock import CudaTimingAdapter, SystemMonotonicClock
from .energy import integrate_energy
from .features import FEATURE_AVAILABILITY, ObservedClientWork, extract_static_client_features
from .multi_gpu_energy import (
    NodeNvmlProcessSampler,
    integrate_physical_devices,
    merge_node_telemetry,
    validate_client_interval_nonoverlap,
    validate_comparative_calibration,
)
from .records import append_jsonl, read_jsonl


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _allocation_identity() -> str:
    job = os.environ.get("SLURM_JOB_ID")
    array = os.environ.get("SLURM_ARRAY_JOB_ID")
    task = os.environ.get("SLURM_ARRAY_TASK_ID")
    if not job:
        raise RuntimeError("measured comparative execution requires SLURM_JOB_ID")
    return f"{array}_{task}:{job}" if array and task else job


def _next_attempt(run_dir: Path) -> int:
    root = run_dir / "measurement_attempts"
    attempts = []
    if root.is_dir():
        for value in root.iterdir():
            if value.is_dir() and value.name.startswith("attempt_"):
                try:
                    attempts.append(int(value.name.removeprefix("attempt_")))
                except ValueError:
                    continue
    return max(attempts or [0]) + 1


def _load_frozen_model(path: Path, expected_sha256: str, feature_order: list[str]) -> dict:
    if _sha256(path) != expected_sha256:
        raise RuntimeError(f"frozen diagnostic SHA-256 differs: {path}")
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if artifact.get("feature_order") != feature_order:
        raise RuntimeError(f"frozen diagnostic feature order differs: {path}")
    required = ("coefficients", "intercept", "standardization_means", "standardization_scales")
    if any(not isinstance(artifact.get(value), (list, int, float)) for value in required):
        raise RuntimeError(f"frozen diagnostic model is malformed: {path}")
    lengths = {
        len(feature_order),
        len(artifact["coefficients"]),
        len(artifact["standardization_means"]),
        len(artifact["standardization_scales"]),
    }
    if len(lengths) != 1 or any(float(value) <= 0 for value in artifact["standardization_scales"]):
        raise RuntimeError(f"frozen diagnostic dimensions are incompatible: {path}")
    return artifact


def _predict(model: dict, features: dict) -> float:
    values = [float(features[name]) for name in model["feature_order"]]
    standardized = [
        (value - float(mean)) / float(scale)
        for value, mean, scale in zip(
            values,
            model["standardization_means"],
            model["standardization_scales"],
            strict=True,
        )
    ]
    return float(model["intercept"]) + sum(
        float(coefficient) * value for coefficient, value in zip(model["coefficients"], standardized, strict=True)
    )


class _ComparativeClientHook:
    def __init__(self, session, identity, static_features, device) -> None:
        self.session = session
        self.identity = identity
        self.static_features = static_features
        self.timing = CudaTimingAdapter(device, session.clock)
        self.observed = ObservedClientWork()
        self.timing_result = None
        self.data_wait_seconds = 0.0
        self._open = False
        self._excluded = False

    def start(self) -> None:
        self.timing.start()
        self._open = True

    def begin_device_work(self):
        return self.timing.begin_device_work()

    def end_device_work(self, token) -> None:
        self.timing.end_device_work(token)

    def observe_batch(self, batch, rates) -> None:
        self.observed.observe(batch, rates, self.session.layer_widths)

    def finish(self, data_wait_seconds: float) -> None:
        self.timing_result = self.timing.finish()
        self.data_wait_seconds = float(data_wait_seconds)
        self._open = False

    def _exclude(self, start_ns: int, end_ns: int, reason: str, data_wait: float = 0.0) -> None:
        if self._excluded:
            return
        record = self.session.intervals.record_client(
            self.identity,
            start_ns,
            end_ns,
            data_wait,
            0.0,
            accepted=False,
            exclusion_reason=reason,
        )
        append_jsonl(self.session.excluded_interval_path, {"interval_id": record.interval_id, "reason": reason})
        self._excluded = True

    def abort_if_open(self) -> None:
        if self._open and not self._excluded:
            start_ns = getattr(self.timing, "_start_ns", self.session.clock.now_ns())
            self._exclude(int(start_ns), self.session.clock.now_ns(), "client_training_interrupted")
            self._open = False

    def reject_after_finish(self, reason: str) -> None:
        if self.timing_result is not None:
            self._exclude(
                self.timing_result.start_ns,
                self.timing_result.end_ns,
                reason,
                self.data_wait_seconds,
            )


class ComparativeMeasurementSession:
    """Measure one attempt on every rank while one sampler process writes per node."""

    def __init__(self, config, run_dir, bundle, model, context, calibration_path: str | Path) -> None:
        if context.device.type != "cuda" or context.client_processes_per_device != 1:
            raise RuntimeError("comparative measurement requires one exclusive CUDA process per GPU")
        if any(name.startswith("CUDA_MPS") and value for name, value in os.environ.items()):
            raise RuntimeError("CUDA MPS is incompatible with comparative measurement")
        self.config = config
        self.run_dir = Path(run_dir)
        self.bundle = bundle
        self.model = model
        self.context = context
        self.measurement = config["energy_measurement"]
        self.clock = SystemMonotonicClock()
        self.gpu_uuid = canonical_gpu_uuid(context.gpu_uuid)
        self.git_commit = git_metadata().get("commit")
        self.allocation_identity = _allocation_identity()
        mappings = [None for _ in range(context.world_size)]
        dist.all_gather_object(
            mappings,
            {
                "rank": context.rank,
                "node_rank": context.node_rank,
                "host": context.host,
                "gpu_uuid": self.gpu_uuid,
                "gpu_uuid_raw": context.gpu_uuid_raw,
            },
        )
        self.process_mappings = sorted(
            (value for value in mappings if value is not None), key=lambda value: value["rank"]
        )
        if len(self.process_mappings) != context.world_size:
            raise RuntimeError("measurement process-to-device mapping is incomplete")
        attempt_value = [_next_attempt(self.run_dir) if context.is_coordinator else None]
        dist.broadcast_object_list(attempt_value, src=0, device=context.control_device)
        self.execution_attempt = int(attempt_value[0])
        self.attempt_dir = self.run_dir / "measurement_attempts" / f"attempt_{self.execution_attempt}"
        self.rank_dir = self.attempt_dir / f"rank_{context.rank}"
        self.interval_path = self.rank_dir / "execution_intervals.jsonl"
        self.provisional_path = self.rank_dir / "client_resource_records.jsonl"
        self.excluded_interval_path = self.rank_dir / "excluded_intervals.jsonl"
        self.intervals = IntervalRecorder(self.interval_path, self.clock)
        self._started = False
        self._open_scopes = []
        self._static_cache = {}
        hidden = config["model"]["hidden_dims"]
        self.layer_widths = {"layer1": int(hidden[0]), "layer2": int(hidden[1])}
        self.model_initialization_id = state_identity(model.state_dict())
        self.model_configuration_identity = hashlib.sha256(
            json.dumps(config["model"], sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        root = Path(__file__).resolve().parents[3]
        diagnostics = config["frozen_model_diagnostics"]
        self.runtime_model = _load_frozen_model(
            root / diagnostics["runtime"]["model_path"],
            diagnostics["runtime"]["model_sha256"],
            diagnostics["runtime"]["feature_order"],
        )
        self.energy_model = _load_frozen_model(
            root / diagnostics["gross_energy"]["model_path"],
            diagnostics["gross_energy"]["model_sha256"],
            diagnostics["gross_energy"]["feature_order"],
        )
        if self.runtime_model.get("target") != "client_wall_time_seconds":
            raise RuntimeError("frozen runtime diagnostic target is incompatible")
        if self.energy_model.get("target") != "gross_energy_joules":
            raise RuntimeError("frozen energy diagnostic target is incompatible")
        calibration_source = Path(calibration_path).resolve()
        artifact = json.loads(calibration_source.read_text(encoding="utf-8"))
        calibration_sha256 = _sha256(calibration_source)
        configured_calibration = config.get("instrumentation_calibration_identity")
        if not isinstance(configured_calibration, dict) or configured_calibration.get("sha256") != calibration_sha256:
            raise RuntimeError("instrumentation calibration identity differs from the resolved execution")
        calibration_validation = validate_comparative_calibration(
            artifact,
            requirements=config["calibration_requirements"],
            execution_gpu_uuids=[value["gpu_uuid"] for value in self.process_mappings],
            execution_commit=str(self.git_commit),
        )
        self.calibration_reference = {
            "path": str(calibration_source),
            "sha256": calibration_sha256,
            "artifact": artifact,
            "calibration_allocation_gpu_uuids": calibration_validation[
                "calibration_allocation_gpu_uuids"
            ],
            "execution_allocation_gpu_uuids": calibration_validation[
                "execution_allocation_gpu_uuids"
            ],
            "compatibility_checks": calibration_validation["checks"],
        }
        self.node_sampler = None
        if context.local_rank == 0:
            node_records = [value for value in self.process_mappings if value["node_rank"] == context.node_rank]
            self.node_sampler = NodeNvmlProcessSampler(
                [value["gpu_uuid_raw"] for value in node_records],
                self.attempt_dir / f"node_{context.node_rank}_samples.jsonl",
                node_identity=context.host,
                execution_attempt=self.execution_attempt,
                slurm_allocation_identity=self.allocation_identity,
            )
        self.idle_record = None

    def start(self) -> None:
        self.rank_dir.mkdir(parents=True, exist_ok=True)
        for path in (self.interval_path, self.provisional_path, self.excluded_interval_path):
            path.touch(exist_ok=True)
        if self.context.is_coordinator:
            atomic_write_json(self.attempt_dir / "measurement_config.json", self.measurement)
            atomic_write_json(self.attempt_dir / "calibration_reference.json", self.calibration_reference)
        dist.barrier()
        startup_error = None
        startup_status = None
        if self.node_sampler is not None:
            try:
                self.node_sampler.start()
            except BaseException as error:
                startup_error = error
                if self.node_sampler.process_is_alive:
                    self.node_sampler.abort()
                startup_status = {
                    "rank": self.context.rank,
                    "node_rank": self.context.node_rank,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                }
        startup_statuses = [None for _ in range(self.context.world_size)]
        dist.all_gather_object(startup_statuses, startup_status)
        startup_failure = next((value for value in startup_statuses if value is not None), None)
        if startup_failure is not None:
            if startup_error is not None:
                raise startup_error
            raise RuntimeError(
                "node sampler startup failed on rank "
                f"{startup_failure['rank']}: {startup_failure['error_type']}: {startup_failure['error_message']}"
            )
        if self.context.local_rank == 0:
            start_ns = self.clock.now_ns()
            time.sleep(float(self.measurement["idle_before_seconds"]))
            self.idle_record = {
                "node_rank": self.context.node_rank,
                "node_identity": self.context.host,
                "idle_before": {"start_ns": start_ns, "end_ns": self.clock.now_ns()},
            }
        dist.barrier()
        self._started = True

    def prepare_selected_clients(self, selected: list[str], round_number: int) -> None:
        for client_id in selected:
            key = (round_number, client_id)
            if key in self._static_cache:
                continue
            seed = derive_seed(
                self.config["seed"],
                self.config["seed_streams"]["client_training"],
                round_number,
                client_id,
            )
            features = extract_static_client_features(
                self.bundle.client_dataset(client_id),
                seed,
                int(self.config["federated"]["local_batch_size"]),
                int(self.config["dataset"]["input_features"]),
                int(self.config["federated"]["local_epochs"]),
                bool(self.config["federated"]["drop_last_local_batch"]),
            ).record()
            features.update(
                {
                    "pre_execution_predicted_client_wall_time_seconds": _predict(self.runtime_model, features),
                    "pre_execution_predicted_gross_energy_joules": _predict(self.energy_model, features),
                    "pre_execution_predicted_cuda_event_time_seconds": None,
                    "pre_execution_predicted_idle_adjusted_energy_joules": None,
                    "cuda_event_prediction_available": False,
                    "idle_adjusted_energy_prediction_available": False,
                    "unavailable_prediction_reason": "no compatible frozen Week 5 target artifact",
                    "prediction_role": "frozen_transfer_evaluation",
                    "current_execution_observations_used_for_prediction": False,
                }
            )
            self._static_cache[key] = features

    def assignment_loads(self, assignments, round_number: int) -> dict:
        """Return diagnostic batch and raw-event loads without influencing assignment."""

        batches = {str(rank): 0 for rank in range(self.context.world_size)}
        events = {str(rank): 0 for rank in range(self.context.world_size)}
        for assignment in assignments:
            features = self._static_cache[(round_number, assignment.client_id)]
            rank = str(assignment.process_rank)
            batches[rank] += int(features["local_batch_count"])
            events[rank] += int(features["total_raw_input_events"])
        return {
            "batch_count_by_process_rank": batches,
            "event_count_by_process_rank": events,
            "used_for_assignment": False,
            "feature_scope": "selected-client training indices only",
        }

    def train_assigned_client(
        self,
        model,
        dataset,
        client_id,
        round_number,
        config,
        device,
        training_seed,
        model_payload,
        *,
        selected_position,
        process_rank,
    ):
        if process_rank != self.context.rank:
            raise RuntimeError("measurement hook process rank differs from its assigned rank")
        identity = ClientIntervalIdentity(
            config["dataset"]["name"],
            config["metadata"]["experiment"],
            int(config["seed"]),
            int(round_number),
            int(selected_position),
            str(client_id),
            int(training_seed),
            self.execution_attempt,
            self.gpu_uuid,
        )
        features = self._static_cache[(round_number, client_id)]
        hook = _ComparativeClientHook(self, identity, features, device)
        try:
            result = train_client(
                model,
                dataset,
                client_id,
                round_number,
                config,
                device,
                training_seed,
                model_payload,
                measurement_hook=hook,
            )
            if hook.timing_result is None:
                raise RuntimeError("client timing did not finalize")
            interval = self.intervals.record_client(
                identity,
                hook.timing_result.start_ns,
                hook.timing_result.end_ns,
                hook.data_wait_seconds,
                hook.timing_result.cuda_seconds,
            )
            row = {
                **identity.__dict__,
                **features,
                **hook.observed.record(),
                "interval_id": interval.interval_id,
                "process_rank": self.context.rank,
                "node_rank": self.context.node_rank,
                "model_identity": type(model).__name__,
                "model_configuration_identity": self.model_configuration_identity,
                "model_initialization_id": self.model_initialization_id,
                "dataset_identity": self.bundle.split_artifact.get("dataset_identity"),
                "split_id": self.bundle.split_artifact["split_id"],
                "partition_id": self.bundle.partition.partition_id,
                "git_commit": self.git_commit,
                "parameter_count": sum(value.numel() for value in model.parameters()),
                "sampling_interval_ms": 100,
                "physical_device_count": self.context.physical_device_count,
                "process_count": self.context.world_size,
                "client_processes_per_gpu": 1,
                "cuda_process_service": "none",
                "feature_source_scope": "client_training_indices",
                "validation_indices_in_features": False,
                "official_test_indices_in_features": False,
                "feature_availability": FEATURE_AVAILABILITY,
                "client_wall_time_seconds": interval.wall_seconds,
                "data_wait_time_seconds": interval.data_wait_seconds,
                "cuda_event_time_seconds": interval.cuda_event_seconds,
                "residual_host_time_seconds": interval.residual_host_seconds,
                "peak_allocated_cuda_memory_bytes": result.peak_cuda_memory_bytes,
                "peak_reserved_cuda_memory_bytes": result.peak_cuda_reserved_bytes,
                "reported_spike_rates": result.spike_rates,
                "gross_energy_joules": None,
                "idle_adjusted_energy_joules": None,
                "energy_sample_count": None,
                "energy_coverage_seconds": None,
                "accepted": False,
                "exclusion_reason": "energy_pending",
            }
            append_jsonl(self.provisional_path, row)
            return result
        except BaseException:
            hook.reject_after_finish("client_result_not_accepted")
            raise

    def measure(self, category: str, identity: dict | None = None):
        if not self._started:
            return nullcontext()
        return self.intervals.interval(category, self.execution_attempt, self.gpu_uuid, identity)

    def begin(self, category: str, identity: dict | None = None):
        scope = self.measure(category, identity)
        scope.__enter__()
        token = f"{category}-{len(self._open_scopes) + 1}"
        self._open_scopes.append((scope, token))
        return token

    def end(self, token: str) -> None:
        if not self._open_scopes or self._open_scopes[-1][1] != token:
            raise RuntimeError("measurement interval nesting is incompatible")
        scope, _ = self._open_scopes.pop()
        scope.__exit__(None, None, None)

    def _abort_scopes(self) -> None:
        while self._open_scopes:
            scope, _ = self._open_scopes.pop()
            error = RuntimeError("execution_interrupted")
            try:
                scope.__exit__(type(error), error, error.__traceback__)
            except RuntimeError:
                pass

    def _node_mapping(self) -> dict[str, list[str]]:
        return {
            next(value["host"] for value in self.process_mappings if value["node_rank"] == node_rank): [
                value["gpu_uuid"] for value in self.process_mappings if value["node_rank"] == node_rank
            ]
            for node_rank in range(self.context.node_count)
        }

    @staticmethod
    def _samples_for_interval(samples, gpu_uuid: str, interval: dict) -> list:
        return [
            row
            for row in samples
            if row.gpu_uuid == gpu_uuid
            and interval["start_ns"] <= row.monotonic_timestamp_ns <= interval["end_ns"]
            and row.sampling_error_status is None
            and row.power_watts is not None
        ]

    def _finalize_attempt(
        self,
        execution_completed: bool,
        idle_records: list[dict],
        *,
        node_mapping: dict[str, list[str]] | None = None,
        node_files: dict[str, Path] | None = None,
        allow_missing_idle_after: bool = False,
    ) -> dict:
        node_mapping = self._node_mapping() if node_mapping is None else node_mapping
        if node_files is None:
            node_files = {}
            for value in self.process_mappings:
                node_files.setdefault(
                    value["host"],
                    self.attempt_dir / "node_{}_samples.jsonl".format(value["node_rank"]),
                )
        samples = merge_node_telemetry(
            node_files,
            self.attempt_dir / "merged_device_samples.jsonl",
            expected_uuids_by_node=node_mapping,
            execution_attempt=self.execution_attempt,
            slurm_allocation_identity=self.allocation_identity,
        )
        by_node = {value["node_identity"]: value for value in idle_records}
        baselines = {}
        idle_evidence = {}
        for node_identity, uuids in node_mapping.items():
            record = by_node[node_identity]
            for gpu_uuid in uuids:
                before = self._samples_for_interval(samples, gpu_uuid, record["idle_before"])
                after = (
                    []
                    if record.get("idle_after") is None
                    else self._samples_for_interval(samples, gpu_uuid, record["idle_after"])
                )
                powers = [float(row.power_watts) for row in (*before, *after)]
                if not before or (not after and not allow_missing_idle_after) or not powers:
                    raise RuntimeError(f"idle baseline coverage is incomplete for {gpu_uuid}")
                baselines[gpu_uuid] = statistics.median(powers)
                idle_evidence[gpu_uuid] = {
                    "idle_before": record["idle_before"],
                    "idle_after": record.get("idle_after"),
                    "sample_count": len(powers),
                    "median_power_watts": baselines[gpu_uuid],
                    "interrupted_attempt_missing_idle_after": not after,
                }
        intervals = [
            value
            for rank in range(self.context.world_size)
            for value in read_jsonl(self.attempt_dir / f"rank_{rank}" / "execution_intervals.jsonl")
        ]
        validate_client_interval_nonoverlap(intervals)
        interval_by_id = {value["interval_id"]: value for value in intervals}
        training_intervals = [
            value
            for value in intervals
            if value["category"] == "complete_treatment" and (value["accepted"] or not execution_completed)
        ]
        by_uuid = {}
        for gpu_uuid in node_mapping.values():
            for value in gpu_uuid:
                selected = [item for item in training_intervals if canonical_gpu_uuid(item["gpu_uuid"]) == value]
                if len(selected) != 1:
                    raise RuntimeError(f"complete treatment interval is missing for {value}")
                by_uuid[value] = (int(selected[0]["start_ns"]), int(selected[0]["end_ns"]))
        treatment = integrate_physical_devices(
            samples,
            intervals_by_uuid=by_uuid,
            idle_baseline_watts_by_uuid=baselines,
            maximum_gap_ms=int(self.measurement["maximum_sample_gap_ms"]),
        )
        completed_keys = {
            (int(value["round_number"]), int(value["selected_position"]), str(value["client_id"]))
            for value in read_jsonl(self.run_dir / "client_metrics.jsonl")
        }
        completed_rounds = {value[0] for value in completed_keys}
        client_rows = []
        excluded = []
        for rank in range(self.context.world_size):
            for value in read_jsonl(self.attempt_dir / f"rank_{rank}" / "client_resource_records.jsonl"):
                key = (
                    int(value["communication_round"]),
                    int(value["selected_position"]),
                    str(value["client_id"]),
                )
                if key not in completed_keys:
                    value["exclusion_reason"] = "communication_round_incomplete"
                    excluded.append(value)
                    continue
                interval = interval_by_id[value["interval_id"]]
                gpu_uuid = canonical_gpu_uuid(value["gpu_uuid"])
                estimate = integrate_energy(
                    [row.device_sample() for row in samples if row.gpu_uuid == gpu_uuid],
                    int(interval["start_ns"]),
                    int(interval["end_ns"]),
                    baselines[gpu_uuid],
                    configured_interval_ms=100,
                    maximum_gap_multiplier=int(self.measurement["maximum_sample_gap_ms"]) / 100,
                )
                value.update(
                    {
                        "gross_energy_joules": estimate.gross_energy_joules,
                        "idle_adjusted_energy_joules": estimate.idle_adjusted_energy_joules,
                        "energy_sample_count": estimate.sample_count,
                        "energy_coverage_seconds": estimate.coverage_seconds,
                        "cumulative_energy_crosscheck_joules": estimate.cumulative_energy_crosscheck_joules,
                        "accepted": True,
                        "exclusion_reason": None,
                    }
                )
                client_rows.append(value)
        atomic_write_text(
            self.attempt_dir / "accepted_client_resource_records.jsonl",
            "".join(json.dumps(value, sort_keys=True, allow_nan=False) + "\n" for value in client_rows),
        )
        atomic_write_text(
            self.attempt_dir / "excluded_client_resource_records.jsonl",
            "".join(json.dumps(value, sort_keys=True, allow_nan=False) + "\n" for value in excluded),
        )
        categories = {}
        leaf_categories = {
            "model_distribution",
            "result_collection",
            "aggregation",
            "validation",
            "official_test",
            "checkpoint_writing",
        }
        for category in leaf_categories:
            selected = [
                value
                for value in intervals
                if value["category"] == category
                and value["accepted"]
                and (
                    (value.get("identity") or {}).get("round_number") is None
                    or int(value["identity"]["round_number"]) in completed_rounds
                )
            ]
            grouped = {}
            for value in selected:
                key = json.dumps(value.get("identity"), sort_keys=True)
                grouped.setdefault(key, []).append(value)
            gross = dynamic = 0.0
            for values in grouped.values():
                start_ns = min(int(value["start_ns"]) for value in values)
                end_ns = max(int(value["end_ns"]) for value in values)
                estimate = integrate_physical_devices(
                    samples,
                    intervals_by_uuid={uuid: (start_ns, end_ns) for uuid in baselines},
                    idle_baseline_watts_by_uuid=baselines,
                    maximum_gap_ms=int(self.measurement["maximum_sample_gap_ms"]),
                )
                gross += estimate["gross_energy_joules"]
                dynamic += estimate["idle_adjusted_energy_joules"]
            categories[category] = {"gross_energy_joules": gross, "idle_adjusted_energy_joules": dynamic}
        client_gross = sum(float(value["gross_energy_joules"]) for value in client_rows)
        client_dynamic = sum(float(value["idle_adjusted_energy_joules"]) for value in client_rows)
        accounted_gross = client_gross + sum(value["gross_energy_joules"] for value in categories.values())
        accounted_dynamic = client_dynamic + sum(value["idle_adjusted_energy_joules"] for value in categories.values())
        summary = {
            "schema_version": 1,
            "execution_attempt": self.execution_attempt,
            "execution_completed": execution_completed,
            "slurm_allocation_identity": self.allocation_identity,
            "node_files": [
                str(self.attempt_dir / f"node_{index}_samples.jsonl") for index in range(self.context.node_count)
            ],
            "merged_file": str(self.attempt_dir / "merged_device_samples.jsonl"),
            "device_count": self.context.physical_device_count,
            "node_count": self.context.node_count,
            "idle_baselines": idle_evidence,
            "complete_treatment_energy": treatment,
            "accepted_client_training_energy": {
                "gross_energy_joules": client_gross,
                "idle_adjusted_energy_joules": client_dynamic,
            },
            "phase_energy": categories,
            "model_distribution_energy": categories["model_distribution"],
            "result_collection_energy": categories["result_collection"],
            "aggregation_energy": categories["aggregation"],
            "validation_energy": categories["validation"],
            "official_test_energy": categories["official_test"],
            "checkpoint_energy": categories["checkpoint_writing"],
            "other_measured_execution_energy": {
                "gross_energy_joules": treatment["gross_energy_joules"] - accounted_gross,
                "idle_adjusted_energy_joules": treatment["idle_adjusted_energy_joules"] - accounted_dynamic,
            },
            "idle_baseline_contribution": {
                "energy_joules": treatment["idle_baseline_contribution_joules"],
                "per_device": idle_evidence,
            },
            "unattributed_energy": {
                "gross_energy_joules": treatment["gross_energy_joules"] - accounted_gross,
                "idle_adjusted_energy_joules": treatment["idle_adjusted_energy_joules"] - accounted_dynamic,
            },
            "interrupted_attempt_energy": treatment if not execution_completed else None,
            "accepted_client_count": len(client_rows),
            "excluded_client_count": len(excluded),
            "sampling_error_count": sum(row.sampling_error_status is not None for row in samples),
        }
        tolerance = float(self.measurement["boundary_reconciliation_tolerance_joules"])
        findings = []
        if summary["unattributed_energy"]["gross_energy_joules"] < -tolerance:
            findings.append("attributed gross energy exceeds complete treatment energy")
        if summary["sampling_error_count"]:
            findings.append("sampling errors were observed")
        summary["validation_findings"] = findings
        atomic_write_json(self.attempt_dir / "attempt_energy_summary.json", summary)
        return summary

    def _recover_interrupted_attempts(self) -> list[str]:
        """Finalize prior aborted attempts without mixing their device telemetry."""

        findings = []
        attempts_root = self.run_dir / "measurement_attempts"
        for attempt_dir in sorted(
            attempts_root.glob("attempt_*"),
            key=lambda path: int(path.name.removeprefix("attempt_")),
        ):
            if attempt_dir == self.attempt_dir or (attempt_dir / "attempt_energy_summary.json").is_file():
                continue
            failure_paths = sorted(attempt_dir.glob("rank_*_failure.json"))
            if not failure_paths:
                findings.append(f"interrupted attempt lacks failure evidence: {attempt_dir.name}")
                continue
            original_attempt_dir = self.attempt_dir
            original_execution_attempt = self.execution_attempt
            original_allocation_identity = self.allocation_identity
            try:
                failures = [json.loads(path.read_text(encoding="utf-8")) for path in failure_paths]
                ranks = {int(value["rank"]) for value in failures}
                world_sizes = {int(value["world_size"]) for value in failures}
                if len(failures) != self.context.world_size or ranks != set(range(self.context.world_size)):
                    raise RuntimeError("per-rank failure evidence is incomplete")
                if world_sizes != {self.context.world_size}:
                    raise RuntimeError("failure evidence world size differs")
                attempts = {int(value["execution_attempt"]) for value in failures}
                allocations = {str(value["slurm_allocation_identity"]) for value in failures}
                if len(attempts) != 1 or len(allocations) != 1:
                    raise RuntimeError("failure evidence identities disagree")
                execution_attempt = attempts.pop()
                if execution_attempt != int(attempt_dir.name.removeprefix("attempt_")):
                    raise RuntimeError("failure evidence attempt differs from its directory")
                treatment_intervals = [value.get("complete_treatment_interval") for value in failures]
                if any(
                    not isinstance(interval, dict)
                    or interval.get("category") != "complete_treatment"
                    or interval.get("accepted") is not False
                    or int(interval["end_ns"]) < int(interval["start_ns"])
                    for interval in treatment_intervals
                ):
                    raise RuntimeError("complete-treatment interruption evidence is missing")
                node_mapping: dict[str, list[str]] = {}
                node_ranks: dict[str, int] = {}
                idle_records = []
                for value in failures:
                    host = str(value["host"])
                    node_rank = int(value["node_rank"])
                    if host in node_ranks and node_ranks[host] != node_rank:
                        raise RuntimeError("node identity maps to multiple node ranks")
                    node_ranks[host] = node_rank
                    node_mapping.setdefault(host, []).append(canonical_gpu_uuid(value["gpu_uuid"]))
                    if int(value["local_rank"]) == 0:
                        idle_record = value.get("idle_record")
                        if not isinstance(idle_record, dict):
                            raise RuntimeError("node-leader idle evidence is missing")
                        idle_records.append(idle_record)
                if len(node_mapping) != self.context.node_count:
                    raise RuntimeError("interrupted attempt node coverage differs")
                if set(node_ranks.values()) != set(range(self.context.node_count)):
                    raise RuntimeError("interrupted attempt node ranks differ")
                if sum(len(values) for values in node_mapping.values()) != self.context.physical_device_count:
                    raise RuntimeError("interrupted attempt device coverage differs")
                for values in node_mapping.values():
                    if len(values) != len(set(values)):
                        raise RuntimeError("interrupted attempt maps multiple ranks to one device")
                    values.sort()
                node_files = {
                    host: attempt_dir / f"node_{node_rank}_samples.jsonl" for host, node_rank in node_ranks.items()
                }
                self.attempt_dir = attempt_dir
                self.execution_attempt = execution_attempt
                self.allocation_identity = allocations.pop()
                summary = self._finalize_attempt(
                    False,
                    idle_records,
                    node_mapping=node_mapping,
                    node_files=node_files,
                    allow_missing_idle_after=True,
                )
                sampler_shutdown_errors = [
                    value["sampler_shutdown_error"]
                    for value in failures
                    if value.get("sampler_shutdown_error") is not None
                ]
                if sampler_shutdown_errors:
                    summary["interrupted_sampler_shutdown_errors"] = sampler_shutdown_errors
                    summary["validation_findings"].append("an interrupted attempt sampler did not shut down cleanly")
                    atomic_write_json(attempt_dir / "attempt_energy_summary.json", summary)
            except BaseException as error:
                findings.append(
                    f"interrupted attempt {attempt_dir.name} recovery failed: {type(error).__name__}: {error}"
                )
            finally:
                self.attempt_dir = original_attempt_dir
                self.execution_attempt = original_execution_attempt
                self.allocation_identity = original_allocation_identity
        return findings

    def _aggregate_attempts(self, execution_completed: bool, current: dict) -> dict:
        recovery_findings = self._recover_interrupted_attempts()
        summaries = []
        rows = []
        for attempt_dir in sorted(
            (self.run_dir / "measurement_attempts").glob("attempt_*"),
            key=lambda path: int(path.name.removeprefix("attempt_")),
        ):
            summary_path = attempt_dir / "attempt_energy_summary.json"
            if summary_path.is_file():
                summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
                rows.extend(read_jsonl(attempt_dir / "accepted_client_resource_records.jsonl"))
        latest = {}
        for value in rows:
            key = (
                value["dataset"],
                int(value["scientific_seed"]),
                int(value["communication_round"]),
                int(value["selected_position"]),
            )
            if key not in latest or int(value["execution_attempt"]) > int(latest[key]["execution_attempt"]):
                latest[key] = value
        accepted_rows = sorted(
            latest.values(),
            key=lambda value: (
                value["dataset"],
                int(value["scientific_seed"]),
                int(value["communication_round"]),
                int(value["selected_position"]),
            ),
        )
        expected = int(self.config["federated"]["rounds"]) * int(self.config["federated"]["clients_per_round"])
        findings = recovery_findings + [
            value for summary in summaries for value in summary.get("validation_findings", [])
        ]
        if execution_completed and len(accepted_rows) != expected:
            findings.append(f"accepted client record count differs: expected {expected}, observed {len(accepted_rows)}")
        official_access = int(getattr(self.bundle, "official_test_access_count", 0))
        if execution_completed and official_access != 1:
            findings.append("official test access count is not one")
        interrupted_gross = sum(
            summary["complete_treatment_energy"]["gross_energy_joules"]
            for summary in summaries
            if not summary["execution_completed"]
        )
        interrupted_dynamic = sum(
            summary["complete_treatment_energy"]["idle_adjusted_energy_joules"]
            for summary in summaries
            if not summary["execution_completed"]
        )
        completed = [value for value in summaries if value["execution_completed"]]
        complete_attempt = completed[-1] if completed else current
        total_treatment_energy = {
            "gross_energy_joules": sum(
                value["complete_treatment_energy"]["gross_energy_joules"] for value in summaries
            ),
            "idle_adjusted_energy_joules": sum(
                value["complete_treatment_energy"]["idle_adjusted_energy_joules"] for value in summaries
            ),
            "idle_baseline_contribution_joules": sum(
                value["complete_treatment_energy"]["idle_baseline_contribution_joules"] for value in summaries
            ),
            "attempt_count": len(summaries),
            "integration_scope": "sum of each separately validated physical-device attempt total",
        }
        phase_energy = {
            category: {
                field: sum(value["phase_energy"][category][field] for value in summaries)
                for field in ("gross_energy_joules", "idle_adjusted_energy_joules")
            }
            for category in complete_attempt["phase_energy"]
        }
        other_energy = {
            field: sum(value["other_measured_execution_energy"][field] for value in summaries)
            for field in ("gross_energy_joules", "idle_adjusted_energy_joules")
        }
        unattributed_energy = {
            field: sum(value["unattributed_energy"][field] for value in summaries)
            for field in ("gross_energy_joules", "idle_adjusted_energy_joules")
        }
        energy_summary = {
            "schema_version": 1,
            "execution_completed": execution_completed,
            "attempt_count": len(summaries),
            "accepted_client_record_count": len(accepted_rows),
            "expected_client_record_count": expected,
            "complete_treatment_energy": total_treatment_energy,
            "successful_completion_attempt_energy": complete_attempt["complete_treatment_energy"],
            "interrupted_attempt_energy": {
                "gross_energy_joules": interrupted_gross,
                "idle_adjusted_energy_joules": interrupted_dynamic,
            },
            "accepted_client_training_energy": {
                "gross_energy_joules": sum(float(value["gross_energy_joules"]) for value in accepted_rows),
                "idle_adjusted_energy_joules": sum(
                    float(value["idle_adjusted_energy_joules"]) for value in accepted_rows
                ),
            },
            "phase_energy": phase_energy,
            "model_distribution_energy": phase_energy["model_distribution"],
            "result_collection_energy": phase_energy["result_collection"],
            "aggregation_energy": phase_energy["aggregation"],
            "validation_energy": phase_energy["validation"],
            "official_test_energy": phase_energy["official_test"],
            "checkpoint_energy": phase_energy["checkpoint_writing"],
            "other_measured_execution_energy": other_energy,
            "idle_baseline_contribution": {
                "energy_joules": total_treatment_energy["idle_baseline_contribution_joules"],
                "scope": "sum across separately validated attempts",
            },
            "unattributed_energy": unattributed_energy,
            "official_test_access_count": official_access,
            "attempts": summaries,
            "validation_findings": findings,
        }
        accepted = execution_completed and not findings
        acceptance = {
            "schema_version": 1,
            "execution_completion": execution_completed,
            "measurement_completeness": accepted,
            "energy_completeness": accepted,
            "scientific_hypothesis_outcome": self.config["comparative_evaluation"]["evidence_complete_outcome"]
            if accepted
            else "not_classified",
            "accepted": accepted,
            "validation_findings": findings,
            "accepted_client_record_count": len(accepted_rows),
            "expected_client_record_count": expected,
        }
        atomic_write_text(
            self.run_dir / "client_resource_records.jsonl",
            "".join(json.dumps(value, sort_keys=True, allow_nan=False) + "\n" for value in accepted_rows),
        )
        atomic_write_json(self.run_dir / "energy_summary.json", energy_summary)
        atomic_write_json(self.run_dir / "measurement_acceptance.json", acceptance)
        return acceptance

    def stop(self, execution_completed: bool) -> dict:
        if not self._started:
            raise RuntimeError("comparative measurement session was not started")
        self._abort_scopes()
        dist.barrier()
        sampler_error = None
        sampler_status = None
        if self.context.local_rank == 0:
            try:
                start_ns = self.clock.now_ns()
                time.sleep(float(self.measurement["idle_after_seconds"]))
                self.idle_record["idle_after"] = {"start_ns": start_ns, "end_ns": self.clock.now_ns()}
                self.node_sampler.stop()
            except BaseException as error:
                sampler_error = error
                if self.node_sampler.process_is_alive:
                    self.node_sampler.abort()
                sampler_status = {
                    "rank": self.context.rank,
                    "node_rank": self.context.node_rank,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                }
        sampler_statuses = [None for _ in range(self.context.world_size)]
        dist.all_gather_object(sampler_statuses, sampler_status)
        sampler_failure = next((value for value in sampler_statuses if value is not None), None)
        if sampler_failure is not None:
            if sampler_error is not None:
                raise sampler_error
            raise RuntimeError(
                "node sampler shutdown failed on rank "
                f"{sampler_failure['rank']}: {sampler_failure['error_type']}: {sampler_failure['error_message']}"
            )
        gathered_idle = [None for _ in range(self.context.world_size)]
        dist.all_gather_object(gathered_idle, self.idle_record)
        finalization_error = None
        finalization_status = {"acceptance": None, "error": None}
        if self.context.is_coordinator:
            try:
                idle_records = [value for value in gathered_idle if value is not None]
                current = self._finalize_attempt(execution_completed, idle_records)
                finalization_status["acceptance"] = self._aggregate_attempts(execution_completed, current)
            except BaseException as error:
                finalization_error = error
                finalization_status["error"] = {
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                }
        values = [finalization_status if self.context.is_coordinator else None]
        dist.broadcast_object_list(values, src=0, device=self.context.control_device)
        if values[0]["error"] is not None:
            if finalization_error is not None:
                raise finalization_error
            failure = values[0]["error"]
            raise RuntimeError(f"measurement finalization failed: {failure['error_type']}: {failure['error_message']}")
        self._started = False
        return values[0]["acceptance"]

    def abort(self, error: BaseException) -> None:
        """Flush local failure evidence and stop the node sampler without collectives."""

        self._abort_scopes()
        sampler_shutdown_error = None
        if self.node_sampler is not None:
            try:
                if self.node_sampler.process_is_alive:
                    self.node_sampler.stop()
                else:
                    self.node_sampler.abort()
            except BaseException as sampler_error:
                sampler_shutdown_error = {
                    "error_type": type(sampler_error).__name__,
                    "error_message": str(sampler_error),
                }
                self.node_sampler.abort()
        complete_treatment_interval = next(
            (value for value in reversed(read_jsonl(self.interval_path)) if value["category"] == "complete_treatment"),
            None,
        )
        atomic_write_json(
            self.attempt_dir / f"rank_{self.context.rank}_failure.json",
            {
                "schema_version": 1,
                "execution_attempt": self.execution_attempt,
                "rank": self.context.rank,
                "node_rank": self.context.node_rank,
                "local_rank": self.context.local_rank,
                "world_size": self.context.world_size,
                "host": self.context.host,
                "gpu_uuid": self.gpu_uuid,
                "slurm_allocation_identity": self.allocation_identity,
                "idle_record": self.idle_record,
                "complete_treatment_interval": complete_treatment_interval,
                "sampler_shutdown_error": sampler_shutdown_error,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "execution_completed": False,
            },
        )
        self._started = False
