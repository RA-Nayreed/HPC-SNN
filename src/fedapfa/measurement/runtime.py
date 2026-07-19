"""Lifecycle integration for one-process resource measurement runs."""

from __future__ import annotations

import hashlib
import json
import os
import statistics
import time
from contextlib import nullcontext
from pathlib import Path

import torch

from fedapfa.federated.checkpointing import state_identity
from fedapfa.federated.client import train_client
from fedapfa.federated.randomness import derive_seed
from fedapfa.utilities.git_metadata import git_metadata
from fedapfa.utilities.serialization import atomic_write_json, atomic_write_text

from .client_interval import ClientIntervalIdentity, IntervalRecorder
from .clock import CudaTimingAdapter, SystemMonotonicClock
from .energy import integrate_energy
from .features import FEATURE_AVAILABILITY, ObservedClientWork, extract_static_client_features
from .power import DeviceSample, NvmlAdapter, PowerSampler
from .records import append_jsonl, read_jsonl, resource_row_key, write_measurement_acceptance


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cuda_uuid(device: torch.device) -> str:
    declared = os.environ.get("FEDAPFA_GPU_UUID")
    if declared:
        if not declared.startswith("GPU-"):
            raise RuntimeError("FEDAPFA_GPU_UUID is invalid")
        return declared
    properties = torch.cuda.get_device_properties(device)
    value = getattr(properties, "uuid", None)
    if value is None:
        raise RuntimeError("the CUDA device UUID is unavailable")
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    value = str(value)
    return value if value.startswith("GPU-") else f"GPU-{value}"


def _device_samples(path: Path) -> list[DeviceSample]:
    return [DeviceSample(**value) for value in read_jsonl(path)]


def _next_execution_attempt(run_dir: Path, idle_attempts: list[dict]) -> int:
    observed = {
        int(value["execution_attempt"])
        for value in idle_attempts
        if value.get("execution_attempt") is not None
    }
    for name in (
        "device_samples.jsonl",
        "execution_intervals.jsonl",
        "client_resource_records.jsonl",
    ):
        for value in read_jsonl(run_dir / name):
            if value.get("execution_attempt") is not None:
                observed.add(int(value["execution_attempt"]))
    return max(observed or {0}) + 1


class ClientMeasurementHook:
    def __init__(
        self,
        session: ResourceMeasurementSession,
        identity: ClientIntervalIdentity,
        static_features: dict,
        device: torch.device,
    ) -> None:
        self.session = session
        self.identity = identity
        self.static_features = static_features
        self.timing = session.timing_factory(device)
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

    def abort_if_open(self) -> None:
        if not self._open or self._excluded:
            return
        ended = self.session.clock.now_ns()
        started = getattr(self.timing, "_start_ns", None)
        if started is not None:
            record = self.session.intervals.record_client(
                self.identity,
                int(started),
                ended,
                0.0,
                0.0,
                accepted=False,
                exclusion_reason="client_training_interrupted",
            )
            append_jsonl(
                self.session.run_dir / "excluded_intervals.jsonl",
                {"interval_id": record.interval_id, "reason": "client_training_interrupted"},
            )
        self._excluded = True
        self._open = False

    def reject_after_finish(self, reason: str) -> None:
        if self._excluded or self.timing_result is None:
            return
        record = self.session.intervals.record_client(
            self.identity,
            self.timing_result.start_ns,
            self.timing_result.end_ns,
            self.data_wait_seconds,
            self.timing_result.cuda_seconds,
            accepted=False,
            exclusion_reason=reason,
        )
        append_jsonl(
            self.session.run_dir / "excluded_intervals.jsonl",
            {"interval_id": record.interval_id, "reason": reason},
        )
        self._excluded = True


class ResourceMeasurementSession:
    """Own one attempt's sampler and preserve accepted work across resumption."""

    def __init__(
        self,
        config: dict,
        run_dir: str | Path,
        bundle,
        model,
        context,
        calibration_path: str | Path,
        *,
        adapter=None,
        timing_factory=None,
        sleeper=time.sleep,
        clock=None,
    ) -> None:
        measurement = config["resource_measurement"]
        if not context.is_coordinator or context.world_size != 1 or context.visible_device_count != 1:
            raise RuntimeError("resource measurement is permitted only on rank zero with one process and one device")
        if context.device.type != "cuda":
            raise RuntimeError("scientific resource measurement requires CUDA")
        if any("MPS" in name and value for name, value in os.environ.items()):
            raise RuntimeError("CUDA MPS environment settings are incompatible with this collection")
        self.config = config
        self.run_dir = Path(run_dir)
        self.bundle = bundle
        self.model = model
        self.context = context
        self.measurement = measurement
        self.clock = clock or SystemMonotonicClock()
        self.sleeper = sleeper
        self.gpu_uuid = adapter.uuid if adapter is not None else _cuda_uuid(context.device)
        calibration_source = Path(calibration_path).resolve()
        if not calibration_source.is_file():
            raise FileNotFoundError("passing calibration artifact is required")
        calibration = json.loads(calibration_source.read_text(encoding="utf-8"))
        if not calibration.get("passed"):
            raise RuntimeError("resource measurement calibration did not pass")
        if int(calibration.get("sampling_interval_ms", measurement["sampling_interval_ms"])) != int(
            measurement["sampling_interval_ms"]
        ):
            raise RuntimeError("calibration sampling interval differs from the scientific configuration")
        idle_record = self.run_dir / "idle_power.json"
        previous_attempts = []
        if idle_record.is_file():
            previous_attempts = json.loads(idle_record.read_text(encoding="utf-8")).get("attempts", [])
        self.execution_attempt = _next_execution_attempt(self.run_dir, previous_attempts)
        self.idle_attempts = list(previous_attempts)
        self.intervals = IntervalRecorder(self.run_dir / "execution_intervals.jsonl", self.clock)
        self.timing_factory = timing_factory or (lambda device: CudaTimingAdapter(device, self.clock))
        nvml_adapter = adapter or NvmlAdapter(self.gpu_uuid, context.visible_device_count)
        self.sampler = PowerSampler(
            nvml_adapter,
            self.run_dir / "device_samples.jsonl",
            int(measurement["sampling_interval_ms"]),
            self.execution_attempt,
        )
        self._started = False
        self._open_scopes: list[tuple[object, str]] = []
        self._static_cache: dict[tuple[int, str], dict] = {}
        hidden = config["model"]["hidden_dims"]
        self.layer_widths = {"layer1": int(hidden[0]), "layer2": int(hidden[1])}
        self.model_initialization_id = state_identity(model.state_dict())
        self.model_configuration_identity = hashlib.sha256(
            json.dumps(config["model"], sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        self.git_commit = git_metadata().get("commit")
        calibration_entry = {
            "execution_attempt": self.execution_attempt,
            "path": str(calibration_source),
            "sha256": _sha256_file(calibration_source),
            "artifact": calibration,
        }
        calibration_record = self.run_dir / "calibration_reference.json"
        prior_calibrations = []
        if calibration_record.is_file():
            stored = json.loads(calibration_record.read_text(encoding="utf-8"))
            prior_calibrations = list(stored.get("attempts", []))
        if any(
            int(value["execution_attempt"]) == self.execution_attempt
            for value in prior_calibrations
        ):
            raise RuntimeError("calibration attempt identity is duplicated")
        self.calibration_reference = {
            "schema_version": 1,
            "attempts": [*prior_calibrations, calibration_entry],
        }

    def start(self) -> None:
        atomic_write_json(self.run_dir / "measurement_config.json", self.measurement)
        atomic_write_json(self.run_dir / "calibration_reference.json", self.calibration_reference)
        for name in (
            "device_samples.jsonl",
            "execution_intervals.jsonl",
            "client_resource_records.jsonl",
            "excluded_intervals.jsonl",
        ):
            (self.run_dir / name).touch(exist_ok=True)
        self.sampler.start()
        self._started = True
        self._record_idle("idle_before", float(self.measurement["idle_before_seconds"]))

    def _samples_for(self, start_ns: int, end_ns: int, attempt: int) -> list[DeviceSample]:
        return [
            value
            for value in _device_samples(self.run_dir / "device_samples.jsonl")
            if value.execution_attempt == attempt
            and start_ns <= value.monotonic_timestamp_ns <= end_ns
            and value.sampling_error_status is None
        ]

    def _record_idle(self, category: str, seconds: float) -> None:
        start_ns = self.clock.now_ns()
        with self.intervals.interval(category, self.execution_attempt, self.gpu_uuid):
            self.sleeper(seconds)
        end_ns = self.clock.now_ns()
        samples = self._samples_for(start_ns, end_ns, self.execution_attempt)
        powers = [float(value.power_watts) for value in samples if value.power_watts is not None]
        temperatures = [
            float(value.temperature_celsius) for value in samples if value.temperature_celsius is not None
        ]
        if not powers:
            raise RuntimeError("idle interval contains no accepted power samples")
        existing = next(
            (value for value in self.idle_attempts if value["execution_attempt"] == self.execution_attempt),
            None,
        )
        if existing is None:
            existing = {"execution_attempt": self.execution_attempt, "gpu_uuid": self.gpu_uuid}
            self.idle_attempts.append(existing)
        existing[category] = {
            "start_ns": start_ns,
            "end_ns": end_ns,
            "sample_count": len(powers),
            "median_power_watts": statistics.median(powers),
            "median_temperature_celsius": statistics.median(temperatures) if temperatures else None,
        }
        sources = [
            name for name in ("idle_before", "idle_after") if name in existing
        ]
        source_intervals = [existing[name] for name in sources]
        existing["combined_median_power_watts"] = statistics.median(
            [
                float(value.power_watts)
                for value in _device_samples(self.run_dir / "device_samples.jsonl")
                if value.execution_attempt == self.execution_attempt
                and any(
                    interval["start_ns"]
                    <= value.monotonic_timestamp_ns
                    <= interval["end_ns"]
                    for interval in source_intervals
                )
                and value.sampling_error_status is None
                and value.power_watts is not None
            ]
        )
        existing["baseline_sources"] = sources
        if "idle_before" in existing and "idle_after" in existing:
            before = existing["idle_before"]
            after = existing["idle_after"]
            start_temperature = before["median_temperature_celsius"]
            end_temperature = after["median_temperature_celsius"]
            existing["temperature_drift_celsius"] = (
                None
                if start_temperature is None or end_temperature is None
                else end_temperature - start_temperature
            )
        atomic_write_json(
            self.run_dir / "idle_power.json",
            {"schema_version": 1, "attempts": sorted(self.idle_attempts, key=lambda value: value["execution_attempt"])},
        )

    def prepare_selected_clients(self, selected: list[str], round_number: int) -> None:
        completed_records = read_jsonl(self.run_dir / "client_metrics.jsonl")
        for client_id in selected:
            training_seed = derive_seed(
                self.config["seed"],
                self.config["seed_streams"]["client_training"],
                round_number,
                client_id,
            )
            dataset = self.bundle.client_dataset(client_id)
            features = extract_static_client_features(
                dataset,
                training_seed,
                int(self.config["federated"]["local_batch_size"]),
                int(self.config["dataset"]["input_features"]),
                int(self.config["federated"]["local_epochs"]),
                bool(self.config["federated"]["drop_last_local_batch"]),
            ).record()
            history_count = sum(
                int(value["round_number"]) < int(round_number)
                and str(value["client_id"]) == str(client_id)
                for value in completed_records
            )
            features["has_historical_observations"] = history_count > 0
            features["historical_observation_count"] = history_count
            self._static_cache[(round_number, client_id)] = features

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
        if process_rank != 0:
            raise RuntimeError("resource-measured clients must execute on rank zero")
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
        static_features = self._static_cache[(round_number, client_id)]
        hook = ClientMeasurementHook(self, identity, static_features, device)
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
            wall = interval.wall_seconds
            provisional = {
                **identity.__dict__,
                **static_features,
                **hook.observed.record(),
                "interval_id": interval.interval_id,
                "model_identity": type(model).__name__,
                "model_configuration_identity": self.model_configuration_identity,
                "model_initialization_id": self.model_initialization_id,
                "dataset_identity": self.bundle.split_artifact.get("dataset_identity"),
                "split_id": self.bundle.split_artifact["split_id"],
                "partition_id": self.bundle.partition.partition_id,
                "git_commit": self.git_commit,
                "parameter_count": sum(value.numel() for value in model.parameters()),
                "sampling_interval_ms": int(self.measurement["sampling_interval_ms"]),
                "physical_device_count": 1,
                "process_count": 1,
                "client_processes_per_gpu": 1,
                "cuda_process_service": "none",
                "feature_source_scope": "client_training_indices",
                "validation_indices_in_features": False,
                "official_test_indices_in_features": False,
                "feature_availability": FEATURE_AVAILABILITY,
                "client_wall_time_seconds": wall,
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
            append_jsonl(self.run_dir / "client_resource_records.jsonl", provisional)
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

    def _abort_open_scopes(self) -> None:
        while self._open_scopes:
            scope, _ = self._open_scopes.pop()
            error = RuntimeError("execution_interrupted")
            try:
                scope.__exit__(type(error), error, error.__traceback__)
            except RuntimeError:
                pass

    def stop(self, execution_completed: bool) -> dict:
        if not self._started:
            raise RuntimeError("resource measurement session was not started")
        idle_error = None
        self._abort_open_scopes()
        try:
            if execution_completed:
                self._record_idle("idle_after", float(self.measurement["idle_after_seconds"]))
            else:
                idle_error = "execution ended before the post-execution idle interval"
        except BaseException as error:
            idle_error = f"{type(error).__name__}: {error}"
        finally:
            self.sampler.stop()
            self._started = False
        return self._finalize(execution_completed, idle_error)

    def _finalize(self, execution_completed: bool, idle_error: str | None) -> dict:
        findings = [] if idle_error is None else [f"idle_after_failed: {idle_error}"]
        intervals = {value["interval_id"]: value for value in read_jsonl(self.run_dir / "execution_intervals.jsonl")}
        samples = _device_samples(self.run_dir / "device_samples.jsonl")
        baseline_by_attempt = {
            int(value["execution_attempt"]): float(value["combined_median_power_watts"])
            for value in self.idle_attempts
            if "combined_median_power_watts" in value
        }
        scientific_records = read_jsonl(self.run_dir / "client_metrics.jsonl")
        completed_keys = {
            (int(value["round_number"]), int(value["selected_position"]), str(value["client_id"]))
            for value in scientific_records
        }
        provisional = read_jsonl(self.run_dir / "client_resource_records.jsonl")
        by_interval = {value["interval_id"]: value for value in provisional}
        candidate_rows = []
        for value in by_interval.values():
            key = (
                int(value["communication_round"]),
                int(value["selected_position"]),
                str(value["client_id"]),
            )
            if key not in completed_keys:
                append_jsonl(
                    self.run_dir / "excluded_intervals.jsonl",
                    {"interval_id": value["interval_id"], "reason": "communication_round_incomplete"},
                )
                continue
            candidate_rows.append(value)
        latest = {}
        for value in candidate_rows:
            key = resource_row_key(value)
            if key not in latest or int(value["execution_attempt"]) > int(latest[key]["execution_attempt"]):
                latest[key] = value
        accepted_rows = []
        for value in sorted(latest.values(), key=resource_row_key):
            interval = intervals[value["interval_id"]]
            attempt = int(value["execution_attempt"])
            baseline = baseline_by_attempt.get(attempt)
            if baseline is None:
                append_jsonl(
                    self.run_dir / "excluded_intervals.jsonl",
                    {"interval_id": value["interval_id"], "reason": "idle_baseline_incomplete"},
                )
                continue
            attempt_samples = [item for item in samples if item.execution_attempt == attempt]
            try:
                energy = integrate_energy(
                    attempt_samples,
                    int(interval["start_ns"]),
                    int(interval["end_ns"]),
                    baseline,
                    int(self.measurement["sampling_interval_ms"]),
                    float(self.measurement["maximum_sample_gap_multiplier"]),
                )
            except ValueError as error:
                append_jsonl(
                    self.run_dir / "excluded_intervals.jsonl",
                    {"interval_id": value["interval_id"], "reason": f"energy_incomplete: {error}"},
                )
                findings.append(f"{value['interval_id']}: {error}")
                continue
            value.update(
                {
                    "gross_energy_joules": energy.gross_energy_joules,
                    "idle_adjusted_energy_joules": energy.idle_adjusted_energy_joules,
                    "energy_sample_count": energy.sample_count,
                    "energy_coverage_seconds": energy.coverage_seconds,
                    "cumulative_energy_crosscheck_joules": energy.cumulative_energy_crosscheck_joules,
                    "accepted": True,
                    "exclusion_reason": None,
                }
            )
            accepted_rows.append(value)
        text = "".join(json.dumps(value, sort_keys=True, allow_nan=False) + "\n" for value in accepted_rows)
        atomic_write_text(self.run_dir / "client_resource_records.jsonl", text)
        expected = int(self.config["federated"]["rounds"]) * int(
            self.config["federated"]["clients_per_round"]
        )
        if len(scientific_records) != expected:
            findings.append(
                f"scientific_client_record_count: expected {expected}, observed {len(scientific_records)}"
            )
        if execution_completed and int(getattr(self.bundle, "official_test_access_count", 0)) != 1:
            findings.append("official_test_access_count_is_not_one")
        sampling_error_count = sum(value.sampling_error_status is not None for value in samples)
        if sampling_error_count:
            findings.append(f"sampling_errors_observed: {sampling_error_count}")
        measurement_complete = len(accepted_rows) == expected and not findings
        energy_complete = measurement_complete and all(
            value["gross_energy_joules"] is not None for value in accepted_rows
        )
        acceptance = write_measurement_acceptance(
            self.run_dir / "measurement_acceptance.json",
            execution_completion=execution_completed,
            measurement_completeness=measurement_complete,
            energy_completeness=energy_complete,
            scientific_hypothesis_outcome="not_evaluated",
            findings=findings,
        )
        acceptance["accepted_client_record_count"] = len(accepted_rows)
        acceptance["expected_client_record_count"] = expected
        acceptance["sampling_error_count"] = sampling_error_count
        reconciliation = self._energy_reconciliation(intervals, samples, accepted_rows, baseline_by_attempt)
        acceptance["energy_reconciliation"] = reconciliation
        if reconciliation.get("validation_findings"):
            acceptance["validation_findings"].extend(reconciliation["validation_findings"])
            acceptance["accepted"] = False
            acceptance["energy_completeness"] = False
        atomic_write_json(self.run_dir / "measurement_acceptance.json", acceptance)
        return acceptance

    def _energy_reconciliation(
        self,
        intervals: dict[str, dict],
        samples: list[DeviceSample],
        accepted_rows: list[dict],
        baseline_by_attempt: dict[int, float],
    ) -> dict:
        findings = []
        totals = []
        category_energy: dict[str, float] = {}
        for interval in intervals.values():
            if interval["category"] != "training_execution" and not interval["accepted"]:
                continue
            if interval["category"] == "communication_round":
                continue
            attempt = int(interval["execution_attempt"])
            if attempt not in baseline_by_attempt:
                continue
            try:
                estimate = integrate_energy(
                    [value for value in samples if value.execution_attempt == attempt],
                    int(interval["start_ns"]),
                    int(interval["end_ns"]),
                    baseline_by_attempt[attempt],
                    int(self.measurement["sampling_interval_ms"]),
                    float(self.measurement["maximum_sample_gap_multiplier"]),
                )
            except ValueError as error:
                findings.append(f"{interval['interval_id']}: {error}")
                continue
            category = interval["category"]
            category_energy[category] = category_energy.get(category, 0.0) + estimate.gross_energy_joules
            if category == "training_execution":
                totals.append((interval, estimate))
        total_energy = sum(value.gross_energy_joules for _, value in totals)
        total_seconds = sum(value.coverage_seconds for _, value in totals)
        client_energy = sum(float(value["gross_energy_joules"]) for value in accepted_rows)
        aggregation = category_energy.get("aggregation", 0.0)
        validation = category_energy.get("validation", 0.0)
        checkpoint = category_energy.get("checkpoint_writing", 0.0)
        other = category_energy.get("model_distribution", 0.0) + category_energy.get(
            "result_collection", 0.0
        )
        unattributed = total_energy - (client_energy + aggregation + validation + checkpoint + other)
        tolerance = float(self.measurement["boundary_reconciliation_tolerance_joules"])
        if unattributed < -tolerance:
            findings.append(
                f"accounted interval energy exceeds execution energy by {-unattributed:.12g} joules"
            )
        idle_contribution = sum(
            baseline_by_attempt[int(interval["execution_attempt"])] * estimate.coverage_seconds
            for interval, estimate in totals
        )
        return {
            "client_training_energy_joules": client_energy,
            "aggregation_energy_joules": aggregation,
            "validation_energy_joules": validation,
            "checkpoint_energy_joules": checkpoint,
            "other_measured_energy_joules": other,
            "excluded_execution_energy_joules": sum(
                estimate.gross_energy_joules for interval, estimate in totals if not interval["accepted"]
            ),
            "idle_baseline_contribution_joules": idle_contribution,
            "complete_training_execution_energy_joules": total_energy,
            "complete_training_execution_seconds": total_seconds,
            "unattributed_energy_joules": unattributed,
            "boundary_tolerance_joules": tolerance,
            "validation_findings": findings,
        }
