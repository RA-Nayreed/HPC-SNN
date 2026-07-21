import copy
import hashlib
import json
import random

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import TensorDataset

from analysis.resource_measurement import write_resource_figures
from fedapfa.cost_estimation.artifacts import fit_client_cost_models
from fedapfa.cost_estimation.dataset import build_client_cost_dataset
from fedapfa.federated.client import train_client
from fedapfa.measurement.clock import CpuTimingAdapter
from fedapfa.measurement.features import ObservedClientWork
from fedapfa.utilities.serialization import atomic_write_json, canonical_json


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 2)

    def forward(self, inputs, generator=None):
        logits = self.linear(inputs)
        rate = torch.sigmoid(logits).mean()
        return logits, {"layer1": rate, "layer2": rate / 2}


class MeasurementHook:
    def __init__(self):
        self.timing = CpuTimingAdapter()
        self.observed = ObservedClientWork()
        self.result = None
        self.data_wait = None
        self.open = False

    def start(self):
        self.timing.start()
        self.open = True

    def begin_device_work(self):
        return self.timing.begin_device_work()

    def end_device_work(self, token):
        self.timing.end_device_work(token)

    def observe_batch(self, batch, rates):
        self.observed.observe(batch, rates, {"layer1": 2, "layer2": 2})

    def finish(self, data_wait):
        self.result = self.timing.finish()
        self.data_wait = data_wait
        self.open = False

    def abort_if_open(self):
        self.open = False


def _client_config():
    return {
        "federated": {
            "local_batch_size": 4,
            "data_loader_workers": 0,
            "persistent_workers": False,
            "drop_last_local_batch": False,
            "optimizer": "adam",
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "local_epochs": 1,
            "record_extended_diagnostics": False,
        }
    }


def _rng_snapshot():
    return random.getstate(), np.random.get_state(), torch.get_rng_state().clone()


def _assert_rng_equal(left, right):
    assert left[0] == right[0]
    assert left[1][0] == right[1][0]
    assert np.array_equal(left[1][1], right[1][1])
    assert left[1][2:] == right[1][2:]
    assert torch.equal(left[2], right[2])


def test_shared_client_path_preserves_update_and_random_state_with_measurement():
    generator = torch.Generator().manual_seed(23)
    dataset = TensorDataset(torch.rand((12, 3), generator=generator), torch.arange(12) % 2)
    torch.manual_seed(31)
    model = Classifier()
    initial = copy.deepcopy(model.state_dict())
    config = _client_config()

    random.seed(41)
    np.random.seed(41)
    torch.manual_seed(41)
    before = _rng_snapshot()
    unmeasured = train_client(model, dataset, "client_00", 1, config, torch.device("cpu"), 71, 100)
    after = _rng_snapshot()
    _assert_rng_equal(before, after)

    model.load_state_dict(initial)
    random.seed(41)
    np.random.seed(41)
    torch.manual_seed(41)
    before = _rng_snapshot()
    hook = MeasurementHook()
    measured = train_client(
        model,
        dataset,
        "client_00",
        1,
        config,
        torch.device("cpu"),
        71,
        100,
        measurement_hook=hook,
    )
    after = _rng_snapshot()
    _assert_rng_equal(before, after)
    assert all(torch.equal(unmeasured.state_dict[name], measured.state_dict[name]) for name in unmeasured.state_dict)
    assert hook.result is not None and hook.data_wait is not None and not hook.open
    observed = hook.observed.record()
    assert observed["actual_batch_count"] == 3
    assert observed["actual_presented_examples"] == 12


def _resource_row(position=0):
    return {
        "dataset": "shd",
        "experiment": "resource",
        "scientific_seed": 7,
        "communication_round": 1,
        "selected_position": position,
        "client_id": f"client_{position:02d}",
        "training_seed": 100 + position,
        "execution_attempt": 1,
        "gpu_uuid": "GPU-a",
        "example_count": 10,
        "local_batch_count": 1,
        "total_raw_input_events": 20,
        "mean_sequence_length": 3.0,
        "median_sequence_length": 3.0,
        "maximum_sequence_length": 4,
        "total_valid_time_bins": 30,
        "estimated_padded_time_bins": 32,
        "padding_fraction": 0.0625,
        "event_density": 0.01,
        "represented_class_count": 2,
        "label_entropy": 0.69,
        "client_wall_time_seconds": 1.0,
        "data_wait_time_seconds": 0.1,
        "cuda_event_time_seconds": 0.7,
        "residual_host_time_seconds": 0.2,
        "gross_energy_joules": 120.0,
        "idle_adjusted_energy_joules": 80.0,
        "energy_sample_count": 12,
        "energy_coverage_seconds": 1.0,
        "sampling_interval_ms": 100,
        "feature_source_scope": "client_training_indices",
        "validation_indices_in_features": False,
        "official_test_indices_in_features": False,
        "accepted": True,
        "split_id": "split-a",
        "partition_id": "partition-a",
        "model_initialization_id": "initialization-a",
        "model_configuration_identity": hashlib.sha256(
            json.dumps({"name": "model-a"}, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "git_commit": "abc",
        "model_identity": "Classifier",
        "parameter_count": 8,
    }


def _write_run(path):
    path.mkdir()
    atomic_write_json(
        path / "measurement_acceptance.json",
        {
            "accepted": True,
            "execution_completion": True,
            "measurement_completeness": True,
            "energy_completeness": True,
            "validation_findings": [],
            "sampling_error_count": 0,
        },
    )
    atomic_write_json(path / "measurement_config.json", {"sampling_interval_ms": 100})
    atomic_write_json(
        path / "calibration_reference.json",
        {"artifact": {"passed": True, "official_test_access_count": 0, "sampling_errors": []}},
    )
    execution_identity = {
        "git_commit": "abc",
        "configuration_id": "config-a",
        "hardware_allocation": {
            "visible_device_count": 1,
            "device_names": ["NVIDIA GH200"],
        },
    }
    atomic_write_json(
        path / "final_metrics.json",
        {
            "accepted": True,
            "completed": True,
            "execution_identity": execution_identity,
            "configuration_id": "config-a",
            "model_class": "Classifier",
            "parameter_count": 8,
            "parallel_execution": {
                "node_count": 1,
                "device_count": 1,
                "process_count": 1,
                "client_processes_per_device": 1,
                "cuda_process_service": "none",
                "control_backend": "nccl",
            },
            "data_protocol": {
                "official_test_access_count": 1,
                "official_test_monitored_during_training": False,
            },
        },
    )
    atomic_write_json(path / "git.json", {"commit": "abc"})
    atomic_write_json(path / "split.json", {"split_id": "split-a"})
    atomic_write_json(
        path / "partition.json",
        {
            "partition_id": "partition-a",
            "clients": [
                {"client_id": "client_00", "size": 10},
                {"client_id": "client_01", "size": 10},
            ],
        },
    )
    atomic_write_json(
        path / "model_initialization.json",
        {"model_initialization_id": "initialization-a"},
    )
    atomic_write_json(path / "execution_provenance.json", execution_identity)
    atomic_write_json(
        path / "execution_measurements.json", {"resource_allocation": {"job_id": "123"}}
    )
    atomic_write_json(path / "idle_power.json", {"attempts": []})
    for name in ("device_samples.jsonl", "execution_intervals.jsonl", "excluded_intervals.jsonl"):
        (path / name).write_text("")
    (path / "resolved_config.yaml").write_text("model:\n  name: model-a\n")
    records = [_resource_row(0), _resource_row(1)]
    (path / "client_resource_records.jsonl").write_text(
        "".join(json.dumps(value, sort_keys=True) + "\n" for value in records)
    )


def test_cost_dataset_construction_writes_schema_provenance_and_exclusions(tmp_path):
    run = tmp_path / "run"
    _write_run(run)
    result = tmp_path / "cost"
    record = build_client_cost_dataset([run], result, expected_rows=2)
    assert len(record["rows"]) == 2
    assert (result / "client_cost_data.csv").is_file()
    assert (result / "client_cost_schema.json").is_file()
    assert (result / "client_cost_provenance.json").is_file()
    schema = json.loads((result / "client_cost_schema.json").read_text())
    assert schema["$schema"].endswith("2020-12/schema")
    assert schema["feature_dictionary"]["client_wall_time_seconds"]["scheduler_input"] is False
    rows = (run / "client_resource_records.jsonl").read_text().splitlines()
    leaking = json.loads(rows[0])
    leaking["official_test_indices_in_features"] = True
    rows[0] = json.dumps(leaking)
    (run / "client_resource_records.jsonl").write_text("\n".join(rows) + "\n")
    with pytest.raises(ValueError, match="validation failed"):
        build_client_cost_dataset([run], tmp_path / "rejected", expected_rows=2)


def _cost_rows():
    rows = []
    for dataset_index, dataset in enumerate(("shd", "ssc")):
        for seed in (7, 17, 27):
            for round_number in (1, 2):
                for position in range(10):
                    client = position + (round_number - 1) * 10
                    base = 1.0 + dataset_index + client / 20 + round_number / 10 + seed / 100
                    rows.append(
                        {
                            "dataset": dataset,
                            "dataset_identity": f"{dataset}-a",
                            "scientific_seed": seed,
                            "communication_round": round_number,
                            "selected_position": position,
                            "client_id": f"client_{client:02d}",
                            "example_count": 20 + client,
                            "local_batch_count": 2 + client % 3,
                            "total_raw_input_events": 100 + client * 4,
                            "mean_sequence_length": 10 + client / 2,
                            "median_sequence_length": 9 + client / 2,
                            "maximum_sequence_length": 12 + client,
                            "total_valid_time_bins": 200 + client * 3,
                            "estimated_padded_time_bins": 220 + client * 4,
                            "padding_fraction": 0.05 + client / 1000,
                            "event_density": 0.01 + client / 10000,
                            "layer1_spike_count": 300 + client,
                            "layer2_spike_count": 200 + client,
                            "layer1_spike_rate": 0.1 + client / 1000,
                            "layer2_spike_rate": 0.05 + client / 2000,
                            "client_wall_time_seconds": base,
                            "cuda_event_time_seconds": base * 0.7,
                            "gross_energy_joules": base * 100,
                            "idle_adjusted_energy_joules": base * 70,
                        }
                    )
    return rows


def test_cost_fitting_writes_json_models_and_offline_assignment_artifacts(tmp_path):
    config = {
        "historical_weight_candidates": [0.3, 0.7],
        "ridge_regularization": [0.1],
        "percentage_denominator_floor": 0.001,
        "rank_correlation_tolerance": 0.01,
        "prediction_time_fraction_limit": 0.001,
        "minimum_runtime_error_improvement_fraction": 0.05,
    }
    result = fit_client_cost_models(_cost_rows(), tmp_path, config, {"git_commit": "abc"})
    assert result["evaluation"]["data_separation"]["evaluation_seed"] == 27
    assert "prequential_seed_27" in result["evaluation"]["settings"]
    assert result["evaluation"]["data_separation"]["historical_weight_selection"]["seed_27_used"] is False
    fitting_rows = [row for row in result["rows"] if row["scientific_seed"] in {7, 17}]
    evaluation_rows = [row for row in result["rows"] if row["scientific_seed"] == 27]
    fitting_hashes = {hashlib.sha256(canonical_json(row).encode()).hexdigest() for row in fitting_rows}
    evaluation_hashes = {hashlib.sha256(canonical_json(row).encode()).hexdigest() for row in evaluation_rows}
    exported_hashes = set(result["selected_scheduler_model"].fitting_row_hashes)
    assert exported_hashes == fitting_hashes
    assert not exported_hashes & evaluation_hashes
    assert len(exported_hashes) + len(evaluation_hashes) == len(result["rows"])
    assert (tmp_path / "client_cost_model.json").is_file()
    assert (tmp_path / "energy_cost_model.json").is_file()
    assert (tmp_path / "assignment_readiness.json").is_file()
    assert set(result["assignment"]["process_counts"]) == {2, 4}
    assert all(value["offline_evaluation_only"] for value in result["assignment"]["records"])
    run = tmp_path / "representative_run"
    run.mkdir()
    samples = [
        {
            "monotonic_timestamp_ns": timestamp,
            "power_watts": watts,
        }
        for timestamp, watts in ((0, 100.0), (100_000_000, 120.0), (200_000_000, 110.0))
    ]
    (run / "device_samples.jsonl").write_text(
        "".join(json.dumps(value) + "\n" for value in samples),
        encoding="utf-8",
    )
    interval = {
        "interval_id": "client-1",
        "category": "client_training",
        "accepted": True,
        "start_ns": 0,
        "end_ns": 200_000_000,
    }
    (run / "execution_intervals.jsonl").write_text(
        json.dumps(interval) + "\n",
        encoding="utf-8",
    )
    figure_root = tmp_path / "figures"
    figures = write_resource_figures(tmp_path, figure_root, [run])
    assert len(figures) == 10
    assert len(list(figure_root.glob("source_*.csv"))) == 10
