import json

import pytest
import yaml

from fedapfa.cli.summarize_federated import summarize_federated
from fedapfa.configuration import load_federated_manifest

MANIFEST = "experiments/federated_baselines/manifest.yaml"


def _write_runs(root, missing=None):
    for task in load_federated_manifest(MANIFEST):
        key = (task.experiment, task.seed)
        if key == missing:
            continue
        run = root / f"{task.config['name']}-seed{task.seed}-synthetic"
        run.mkdir(parents=True)
        (run / "resolved_config.yaml").write_text(yaml.safe_dump(task.config), encoding="utf-8")
        participation = task.config["federated"]["participation_fraction"]
        offset = {7: -0.01, 17: 0.0, 27: 0.01}[task.seed]
        accuracy = (0.7 if participation == 0.5 else 0.6) + offset
        identity = f"seed-{task.seed}"
        acceptance = {
            "accepted": True,
            "completed": True,
            "scientific_status": "not_claimed",
            "reference_test_accuracy": None,
            "protocol": "independent_evaluation",
            "seed": task.seed,
            "split_id": f"split-{identity}",
            "partition_id": f"partition-{identity}",
            "model_initialization_id": f"model-{identity}",
            "git_commit": "abc",
        }
        final = {
            "best_validation_accuracy": accuracy + 0.02,
            "test": {"accuracy": accuracy},
            "selected_round": 50 + task.seed % 3,
            "final_validation_accuracy": accuracy + 0.01,
            "logical_communication": {"cumulative_total_bytes": participation * 1000},
            "execution_time_seconds": participation * 100,
            "mean_client_update_l2_norm": participation,
            "mean_client_spike_rates": {"layer1": 0.1 + offset, "layer2": 0.05 + offset},
            "parameter_count": 107028,
        }
        (run / "acceptance.json").write_text(json.dumps(acceptance), encoding="utf-8")
        (run / "final_metrics.json").write_text(json.dumps(final), encoding="utf-8")


def test_federated_summary_statistics_pairing_and_centralized_context(tmp_path):
    runs = tmp_path / "runs"
    output = tmp_path / "output"
    _write_runs(runs)
    summary = summarize_federated(MANIFEST, runs, output)
    assert summary["valid"]
    assert len(summary["experiments"]) == 2
    higher = next(item for item in summary["experiments"] if item["participation_fraction"] == 0.5)
    assert higher["metrics"]["official_test_accuracy"]["mean"] == pytest.approx(0.7)
    assert higher["metrics"]["official_test_accuracy"]["sample_standard_deviation"] == pytest.approx(0.01)
    paired = summary["paired_participation_differences"]
    assert [item["official_test_accuracy_difference"] for item in paired["runs"]] == pytest.approx([0.1] * 3)
    assert summary["centralized_context"]["experiment"] == "shd_lif_independent_evaluation"
    assert all(item["scientific_status"] == "not_claimed" for item in summary["experiments"])
    assert all(
        (output / name).is_file()
        for name in ("federated_summary.json", "federated_summary.csv", "federated_summary.md")
    )


def test_federated_summary_detects_missing_seed(tmp_path):
    tasks = load_federated_manifest(MANIFEST)
    missing = (tasks[-1].experiment, 27)
    runs = tmp_path / "runs"
    _write_runs(runs, missing=missing)
    summary = summarize_federated(MANIFEST, runs, tmp_path / "output")
    assert not summary["valid"]
    assert any("seed 27: missing mandatory execution" in finding for finding in summary["validation_findings"])
