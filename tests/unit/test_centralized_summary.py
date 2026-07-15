import json

import pytest
import yaml

from fedapfa.cli.summarize_centralized import summarize_centralized
from fedapfa.configuration import load_centralized_manifest

MANIFEST = "experiments/centralized/manifest.yaml"


def _write_runs(root, skip=None, wrong_protocol=None):
    tasks = load_centralized_manifest(MANIFEST)
    for task in tasks:
        key = (task.experiment, task.seed)
        if key == skip:
            continue
        run = root / f"{task.config['name']}-seed{task.seed}-synthetic"
        run.mkdir(parents=True)
        config = task.config
        if key == wrong_protocol:
            config = dict(config)
            config["protocol"] = (
                "published_protocol" if task.protocol == "independent_evaluation" else "independent_evaluation"
            )
        (run / "resolved_config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
        seed_offset = {7: -0.1, 17: 0.0, 27: 0.1}[task.seed]
        protocol_base = 0.8 if task.protocol == "independent_evaluation" else 0.4
        accuracy = protocol_base + seed_offset
        acceptance = {
            "mode": "scientific_evaluation",
            "completed": True,
            "completion_failures": [],
            "scientific_status": "not_claimed",
            "reference_test_accuracy": None,
            "achieved_test_accuracy": accuracy,
            "absolute_accuracy_difference": None,
            "tolerance": None,
            "protocol": config["protocol"],
            "seed": task.seed,
            "git_commit": "abc",
        }
        final = {
            "best_selection_accuracy": accuracy + 0.01,
            "test": {
                "accuracy": accuracy,
                "spike_rates": {"layer1": 0.1 + seed_offset / 10, "layer2": 0.2 + seed_offset / 10},
            },
            "runtime_seconds": 100.0 + seed_offset * 10,
            "peak_cuda_memory_bytes": 1000.0 + seed_offset * 100,
            "parameter_count": 1234 + 100 * list(dict.fromkeys(t.experiment for t in tasks)).index(task.experiment),
        }
        (run / "acceptance.json").write_text(json.dumps(acceptance), encoding="utf-8")
        (run / "final_metrics.json").write_text(json.dumps(final), encoding="utf-8")


def test_summary_mean_sample_standard_deviation_and_protocol_separation(tmp_path):
    runs = tmp_path / "runs"
    output = tmp_path / "summary"
    _write_runs(runs)
    summary = summarize_centralized(MANIFEST, runs, output)
    assert summary["valid"]
    assert len(summary["experiments"]) == 6
    independent = next(
        item for item in summary["experiments"] if item["experiment"] == "shd_lif_independent_evaluation"
    )
    paper = next(item for item in summary["experiments"] if item["experiment"] == "shd_dcls_published_protocol")
    assert independent["metrics"]["official_test_accuracy"]["mean"] == pytest.approx(0.8)
    assert independent["metrics"]["official_test_accuracy"]["sample_standard_deviation"] == pytest.approx(0.1)
    assert paper["metrics"]["official_test_accuracy"]["mean"] == pytest.approx(0.4)
    assert independent["protocol"] == "independent_evaluation"
    assert paper["protocol"] == "published_protocol"
    assert all(item["scientific_status"] == "not_claimed" for item in summary["experiments"])
    assert all(
        (output / name).is_file()
        for name in ("centralized_summary.json", "centralized_summary.csv", "centralized_summary.md")
    )


def test_summary_detects_missing_seed(tmp_path):
    runs = tmp_path / "runs"
    _write_runs(runs, skip=("ssc_lif_128_independent_evaluation", 27))
    summary = summarize_centralized(MANIFEST, runs, tmp_path / "summary")
    assert not summary["valid"]
    assert any("ssc_lif_128_independent_evaluation seed 27: missing" in error for error in summary["errors"])


def test_summary_rejects_protocol_mismatch_instead_of_mixing(tmp_path):
    runs = tmp_path / "runs"
    _write_runs(runs, wrong_protocol=("shd_lif_independent_evaluation", 17))
    summary = summarize_centralized(MANIFEST, runs, tmp_path / "summary")
    assert not summary["valid"]
    assert any("protocol mismatch would mix" in error for error in summary["errors"])
