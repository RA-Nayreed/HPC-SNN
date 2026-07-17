import json

import pytest
import yaml

from fedapfa.cli.summarize_heterogeneity import summarize_heterogeneity
from fedapfa.cli.summarize_published_fedsnn import summarize_published_fedsnn
from fedapfa.configuration import load_heterogeneity_manifest, load_published_fedsnn_manifest

MANIFEST = "experiments/heterogeneity_evaluation/manifest.yaml"
FEDERATED_SUMMARY = "results/federated/federated_summary.json"


def _candidate(root, task, suffix):
    path = root / f"{task.experiment}-seed{task.seed}-{suffix}"
    path.mkdir(parents=True)
    (path / "resolved_config.yaml").write_text(yaml.safe_dump(task.config), encoding="utf-8")
    return path


@pytest.mark.parametrize("condition", ["missing", "duplicate", "incompatible"])
def test_heterogeneity_summary_rejects_incomplete_execution_sets(tmp_path, condition):
    runs = tmp_path / "runs"
    runs.mkdir()
    task = load_heterogeneity_manifest(MANIFEST)[0]
    if condition == "duplicate":
        _candidate(runs, task, "a")
        _candidate(runs, task, "b")
    elif condition == "incompatible":
        path = _candidate(runs, task, "a")
        (path / "acceptance.json").write_text(json.dumps({"accepted": False, "completed": False}), encoding="utf-8")
        (path / "final_metrics.json").write_text("{}", encoding="utf-8")
        (path / "partition.json").write_text("{}", encoding="utf-8")
        (path / "round_metrics.jsonl").write_text(
            "".join(json.dumps({"round_number": value}) + "\n" for value in range(1, 101)),
            encoding="utf-8",
        )
    summary = summarize_heterogeneity(
        MANIFEST,
        runs,
        tmp_path / "context",
        FEDERATED_SUMMARY,
        tmp_path / "output",
    )
    assert not summary["valid"]
    joined = " ".join(summary["validation_findings"])
    if condition == "duplicate":
        assert "found 2" in joined
    elif condition == "incompatible":
        assert "not accepted and completed" in joined
    else:
        assert "found 0" in joined


def test_published_summary_keeps_completion_and_scientific_status_independent(tmp_path):
    summary = summarize_published_fedsnn(
        "experiments/published_fedsnn/manifest.yaml", tmp_path / "runs", tmp_path / "output"
    )
    assert not summary["completed"]
    assert summary["scientific_status"] == "equivalence_not_established"
    assert summary["acceptance_reference_accuracy"] is None
    assert summary["expected_task_count"] == 6
    assert len(summary["treatment_summaries"]) == 2
    assert not summary["treatments_pooled"]


def test_published_summary_reports_separate_table_i_rows_without_validation_metrics(tmp_path):
    tasks = load_published_fedsnn_manifest("experiments/published_fedsnn/manifest.yaml")
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    for task in tasks:
        path = _candidate(runs_root, task, "complete")
        reference = task.config["acceptance"]["descriptive_reference_accuracy"]
        signed_offset = 0.01 if task.config["federated"]["partition"]["alpha"] is None else -0.02
        distribution = (
            "iid"
            if task.config["federated"]["partition"]["method"] == "fedsnn_random_iid"
            else "label_dirichlet_non_iid"
        )
        (path / "acceptance.json").write_text(
            json.dumps(
                {
                    "accepted": True,
                    "completed": True,
                    "scientific_status": "equivalence_not_established",
                    "reference_test_accuracy": None,
                    "tolerance": None,
                    "descriptive_reference_accuracy": reference,
                }
            ),
            encoding="utf-8",
        )
        (path / "final_metrics.json").write_text(
            json.dumps(
                {
                    "selected_round": 100,
                    "checkpoint_selection": "final_round",
                    "best_validation_accuracy": None,
                    "final_validation_accuracy": None,
                    "selected_validation": None,
                    "client_distribution_weighted_validation_accuracy": None,
                    "test": {"accuracy": reference + signed_offset, "macro_f1": 0.7},
                    "data_protocol": {
                        "examples_available_before_validation_separation": 50000,
                        "examples_used_for_client_training": 50000,
                        "examples_used_for_validation": 0,
                        "official_test_examples": 10000,
                        "official_test_access_count": 1,
                    },
                    "distribution": distribution,
                    "partition_alpha": task.config["federated"]["partition"].get("alpha"),
                    "local_epochs": 5,
                    "total_clients": 10,
                    "participating_clients": 2,
                    "timesteps": 20,
                    "momentum": 0.95,
                    "weight_decay": 0.0001,
                    "aggregation_weighting": "uniform",
                }
            ),
            encoding="utf-8",
        )
        (path / "round_metrics.jsonl").write_text(
            "".join(
                json.dumps(
                    {
                        "round_number": round_number,
                        "total_selected_examples": 10000,
                        "total_training_examples_presented": 49600,
                    }
                )
                + "\n"
                for round_number in range(1, 101)
            ),
            encoding="utf-8",
        )
    summary = summarize_published_fedsnn(
        "experiments/published_fedsnn/manifest.yaml",
        runs_root,
        tmp_path / "summary",
    )
    assert summary["valid"] and summary["completed_task_count"] == 6
    by_distribution = {value["distribution"]: value for value in summary["treatment_summaries"]}
    assert by_distribution["iid"]["seed_count"] == 3
    assert by_distribution["label_dirichlet_non_iid"]["seed_count"] == 3
    iid_run = by_distribution["iid"]["runs"][0]
    noniid_run = by_distribution["label_dirichlet_non_iid"]["runs"][0]
    assert "best_validation_accuracy" not in iid_run
    assert iid_run["signed_difference_percentage_points"] == pytest.approx(1.0)
    assert noniid_run["signed_difference_percentage_points"] == pytest.approx(-2.0)
    assert iid_run["complete_training_examples"] == 50000
    assert iid_run["internal_validation_examples"] == 0
    assert iid_run["official_test_access_count"] == 1
