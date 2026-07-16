import json

import pytest
import yaml

from fedapfa.cli.summarize_heterogeneity import summarize_heterogeneity
from fedapfa.cli.summarize_published_fedsnn import summarize_published_fedsnn
from fedapfa.configuration import load_heterogeneity_manifest

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
        (path / "acceptance.json").write_text(
            json.dumps({"accepted": False, "completed": False}), encoding="utf-8"
        )
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
    assert summary["scientific_status"] == "not_claimed"
    assert summary["reference_test_accuracy"] is None
