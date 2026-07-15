"""Truthful execution-completion and scientific-reproduction records."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _finite_tree(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_tree(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def _read_metrics(path: Path) -> tuple[list[dict], list[str]]:
    failures = []
    records = []
    if not path.is_file() or path.stat().st_size == 0:
        return records, ["metrics.jsonl is missing or empty"]
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            failures.append(f"metrics.jsonl line {line_number} is invalid")
            continue
        if not isinstance(record, dict):
            failures.append(f"metrics.jsonl line {line_number} is not an object")
            continue
        records.append(record)
    if not records:
        failures.append("metrics.jsonl has no metric records")
    return records, failures


def _git_commit(run_dir: Path) -> str | None:
    try:
        metadata = json.loads((run_dir / "git.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        return None
    return metadata.get("commit") if isinstance(metadata, dict) else None


def evaluate_acceptance(
    config: Mapping[str, Any],
    run_dir: str | Path,
    final_metrics: Mapping[str, Any],
    protocol_metadata: Mapping[str, Any],
    termination: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate completion independently from any paper-accuracy claim."""

    path = Path(run_dir)
    mode = config["mode"]
    test_metrics = final_metrics.get("test")
    achieved = test_metrics.get("accuracy") if isinstance(test_metrics, Mapping) else None
    achieved = float(achieved) if isinstance(achieved, (int, float)) and math.isfinite(achieved) else None
    acceptance_config = config.get("acceptance", {})
    reference = acceptance_config.get("reference_test_accuracy")
    tolerance = acceptance_config.get("absolute_tolerance")
    failures: list[str] = []

    if mode == "scientific_evaluation":
        records, metric_failures = _read_metrics(path / "metrics.jsonl")
        failures.extend(metric_failures)
        if records and not _finite_tree(records):
            failures.append("training metrics contain NaN or Inf")
        if not _finite_tree(final_metrics):
            failures.append("final metrics contain NaN or Inf")

        reason = termination.get("reason")
        last_epoch = termination.get("last_epoch")
        if reason == "epochs_completed":
            if last_epoch != config["training"]["epochs"] - 1:
                failures.append("configured epochs were not completed")
        elif reason == "early_stopped":
            patience = config["training"].get("early_stop_patience")
            if (
                patience is None
                or termination.get("epochs_without_improvement", 0) < patience
                or termination.get("early_stop_documented") is not True
            ):
                failures.append("early-stop condition is not documented or valid")
        else:
            failures.append("training has no valid completion condition")

        if records:
            record_epochs = [record.get("epoch") for record in records]
            expected_epochs = (
                list(range(int(last_epoch) + 1)) if isinstance(last_epoch, int) and last_epoch >= 0 else []
            )
            if record_epochs != expected_epochs:
                failures.append("epoch metric records are missing, duplicated, or out of order")

        for checkpoint in ("best_validation.pt", "last.pt"):
            checkpoint_path = path / "checkpoints" / checkpoint
            if not checkpoint_path.is_file() or checkpoint_path.stat().st_size == 0:
                failures.append(f"missing checkpoint: checkpoints/{checkpoint}")

        batch_keys = ("max_train_batches", "max_validation_batches", "max_test_batches")
        if any(config["training"].get(key) is not None for key in batch_keys):
            failures.append("a batch limit was active")
        if any(config["subset"].get(key) != 0 for key in ("train_examples", "validation_examples", "test_examples")):
            failures.append("a dataset subset limit was active")
        if protocol_metadata.get("complete_dataset_used") is not True:
            failures.append("complete dataset use was not verified")
        if protocol_metadata.get("official_test_accessed") is not True:
            failures.append("official test evaluation is missing")
        if protocol_metadata.get("official_test_evaluated_after_model_selection") is not True:
            failures.append("official test was not evaluated after model selection")
        if not isinstance(test_metrics, Mapping) or not test_metrics:
            failures.append("official test metrics are missing")
        expected_class = acceptance_config.get("expected_model_class")
        if final_metrics.get("model_class") != expected_class:
            failures.append(f"model class mismatch: expected {expected_class}, got {final_metrics.get('model_class')}")
        if _git_commit(path) is None:
            failures.append("Git commit metadata is missing")
        log_path = path / "training.log"
        if not log_path.is_file() or log_path.stat().st_size == 0:
            failures.append("training.log is missing or empty")
    else:
        legacy_accepted = bool(final_metrics.get("accepted"))
        if not legacy_accepted:
            failures.append(f"{mode} acceptance condition was not met")

    completed = not failures
    difference = abs(achieved - reference) if achieved is not None and reference is not None else None
    if reference is None:
        scientific_status = "not_claimed"
    elif completed and difference is not None and tolerance is not None and difference <= tolerance:
        scientific_status = "passed"
    else:
        scientific_status = "failed"

    official_test = {
        "accessed": bool(protocol_metadata.get("official_test_accessed")),
        "monitored_during_training": bool(protocol_metadata.get("official_test_monitored_during_training")),
        "evaluated_after_model_selection": bool(protocol_metadata.get("official_test_evaluated_after_model_selection")),
        "role": protocol_metadata.get("official_test_role"),
        "examples": protocol_metadata.get("official_test_examples"),
        "complete_split": bool(protocol_metadata.get("complete_dataset_used")),
    }
    return {
        "mode": mode,
        "accepted": completed,
        "completed": completed,
        "completion_failures": failures,
        "scientific_status": scientific_status,
        "reference_test_accuracy": reference,
        "achieved_test_accuracy": achieved,
        "absolute_accuracy_difference": difference,
        "tolerance": tolerance,
        "protocol": config["protocol"],
        "seed": config["seed"],
        "git_commit": _git_commit(path),
        "dataset": config["dataset"]["name"],
        "model_class": final_metrics.get("model_class"),
        "official_test_access_information": official_test,
        "termination": dict(termination),
    }
