"""Derive SHD validation diagnostics from accepted historical FedAvg executions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

from fedapfa.configuration.scientific_manifests import ContextTask
from fedapfa.datasets.partition_diagnostics import diagnostics_from_artifact
from fedapfa.datasets.shd import EventAudioDataset
from fedapfa.federated.checkpointing import state_identity
from fedapfa.federated.client import evaluate_model
from fedapfa.federated.data_protocol import file_identity
from fedapfa.metrics.client_fairness import fairness_proxy_record
from fedapfa.models.model_factory import make_model
from fedapfa.utilities.serialization import atomic_write_json, sha256_json


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _context_records(summary_path: Path) -> list[ContextTask]:
    summary = _read_json(summary_path)
    matches = [
        value
        for value in summary.get("experiments", [])
        if value.get("experiment") == "shd_lif_dirichlet_alpha_0_5_participation_0_50"
    ]
    if len(matches) != 1 or matches[0].get("completed") is not True:
        raise ValueError("the committed summary has no unique completed alpha 0.5 participation 0.50 treatment")
    runs = matches[0].get("runs", [])
    if sorted(value.get("seed") for value in runs) != [7, 17, 27]:
        raise ValueError("historical context must contain seeds 7, 17, and 27 exactly once")
    return [ContextTask(int(value["seed"]), str(matches[0]["experiment"]), dict(value)) for value in runs]


def _locate_run(root: Path, task: ContextTask) -> Path:
    candidates: list[Path] = []
    named = root / Path(task.source_record["run_directory"]).name
    if named.is_dir():
        candidates.append(named)
    for path in sorted(root.iterdir() if root.is_dir() else []):
        config_path = path / "resolved_config.yaml"
        if not path.is_dir() or not config_path.is_file() or path in candidates:
            continue
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if config.get("name") == task.experiment and config.get("seed") == task.seed:
            candidates.append(path)
    if len(candidates) != 1:
        raise ValueError(f"seed {task.seed} requires exactly one historical execution, found {len(candidates)}")
    return candidates[0]


def _verify_source(run_dir: Path, task: ContextTask) -> tuple[dict, dict, dict, dict, str]:
    config = yaml.safe_load((run_dir / "resolved_config.yaml").read_text(encoding="utf-8")) or {}
    acceptance = _read_json(run_dir / "acceptance.json")
    split = _read_json(run_dir / "split.json")
    partition = _read_json(run_dir / "partition.json")
    initialization = _read_json(run_dir / "model_initialization.json")
    git_commit = _read_json(run_dir / "git.json").get("commit")
    if config.get("name") != task.experiment or config.get("seed") != task.seed:
        raise ValueError("historical execution configuration or seed is incompatible")
    federation = config.get("federated", {})
    partition_config = federation.get("partition", {})
    if (
        config.get("protocol") != "independent_evaluation"
        or federation.get("participation_fraction") != 0.5
        or partition_config.get("method") != "label_dirichlet"
        or partition_config.get("alpha") != 0.5
    ):
        raise ValueError("historical execution is not the required alpha 0.5 participation 0.50 protocol")
    if acceptance.get("accepted") is not True or acceptance.get("completed") is not True:
        raise ValueError("historical execution is not accepted and completed")
    identities = {
        "split_id": split.get("split_id"),
        "partition_id": partition.get("partition_id"),
        "model_initialization_id": initialization.get("model_initialization_id"),
        "git_commit": git_commit,
    }
    for key, expected in identities.items():
        if not expected or task.source_record.get(key) != expected or acceptance.get(key) != expected:
            raise ValueError(f"historical {key} does not match the committed summary and acceptance record")
    if initialization.get("model_class") != "AudioLIFSNN" or acceptance.get("model_class") != "AudioLIFSNN":
        raise ValueError("historical model class is not AudioLIFSNN")
    split_core = dict(split)
    recorded_split_id = split_core.pop("split_id", None)
    partition_core = dict(partition)
    recorded_partition_id = partition_core.pop("partition_id", None)
    if recorded_split_id != sha256_json(split_core) or recorded_partition_id != sha256_json(partition_core):
        raise ValueError("historical split or partition artifact identity is invalid")
    return config, split, partition, initialization, str(git_commit)


def derive_context_record(
    task: ContextTask,
    runs_root: str | Path,
    data_root: str | Path,
    output_root: str | Path,
) -> dict:
    """Derive one seed record without constructing or opening the official test split."""

    run_dir = _locate_run(Path(runs_root), task)
    config, split, partition, initialization, git_commit = _verify_source(run_dir, task)
    train_path = Path(data_root) / config["dataset"]["train_file"]
    if file_identity(train_path) != split.get("dataset_identity"):
        raise ValueError("supplied SHD training file identity differs from the historical split")
    with h5py.File(train_path, "r") as handle:
        labels = np.asarray(handle["labels"][:], dtype=np.int64)
    validation_indices = np.asarray(split["validation_indices"], dtype=np.int64)
    validation = EventAudioDataset(
        train_path,
        validation_indices,
        temporal_bin_ms=config["dataset"]["temporal_bin_ms"],
        frequency_bin_factor=config["dataset"]["frequency_bin_factor"],
        validate=False,
    )
    model = make_model(config)
    checkpoint = torch.load(run_dir / "checkpoints" / "best.pt", map_location="cpu", weights_only=False)
    checkpoint_expected = {
        "model_class": "AudioLIFSNN",
        "git_commit": git_commit,
        "split_id": split["split_id"],
        "partition_id": partition["partition_id"],
        "model_initialization_id": initialization["model_initialization_id"],
    }
    if any(checkpoint.get(key) != value for key, value in checkpoint_expected.items()):
        raise ValueError("historical selected checkpoint identities are incompatible")
    model.load_state_dict(checkpoint["global_model_state"], strict=True)
    if state_identity(checkpoint["global_model_state"]) == initialization["model_initialization_id"]:
        raise ValueError("selected checkpoint unexpectedly equals the initialization state")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    evaluation = evaluate_model(
        model,
        validation,
        device,
        int(config["federated"]["local_batch_size"]),
        int(config["seed"]),
        0,
        False,
    )
    diagnostics = diagnostics_from_artifact(partition, labels, split["training_indices"])
    source_final = _read_json(run_dir / "final_metrics.json")
    round_records = [
        json.loads(line)
        for line in (run_dir / "round_metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(round_records) != int(config["federated"]["rounds"]):
        raise ValueError("historical communication-round records are incomplete")
    if source_final.get("test", {}).get("accuracy") != task.source_record.get("official_test_accuracy"):
        raise ValueError("historical official-test metric differs from the committed summary")
    fairness = fairness_proxy_record(evaluation.per_class_accuracy, partition)
    record = {
        "schema_version": 1,
        "seed": task.seed,
        "experiment": task.experiment,
        "source_run_directory": str(run_dir),
        "source_identities": {
            **checkpoint_expected,
            "dataset_identity": split["dataset_identity"],
        },
        "protocol_settings": {
            "dataset": {key: value for key, value in config["dataset"].items() if key != "root"},
            "model": config["model"],
            "federated": {
                key: config["federated"].get(key)
                for key in (
                    "clients", "clients_per_round", "participation_fraction", "rounds",
                    "local_epochs", "local_batch_size", "optimizer", "learning_rate",
                    "weight_decay", "gradient_clip", "client_sampling",
                )
            },
        },
        "validation": {
            "loss": evaluation.loss,
            "accuracy": evaluation.accuracy,
            "macro_f1": evaluation.macro_f1,
            "per_class_accuracy": evaluation.per_class_accuracy,
            "confusion_matrix": evaluation.confusion_matrix,
            "spike_rates": evaluation.spike_rates,
        },
        "client_distribution_weighted_validation_accuracy": fairness,
        "historical_metrics": {
            "best_validation_accuracy": source_final["best_validation_accuracy"],
            "official_test_accuracy": source_final["test"]["accuracy"],
            "selected_round": source_final["selected_round"],
            "final_validation_accuracy": source_final["final_validation_accuracy"],
            "logical_communication_bytes": source_final["logical_communication"]["cumulative_total_bytes"],
            "execution_time_seconds": source_final["execution_time_seconds"],
            "mean_client_update_l2_norm": source_final["mean_client_update_l2_norm"],
            "mean_client_spike_rates": source_final["mean_client_spike_rates"],
            "validation_curve": [
                [record["round_number"], record["validation_accuracy"]] for record in round_records
            ],
        },
        "partition_diagnostics": diagnostics,
        "update_alignment": {
            "available": False,
            "reason": "Historical local update tensors were not retained.",
        },
        "official_test_reevaluated": False,
    }
    output = Path(output_root) / f"context-seed{task.seed}-{partition['partition_id'][:12]}"
    output.mkdir(parents=True, exist_ok=True)
    target = output / "context.json"
    if target.is_file() and _read_json(target) != record:
        raise RuntimeError(f"incompatible derived context record already exists: {target}")
    atomic_write_json(target, record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive validation-only diagnostics for historical SHD context.")
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--federated-summary", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, choices=(7, 17, 27))
    args = parser.parse_args()
    tasks = _context_records(Path(args.federated_summary))
    selected = tasks if args.seed is None else [task for task in tasks if task.seed == args.seed]
    for task in selected:
        derive_context_record(task, args.runs_root, args.data_root, args.output_root)


if __name__ == "__main__":
    main()
