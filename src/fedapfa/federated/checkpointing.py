"""Atomic FedAvg checkpoints with exact deterministic-resumption state."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from fedapfa.configuration.experiment_id import canonical_config
from fedapfa.utilities.serialization import atomic_torch_save

from .randomness import global_rng_state, restore_global_rng_state


def configuration_identity(config: dict) -> str:
    identity = json.loads(canonical_config(config))
    identity.pop("output_root", None)
    identity.pop("resume", None)
    return hashlib.sha256(canonical_config(identity).encode("utf-8")).hexdigest()


def state_identity(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def read_git_commit(run_dir: str | Path) -> str:
    metadata = json.loads((Path(run_dir) / "git.json").read_text(encoding="utf-8"))
    commit = metadata.get("commit")
    if not isinstance(commit, str) or not commit:
        raise RuntimeError("run Git commit identity is missing")
    return commit


def save_federated_checkpoint(
    path: str | Path,
    model: nn.Module,
    config: dict,
    run_dir: str | Path,
    next_round: int,
    best_validation_accuracy: float,
    best_validation_round: int,
    selection_state: dict,
    split_id: str,
    partition_id: str,
    model_initialization_id: str,
    cumulative_download_bytes: int,
    cumulative_upload_bytes: int,
    client_records: list[dict],
    round_records: list[dict],
) -> None:
    atomic_torch_save(
        path,
        {
            "schema_version": 1,
            "global_model_state": model.state_dict(),
            "model_class": type(model).__name__,
            "next_round": next_round,
            "best_validation_accuracy": best_validation_accuracy,
            "best_validation_round": best_validation_round,
            "selection_generator_state": selection_state,
            "global_random_states": global_rng_state(),
            "split_id": split_id,
            "partition_id": partition_id,
            "model_initialization_id": model_initialization_id,
            "configuration_id": configuration_identity(config),
            "resolved_config": config,
            "git_commit": read_git_commit(run_dir),
            "cumulative_download_bytes": cumulative_download_bytes,
            "cumulative_upload_bytes": cumulative_upload_bytes,
            "client_records": client_records,
            "round_records": round_records,
        },
    )


def load_federated_checkpoint(
    path: str | Path,
    model: nn.Module,
    config: dict,
    run_dir: str | Path,
    split_id: str,
    partition_id: str,
    model_initialization_id: str,
    restore_random_states: bool = True,
) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    expected = {
        "model_class": type(model).__name__,
        "configuration_id": configuration_identity(config),
        "resolved_config": config,
        "git_commit": read_git_commit(run_dir),
        "split_id": split_id,
        "partition_id": partition_id,
        "model_initialization_id": model_initialization_id,
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise RuntimeError(f"federated checkpoint {key} is incompatible")
    if not isinstance(checkpoint.get("next_round"), int) or checkpoint["next_round"] < 1:
        raise RuntimeError("federated checkpoint next_round is invalid")
    model.load_state_dict(checkpoint["global_model_state"], strict=True)
    if restore_random_states:
        restore_global_rng_state(checkpoint["global_random_states"])
    return checkpoint
