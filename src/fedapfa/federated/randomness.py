"""Stable identities for independent deterministic random streams."""

from __future__ import annotations

import hashlib
import random
from typing import Any

import numpy as np
import torch


def derive_seed(experiment_seed: int, stream_identity: str, *coordinates: object) -> int:
    payload = ":".join([str(experiment_seed), stream_identity, *(str(value) for value in coordinates)])
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big") % (2**63 - 1)


def resolved_seeds(config: dict) -> dict[str, int]:
    return {
        name: (config["seed"] if name == "split" else derive_seed(config["seed"], identity))
        for name, identity in config["seed_streams"].items()
    }


def global_rng_state(cuda_device: torch.device | None = None) -> dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": None,
    }
    if torch.cuda.is_available():
        if cuda_device is None:
            state["torch_cuda"] = torch.cuda.get_rng_state_all()
        else:
            state["torch_cuda_device"] = torch.cuda.get_rng_state(cuda_device)
            state["torch_cuda_device_index"] = cuda_device.index
    return state


def restore_global_rng_state(state: dict[str, Any], cuda_device: torch.device | None = None) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    if state.get("torch_cuda_device") is not None and torch.cuda.is_available():
        if cuda_device is None or cuda_device.index != state.get("torch_cuda_device_index"):
            raise RuntimeError("CUDA random state device is incompatible")
        torch.cuda.set_rng_state(state["torch_cuda_device"], cuda_device)
