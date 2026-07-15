"""Logical model-transfer accounting, distinct from measured network traffic."""

from __future__ import annotations

import torch


def model_payload_bytes(state: dict[str, torch.Tensor]) -> int:
    if not state:
        raise ValueError("model state is empty")
    total = 0
    for name, tensor in state.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"communicated state value {name} is not a tensor")
        total += tensor.numel() * tensor.element_size()
    return total


def communication_for_clients(payload_bytes: int, selected_clients: int) -> dict[str, int]:
    if payload_bytes <= 0 or selected_clients <= 0:
        raise ValueError("payload bytes and selected-client count must be positive")
    download = payload_bytes * selected_clients
    upload = payload_bytes * selected_clients
    return {"download_bytes": download, "upload_bytes": upload, "total_bytes": download + upload}
