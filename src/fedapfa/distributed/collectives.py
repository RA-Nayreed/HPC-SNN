"""Transient distributed collectives for model state and client results."""

from __future__ import annotations

import io

import torch
import torch.distributed as dist
from torch import nn

from fedapfa.federated.checkpointing import state_identity

from .process_context import ProcessContext


def broadcast_model_state(model: nn.Module, context: ProcessContext) -> str:
    """Broadcast the authoritative rank-zero state and verify its identity."""

    if context.backend == "nccl":
        for tensor in model.state_dict().values():
            dist.broadcast(tensor, src=0)
    else:
        received = {}
        for name, tensor in model.state_dict().items():
            control_tensor = (
                tensor.detach().cpu().clone()
                if context.is_coordinator
                else torch.empty_like(tensor, device="cpu")
            )
            dist.broadcast(control_tensor, src=0)
            received[name] = control_tensor
        model.load_state_dict(received)
    identity = state_identity(model.state_dict())
    identities: list[str | None] = [None for _ in range(context.world_size)]
    dist.all_gather_object(identities, identity)
    if any(value != identity for value in identities):
        raise RuntimeError("distributed processes received incompatible global model states")
    return identity


def _serialize(value) -> bytes:
    buffer = io.BytesIO()
    torch.save(value, buffer)
    return buffer.getvalue()


def _deserialize(payload: bytes):
    return torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)


def gather_rank_payloads(value, context: ProcessContext) -> tuple[list | None, list[int]]:
    """Collect CPU payloads on rank zero using one transient device buffer per sender."""

    serialized = _serialize(value)
    local_size = torch.tensor([len(serialized)], dtype=torch.int64, device=context.control_device)
    size_tensors = [torch.zeros_like(local_size) for _ in range(context.world_size)]
    dist.all_gather(size_tensors, local_size)
    sizes = [int(item.item()) for item in size_tensors]
    local_bytes = torch.frombuffer(bytearray(serialized), dtype=torch.uint8).to(context.control_device)
    if context.is_coordinator:
        values = [value]
        del local_bytes
        for source in range(1, context.world_size):
            receive = torch.empty(sizes[source], dtype=torch.uint8, device=context.control_device)
            dist.recv(receive, src=source)
            payload = receive.cpu().numpy().tobytes()
            values.append(_deserialize(payload))
            del receive
        return values, sizes
    dist.send(local_bytes, dst=0)
    del local_bytes
    return None, sizes
