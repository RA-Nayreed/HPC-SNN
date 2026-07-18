"""Process-group initialization and single-node identity verification."""

from __future__ import annotations

import os
import socket
from dataclasses import asdict, dataclass
from datetime import timedelta

import torch
import torch.distributed as dist


def process_resident_memory_bytes() -> int | None:
    try:
        with open("/proc/self/statm", encoding="utf-8") as stream:
            resident_pages = int(stream.read().split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        return None


@dataclass(frozen=True)
class ProcessContext:
    rank: int
    local_rank: int
    world_size: int
    backend: str
    device: torch.device
    control_device: torch.device
    host: str
    visible_device_count: int
    physical_device_count: int
    device_index: int
    device_slot: int
    client_processes_per_device: int
    cuda_process_service: str
    device_name: str | None
    device_total_memory_bytes: int | None
    device_compute_capability: tuple[int, int] | None

    @property
    def is_coordinator(self) -> bool:
        return self.rank == 0

    def record(self) -> dict:
        value = asdict(self)
        value["device"] = str(self.device)
        value["control_device"] = str(self.control_device)
        if self.device_compute_capability is not None:
            value["device_compute_capability"] = list(self.device_compute_capability)
        value["process_resident_memory_bytes"] = process_resident_memory_bytes()
        return value


def _environment_integer(name: str) -> int:
    value = os.environ.get(name)
    if value is None:
        raise RuntimeError(f"distributed execution requires torchrun environment variable {name}")
    try:
        return int(value)
    except ValueError as error:
        raise RuntimeError(f"torchrun environment variable {name} must be an integer") from error


def initialize_process_context(parallel: dict, *, allow_gloo: bool = False) -> ProcessContext:
    """Initialize exclusive-device or MPS client processes on one node."""

    if dist.is_initialized():
        raise RuntimeError("distributed process group is already initialized")
    rank = _environment_integer("RANK")
    local_rank = _environment_integer("LOCAL_RANK")
    world_size = _environment_integer("WORLD_SIZE")
    local_world_size = _environment_integer("LOCAL_WORLD_SIZE")
    if local_world_size != world_size:
        raise RuntimeError("distributed evaluation requires every process on one node")
    if not 0 <= rank < world_size or local_rank != rank:
        raise RuntimeError("single-node torchrun ranks are incompatible")
    if parallel["node_count"] != 1:
        raise RuntimeError("distributed evaluation requires node_count 1")
    devices = int(parallel["device_count"])
    per_device = int(parallel["client_processes_per_device"])
    if int(parallel["process_count"]) != world_size or world_size != devices * per_device:
        raise RuntimeError("configured process topology must equal torchrun world size")
    device_index = rank % devices
    device_slot = rank // devices
    backend = str(parallel["control_backend"])
    service = str(parallel["cuda_process_service"])
    use_cuda = backend == "nccl" or service == "mps"
    if use_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("distributed CUDA execution requires CUDA")
        visible_count = torch.cuda.device_count()
        if visible_count != devices:
            raise RuntimeError("visible CUDA device count must equal configured physical device count")
        if not 0 <= device_index < visible_count:
            raise RuntimeError("process rank does not map to a visible CUDA device")
        torch.cuda.set_device(device_index)
        device = torch.device("cuda", device_index)
        properties = torch.cuda.get_device_properties(device)
        device_name = properties.name
        device_total_memory = properties.total_memory
        device_capability = torch.cuda.get_device_capability(device)
    elif backend == "gloo" and allow_gloo:
        visible_count = 0
        device = torch.device("cpu")
        device_name = None
        device_total_memory = None
        device_capability = None
    else:
        raise RuntimeError("scientific execution requires NCCL for exclusive devices or Gloo with CUDA MPS")
    if backend == "nccl":
        if per_device != 1 or service != "none":
            raise RuntimeError("NCCL ranks require exclusive physical CUDA devices")
        control_device = device
    elif backend == "gloo":
        if not allow_gloo and (per_device == 1 or service != "mps"):
            raise RuntimeError("scientific Gloo coordination requires same-device CUDA MPS")
        control_device = torch.device("cpu")
    else:
        raise RuntimeError(f"unsupported distributed control backend: {backend}")
    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(hours=36),
    )
    if dist.get_rank() != rank or dist.get_world_size() != world_size:
        raise RuntimeError("initialized process-group identity is incompatible with torchrun")
    context = ProcessContext(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        backend=backend,
        device=device,
        control_device=control_device,
        host=socket.gethostname(),
        visible_device_count=visible_count,
        physical_device_count=devices,
        device_index=device_index,
        device_slot=device_slot,
        client_processes_per_device=per_device,
        cuda_process_service=service,
        device_name=device_name,
        device_total_memory_bytes=device_total_memory,
        device_compute_capability=device_capability,
    )
    hosts: list[str | None] = [None for _ in range(world_size)]
    dist.all_gather_object(hosts, context.host)
    if len(set(hosts)) != 1:
        raise RuntimeError("distributed evaluation detected processes on more than one host")
    return context


def verify_identity_consensus(
    context: ProcessContext,
    identity: dict,
    process_resident_memory_before_workload_bytes: int | None = None,
) -> list[dict]:
    """Require every process to resolve the same scientific identity."""

    gathered: list[dict | None] = [None for _ in range(context.world_size)]
    dist.all_gather_object(gathered, identity)
    if any(value != identity for value in gathered):
        raise RuntimeError("distributed processes disagree on scientific execution identities")
    local_record = context.record()
    resident_after = local_record["process_resident_memory_bytes"]
    local_record["process_resident_memory_before_workload_bytes"] = (
        process_resident_memory_before_workload_bytes
    )
    local_record["process_resident_memory_after_workload_bytes"] = resident_after
    local_record["workload_resident_memory_delta_bytes"] = (
        None
        if process_resident_memory_before_workload_bytes is None or resident_after is None
        else resident_after - process_resident_memory_before_workload_bytes
    )
    mappings: list[dict | None] = [None for _ in range(context.world_size)]
    dist.all_gather_object(mappings, local_record)
    records = [value for value in mappings if value is not None]
    if len(records) != context.world_size or {value["rank"] for value in records} != set(range(context.world_size)):
        raise RuntimeError("process-to-device mapping is incomplete")
    if len({value["host"] for value in records}) != 1:
        raise RuntimeError("process-to-device mapping spans more than one host")
    expected = {
        value["rank"]: (
            value["rank"] % context.physical_device_count,
            value["rank"] // context.physical_device_count,
        )
        for value in records
    }
    if any((value["device_index"], value["device_slot"]) != expected[value["rank"]] for value in records):
        raise RuntimeError("process-to-device mapping differs from the configured deterministic mapping")
    return sorted(records, key=lambda value: value["rank"])


def close_process_context() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()
