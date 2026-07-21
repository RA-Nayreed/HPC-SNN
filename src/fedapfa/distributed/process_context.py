"""Process-group initialization and node-major identity verification."""

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
    node_rank: int
    local_world_size: int
    node_count: int
    devices_per_node: int
    node_leader_rank: int
    node_group_ranks: tuple[int, ...]
    node_leader_ranks: tuple[int, ...]
    gpu_uuid: str | None

    @property
    def is_coordinator(self) -> bool:
        return self.rank == 0

    def record(self) -> dict:
        value = asdict(self)
        value["device"] = str(self.device)
        value["control_device"] = str(self.control_device)
        if self.device_compute_capability is not None:
            value["device_compute_capability"] = list(self.device_compute_capability)
        value["node_group_ranks"] = list(self.node_group_ranks)
        value["node_leader_ranks"] = list(self.node_leader_ranks)
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


def allocated_gpu_uuids(value: str | None, *, expected_count: int) -> tuple[str, ...]:
    """Parse an exact, nonempty, distinct allocation-level GPU UUID set."""

    if value is None:
        raise RuntimeError("FEDAPFA_ALLOCATED_GPU_UUIDS must enumerate the allocated GPUs")
    values = tuple(item.strip() for item in value.split(","))
    if len(values) != expected_count or any(not item for item in values):
        raise RuntimeError(f"allocated GPU UUID evidence must contain exactly {expected_count} values")
    if len(set(values)) != expected_count:
        raise RuntimeError("allocated GPU UUID evidence contains duplicate devices")
    return values


def validate_process_gpu_uuid_mapping(
    records: list[dict],
    allocation_uuids: tuple[str, ...],
    *,
    world_size: int,
    node_count: int,
    devices_per_node: int,
    local_world_size: int,
) -> list[dict]:
    """Validate the exact exclusive rank-to-device-to-UUID allocation mapping."""

    if len(allocation_uuids) != world_size or len(set(allocation_uuids)) != world_size:
        raise RuntimeError("exclusive NCCL execution requires one distinct allocated GPU UUID per rank")
    if len(records) != world_size or {value.get("rank") for value in records} != set(range(world_size)):
        raise RuntimeError("process-to-GPU UUID mapping is incomplete")
    ordered = sorted(records, key=lambda value: value["rank"])
    for value in ordered:
        rank = value["rank"]
        expected_local_rank = rank % local_world_size
        if (
            value.get("node_rank") != rank // local_world_size
            or value.get("local_rank") != expected_local_rank
            or value.get("device_index") != expected_local_rank
        ):
            raise RuntimeError("process-to-GPU UUID mapping differs from node-major exclusive-device mapping")
    observed = [value.get("gpu_uuid") for value in ordered]
    if any(not isinstance(value, str) or not value for value in observed):
        raise RuntimeError("every NCCL process must report a nonempty GPU UUID")
    if len(set(observed)) != world_size:
        raise RuntimeError("multiple NCCL processes map to the same GPU UUID")
    if set(observed) != set(allocation_uuids):
        raise RuntimeError("process GPU UUIDs do not match the exact allocated GPU UUID set")
    if node_count * devices_per_node != world_size:
        raise RuntimeError("GPU UUID mapping topology does not cover the configured physical devices")
    return ordered


def initialize_process_context(parallel: dict, *, allow_gloo: bool = False) -> ProcessContext:
    """Initialize exclusive-device or MPS processes with stable node-major ranks."""

    if dist.is_initialized():
        raise RuntimeError("distributed process group is already initialized")
    rank = _environment_integer("RANK")
    local_rank = _environment_integer("LOCAL_RANK")
    world_size = _environment_integer("WORLD_SIZE")
    local_world_size = _environment_integer("LOCAL_WORLD_SIZE")
    if not 0 <= rank < world_size or not 0 <= local_rank < local_world_size:
        raise RuntimeError("torchrun rank environment is incompatible")
    nodes = int(parallel["node_count"])
    devices = int(parallel["device_count"])
    devices_per_node = int(parallel.get("devices_per_node", devices))
    per_device = int(parallel["client_processes_per_device"])
    expected_local_world = devices_per_node * per_device
    if (
        int(parallel["process_count"]) != world_size
        or world_size != devices * per_device
        or devices != nodes * devices_per_node
        or local_world_size != expected_local_world
    ):
        raise RuntimeError("configured process topology must equal torchrun world size")
    node_rank = rank // local_world_size
    if node_rank >= nodes or rank != node_rank * local_world_size + local_rank:
        raise RuntimeError("torchrun ranks do not follow node-major local-rank mapping")
    device_index = local_rank % devices_per_node
    device_slot = local_rank // devices_per_node
    backend = str(parallel["control_backend"])
    service = str(parallel["cuda_process_service"])
    use_cuda = backend == "nccl" or service == "mps"
    if use_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("distributed CUDA execution requires CUDA")
        visible_count = torch.cuda.device_count()
        if visible_count != devices_per_node:
            raise RuntimeError("visible CUDA device count must equal configured devices per node")
        if not 0 <= device_index < visible_count:
            raise RuntimeError("process rank does not map to a visible CUDA device")
        torch.cuda.set_device(device_index)
        device = torch.device("cuda", device_index)
        properties = torch.cuda.get_device_properties(device)
        device_name = properties.name
        device_total_memory = properties.total_memory
        device_capability = torch.cuda.get_device_capability(device)
        gpu_uuid = getattr(properties, "uuid", None)
        if gpu_uuid is not None:
            gpu_uuid = str(gpu_uuid)
    elif backend == "gloo" and allow_gloo:
        visible_count = 0
        device = torch.device("cpu")
        device_name = None
        device_total_memory = None
        device_capability = None
        gpu_uuid = None
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
    node_group_ranks = tuple(range(node_rank * local_world_size, (node_rank + 1) * local_world_size))
    node_leader_ranks = tuple(value * local_world_size for value in range(nodes))
    for group_node_rank in range(nodes):
        ranks = tuple(
            range(
                group_node_rank * local_world_size,
                (group_node_rank + 1) * local_world_size,
            )
        )
        _PROCESS_GROUPS[("node", ranks)] = dist.new_group(ranks=list(ranks))
    _PROCESS_GROUPS[("leaders", node_leader_ranks)] = dist.new_group(ranks=list(node_leader_ranks))
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
        node_rank=node_rank,
        local_world_size=local_world_size,
        node_count=nodes,
        devices_per_node=devices_per_node,
        node_leader_rank=node_group_ranks[0],
        node_group_ranks=node_group_ranks,
        node_leader_ranks=node_leader_ranks,
        gpu_uuid=gpu_uuid,
    )
    hosts: list[str | None] = [None for _ in range(world_size)]
    dist.all_gather_object(hosts, context.host)
    host_values = [str(value) for value in hosts]
    synthetic_same_host_nodes = allow_gloo and nodes > 1 and len(set(host_values)) == 1
    if len(set(host_values)) != nodes and not synthetic_same_host_nodes:
        raise RuntimeError("distributed execution host count differs from configured node count")
    for group_node_rank in range(nodes):
        first = group_node_rank * local_world_size
        if len(set(host_values[first : first + local_world_size])) != 1:
            raise RuntimeError("one configured node rank spans multiple hostnames")
    if backend == "nccl":
        local_mapping = {
            "rank": context.rank,
            "node_rank": context.node_rank,
            "local_rank": context.local_rank,
            "device_index": context.device_index,
            "gpu_uuid": context.gpu_uuid,
        }
        gathered_mappings: list[dict | None] = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_mappings, local_mapping)
        validate_process_gpu_uuid_mapping(
            [value for value in gathered_mappings if value is not None],
            allocated_gpu_uuids(os.environ.get("FEDAPFA_ALLOCATED_GPU_UUIDS"), expected_count=devices),
            world_size=world_size,
            node_count=nodes,
            devices_per_node=devices_per_node,
            local_world_size=local_world_size,
        )
    return context


_PROCESS_GROUPS: dict[tuple[str, tuple[int, ...]], dist.ProcessGroup] = {}


def node_process_group(context: ProcessContext) -> dist.ProcessGroup:
    return _PROCESS_GROUPS[("node", context.node_group_ranks)]


def node_leader_process_group(context: ProcessContext) -> dist.ProcessGroup:
    return _PROCESS_GROUPS[("leaders", context.node_leader_ranks)]


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
    local_record["process_resident_memory_before_workload_bytes"] = process_resident_memory_before_workload_bytes
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
    host_count = len({value["host"] for value in records})
    synthetic_same_host_nodes = context.backend == "gloo" and context.node_count > 1 and host_count == 1
    if host_count != context.node_count and not synthetic_same_host_nodes:
        raise RuntimeError("process-to-device mapping has an incompatible host count")
    expected = {
        value["rank"]: (
            value["local_rank"] % context.devices_per_node,
            value["local_rank"] // context.devices_per_node,
        )
        for value in records
    }
    if any((value["device_index"], value["device_slot"]) != expected[value["rank"]] for value in records):
        raise RuntimeError("process-to-device mapping differs from the configured deterministic mapping")
    return sorted(records, key=lambda value: value["rank"])


def close_process_context() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()
    _PROCESS_GROUPS.clear()
