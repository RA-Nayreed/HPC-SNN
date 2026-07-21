"""Strict validation and deterministic merging for node-local GPU telemetry."""

from __future__ import annotations

import csv
import io
import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from fedapfa.distributed.process_context import allocated_gpu_uuids, canonical_gpu_uuid
from fedapfa.utilities.serialization import atomic_write_text

HIERARCHICAL_TELEMETRY_HEADER = (
    "timestamp",
    "index",
    "uuid",
    "name",
    "utilization.gpu",
    "utilization.memory",
    "node_rank",
)


@dataclass(frozen=True)
class GpuTelemetryRow:
    timestamp: str
    device_index: int
    gpu_uuid_raw: str
    gpu_uuid: str
    device_name: str
    gpu_utilization_text: str
    gpu_utilization: float
    memory_utilization_text: str
    memory_utilization: float
    node_rank: int
    source_line: int

    def csv_values(self) -> tuple[str, ...]:
        return (
            self.timestamp,
            str(self.device_index),
            self.gpu_uuid_raw,
            self.device_name,
            self.gpu_utilization_text,
            self.memory_utilization_text,
            str(self.node_rank),
        )


def _percentage(value: str, *, path: Path, line_number: int, field: str) -> tuple[str, float]:
    text = value.strip()
    try:
        number = float(text)
    except ValueError as error:
        raise RuntimeError(f"{path}:{line_number} has malformed {field}") from error
    if not math.isfinite(number) or not 0.0 <= number <= 100.0:
        raise RuntimeError(f"{path}:{line_number} has invalid {field}")
    return text, number


def _read_node_telemetry(
    path: str | Path,
    *,
    node_rank: int,
    devices_per_node: int,
) -> list[GpuTelemetryRow]:
    source = Path(path)
    if not source.is_file():
        raise RuntimeError(f"hierarchical telemetry node file is missing: {source}")
    with source.open(encoding="utf-8", newline="") as stream:
        values = list(csv.reader(stream))
    if not values or tuple(values[0]) != HIERARCHICAL_TELEMETRY_HEADER:
        raise RuntimeError(f"hierarchical telemetry header is invalid: {source}")
    rows: list[GpuTelemetryRow] = []
    for line_number, value in enumerate(values[1:], start=2):
        if len(value) != len(HIERARCHICAL_TELEMETRY_HEADER):
            raise RuntimeError(f"{source}:{line_number} has a malformed telemetry row")
        timestamp, index_text, raw_uuid, device_name, gpu_text, memory_text, row_node_text = (
            item.strip() for item in value
        )
        if not timestamp or not device_name:
            raise RuntimeError(f"{source}:{line_number} has an empty timestamp or device name")
        try:
            device_index = int(index_text)
            row_node_rank = int(row_node_text)
            gpu_uuid = canonical_gpu_uuid(raw_uuid)
        except ValueError as error:
            raise RuntimeError(f"{source}:{line_number} has an invalid device identity") from error
        if row_node_rank != node_rank:
            raise RuntimeError(f"{source}:{line_number} has an unexpected node rank")
        expected_indexes = set(
            range(node_rank * devices_per_node, (node_rank + 1) * devices_per_node)
        )
        if device_index not in expected_indexes:
            raise RuntimeError(f"{source}:{line_number} has an unexpected global device index")
        gpu_text, gpu_utilization = _percentage(
            gpu_text,
            path=source,
            line_number=line_number,
            field="GPU utilization",
        )
        memory_text, memory_utilization = _percentage(
            memory_text,
            path=source,
            line_number=line_number,
            field="memory utilization",
        )
        rows.append(
            GpuTelemetryRow(
                timestamp=timestamp,
                device_index=device_index,
                gpu_uuid_raw=raw_uuid,
                gpu_uuid=gpu_uuid,
                device_name=device_name,
                gpu_utilization_text=gpu_text,
                gpu_utilization=gpu_utilization,
                memory_utilization_text=memory_text,
                memory_utilization=memory_utilization,
                node_rank=node_rank,
                source_line=line_number,
            )
        )
    if not rows:
        raise RuntimeError(f"hierarchical telemetry contains no samples: {source}")
    index_to_uuids: dict[int, set[str]] = {}
    uuid_to_indexes: dict[str, set[int]] = {}
    for row in rows:
        index_to_uuids.setdefault(row.device_index, set()).add(row.gpu_uuid)
        uuid_to_indexes.setdefault(row.gpu_uuid, set()).add(row.device_index)
    expected_indexes = set(range(node_rank * devices_per_node, (node_rank + 1) * devices_per_node))
    if set(index_to_uuids) != expected_indexes:
        raise RuntimeError(f"hierarchical telemetry device coverage is incomplete: {source}")
    if any(len(values) != 1 for values in index_to_uuids.values()) or any(
        len(values) != 1 for values in uuid_to_indexes.values()
    ):
        raise RuntimeError(f"hierarchical telemetry device identities are inconsistent: {source}")
    if len(uuid_to_indexes) != devices_per_node:
        raise RuntimeError(f"hierarchical telemetry must contain distinct GPUs per node: {source}")
    return rows


def validate_hierarchical_gpu_telemetry(
    node_files: Sequence[str | Path],
    allocated_uuid_text: str,
    *,
    node_count: int = 2,
    devices_per_node: int = 2,
) -> list[GpuTelemetryRow]:
    """Validate complete node files and return rows in deterministic merge order."""

    if len(node_files) != node_count:
        raise RuntimeError(f"hierarchical telemetry requires exactly {node_count} node files")
    expected_count = node_count * devices_per_node
    allocation_uuids = allocated_gpu_uuids(allocated_uuid_text, expected_count=expected_count)
    rows = [
        row
        for node_rank, path in enumerate(node_files)
        for row in _read_node_telemetry(
            path,
            node_rank=node_rank,
            devices_per_node=devices_per_node,
        )
    ]
    observed_uuids = {row.gpu_uuid for row in rows}
    if len(observed_uuids) != expected_count:
        raise RuntimeError("hierarchical telemetry does not contain four distinct canonical GPU UUIDs")
    if observed_uuids != set(allocation_uuids):
        missing = sorted(set(allocation_uuids) - observed_uuids)
        unexpected = sorted(observed_uuids - set(allocation_uuids))
        raise RuntimeError(
            "hierarchical telemetry UUID coverage differs from the allocation: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return sorted(
        rows,
        key=lambda row: (
            row.timestamp,
            row.device_index,
            row.node_rank,
            row.gpu_uuid,
            row.device_name,
            row.gpu_utilization_text,
            row.memory_utilization_text,
            row.source_line,
        ),
    )


def hierarchical_gpu_utilization_record(
    node_files: Sequence[str | Path],
    allocated_uuid_text: str,
) -> dict:
    rows = validate_hierarchical_gpu_telemetry(node_files, allocated_uuid_text)
    by_device: dict[str, list[float]] = {}
    for row in rows:
        by_device.setdefault(str(row.device_index), []).append(row.gpu_utilization)
    values = [value for device_values in by_device.values() for value in device_values]
    return {
        "source": "validated node-local nvidia-smi physical-device samples",
        "sample_count": len(values),
        "mean_percent": statistics.fmean(values),
        "minimum_percent": min(values),
        "maximum_percent": max(values),
        "by_device_index": {
            device: {
                "sample_count": len(device_values),
                "mean_percent": statistics.fmean(device_values),
                "minimum_percent": min(device_values),
                "maximum_percent": max(device_values),
            }
            for device, device_values in sorted(by_device.items(), key=lambda item: int(item[0]))
        },
    }


def merge_hierarchical_gpu_telemetry(
    node_files: Sequence[str | Path],
    output_path: str | Path,
    allocated_uuid_text: str,
) -> None:
    rows = validate_hierarchical_gpu_telemetry(node_files, allocated_uuid_text)
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(HIERARCHICAL_TELEMETRY_HEADER)
    writer.writerows(row.csv_values() for row in rows)
    atomic_write_text(output_path, stream.getvalue())
