import csv

import pytest

from fedapfa.measurement.gpu_telemetry import (
    HIERARCHICAL_TELEMETRY_HEADER,
    hierarchical_gpu_utilization_record,
    merge_hierarchical_gpu_telemetry,
)

UUIDS = (
    "11111111-1111-4111-8111-111111111111",
    "22222222-2222-4222-8222-222222222222",
    "33333333-3333-4333-8333-333333333333",
    "44444444-4444-4444-8444-444444444444",
)


def _write_node(path, node_rank, *, uuids=None, indexes=None, reverse=False):
    uuids = UUIDS[node_rank * 2 : (node_rank + 1) * 2] if uuids is None else uuids
    indexes = range(node_rank * 2, (node_rank + 1) * 2) if indexes is None else indexes
    rows = [
        [
            f"2026/07/21 12:00:0{device_index % 2}",
            str(device_index),
            f"GPU-{gpu_uuid.upper()}",
            "NVIDIA GH200 480GB",
            str(20 + device_index),
            str(5 + device_index),
            str(node_rank),
        ]
        for device_index, gpu_uuid in zip(indexes, uuids, strict=True)
    ]
    if reverse:
        rows.reverse()
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(HIERARCHICAL_TELEMETRY_HEADER)
        writer.writerows(rows)


def _valid_files(tmp_path, *, reverse=False):
    paths = [tmp_path / "node-0.csv", tmp_path / "node-1.csv"]
    _write_node(paths[0], 0, reverse=reverse)
    _write_node(paths[1], 1, reverse=not reverse)
    return paths


def _allocation():
    return ",".join(UUIDS)


def test_successful_hierarchical_telemetry_merge_and_utilization(tmp_path):
    paths = _valid_files(tmp_path, reverse=True)
    output = tmp_path / "merged.csv"
    merge_hierarchical_gpu_telemetry(paths, output, _allocation())

    with output.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))
    assert tuple(rows[0]) == HIERARCHICAL_TELEMETRY_HEADER
    assert [int(row[1]) for row in rows[1:]] == [0, 2, 1, 3]
    assert [row[2] for row in rows[1:]] == [
        f"GPU-{UUIDS[index].upper()}" for index in (0, 2, 1, 3)
    ]
    utilization = hierarchical_gpu_utilization_record(paths, _allocation())
    assert utilization["sample_count"] == 4
    assert set(utilization["by_device_index"]) == {"0", "1", "2", "3"}


def test_hierarchical_telemetry_rejects_malformed_rows(tmp_path):
    paths = _valid_files(tmp_path)
    with paths[1].open("a", encoding="utf-8") as stream:
        stream.write("malformed,row\n")
    with pytest.raises(RuntimeError, match="malformed telemetry row"):
        merge_hierarchical_gpu_telemetry(paths, tmp_path / "merged.csv", _allocation())


def test_hierarchical_telemetry_rejects_missing_node_file(tmp_path):
    paths = _valid_files(tmp_path)
    missing = tmp_path / "absent.csv"
    with pytest.raises(RuntimeError, match="node file is missing"):
        merge_hierarchical_gpu_telemetry([paths[0], missing], tmp_path / "merged.csv", _allocation())


def test_hierarchical_telemetry_rejects_duplicate_uuid_across_nodes(tmp_path):
    paths = _valid_files(tmp_path)
    _write_node(paths[1], 1, uuids=(UUIDS[0], UUIDS[3]))
    with pytest.raises(RuntimeError, match="four distinct canonical"):
        merge_hierarchical_gpu_telemetry(paths, tmp_path / "merged.csv", _allocation())


def test_hierarchical_telemetry_rejects_incomplete_device_coverage(tmp_path):
    paths = _valid_files(tmp_path)
    _write_node(paths[1], 1, uuids=(UUIDS[2],), indexes=(2,))
    with pytest.raises(RuntimeError, match="coverage is incomplete"):
        merge_hierarchical_gpu_telemetry(paths, tmp_path / "merged.csv", _allocation())


def test_hierarchical_telemetry_merge_order_is_deterministic(tmp_path):
    first_paths = _valid_files(tmp_path, reverse=False)
    first_output = tmp_path / "first.csv"
    merge_hierarchical_gpu_telemetry(first_paths, first_output, _allocation())

    second_root = tmp_path / "second"
    second_root.mkdir()
    second_paths = _valid_files(second_root, reverse=True)
    second_output = tmp_path / "second.csv"
    merge_hierarchical_gpu_telemetry(second_paths, second_output, _allocation())
    assert first_output.read_bytes() == second_output.read_bytes()
