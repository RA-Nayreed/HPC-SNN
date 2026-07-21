import pytest

from fedapfa.distributed.process_context import (
    allocated_gpu_uuids,
    validate_process_gpu_uuid_mapping,
)


def _mapping():
    return [
        {
            "rank": rank,
            "node_rank": rank // 2,
            "local_rank": rank % 2,
            "device_index": rank % 2,
            "gpu_uuid": f"GPU-{rank}",
        }
        for rank in range(4)
    ]


def test_exact_four_distinct_allocated_uuids_and_process_mapping():
    uuids = allocated_gpu_uuids("GPU-0,GPU-1,GPU-2,GPU-3", expected_count=4)
    assert validate_process_gpu_uuid_mapping(
        list(reversed(_mapping())),
        uuids,
        world_size=4,
        node_count=2,
        devices_per_node=2,
        local_world_size=2,
    ) == _mapping()


@pytest.mark.parametrize(
    "value,match",
    [
        (None, "must enumerate"),
        ("GPU-0,GPU-1,GPU-2", "exactly 4"),
        ("GPU-0,GPU-1,,GPU-3", "exactly 4"),
        ("GPU-0,GPU-1,GPU-1,GPU-3", "duplicate"),
    ],
)
def test_allocated_uuid_parser_rejects_incomplete_or_duplicate_evidence(value, match):
    with pytest.raises(RuntimeError, match=match):
        allocated_gpu_uuids(value, expected_count=4)


def test_process_mapping_rejects_wrong_uuid_set_duplicate_or_rank_device_mapping():
    uuids = allocated_gpu_uuids("GPU-0,GPU-1,GPU-2,GPU-3", expected_count=4)
    wrong_set = _mapping()
    wrong_set[3]["gpu_uuid"] = "GPU-other"
    with pytest.raises(RuntimeError, match="exact allocated"):
        validate_process_gpu_uuid_mapping(
            wrong_set,
            uuids,
            world_size=4,
            node_count=2,
            devices_per_node=2,
            local_world_size=2,
        )
    duplicate = _mapping()
    duplicate[3]["gpu_uuid"] = "GPU-2"
    with pytest.raises(RuntimeError, match="same GPU UUID"):
        validate_process_gpu_uuid_mapping(
            duplicate,
            uuids,
            world_size=4,
            node_count=2,
            devices_per_node=2,
            local_world_size=2,
        )
    wrong_device = _mapping()
    wrong_device[2]["device_index"] = 1
    with pytest.raises(RuntimeError, match="node-major"):
        validate_process_gpu_uuid_mapping(
            wrong_device,
            uuids,
            world_size=4,
            node_count=2,
            devices_per_node=2,
            local_world_size=2,
        )
