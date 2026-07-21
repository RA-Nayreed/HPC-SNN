from types import SimpleNamespace

import pytest

from fedapfa.distributed import process_context
from fedapfa.distributed.process_context import (
    allocated_gpu_uuids,
    canonical_gpu_uuid,
    initialize_process_context,
    validate_process_gpu_uuid_mapping,
)

UUIDS = (
    "11111111-1111-4111-8111-111111111111",
    "22222222-2222-4222-8222-222222222222",
    "33333333-3333-4333-8333-333333333333",
    "44444444-4444-4444-8444-444444444444",
)


def _mapping(*, node_count=2, devices_per_node=2, prefixed=True):
    records = []
    for rank, value in enumerate(UUIDS):
        local_rank = rank % devices_per_node
        raw_uuid = f"GPU-{value}" if prefixed else value
        records.append(
            {
                "rank": rank,
                "node_rank": rank // devices_per_node,
                "local_rank": local_rank,
                "device_index": local_rank,
                "gpu_uuid": raw_uuid,
            }
        )
    assert node_count * devices_per_node == len(records)
    return records


@pytest.mark.parametrize(
    "value",
    [
        f"GPU-{UUIDS[0]}",
        UUIDS[0],
        f"gpu-{UUIDS[0].upper()}",
        f"  GPU-{UUIDS[0].upper()}  ",
    ],
)
def test_gpu_uuid_canonicalization_equates_prefix_case_and_whitespace(value):
    assert canonical_gpu_uuid(value) == UUIDS[0]


@pytest.mark.parametrize(
    "value,match",
    [
        ("", "empty"),
        ("   ", "empty"),
        ("GPU-not-a-uuid", "malformed"),
        (f"MIG-GPU-{UUIDS[0]}/1/2", "MIG"),
        (f"mig-{UUIDS[0]}", "MIG"),
    ],
)
def test_gpu_uuid_canonicalization_rejects_empty_malformed_and_mig(value, match):
    with pytest.raises(ValueError, match=match):
        canonical_gpu_uuid(value)


def test_allocated_uuid_parser_detects_duplicates_only_visible_after_canonicalization():
    with pytest.raises(RuntimeError, match="duplicate"):
        allocated_gpu_uuids(
            f"GPU-{UUIDS[0]},{UUIDS[0]},GPU-{UUIDS[2]},GPU-{UUIDS[3]}",
            expected_count=4,
        )


@pytest.mark.parametrize(
    "value,match",
    [
        (None, "must enumerate"),
        (f"GPU-{UUIDS[0]},GPU-{UUIDS[1]},GPU-{UUIDS[2]}", "exactly 4"),
        (f"GPU-{UUIDS[0]},GPU-{UUIDS[1]},,GPU-{UUIDS[3]}", "empty"),
        (f"GPU-{UUIDS[0]},GPU-{UUIDS[1]},MIG-{UUIDS[2]},GPU-{UUIDS[3]}", "MIG"),
    ],
)
def test_allocated_uuid_parser_rejects_incomplete_or_invalid_evidence(value, match):
    with pytest.raises(RuntimeError, match=match):
        allocated_gpu_uuids(value, expected_count=4)


def test_valid_one_node_four_gpu_mapping_canonicalizes_process_records():
    allocation = tuple(f"GPU-{value.upper()}" for value in UUIDS)
    observed = validate_process_gpu_uuid_mapping(
        list(reversed(_mapping(node_count=1, devices_per_node=4, prefixed=False))),
        allocation,
        world_size=4,
        node_count=1,
        devices_per_node=4,
        local_world_size=4,
    )
    assert [value["gpu_uuid"] for value in observed] == list(UUIDS)
    assert [value["gpu_uuid_raw"] for value in observed] == list(UUIDS)


def test_valid_two_node_two_gpu_per_node_mapping_accepts_prefixed_allocation():
    allocation = allocated_gpu_uuids(
        ",".join(f" GPU-{value.upper()} " for value in UUIDS),
        expected_count=4,
    )
    observed = validate_process_gpu_uuid_mapping(
        list(reversed(_mapping())),
        allocation,
        world_size=4,
        node_count=2,
        devices_per_node=2,
        local_world_size=2,
    )
    assert [value["gpu_uuid"] for value in observed] == list(UUIDS)
    assert all(value["gpu_uuid_raw"].startswith("GPU-") for value in observed)


def test_process_mapping_rejects_missing_unexpected_duplicate_and_wrong_device():
    allocation = tuple(UUIDS)
    wrong_set = _mapping()
    wrong_set[3]["gpu_uuid"] = "55555555-5555-4555-8555-555555555555"
    with pytest.raises(RuntimeError, match="missing=.*unexpected="):
        validate_process_gpu_uuid_mapping(
            wrong_set,
            allocation,
            world_size=4,
            node_count=2,
            devices_per_node=2,
            local_world_size=2,
        )
    duplicate = _mapping()
    duplicate[3]["gpu_uuid"] = UUIDS[2]
    with pytest.raises(RuntimeError, match="same GPU UUID"):
        validate_process_gpu_uuid_mapping(
            duplicate,
            allocation,
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
            allocation,
            world_size=4,
            node_count=2,
            devices_per_node=2,
            local_world_size=2,
        )


def test_uuid_mismatch_after_process_group_initialization_cleans_every_group(monkeypatch):
    initialized = {"value": False}
    destroyed = []
    groups = []

    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")
    monkeypatch.setenv("FEDAPFA_ALLOCATED_GPU_UUIDS", UUIDS[1])
    monkeypatch.setattr(process_context.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(process_context.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(process_context.torch.cuda, "set_device", lambda _device: None)
    monkeypatch.setattr(
        process_context.torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(
            name="GH200",
            total_memory=96_000_000_000,
            uuid=f"GPU-{UUIDS[0]}",
        ),
    )
    monkeypatch.setattr(process_context.torch.cuda, "get_device_capability", lambda _device: (9, 0))
    monkeypatch.setattr(process_context.dist, "is_initialized", lambda: initialized["value"])

    def init_process_group(**_kwargs):
        initialized["value"] = True

    def new_group(*, ranks):
        group = SimpleNamespace(ranks=tuple(ranks))
        groups.append(group)
        return group

    def all_gather_object(output, value):
        for index in range(len(output)):
            output[index] = value

    def destroy_process_group(group=None):
        destroyed.append(group)
        if group is None:
            initialized["value"] = False

    monkeypatch.setattr(process_context.dist, "init_process_group", init_process_group)
    monkeypatch.setattr(process_context.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(process_context.dist, "get_world_size", lambda: 1)
    monkeypatch.setattr(process_context.dist, "new_group", new_group)
    monkeypatch.setattr(process_context.dist, "all_gather_object", all_gather_object)
    monkeypatch.setattr(process_context.dist, "destroy_process_group", destroy_process_group)
    process_context._PROCESS_GROUPS.clear()

    parallel = {
        "node_count": 1,
        "device_count": 1,
        "devices_per_node": 1,
        "client_processes_per_device": 1,
        "process_count": 1,
        "control_backend": "nccl",
        "cuda_process_service": "none",
    }
    with pytest.raises(RuntimeError, match="exact allocated GPU UUID set"):
        initialize_process_context(parallel)

    assert initialized["value"] is False
    assert process_context._PROCESS_GROUPS == {}
    assert {id(value) for value in destroyed if value is not None} == {id(value) for value in groups}
    assert destroyed[-1] is None
