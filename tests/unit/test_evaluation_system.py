import copy
from collections import Counter

import pytest

from fedapfa.configuration import (
    RESOLVED_PAIRED_INVARIANT_PATHS,
    ConfigurationError,
    evaluation_execution_identity,
    evaluation_scientific_identity,
    load_evaluation_allocations,
    load_evaluation_config,
    load_evaluation_manifest,
    validate_collection_path_disjointness,
    validate_evaluation_config,
    validate_resolved_evaluation_pair,
)

SCHEDULING = "experiments/scheduling_evaluation/manifest.yaml"
HIERARCHY = "experiments/hierarchical_reduction_evaluation/manifest.yaml"


def test_evaluation_manifests_have_exact_matrices_orders_and_disjoint_paths():
    scheduling = load_evaluation_manifest(SCHEDULING)
    hierarchy = load_evaluation_manifest(HIERARCHY)
    assert len(scheduling) == 18
    assert len(hierarchy) == 12
    assert set(Counter((value.dataset, value.config["scheduler"]["strategy"]) for value in scheduling).values()) == {3}
    assert set(
        Counter((value.dataset, value.config["aggregation_execution"]["topology"]) for value in hierarchy).values()
    ) == {3}
    assert [value.execution_order for value in load_evaluation_allocations(SCHEDULING)] == [
        ("round_robin", "example_count_longest_processing_time", "event_structure_longest_processing_time"),
        ("example_count_longest_processing_time", "event_structure_longest_processing_time", "round_robin"),
        ("event_structure_longest_processing_time", "round_robin", "example_count_longest_processing_time"),
    ] * 2
    assert [value.execution_order for value in load_evaluation_allocations(HIERARCHY)] == [
        ("flat_ordered", "node_hierarchical"),
        ("node_hierarchical", "flat_ordered"),
        ("flat_ordered", "node_hierarchical"),
    ] * 2
    validate_collection_path_disjointness(SCHEDULING, HIERARCHY)


def test_treatments_share_science_but_resume_identity_contains_scheduler_and_topology():
    scheduling = load_evaluation_manifest(SCHEDULING)
    shd = [value.config for value in scheduling if value.dataset == "shd" and value.seed == 37]
    assert len({repr(evaluation_scientific_identity(value)) for value in shd}) == 1
    assert len({repr(evaluation_execution_identity(value)) for value in shd}) == 3


@pytest.mark.parametrize(
    ("manifest", "collection", "expected_differences"),
    [
        (
            SCHEDULING,
            "scheduling_evaluation",
            {
                "metadata.experiment",
                "name",
                "parallel_execution.client_assignment",
                "scheduler.strategy",
            },
        ),
        (
            HIERARCHY,
            "hierarchical_reduction_evaluation",
            {
                "aggregation_execution.topology",
                "metadata.experiment",
                "name",
                "parallel_execution.aggregation_topology",
            },
        ),
    ],
)
def test_resolved_pair_whitelist_proves_every_named_invariant(manifest, collection, expected_differences):
    tasks = load_evaluation_manifest(manifest)
    for dataset in ("shd", "ssc"):
        paired = [task for task in tasks if task.dataset == dataset and task.seed == 37]
        reference_name = paired[0].config["evaluation"]["comparison_reference"]
        reference = next(task.config for task in paired if task.experiment == reference_name)
        for candidate in paired:
            record = validate_resolved_evaluation_pair(reference, candidate.config, collection)
            assert set(record["invariants_equal"]) == set(RESOLVED_PAIRED_INVARIANT_PATHS)
            assert all(record["invariants_equal"].values())
            if candidate.config is not reference:
                assert set(record["observed_difference_paths"]) == expected_differences


def test_manifest_rejects_resolved_scientific_drift_after_yaml_composition(monkeypatch):
    from fedapfa.configuration import evaluation_system

    original = evaluation_system.load_resolved_config

    def drift(path):
        config = original(path)
        if "example_count_longest_processing_time" in str(path):
            config["training"]["learning_rate"] *= 2
        return config

    monkeypatch.setattr(evaluation_system, "load_resolved_config", drift)
    with pytest.raises(ConfigurationError, match="outside the whitelist"):
        load_evaluation_manifest(SCHEDULING)


def test_general_validation_command_accepts_evaluation_configuration(capsys, monkeypatch):
    from fedapfa.cli.validate_run import main

    config = "experiments/scheduling_evaluation/shd/lif_fedavg_round_robin.yaml"
    monkeypatch.setattr("sys.argv", ["fedapfa-validate-config", config])
    main()
    assert "shd_lif_fedavg_round_robin-seed7-" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("scheduler", "strategy", "oracle", "unsupported scheduler"),
        ("aggregation_execution", "topology", "tree", "unsupported aggregation"),
        ("parallel_execution", "process_count", 3, "inconsistent"),
        ("parallel_execution", "cuda_process_service", "mps", "without CUDA MPS"),
        ("parallel_execution", "node_count", 2, "inconsistent"),
    ],
)
def test_configuration_rejects_scheduler_aggregation_and_topology_drift(section, key, value, message):
    config = load_evaluation_config("experiments/scheduling_evaluation/shd/lif_fedavg_round_robin.yaml")
    changed = copy.deepcopy(config)
    changed[section][key] = value
    with pytest.raises(ConfigurationError, match=message):
        validate_evaluation_config(changed)
