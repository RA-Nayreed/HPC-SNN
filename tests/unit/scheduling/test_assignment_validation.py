import pytest

from fedapfa.scheduling.assignment import assign_selected_clients, validate_assignments


def test_every_selected_client_is_assigned_once_and_restored_to_selected_order():
    clients = ["c3", "c1", "c2", "c0"]
    assignments, _ = assign_selected_clients(
        clients,
        3,
        "example_count_longest_processing_time",
        {"c0": 1, "c1": 4, "c2": 2, "c3": 8},
        cost_source="training_example_count",
    )
    validate_assignments(assignments, clients, 3)
    assert [value.client_id for value in assignments] == clients
    assert len({value.client_id for value in assignments}) == len(clients)


def test_assignment_rejects_duplicates_missing_costs_and_nonfinite_costs():
    with pytest.raises(ValueError, match="exactly once"):
        assign_selected_clients(["a", "a"], 2, "round_robin", cost_source="position")
    with pytest.raises(ValueError, match="exactly"):
        assign_selected_clients(
            ["a", "b"],
            2,
            "example_count_longest_processing_time",
            {"a": 1},
            cost_source="training_example_count",
        )
    with pytest.raises(ValueError, match="finite"):
        assign_selected_clients(
            ["a"],
            2,
            "event_structure_longest_processing_time",
            {"a": float("nan")},
            cost_source="prediction",
        )
