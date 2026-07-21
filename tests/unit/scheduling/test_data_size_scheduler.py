from fedapfa.scheduling.assignment import assign_selected_clients
from fedapfa.scheduling.data_size_scheduler import example_count_longest_processing_time


def test_example_count_longest_processing_time_balances_deterministically():
    clients = ["a", "b", "c", "d"]
    assignments, loads = example_count_longest_processing_time(clients, 2, {"d": 5, "b": 7, "a": 8, "c": 6})
    assert [value.process_rank for value in assignments] == [0, 1, 1, 0]
    assert loads == {"0": 13.0, "1": 13.0}


def test_event_structure_ties_and_empty_process_workloads_are_stable():
    clients = ["client_b", "client_a"]
    first, loads = assign_selected_clients(
        clients,
        4,
        "event_structure_longest_processing_time",
        {"client_a": 3.0, "client_b": 3.0},
        cost_source="frozen_event_structure_wall_time_prediction",
    )
    second, _ = assign_selected_clients(
        clients,
        4,
        "event_structure_longest_processing_time",
        dict(reversed(list({"client_a": 3.0, "client_b": 3.0}.items()))),
        cost_source="frozen_event_structure_wall_time_prediction",
    )
    assert [value.process_rank for value in first] == [0, 1]
    assert first == second
    assert loads == {"0": 3.0, "1": 3.0, "2": 0.0, "3": 0.0}
