import pytest

from fedapfa.cost_estimation.assignment import (
    assignment_loads,
    evaluate_round_assignments,
    longest_first_assignment,
    round_robin_assignment,
)


def _rows():
    return [
        {
            "dataset": "shd",
            "scientific_seed": 27,
            "communication_round": 1,
            "selected_position": index,
            "client_wall_time_seconds": float(index + 1),
            "example_count": 20 - index,
        }
        for index in range(10)
    ]


@pytest.mark.parametrize("process_count", [2, 4])
def test_assignments_are_deterministic_complete_and_report_regret(process_count):
    costs = [float(index + 1) for index in range(10)]
    assert round_robin_assignment(costs, process_count) == round_robin_assignment(costs, process_count)
    assignment = longest_first_assignment(costs, process_count)
    assert len(assignment) == 10 and all(0 <= value < process_count for value in assignment)
    assert sum(assignment_loads(assignment, costs, process_count)) == sum(costs)
    records = evaluate_round_assignments(_rows(), list(reversed(costs)), process_count)
    assert {value["strategy"] for value in records} == {
        "round_robin",
        "example_count_longest_first",
        "predicted_cost_longest_first",
        "measured_cost_oracle",
    }
    oracle = next(value for value in records if value["strategy"] == "measured_cost_oracle")
    assert oracle["makespan_regret_seconds"] == pytest.approx(0.0)
    assert all(value["offline_evaluation_only"] for value in records)


def test_assignment_has_no_external_side_effects():
    rows = _rows()
    before = [dict(value) for value in rows]
    evaluate_round_assignments(rows, [float(index + 1) for index in range(10)], 2)
    assert rows == before
