"""Deterministic offline assignment comparisons with no training side effects."""

from __future__ import annotations

import time


def _validate(costs: list[float], process_count: int) -> None:
    if process_count not in {2, 4}:
        raise ValueError("offline assignment process count must be two or four")
    if not costs or any(value < 0 for value in costs):
        raise ValueError("assignment costs must be non-negative")


def round_robin_assignment(costs: list[float], process_count: int) -> list[int]:
    _validate(costs, process_count)
    return [index % process_count for index in range(len(costs))]


def longest_first_assignment(costs: list[float], process_count: int) -> list[int]:
    _validate(costs, process_count)
    assignments = [-1] * len(costs)
    loads = [0.0] * process_count
    counts = [0] * process_count
    for index in sorted(range(len(costs)), key=lambda value: (-costs[value], value)):
        process = min(range(process_count), key=lambda value: (loads[value], counts[value], value))
        assignments[index] = process
        loads[process] += costs[index]
        counts[process] += 1
    return assignments


def assignment_loads(assignments: list[int], measured_costs: list[float], process_count: int) -> list[float]:
    if len(assignments) != len(measured_costs) or sorted(set(assignments)) != list(range(process_count)):
        raise ValueError("assignment must map clients across every requested process")
    loads = [0.0] * process_count
    for process, cost in zip(assignments, measured_costs, strict=True):
        if not 0 <= process < process_count:
            raise ValueError("assignment process index is invalid")
        loads[process] += cost
    return loads


def evaluate_round_assignments(
    rows: list[dict], predicted_costs: list[float], process_count: int
) -> list[dict]:
    if len(rows) != len(predicted_costs):
        raise ValueError("predicted assignment costs differ from row count")
    grouped: dict[tuple, list[tuple[dict, float]]] = {}
    for row, predicted in zip(rows, predicted_costs, strict=True):
        key = (row["dataset"], int(row["scientific_seed"]), int(row["communication_round"]))
        grouped.setdefault(key, []).append((row, float(predicted)))
    records = []
    for key, values in sorted(grouped.items()):
        values.sort(key=lambda value: int(value[0]["selected_position"]))
        measured = [float(value[0]["client_wall_time_seconds"]) for value in values]
        predicted = [value[1] for value in values]
        examples = [float(value[0]["example_count"]) for value in values]
        strategy_inputs = {
            "round_robin": measured,
            "example_count_longest_first": examples,
            "predicted_cost_longest_first": predicted,
            "measured_cost_oracle": measured,
        }
        strategies = {}
        assignment_times = {}
        for strategy, costs in strategy_inputs.items():
            started = time.perf_counter_ns()
            strategies[strategy] = (
                round_robin_assignment(costs, process_count)
                if strategy == "round_robin"
                else longest_first_assignment(costs, process_count)
            )
            assignment_times[strategy] = (time.perf_counter_ns() - started) / 1_000_000_000
        oracle_loads = assignment_loads(strategies["measured_cost_oracle"], measured, process_count)
        oracle_makespan = max(oracle_loads)
        for strategy, assignments in strategies.items():
            predicted_loads = assignment_loads(assignments, predicted, process_count)
            measured_loads = assignment_loads(assignments, measured, process_count)
            makespan = max(measured_loads)
            records.append(
                {
                    "dataset": key[0],
                    "scientific_seed": key[1],
                    "communication_round": key[2],
                    "process_count": process_count,
                    "strategy": strategy,
                    "assignments": assignments,
                    "predicted_process_loads_seconds": predicted_loads,
                    "measured_process_loads_seconds": measured_loads,
                    "predicted_assignment_makespan_seconds": max(predicted_loads),
                    "measured_assignment_makespan_seconds": makespan,
                    "measured_oracle_makespan_seconds": oracle_makespan,
                    "makespan_regret_seconds": makespan - oracle_makespan,
                    "load_imbalance": (
                        (makespan - min(measured_loads)) / makespan if makespan else 0.0
                    ),
                    "assignment_computation_time_seconds": assignment_times[strategy],
                    "offline_evaluation_only": True,
                }
            )
    return records
