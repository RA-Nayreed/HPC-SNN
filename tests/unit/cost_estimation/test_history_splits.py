import copy

from fedapfa.cost_estimation.history import add_causal_history
from fedapfa.cost_estimation.splits import evaluation_rows, evaluation_settings, grouped_fit_validation


def _row(seed, round_number, client_id, dataset="shd", value=1.0):
    return {
        "dataset": dataset,
        "scientific_seed": seed,
        "communication_round": round_number,
        "selected_position": 0,
        "client_id": client_id,
        "client_wall_time_seconds": value,
        "gross_energy_joules": value * 10,
        "idle_adjusted_energy_joules": value * 5,
        "example_count": 2,
        "layer1_spike_count": value * 2,
        "layer2_spike_count": value * 4,
        "reported_spike_rates": {"layer1": value / 10, "layer2": value / 20},
    }


def test_history_is_causal_and_later_observation_cannot_change_earlier_row():
    rows = [_row(27, 1, "client_00", value=1.0), _row(27, 2, "client_00", value=2.0)]
    first = add_causal_history(rows, 0.3)
    changed = copy.deepcopy(rows)
    changed[1]["client_wall_time_seconds"] = 900.0
    second = add_causal_history(changed, 0.3)
    assert first[0] == second[0]
    assert not first[0]["has_historical_observations"]
    assert first[1]["has_historical_observations"]
    assert first[1]["previous_wall_duration"] == 1.0
    assert first[1]["historical_observation_count"] == 1


def test_grouped_split_excludes_seed_27_and_keeps_clients_separate():
    rows = [
        _row(seed, round_number, f"client_{client:02d}", dataset)
        for dataset in ("shd", "ssc")
        for seed in (7, 17, 27)
        for client in range(20)
        for round_number in (1, 2)
    ]
    fitting, validation = grouped_fit_validation(rows)
    assert {row["scientific_seed"] for row in fitting + validation} == {7, 17}
    fit_clients = {(row["dataset"], row["client_id"]) for row in fitting}
    validation_clients = {(row["dataset"], row["client_id"]) for row in validation}
    assert fit_clients.isdisjoint(validation_clients)
    assert {row["scientific_seed"] for row in evaluation_rows(rows, {"shd", "ssc"})} == {27}
    assert set(evaluation_settings(rows)) == {
        "shd_within_dataset",
        "ssc_within_dataset",
        "joint",
        "shd_to_ssc_transfer",
        "ssc_to_shd_transfer",
        "prequential_seed_27",
    }
