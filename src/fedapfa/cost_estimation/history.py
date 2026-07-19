"""Causal per-client history features constructed in communication-round order."""

from __future__ import annotations

import copy


def add_causal_history(rows: list[dict], coefficient: float) -> list[dict]:
    if not 0 < coefficient < 1:
        raise ValueError("history coefficient must be between zero and one")
    ordered = sorted(
        (copy.deepcopy(value) for value in rows),
        key=lambda value: (
            value["dataset"],
            value["scientific_seed"],
            value["communication_round"],
            value["selected_position"],
        ),
    )
    state: dict[tuple, dict] = {}
    for row in ordered:
        key = (row["dataset"], int(row["scientific_seed"]), str(row["client_id"]))
        prior = state.get(key)
        row["has_historical_observations"] = prior is not None
        row["historical_observation_count"] = 0 if prior is None else prior["count"]
        for name in (
            "wall_duration",
            "gross_energy",
            "idle_adjusted_energy",
            "layer1_spike_rate",
            "layer2_spike_rate",
            "spikes_per_example",
        ):
            row[f"previous_{name}"] = 0.0 if prior is None else prior[name]
            row[f"missing_previous_{name}"] = prior is None
        for name in ("duration", "gross_energy", "idle_adjusted_energy", "layer1_spike_rate", "layer2_spike_rate"):
            row[f"exponentially_weighted_{name}"] = 0.0 if prior is None else prior[f"ew_{name}"]
            row[f"missing_exponentially_weighted_{name}"] = prior is None
        rates = row.get("reported_spike_rates", {})
        current = {
            "wall_duration": float(row["client_wall_time_seconds"]),
            "gross_energy": float(row["gross_energy_joules"]),
            "idle_adjusted_energy": float(row["idle_adjusted_energy_joules"]),
            "layer1_spike_rate": float(row.get("layer1_spike_rate", rates.get("layer1", 0.0))),
            "layer2_spike_rate": float(row.get("layer2_spike_rate", rates.get("layer2", 0.0))),
            "spikes_per_example": (
                float(row.get("layer1_spike_count", 0.0) + row.get("layer2_spike_count", 0.0))
                / float(row["example_count"])
            ),
        }
        if prior is None:
            next_state = {**current, "count": 1}
            for name, source in (
                ("duration", "wall_duration"),
                ("gross_energy", "gross_energy"),
                ("idle_adjusted_energy", "idle_adjusted_energy"),
                ("layer1_spike_rate", "layer1_spike_rate"),
                ("layer2_spike_rate", "layer2_spike_rate"),
            ):
                next_state[f"ew_{name}"] = current[source]
        else:
            next_state = {**current, "count": prior["count"] + 1}
            for name, source in (
                ("duration", "wall_duration"),
                ("gross_energy", "gross_energy"),
                ("idle_adjusted_energy", "idle_adjusted_energy"),
                ("layer1_spike_rate", "layer1_spike_rate"),
                ("layer2_spike_rate", "layer2_spike_rate"),
            ):
                next_state[f"ew_{name}"] = coefficient * current[source] + (1 - coefficient) * prior[f"ew_{name}"]
        state[key] = next_state
    return ordered
