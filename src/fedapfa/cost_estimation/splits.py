"""Seed-separated and client-grouped fitting collections."""

from __future__ import annotations

import hashlib


def row_identity(row: dict) -> str:
    value = "|".join(
        str(row[name])
        for name in ("dataset", "scientific_seed", "communication_round", "selected_position", "client_id")
    )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def grouped_fit_validation(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    candidates = [value for value in rows if int(value["scientific_seed"]) in {7, 17}]
    fitting = []
    validation = []
    for row in candidates:
        group = hashlib.sha256(f"{row['dataset']}|{row['client_id']}".encode()).digest()[0] % 5
        (validation if group == 0 else fitting).append(row)
    if not fitting or not validation:
        raise ValueError("client-grouped split produced an empty collection")
    fitting_clients = {(value["dataset"], value["client_id"]) for value in fitting}
    validation_clients = {(value["dataset"], value["client_id"]) for value in validation}
    if fitting_clients & validation_clients:
        raise RuntimeError("client-grouped fitting and validation collections overlap")
    return fitting, validation


def evaluation_rows(rows: list[dict], datasets: set[str]) -> list[dict]:
    return [
        value
        for value in rows
        if int(value["scientific_seed"]) == 27 and value["dataset"] in datasets
    ]


def evaluation_settings(rows: list[dict]) -> dict[str, tuple[list[dict], list[dict]]]:
    fit = [value for value in rows if int(value["scientific_seed"]) in {7, 17}]
    return {
        "shd_within_dataset": (
            [value for value in fit if value["dataset"] == "shd"],
            evaluation_rows(rows, {"shd"}),
        ),
        "ssc_within_dataset": (
            [value for value in fit if value["dataset"] == "ssc"],
            evaluation_rows(rows, {"ssc"}),
        ),
        "joint": (fit, evaluation_rows(rows, {"shd", "ssc"})),
        "shd_to_ssc_transfer": (
            [value for value in fit if value["dataset"] == "shd"],
            evaluation_rows(rows, {"ssc"}),
        ),
        "ssc_to_shd_transfer": (
            [value for value in fit if value["dataset"] == "ssc"],
            evaluation_rows(rows, {"shd"}),
        ),
        "prequential_seed_27": (fit, evaluation_rows(rows, {"shd", "ssc"})),
    }
