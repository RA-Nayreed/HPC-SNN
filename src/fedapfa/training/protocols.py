"""Dataset selection semantics for centralized evaluation protocols."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import h5py
from torch.utils.data import Dataset

from fedapfa.configuration.validation import resolve_dataset_paths
from fedapfa.datasets.centralized_split import select_subset, stratified_split
from fedapfa.datasets.shd import EventAudioDataset


@dataclass
class DatasetBundle:
    train: Dataset
    validation: Dataset | None
    test: Dataset | Callable[[], Dataset] | None
    selected_indices: dict[str, list[int]]
    metadata: dict[str, object]


def _labels(path: Path):
    with h5py.File(path, "r") as handle:
        return handle["labels"][:]


def _dataset(path, indices, config, validate=True):
    dataset = config["dataset"]
    return EventAudioDataset(
        path,
        indices,
        temporal_bin_ms=dataset["temporal_bin_ms"],
        frequency_bin_factor=dataset["frequency_bin_factor"],
        validate=validate,
    )


def prepare_datasets(config) -> DatasetBundle:
    paths = resolve_dataset_paths(config)
    dataset = config["dataset"]
    subset = config["subset"]
    seed = config["seed"]
    protocol = config["protocol"]
    train_labels = _labels(paths["train"])
    if protocol == "tiny_overfit":
        chosen = select_subset(train_labels, subset["train_examples"], seed, subset["stratified"])
        tiny = _dataset(paths["train"], chosen, config)
        return DatasetBundle(
            tiny,
            tiny,
            None,
            {"train": chosen.tolist()},
            {"protocol": protocol, "official_test_accessed": False, "scientific_result": False},
        )
    if dataset["name"] == "shd" and protocol in {"thesis_clean", "smoke"}:
        train_indices, validation_indices = stratified_split(train_labels, dataset["validation_fraction"], seed)
    else:
        train_indices = range(len(train_labels))
        validation_indices = None
    if subset["train_examples"]:
        local = select_subset(train_labels[list(train_indices)], subset["train_examples"], seed, subset["stratified"])
        train_indices = list(train_indices)
        train_indices = [train_indices[i] for i in local]
    train = _dataset(paths["train"], train_indices, config)
    selected = {"train": [int(i) for i in train_indices]}
    if dataset["name"] == "shd" and protocol in {"thesis_clean", "smoke"}:
        if subset["validation_examples"]:
            local = select_subset(
                train_labels[validation_indices], subset["validation_examples"], seed + 1, subset["stratified"]
            )
            validation_indices = validation_indices[local]
        validation = _dataset(paths["train"], validation_indices, config, validate=False)
        selected["validation"] = [int(i) for i in validation_indices]
    else:
        validation_path = paths.get("validation")
        validation = None if validation_path is None else _dataset(validation_path, None, config)
        if validation is not None and subset["validation_examples"]:
            chosen = select_subset(validation.labels(), subset["validation_examples"], seed + 1, subset["stratified"])
            validation = _dataset(validation_path, chosen, config, validate=False)
            selected["validation"] = chosen.tolist()
    test = None
    if protocol in {"thesis_clean", "paper_compatible"} and config["mode"] == "full":

        def create_test_dataset():
            test_indices = None
            if subset["test_examples"]:
                test_labels = _labels(paths["test"])
                test_indices = select_subset(test_labels, subset["test_examples"], seed + 2, subset["stratified"])
                selected["test"] = test_indices.tolist()
            return _dataset(paths["test"], test_indices, config)

        test = create_test_dataset
    return DatasetBundle(
        train,
        validation,
        test,
        selected,
        {
            "protocol": protocol,
            "official_test_monitored_during_training": protocol == "paper_compatible",
            "official_test_accessed": protocol == "paper_compatible",
            "scientific_result": config["mode"] == "full",
            "metric_label": "reproduction" if protocol == "paper_compatible" else "clean",
        },
    )
