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
    if protocol == "memorization_validation":
        chosen = select_subset(train_labels, subset["train_examples"], seed, subset["stratified"])
        memorization_dataset = _dataset(paths["train"], chosen, config)
        return DatasetBundle(
            memorization_dataset,
            memorization_dataset,
            None,
            {"train": chosen.tolist()},
            {
                "protocol": protocol,
                "official_test_accessed": False,
                "official_test_monitored_during_training": False,
                "official_test_evaluated_after_model_selection": False,
                "scientific_result": False,
                "complete_training_data_used": False,
                "complete_dataset_used": False,
            },
        )

    if dataset["name"] == "shd" and protocol in {"independent_evaluation", "reduced_sample_evaluation"}:
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

    if dataset["name"] == "shd" and protocol in {"independent_evaluation", "reduced_sample_evaluation"}:
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

    no_training_subset = subset["train_examples"] == 0 and subset["validation_examples"] == 0
    if dataset["name"] == "shd" and protocol in {"independent_evaluation", "reduced_sample_evaluation"}:
        complete_training_data_used = (
            no_training_subset and validation is not None and len(train) + len(validation) == len(train_labels)
        )
        selection_split = "derived_train_validation"
    else:
        complete_training_data_used = no_training_subset and len(train) == len(train_labels) and validation is not None
        selection_split = "official_test" if protocol == "published_protocol" else "official_validation"

    test = None
    if protocol in {"independent_evaluation", "published_protocol"} and config["mode"] == "scientific_evaluation":

        def create_test_dataset():
            test_indices = None
            if subset["test_examples"]:
                test_labels = _labels(paths["test"])
                test_indices = select_subset(test_labels, subset["test_examples"], seed + 2, subset["stratified"])
                selected["test"] = test_indices.tolist()
            return _dataset(paths["test"], test_indices, config)

        test = create_test_dataset

    published = protocol == "published_protocol"
    return DatasetBundle(
        train,
        validation,
        test,
        selected,
        {
            "protocol": protocol,
            "selection_split": selection_split,
            "official_test_monitored_during_training": published,
            "official_test_accessed": published,
            "official_test_evaluated_after_model_selection": False,
            "official_test_role": "selection_and_final_test" if published else "final_test_only",
            "official_test_examples": None,
            "scientific_result": config["mode"] == "scientific_evaluation",
            "metric_label": "reproduction" if published else "independent_evaluation",
            "source_train_examples": len(train_labels),
            "train_examples": len(train),
            "validation_examples": len(validation) if validation is not None else 0,
            "complete_training_data_used": complete_training_data_used,
            "complete_dataset_used": False,
        },
    )
