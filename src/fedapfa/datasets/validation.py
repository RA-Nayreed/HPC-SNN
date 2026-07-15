"""Executable validation for official and synthetic event-audio HDF5 files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


@dataclass(frozen=True)
class DatasetExpectation:
    examples: int
    classes: int


OFFICIAL_EXPECTATIONS = {
    "shd_train.h5": DatasetExpectation(8156, 20),
    "shd_test.h5": DatasetExpectation(2264, 20),
    "ssc_train.h5": DatasetExpectation(75466, 35),
    "ssc_valid.h5": DatasetExpectation(9981, 35),
    "ssc_test.h5": DatasetExpectation(20382, 35),
}


class DatasetValidationError(ValueError):
    pass


def validate_hdf5(path: str | Path, expectation: DatasetExpectation | None = None) -> dict[str, int]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    expectation = expectation or OFFICIAL_EXPECTATIONS.get(path.name)
    try:
        with h5py.File(path, "r") as handle:
            if (
                "spikes" not in handle
                or "times" not in handle["spikes"]
                or "units" not in handle["spikes"]
                or "labels" not in handle
            ):
                raise DatasetValidationError(
                    f"{path}: required spikes/times, spikes/units, and labels arrays are missing"
                )
            times, units, labels = handle["spikes/times"], handle["spikes/units"], handle["labels"]
            examples = len(labels)
            if len(times) != examples or len(units) != examples:
                raise DatasetValidationError(f"{path}: event and label example counts differ")
            labels_array = np.asarray(labels[:], dtype=np.int64)
            if expectation and examples != expectation.examples:
                raise DatasetValidationError(f"{path}: expected {expectation.examples} examples, found {examples}")
            classes = expectation.classes if expectation else (int(labels_array.max()) + 1 if examples else 0)
            if examples == 0 or labels_array.min(initial=0) < 0 or labels_array.max(initial=-1) >= classes:
                raise DatasetValidationError(f"{path}: labels are outside [0, {classes - 1}]")
            present_classes = len(np.unique(labels_array))
            if expectation and present_classes != expectation.classes:
                raise DatasetValidationError(f"{path}: expected {expectation.classes} classes, found {present_classes}")
            for index in range(examples):
                event_times = np.asarray(times[index])
                event_units = np.asarray(units[index])
                if len(event_times) != len(event_units):
                    raise DatasetValidationError(f"{path}: sample {index} has mismatched event arrays")
                if np.any(~np.isfinite(event_times)) or np.any(event_times < 0):
                    raise DatasetValidationError(f"{path}: sample {index} has invalid event times")
                if np.any(event_units < 0) or np.any(event_units > 699):
                    raise DatasetValidationError(f"{path}: sample {index} has units outside [0, 699]")
    except OSError as error:
        raise DatasetValidationError(f"{path}: invalid HDF5 file: {error}") from error
    return {"examples": examples, "classes": classes, "raw_channels": 700}
