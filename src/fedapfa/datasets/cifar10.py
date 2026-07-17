"""Lazy CIFAR-10 access with deterministic transforms and strict test isolation."""

from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from fedapfa.datasets.centralized_split import stratified_split
from fedapfa.datasets.dirichlet_partition import DirichletPartition, label_dirichlet_partition
from fedapfa.datasets.fedsnn_partition import (
    fedsnn_balanced_label_dirichlet_partition,
    fedsnn_random_iid_partition,
)
from fedapfa.federated.randomness import resolved_seeds
from fedapfa.utilities.serialization import sha256_json


class CIFAR10DependencyError(RuntimeError):
    """Raised when CIFAR-10 support is requested without torchvision."""


class CIFAR10AccessError(RuntimeError):
    """Raised when CIFAR-10 files or split access violate the protocol."""


def normalize_cifar10_tensor(tensor: torch.Tensor, normalization: str) -> torch.Tensor:
    """Apply one explicitly selected CIFAR-10 value representation."""

    if not tensor.is_floating_point():
        raise TypeError("CIFAR-10 normalization requires a floating tensor")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError("CIFAR-10 input contains NaN or infinity")
    tolerance = 8 * torch.finfo(tensor.dtype).eps
    if bool(torch.any(tensor < -tolerance)) or bool(torch.any(tensor > 1 + tolerance)):
        raise ValueError("CIFAR-10 input before normalization must be in [0, 1]")
    if normalization == "scale_0_1":
        result = tensor
        lower, upper = 0.0, 1.0
    elif normalization == "signed_minus_one_one":
        result = tensor.mul(2.0).sub(1.0)
        lower, upper = -1.0, 1.0
    else:
        raise ValueError(f"unsupported CIFAR-10 normalization: {normalization}")
    if not bool(torch.isfinite(result).all()):
        raise ValueError("normalized CIFAR-10 tensor contains NaN or infinity")
    if bool(torch.any(result < lower - tolerance)) or bool(torch.any(result > upper + tolerance)):
        raise ValueError(f"normalized CIFAR-10 tensor must remain in [{lower:g}, {upper:g}]")
    return result


def _torchvision():
    try:
        from torchvision import datasets, transforms
        from torchvision.transforms import functional
    except (ImportError, RuntimeError) as error:
        raise CIFAR10DependencyError(
            "CIFAR-10 support requires torchvision from the Roihu python-pytorch/2.10 module "
            "or a compatible local installation"
        ) from error
    return datasets, transforms, functional


def _file_identity(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return {"name": path.name, "size_bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def _training_files(root: Path) -> list[Path]:
    base = root / "cifar-10-batches-py"
    return [base / f"data_batch_{index}" for index in range(1, 6)] + [base / "batches.meta"]


def cifar10_training_identity(root: str | Path) -> dict:
    """Hash only the extracted standard training batches and class metadata."""

    paths = _training_files(Path(root))
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise CIFAR10AccessError(
            "CIFAR-10 training files are unavailable; run "
            f"fedapfa-download-data cifar10 --root {Path(root).parent}. Missing: {missing}"
        )
    records = [_file_identity(path) for path in paths]
    return {
        "name": "cifar10",
        "source": "torchvision.CIFAR10",
        "standard_training_examples": 50000,
        "files": records,
        "sha256": sha256_json(records),
    }


class CIFAR10TrainingData(Dataset):
    """Standard CIFAR-10 training batches loaded without touching the test batch."""

    def __init__(self, root: str | Path):
        root = Path(root)
        cifar10_training_identity(root)
        data_parts = []
        targets: list[int] = []
        for path in _training_files(root)[:5]:
            try:
                with path.open("rb") as handle:
                    value = pickle.load(handle, encoding="latin1")
            except (OSError, pickle.UnpicklingError) as error:
                raise CIFAR10AccessError(f"cannot read standard CIFAR-10 training batch {path}: {error}") from error
            data = value.get("data", value.get(b"data"))
            labels = value.get("labels", value.get(b"labels"))
            if data is None or labels is None:
                raise CIFAR10AccessError(f"CIFAR-10 training batch has incompatible fields: {path}")
            data_parts.append(np.asarray(data, dtype=np.uint8).reshape(-1, 3, 32, 32))
            targets.extend(int(label) for label in labels)
        self.data = np.concatenate(data_parts, axis=0)
        self.targets = targets
        if self.data.shape != (50000, 3, 32, 32) or len(self.targets) != 50000:
            raise CIFAR10AccessError(f"expected 50000 CIFAR-10 training examples, found {len(self.targets)}")

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int):
        return torch.from_numpy(self.data[index].copy()).to(torch.float32).div_(255), self.targets[index]


class CIFAR10IndexedDataset(Dataset):
    """Indexed view with configuration-driven deterministic augmentation."""

    fedapfa_batch_kind = "image"

    def __init__(self, base, indices, transform_config: dict, augment: bool):
        self.base = base
        self.indices = np.asarray(indices, dtype=np.int64)
        self.transform_config = transform_config
        self.augment = augment

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item):
        _, _, functional = _torchvision()
        image, label = self.base[int(self.indices[item])]
        tensor = image if isinstance(image, torch.Tensor) else functional.to_tensor(image)
        tensor = tensor.to(torch.float32)
        if tensor.numel() and float(tensor.max()) > 1:
            tensor = tensor / 255
        augmentation = self.transform_config["augmentation"]
        if self.augment and augmentation["random_crop"]:
            padding = int(augmentation["crop_padding"])
            tensor = functional.pad(tensor, [padding] * 4, fill=0)
            size = int(self.transform_config["image_size"])
            maximum = 2 * padding
            top = int(torch.randint(maximum + 1, (1,)).item())
            left = int(torch.randint(maximum + 1, (1,)).item())
            tensor = functional.crop(tensor, top, left, size, size)
        if self.augment and augmentation["horizontal_flip"]:
            if float(torch.rand(())) < float(augmentation["horizontal_flip_probability"]):
                tensor = functional.hflip(tensor)
        tensor = normalize_cifar10_tensor(tensor, self.transform_config["normalization"])
        return tensor, int(label)


@dataclass
class FederatedCIFAR10Bundle:
    root: Path
    config: dict
    labels: np.ndarray
    train_indices: np.ndarray
    validation_indices: np.ndarray
    validation_dataset: CIFAR10IndexedDataset | None
    partition: DirichletPartition
    split_artifact: dict
    resolved_seed_values: dict[str, int]
    training_base: CIFAR10TrainingData
    official_test_access_count: int = 0
    official_test_identity: dict | None = None

    def client_dataset(self, client_id: str) -> CIFAR10IndexedDataset:
        return CIFAR10IndexedDataset(
            self.training_base,
            self.partition.client_indices[client_id],
            self.config["dataset"]["transforms"],
            augment=True,
        )

    def official_test_dataset(self, model_selected: bool) -> CIFAR10IndexedDataset:
        if not model_selected:
            raise CIFAR10AccessError("official CIFAR-10 test access is prohibited before model selection")
        if self.official_test_access_count:
            raise CIFAR10AccessError("official CIFAR-10 test evaluation is permitted exactly once")
        datasets, _, _ = _torchvision()
        try:
            base = datasets.CIFAR10(root=str(self.root), train=False, download=False, transform=None)
        except RuntimeError as error:
            raise CIFAR10AccessError(
                f"CIFAR-10 test files are unavailable; run fedapfa-download-data cifar10 --root {self.root.parent}"
            ) from error
        test_batch = self.root / "cifar-10-batches-py" / "test_batch"
        test_file_identity = _file_identity(test_batch)
        self.official_test_identity = {
            "standard_test_examples": len(base),
            "files": [test_file_identity],
            "sha256": sha256_json([test_file_identity]),
        }
        self.official_test_access_count += 1
        return CIFAR10IndexedDataset(
            base,
            np.arange(len(base), dtype=np.int64),
            self.config["dataset"]["transforms"],
            augment=False,
        )


def prepare_federated_cifar10(config: dict) -> FederatedCIFAR10Bundle:
    """Prepare the standard training split without opening the official test batch."""

    _torchvision()
    root = Path(config["dataset"]["root"])
    base = CIFAR10TrainingData(root)
    labels = np.asarray(base.targets, dtype=np.int64)
    seeds = resolved_seeds(config)
    train_indices, validation_indices = stratified_split(
        labels, config["dataset"]["validation_fraction"], seeds["split"]
    )
    dataset_identity = cifar10_training_identity(root)
    split_core = {
        "schema_version": 1,
        "split_seed": seeds["split"],
        "validation_fraction": config["dataset"]["validation_fraction"],
        "dataset_identity": dataset_identity,
        "training_indices": [int(value) for value in train_indices],
        "validation_indices": [int(value) for value in validation_indices],
    }
    split_artifact = dict(split_core)
    split_artifact["split_id"] = sha256_json(split_core)
    partition_config = config["federated"]["partition"]
    common_partition = {
        "labels": labels,
        "eligible_indices": train_indices,
        "clients": config["federated"]["clients"],
        "minimum_size": partition_config["minimum_examples_per_client"],
        "seed": seeds["partition"],
        "validation_split_id": split_artifact["split_id"],
        "dataset_identity": dataset_identity,
    }
    if partition_config["method"] == "fedsnn_balanced_label_dirichlet":
        partition = fedsnn_balanced_label_dirichlet_partition(
            **common_partition,
            alpha=partition_config["alpha"],
            maximum_attempts=partition_config["maximum_attempts"],
        )
    elif partition_config["method"] == "fedsnn_random_iid":
        partition = fedsnn_random_iid_partition(**common_partition)
    elif partition_config["method"] == "label_dirichlet":
        partition = label_dirichlet_partition(
            **common_partition,
            alpha=partition_config["alpha"],
            maximum_attempts=partition_config["maximum_attempts"],
        )
    else:
        raise ValueError(f"unsupported CIFAR-10 partition method: {partition_config['method']}")
    validation = (
        CIFAR10IndexedDataset(base, validation_indices, config["dataset"]["transforms"], augment=False)
        if len(validation_indices)
        else None
    )
    return FederatedCIFAR10Bundle(
        root=root,
        config=config,
        labels=labels,
        train_indices=train_indices,
        validation_indices=validation_indices,
        validation_dataset=validation,
        partition=partition,
        split_artifact=split_artifact,
        resolved_seed_values=seeds,
        training_base=base,
    )
