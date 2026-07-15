import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset

from fedapfa.training.centralized import train_centralized
from fedapfa.training.protocols import DatasetBundle
from fedapfa.utilities.run_records import initialize_run


class SyntheticEvents(Dataset):
    def __init__(self, count=32, guard=None):
        self.count = count
        self.guard = guard

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        if self.guard is not None:
            assert self.guard.exists(), "test data accessed before model selection"
        label = index % 2
        value = torch.zeros(2, 140)
        value[:, label] = 1
        return value, label


class LinearEventModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(140, 20)
        self.model_metadata = {"class": type(self).__name__}

    def forward(self, inputs, lengths):
        return self.linear(inputs.sum(1)), {
            "layer1": torch.zeros((), device=inputs.device),
            "layer2": torch.zeros((), device=inputs.device),
        }


def config(tmp_path, mode="memorization_validation"):
    return {
        "name": "synthetic-" + mode,
        "seed": 7,
        "mode": mode,
        "protocol": "memorization_validation" if mode == "memorization_validation" else "reduced_sample_evaluation",
        "device": "cpu",
        "output_root": str(tmp_path),
        "dataset": {
            "name": "shd",
            "root": "unused",
            "train_file": "unused",
            "test_file": "unused",
            "classes": 20,
            "raw_channels": 700,
            "input_features": 140,
            "frequency_bin_factor": 5,
            "temporal_bin_ms": 10.0,
            "validation_fraction": 0.1,
        },
        "model": {
            "name": "lif_2layer",
            "hidden_dims": [8, 8],
            "dropout": 0.0,
            "batch_normalization": False,
            "bias": True,
            "neuron": {
                "name": "euler_lif",
                "tau_ms": 20.0,
                "threshold": 1.0,
                "reset": "subtract",
                "detach_reset": True,
                "surrogate": {"name": "atan", "alpha": 5.0},
            },
            "attention": {"variant": "none", "lambda": 0.01},
        },
        "training": {
            "optimizer": "adam",
            "learning_rate": 0.1,
            "delay_lr_multiplier": 100.0,
            "weight_decay": 0.0,
            "batch_size": 8,
            "epochs": 30,
            "gradient_clip": 1.0,
            "max_train_batches": 2 if mode == "reduced_sample_evaluation" else None,
            "max_validation_batches": 1 if mode == "reduced_sample_evaluation" else None,
            "max_test_batches": None,
            "target_accuracy": 0.95 if mode == "memorization_validation" else None,
            "early_stop_patience": None,
            "data_loader_workers": 0,
            "persistent_workers": False,
        },
        "subset": {"train_examples": 32, "validation_examples": 0, "test_examples": 0, "stratified": True},
        "resume": None,
    }


def test_synthetic_memorization_validation_and_artifacts(tmp_path):
    cfg = config(tmp_path)
    data = SyntheticEvents()
    bundle = DatasetBundle(
        data,
        data,
        None,
        {"train": list(range(32))},
        {"protocol": "memorization_validation", "official_test_accessed": False, "scientific_result": False},
    )
    run = initialize_run(cfg, bundle.selected_indices, "synthetic command")
    result = train_centralized(LinearEventModel(), bundle, cfg, run)
    assert result["accepted"] and result["best_selection_accuracy"] >= 0.95
    required = [
        "resolved_config.yaml",
        "command.txt",
        "environment.json",
        "git.json",
        "metrics.jsonl",
        "training.log",
        "checkpoints/last.pt",
        "checkpoints/best_validation.pt",
        "final_metrics.json",
        "selected_indices.json",
        "acceptance.json",
    ]
    assert all((run / name).stat().st_size > 0 for name in required)


def test_reduced_sample_limits_and_test_waits_for_selection(tmp_path):
    cfg = config(tmp_path, "reduced_sample_evaluation")
    data = SyntheticEvents(64)
    run_path = Path(cfg["output_root"])
    placeholder = DatasetBundle(
        data,
        data,
        None,
        {},
        {"protocol": "reduced_sample_evaluation", "official_test_accessed": False, "scientific_result": False},
    )
    run = initialize_run(cfg, {}, "reduced-sample command")
    result = train_centralized(LinearEventModel(), placeholder, cfg, run)
    record = json.loads((run / "metrics.jsonl").read_text().splitlines()[0])
    assert record["train"]["batches"] == 2 and record["validation"]["batches"] == 1 and result["accepted"]
    scientific = config(run_path / "scientific", "reduced_sample_evaluation")
    scientific["mode"] = "scientific_evaluation"
    scientific["protocol"] = "independent_evaluation"
    scientific["training"]["epochs"] = 1
    scientific["training"]["max_train_batches"] = 1
    scientific["training"]["max_validation_batches"] = 1
    run2 = initialize_run(scientific, {}, "scientific command")
    guarded = SyntheticEvents(4, run2 / "checkpoints/best_validation.pt")
    bundle = DatasetBundle(
        data,
        data,
        guarded,
        {},
        {"protocol": "independent_evaluation", "official_test_accessed": True, "scientific_result": True},
    )
    train_centralized(LinearEventModel(), bundle, scientific, run2)
