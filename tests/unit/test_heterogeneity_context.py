import json

import h5py
import numpy as np
import torch

from fedapfa.cli import derive_heterogeneity_context as context_module
from fedapfa.configuration.scientific_manifests import ContextTask
from fedapfa.federated.round_state import EvaluationResult


def test_context_derivation_never_accesses_official_test_split(tmp_path, monkeypatch):
    run = tmp_path / "historical"
    run.mkdir()
    train = tmp_path / "data" / "shd_train.h5"
    train.parent.mkdir()
    with h5py.File(train, "w") as handle:
        handle.create_dataset("labels", data=np.asarray([0, 1, 0, 1, 0, 1]))
    source_final = {
        "best_validation_accuracy": 0.8,
        "test": {"accuracy": 0.7},
        "selected_round": 90,
        "final_validation_accuracy": 0.75,
        "logical_communication": {"cumulative_total_bytes": 1000},
        "execution_time_seconds": 10.0,
        "mean_client_update_l2_norm": 0.2,
        "mean_client_spike_rates": {"layer1": 0.1},
    }
    (run / "final_metrics.json").write_text(json.dumps(source_final), encoding="utf-8")
    (run / "round_metrics.jsonl").write_text(
        "".join(
            json.dumps({"round_number": value, "validation_accuracy": value / 100}) + "\n" for value in range(1, 101)
        ),
        encoding="utf-8",
    )
    config = {
        "seed": 7,
        "protocol": "independent_evaluation",
        "dataset": {
            "name": "shd",
            "train_file": "shd_train.h5",
            "test_file": "must_not_be_opened.h5",
            "temporal_bin_ms": 10.0,
            "frequency_bin_factor": 5,
            "classes": 2,
        },
        "model": {"name": "fixture"},
        "federated": {
            "rounds": 100,
            "clients": 2,
            "clients_per_round": 1,
            "participation_fraction": 0.5,
            "local_epochs": 1,
            "local_batch_size": 2,
            "optimizer": "adam",
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "gradient_clip": 1.0,
            "client_sampling": "without_replacement",
            "partition": {"method": "label_dirichlet", "alpha": 0.5},
        },
    }
    split = {
        "split_id": "split",
        "dataset_identity": {"name": "shd_train.h5", "sha256": "fixture"},
        "training_indices": [0, 1, 2, 3],
        "validation_indices": [4, 5],
    }
    partition = {
        "partition_id": "partition",
        "clients": [
            {"client_id": "client_00", "indices": [0, 1], "class_counts": {"0": 1, "1": 1}},
            {"client_id": "client_01", "indices": [2, 3], "class_counts": {"0": 1, "1": 1}},
        ],
    }
    initialization = {"model_initialization_id": "initialization"}
    model = torch.nn.Linear(1, 2)
    checkpoint = {
        "model_class": "AudioLIFSNN",
        "git_commit": "commit",
        "split_id": "split",
        "partition_id": "partition",
        "model_initialization_id": "initialization",
        "global_model_state": model.state_dict(),
    }
    task = ContextTask(
        7,
        "shd_lif_dirichlet_alpha_0_5_participation_0_50",
        {"run_directory": str(run), "official_test_accuracy": 0.7},
    )
    accessed = []
    real_dataset = context_module.EventAudioDataset

    def guarded_dataset(path, *args, **kwargs):
        accessed.append(str(path))
        assert path == train
        return real_dataset(path, *args, **kwargs)

    monkeypatch.setattr(context_module, "_locate_run", lambda root, value: run)
    monkeypatch.setattr(
        context_module,
        "_verify_source",
        lambda run_dir, value: (config, split, partition, initialization, "commit"),
    )
    monkeypatch.setattr(context_module, "file_identity", lambda path: split["dataset_identity"])
    monkeypatch.setattr(context_module, "EventAudioDataset", guarded_dataset)
    monkeypatch.setattr(context_module, "make_model", lambda value: model)
    monkeypatch.setattr(context_module.torch, "load", lambda *args, **kwargs: checkpoint)
    monkeypatch.setattr(
        context_module,
        "evaluate_model",
        lambda *args, **kwargs: EvaluationResult(
            loss=0.5,
            accuracy=0.75,
            examples=2,
            batches=1,
            spike_rates={"layer1": 0.1},
            confusion_matrix=[[1, 0], [1, 0]],
            per_class_accuracy=[1.0, 0.0],
            macro_f1=1 / 3,
            peak_cuda_memory_bytes=None,
        ),
    )
    record = context_module.derive_context_record(task, tmp_path, train.parent, tmp_path / "output")
    assert accessed == [str(train)]
    assert record["official_test_reevaluated"] is False
    assert record["update_alignment"]["available"] is False
