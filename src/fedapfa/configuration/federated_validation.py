"""Strict validation for federated scientific protocols."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from .validation import ConfigurationError

REQUIRED_STREAMS = {
    "split",
    "partition",
    "model_initialization",
    "client_selection",
    "client_training",
    "validation",
    "final_test",
}
PARTITION_METHODS = {
    "label_dirichlet",
    "stratified_iid",
    "fedsnn_random_iid",
    "fedsnn_balanced_label_dirichlet",
}
CIFAR_PROTOCOLS = {"paper_reported_evaluation"}
PAPER_EXPERIMENTS = {
    "cifar10_fedsnn_paper_reported_iid_evaluation": ("fedsnn_random_iid", None, 0.7644),
    "cifar10_fedsnn_paper_reported_noniid_evaluation": (
        "fedsnn_balanced_label_dirichlet",
        0.5,
        0.7394,
    ),
}
PAPER_DISTRIBUTED_EXPERIMENTS = {
    "cifar10_svgg9_bntt_noniid_1_gpu": "cifar10_fedsnn_paper_reported_noniid_evaluation",
    "cifar10_svgg9_bntt_noniid_2_gpu": "cifar10_fedsnn_paper_reported_noniid_evaluation",
}


def _paper_experiment_name(config: Mapping[str, Any]) -> str:
    name = str(config.get("name", ""))
    return PAPER_DISTRIBUTED_EXPERIMENTS.get(name, name)


def _section(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{key} must be a mapping")
    return value


def _positive_integer(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigurationError(f"{label} must be a positive integer")
    return value


def _finite_positive(value: Any, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ConfigurationError(f"{label} must be finite and positive")
    return float(value)


def _nonnegative(value: Any, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
        raise ConfigurationError(f"{label} must be finite and nonnegative")
    return float(value)


def _validate_dataset_and_model(config: Mapping[str, Any]) -> None:
    dataset = _section(config, "dataset")
    model = _section(config, "model")
    acceptance = _section(config, "acceptance")
    dataset_name = dataset.get("name")
    model_name = model.get("name")
    if (dataset_name, model_name) == ("shd", "lif_2layer"):
        expected_dataset = {
            "train_file": "shd_train.h5",
            "test_file": "shd_test.h5",
            "validation_file": None,
            "classes": 20,
            "raw_channels": 700,
            "input_features": 140,
            "frequency_bin_factor": 5,
            "temporal_bin_ms": 10.0,
            "validation_fraction": 0.1,
        }
        for key, expected in expected_dataset.items():
            if dataset.get(key) != expected:
                raise ConfigurationError(f"dataset.{key} must be {expected!r}")
        if model.get("hidden_dims") != [256, 256]:
            raise ConfigurationError("SHD federated evaluation requires hidden_dims [256, 256]")
        if (
            model.get("dropout") != 0.4
            or model.get("batch_normalization") is not False
            or model.get("bias") is not True
        ):
            raise ConfigurationError("SHD federated evaluation requires dropout 0.4, no batch normalization, and bias")
        attention = _section(model, "attention")
        if attention.get("variant") != "none":
            raise ConfigurationError("SHD federated evaluation requires no PfA attention; variant must be none")
        neuron = _section(model, "neuron")
        surrogate = _section(neuron, "surrogate")
        expected_neuron = {
            "name": "spikingjelly_lif",
            "tau_ms": 10.05,
            "threshold": 1.0,
            "reset": "subtract",
            "detach_reset": True,
        }
        for key, expected in expected_neuron.items():
            if neuron.get(key) != expected:
                raise ConfigurationError(f"model.neuron.{key} must be {expected!r}")
        if surrogate.get("name") != "atan" or surrogate.get("alpha") != 5.0:
            raise ConfigurationError("SHD federated evaluation requires ATan surrogate alpha 5")
        if acceptance.get("expected_model_class") != "AudioLIFSNN":
            raise ConfigurationError("acceptance.expected_model_class must be AudioLIFSNN")
    elif (dataset_name, model_name) == ("ssc", "lif_2layer_128"):
        expected_dataset = {
            "train_file": "ssc_train.h5",
            "validation_file": "ssc_valid.h5",
            "test_file": "ssc_test.h5",
            "classes": 35,
            "raw_channels": 700,
            "input_features": 140,
            "frequency_bin_factor": 5,
            "temporal_bin_ms": 10.0,
        }
        for key, expected in expected_dataset.items():
            if dataset.get(key) != expected:
                raise ConfigurationError(f"dataset.{key} must be {expected!r}")
        if model.get("hidden_dims") != [128, 128]:
            raise ConfigurationError("SSC federated evaluation requires hidden_dims [128, 128]")
        if (
            model.get("dropout") != 0.4
            or model.get("batch_normalization") is not False
            or model.get("bias") is not True
        ):
            raise ConfigurationError("SSC federated evaluation requires dropout 0.4, no batch normalization, and bias")
        attention = _section(model, "attention")
        if attention.get("variant") != "none":
            raise ConfigurationError("SSC federated evaluation requires attention variant none")
        neuron = _section(model, "neuron")
        surrogate = _section(neuron, "surrogate")
        expected_neuron = {
            "name": "spikingjelly_lif",
            "tau_ms": 10.05,
            "threshold": 1.0,
            "reset": "subtract",
            "detach_reset": True,
        }
        for key, expected in expected_neuron.items():
            if neuron.get(key) != expected:
                raise ConfigurationError(f"model.neuron.{key} must be {expected!r}")
        if surrogate.get("name") != "atan" or surrogate.get("alpha") != 5.0:
            raise ConfigurationError("SSC federated evaluation requires ATan surrogate alpha 5")
        if acceptance.get("expected_model_class") != "AudioLIFSNN":
            raise ConfigurationError("acceptance.expected_model_class must be AudioLIFSNN")
    elif (dataset_name, model_name) == ("cifar10", "svgg9_bntt"):
        expected_dataset = {
            "classes": 10,
            "channels": 3,
            "image_size": 32,
            "validation_fraction": 0.0,
            "standard_train_split": True,
            "standard_test_split": True,
            "download_during_training": False,
        }
        for key, expected in expected_dataset.items():
            if dataset.get(key) != expected:
                raise ConfigurationError(f"dataset.{key} must be {expected!r}")
        transforms = _section(dataset, "transforms")
        if transforms.get("image_size") != 32 or transforms.get("normalization") != "signed_minus_one_one":
            raise ConfigurationError("paper-reported CIFAR-10 transforms require 32x32 signed_minus_one_one inputs")
        augmentation = _section(transforms, "augmentation")
        if (
            augmentation.get("random_crop") is not False
            or augmentation.get("crop_padding") != 0
            or augmentation.get("horizontal_flip") is not False
            or augmentation.get("horizontal_flip_probability") != 0.0
        ):
            raise ConfigurationError("paper-reported CIFAR-10 source configurations disable crop and flip")
        if model.get("channels") != [64, 64, 128, 128, 256, 256, 256]:
            raise ConfigurationError("S-VGG9 BNTT channel layout is incompatible")
        if model.get("average_pool_after_convolution") != [2, 4, 7] or model.get("linear_hidden") != 1024:
            raise ConfigurationError("S-VGG9 BNTT pooling or linear layout is incompatible")
        if _positive_integer(model.get("timesteps"), "model.timesteps") != 20:
            raise ConfigurationError("paper-reported S-VGG9 BNTT requires 20 configured timesteps")
        if model.get("leak") != 0.95 or model.get("threshold") != 1.0 or model.get("surrogate_scale") != 0.3:
            raise ConfigurationError("S-VGG9 BNTT neuron assumptions are incompatible")
        if model.get("bntt_momentum") != 0.1 or model.get("bntt_epsilon") != 0.0001:
            raise ConfigurationError("paper-reported S-VGG9 BNTT requires momentum 0.1 and epsilon 1e-4")
        if model.get("input_encoding") != "signed_poisson":
            raise ConfigurationError("paper-reported S-VGG9 BNTT requires signed_poisson input encoding")
        if model.get("poisson_rescale_factor") != 2.0:
            raise ConfigurationError("paper-reported S-VGG9 BNTT requires Poisson rescale factor 2.0")
        if model.get("readout") != "temporal_mean":
            raise ConfigurationError("paper-reported S-VGG9 BNTT requires temporal_mean readout")
        if model.get("weight_initialization") != "xavier_uniform_gain_2":
            raise ConfigurationError("paper-reported S-VGG9 BNTT requires xavier_uniform_gain_2 initialization")
        if acceptance.get("expected_model_class") != "SVGG9BNTT":
            raise ConfigurationError("acceptance.expected_model_class must be SVGG9BNTT")
    elif dataset_name == "shd":
        raise ConfigurationError("SHD cannot be paired with S-VGG9 BNTT")
    elif dataset_name == "ssc":
        raise ConfigurationError("SSC cannot be paired with S-VGG9 BNTT")
    elif dataset_name == "cifar10":
        raise ConfigurationError("CIFAR-10 cannot be paired with AudioLIFSNN")
    else:
        raise ConfigurationError(f"unsupported federated dataset/model combination: {(dataset_name, model_name)}")
    if not isinstance(dataset.get("root"), str) or not dataset["root"]:
        raise ConfigurationError("dataset.root must be a path string")


def _validate_optimizer(federation: Mapping[str, Any], protocol: str) -> None:
    optimizer = federation.get("optimizer")
    learning_rate = _finite_positive(federation.get("learning_rate"), "federated.learning_rate")
    _nonnegative(federation.get("weight_decay"), "federated.weight_decay")
    gradient_clip = federation.get("gradient_clip")
    if gradient_clip is not None:
        _finite_positive(gradient_clip, "federated.gradient_clip")
    reductions = federation.get("learning_rate_reduction_rounds")
    if not isinstance(reductions, list) or any(
        not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in reductions
    ):
        raise ConfigurationError("federated.learning_rate_reduction_rounds must be a list of positive integers")
    if reductions != sorted(set(reductions)):
        raise ConfigurationError("learning-rate reduction rounds must be sorted and unique")
    factor = _finite_positive(
        federation.get("learning_rate_reduction_factor"), "federated.learning_rate_reduction_factor"
    )
    if optimizer == "adam":
        if protocol in CIFAR_PROTOCOLS or protocol == "published_protocol":
            raise ConfigurationError("published Fed-SNN protocol requires SGD")
        if "momentum" in federation and federation.get("momentum") is not None:
            raise ConfigurationError("momentum cannot be configured for Adam")
        if reductions or factor != 1.0:
            raise ConfigurationError("Adam evaluations do not use learning-rate reductions")
    elif optimizer == "sgd":
        _nonnegative(federation.get("momentum"), "federated.momentum")
        if not reductions or factor <= 1:
            raise ConfigurationError("SGD published protocol requires a reduction schedule")
    else:
        raise ConfigurationError(f"unsupported federated optimizer: {optimizer}")
    if protocol == "independent_evaluation" and (optimizer != "adam" or learning_rate != 0.001):
        raise ConfigurationError("independent federated evaluation requires Adam at learning rate 0.001")
    if protocol in CIFAR_PROTOCOLS | {"published_protocol"} and (
        optimizer != "sgd" or learning_rate != 0.1 or federation.get("momentum") != 0.95
    ):
        raise ConfigurationError("published Fed-SNN protocol requires SGD, learning rate 0.1, and momentum 0.95")


def validate_federated_config(config: Mapping[str, Any]) -> None:
    """Reject incomplete or scientifically incompatible federated configurations."""

    for key in (
        "name",
        "seed",
        "mode",
        "execution",
        "protocol",
        "device",
        "output_root",
        "pairing_group",
        "dataset",
        "model",
        "training",
        "subset",
        "federated",
        "seed_streams",
        "acceptance",
        "provenance",
    ):
        if key not in config:
            raise ConfigurationError(f"missing required setting: {key}")
    if not isinstance(config["name"], str) or not config["name"].strip():
        raise ConfigurationError("name must be a non-empty string")
    if not isinstance(config["seed"], int) or isinstance(config["seed"], bool):
        raise ConfigurationError("seed must be an integer")
    if config["mode"] != "scientific_evaluation" or config["execution"] != "federated":
        raise ConfigurationError("federated protocol requires scientific_evaluation and federated execution")
    if config["protocol"] not in {
        "independent_evaluation",
        "published_protocol",
        *CIFAR_PROTOCOLS,
    }:
        raise ConfigurationError("unsupported federated scientific protocol")
    if config["device"] != "cuda":
        raise ConfigurationError("scientific federated configurations require CUDA")
    if not isinstance(config["output_root"], str) or not config["output_root"]:
        raise ConfigurationError("output_root must be a path string")
    if not isinstance(config["pairing_group"], str) or not config["pairing_group"]:
        raise ConfigurationError("pairing_group must identify comparable executions")
    _validate_dataset_and_model(config)

    subset = _section(config, "subset")
    if any(subset.get(key) != 0 for key in ("train_examples", "validation_examples", "test_examples")):
        raise ConfigurationError("scientific federated evaluation prohibits dataset caps")
    if subset.get("stratified") is not True:
        raise ConfigurationError("validation selection must be stratified")
    training = _section(config, "training")
    if any(
        training.get(key) is not None for key in ("max_train_batches", "max_validation_batches", "max_test_batches")
    ):
        raise ConfigurationError("scientific federated evaluation prohibits batch caps")

    federation = _section(config, "federated")
    if federation.get("algorithm") != "fedavg":
        raise ConfigurationError("unsupported federated algorithm")
    clients = _positive_integer(federation.get("clients"), "federated.clients")
    if clients < 2:
        raise ConfigurationError("federated.clients must contain at least two clients")
    rounds = _positive_integer(federation.get("rounds"), "federated.rounds")
    selected = _positive_integer(federation.get("clients_per_round"), "federated.clients_per_round")
    if selected > clients:
        raise ConfigurationError("federated.clients_per_round cannot exceed federated.clients")
    participation = federation.get("participation_fraction")
    if (
        not isinstance(participation, (int, float))
        or isinstance(participation, bool)
        or not math.isfinite(participation)
    ):
        raise ConfigurationError("federated.participation_fraction must be finite and in (0, 1]")
    if not 0 < float(participation) <= 1 or not math.isclose(
        float(participation), selected / clients, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ConfigurationError(
            "federated.participation_fraction is inconsistent with clients_per_round; it must be in (0, 1]"
        )
    local_epochs = _positive_integer(federation.get("local_epochs"), "federated.local_epochs")
    batch_size = _positive_integer(federation.get("local_batch_size"), "federated.local_batch_size")
    if rounds != 100:
        raise ConfigurationError("scientific federated protocols require 100 communication rounds")
    if federation.get("client_sampling") != "without_replacement":
        raise ConfigurationError("client sampling must be deterministic without replacement")
    if federation.get("retain_optimizer_state") is not False:
        raise ConfigurationError("client optimizer state must not persist")
    if federation.get("official_test_evaluation_during_rounds") is not False:
        raise ConfigurationError("official-test monitoring during training is prohibited")
    if not isinstance(federation.get("record_extended_diagnostics"), bool):
        raise ConfigurationError("federated.record_extended_diagnostics must be boolean")
    if not isinstance(federation.get("drop_last_local_batch"), bool):
        raise ConfigurationError("federated.drop_last_local_batch must be boolean")
    if federation.get("aggregation_weighting") not in {"uniform", "example_count"}:
        raise ConfigurationError("unsupported federated aggregation weighting")
    if federation.get("checkpoint_selection") not in {"best_validation", "final_round"}:
        raise ConfigurationError("unsupported federated checkpoint selection policy")
    _validate_optimizer(federation, config["protocol"])

    if config["protocol"] == "independent_evaluation":
        dataset_name = config["dataset"]["name"]
        if dataset_name not in {"shd", "ssc"}:
            raise ConfigurationError("independent federated evaluation is defined for SHD and SSC")
        compatible = (
            (clients, selected, local_epochs, batch_size) in {(20, 10, 1, 32), (20, 5, 1, 32)}
            if dataset_name == "shd"
            else (clients, selected, local_epochs, batch_size) == (20, 10, 1, 256)
        )
        if not compatible:
            raise ConfigurationError(
                f"{dataset_name.upper()} federated client, participation, epoch, or batch settings are incompatible"
            )
        if federation.get("weight_decay") != 0.0 or federation.get("gradient_clip") != 1.0:
            raise ConfigurationError("independent federated evaluation requires zero weight decay and clipping 1.0")
        if (
            federation.get("drop_last_local_batch") is not False
            or federation.get("aggregation_weighting") != "example_count"
            or federation.get("checkpoint_selection") != "best_validation"
        ):
            raise ConfigurationError(
                "independent federated evaluation requires retained local batches, "
                "example-count weighting, and best-validation selection"
            )
    elif config["protocol"] in CIFAR_PROTOCOLS:
        if config["dataset"]["name"] != "cifar10":
            raise ConfigurationError("published Fed-SNN protocol is defined for CIFAR-10")
        paper_name = _paper_experiment_name(config)
        if paper_name not in PAPER_EXPERIMENTS:
            raise ConfigurationError("paper-reported Fed-SNN experiment identity is unsupported")
        normalized_output_root = config["output_root"].replace("\\", "/").rstrip("/")
        distributed_output = "parallel_execution" in config and (
            normalized_output_root == "runs/distributed_evaluation"
            or normalized_output_root.endswith("/runs/distributed_evaluation")
        )
        if not distributed_output and not (
            normalized_output_root == "runs/fedsnn_paper_evaluation"
            or normalized_output_root.endswith("/runs/fedsnn_paper_evaluation")
        ):
            raise ConfigurationError("paper-reported Fed-SNN output_root must isolate Table I evaluations")
        if (clients, selected, local_epochs, batch_size) != (10, 2, 5, 32):
            raise ConfigurationError(
                "published Fed-SNN client, participation, epoch, or batch settings are incompatible"
            )
        if federation.get("learning_rate_reduction_rounds") != [40, 60, 80]:
            raise ConfigurationError("published Fed-SNN reduction rounds must be [40, 60, 80]")
        if federation.get("learning_rate_reduction_factor") != 5:
            raise ConfigurationError("published Fed-SNN reduction factor must be 5")
        if federation.get("weight_decay") != 0.0001 or federation.get("gradient_clip") is not None:
            raise ConfigurationError("Table I Fed-SNN configurations require source-default weight decay 1e-4")
        if (
            federation.get("drop_last_local_batch") is not True
            or federation.get("aggregation_weighting") != "uniform"
            or federation.get("checkpoint_selection") != "final_round"
        ):
            raise ConfigurationError(
                "paper-reported Fed-SNN configurations require dropped local remainders, "
                "uniform aggregation, and final-round selection"
            )
        assumptions = config.get("protocol_assumptions")
        if (
            not isinstance(assumptions, list)
            or not assumptions
            or any(not isinstance(value, str) or not value for value in assumptions)
        ):
            raise ConfigurationError("published protocol requires explicit protocol_assumptions")
        setting_sources = config.get("setting_sources")
        if not isinstance(setting_sources, Mapping) or set(setting_sources) != {
            "paper",
            "released_source",
            "interpretations",
        }:
            raise ConfigurationError("paper-reported Fed-SNN protocol requires paper/source/interpretation disclosures")
        if any(
            not isinstance(values, list)
            or not values
            or any(not isinstance(value, str) or not value for value in values)
            for values in setting_sources.values()
        ):
            raise ConfigurationError("Fed-SNN setting-source disclosures must be nonempty string lists")

    partition = _section(federation, "partition")
    method = partition.get("method")
    if method not in PARTITION_METHODS:
        raise ConfigurationError(f"unknown partition method: {method}")
    minimum_examples = _positive_integer(
        partition.get("minimum_examples_per_client"),
        "federated.partition.minimum_examples_per_client",
    )
    if method in {"stratified_iid", "fedsnn_random_iid"}:
        if partition.get("alpha") is not None:
            raise ConfigurationError(f"{method} configurations cannot set alpha")
    else:
        _finite_positive(partition.get("alpha"), "federated.partition.alpha")
        _positive_integer(partition.get("maximum_attempts"), "federated.partition.maximum_attempts")
    if config["protocol"] == "paper_reported_evaluation":
        expected_method, expected_alpha, _ = PAPER_EXPERIMENTS[_paper_experiment_name(config)]
        if method != expected_method or partition.get("alpha") != expected_alpha or minimum_examples != 10:
            raise ConfigurationError(
                "paper-reported evaluation has an incompatible distribution or minimum client size"
            )

    workers = federation.get("data_loader_workers")
    if not isinstance(workers, int) or workers < 0:
        raise ConfigurationError("federated.data_loader_workers must be a non-negative integer")
    if not isinstance(federation.get("persistent_workers"), bool):
        raise ConfigurationError("federated.persistent_workers must be boolean")
    if federation["persistent_workers"] and workers == 0:
        raise ConfigurationError("persistent workers require at least one data-loader worker")
    if "pin_memory" in federation and not isinstance(federation["pin_memory"], bool):
        raise ConfigurationError("federated.pin_memory must be boolean")
    prefetch = federation.get("prefetch_factor")
    if prefetch is not None and (not isinstance(prefetch, int) or isinstance(prefetch, bool) or prefetch <= 0):
        raise ConfigurationError("federated.prefetch_factor must be a positive integer or null")
    if workers == 0 and prefetch is not None:
        raise ConfigurationError("federated.prefetch_factor requires at least one data-loader worker")
    if "non_blocking_transfer" in federation and not isinstance(federation["non_blocking_transfer"], bool):
        raise ConfigurationError("federated.non_blocking_transfer must be boolean")
    communication = _section(federation, "communication")
    if (
        communication.get("model_downloads_per_selected_client") != 1
        or communication.get("model_uploads_per_selected_client") != 1
        or communication.get("include_optimizer_state") is not False
        or communication.get("include_dataset_transfer") is not False
    ):
        raise ConfigurationError("logical communication configuration is incompatible")

    streams = _section(config, "seed_streams")
    if set(streams) != REQUIRED_STREAMS or any(not isinstance(value, str) or not value for value in streams.values()):
        raise ConfigurationError(f"seed_streams must contain exactly {sorted(REQUIRED_STREAMS)}")
    if len(set(streams.values())) != len(streams):
        raise ConfigurationError("seed stream identities must be distinct")
    provenance = _section(config, "provenance")
    if any(
        provenance.get(key) is not True
        for key in (
            "require_git_commit",
            "require_dataset_identity",
            "require_split_identity",
            "require_partition_identity",
            "require_model_initialization_identity",
        )
    ):
        raise ConfigurationError("all federated provenance requirements must be enabled")

    acceptance = _section(config, "acceptance")
    reference = acceptance.get("reference_test_accuracy")
    tolerance = acceptance.get("absolute_tolerance")
    descriptive_reference = acceptance.get("descriptive_reference_accuracy")
    if reference is None:
        if tolerance is not None:
            raise ConfigurationError("acceptance.absolute_tolerance must be null when reference accuracy is null")
    else:
        if not isinstance(reference, (int, float)) or not 0 <= reference <= 1:
            raise ConfigurationError("acceptance.reference_test_accuracy must be in [0, 1]")
        if not isinstance(tolerance, (int, float)) or not 0 <= tolerance <= 1:
            raise ConfigurationError("acceptance.absolute_tolerance must be in [0, 1]")
    if config["protocol"] == "independent_evaluation" and reference is not None:
        raise ConfigurationError("independent evaluation cannot configure a paper accuracy target")
    if config["dataset"]["name"] == "cifar10":
        if reference is not None or tolerance is not None:
            raise ConfigurationError("paper-reported Fed-SNN protocols cannot configure an acceptance target")
        expected_descriptive = PAPER_EXPERIMENTS[_paper_experiment_name(config)][2]
        if descriptive_reference != expected_descriptive:
            raise ConfigurationError("the paper-reported protocol has the wrong descriptive reference")
    elif descriptive_reference is not None:
        raise ConfigurationError("descriptive_reference_accuracy is reserved for paper-reported CIFAR-10 protocols")


def paired_configuration_identity(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return fields that must match across paired participation treatments."""

    federation = dict(_section(config, "federated"))
    federation.pop("participation_fraction", None)
    federation.pop("clients_per_round", None)
    return {
        "mode": config.get("mode"),
        "execution": config.get("execution"),
        "protocol": config.get("protocol"),
        "pairing_group": config.get("pairing_group"),
        "dataset": dict(_section(config, "dataset")),
        "model": dict(_section(config, "model")),
        "subset": dict(_section(config, "subset")),
        "federated": federation,
        "seed_streams": dict(_section(config, "seed_streams")),
    }
