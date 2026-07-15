"""Strict validation for fully resolved centralized experiment configurations."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

DATASETS = {"shd": 20, "ssc": 35}
MODELS = {("shd", "lif_2layer"), ("shd", "dcls_shd"), ("ssc", "lif_2layer_128"), ("ssc", "lif_2layer_512")}
ATTENTION = {"none", "equation", "public_behavior"}
PROTOCOLS = {"paper_compatible", "thesis_clean", "tiny_overfit", "smoke"}
MODES = {"tiny_overfit", "smoke", "full", "sweep", "deferred"}


class ConfigurationError(ValueError):
    """The resolved experiment is incomplete or unsafe."""


def _mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{key} must be a mapping")
    return value


def _positive(mapping: Mapping[str, Any], key: str, allow_none: bool = False) -> None:
    value = mapping.get(key)
    if allow_none and value is None:
        return
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise ConfigurationError(f"{key} must be positive")


def validate_config(config: Mapping[str, Any]) -> None:
    required = ("name", "seed", "mode", "dataset", "model", "training", "subset", "protocol", "device", "output_root")
    for key in required:
        if key not in config:
            raise ConfigurationError(f"missing required setting: {key}")
    if not isinstance(config["name"], str) or not config["name"].strip():
        raise ConfigurationError("name must be a non-empty string")
    if not isinstance(config["seed"], int) or isinstance(config["seed"], bool):
        raise ConfigurationError("seed must be an integer")
    if config["mode"] not in MODES:
        raise ConfigurationError(f"unknown mode: {config['mode']}")
    if config["protocol"] not in PROTOCOLS:
        raise ConfigurationError(f"unknown protocol: {config['protocol']}")
    if config["device"] not in {"cpu", "cuda"}:
        raise ConfigurationError("device must be cpu or cuda")
    if not isinstance(config["output_root"], str) or not config["output_root"]:
        raise ConfigurationError("output_root must be a path string")

    dataset = _mapping(config, "dataset")
    dataset_name = dataset.get("name")
    if dataset_name not in DATASETS:
        raise ConfigurationError(f"unknown dataset: {dataset_name}")
    for key in ("root", "train_file", "test_file"):
        if not isinstance(dataset.get(key), str) or not dataset[key]:
            raise ConfigurationError(f"dataset.{key} must be a path string")
    validation_file = dataset.get("validation_file")
    if dataset_name == "ssc" and (not isinstance(validation_file, str) or not validation_file):
        raise ConfigurationError("ssc requires dataset.validation_file")
    if dataset_name == "shd" and validation_file is not None and not isinstance(validation_file, str):
        raise ConfigurationError("shd dataset.validation_file must be null or a path string")
    if dataset.get("classes") != DATASETS[dataset_name]:
        raise ConfigurationError(f"{dataset_name} requires {DATASETS[dataset_name]} classes")
    if dataset.get("raw_channels") != 700 or dataset.get("input_features") != 140:
        raise ConfigurationError("event audio requires 700 raw channels and 140 input features")
    if dataset.get("frequency_bin_factor") != 5:
        raise ConfigurationError("frequency_bin_factor must be 5")
    if dataset.get("temporal_bin_ms") != 10.0:
        raise ConfigurationError("temporal_bin_ms must be 10.0")
    validation_fraction = dataset.get("validation_fraction", 0.1)
    if not 0 < validation_fraction < 1:
        raise ConfigurationError("validation_fraction must be between zero and one")

    model = _mapping(config, "model")
    combination = (dataset_name, model.get("name"))
    if combination not in MODELS:
        raise ConfigurationError(f"unknown dataset/model combination: {combination}")
    hidden = model.get("hidden_dims")
    if not isinstance(hidden, list) or len(hidden) != 2 or any(not isinstance(v, int) or v <= 0 for v in hidden):
        raise ConfigurationError("model.hidden_dims must contain two positive integers")
    required_hidden = {
        "lif_2layer": [256, 256],
        "dcls_shd": [256, 256],
        "lif_2layer_128": [128, 128],
        "lif_2layer_512": [512, 512],
    }
    if hidden != required_hidden[model["name"]]:
        raise ConfigurationError(f"{model['name']} requires hidden_dims={required_hidden[model['name']]}")
    if not isinstance(model.get("dropout"), (int, float)) or not 0 <= model["dropout"] < 1:
        raise ConfigurationError("model.dropout must be in [0, 1)")
    for key in ("batch_normalization", "bias"):
        if not isinstance(model.get(key), bool):
            raise ConfigurationError(f"model.{key} must be boolean")
    neuron = _mapping(model, "neuron")
    if neuron.get("name") not in {"euler_lif", "spikingjelly_lif"}:
        raise ConfigurationError(f"unknown neuron: {neuron.get('name')}")
    _positive(neuron, "tau_ms")
    _positive(neuron, "threshold")
    if neuron.get("reset") not in {"subtract", "zero"} or neuron.get("detach_reset") is not True:
        raise ConfigurationError("neuron requires subtract/zero reset and detach_reset=true")
    surrogate = _mapping(neuron, "surrogate")
    if surrogate.get("name") != "atan":
        raise ConfigurationError("only atan surrogate is supported")
    _positive(surrogate, "alpha")
    attention = _mapping(model, "attention")
    if attention.get("variant") not in ATTENTION:
        raise ConfigurationError(f"unknown attention variant: {attention.get('variant')}")
    _positive(attention, "lambda")

    training = _mapping(config, "training")
    if training.get("optimizer") != "adam":
        raise ConfigurationError("only adam optimizer is supported")
    for key in ("learning_rate", "gradient_clip", "delay_lr_multiplier"):
        _positive(training, key)
    for key in ("batch_size", "epochs"):
        value = training.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ConfigurationError(f"training.{key} must be a positive integer")
    if not isinstance(training.get("weight_decay"), (int, float)) or training["weight_decay"] < 0:
        raise ConfigurationError("weight_decay must be non-negative")
    for key in ("max_train_batches", "max_validation_batches", "max_test_batches"):
        value = training.get(key)
        if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value <= 0):
            raise ConfigurationError(f"training.{key} must be null or a positive integer")
    target_accuracy = training.get("target_accuracy")
    if target_accuracy is not None and (
        not isinstance(target_accuracy, (int, float))
        or isinstance(target_accuracy, bool)
        or not 0 < target_accuracy <= 1
    ):
        raise ConfigurationError("training.target_accuracy must be null or in (0, 1]")
    early_stop_patience = training.get("early_stop_patience")
    if early_stop_patience is not None and (
        not isinstance(early_stop_patience, int) or isinstance(early_stop_patience, bool) or early_stop_patience <= 0
    ):
        raise ConfigurationError("training.early_stop_patience must be null or a positive integer")
    workers = training.get("data_loader_workers")
    if not isinstance(workers, int) or workers < 0:
        raise ConfigurationError("data_loader_workers must be a non-negative integer")
    if not isinstance(training.get("persistent_workers"), bool):
        raise ConfigurationError("persistent_workers must be boolean")
    if training["persistent_workers"] and workers == 0:
        raise ConfigurationError("persistent_workers requires data_loader_workers > 0")

    subset = _mapping(config, "subset")
    for key in ("train_examples", "validation_examples", "test_examples"):
        value = subset.get(key)
        if not isinstance(value, int) or value < 0:
            raise ConfigurationError(f"subset.{key} must be a non-negative integer")
    if not isinstance(subset.get("stratified"), bool):
        raise ConfigurationError("subset.stratified must be boolean")
    if config["mode"] == "tiny_overfit" and config["protocol"] != "tiny_overfit":
        raise ConfigurationError("tiny_overfit mode requires tiny_overfit protocol")
    if config["mode"] in {"smoke", "sweep"} and config["protocol"] != "smoke":
        raise ConfigurationError("smoke and sweep modes require smoke protocol")
    if config["mode"] == "tiny_overfit":
        if not 32 <= subset["train_examples"] <= 64:
            raise ConfigurationError("tiny_overfit requires 32-64 train examples")
        if subset["validation_examples"] or subset["test_examples"]:
            raise ConfigurationError("tiny_overfit must not use validation or test subsets")
        if model["dropout"] != 0:
            raise ConfigurationError("tiny_overfit requires dropout=0")
        target = training.get("target_accuracy")
        if not isinstance(target, (int, float)) or not 0.95 <= target <= 1:
            raise ConfigurationError("tiny_overfit target_accuracy must be at least 0.95")
    if config["mode"] == "smoke":
        if training["max_train_batches"] is None or training["max_validation_batches"] is None:
            raise ConfigurationError("smoke runs require hard train and validation batch limits")
        if training["max_test_batches"] is not None:
            raise ConfigurationError("smoke runs must not evaluate the official test set")
    if config["protocol"] == "paper_compatible" and dataset_name != "shd":
        raise ConfigurationError("paper_compatible is defined only for SHD")
    if model["name"] == "dcls_shd":
        if attention["variant"] != "none":
            raise ConfigurationError("dcls_shd does not implement PfA; attention must be none")
        if training["delay_lr_multiplier"] != 100.0:
            raise ConfigurationError("dcls_shd delay_lr_multiplier must be 100")
        _positive(model, "maximum_delay_ms")
        if model["maximum_delay_ms"] != 250:
            raise ConfigurationError("dcls_shd maximum_delay_ms must be 250")

    if config["mode"] == "sweep":
        sweep = _mapping(config, "sweep")
        if sweep.get("parameter") != "model.attention.lambda":
            raise ConfigurationError("lambda sweep must target model.attention.lambda")
        if sweep.get("values") != [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
            raise ConfigurationError("lambda sweep values do not match the required grid")


def resolve_dataset_paths(config: Mapping[str, Any]) -> dict[str, Path]:
    dataset = _mapping(config, "dataset")
    root = Path(dataset["root"])
    return {
        split: root / dataset[key]
        for split, key in (("train", "train_file"), ("validation", "validation_file"), ("test", "test_file"))
        if dataset.get(key)
    }
