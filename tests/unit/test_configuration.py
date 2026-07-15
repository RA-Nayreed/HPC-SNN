import copy

import pytest

from fedapfa.configuration import ConfigurationError, expand_sweep, experiment_id, load_config, validate_config

MEMORIZATION_CONFIG = "tests/data/configurations/centralized/shd_memorization_validation.yaml"
CANONICAL = [
    "experiments/centralized/shd/lif_independent_evaluation.yaml",
    "experiments/centralized/shd/dcls_published_protocol.yaml",
    "experiments/centralized/shd/pfa_equation_independent_evaluation.yaml",
    "experiments/centralized/shd/pfa_public_published_protocol.yaml",
    "experiments/centralized/ssc/lif_128_independent_evaluation.yaml",
    "experiments/centralized/ssc/pfa_equation_128_independent_evaluation.yaml",
]
FIXTURES = [
    MEMORIZATION_CONFIG,
    "tests/data/configurations/centralized/shd_pfa_lambda_grid.yaml",
    "tests/data/configurations/centralized/ssc_memorization_validation.yaml",
    "tests/data/configurations/centralized/ssc_reduced_sample_lif.yaml",
]


def test_all_required_experiments_validate():
    for path in CANONICAL + FIXTURES:
        validate_config(load_config(path))


def test_invalid_names_and_values_are_rejected():
    base = load_config(MEMORIZATION_CONFIG)
    mutations = [
        ("dataset", "name", "bad"),
        ("model", "name", "bad"),
        ("model", "attention", {"variant": "bad", "lambda": 0.01}),
        (None, "protocol", "bad"),
        ("training", "batch_size", 0),
        ("training", "epochs", 1.5),
        ("training", "early_stop_patience", 0),
        ("dataset", "root", ""),
        ("model", "dropout", 0.5),
    ]
    for section, key, value in mutations:
        config = copy.deepcopy(base)
        target = config if section is None else config[section]
        target[key] = value
        with pytest.raises(ConfigurationError):
            validate_config(config)


def test_missing_ssc_split_and_ignored_dcls_attention_are_rejected():
    ssc = load_config("tests/data/configurations/centralized/ssc_reduced_sample_lif.yaml")
    ssc["dataset"]["validation_file"] = None
    with pytest.raises(ConfigurationError, match="validation_file"):
        validate_config(ssc)
    dcls = load_config("experiments/centralized/shd/dcls_published_protocol.yaml")
    dcls["model"]["attention"]["variant"] = "equation"
    with pytest.raises(ConfigurationError, match="does not implement PfA"):
        validate_config(dcls)


def test_ids_hash_every_result_setting_and_do_not_collide():
    configs = [load_config(path) for path in CANONICAL + FIXTURES]
    ids = [experiment_id(item) for item in configs]
    assert len(ids) == len(set(ids))
    assert all(item.startswith(config["name"] + "-seed7-") for item, config in zip(ids, configs, strict=True))
    changed = copy.deepcopy(configs[0])
    changed["model"]["dropout"] = 0.01
    assert experiment_id(changed) != ids[0]
    operational = copy.deepcopy(configs[0])
    operational["resume"] = "/tmp/checkpoint.pt"
    operational["output_root"] = "/tmp/runs"
    assert experiment_id(operational) == ids[0]


def test_sweep_expands_required_grid_with_unique_ids():
    runs = expand_sweep(load_config("tests/data/configurations/centralized/shd_pfa_lambda_grid.yaml"))
    assert [run["model"]["attention"]["lambda"] for run in runs] == [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    assert len({experiment_id(run) for run in runs}) == 6


CENTRALIZED_MANIFEST = "experiments/centralized/manifest.yaml"


def test_centralized_manifest_is_exact_and_uncapped():
    from fedapfa.configuration import load_centralized_manifest

    tasks = load_centralized_manifest(CENTRALIZED_MANIFEST)
    assert len(tasks) == 18
    assert sorted({task.seed for task in tasks}) == [7, 17, 27]
    assert len({task.experiment for task in tasks}) == 6
    assert all(task.mode == "scientific_evaluation" and task.config["device"] == "cuda" for task in tasks)
    for task in tasks:
        training = task.config["training"]
        assert training["max_train_batches"] is None
        assert training["max_validation_batches"] is None
        assert training["max_test_batches"] is None
        assert task.config["subset"] == {
            "train_examples": 0,
            "validation_examples": 0,
            "test_examples": 0,
            "stratified": True,
        }
        label = f"{task.experiment} {task.config_path}".lower()
        assert not any(word in label for word in ("reduced_sample", "sweep", "memorization_validation"))


def test_centralized_seed_override_changes_experiment_id():
    from fedapfa.configuration import load_centralized_manifest

    tasks = load_centralized_manifest(CENTRALIZED_MANIFEST)
    ids = [experiment_id(task.config) for task in tasks if task.experiment == "shd_lif_independent_evaluation"]
    assert len(ids) == 3 and len(set(ids)) == 3
    assert any("-seed7-" in value for value in ids)
    assert any("-seed17-" in value for value in ids)
    assert any("-seed27-" in value for value in ids)


def test_scientific_evaluation_requires_strict_acceptance_and_no_caps():
    config = load_config("experiments/centralized/shd/lif_independent_evaluation.yaml")
    invalid = copy.deepcopy(config)
    invalid.pop("acceptance")
    with pytest.raises(ConfigurationError, match="acceptance"):
        validate_config(invalid)
    invalid = copy.deepcopy(config)
    invalid["training"]["max_test_batches"] = 1
    with pytest.raises(ConfigurationError, match="max_test_batches"):
        validate_config(invalid)


def test_cli_seed_override_precedes_experiment_id_generation():
    from argparse import Namespace

    from fedapfa.cli.train_centralized import _override

    config = load_config("experiments/centralized/shd/lif_independent_evaluation.yaml")
    args = Namespace(data_root=None, output_root=None, device=None, seed=17)
    overridden = _override(config, args)
    assert overridden["seed"] == 17
    assert "-seed17-" in experiment_id(overridden)
    assert experiment_id(overridden) != experiment_id(config)
