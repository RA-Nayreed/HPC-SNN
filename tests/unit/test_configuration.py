import copy

import pytest

from fedapfa.configuration import ConfigurationError, expand_sweep, experiment_id, load_config, validate_config

TINY = "experiments/week01_pfa_reproduction/01_tiny_overfit.yaml"


def test_all_required_experiments_validate():
    paths = [
        TINY,
        "experiments/week01_pfa_reproduction/02_plain_lif_shd.yaml",
        "experiments/week01_pfa_reproduction/03_dcls_shd.yaml",
        "experiments/week01_pfa_reproduction/04_pfa_equation.yaml",
        "experiments/week01_pfa_reproduction/05_pfa_public_behavior.yaml",
        "experiments/week01_pfa_reproduction/06_lambda_sweep.yaml",
        "experiments/week01_pfa_reproduction/07_spike_statistics_smoke.yaml",
        "experiments/week01_pfa_reproduction/08_ssc_tiny_overfit.yaml",
        "experiments/week01_pfa_reproduction/09_ssc_smoke.yaml",
        "experiments/week01_pfa_reproduction/10_shd_plain_lif_full.yaml",
        "experiments/week01_pfa_reproduction/11_shd_dcls_reference.yaml",
        "experiments/week01_pfa_reproduction/12_ssc_512_deferred.yaml",
    ]
    for path in paths:
        validate_config(load_config(path))


def test_invalid_names_and_values_are_rejected():
    base = load_config(TINY)
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
    ssc = load_config("experiments/week01_pfa_reproduction/09_ssc_smoke.yaml")
    ssc["dataset"]["validation_file"] = None
    with pytest.raises(ConfigurationError, match="validation_file"):
        validate_config(ssc)
    dcls = load_config("experiments/week01_pfa_reproduction/03_dcls_shd.yaml")
    dcls["model"]["attention"]["variant"] = "equation"
    with pytest.raises(ConfigurationError, match="does not implement PfA"):
        validate_config(dcls)


def test_ids_hash_every_result_setting_and_do_not_collide():
    configs = [
        load_config(path)
        for path in [
            TINY,
            "experiments/week01_pfa_reproduction/02_plain_lif_shd.yaml",
            "experiments/week01_pfa_reproduction/04_pfa_equation.yaml",
            "experiments/week01_pfa_reproduction/05_pfa_public_behavior.yaml",
            "experiments/week01_pfa_reproduction/06_lambda_sweep.yaml",
            "experiments/week01_pfa_reproduction/07_spike_statistics_smoke.yaml",
            "experiments/week01_pfa_reproduction/08_ssc_tiny_overfit.yaml",
            "experiments/week01_pfa_reproduction/09_ssc_smoke.yaml",
        ]
    ]
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
    runs = expand_sweep(load_config("experiments/week01_pfa_reproduction/06_lambda_sweep.yaml"))
    assert [run["model"]["attention"]["lambda"] for run in runs] == [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    assert len({experiment_id(run) for run in runs}) == 6
