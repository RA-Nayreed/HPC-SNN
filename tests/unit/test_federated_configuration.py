import copy

import pytest

from fedapfa.configuration import (
    ConfigurationError,
    experiment_id,
    load_federated_config,
    load_federated_manifest,
    validate_federated_config,
)

MANIFEST = "experiments/federated_baselines/manifest.yaml"
CONFIG = "experiments/federated_baselines/shd/lif_dirichlet_alpha_0_5_participation_0_50.yaml"


def test_federated_manifest_expands_exact_six_scientific_tasks():
    tasks = load_federated_manifest(MANIFEST)
    assert len(tasks) == 6
    assert sorted({task.seed for task in tasks}) == [7, 17, 27]
    assert len({task.experiment for task in tasks}) == 2
    assert {task.config["federated"]["participation_fraction"] for task in tasks} == {0.25, 0.5}
    assert len({experiment_id(task.config) for task in tasks}) == 6


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("federated", "clients", 1, "at least two"),
        ("partition", "alpha", 0, "finite and positive"),
        ("federated", "participation_fraction", 0, r"in \(0, 1\]"),
        ("federated", "clients_per_round", 0, "positive integer"),
        ("federated", "rounds", 0, "positive integer"),
        ("federated", "local_epochs", -1, "positive integer"),
        ("federated", "local_batch_size", 0, "positive integer"),
        ("federated", "algorithm", "unsupported", "unsupported federated algorithm"),
        ("federated", "official_test_evaluation_during_rounds", True, "prohibited"),
        ("model", "batch_normalization", True, "no batch normalization"),
    ],
)
def test_invalid_reference_settings_are_rejected(section, key, value, message):
    config = load_federated_config(CONFIG)
    changed = copy.deepcopy(config)
    target = changed["federated"]["partition"] if section == "partition" else changed[section]
    target[key] = value
    with pytest.raises(ConfigurationError, match=message):
        validate_federated_config(changed)


def test_missing_seed_identity_and_attention_are_rejected():
    config = load_federated_config(CONFIG)
    missing = copy.deepcopy(config)
    missing["seed_streams"].pop("final_test")
    with pytest.raises(ConfigurationError, match="seed_streams"):
        validate_federated_config(missing)
    attention = copy.deepcopy(config)
    attention["model"]["attention"]["variant"] = "equation"
    with pytest.raises(ConfigurationError, match="PfA"):
        validate_federated_config(attention)


def test_general_validation_command_accepts_federated_configuration(capsys, monkeypatch):
    from fedapfa.cli.validate_run import main

    monkeypatch.setattr("sys.argv", ["fedapfa-validate-config", CONFIG])
    main()
    assert "shd_lif_dirichlet_alpha_0_5_participation_0_50-seed7-" in capsys.readouterr().out
