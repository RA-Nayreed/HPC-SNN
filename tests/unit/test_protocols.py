import copy

from conftest import write_event_h5

from fedapfa.configuration import load_config
from fedapfa.training import protocols as protocols_module
from fedapfa.training.protocols import prepare_datasets


def _paths(config, tmp_path):
    classes = config["dataset"]["classes"]
    train = write_event_h5(tmp_path / "train.h5", tuple(range(classes)) * 8)
    valid = write_event_h5(tmp_path / "valid.h5", tuple(range(classes)) * 4)
    test = write_event_h5(tmp_path / "test.h5", tuple(range(classes)) * 4)
    config["dataset"].update(
        {
            "root": str(tmp_path),
            "train_file": train.name,
            "validation_file": valid.name,
            "test_file": test.name,
        }
    )
    return config


def test_memorization_protocol_reuses_subset_and_never_opens_test(tmp_path):
    config = _paths(load_config("tests/data/configurations/centralized/shd_memorization_validation.yaml"), tmp_path)
    bundle = prepare_datasets(config)
    assert bundle.train is bundle.validation
    assert bundle.test is None
    assert len(bundle.train) == 64
    assert not bundle.metadata["official_test_accessed"]


def test_independent_and_published_protocols_change_behavior(tmp_path):
    independent = _paths(load_config("experiments/centralized/shd/lif_independent_evaluation.yaml"), tmp_path)
    independent_bundle = prepare_datasets(independent)
    assert not independent_bundle.metadata["official_test_monitored_during_training"]
    assert not independent_bundle.metadata["official_test_accessed"]
    assert independent_bundle.test is not None
    published = copy.deepcopy(independent)
    published["protocol"] = "published_protocol"
    published["dataset"]["validation_file"] = "test.h5"
    published_bundle = prepare_datasets(published)
    assert published_bundle.metadata["official_test_monitored_during_training"]
    assert published_bundle.metadata["metric_label"] == "reproduction"


def test_ssc_reduced_sample_evaluation_uses_official_validation_split_without_opening_test(tmp_path):
    config = _paths(load_config("tests/data/configurations/centralized/ssc_reduced_sample_lif.yaml"), tmp_path)
    bundle = prepare_datasets(config)
    assert bundle.validation.path.name == "valid.h5"
    assert bundle.test is None
    assert not bundle.metadata["official_test_accessed"]
    assert set(bundle.selected_indices) == {"train", "validation"}


def test_independent_evaluation_defers_test_hdf5_construction_until_requested(tmp_path, monkeypatch):
    config = _paths(load_config("experiments/centralized/shd/lif_independent_evaluation.yaml"), tmp_path)
    opened = []
    original = protocols_module._dataset

    def recording_dataset(path, *args, **kwargs):
        opened.append(path.name)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(protocols_module, "_dataset", recording_dataset)
    bundle = prepare_datasets(config)
    assert "test.h5" not in opened
    bundle.test()
    assert "test.h5" in opened
