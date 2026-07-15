import pytest

from fedapfa.configuration.loader import load_config


def test_loads_debug_configuration() -> None:
    config = load_config("configs/base/debug.yaml")
    assert config["name"] == "debug"


def test_rejects_missing_configuration_field(tmp_path) -> None:
    config = tmp_path / "bad.yaml"
    config.write_text("name: bad\nseed: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="dataset"):
        load_config(config)
