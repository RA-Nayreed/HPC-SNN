from collections.abc import Mapping

REQUIRED_FIELDS = ("name", "seed", "dataset", "model")


def validate_config(config: Mapping[str, object]) -> None:
    missing = [field for field in REQUIRED_FIELDS if field not in config]
    if missing:
        raise ValueError(f"Missing required configuration fields: {', '.join(missing)}")
    if not isinstance(config["seed"], int):
        raise ValueError("'seed' must be an integer")
