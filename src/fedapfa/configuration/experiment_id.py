from collections.abc import Mapping


def experiment_id(config: Mapping[str, object]) -> str:
    return f"{config['name']}-seed{config['seed']}"
