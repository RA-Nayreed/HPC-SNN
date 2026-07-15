"""Creation of non-overwriting centralized run records."""

import json
import shlex
import sys
from pathlib import Path

import yaml

from fedapfa.configuration import experiment_id

from .environment_capture import environment_metadata
from .git_metadata import git_metadata


def initialize_run(config, selected_indices, command=None, resume_checkpoint=None):
    path = Path(config["output_root"]) / experiment_id(config)
    if resume_checkpoint:
        checkpoint = Path(resume_checkpoint).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        expected = (path / "checkpoints").resolve()
        if checkpoint.parent != expected:
            raise ValueError(f"resume checkpoint must belong to {expected}")
        if not path.is_dir():
            raise FileNotFoundError(path)
        return path
    if path.exists():
        raise FileExistsError(f"run already exists: {path}; use --resume explicitly")
    (path / "checkpoints").mkdir(parents=True)
    (path / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    (path / "command.txt").write_text(command or shlex.join(sys.argv), encoding="utf-8")
    (path / "environment.json").write_text(
        json.dumps(environment_metadata(), indent=2, sort_keys=True), encoding="utf-8"
    )
    (path / "git.json").write_text(json.dumps(git_metadata(), indent=2, sort_keys=True), encoding="utf-8")
    if selected_indices:
        (path / "selected_indices.json").write_text(
            json.dumps(selected_indices, indent=2, sort_keys=True), encoding="utf-8"
        )
    return path
