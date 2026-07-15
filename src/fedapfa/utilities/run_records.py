"""Creation, compatibility checks, and automatic resumption of run records."""

from __future__ import annotations

import json
import shlex
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml

from fedapfa.configuration import experiment_id

from .environment_capture import environment_metadata
from .git_metadata import git_metadata


class RunCompatibilityError(RuntimeError):
    """An existing run was produced by a different configuration or Git state."""


@dataclass(frozen=True)
class RunAction:
    run_dir: Path
    resume_checkpoint: Path | None = None
    skip_completed: bool = False


def run_directory(config) -> Path:
    return Path(config["output_root"]) / experiment_id(config)


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as error:
        raise RunCompatibilityError(f"missing or invalid compatibility metadata: {path}") from error


def _append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _event(event: str, command: str, checkpoint: Path | None = None) -> dict:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "command": command,
        "checkpoint": str(checkpoint) if checkpoint else None,
    }


def _assert_compatible(path: Path, config) -> None:
    resolved_path = path / "resolved_config.yaml"
    try:
        stored_config = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, yaml.YAMLError) as error:
        raise RunCompatibilityError(f"missing or invalid resolved configuration: {resolved_path}") from error
    if stored_config != config:
        raise RunCompatibilityError("existing run resolved configuration is incompatible")
    stored_git = _read_json(path / "git.json")
    current_git = git_metadata()
    if stored_git != current_git:
        raise RunCompatibilityError(
            "existing run Git metadata is incompatible "
            f"(stored commit={stored_git.get('commit')}, current commit={current_git.get('commit')})"
        )


def _is_completed(path: Path) -> bool:
    try:
        acceptance = json.loads((path / "acceptance.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        return False
    return isinstance(acceptance, dict) and acceptance.get("completed") is True


def plan_run(config, command: str, resume_checkpoint=None, resume_auto: bool = False) -> RunAction:
    """Resolve skip/resume/new behavior before model or dataset construction."""

    if resume_checkpoint and resume_auto:
        raise ValueError("--resume and --resume-auto are mutually exclusive")
    path = run_directory(config)

    if resume_auto:
        if not path.exists():
            return RunAction(path)
        if not path.is_dir():
            raise RunCompatibilityError(f"resolved run path is not a directory: {path}")
        _assert_compatible(path, config)
        if _is_completed(path):
            record = _event("resume_auto_skip_completed", command)
            _append_jsonl(path / "command_history.jsonl", record)
            _append_jsonl(path / "resume_events.jsonl", record)
            return RunAction(path, skip_completed=True)
        checkpoint = path / "checkpoints" / "last.pt"
        if not checkpoint.is_file():
            raise RunCompatibilityError("incomplete compatible run has no checkpoints/last.pt to resume")
        record = _event("resume_auto_resume_incomplete", command, checkpoint)
        _append_jsonl(path / "command_history.jsonl", record)
        _append_jsonl(path / "resume_events.jsonl", record)
        return RunAction(path, checkpoint)

    if resume_checkpoint:
        checkpoint = Path(resume_checkpoint).resolve()
        expected = (path / "checkpoints").resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        if checkpoint.parent != expected:
            raise ValueError(f"resume checkpoint must belong to {expected}")
        if not path.is_dir():
            raise FileNotFoundError(path)
        _assert_compatible(path, config)
        if _is_completed(path):
            raise FileExistsError(f"completed run cannot be resumed or overwritten: {path}")
        record = _event("explicit_resume", command, checkpoint)
        _append_jsonl(path / "command_history.jsonl", record)
        _append_jsonl(path / "resume_events.jsonl", record)
        return RunAction(path, checkpoint)

    return RunAction(path)


def initialize_run(config, selected_indices, command=None, resume_checkpoint=None):
    path = run_directory(config)
    if resume_checkpoint:
        checkpoint = Path(resume_checkpoint).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        expected = (path / "checkpoints").resolve()
        if checkpoint.parent != expected:
            raise ValueError(f"resume checkpoint must belong to {expected}")
        if not path.is_dir():
            raise FileNotFoundError(path)
        _assert_compatible(path, config)
        if _is_completed(path):
            raise FileExistsError(f"completed run cannot be resumed or overwritten: {path}")
        return path
    if path.exists():
        raise FileExistsError(f"run already exists: {path}; use --resume or --resume-auto")
    (path / "checkpoints").mkdir(parents=True)
    resolved_command = command or shlex.join(sys.argv)
    (path / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    (path / "command.txt").write_text(resolved_command, encoding="utf-8")
    (path / "environment.json").write_text(
        json.dumps(environment_metadata(), indent=2, sort_keys=True), encoding="utf-8"
    )
    (path / "git.json").write_text(json.dumps(git_metadata(), indent=2, sort_keys=True), encoding="utf-8")
    start_event = _event("started", resolved_command)
    _append_jsonl(path / "command_history.jsonl", start_event)
    if selected_indices:
        (path / "selected_indices.json").write_text(
            json.dumps(selected_indices, indent=2, sort_keys=True), encoding="utf-8"
        )
    return path
