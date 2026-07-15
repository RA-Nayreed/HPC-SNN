import json

import pytest

from fedapfa.configuration import experiment_id, load_config
from fedapfa.utilities.run_records import (
    RunCompatibilityError,
    initialize_run,
    plan_run,
    run_directory,
)


def _config(tmp_path):
    config = load_config("tests/data/configurations/centralized/shd_memorization_validation.yaml")
    config["output_root"] = str(tmp_path)
    return config


def test_auto_resume_starts_only_when_run_does_not_exist(tmp_path):
    config = _config(tmp_path)
    action = plan_run(config, "train --resume-auto", resume_auto=True)
    assert action.run_dir == run_directory(config)
    assert action.resume_checkpoint is None and not action.skip_completed


def test_auto_resume_uses_last_checkpoint_and_records_event(tmp_path):
    config = _config(tmp_path)
    run = initialize_run(config, {}, "initial command")
    checkpoint = run / "checkpoints" / "last.pt"
    checkpoint.write_bytes(b"checkpoint")
    action = plan_run(config, "resumed command", resume_auto=True)
    assert action.resume_checkpoint == checkpoint
    events = [json.loads(line) for line in (run / "resume_events.jsonl").read_text().splitlines()]
    assert events[-1]["event"] == "resume_auto_resume_incomplete"
    assert events[-1]["command"] == "resumed command"


def test_auto_resume_skips_completed_run_without_retraining(tmp_path):
    config = _config(tmp_path)
    run = initialize_run(config, {}, "initial command")
    (run / "acceptance.json").write_text(json.dumps({"completed": True}), encoding="utf-8")
    action = plan_run(config, "retry command", resume_auto=True)
    assert action.skip_completed and action.resume_checkpoint is None
    assert json.loads((run / "resume_events.jsonl").read_text().splitlines()[-1])["event"].endswith("skip_completed")


def test_resume_refuses_incompatible_resolved_config_or_git_metadata(tmp_path):
    config = _config(tmp_path)
    run = initialize_run(config, {}, "initial command")
    (run / "checkpoints" / "last.pt").write_bytes(b"checkpoint")
    changed = dict(config)
    changed["resume"] = "different"
    assert experiment_id(changed) == experiment_id(config)
    with pytest.raises(RunCompatibilityError, match="configuration"):
        plan_run(changed, "retry", resume_auto=True)

    git = json.loads((run / "git.json").read_text())
    git["commit"] = "different"
    (run / "git.json").write_text(json.dumps(git), encoding="utf-8")
    with pytest.raises(RunCompatibilityError, match="Git metadata"):
        plan_run(config, "retry", resume_auto=True)


def test_auto_resume_refuses_incomplete_run_without_last_checkpoint(tmp_path):
    config = _config(tmp_path)
    initialize_run(config, {}, "initial command")
    with pytest.raises(RunCompatibilityError, match="last.pt"):
        plan_run(config, "retry", resume_auto=True)
