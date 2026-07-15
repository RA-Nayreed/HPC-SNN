# Reproducibility

## Identity and provenance

Each run ID hashes the resolved scientific configuration and includes the seed. The `--seed` override is applied before ID generation, so seeds 7, 17, and 27 cannot collide. Operational `output_root` and `resume` fields do not alter scientific identity.

A new run records `resolved_config.yaml`, `command.txt`, `command_history.jsonl`, `environment.json`, `git.json`, selected indices when applicable, epoch metrics, logs, checkpoints, final metrics, and acceptance. Git metadata includes the commit, dirty state, and a worktree digest. Scientific records therefore retain both the resolved configuration and Git provenance.

## Automatic resumption

`fedapfa-train-centralized --resume-auto` resolves the run directory before model or dataset construction:

- completed `acceptance.json`: record the retry command and event, then exit successfully without training;
- compatible incomplete run with `checkpoints/last.pt`: record the resume event and continue;
- no run directory: create a new non-overwriting run;
- incompatible resolved configuration or Git metadata: refuse;
- incomplete directory without `last.pt`: refuse rather than overwrite.

The checkpoint includes model, optimizer, RNG, seed-loader generator, epoch, global step, best selection accuracy, early-stop counter, and accumulated runtime. Metrics beyond the resumed checkpoint are reconciled before appending.

## Acceptance records

Execution completion and scientific reproduction are distinct. Completion requires a valid terminal condition, finite metrics, both checkpoints, verified complete-dataset use, no active batch limit, official test evaluation, the expected model class, and nonempty metrics and logs.

Scientific status is `passed` or `failed` only when a verified `reference_test_accuracy` and `absolute_tolerance` are configured. A null reference always produces `not_claimed`; it never produces an automatic pass.

## Three-seed aggregation

`fedapfa-summarize-centralized` requires exactly seeds 7, 17, and 27 for every mandatory experiment. It detects missing, duplicate, invalid, and incomplete runs; refuses protocol mixing; and reports mean, sample standard deviation, minimum, and maximum for validation accuracy, official test accuracy, runtime, peak CUDA memory, and per-layer spike rates. It also reports parameter counts and configured reproduction deltas.
