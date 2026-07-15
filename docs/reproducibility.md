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

## Committed evidence artifacts

The centralized aggregation validated 18 executions: six experiments with completed seeds 7, 17, and 27 for each experiment. The committed evidence has distinct roles:

- [`centralized_summary.json`](../results/centralized/centralized_summary.json) is the authoritative machine-readable aggregation and records validity, required seeds, per-run evidence, aggregate statistics, provenance, and scientific status.
- [`centralized_summary.csv`](../results/centralized/centralized_summary.csv) is the tabular export of the aggregate experiment records.
- [`centralized_summary.md`](../results/centralized/centralized_summary.md) is the generated readable summary.
- [`slurm-accounting.txt`](../results/centralized/provenance/slurm-accounting.txt) records scheduler outcomes for array `186611`; all 18 tasks completed with exit code `0:0`.
- [`execution-commit.txt`](../results/centralized/provenance/execution-commit.txt) identifies the executed source state as commit `44c7e4aa52a8dd6103256d17da3b79d2aa075056`.

The scientific status is `not_claimed` for every experiment because the reference accuracy and tolerance are null. This is distinct from execution completion and does not indicate execution failure.

## Measurement limitations

Array concurrency changed from four simultaneous tasks to one and then six. This does not invalidate accuracy aggregation, but runtime values must not be interpreted as a controlled scaling comparison.

The retained scheduler evidence does not contain historical GPU utilization, GPU-memory telemetry, or GPU energy measurements. Peak CUDA memory in the aggregation is a framework-recorded allocation metric, not GPU utilization. Allocated GPU-hours and billing units are operational accounting and must not be interpreted as energy consumption.
