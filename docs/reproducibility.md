# Reproducibility

## Identity and provenance

Each run ID hashes the resolved scientific configuration and includes the seed. The `--seed` override is applied before ID generation, so seeds 7, 17, and 27 cannot collide. Operational `output_root` and `resume` fields do not alter scientific identity.

A new run records `resolved_config.yaml`, `command.txt`, `command_history.jsonl`, `environment.json`, `git.json`, selected indices when applicable, epoch metrics, logs, checkpoints, final metrics, and acceptance. Git metadata includes the commit, dirty state, and a worktree digest. Scientific records therefore retain both the resolved configuration and Git provenance.

## Automatic resumption

`fedapfa-train-centralized --resume-auto` resolves the run directory before model or dataset construction:

- completed `acceptance.json`: record the retry command and event, then exit with status zero without training;
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

## Federated execution identity

The federated experiment ID uses the same canonical configuration hashing as centralized execution and includes the scientific seed. Operational output paths do not alter identity. The resolved record additionally stores independent numeric seeds for split construction, client partitioning, global model initialization, client selection, client training, validation, and final test evaluation.

Paired participation treatments verify identical split, partition, and model-initialization identities for each seed. Client selection uses an isolated generator whose state is checkpointed after every round. Per-client training seeds are derived with SHA-256 from the experiment seed, stream identity, round, and client identifier; Python's process-randomized hashing is not used.

## Federated records and resumption

Each run contains `split.json`, `partition.json`, `resolved_seeds.json`, `model_initialization.json`, `client_metrics.jsonl`, `round_metrics.jsonl`, `official_test_metrics.json`, `final_metrics.json`, and `acceptance.json`, together with the shared resolved configuration, command history, environment, Git provenance, logs, and checkpoints.

`checkpoints/last.pt` is written atomically after every completed round. It contains the global model, next round, optional best-validation score and round, selection-generator state, global Python, NumPy, CPU Torch, and CUDA random states, scientific identities, Git commit, cumulative communication totals, and completed client and round records. Best-validation protocols replace `checkpoints/best.pt` atomically when validation accuracy improves. Final-round protocols without internal validation use only `last.pt` and do not invent a best checkpoint or best-validation metric.

`fedapfa-train-federated --resume-auto` uses the shared run-directory compatibility checks. It skips a completed compatible execution, resumes an incomplete compatible execution from `checkpoints/last.pt`, rejects configuration or Git mismatches, and never overwrites a completed record. Federated checkpoint loading separately rejects split, partition, model-initialization, model-class, configuration, or Git incompatibility. A durable official-test record prevents a resumed process from evaluating that split a second time.

## Federated completion and aggregation

Completion requires every configured round, valid client selections and aggregation weights, complete partition integrity, finite records, the checkpoint artifacts required by the configured selection policy, nonempty client and round logs, exactly one official-test evaluation after model selection, Git provenance, all scientific identities, and consistent logical communication totals. Accuracy does not determine completion. With no verified FedAvg literature target, scientific status remains `not_claimed`; the centralized SHD LIF result is context rather than a reproduction threshold.

`fedapfa-summarize-federated` requires the two manifest treatments and seeds 7, 17, and 27. It rejects missing, duplicate, incompatible, or incomplete executions; preserves the participation rows; checks paired identities; calculates aggregate statistics and per-seed 50%-minus-25% differences; and reports the centralized-to-federated accuracy gap as context. Its outputs are `federated_summary.json`, `federated_summary.csv`, and `federated_summary.md`.

## Committed federated evidence

The exact manifest is [`experiments/federated_baselines/manifest.yaml`](../experiments/federated_baselines/manifest.yaml). Its two participation treatments crossed with seeds 7, 17, and 27 produce six scientific executions. Aggregation verified that every execution passed completion acceptance, that paired treatments share split, partition, and model-initialization identities for each seed, and that scientific status is `not_claimed` because no verified published target is configured.

The committed evidence has distinct roles:

- [`federated_summary.json`](../results/federated/federated_summary.json) is the authoritative machine-readable aggregation, including validity findings, per-seed records, aggregate statistics, paired differences, centralized context, and scientific status.
- [`federated_summary.csv`](../results/federated/federated_summary.csv) is the tabular aggregate export.
- [`federated_summary.md`](../results/federated/federated_summary.md) is the generated readable summary.
- [`slurm-accounting.txt`](../results/federated/provenance/slurm-accounting.txt) records all six tasks in array `189464` as completed with exit code `0:0`.
- [`execution-commit.txt`](../results/federated/provenance/execution-commit.txt) records executed source commit `29ad1558dff52b856ee35b6ce2f538ec2006594a`.
- [`slurm-array-id.txt`](../results/federated/provenance/slurm-array-id.txt) records array identifier `189464`.

Each accepted run must retain its resolved configuration, Git record, split, partition, model-initialization identity, resolved seed streams, checkpoints, client and round logs, final metrics, official-test record, and acceptance record. The aggregation is valid only when these identities and records agree with the manifest and committed execution provenance.

## Federated reproduction commands

The repository's Roihu reproduction command creates one six-task `gpumedium` array with one GH200 per task and a default concurrency of one:

~~~bash
bash scripts/slurm/submit_roihu_federated.sh \
  --work-dir "/scratch/$CSC_PROJECT/$USER/hpc-snn" \
  --max-parallel 1
~~~

This is a reproduction command, not an action required to validate the committed evidence. A live reproduction array can be monitored with:

~~~bash
squeue --job <JOB_ID> --array -o "%.18i %.9P %.28j %.2t %.10M %.10l %R"
~~~

Scheduler outcomes can be recorded with:

~~~bash
sacct -j <JOB_ID> --array \
  --format=JobID,State,ExitCode,Elapsed,Start,End,AllocTRES
~~~

Aggregate compatible accepted runs with:

~~~bash
fedapfa-summarize-federated \
  --manifest experiments/federated_baselines/manifest.yaml \
  --runs-root "/scratch/$CSC_PROJECT/$USER/hpc-snn/runs/federated" \
  --output-dir "/scratch/$CSC_PROJECT/$USER/hpc-snn/results/federated"
~~~

Check the six acceptance records before aggregation:

~~~bash
WORK_DIR="/scratch/$CSC_PROJECT/$USER/hpc-snn"
mapfile -t acceptance_records < <(
  find "$WORK_DIR/runs/federated" -mindepth 2 -maxdepth 2 \
    -name acceptance.json -print | sort
)
[[ "${#acceptance_records[@]}" -eq 6 ]]
for record in "${acceptance_records[@]}"; do
  jq -e '
    .accepted == true and
    .completed == true and
    (.completion_failures | length == 0) and
    .scientific_status == "not_claimed"
  ' "$record"
done
~~~

The generated output paths are `$WORK_DIR/results/federated/federated_summary.json`, `$WORK_DIR/results/federated/federated_summary.csv`, and `$WORK_DIR/results/federated/federated_summary.md`. Run-level evidence remains below `$WORK_DIR/runs/federated`, Slurm output below `$WORK_DIR/slurm-logs/federated`, and device telemetry below `$WORK_DIR/telemetry/federated`.

## GPU telemetry scope

The Roihu array script samples supported `nvidia-smi` fields every two seconds and records the command, interval, unsupported fields, and CSV observations. Sampling begins before training and is stopped on normal exit, failure, or signal. These observations are job-level GPU telemetry. They are not measured network traffic, per-client energy, neural-model energy, or neuromorphic energy, and their sampling interval limits short-event attribution.

The committed aggregation does not contain device-utilization or energy estimates. Logical communication counts communicated tensors and cannot be interpreted as network traffic or energy. A scheduler time limit is a reservation constraint, not an observed execution time.

## Corrected CIFAR-10 Fed-SNN identity

The corrected configurations use experiment names `cifar10_fedsnn_paper_reported_iid_evaluation` and `cifar10_fedsnn_paper_reported_noniid_evaluation` under `runs/fedsnn_paper_evaluation`. The unsuccessful independent implementation retains its original name and `runs/published_fedsnn` root; the superseded, unexecuted released-command identity is also different. Automatic resumption therefore cannot resolve either identity into a corrected directory.

Checkpoint compatibility hashes the entire resolved configuration and also stores and verifies `aggregation_weighting` and `checkpoint_selection` explicitly. A change between `uniform` and `example_count`, or between `final_round` and `best_validation`, is incompatible. Split, partition, model-initialization, Git, and experiment identities remain mandatory.

Each client record distinguishes original client population, examples presented per local epoch, and total examples presented over local epochs. Round records store the aggregation policy, exact resolved weights, selected populations, and presented counts. Communication accounting is unchanged: one model download and one upload per selected client.

The corrected manifest expands to six tasks: IID and non-IID treatments times seeds 7, 17, and 27. Both use all 50,000 training examples, no internal validation collection, final-round selection, and exactly one post-training official-test access. Their resolved configurations differ only in experiment identity and distribution-specific fields. Resume from `last.pt` preserves the final-round policy; compatibility rejects changes to configuration, aggregation weighting, selection policy, split, partition, initialization, Git, or experiment identity.

The summarizer refuses to pool the two treatments. It reports distribution and alpha, completed seeds, final round, official-test accuracy and macro-F1, both descriptive references, signed and absolute percentage-point differences, optimizer settings, client counts, timesteps, aggregation, 50,000/0/10,000 data counts, official-test access count, and `equivalence_not_established`. It does not report best-validation accuracy when no internal validation collection exists, and no tolerance exists.

The paper supplies five local epochs, momentum 0.95, reductions after rounds 40, 60, and 80, all 50,000 training examples, final reporting after round 100, and three repetitions. The pinned source supplies signed normalization and Poisson encoding, Xavier gain-2 initialization, temporal-mean output, BNTT epsilon `1e-4`, 20 timesteps, uniform aggregation, and default weight decay `1e-4`. Remaining discrepancies are documented in the literature record: paper example-count versus source-uniform aggregation, omitted paper timestep and SNN weight-decay values, source test monitoring versus one final project evaluation, and source computation of all client updates before upload selection versus project computation for selected clients only.

The authors’ source comparison is pinned to `1ab26154b064119850bc9f84c55304b5b45f7df6`; inspected paths and paper/source conflicts are recorded in `literature/paper_notes/fedsnn_2021.md`.
