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

## Distributed evaluation identity and resumption

The distributed [manifest](../experiments/distributed_evaluation/manifest.yaml) expands to 24 tasks: SHD and SSC each contribute one-, two-, and four-GPU treatments, CIFAR-10 contributes one- and two-GPU treatments, and all use seeds 7, 17, and 27. CIFAR-10 cannot request four devices because only two clients participate. Scientific identity is equal across physical-device treatments within a dataset and different across SHD, SSC, and CIFAR-10. Each dataset’s one-GPU distributed execution is its reference; sequential runtimes and other datasets are outside its paired denominator.

Every process agrees on configuration, Git, dataset, split, partition, model initialization, seed identities, process count, physical-device count, control backend, CUDA process service, and single host. Exclusive execution maps one process to each visible GPU through `LOCAL_RANK`, uses NCCL, and rejects mismatched visibility. Same-GPU packing uses Gloo for CPU control and detached state movement while CUDA MPS serves local client work. Its mapping is deterministic by process rank and configured physical-device count. CPU Gloo remains restricted to integration testing.

Run provenance includes `execution_provenance.json` and `process_mapping.json`; the latter contains stable rank, process, and device topology fields so a compatible allocation can resume on another host. `execution_measurements.json` records every distinct Slurm allocation used by an interrupted execution, each attempt's host and process mapping, resident host memory before and after workload construction, the signed memory difference, MPS archive and active-thread percentage when applicable, client/process/device counts, busy and idle durations, load imbalance, allocated and reserved memory, internal data movement, resumption count, and official-test duration. Summary accounting requires a row for every recorded allocation, sums their GPU-hours and elapsed time, and requires the completing allocation to report `COMPLETED` with exit code `0:0`; non-completing allocation states remain visible instead of being discarded. When the Roihu CSV is available, it also records finite physical-device utilization sample counts and aggregate and per-device minima, means, and maxima. Job-level utilization covers only its named execution attempt; the summarizer excludes that value from treatment statistics after resumption rather than presenting it as complete-run utilization. `round_metrics.jsonl` records ordered assignments and model/update identities. `client_metrics.jsonl` records process/device placement, population and presented examples, batches, data wait, seed, spike statistics, duration, and memory.

Distributed checkpoints retain the entire resolved configuration and complete `parallel_execution` identity. Resumption rejects another dataset, scientific identity, device count, client processes per device, total process count, process mapping, control backend, CUDA process service, assignment rule, aggregation order, Git state, split, partition, or initialization. Only a completely collected and aggregated round, including configured validation and `last.pt`, advances durability. A failure during client work repeats the entire round from its incoming checkpoint. Rank 0 is the sole owner of shared checkpoints, metrics, validation, official-test, and acceptance records.

The logical communication count excludes process transfers and remains directly comparable only within one scientific workload. Internal model distribution and collected update bytes are execution data movement. The summarizer groups by workload and then execution treatment, compares each seed with its own workload’s one-GPU execution, checks scientific, split, partition, selection, seed, example-count, aggregation, checkpoint, communication, and test-access identities, and measures maximum absolute and relative checkpoint-parameter differences, selected-round equality, accuracy and macro-F1 differences, and prediction agreement when predictions exist. Accuracy similarity alone is not numerical equivalence, and no tolerance is invented.

Validate the collection locally:

~~~bash
fedapfa-validate-config experiments/distributed_evaluation/shd/lif_fedavg_1_gpu.yaml
fedapfa-validate-config experiments/distributed_evaluation/shd/lif_fedavg_2_gpu.yaml
fedapfa-validate-config experiments/distributed_evaluation/shd/lif_fedavg_4_gpu.yaml
fedapfa-validate-config experiments/distributed_evaluation/ssc/lif_128_fedavg_1_gpu.yaml
fedapfa-validate-config experiments/distributed_evaluation/ssc/lif_128_fedavg_2_gpu.yaml
fedapfa-validate-config experiments/distributed_evaluation/ssc/lif_128_fedavg_4_gpu.yaml
fedapfa-validate-config experiments/distributed_evaluation/cifar10/svgg9_bntt_noniid_1_gpu.yaml
fedapfa-validate-config experiments/distributed_evaluation/cifar10/svgg9_bntt_noniid_2_gpu.yaml
python3 -m fedapfa.cli.scientific_manifest validate \
  --manifest experiments/distributed_evaluation/manifest.yaml
python3 -m fedapfa.cli.scientific_manifest validate \
  --manifest experiments/device_capacity_evaluation/manifest.yaml
python3 -m pytest -q tests/unit/distributed/test_gloo_integration.py
~~~

Optional profiler traces require `profiler_enabled: true` and explicit `profiled_rounds`; they add overhead and are excluded from ordinary runtime interpretation. The 24-task manifest leaves profiling disabled. The separate device-capacity collection represents unmeasured MPS configurations and does not identify a preferred packing level.

No distributed CUDA, NCCL, or MPS run is committed. Once all 24 compatible tasks exist, aggregate them with:

~~~bash
WORK_DIR="/scratch/$CSC_PROJECT/$USER/hpc-snn"
mkdir -p "$WORK_DIR/results/distributed_evaluation"
sacct -j <JOB_ID> --array -X -P \
  --format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocTRES \
  > "$WORK_DIR/results/distributed_evaluation/slurm-accounting.txt"
fedapfa-summarize-distributed-evaluation \
  --manifest experiments/distributed_evaluation/manifest.yaml \
  --runs-root "$WORK_DIR/runs/distributed_evaluation" \
  --output-dir "$WORK_DIR/results/distributed_evaluation" \
  --slurm-accounting "$WORK_DIR/results/distributed_evaluation/slurm-accounting.txt"
~~~

## GPU telemetry scope

The Roihu array script samples supported `nvidia-smi` fields every two seconds and records the command, interval, unsupported fields, and CSV observations. Sampling begins before training and is stopped on normal exit, failure, or signal. These observations are job-level GPU telemetry. They are not measured network traffic, per-client energy, neural-model energy, or neuromorphic energy, and their sampling interval limits short-event attribution.

The committed aggregation does not contain device-utilization or energy estimates. Logical communication counts communicated tensors and cannot be interpreted as network traffic or energy. A scheduler time limit is a reservation constraint, not an observed execution time.

## CIFAR-10 Fed-SNN identity

The configurations use experiment names `cifar10_fedsnn_paper_reported_iid_evaluation` and `cifar10_fedsnn_paper_reported_noniid_evaluation` under `runs/fedsnn_paper_evaluation`. The unsuccessful independent implementation retains its original name and `runs/published_fedsnn` root; the superseded, unexecuted released-command identity is also different. Automatic resumption therefore cannot resolve either identity into a directory.

Checkpoint compatibility hashes the entire resolved configuration and also stores and verifies `aggregation_weighting` and `checkpoint_selection` explicitly. A change between `uniform` and `example_count`, or between `final_round` and `best_validation`, is incompatible. Split, partition, model-initialization, Git, and experiment identities remain mandatory.

Each client record distinguishes original client population, examples presented per local epoch, and total examples presented over local epochs. Round records store the aggregation policy, exact resolved weights, selected populations, and presented counts. Communication accounting is unchanged: one model download and one upload per selected client.

The manifest expands to six tasks: IID and non-IID treatments times seeds 7, 17, and 27. Both use all 50,000 training examples, no internal validation collection, final-round selection, and exactly one post-training official-test access. Their resolved configurations differ only in experiment identity and distribution-specific fields. Resume from `last.pt` preserves the final-round policy; compatibility rejects changes to configuration, aggregation weighting, selection policy, split, partition, initialization, Git, or experiment identity.

The summarizer refuses to pool the two treatments. It reports distribution and alpha, completed seeds, final round, official-test accuracy and macro-F1, both descriptive references, signed and absolute percentage-point differences, optimizer settings, client counts, timesteps, aggregation, 50,000/0/10,000 data counts, official-test access count, and `equivalence_not_established`. It does not report best-validation accuracy when no internal validation collection exists, and no tolerance exists.

The paper supplies five local epochs, momentum 0.95, reductions after rounds 40, 60, and 80, all 50,000 training examples, final reporting after round 100, and three repetitions. The pinned source supplies signed normalization and Poisson encoding, Xavier gain-2 initialization, temporal-mean output, BNTT epsilon `1e-4`, 20 timesteps, uniform aggregation, and default weight decay `1e-4`. Remaining discrepancies are documented in the literature record: paper example-count versus source-uniform aggregation, omitted paper timestep and SNN weight-decay values, source test monitoring versus one final project evaluation, and source computation of all client updates before upload selection versus project computation for selected clients only.

The authors’ source comparison is pinned to `1ab26154b064119850bc9f84c55304b5b45f7df6`; inspected paths and paper/source conflicts are recorded in `literature/paper_notes/fedsnn_2021.md`.

## Committed CIFAR-10 Fed-SNN evidence

The [generated summary](../results/fedsnn_paper_evaluation/published_fedsnn_summary.md) validates all six manifest tasks as complete: IID and balanced label-Dirichlet alpha 0.5, each with seeds 7, 17, and 27. Every execution selected round 100, used 50,000 training and zero internal-validation examples, evaluated all 10,000 official-test examples exactly once, and retained scientific status `equivalence_not_established`.

IID official-test accuracy was 81.50%, 82.16%, and 81.55%, with mean 81.7367% and sample standard deviation 0.3675 percentage points. Non-IID accuracy was 72.01%, 75.80%, and 73.32%, with mean 73.7100% and sample standard deviation 1.9249 percentage points. The signed mean differences from the descriptive paper references are +5.2967 and -0.2300 percentage points. The descriptive IID-to-non-IID reduction is 8.0267 percentage points. Macro-F1 closely follows accuracy and gives no evidence of class collapse.

The recorded execution commit is [`d71cfe4c4fdc7c3480806a5f1302c164273dfb82`](../results/fedsnn_paper_evaluation/provenance/execution-commit.txt). The manifest at that commit hashes to `e25d4d760cffe9475b09673888c6b0e21ba56d1dd8ddc436ffe84b3029db672c`, exactly matching the [stored manifest hash](../results/fedsnn_paper_evaluation/provenance/manifest-sha256.txt). The [Slurm accounting record](../results/fedsnn_paper_evaluation/provenance/slurm-accounting.txt) identifies tasks `236880_0` through `236880_5`; every task is `COMPLETED` with exit code `0:0`. The separate `slurm-array-id.txt` contains no identifier, so array identity is taken from the accounting record rather than inferred from that empty field.

The evidence is the active Fed-SNN reference. The unsuccessful superseded 18.23–26.79% independent implementation remains unchanged and separate; it is neither pooled with nor averaged into these results. The three-seed results support protocol-aligned learning validation but not statistical significance, causality, novelty, energy efficiency, implementation equivalence, or an exact reproduction pass.
