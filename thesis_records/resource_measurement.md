# Client resource measurement and cost estimation

## Scientific question and evidence status

This evaluation asks whether the runtime and device energy of an unseen federated SNN client can be predicted before process assignment, and whether causal spike history improves that prediction beyond example count, event count, sequence length, and padding.

The implementation and protocol are present, but the six Roihu executions and their aggregation have not been committed. No scientific conclusion, runtime or energy value, fitted coefficient, prediction improvement, spike-history decision, statistical-significance claim, or novelty claim is available. This record must retain that status until the accepted execution records, accounting, cost-model artifacts, and summary are committed.

## Integration with FedAvg

Measurement extends the established client path rather than introducing another trainer. The distributed entry point constructs the established dataset bundle, split, partition, model, client selections, local Adam optimizer, sample-count aggregation, checkpoint selection, and one-time official-test evaluation. When measurement is disabled, the hook is absent and the established random-number consumption, data-loader seeds, iterator order, client selection, update order, aggregation, checkpoints, and record fields are unchanged.

The scientific topology is rank zero on one visible GH200, one distributed process, and one client process per GPU. Rank zero invokes the established `train_client` function through a measurement wrapper. Measurements never control a tensor operation. Sampler threads do not use Torch random functions. The client features needed before assignment are computed from the client training indices and resolved training seed before rank zero assigns work.

Client intervals are identified by dataset, experiment, scientific seed, communication round, selected position, client ID, training seed, execution attempt, and GPU UUID. Accepted client intervals cannot overlap within an attempt. Other interval categories cover model distribution, result collection, aggregation, validation, checkpoint writing, a communication round, and the training execution.

## Scientific matrix

The canonical [manifest](../experiments/resource_measurement/manifest.yaml) expands to exactly six tasks.

| Dataset | Experiment | Seeds | Model | Batch size | Clients selected / total | Rounds |
|---|---|---|---|---:|---:|---:|
| SHD | `shd_lif_client_resource` | 7, 17, 27 | 256/256 LIF | 32 | 10 / 20 | 100 |
| SSC | `ssc_lif_128_client_resource` | 7, 17, 27 | 128/128 LIF | 256 | 10 / 20 | 100 |

Both datasets use one local epoch, Adam at 0.001, sample-count FedAvg, and label-Dirichlet alpha 0.5. The expected accepted client-record count is `2 × 3 × 100 × 10 = 6,000`.

SHD uses the established stratified validation split, 10 ms integration, 700-to-140 input reduction, and the established 256/256 LIF model. Its official test collection is constructed only after validation has selected a checkpoint. SSC uses all 75,466 official training examples, all 9,981 official validation examples, and all 20,382 official test examples, with 10 ms integration, the same input reduction, and the established 128/128 LIF model. Official validation selects the SSC checkpoint, and the official test collection is accessed once afterward. Dataset results remain separate; cross-dataset rows are reported only in the declared joint and transfer evaluations.

The validator rejects another dataset, seed, topology, backend, process count, GPU count, MPS setting, sampling interval, idle duration, model, batch size, client protocol, round count, example cap, batch cap, validation policy, or official-test policy.

## Timing boundaries

Every client interval begins with a monotonic nanosecond timestamp. Data-loader iterator waits are timed around batch retrieval. CUDA events delimit GPU work on the active stream, and CUDA is synchronized before the final event duration is read. The interval ends with another monotonic nanosecond timestamp.

For an accepted interval,

\[
t_{wall}=t_{data\ wait}+t_{CUDA}+t_{residual\ host}.
\]

The residual is computed from the other three values, and reconciliation uses a declared floating-point tolerance. The CPU test adapter supplies controlled event durations; it does not represent CUDA availability. CUDA-event behavior and NVML sampling require a compatible accelerator host.

## Persistent power sampling and idle reference

One persistent NVML sampler resolves the allocated GPU by UUID rather than assuming CUDA index zero is NVML index zero. It writes each sample to `device_samples.jsonl` and flushes incrementally. A sample contains monotonic nanoseconds, UTC time, GPU UUID, power, GPU and memory utilization, allocated device memory, temperature, optional graphics and memory clocks, optional cumulative hardware energy, backend, configured interval, execution attempt, and error status.

The scientific sampling interval is exactly 100 ms. Requested measurement fails when NVML is unavailable, the visible-device count is not one, UUID resolution is ambiguous, or required fields cannot be sampled. Shutdown closes and flushes records on ordinary completion, Python exceptions, `SIGTERM`, `SIGINT`, and Slurm cancellation.

Each dataset/seed attempt records 30 seconds of idle power after model, CUDA, and measurement initialization and another 30 seconds after evaluation. Data loading, training, validation, and checkpoint I/O are excluded from these idle intervals. Pre-execution and post-execution medians and temperature drift remain separate. The client idle reference is the median of the combined accepted idle samples, while every original idle sample is preserved.

## Device-energy integration

Power is interpolated at each interval boundary from adjacent samples. A sample immediately before and after the interval is mandatory, duplicated or nonmonotonic timestamps are rejected, and a gap above 2.5 times the configured interval invalidates coverage. Nanoseconds are converted to seconds once.

Gross device energy is the trapezoidal integral

\[
E_{gross}=\sum_i \frac{P_i+P_{i+1}}{2}(t_{i+1}-t_i),
\]

and idle-adjusted energy is retained separately as

\[
E_{dynamic}=\sum_i \max\left(\frac{P_i+P_{i+1}}{2}-P_{idle},0\right)(t_{i+1}-t_i).
\]

Both are expressed in joules. Gross energy is never replaced by the idle-adjusted quantity. Sample count, coverage, maximum gap, and an optional cumulative-device-energy difference are recorded. Missing samples are not fabricated.

Execution energy is reconciled into client training, aggregation, validation, checkpoint writing, other measured work, idle-reference contribution, and any unattributed interval. Model distribution and result collection belong to other measured work. A declared interpolation tolerance permits only numerical boundary effects; reconciliation does not force an unexplained remainder to zero. Allocated GPU-hours, utilization, billing units, and device energy are distinct quantities.

## Measurement calibration

The [SHD resource configuration](../experiments/resource_measurement/shd/lif_client_resource.yaml) defines the training-only calibration client. The calibration never constructs or accesses the official test dataset. At least ten paired repetitions alternate measured-first and unmeasured-first order. Before each repetition it restores identical model parameters, Python state, NumPy state, Torch CPU state, Torch CUDA state, and deterministic loader order.

Calibration passes only if all of these conditions hold:

- median paired runtime overhead is at most 2%;
- at least 90% of representative measured intervals contain ten or more samples;
- there are no sampling errors;
- exactly one GPU UUID is observed;
- measured and unmeasured parameter updates are numerically identical.

All individual durations, relative overheads, sample counts, UUIDs, and update comparisons are recorded in `instrumentation_calibration.json`. The scientific runner requires a passing artifact and never changes the 100 ms interval in response to calibration.

## Client-resource schema

Features available before any client execution are example count, local batch count, raw input events, mean/median/maximum sequence length, valid time bins, estimated padded time bins, padding fraction, event density, represented class count, label entropy, communication round, dataset identity, model identity, and parameter count. They are computed only from that client's training indices. Deterministic padded-work estimation reproduces the client's seeded batch order without constructing the official test dataset.

Before assignment, the record also indicates whether earlier accepted observations exist and how many. Features available only from prior observations are previous duration, gross energy, idle-adjusted energy, layer spike rates, spikes per example, exponentially weighted duration, energy, and spike rates, observation count, and missing-history indicators. Coefficient candidates 0.10, 0.30, 0.50, 0.70, and 0.90 are declared before execution. The selected coefficient is chosen using fitting and client-grouped validation observations only, with seed 27 excluded.

Values available only after the current client finishes are actual batches, presented examples, valid and padded time bins, padding fraction, input events, layer-one and layer-two spike counts and rates, wall time, data-wait time, CUDA-event time, residual host time, gross and idle-adjusted energy, and peak allocated and reserved CUDA memory. Current-execution spike values are never scheduler inputs.

Each record carries an availability dictionary so these three feature times are machine-readable. Validation and official-test examples are rejected from client features.

## Causal history and leakage protection

Rows are ordered by dataset, scientific seed, communication round, selected position, and client ID. Before predicting round `t`, a client's history is constructed only from accepted observations for the same dataset, seed, and client in rounds less than `t`. Later rows are therefore unable to alter an earlier feature row. Client ID is used for grouping and history lookup but is never a predictor.

Seeds 7 and 17 provide fitting and client-grouped model selection. Seed 27 remains outside fitting and selection. The evaluation settings are SHD within-dataset, SSC within-dataset, joint fitting and evaluation, SHD-to-SSC transfer, SSC-to-SHD transfer, and prequential seed-27 evaluation. Exact row identities and hashes are stored for fitting, validation, and evaluation collections.

## Cost models and evaluation

The candidates are:

1. a fitting-set median constant;
2. a size model using example and batch count;
3. an event-structure model adding events, sequence statistics, valid and padded time bins, padding fraction, and density;
4. a historical-spike model adding only causal spike, duration, energy, and missing-history features;
5. a diagnostic oracle that may use current-execution spike information and is prohibited from scheduler export.

Implemented regressors are the median predictor, standardized ridge regression over declared penalties, robust linear regression, and optional log-target variants chosen by grouped validation. Means and scales come only from fitting rows; constant features use a safe scale. JSON model artifacts contain feature order, coefficients, intercept, standardization, target transform, fitting row hashes, dataset and seed identities, validation decision, and software and Git provenance. They contain no executable serialization. Reloaded JSON must reproduce stored predictions within strict tolerance.

The required targets are client wall seconds, CUDA-event seconds, gross device joules, and idle-adjusted device joules. Wall time is primary for assignment and gross energy is primary for energy interpretation. Every target and setting reports mean absolute error, root mean squared error, median and 90th-percentile absolute percentage error, R-squared, Spearman correlation, mean and median signed error, and count. Percentage errors use a declared positive denominator floor. Slices cover dataset, seed, joint rows, client-size quartiles, sequence-length quartiles, round intervals, clients without history, and clients with history.

## Spike-history decision

Historical spikes are adopted only if the untouched seed-27 evaluation meets every predeclared condition: median absolute runtime error improves by at least 5%, 90th-percentile runtime error does not worsen, rank correlation improves or remains within tolerance, both SHD and SSC benefit, prediction time is negligible beside client training, and offline assignment is closer to the measured oracle. If any condition fails, the recorded decision is `spike_history_not_adopted`, the strongest event/size model is exported, and the spike result remains valid negative evidence. The decision values cannot be altered after inspecting seed 27.

## Deterministic offline assignment evaluation

No production scheduler is introduced. For the already selected clients in every measured round, the analysis compares round-robin, example-count longest-first, predicted-cost longest-first, and measured-cost oracle assignment for two and four candidate processes. Every client is assigned once. Reports include predicted and measured process loads, predicted and measured makespan, oracle makespan, regret, load imbalance, and assignment-computation time.

## Interruption, resumption, and acceptance

Every power sample, interval, idle reference, and client record carries an execution-attempt identity. An interval interrupted during a client is retained in attempt records but is not accepted. The established training rule repeats an incomplete round; accepted prior rounds remain usable. An interrupted attempt uses only its accepted 30-second pre-execution idle samples for its attempt-local baseline, allowing immediate sampler shutdown. The completed resumed execution must still contain a 30-second post-evaluation idle interval. Exclusion records state why interrupted clients or rounds were omitted. Resumed accepted records keep their original attempt and GPU UUID, and samples are filtered by attempt so traces from separate attempts are never integrated together.

Each scientific run writes `measurement_config.json`, `measurement_acceptance.json`, `idle_power.json`, `device_samples.jsonl`, `execution_intervals.jsonl`, `client_resource_records.jsonl`, `excluded_intervals.jsonl`, and `calibration_reference.json`, beside established resolved configuration, Git provenance, checkpoints, and federated records. Raw power samples are not stored in `final_metrics.json`.

The collection can be valid only with six completed compatible runs, exactly 6,000 accepted client records, three seeds per dataset, passing attempt-specific calibration, complete timing and energy coverage, no leakage, one official-test access per run, finite metrics, model JSON prediction reproduction, and complete Git, configuration, hardware, Slurm, and input-hash provenance. If compatible resumption spans allocations, every parent allocation and its allocated GPU-hours remain in accounting. Execution completion, measurement completeness, energy completeness, and scientific hypothesis outcome remain distinct. Accuracy is not a completion gate, and a valid collection does not predetermine the spike-history decision.

## Planned artifacts and limitations

The planned summary outputs are `resource_measurement_summary.{json,csv,md}`, `cost_model_evaluation.{json,csv,md}`, `client_cost_model.json`, an accepted `energy_cost_model.json`, `assignment_readiness.json`, `instrumentation_calibration.json`, ten deterministic figures, and one source-data CSV per figure.

Interpretation will be limited by three seeds, one HPC system, single-node execution, two datasets and two model sizes, 100 ms power sampling, NVML device-level rather than component-level energy, and offline rather than deployed assignment. Concurrent system load may vary. No direct energy comparison to another hardware platform, statistical-significance conclusion, causal explanation, multinode scalability claim, or thesis novelty follows from this protocol alone.

Submission, monitoring, accounting, fitting, and summary commands are in the [reproducibility guide](../docs/reproducibility.md#client-resource-measurement-and-cost-estimation). Roihu allocation and interpreter requirements are in the [environment guide](../environment/roihu/README.md#client-resource-measurement-allocation).
