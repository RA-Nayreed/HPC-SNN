# Experimental protocol

## Centralized evaluation matrix

The canonical manifest is `experiments/centralized/manifest.yaml`. It contains six mandatory `scientific_evaluation` configurations crossed with seeds 7, 17, and 27, producing 18 independent single-GPU tasks:

| Experiment | Dataset and model | Attention | Protocol |
|---|---|---|---|
| shd_lif_independent_evaluation | SHD 256/256 LIF | none | independent_evaluation |
| shd_dcls_published_protocol | SHD 256/256 DCLS | none | published_protocol |
| shd_pfa_equation_independent_evaluation | SHD 256/256 LIF | equation | independent_evaluation |
| shd_pfa_public_published_protocol | SHD 256/256 LIF | public_behavior | published_protocol |
| ssc_lif_128_independent_evaluation | SSC 128/128 LIF | none | independent_evaluation |
| ssc_pfa_equation_128_independent_evaluation | SSC 128/128 LIF | equation | independent_evaluation |

Every canonical configuration uses CUDA, all examples required by its protocol, null train, validation, and test batch limits, eight data-loader workers, persistent workers, and an explicit early-stopping setting. The SSC 512-neuron model is outside the current evaluation scope.

## Split policies

For SHD, `independent_evaluation` creates a seed-specific stratified validation split from the official training file. Checkpoint selection uses only that derived validation split. The official SHD test file is constructed and evaluated only after checkpoint selection.

For SSC, `independent_evaluation` uses the official training and validation files for fitting and checkpoint selection. The official SSC test file is constructed and evaluated only after checkpoint selection.

For SHD, `published_protocol` reproduces source evaluation behaviour by using the official test file for model selection and evaluating the selected checkpoint on that file again. This monitoring is recorded explicitly in `acceptance.json`. It is a paper-compatible reproduction measurement, not an unbiased held-out estimate.

Results from `independent_evaluation` and `published_protocol` are never pooled. The centralized summarizer groups by manifest experiment and verifies each resolved protocol.

## PfA variants and model boundary

Equation PfA and pinned public-code behaviour are separate treatments until their variance-versus-squared-deviation discrepancy is resolved. DCLS and PfA are also separate: no verified source in the repository specifies a combined DCLS+PfA architecture, so `dcls_shd` rejects PfA.

## Terminal conditions

Independent evaluations permit validation early stopping with patience 20. Published-protocol configurations use fixed 150-epoch training with `early_stop_patience: null`. An evaluation is terminal only after every configured epoch or a documented valid early-stop condition. Interruption is not completion; `--resume-auto` continues from `checkpoints/last.pt`.

## Collected evidence

All 18 mandatory executions completed for the six experiments and seeds 7, 17, and 27. The generated [centralized summary](../results/centralized/centralized_summary.md) reports the aggregated measurements while preserving the protocol distinctions defined above. Observed results do not alter the split, selection, or acceptance rules in this methods document.

## Federated SHD LIF reference

SHD is the sole dataset in the FedAvg baseline because it has a validated centralized LIF comparison and permits federated correctness to be examined without changing the dataset, model, and aggregation algorithm together. SSC remains reserved for cross-dataset generalization and resource evaluation. The ordinary two-hidden-layer LIF model is used before any attention mechanism. DCLS, PfA, adaptive attention, hierarchical aggregation, multiple GPUs, and multiple nodes are excluded. This baseline is reference infrastructure and is not a novelty claim.

The manifest `experiments/federated_baselines/manifest.yaml` contains two treatments: 20 clients with label-wise Dirichlet alpha 0.5 and either 10 selected clients (50%) or 5 selected clients (25%) per round. Seeds 7, 17, and 27 produce six scientific executions. Each execution uses 100 rounds, one local epoch, batch size 32, Adam learning rate 0.001, zero weight decay, gradient clipping at 1, and no learning-rate scheduler or early stopping.

Both treatments use the official `shd_train.h5` and `shd_test.h5` files, 20 classes, 10 ms temporal integration, and deterministic reduction from 700 cochlear channels to 140 inputs. The model has two hidden LIF layers with 256 neurons each, tau 10.05, threshold 1, subtractive detached reset, an ATan surrogate with alpha 5, dropout 0.4, and no batch normalization. The global model starts from seed-specific random initialization rather than a centralized checkpoint.

### Split and partition isolation

The shared SHD `independent_evaluation` splitter creates the seed-specific stratified 10% validation split. Validation indices are removed before client partitioning. The Dirichlet procedure operates only on the remaining training indices, assigns every eligible index exactly once, rejects any construction with a client below 32 examples, and uses a finite deterministic resampling limit. Validation and official-test examples are never assigned to clients.

The partition artifact records exact client indices, client sizes, per-client class counts, size statistics, label entropy, construction attempts, dataset identity, split identity, and integrity checks. For a given seed, the two participation treatments must have identical split, partition, and model-initialization identities.

### Client selection and optimization

A separate selection generator produces one deterministic permutation of the 20 clients for each round. The 25% treatment uses the first five clients and the 50% treatment uses the first ten, making the lower-participation selection a subset of the higher-participation selection. Client data ordering and dropout use separate per-round, per-client seeds and cannot advance the selection generator.

Every selected client receives an isolated copy of the current global model. A new Adam optimizer is created for that selection; optimizer state is not retained across selections and is not aggregated. The global model begins from random initialization associated with the scientific seed and never from a centralized checkpoint. SNN state is reset between batches, clients, validation, and final test evaluation.

### Federated Averaging

For selected clients `k`, the server computes:

`w_next = sum(n_k * w_k) / sum(n_k)`

where `n_k` is the number of training examples owned by client `k`. Aggregation weights must be finite, nonnegative, and sum to one. Empty updates, incompatible state dictionaries, non-finite values, and unsafe non-floating state differences are rejected.

### Selection and official-test access

Global validation accuracy selects `checkpoints/best.pt`, while `checkpoints/last.pt` records every completed round for resumption. The official SHD test dataset is not constructed or accessed during communication rounds. It is constructed only after all 100 rounds and evaluated exactly once using the selected global checkpoint. The centralized SHD LIF accuracy is reported only as contextual evidence and does not determine completion or scientific status.

### Communication definition

Logical communication includes one global-model download and one client-model upload for every selected client. Bytes equal each communicated tensor's element count multiplied by its element size. Optimizer state, dataset transfer, checkpoint I/O, and telemetry are excluded. This value is a deterministic accounting quantity, not measured network traffic.

No compression, quantization, sparsification, or event-based communication method is part of this baseline. With the model and round count fixed, doubling selected clients doubles the logical communication count. Logical communication is neither physical network measurement nor energy consumption.

### Completion rules

Completion requires all 100 rounds, valid partition integrity, 20 clients, the configured number of distinct selected clients in each round, finite records, valid aggregation weights, both nonempty checkpoints, nonempty client and round logs, complete and nonduplicated assignment of eligible training indices, validation and test isolation, exactly one official-test evaluation after model selection, Git and configuration provenance, scientific identities, and consistent logical communication accounting. Accuracy is not a completion condition. A missing verified reproduction target yields `scientific_status: not_claimed`, not execution failure.

### Collected federated evidence

All six mandatory scientific executions completed and passed aggregation validation for both treatments and seeds 7, 17, and 27. The [federated summary](../results/federated/federated_summary.md) records the measurements, and the [federated scientific record](../thesis_records/federated_baseline.md) separates evidence, interpretation, and limitations. Observed results do not alter the partition, selection, test-isolation, or acceptance rules in this methods document.

## Single-node distributed FedAvg protocol

The authoritative physical-device collection is `experiments/distributed_evaluation/manifest.yaml`. Its eight treatments crossed with seeds 7, 17, and 27 produce 24 tasks:

| Workload | Physical GPUs | Client processes per GPU | Total processes | Tasks |
|---|---|---:|---:|---:|
| SHD | 1, 2, 4 | 1 | 1, 2, 4 | 9 |
| SSC | 1, 2, 4 | 1 | 1, 2, 4 | 9 |
| CIFAR-10 | 1, 2 | 1 | 1, 2 | 6 |

The separate `experiments/device_capacity_evaluation/manifest.yaml` contains nine SHD tasks: one physical GPU with one, two, or four client processes, each at seeds 7, 17, and 27. Two and four processes use CUDA MPS and Gloo control. Capacity results are not pooled with exclusive one-process-per-GPU results and are not used as authoritative scientific-accuracy comparisons.

### Workload protocols

SHD retains 20 clients, ten selected per round, 100 rounds, one local epoch, local batch 32, Adam at `0.001`, gradient clipping 1, example-count FedAvg, a seed-specific stratified validation split removed before client partitioning, best-validation checkpoint selection, and one later access to the official test collection.

SSC uses all 75,466 official training examples for a 20-client label-Dirichlet alpha-0.5 partition, selects ten clients, uses one local epoch, local batch 256, the 128/128 LIF model, 700-to-140 channel reduction, 10 ms bins, and example-count FedAvg. The official 9,981-example validation collection selects the checkpoint, followed by one evaluation of all 20,382 official test examples.

CIFAR-10 uses the non-IID alpha-0.5 S-VGG9 BNTT configuration with ten clients, two selected per round, five local epochs, 20 timesteps, uniform aggregation, no internal validation collection, round-100 selection, and one evaluation of all 10,000 official test examples. CIFAR-10 has no four-GPU treatment because each round selects two clients.

Each run constructs or accesses the official test collection exactly once after checkpoint selection. Nonzero ranks do not construct validation or official-test datasets. SHD, SSC, and CIFAR-10 remain separate groups in every summary; cross-dataset pooling is prohibited.

### Coordination and paired comparisons

Exclusive execution uses one node, one process per physical GPU, NCCL, synchronous rounds, selected-order round-robin assignment, and aggregation in selected-client order. Rank zero selects clients. Client position `i` is assigned to process `i % process_count`. Rank zero restores received updates to the established selected-client order, invokes the configured FedAvg aggregation once, performs configured validation and checkpoint selection, and evaluates the official test collection. SHD and SSC use `w_next = sum(n_k * w_k) / sum(n_k)`; CIFAR-10 uses equal weights for its two selected clients.

Client randomness derives from scientific seed streams, communication round, and client identity. It is independent of physical-device count, client-process count, process rank, device index, topology, arrival order, and completion order. Within each workload and seed, the one-GPU distributed run is the reference. Comparisons are paired by seed; sequential paths and other workloads are not valid denominators.

Structural identity is exact only when scientific, dataset, split, partition, initialization, selected-client order, client seeds, example counts, aggregation, logical communication, checkpoint policy, and official-test access records agree with the paired reference and no structural difference is recorded. Numerical identity is exact only when selected checkpoints match, maximum absolute and relative parameter differences are zero, and official-test accuracy and macro-F1 differences are zero. Prediction agreement is checked when prediction records exist. No tolerance is inferred from similar accuracy.

The completed physical-device collection has exact structural and numerical identity for every paired treatment. The capacity collection has exact structural identity, but every two- and four-process comparison records `difference_observed`; it therefore does not establish numerical transparency under MPS.

### Measurement definitions

- Total runtime is the sum of communication-round total durations.
- Paired speedup for seed `s` is `T_one_process(s) / T_treatment(s)` within the same workload and collection.
- Parallel efficiency is paired speedup divided by total client-process count. For exclusive execution, process count equals physical-GPU count; for capacity execution, several processes share one GPU.
- GPU utilization is the arithmetic mean of job-level physical-device `nvidia-smi` utilization samples collected every two seconds. Treatment values are means ± sample standard deviations across the three seeds.
- Peak CUDA allocation is the largest per-process `torch.cuda` allocated-tensor peak recorded by a run. Peak CUDA reservation is recorded separately. Neither quantity is GPU HBM capacity, Grace CPU memory, or Slurm memory.
- Per-round process load imbalance is `(maximum busy time - minimum busy time) / maximum busy time`; the run value is the largest round value. A one-process run has value zero.
- Allocated GPU-hours and Slurm elapsed time come from scheduler accounting. Pending queue time is excluded.
- Logical communication counts model downloads and uploads. Process-control and state movement are separate execution data movement and are not physical network-traffic measurements.

`valid` means a collection passed its declared consistency gates. It does not imply numerical equivalence for every treatment. Likewise, `not_claimed` and `equivalence_not_established` are scientific-status outcomes, not execution failures.

## CIFAR-10 Fed-SNN evaluation

The active manifest `experiments/published_fedsnn/manifest.yaml` contains the two CIFAR-10 SNN 10/2 Table I treatments crossed with seeds 7, 17, and 27. `cifar10_fedsnn_paper_reported_iid_evaluation` assigns the complete training collection evenly to ten clients with nearly equal class proportions. `cifar10_fedsnn_paper_reported_noniid_evaluation` uses the released balanced label-Dirichlet algorithm with alpha 0.5 and minimum client size ten. Distribution and its associated provenance fields are the only intended treatment difference.

Both configurations use ten total clients, two selected clients, 100 communication rounds, five local epochs, local batch size 32, SGD learning rate 0.1, momentum 0.95, source-default weight decay `1e-4`, no gradient clipping, uniform selected-client averaging, and final-round checkpoint selection. The paper supplies five local epochs, momentum 0.95, reductions after rounds 40, 60, and 80, final reporting after round 100, all 50,000 training examples, and three repetitions. The released two-local-epoch IID command is implementation evidence, not an active Table I treatment.

CIFAR-10 is represented in `[-1,1]` as `2x-1`, without crop or flip. Each input timestep uses `1(2U <= |x|) sign(x)` with an explicit generator. S-VGG9 uses seven convolutions, two linear layers, 20 timesteps, leak 0.95, threshold 1, the scale-0.3 triangular surrogate, BNTT momentum 0.1 and epsilon `1e-4`, temporal-mean readout, and Xavier uniform gain-2 initialization for convolution and linear weights only.

Incomplete local batches are dropped. Client records preserve both partition population and examples presented after dropping remainders. Uniform aggregation includes every floating tensor in the state dictionary, including BNTT running means and variances. SHD continues using retained local batches and example-count aggregation.

All 50,000 standard training images are eligible for client partitioning and occur in exactly one client partition. No internal validation collection or validation loader exists. The round-100 `last.pt` checkpoint is selected neutrally as the final-round checkpoint; a `best.pt` checkpoint and best-validation metric are neither created nor required. The official 10,000-image test collection, which the paper calls its validation collection, is inaccessible during training and is evaluated exactly once after round 100. This is not procedure-for-procedure equivalent to the released program, which monitors that collection during training.

The configured rate is 0.1 for rounds 1–40, 0.02 for rounds 41–60, 0.004 for rounds 61–80, and 0.0008 for rounds 81–100. The reduction occurs only after each boundary round has completed. The 20-timestep value, signed representation, dropped local remainder, uniform averaging, Xavier gain-2 initialization, temporal-mean readout, BNTT epsilon `1e-4`, and weight decay `1e-4` come from the pinned source; the paper does not state the SNN timestep count or weight decay.

The paper’s 76.44% IID and 73.94% non-IID values are descriptive only. No tolerance or acceptance threshold is configured. Reports retain `equivalence_not_established` and show signed and absolute percentage-point differences without automatically declaring reproduction success.

Remaining discrepancies include the paper’s example-count aggregation equation versus released `Fed.py` uniform aggregation, omitted paper values for timesteps and SNN weight decay, released-source official-test monitoring versus one final project evaluation, and the project training selected clients only while the authors compute every client update before choosing uploaded states.

### Collected evidence

All six mandatory executions completed successfully for seeds 7, 17, and 27. Every run selected round 100, used all 50,000 training examples with zero internal-validation examples, and accessed the complete 10,000-example official test collection exactly once after training.

| Distribution | Seed 7 | Seed 17 | Seed 27 | Mean ± sample SD | Mean macro-F1 | Paper reference | Mean signed difference |
|---|---:|---:|---:|---:|---:|---:|---:|
| IID | 81.50% | 82.16% | 81.55% | 81.7367% ± 0.3675 pp | 81.7070% | 76.44% | +5.2967 pp |
| Label-Dirichlet non-IID, alpha 0.5 | 72.01% | 75.80% | 73.32% | 73.7100% ± 1.9249 pp | 73.5136% | 73.94% | -0.2300 pp |

The [generated summary](../results/fedsnn_paper_evaluation/published_fedsnn_summary.md) is the active Fed-SNN evidence. The implementation learns successfully. The non-IID mean closely agrees with the paper’s descriptive value, while the IID mean exceeds its reference by 5.2967 percentage points. The descriptive IID-to-non-IID reduction is 8.0267 percentage points. Macro-F1 tracks accuracy closely and supplies no evidence of class collapse.

These three-seed observations do not establish statistical significance, causality, novelty, energy efficiency, exact source equivalence, or an exact reproduction pass. Scientific status remains `equivalence_not_established`.

The earlier CIFAR-10 implementation completed with 18.23–26.79% official-test accuracy but used an incompatible representation, encoder, readout, initialization, timestep count, BNTT epsilon, loader rule, partition, and aggregation rule. Its configuration is retained outside the active manifest, and its generated evidence is unchanged.

## Client resource measurement protocol

### Predeclared matrix and topology

The canonical [resource manifest](../experiments/resource_measurement/manifest.yaml) expands to exactly six executions: SHD 256/256 LIF FedAvg and SSC 128/128 LIF FedAvg, each with seeds 7, 17, and 27. Every execution reserves one physical GH200, starts one distributed process, uses one client process, uses NCCL, and disables CUDA MPS. The six tasks run sequentially in one Slurm allocation when uninterrupted.

| Dataset | Training / validation / test source | Batch size | Clients selected / total | Rounds | Test policy |
|---|---|---:|---:|---:|---|
| SHD | official training file with established stratified validation / official test | 32 | 10 / 20 | 100 | construct and access once after checkpoint selection |
| SSC | 75,466 official training / 9,981 official validation / 20,382 official test examples | 256 | 10 / 20 | 100 | access once after official-validation checkpoint selection |

Both protocols use label-Dirichlet alpha 0.5, one local epoch, Adam learning rate 0.001, sample-count aggregation, 10 ms integration, and 700-to-140 channel reduction. There are no example or batch caps. Static client features use training indices only. Validation and official-test identities are rejected from those features, and official-test access during communication rounds is rejected.

The predeclared accepted row count is 2 datasets × 3 seeds × 100 rounds × 10 clients = 6,000. SHD and SSC results are never pooled for dataset-specific reporting. Joint and transfer evaluations are explicitly labelled. The observed collection contains exactly those 6,000 rows: 1,000 accepted client measurements for every dataset/seed execution.

### Paired identities and measurement transparency

The measurement-disabled and measurement-enabled calibration repetitions restore the same model parameters, Python, NumPy, Torch CPU and Torch CUDA random states, and data-loader order. Client selection, training seeds, updates, update ordering, aggregation, and checkpoint selection remain independent of measurement values.

Calibration update identity means the two state dictionaries have the same parameter keys, shapes, dtypes, and numerically identical tensor values under the declared strict comparison. It is a calibration condition, not a claim that an eventual cost model is exact. A missing scientific conclusion is not an execution failure.

The resource collection is not an MPS treatment. Prior one-GPU MPS capacity evidence remains separate because packed client processes were numerically different. No MPS row may enter resource measurement or client-cost fitting.

### Timing, power, energy, memory, and load definitions

- Client wall time is the interval between monotonic nanosecond boundaries.
- Data-wait time is time spent obtaining client batches.
- CUDA-event time is the summed active-stream event duration after CUDA synchronization.
- Residual host time is wall time minus data-wait and CUDA-event time.
- Gross energy is trapezoidal integration of boundary-interpolated device power.
- Idle-adjusted energy integrates the positive part of power minus the accepted idle median and remains separate from gross energy.
- GPU utilization and memory utilization are 100 ms NVML samples, not energy estimates.
- Peak allocated CUDA memory is live tensor allocation; peak reserved CUDA memory is allocator reservation. Neither is GPU HBM capacity, Grace memory, or Slurm memory.
- Assignment makespan is the maximum measured process load in a candidate assignment.
- Load imbalance is the declared dispersion of process loads relative to their mean.
- Makespan regret is candidate measured makespan minus measured-oracle makespan.

Each energy interval requires a sample immediately before and after its boundary and rejects a sample gap above 2.5 configured intervals. Gross and idle-adjusted joules, sample count, coverage, and optional hardware cumulative-energy difference are retained. Execution reconciliation reports client, aggregation, validation, checkpoint, other, idle-reference, and unattributed energy without substituting allocated GPU-hours.

### Calibration and idle protocol

The 100 ms interval is accepted only if at least ten alternating paired SHD training-client repetitions produce median runtime overhead at most 2%, at least 90% of measured client intervals contain ten or more samples, no sampling error occurs, exactly one UUID is observed, and parameter updates are numerically identical. The official test dataset is never constructed by calibration.

Every scientific attempt records 30 seconds of pre-execution idle power after CUDA and measurement initialization and 30 seconds after evaluation. No data loading, validation, checkpoint I/O, or training occurs within an idle interval. Pre/post medians, temperature drift, the combined accepted median, and original samples are preserved per attempt.

### Predictor availability and causal history

Pre-assignment static predictors are example count, batch count, input-event count, sequence-length statistics, valid and estimated padded time bins, padding fraction, event density, represented classes, label entropy, round, dataset, model, and parameter count. Padded work reproduces the client's deterministic seeded batch ordering.

Earlier-observation predictors include previous and exponentially weighted duration, gross and idle-adjusted energy, layer spike rates, spikes per example, observation count, and missing-history indicators. For round t, only accepted rows for that client in rounds less than t are permitted. Actual current batches, examples, timesteps, input events, spikes, timing, energy, and memory are post-execution observations and cannot be scheduler predictors.

Seeds 7 and 17 are the fitting and client-grouped validation collection. Seed 27 is absent from fitting and model selection. The protocol evaluates SHD within-dataset, SSC within-dataset, joint, both directed transfers, and prequential seed-27 predictions. Client ID is grouping metadata, not a predictor.

### Models, decision, and offline assignment

The candidate hierarchy is fitting-set median, size, event structure, causal historical spikes, and a diagnostic current-spike oracle. Standardized ridge penalties and historical exponential coefficients 0.10, 0.30, 0.50, 0.70, and 0.90 are declared before evaluation. The coefficient and regression variant are chosen only through client-grouped fitting/validation observations from seeds 7 and 17. Robust linear and optional log-target variants follow the same separation. JSON serialization records fit-derived standardization and row hashes and must reproduce stored predictions.

Historical spikes are adopted only if seed 27 shows at least 5% improvement in median absolute runtime error, no worsening of 90th-percentile runtime error, rank correlation within the declared tolerance or better, benefit on both datasets, negligible prediction time, and assignment closer to the measured oracle. Failure of any condition records spike_history_not_adopted and exports the strongest event/size model. The diagnostic oracle is never exported.

Offline assignment uses the clients already selected for each round and compares round-robin, example-count longest-first, predicted-cost longest-first, and measured-cost oracle at process counts two and four. It assigns each client exactly once and has no production scheduling side effect.

### Acceptance and evidence status

A valid summary requires six completed compatible runs, 6,000 accepted rows, passing calibration, complete timing and energy coverage, no leakage, one official-test access per run, finite metrics, JSON prediction reproduction, and complete Git, configuration, hardware, Slurm, and input-hash provenance. Incomplete intervals remain in attempt records with exclusion reasons and cannot enter fitting.

Execution completion, measurement completeness, energy completeness, and hypothesis outcome are separate fields. Accuracy is not a completion condition. A valid summary may adopt or reject spike history.

### Observed evidence

The [committed summary](../results/resource_measurement/resource_measurement_summary.json) reports `valid: true`, six completed executions, 6,000 accepted client records, three SHD and three SSC executions, seeds 7, 17, and 27, and true calibration, power-coverage, timing-completeness, energy-integration, official-test-isolation, finite-metric, and model-JSON-reload fields. The preserved execution commit is `3ddae173c89125bc69922d80bde5732ed6cd050e`; summary, run, and exported-model provenance agree.

The accepted calibration had ten paired repetitions and one measured and one unmeasured warm-up execution before those pairs. Warm-up state was restored before each execution and warm-up observations were excluded from both the paired observations and the overhead statistic. The outcome passed with median relative overhead `0.01713823842158517`, sample coverage `1.0`, identical numerical updates, no sampling errors, the single UUID `GPU-f50ec698-57aa-4b59-108c-fa678abbc391`, and zero official-test accesses. The measured overhead is nonzero and remains below the declared `0.02` threshold.

Slurm job `291481` is `COMPLETED` with exit code `0:0`. The one-GH200 allocation used 72 CPU cores and 217086M Slurm memory for 34,051 seconds, or `9.45861111111111` allocated GPU-hours. Separately, the summary records `33217.69320165331` seconds of internal execution time, `11457.10155192` seconds of summed client wall time, and `8159.062962005615` seconds of CUDA-event time. Pending time is null and is not included in allocation execution time.

Accepted client-training intervals contained `1883612.6948749206 J` of gross device energy and `52380.5986610567 J` of idle-adjusted device energy. These client-interval totals are neither Slurm accounting quantities nor whole-allocation energy.

The fitting and client-grouped selection collection used seeds 7 and 17; seed 27 remained outside fitting and model selection. The exported wall-time and gross-energy models are `event_structure` ridge models, and the stored JSON models reproduce their predictions. The scheduling model uses example count, batch count, raw input events, mean/median/maximum sequence length, valid and estimated padded time bins, padding fraction, and event density. Client identity and current-execution spike fields are absent.

The spike decision is `spike_history_not_adopted`. SHD passed the median-improvement, tail-error, and rank conditions with median absolute runtime-error improvement fraction `0.3331679722893203`. SSC maintained rank but failed the median-improvement and tail-error conditions, with improvement fraction `-2.233354617899182`. Negligible prediction time and assignment closer to the measured oracle both passed, but the required benefit across both datasets did not.

### Scientific interpretation

The declared consistency gates passed, so the measurement and cost-estimation evidence is accepted. Event-structure features were selected for deployment-oriented prediction. Historical spike information helped the declared SHD runtime comparison but performed substantially worse in the declared SSC comparison, so it was not sufficiently robust across both evaluated datasets for adoption. This is valid dataset-dependent negative evidence rather than an execution failure.

The [offline assignment evidence](../results/resource_measurement/assignment_readiness.json) evaluates process counts two and four without changing a production scheduling policy. The measured-cost oracle and the diagnostic current-execution oracle are unavailable before assignment. Consequently, the offline comparisons do not establish an end-to-end distributed speedup, production scheduling superiority, energy advantage, reduced billing, or behavior beyond these two datasets, these models, this hardware, and this execution.

## Scheduling and hierarchical-reduction evaluation protocol

The scheduling evaluation asks whether the already-selected clients can be assigned more evenly without changing FedAvg. Client selection and its original order occur first. The three assignment treatments are `round_robin`, `example_count_longest_processing_time`, and `event_structure_longest_processing_time`. Equal client costs are resolved by selected position and stable client ID; equal process loads are resolved by client count and global rank. Client-training seeds remain functions only of the scientific seed stream, communication round, and client ID.

The event-structure scheduler loads [the committed ridge model](../results/resource_measurement/client_cost_model.json) once per execution and verifies SHA-256 `78ad111c6997999e017b7b29c07fadfdab86a11a83c128b06e2b8924d4a1471c`. The source collection has 6,000 accepted rows, separated into 4,000 seed-7/17 fitting rows and 2,000 untouched seed-27 evaluation rows. Immutable row identities show that seed 27 did not enter normalization, coefficient fitting, regression-family selection, feature selection, or hyperparameter selection. Seed 27 is post-freeze evaluation/adoption evidence only. The evaluation-provenance artifact is verified against SHA-256 `1488d3542e5b0770ab750b9d3812aeeac86bc091e3a11954e1f8a6a73ba9f924`.

The scheduler's permitted predictors are example and batch counts, raw event count, mean/median/maximum sequence length, valid and estimated padded time bins, padding fraction, and event density. Features use client training data and are computed before client training. Validation and official-test information, client ID, labels, current-execution timing, energy, CUDA, and spike observations are prohibited.

The scheduling matrix crosses SHD and SSC, the three treatments, and seeds 37, 47, and 57: exactly 18 independent executions. For each dataset/seed, the treatments run sequentially as independent processes within one one-node/four-GH200 allocation. The counterbalanced orders are round-robin/example-count/event-structure for seed 37; example-count/event-structure/round-robin for seed 47; and event-structure/round-robin/example-count for seed 57. Pairing is only within dataset, seed, round, and topology.

The hierarchical-reduction matrix holds the event-structure scheduler fixed and compares `flat_ordered` with `node_hierarchical` on exactly two nodes with two GH200 GPUs per node. Seeds and datasets are unchanged, yielding 12 executions. Seed 37 runs flat then hierarchical, seed 47 hierarchical then flat, and seed 57 flat then hierarchical. Both paths preserve the established pre-Week-6 FedAvg policy: uniform or example-count weights are normalized as Python floats before tensor arithmetic; every floating or complex input is cast to float64 and added in selected-client order with its normalized weight as `alpha`; and the completed accumulator is cast back to the original dtype without a post-sum division. Consequently, the historical float64 cast of a complex input also preserves its established imaginary-component behavior. Integral and Boolean buffers require exact agreement and preserve their dtype. The same key, shape, dtype, finite-value, normalized-weight, contribution-coverage, and nonfloating checks apply. Hierarchical reduction forms those same weighted terms locally and changes only their grouping by node.

After YAML composition, every pair is checked against a versioned resolved-configuration whitelist. Scheduling permits only scheduler strategy plus treatment/order and run/output identity differences; hierarchy permits only aggregation topology plus treatment/order and run/output identity differences. Named invariant checks require equal dataset/preprocessing, split, partition, client population/participation/selection count, rounds, model/initialization, optimizer/learning-rate policy, epochs/batch/drop-last, client seed derivation, FedAvg weighting, validation/checkpoint/official-test protocols and one test access, and physical GPU/process/processes-per-GPU counts.

Both collections use `gpumedium` for at most 36 hours and exactly four GH200 GPUs without CUDA MPS. Scheduling requests one node, one task, four workers, and 288 CPU cores. Hierarchy requests two nodes, two tasks with one task per node, two GPUs and 144 CPU cores per task/node, one agent and two workers per node, and four global workers. A preflight enumerates exactly four distinct allocation UUIDs; every NCCL process must map to exactly one of them with the declared node-major rank/local-rank/device mapping before any scientific workload is constructed.

Because treatments execute sequentially, allocation elapsed time and billed GPU-hours are stored once per allocation. Per-treatment records contain internal duration and derived four-GPU exposure only. Allocation reconciliation, with a declared two-second tolerance, requires internal durations plus initialization, between-treatment overhead, remaining overhead, and reconciliation error to equal Slurm allocation elapsed time. Derived exposure is not separately billed Slurm accounting.

Logical federated downloads and uploads remain separate from internal process movement. Intra-node and inter-node byte fields count tensor payloads: flat reduction counts each client model state crossing the corresponding boundary, while hierarchical reduction counts client states sent to node leaders and the sufficient-statistics tensor contribution sent by a remote leader. Serialized process-envelope sizes are retained separately as collection diagnostics. None of these fields is a physical network-interface measurement. Measurements also include scheduler stages, parallel client wall time, collection and reduction stages, validation and checkpoint time, round time, load and waiting, GPU utilization, CUDA memory, and model-sized cross-node payload counts.

The scheduling adoption gate uses the six explicit dataset/seed pairs against round-robin and never pools datasets. Within SHD and SSC separately, the 5% condition is the arithmetic mean of the three paired seed-level runtime reductions and at least two of three reductions must be positive. Every dataset/seed pair must have reduction at least -2%. Dataset-level means must be no worse than example-count scheduling. Exact scientific identity, overhead below 1%, identical predictions and selected checkpoints, and only permitted pre-execution information are additional conjunctive gates. The hierarchy retention gate independently requires exact contribution coverage and weights, structural and tolerance-justified mathematical equivalence, prediction and checkpoint agreement, predicted logical inter-node movement reduction, no runtime regression above 2%, and official-test isolation. Execution validity is separate from either performance decision.

This simulation calculates workload metadata centrally from training data. A real federated deployment would require clients to transmit the ten scheduler fields. They contain no labels, but they can reveal workload volume, sequence shape, padding, and event-density characteristics. Static event summaries may be cached; batch-order padding fields are round-dependent. Raw event tensors are never scheduler metadata and do not leave a client for scheduling. This is metadata exposure, not a privacy-preserving claim.

No scientific result exists for these matrices until all declared Roihu executions, accounting, equivalence checks, and acceptance records complete. The implementation alone does not show that scheduling improves performance or that hierarchical reduction reduces runtime.

## Centralized CIFAR-10 learning verification

`experiments/published_fedsnn/cifar10/centralized_learning_verification.yaml` uses the data representation and S-VGG9 model with a distinct `runs/fedsnn_centralized_verification` identity. It fits the 45,000-example training subset, selects by the 5,000-example validation subset, and evaluates the official test once. This configuration remains an independent model-learning check and is not part of the six-task federated manifest. The completed federated evidence above independently demonstrates that the implementation learns under the declared Table I protocol.

## Week 7 scaling, non-IID, and measured-energy protocol

This is a prospective protocol backed by a locally verified implementation. Scientific execution has not occurred, no Week 7 results exist, and no scaling, energy, accuracy, or adoption conclusion is claimed. Weeks 1–6 and their committed evidence remain unchanged; existing `results/` artifacts are not evidence for either collection below.

### Fixed scientific workload

Both collections cross SHD and SSC with seeds 37, 47, and 57. SHD uses the established 256/256 LIF model, 20 clients, ten selected clients, 100 rounds, one local epoch, Adam at 0.001, batch size 32, the deterministic stratified validation split, and checkpoint selection before one official-test evaluation. SSC uses the established 128/128 LIF model with the same federated counts and optimizer, batch size 256, its complete official train and validation collections, and one post-selection official-test evaluation. Both retain 700-to-140 channel reduction, 10 ms integration, isolated client models, fresh client optimizer state, deterministic initialization and round/client seed derivation, and example-count FedAvg.

Rank zero alone selects clients, restores selected-client order, calls `flat_ordered`, validates, selects checkpoints, writes run-level records, and constructs and evaluates the official test. Client seeds cannot depend on rank, node, device, topology, process count, or arrival order. `example_count_longest_processing_time` is the only assignment policy; `event_structure_longest_processing_time`, `node_hierarchical`, CUDA MPS, MPI, and custom OpenMP are outside this protocol.

### System-scaling collection

`system_scaling_energy_evaluation` contains 24 executions: two datasets by three seeds by these four physical layouts.

| Topology | Nodes | GPUs per node | Total GPUs | Processes |
|---|---:|---:|---:|---:|
| `one_node_one_gpu` | 1 | 1 | 1 | 1 |
| `one_node_two_gpu` | 1 | 2 | 2 | 2 |
| `one_node_four_gpu` | 1 | 4 | 4 | 4 |
| `two_nodes_four_gpus` | 2 | 2 | 4 | 4 |

Every row fixes label-Dirichlet alpha 0.5. Within dataset and seed, partition identity, initialization, selection sequence, client seeds, selected order, weights, and checkpoint rules are invariant. The one-GPU row is the numerical reference. Comparisons retain structural identity, exact and bounded parameter identity, prediction identity, checkpoint identity, and metric identity separately; similar accuracy alone never establishes equivalence.

### Non-IID collection

`non_iid_energy_evaluation` contains 24 executions: two datasets by three seeds by deterministic IID and label-Dirichlet alpha 1.0, 0.5, and 0.1. Every row uses one node, four GPUs, and four processes. The partition is the intended treatment difference; all eligible indices must be assigned exactly once, with population, presented-example, class-count, represented-class, entropy, partition-hash, seed, retry, and repair provenance retained. An invalid extreme partition fails with an explanation instead of dropping clients or examples.

Four treatments share each dataset/seed allocation. Their rotated orders are IID/1.0/0.5/0.1 for seed 37; 1.0/0.1/IID/0.5 for seed 47; and 0.5/IID/0.1/1.0 for seed 57. Every treatment starts a fresh process and scientific state.

### Resolved identity and resumption

The `resolved_leaf_paths_v1` comparison runs after YAML composition and data/output overrides. Scaling permits only treatment/experiment, topology and mapping, rendezvous, execution/output, and allocation-description paths. Non-IID permits only treatment/experiment, distribution/alpha, partition identity/provenance, rotated order, and output identity. Named workload invariants provide a second check.

Resume compatibility includes dataset, distribution, alpha, seed, partition, initialization, selection stream, scheduler, aggregation, topology, process/device mapping, backend, measurement, calibration, and Git identity. Hostname and Slurm allocation remain attempt provenance. An incomplete round is repeated; completed earlier rounds remain accepted. Telemetry never crosses attempt boundaries. Failure stops node samplers, flushes local attempt evidence, clears subgroups and the default process group, and preserves the originating exception.

### Measured energy, calibration, and accounting

One sampler process and telemetry file are owned per node. Canonical prefix-free physical GPU UUIDs provide exact one/two/four-device coverage; MIG, missing, malformed, duplicate, and unexpected UUIDs fail before workload construction. Samples retain monotonic and UTC time, raw/canonical UUID, node, power, utilization, memory, temperature, clocks, cumulative energy when available, backend, interval, sampling status, attempt, and Slurm allocation. Node files are schema-validated and atomically merged in deterministic node/device/time order.

Gross and idle-adjusted energy use boundary-interpolated trapezoids independently on each physical device before summation. Negative power, duplicate or nonmonotonic times, errors, missing boundaries, and gaps above 250 ms are rejected. Same-device client intervals cannot overlap; different-device intervals may. Client training, distribution, collection, aggregation, validation, official test, checkpoint, other, idle-baseline, interrupted, unattributed, and complete-treatment energy remain separate. Integrated joules are never replaced by allocated GPU-hours.

Each topology requires its own compatible instrumentation calibration covering every node, process, sampler, UUID, the 100 ms interval, and execution commit. Warm-ups are excluded; at least ten alternating measured/unmeasured pairs must have identical updates, median overhead at most 2%, at least 90% accepted interval coverage, no sampling errors, and zero official-test accesses.

Allocation accounting stores display and raw job IDs, state, exit, elapsed seconds, TRES, timestamps, nodes, physical GPUs, and billed GPU-hours once. Internal treatment durations and derived GPU exposure are separate. Initialization, between-treatment, remaining overhead, and reconciliation error must close to allocation elapsed time within two seconds. Logical communication and movement are payload accounting, not measured physical network traffic.

### Analysis and acceptance

Scaling summaries pair every topology with one GPU by dataset and seed and report runtime, phase timing, accuracy, macro-F1, selected round, energy, utilization, memory, load, communication, inter-node movement, speedup, efficiency, energy ratio, derived energy-delay product, and numerical classifications. Non-IID summaries pair every treatment with IID by dataset and seed and report validation/test metrics, paired differences, time, population/event/load imbalance, energy, per-round and per-client energy, utilization, memory, and communication. Three-seed aggregates are arithmetic means with sample standard deviation; datasets are never pooled and no significance tests are performed.

Complete, non-monotonic, or negative evidence remains valid when execution and evidence gates pass. The evidence-complete classifications are `system_scaling_energy_characterization_complete` and `non_iid_energy_characterization_complete`; neither encodes a favorable result.
