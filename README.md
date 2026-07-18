# HPC-SNN

Federated adaptive Parameter-free Attention for spiking neural networks.

The centralized SHD and SSC evaluation is complete. It covers official event files, deterministic 10 ms and 140-channel preprocessing, two-layer LIF models, an SHD DCLS reference, equation PfA, and pinned public-code-behaviour PfA. All 18 required executions completed, and the generated [centralized summary](results/centralized/centralized_summary.md) records the three-seed evidence.

## Centralized evaluation matrix

| Experiment | Dataset and model | Attention | Protocol |
|---|---|---|---|
| shd_lif_independent_evaluation | SHD, 256/256 LIF | none | independent_evaluation |
| shd_dcls_published_protocol | SHD, 256/256 DCLS | none | published_protocol |
| shd_pfa_equation_independent_evaluation | SHD, 256/256 LIF | equation PfA | independent_evaluation |
| shd_pfa_public_published_protocol | SHD, 256/256 LIF | public-code behaviour | published_protocol |
| ssc_lif_128_independent_evaluation | SSC, 128/128 LIF | none | independent_evaluation |
| ssc_pfa_equation_128_independent_evaluation | SSC, 128/128 LIF | equation PfA | independent_evaluation |

Every experiment uses seeds 7, 17, and 27, giving 18 independent one-GPU tasks. The SSC 512-neuron model is outside the current evaluation scope. DCLS and PfA are not combined because no verified source in the repository specifies that architecture.

The protocols are not statistically interchangeable. For SHD, `independent_evaluation` creates a deterministic stratified validation split from the official training set and accesses the official test set only after checkpoint selection. For SSC, it uses the official validation split for selection and accesses the official test set afterward. The SHD `published_protocol` reproduces source evaluation behaviour that monitors the official test split during training; those results are labelled as reproduction measurements and are not unbiased held-out estimates.

## Centralized results

Official-test accuracy is reported as mean ± sample standard deviation across seeds 7, 17, and 27:

| Experiment | Protocol | Official-test accuracy |
|---|---|---:|
| SHD LIF | `independent_evaluation` | 76.3693% ± 2.26431% |
| SHD DCLS | `published_protocol` | 91.3722% ± 0.243267% |
| SHD equation PfA | `independent_evaluation` | 78.4305% ± 1.91990% |
| SHD public-behaviour PfA | `published_protocol` | 84.0253% ± 0.0919464% |
| SSC LIF-128 | `independent_evaluation` | 54.8981% ± 0.198346% |
| SSC equation PfA-128 | `independent_evaluation` | 55.7812% ± 0.570910% |

Every scientific status is `not_claimed`: the executions completed, but no verified literature target and tolerance are available for a reproduction decision. This status is neither a reproduction pass nor an execution failure.

The independent-evaluation centralized results provide the reference measurements for subsequent federated experiments. Published-protocol measurements remain separate because they monitor the official SHD test split during training.

## Local setup and checks

Install CPU PyTorch from the official PyTorch index, then install the project:

~~~bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
python -m pip install -e ".[dev]"
ruff check src tests
python -m pytest -q
~~~

Raw datasets and run outputs remain excluded from Git. Automated tests use synthetic data and do not download SHD or SSC.

## Centralized execution on Roihu

~~~bash
bash scripts/slurm/submit_roihu_centralized.sh \
  --work-dir "/scratch/$CSC_PROJECT/$USER/hpc-snn" \
  --max-parallel 4
~~~

Monitor the returned job ID with:

~~~bash
squeue --job <JOB_ID> --array -o "%.18i %.9P %.28j %.2t %.10M %.10l %R"
~~~

After all 18 tasks complete:

~~~bash
fedapfa-summarize-centralized \
  --manifest experiments/centralized/manifest.yaml \
  --runs-root "/scratch/$CSC_PROJECT/$USER/hpc-snn/runs/centralized" \
  --output-dir "/scratch/$CSC_PROJECT/$USER/hpc-snn/results/centralized"
~~~

Execution completion requires a valid terminal condition, finite metrics, both checkpoints, complete-dataset use without batch limits, official test evaluation, the expected model class, and nonempty metrics and logs. Scientific reproduction is separate: it requires a verified reference accuracy and tolerance. Current literature targets remain null because no value is verifiable from repository records, so completed runs report `scientific_status: not_claimed` rather than a reproduction pass.

## Federated SHD LIF reference

The federated SHD LIF evaluation is complete. It uses SHD so that FedAvg can be assessed against the centralized SHD LIF reference without also changing the dataset or model family. The ordinary 256/256 LIF network is evaluated before attention mechanisms; FedAvg is reference infrastructure, not a novelty claim. SSC remains necessary for cross-dataset generalization and resource evidence but is outside this two-treatment matrix.

Accuracy is mean ± sample standard deviation across seeds 7, 17, and 27:

| Participation | Selected / total clients | Best validation accuracy | Official-test accuracy | Logical communication | Seeds |
|---:|---:|---:|---:|---:|---:|
| 50% | 10 / 20 | 88.6846% ± 0.935984% | 69.9647% ± 2.99540% | 856,224,000 bytes | 7, 17, 27 |
| 25% | 5 / 20 | 84.3137% ± 1.76743% | 67.0936% ± 1.21527% | 428,112,000 bytes | 7, 17, 27 |

The [generated federated summary](results/federated/federated_summary.md) provides the aggregation, and the [federated scientific record](thesis_records/federated_baseline.md) documents the methods, evidence, interpretation, and limitations.

For each seed, the established stratified SHD validation split is removed before client partitioning. The remaining training indices are assigned exactly once by deterministic label-wise Dirichlet sampling. Both participation treatments reuse the same split, partition, initial global parameters, and round-specific client permutation; the five-client selection is the prefix of the ten-client selection. Validation selects the global checkpoint, and the official test dataset is constructed only after all 100 communication rounds.

FedAvg uses sample-count weighting, `w_next = sum(n_k * w_k) / sum(n_k)`. Each selected client receives an isolated global-model copy and a newly created Adam optimizer. Optimizer state stays local and is neither retained nor aggregated. Logical communication counts one tensor-model download and upload per selected client; it excludes optimizer state, dataset transfer, checkpoint I/O, and telemetry and is not measured network traffic.

Within this protocol, selecting ten rather than five clients increased mean official-test accuracy by 2.87102 percentage points and used exactly twice the logical communication. Both federated means remained below the centralized SHD LIF independent-evaluation mean of 76.3693%, by 6.4046 and 9.2756 percentage points respectively. These are descriptive three-seed comparisons, not significance or causality claims.

All six executions passed completion checks. Their scientific status is `not_claimed` because no verified published FedAvg target is configured; this is expected and is not an execution failure or a reproduction claim. Federated and centralized measurements remain visibly separate even where their independent-evaluation rules permit contextual comparison.

## Single-node distributed FedAvg evaluation

The canonical [manifest](experiments/distributed_evaluation/manifest.yaml) contains 24 exclusive-device tasks: SHD and SSC each use one, two, and four GPUs, CIFAR-10 uses one and two GPUs, and every treatment uses seeds 7, 17, and 27. CIFAR-10 has no four-GPU treatment because only two clients participate in each round. Configuration composition defines each workload once, and GPU-count treatments change execution placement only.

SHD retains the established 256/256 LIF 20/10 FedAvg protocol, derived validation split, example-count aggregation, and best-validation selection. SSC uses the complete 75,466-example official training collection for 20-client alpha-0.5 partitioning, the official 9,981-example validation collection for selection, the 128/128 LIF model and 256-example local batch recorded by the centralized SSC reference, and the official 20,382-example test collection once afterward. SSC is an independent evaluation with no literature target. CIFAR-10 reuses the active non-IID alpha-0.5 S-VGG9 BNTT configuration, selects two of ten clients, uses uniform aggregation, has no internal validation, selects round 100, and evaluates the official test once afterward.

Rank 0 alone selects clients, restores detached CPU updates to selected-client order, invokes the existing configured FedAvg implementation once, validates when the workload provides validation data, checkpoints, and performs official-test evaluation. Client assignment is `selected_position % process_count`; client randomness depends on the scientific seed streams, round, and client ID, never rank or device. Nonzero ranks construct neither validation nor official-test datasets.

Exclusive-device execution uses one process per physical GPU and NCCL. The separate [device-capacity collection](experiments/device_capacity_evaluation/manifest.yaml) provides unmeasured CUDA MPS configurations with two or four client processes on one GPU, Gloo coordination, detached CPU state movement, and deterministic `device_index = process_rank % device_count` mapping. MPS treatments are not part of the 24-task matrix and none is declared preferable. Optional PyTorch profiling is disabled in ordinary executions because traces add overhead.

Each workload’s one-GPU distributed path is its timing and numerical reference. Logical federated communication remains separate from internal process movement, device telemetry, memory, busy time, and Slurm allocation. When Roihu telemetry is available, the execution record stores aggregate and per-device sample counts and utilization minima, means, and maxima; these remain physical-device observations rather than process busy time. No distributed CUDA, NCCL, or MPS execution evidence has been collected, so there is no speedup, utilization, numerical-equivalence, resource, or energy claim.

Validate the matrix and submit it only when execution evidence is required:

~~~bash
python3 -m fedapfa.cli.scientific_manifest validate \
  --manifest experiments/distributed_evaluation/manifest.yaml
bash scripts/slurm/submit_roihu_distributed_evaluation.sh \
  --work-dir "/scratch/$CSC_PROJECT/$USER/hpc-snn" \
  --datasets shd,ssc,cifar10 \
  --device-counts 1,2,4 \
  --max-parallel 1
~~~

The wrapper submits separate one-, two-, and four-GPU arrays so every task receives its configured physical-device count; it automatically finds no four-GPU CIFAR task. It prints a labelled job ID for every submitted allocation group, and `--max-parallel` controls concurrency within each array group. Monitor every returned job and summarize the complete compatible collection with:

~~~bash
squeue --job <ONE_JOB_ID>,<TWO_JOB_ID>,<FOUR_JOB_ID> --array \
  -o "%.18i %.9P %.28j %.2t %.10M %.10l %R"
mkdir -p "/scratch/$CSC_PROJECT/$USER/hpc-snn/results/distributed_evaluation"
sacct -j <ONE_JOB_ID>,<TWO_JOB_ID>,<FOUR_JOB_ID> --array -X -P \
  --format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocTRES \
  > "/scratch/$CSC_PROJECT/$USER/hpc-snn/results/distributed_evaluation/slurm-accounting.txt"
fedapfa-summarize-distributed-evaluation \
  --manifest experiments/distributed_evaluation/manifest.yaml \
  --runs-root "/scratch/$CSC_PROJECT/$USER/hpc-snn/runs/distributed_evaluation" \
  --output-dir "/scratch/$CSC_PROJECT/$USER/hpc-snn/results/distributed_evaluation" \
  --slurm-accounting "/scratch/$CSC_PROJECT/$USER/hpc-snn/results/distributed_evaluation/slurm-accounting.txt"
~~~

## CIFAR-10 Fed-SNN protocols

The active Fed-SNN evidence covers the two CIFAR-10 SNN 10/2 rows in Table I. Distribution is the only intended treatment difference: IID versus balanced label-Dirichlet non-IID with alpha 0.5. Both treatments use all 50,000 standard training examples, no internal validation collection, ten total clients, two selected clients, five local epochs, 20 timesteps, momentum 0.95, weight decay `1e-4`, uniform aggregation, and final-round selection. The official 10,000-example test collection is evaluated once after round 100.

All six executions for seeds 7, 17, and 27 completed successfully:

| Distribution | Seed 7 | Seed 17 | Seed 27 | Mean ± sample SD | Mean macro-F1 | Paper reference | Mean signed difference |
|---|---:|---:|---:|---:|---:|---:|---:|
| IID | 81.50% | 82.16% | 81.55% | 81.7367% ± 0.3675 pp | 81.7070% | 76.44% | +5.2967 pp |
| Label-Dirichlet non-IID, alpha 0.5 | 72.01% | 75.80% | 73.32% | 73.7100% ± 1.9249 pp | 73.5136% | 73.94% | -0.2300 pp |

The [committed corrected summary](results/fedsnn_paper_evaluation/published_fedsnn_summary.md) is the active Fed-SNN reference. The corrected implementation learns successfully; the non-IID mean closely agrees with the descriptive paper value, while the stable IID mean is about 5.30 percentage points higher. The IID-to-non-IID mean reduction is 8.0267 percentage points. Macro-F1 closely follows accuracy and provides no evidence of class collapse. These are descriptive three-seed results, not evidence of statistical significance, causality, novelty, energy efficiency, implementation equivalence, or an exact reproduction pass. Scientific status remains `equivalence_not_established`.

Both treatments retain signed `[-1,1]` input, signed Poisson spikes, 20-timestep S-VGG9 BNTT with temporal-mean readout, Xavier gain-2 initialization, BNTT epsilon `1e-4`, dropped local remainders, and uniform selected-client aggregation. The earlier 18.23–26.79% evidence remains byte-for-byte preserved as an unsuccessful superseded independent implementation; it is excluded from the active manifest and is never pooled with the corrected evidence.

Validate the two federated configurations, centralized verification, and manifest:

~~~bash
fedapfa-validate-config experiments/published_fedsnn/cifar10/paper_reported_iid_evaluation.yaml
fedapfa-validate-config experiments/published_fedsnn/cifar10/paper_reported_noniid_evaluation.yaml
fedapfa-validate-config experiments/published_fedsnn/cifar10/centralized_learning_verification.yaml
python3 -m fedapfa.cli.scientific_manifest validate \
  --manifest experiments/published_fedsnn/manifest.yaml
~~~

Re-run the six corrected federated tasks on Roihu when an independent execution is required:

~~~bash
bash scripts/slurm/submit_roihu_published_fedsnn.sh \
  --work-dir "/scratch/$CSC_PROJECT/$USER/hpc-snn" \
  --max-parallel 1
~~~

Run the centralized learning verification inside a one-GPU Roihu allocation:

~~~bash
python3 -m fedapfa.cli.train_centralized \
  experiments/published_fedsnn/cifar10/centralized_learning_verification.yaml \
  --data-root "/scratch/$CSC_PROJECT/$USER/hpc-snn/data/cifar10" \
  --output-root "/scratch/$CSC_PROJECT/$USER/hpc-snn/runs/fedsnn_centralized_verification" \
  --device cuda --resume-auto
~~~

Monitor and summarize with:

~~~bash
squeue --job <JOB_ID> --array -o "%.18i %.9P %.28j %.2t %.10M %.10l %R"
fedapfa-summarize-published-fedsnn \
  --manifest experiments/published_fedsnn/manifest.yaml \
  --runs-root "/scratch/$CSC_PROJECT/$USER/hpc-snn/runs/fedsnn_paper_evaluation" \
  --output-dir "/scratch/$CSC_PROJECT/$USER/hpc-snn/results/fedsnn_paper_evaluation"
~~~
