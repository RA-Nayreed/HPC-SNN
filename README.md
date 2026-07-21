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

The completed evaluation is recorded in the committed [physical-GPU result collection](results/distributed_evaluation/distributed_evaluation_summary.md), the separate [one-GPU capacity collection](results/device_capacity_evaluation/distributed_evaluation_summary.md), and the [distributed scientific record](thesis_records/distributed_execution.md). Both collections are `valid`: all 33 tasks passed their declared consistency gates. Results are means ± sample standard deviations over paired seeds 7, 17, and 27; datasets are not pooled.

| Workload | Physical GPUs | Test accuracy | Mean total runtime (s) | Paired speedup | Parallel efficiency | Mean GPU utilization | Numerical equivalence |
|---|---:|---:|---:|---:|---:|---:|---|
| SHD | 1 | 0.699647 ± 0.029954 | 1561.09 ± 42.5351 | 1.00000 ± 0 | 1.00000 ± 0 | 16.9111% ± 0.564368% | exact |
| SHD | 2 | 0.699647 ± 0.029954 | 1011.02 ± 8.85204 | 1.54400 ± 0.0342813 | 0.771999 ± 0.0171407 | 25.8264% ± 0.825906% | exact |
| SHD | 4 | 0.699647 ± 0.029954 | 620.950 ± 8.46605 | 2.51471 ± 0.0930752 | 0.628677 ± 0.0232688 | 42.7625% ± 1.25962% | exact |
| SSC | 1 | 0.426635 ± 0.0128545 | 2812.16 ± 31.2943 | 1.00000 ± 0 | 1.00000 ± 0 | 13.0648% ± 0.256050% | exact |
| SSC | 2 | 0.426635 ± 0.0128545 | 1777.76 ± 44.2429 | 1.58222 ± 0.0227238 | 0.791112 ± 0.0113619 | 25.6943% ± 0.341631% | exact |
| SSC | 4 | 0.426635 ± 0.0128545 | 1192.56 ± 22.1012 | 2.35830 ± 0.0176213 | 0.589575 ± 0.00440534 | 45.1094% ± 0.473272% | exact |
| CIFAR-10 | 1 | 0.737100 ± 0.0192486 | 23074.3 ± 739.096 | 1.00000 ± 0 | 1.00000 ± 0 | 32.2526% ± 0.672774% | exact |
| CIFAR-10 | 2 | 0.737100 ± 0.0192486 | 15427.8 ± 641.652 | 1.49625 ± 0.0356666 | 0.748123 ± 0.0178333 | 37.2154% ± 2.88017% | exact |

The capacity treatment used SHD and one physical GH200. CUDA MPS increased throughput, but the packed treatments were numerically different from their paired exclusive-process references and remain separate from authoritative accuracy comparisons.

| Client processes on one GPU | Test accuracy | Mean total runtime (s) | Paired speedup | Parallel efficiency | Mean GPU utilization | Numerical status |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.699647 ± 0.0299540 | 1515.45 ± 85.4437 | 1.00000 ± 0 | 1.00000 ± 0 | 17.2476% ± 1.06860% | exact reference |
| 2 | 0.699647 ± 0.0214393 | 864.519 ± 14.8104 | 1.75221 ± 0.0714499 | 0.876106 ± 0.0357250 | 32.5880% ± 1.56503% | difference observed |
| 4 | 0.701561 ± 0.0271646 | 561.249 ± 5.20705 | 2.69982 ± 0.141900 | 0.674955 ± 0.0354750 | 44.7365% ± 0.912779% | difference observed |

These are descriptive three-seed measurements, not claims of statistical significance, causal superiority, energy efficiency, multinode scalability, published reproduction, or thesis novelty.

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

## Client resource measurement and cost estimation

The committed collection is valid: six accepted executions produced exactly 6,000 accepted client records, with SHD and SSC each evaluated at seeds 7, 17, and 27. Each execution used 100 communication rounds, ten selected clients per round, one local epoch, sample-count FedAvg, one process on one physical GH200, no CUDA MPS, and persistent NVML sampling at 100 ms. The training-only calibration passed with a committed median relative overhead of `0.01713823842158517` (about 1.7138%), below the declared 2% threshold; it had ten paired repetitions, complete sample coverage, identical numerical updates, no sampling errors, one consistent GPU UUID, and zero official-test accesses.

Slurm job `291481` completed with exit code `0:0` at execution commit `3ddae173c89125bc69922d80bde5732ed6cd050e`. Its 34,051-second one-GH200 allocation represents `9.45861111111111` allocated GPU-hours. Accepted client-training intervals contained `1,883,612.6948749206 J` of gross device energy (`1,883.6126948749206 kJ`, `0.5232257485763668 kWh`) and `52,380.5986610567 J` of idle-adjusted device energy (`52.380598661056695 kJ`, `0.014550166294737972 kWh`). These are not whole-allocation energy measurements.

Seeds 7 and 17 supplied fitting and client-grouped selection; untouched seed 27 supplied evaluation. The exported scheduling model is the `event_structure` ridge model for client wall time. It excludes client identity and current-execution spike measurements, and its stored JSON reload reproduces predictions. Headline joint seed-27 metrics are:

| Target | MAE | Median AE | P90 AE | R² | Spearman ρ | n |
|---|---:|---:|---:|---:|---:|---:|
| Client wall time (s) | 0.06884469267330857 | 0.066351972627951 | 0.11281469013244072 | 0.9811472859662216 | 0.994406546671811 | 2000 |
| CUDA-event time (s) | 0.07553541764813324 | 0.07235978633440726 | 0.11167373378190251 | 0.9357145406671056 | 0.9933739515652682 | 2000 |
| Gross device energy (J) | 11.216858334971583 | 10.85648933302295 | 18.579205432184253 | 0.9826846943594206 | 0.994749131529932 | 2000 |
| Idle-adjusted device energy (J) | 0.6729312828219791 | 0.3604094094492716 | 1.357045479752213 | 0.9630307619096935 | 0.9607420713455627 | 2000 |

The predeclared decision is `spike_history_not_adopted`. Historical spike features improved SHD median absolute runtime error by the committed fraction `0.3331679722893203`, with the tail-error and rank conditions passing. For SSC the corresponding fraction was `-2.233354617899182`, and both the median-improvement and tail-error conditions failed while rank was maintained. Prediction time was negligible and historical-spike assignment was closer to the measured oracle, but the required two-dataset rule failed. This is dataset-dependent negative evidence, not an execution failure.

Authoritative evidence is in the generated [resource summary](results/resource_measurement/resource_measurement_summary.md), [summary JSON](results/resource_measurement/resource_measurement_summary.json), [model evaluation](results/resource_measurement/cost_model_evaluation.md), [evaluation JSON](results/resource_measurement/cost_model_evaluation.json), [scheduling model](results/resource_measurement/client_cost_model.json), and [energy model](results/resource_measurement/energy_cost_model.json). Figures include [predicted versus measured runtime](results/resource_measurement/predicted_versus_measured_runtime.png), [predicted versus measured gross energy](results/resource_measurement/predicted_versus_measured_gross_energy.png), [feature ablation](results/resource_measurement/feature_ablation_comparison.png), [cross-dataset transfer](results/resource_measurement/cross_dataset_transfer_comparison.png), and [offline assignment loads](results/resource_measurement/observed_load_versus_predicted_assignment_load.png).

The [protocol](docs/experimental_protocol.md#client-resource-measurement-protocol) preserves the predeclared methodology, the [reproducibility guide](docs/reproducibility.md#client-resource-measurement-and-cost-estimation) records the committed provenance and regeneration commands, the [Roihu guide](environment/roihu/README.md#client-resource-measurement-allocation) documents the allocation, and the [scientific record](thesis_records/resource_measurement.md) provides comprehensive metrics, interpretation, offline assignment evidence, and limitations. Offline assignment did not change a production scheduling policy and does not establish an end-to-end distributed speedup.

## Prospective scheduling and node-hierarchical reduction

The repository includes [scheduling](experiments/scheduling_evaluation/manifest.yaml) and [node-hierarchical reduction](experiments/hierarchical_reduction_evaluation/manifest.yaml) matrices. They use seeds 37, 47, and 57 and validate to 18 and 12 executions. Scheduling verifies a frozen model fitted on 4,000 seed-7/17 rows, with 2,000 seed-27 rows held untouched for post-freeze evaluation. Hierarchy holds that scheduler constant and compares flat ordered aggregation with node-local grouping under the established FedAvg policy: all floating and complex inputs accumulate in float64 from globally normalized weighted terms, exactly as in the pre-Week-6 flat implementation. Fully resolved paired configurations are whitelist-checked. Both prospective collections use four GH200 GPUs on `gpumedium`, validate four distinct allocation UUIDs before workload construction, and reconcile sequential treatments to one allocation-level elapsed time and GPU-hour record. No scientific result or performance claim exists until the Roihu executions and acceptance gates complete. See the [protocol](docs/experimental_protocol.md#scheduling-and-hierarchical-reduction-evaluation-protocol) and [reproducibility guide](docs/reproducibility.md#scheduling-and-hierarchical-reduction-execution).
