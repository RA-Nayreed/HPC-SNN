# Centralized SNN evaluation

## Scientific objective

Establish comparable centralized SHD and SSC references for two-layer LIF networks, equation PfA, pinned public-code-behaviour PfA, and the published SHD DCLS procedure while preserving the distinction between unbiased held-out evaluation and publication-protocol reproduction.

## Methods

The canonical collection contains six experiments and three deterministic seeds, 7, 17, and 27, for exactly 18 independent GH200 executions:

| Experiment | Dataset and model | PfA | Protocol |
|---|---|---|---|
| `shd_lif_independent_evaluation` | SHD 256/256 LIF | none | `independent_evaluation` |
| `shd_dcls_published_protocol` | SHD 256/256 DCLS | none | `published_protocol` |
| `shd_pfa_equation_independent_evaluation` | SHD 256/256 LIF | equation | `independent_evaluation` |
| `shd_pfa_public_published_protocol` | SHD 256/256 LIF | pinned public behaviour | `published_protocol` |
| `ssc_lif_128_independent_evaluation` | SSC 128/128 LIF | none | `independent_evaluation` |
| `ssc_pfa_equation_128_independent_evaluation` | SSC 128/128 LIF | equation | `independent_evaluation` |

For SHD, `independent_evaluation` creates a seed-specific stratified validation split from the official training file. Checkpoint selection uses only that split, and the official test file is accessed only after model selection. For SSC, the protocol uses the official validation file for selection and accesses the official test file afterward.

For SHD, `published_protocol` intentionally monitors the official test file during training to reproduce the source evaluation procedure. Its reported accuracies are reproduction-protocol measurements, not unbiased held-out estimates. Equation PfA and public-code behaviour remain separate because their denominator calculations differ. DCLS and PfA are not combined because no verified repository source specifies that architecture. The SSC 512-neuron model is outside the current evaluation scope.

Every configuration uses all examples required by its protocol, null batch limits, CUDA, eight data-loader workers, deterministic seeds, and explicit stopping settings. Execution completion requires a valid terminal condition, finite metrics, both checkpoints, complete-dataset use, official-test evaluation, the configured model class, nonempty metrics and logs, and resolved-configuration and Git provenance. Scientific reproduction is assessed separately and requires a verified reference accuracy and tolerance.

## Evidence

The evaluation was executed on 2026-07-16 as Slurm array `186611` from Git commit `44c7e4aa52a8dd6103256d17da3b79d2aa075056`. All 18 of 18 array tasks completed with exit code `0:0`, providing three completed seeds for every experiment. The machine-readable aggregation is valid; its readable form is the [centralized summary](../results/centralized/centralized_summary.md). Scheduler outcomes and the executed source identity are retained in [Slurm accounting](../results/centralized/provenance/slurm-accounting.txt) and the [execution commit record](../results/centralized/provenance/execution-commit.txt).

Accuracy, peak CUDA memory, and runtime values below are mean ± sample standard deviation across seeds 7, 17, and 27. Memory is the framework-recorded peak allocation and is expressed in MiB.

| Experiment | Protocol | Best validation accuracy | Official-test accuracy | Parameters | Peak CUDA memory | Runtime | Scientific status |
|---|---|---:|---:|---:|---:|---:|---|
| SHD LIF | `independent_evaluation` | 96.4461% ± 0.648468% | 76.3693% ± 2.26431% | 107,028 | 411.27 ± 0.51 MiB | 375.53 ± 117.24 s | `not_claimed` |
| SHD DCLS | `published_protocol` | 91.3722% ± 0.243267% | 91.3722% ± 0.243267% | 310,272 | 509.15 ± 0.26 MiB | 819.78 ± 21.64 s | `not_claimed` |
| SHD equation PfA | `independent_evaluation` | 96.6503% ± 0.566030% | 78.4305% ± 1.91990% | 107,028 | 890.77 ± 0.51 MiB | 353.12 ± 53.72 s | `not_claimed` |
| SHD public-behaviour PfA | `published_protocol` | 84.0253% ± 0.0919464% | 84.0253% ± 0.0919464% | 107,028 | 890.72 ± 0.41 MiB | 805.88 ± 27.06 s | `not_claimed` |
| SSC LIF-128 | `independent_evaluation` | 57.0484% ± 0.234326% | 54.8981% ± 0.198346% | 39,075 | 198.60 ± 0.00 MiB | 2672.82 ± 823.58 s | `not_claimed` |
| SSC equation PfA-128 | `independent_evaluation` | 57.9234% ± 0.145994% | 55.7812% ± 0.570910% | 39,075 | 373.60 ± 0.00 MiB | 2947.06 ± 378.01 s | `not_claimed` |

The per-seed official-test accuracies are:

| Experiment | Seed 7 | Seed 17 | Seed 27 |
|---|---:|---:|---:|
| SHD LIF | 76.6343% | 78.4894% | 73.9841% |
| SHD DCLS | 91.3869% | 91.6078% | 91.1219% |
| SHD equation PfA | 76.9876% | 77.6943% | 80.6095% |
| SHD public-behaviour PfA | 83.9223% | 84.0989% | 84.0548% |
| SSC LIF-128 | 55.1271% | 54.7787% | 54.7885% |
| SSC equation PfA-128 | 55.1222% | 56.1230% | 56.0985% |

Aggregated layer spike rates are also mean ± sample standard deviation:

| Experiment | Layer 1 spike rate | Layer 2 spike rate |
|---|---:|---:|
| SHD LIF | 14.4048% ± 0.3331% | 4.7986% ± 0.2198% |
| SHD DCLS | 11.4168% ± 0.3793% | 9.5509% ± 0.5927% |
| SHD equation PfA | 17.6300% ± 0.6595% | 4.4907% ± 0.1559% |
| SHD public-behaviour PfA | 17.3495% ± 0.5370% | 4.6765% ± 0.0403% |
| SSC LIF-128 | 13.3469% ± 2.4607% | 1.8668% ± 0.0719% |
| SSC equation PfA-128 | 13.0601% ± 0.5008% | 1.8237% ± 0.0293% |

The array consumed 6.7775 allocated GPU-hours, corresponding to approximately 1,355.5 GPU billing units at the applicable accounting rate. These quantities describe reserved execution resources, not neural-network energy consumption.

The DCLS CUDA executions completed successfully for all three seeds. This supersedes the earlier uncertainty that was based only on a CPU DCLS probe.

## Interpretation

Within the matched `independent_evaluation` protocol, equation PfA exceeded the LIF mean official-test accuracy by 2.0612 percentage points on SHD and 0.8831 percentage points on SSC. These are descriptive differences from three seeds, not claims of statistical significance.

The SHD independent evaluations show large differences between mean checkpoint-selection accuracy and mean official-test accuracy: 20.0768 percentage points for LIF and 18.2198 percentage points for equation PfA. This is an observation requiring investigation; the present evidence does not identify its cause.

Published-protocol and independent-evaluation results cannot be pooled. The former use the official SHD test split during training, whereas the latter reserve it until after checkpoint selection. Consequently, the DCLS and public-behaviour PfA accuracies cannot be interpreted as unbiased held-out estimates or ranked directly against independent-evaluation models.

Equation PfA and public-behaviour PfA also cannot yet be compared directly. Their available SHD measurements use different protocols, and no matched-protocol experiment was executed. The equation-versus-public discrepancy therefore remains unresolved.

## Limitations

- Three seeds do not support strong significance claims.
- No verified literature accuracy or tolerance is available for any experiment. Every scientific status is therefore `not_claimed`, which is neither an execution failure nor a successful reproduction claim.
- Peak CUDA memory is not GPU utilization. Slurm did not retain historical GPU utilization, GPU-memory telemetry, or GPU energy measurements.
- Execution concurrency changed from four simultaneous tasks to one and then six. Accuracy evidence remains usable, but runtime is not a controlled scaling comparison.
- The allocated GPU-hours and billing units are operational accounting, not energy measurements.
- The SHD validation-to-test differences require targeted investigation.
- The equation-versus-public PfA discrepancy lacks a matched-protocol comparison.

## Conclusions permitted by the evidence

The six centralized experiments reached execution completion for all required seeds, including successful CUDA execution of DCLS. The independent-evaluation results provide centralized comparison measurements for subsequent federated experiments. Equation PfA has descriptively higher mean official-test accuracy than LIF under matched protocols on both datasets, but the evidence does not establish statistical significance, published reproduction, or a general superiority claim.

Centralized results alone do not establish the success of the thesis method. Published-protocol measurements must remain separate from independent-evaluation measurements, and all reproduction statuses remain `not_claimed` until verifiable literature targets and tolerances exist.

## Questions carried into federated evaluation

- How should client partitions preserve label and event-distribution validity on SHD and SSC?
- Does the descriptive equation-PfA difference persist under identical federated partitions and optimization settings?
- How do convergence and client sampling vary across seeds?
- What communication, runtime, memory, utilization, and energy evidence must be collected prospectively?
- Which targeted checks explain the SHD validation-to-test difference without accessing held-out test data during selection?

## Reusable commands

Submit:

~~~bash
bash scripts/slurm/submit_roihu_centralized.sh \
  --work-dir "/scratch/$CSC_PROJECT/$USER/hpc-snn" \
  --max-parallel 4
~~~

Monitor:

~~~bash
squeue --job <JOB_ID> --array -o "%.18i %.9P %.28j %.2t %.10M %.10l %R"
~~~

Aggregate:

~~~bash
fedapfa-summarize-centralized \
  --manifest experiments/centralized/manifest.yaml \
  --runs-root "/scratch/$CSC_PROJECT/$USER/hpc-snn/runs/centralized" \
  --output-dir "/scratch/$CSC_PROJECT/$USER/hpc-snn/results/centralized"
~~~
