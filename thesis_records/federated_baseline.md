# Federated SHD LIF baseline

## Scientific objective

Establish a trustworthy single-GPU Federated Averaging baseline on SHD before evaluating attention mechanisms or distributed execution. FedAvg is established reference infrastructure and is not presented as thesis novelty.

## Why SHD was used

SHD has a completed centralized LIF reference in this repository, so federated behaviour can be examined without simultaneously changing the dataset, model family, and aggregation algorithm. Its scale also supports paired three-seed evaluation with explicit split and partition checks. SSC is reserved for cross-dataset generalization and resource evaluation; it is not part of this baseline.

DCLS, equation PfA, public-behaviour PfA, adaptive attention, hierarchical aggregation, multiple GPUs, and multiple nodes were excluded. The baseline therefore isolates ordinary LIF training with FedAvg.

## Experiment matrix

The canonical [manifest](../experiments/federated_baselines/manifest.yaml) contains two participation treatments crossed with seeds 7, 17, and 27:

| Experiment | Model | Clients | Partition | Participation |
|---|---|---:|---|---:|
| `shd_lif_dirichlet_alpha_0_5_participation_0_50` | SHD 256/256 LIF | 20 | label-wise Dirichlet, alpha 0.5 | 10 clients (50%) |
| `shd_lif_dirichlet_alpha_0_5_participation_0_25` | SHD 256/256 LIF | 20 | label-wise Dirichlet, alpha 0.5 | 5 clients (25%) |

This gives six scientific executions. Each treatment uses 100 communication rounds, and client selection is without replacement within a round.

## Dataset isolation protocol

The evaluation uses the official `shd_train.h5` and `shd_test.h5` files, 20 classes, 10 ms temporal integration, and a deterministic reduction from 700 cochlear channels to 140 model inputs. For each seed, the shared SHD splitter derives a stratified 10% validation split only from the official training file. Validation indices are removed before client partition construction.

Client datasets contain only the remaining eligible training indices. The official test dataset is not constructed or accessed during communication rounds. Global validation accuracy selects the best checkpoint; after all 100 rounds, the selected checkpoint is evaluated on the official test split exactly once.

## Client partition construction

For each class, eligible indices are deterministically shuffled and allocated by a label-wise Dirichlet draw with alpha 0.5. Every eligible index is assigned exactly once, with no duplication, omission, validation leakage, or test leakage. A construction is rejected if any of the 20 clients has fewer than 32 examples, and deterministic resampling continues within a finite attempt limit.

Each partition artifact records the exact client indices, client sizes, per-client class counts, size statistics, label entropy, construction attempts, dataset identity, split identity, and integrity checks. The two participation treatments use the same split and partition for a given seed.

## Model and optimization

The model has two hidden LIF layers of 256 neurons, tau 10.05, threshold 1, subtractive detached reset, an ATan surrogate with alpha 5, dropout 0.4, no batch normalization, and 20 output classes. It contains 107,028 trainable parameters. Each global model begins from random initialization associated with its scientific seed and is never initialized from a centralized checkpoint.

Every selected client trains for one local epoch with batch size 32, Adam learning rate 0.001, zero weight decay, and gradient clipping at 1. A new optimizer is created for every client selection. Optimizer state is neither retained between selections nor aggregated. SNN state is reset between batches, clients, validation, and official-test evaluation.

## Federated Averaging

For selected clients `k`, the server computes the sample-count-weighted update

\[
w_{t+1} = \frac{\sum_k n_k w_k}{\sum_k n_k},
\]

where `n_k` is the number of training examples owned by selected client `k`. Each client trains an isolated copy of the incoming global state. Aggregation rejects empty or incompatible updates, invalid weights, and non-finite tensors.

## Determinism and paired comparison

Separate deterministic random streams govern validation splitting, partition construction, global initialization, client selection, client training, validation, and final test evaluation. For each seed, both treatments have identical split, partition, and model-initialization identities. A separate selection generator creates one client permutation per round: the 25% treatment selects its first five clients and the 50% treatment selects its first ten. The five-client selection is therefore nested within the ten-client selection, and client training randomness cannot alter selection order.

Checkpoints retain the global model, next round, best validation score and round, random-generator states, Python, NumPy, CPU Torch and CUDA random states, scientific identities, Git identity, communication totals, and completed client and round records. Compatible interrupted executions resume exactly; incompatible records are rejected, and completed records are never overwritten.

## Acceptance requirements

Execution completion requires all 100 rounds, a valid 20-client partition, the configured number of distinct selected clients in every round, finite recorded values, valid aggregation weights, nonempty best and last checkpoints, nonempty client and round logs, complete assignment of eligible training data, verified split and partition isolation, exactly one official-test evaluation after model selection, Git and configuration provenance, scientific identities, and consistent logical communication accounting. Accuracy is not an execution-completion condition.

All six executions passed these completion checks. Scientific status is assessed separately. Both treatments report `not_claimed` because no verified published FedAvg accuracy and tolerance are configured. This status is expected: it is neither an execution failure nor a claim of published reproduction.

## Results

The authoritative machine-readable evidence is [`federated_summary.json`](../results/federated/federated_summary.json); the generated readable form is the [federated summary](../results/federated/federated_summary.md). Accuracy, selected round, execution time, and spike rates below are mean ± sample standard deviation across seeds 7, 17, and 27.

| Participation | Best validation accuracy | Official-test accuracy | Selected round | Execution time | Logical communication | Scientific status |
|---:|---:|---:|---:|---:|---:|---|
| 50% (10/20) | 88.6846% ± 0.935984% | 69.9647% ± 2.99540% | 96.3333 ± 3.21455 | 1594.26 ± 58.24 s | 856,224,000 bytes | `not_claimed` |
| 25% (5/20) | 84.3137% ± 1.76743% | 67.0936% ± 1.21527% | 94.3333 ± 1.15470 | 869.69 ± 19.79 s | 428,112,000 bytes | `not_claimed` |

Mean client spike rates were:

| Participation | Layer 1 spike rate | Layer 2 spike rate |
|---:|---:|---:|
| 50% | 12.4851% ± 1.17874% | 2.58368% ± 0.122608% |
| 25% | 11.5971% ± 1.05105% | 2.35782% ± 0.127507% |

The paired 50%-minus-25% official-test differences were positive for every recorded seed:

| Seed | Official-test difference |
|---:|---:|
| 7 | +2.47350 percentage points |
| 17 | +0.795053 percentage points |
| 27 | +5.34452 percentage points |
| Mean ± sample standard deviation | +2.87102 ± 2.30064 percentage points |

## Centralized comparison

The committed [centralized summary](../results/centralized/centralized_summary.md) reports 76.3693% ± 2.26431% official-test accuracy for SHD LIF under `independent_evaluation`. The centralized mean exceeds the 50% federated mean by 6.4046 percentage points and the 25% federated mean by 9.2756 percentage points. These are contextual gaps, not acceptance thresholds.

Mean best-validation accuracy exceeds mean official-test accuracy by 18.7200 percentage points for 50% participation and 17.2201 percentage points for 25% participation. A similar 20.0768-point separation occurs in the centralized SHD LIF independent evaluation. These observations require investigation; the evidence does not identify their cause.

## Logical communication comparison

Logical communication counts one model-tensor download and one model-tensor upload for every selected client. Bytes are tensor element counts multiplied by element sizes. Optimizer state, dataset transfer, checkpoint I/O, and telemetry files are excluded. These values are deterministic accounting, not measured network traffic or energy consumption.

With a fixed model and 100 rounds, selecting ten rather than five clients produces exactly twice the logical communication: 856,224,000 versus 428,112,000 bytes. No compression, quantization, sparsification, or event-based communication method was evaluated.

## Interpretation

Within this protocol, increasing participation from five to ten clients raised mean best-validation accuracy by 4.37092 percentage points and mean official-test accuracy by 2.87102 percentage points, while doubling logical model communication. Both treatments remained below the centralized SHD LIF reference. This establishes an executable non-IID federated SNN baseline for later comparisons, but it does not establish FedAvg novelty, a thesis contribution, causality, or statistical significance.

## Limitations

- Three seeds support the present baseline record but not strong distributional or significance claims.
- No published FedAvg target has been verified, so successful reproduction is not claimed.
- Only SHD and one ordinary LIF architecture were evaluated; SSC cross-dataset evidence is absent.
- Participation changed together with the number of client updates and logical communication, so the observed accuracy difference does not isolate a causal mechanism.
- Validation-to-test separation requires investigation without using official-test results for model selection.
- Logical communication is not physical network measurement, and no communication compression method was evaluated.
- Job-level GPU telemetry cannot attribute energy to clients or to individual model operations and does not demonstrate low-energy SNN execution.
- The requested scheduler time limit is not an observed runtime measurement.

## Provenance

The six executions ran on 2026-07-16 as Slurm array `189464` on Roihu `gpumedium`, with one NVIDIA GH200 reserved for each array task. All tasks `189464_0` through `189464_5` completed with exit code `0:0`. The executed source state was commit `29ad1558dff52b856ee35b6ce2f538ec2006594a`.

The [scheduler accounting](../results/federated/provenance/slurm-accounting.txt), [execution commit](../results/federated/provenance/execution-commit.txt), and [array identifier](../results/federated/provenance/slurm-array-id.txt) are committed with the aggregation. CPU cores and memory attached to each GPU reservation are scheduler resources, not separate algorithmic resources. Job-level GPU telemetry is not per-client energy, and logical communication is not a physical network measurement.

## Conclusions permitted by the evidence

The repository now has a completed, deterministic SHD FedAvg baseline with validated three-seed aggregation at two paired participation levels. It can serve as the ordinary-LIF reference for matched heterogeneity and attention comparisons. The evidence does not establish a published reproduction, FedAvg novelty, statistical significance, cross-dataset generalization, communication efficiency under compression, or neural-model energy efficiency.

## Evidence enabled next

Subsequent evaluations can use these split, partition, initialization, acceptance, and communication-accounting rules as matched controls when studying client heterogeneity, attention mechanisms, convergence, SSC generalization, and resource behaviour. Any such comparison must preserve test isolation and must distinguish logical communication, scheduler resources, and measured device telemetry.
