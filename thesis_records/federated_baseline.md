# Federated SHD LIF reference

## Scientific objective

Establish a trustworthy single-GPU Federated Averaging reference on SHD before evaluating attention mechanisms or distributed execution. FedAvg itself is not presented as thesis novelty. SHD is used because the repository already contains a completed centralized LIF reference, the dataset supports careful three-seed comparisons, and federated correctness can be examined without simultaneously changing the dataset, model, and aggregation algorithm. SSC remains necessary for broader generalization and resource evidence but is outside this reference matrix.

## Methods

The canonical manifest contains two treatments crossed with seeds 7, 17, and 27:

| Experiment | Model | Clients | Partition | Participation |
|---|---|---:|---|---:|
| `shd_lif_dirichlet_alpha_0_5_participation_0_50` | SHD 256/256 LIF | 20 | label-wise Dirichlet, alpha 0.5 | 10 clients (50%) |
| `shd_lif_dirichlet_alpha_0_5_participation_0_25` | SHD 256/256 LIF | 20 | label-wise Dirichlet, alpha 0.5 | 5 clients (25%) |

This produces six independent scientific executions. The LIF settings match the centralized SHD reference: tau 10.05, threshold 1, subtractive detached reset, ATan surrogate alpha 5, dropout 0.4, no batch normalization, and 20 output classes. The global model is initialized randomly from a dedicated seed stream and never from a centralized checkpoint.

For each seed, the shared stratified splitter derives a 10% validation split from `shd_train.h5`. Those validation indices are removed before partitioning. A deterministic label-wise Dirichlet construction assigns every remaining training index exactly once to 20 clients, requires at least 32 examples per client, and resamples the entire construction until valid. The partition artifact records exact indices, class counts, size statistics, label entropy, dataset identity, split identity, construction attempts, and integrity checks.

The server performs 100 communication rounds. Each selected client trains for one local epoch with batch size 32, Adam learning rate 0.001, zero weight decay, and gradient clipping at 1. The optimizer is recreated at every selection; optimizer state is not retained or aggregated. Client sampling is without replacement within a round.

The server update is sample-count-weighted:

`w_next = sum(n_k * w_k) / sum(n_k)`

Aggregation rejects empty or incompatible updates, invalid weights, and non-finite tensors. Clients train detached model copies, so local execution cannot mutate server parameters before aggregation. SNN state is reset between batches, clients, validation, and official-test evaluation.

## Paired comparison protocol

Distinct deterministic streams govern splitting, partitioning, global initialization, client selection, client training, validation, and final test evaluation. The two participation treatments reuse the same split, client partition, and initial global parameters for a given seed. A separate client-selection generator creates one permutation per round; the five-client treatment selects its first five entries and the ten-client treatment selects its first ten. Local training randomness cannot alter this generator.

## Model selection and test isolation

Validation accuracy selects `checkpoints/best.pt`; `checkpoints/last.pt` is written after every completed round. The official SHD test dataset is neither constructed nor accessed during communication rounds. After all rounds, the selected global checkpoint is evaluated on the official test split exactly once. A durable official-test record prevents reevaluation after interruption.

## Logical communication accounting

The model payload is the sum of communicated tensor element counts multiplied by their element sizes. Each selected client contributes one model download and one model upload. Optimizer state, dataset transfer, checkpoint I/O, and telemetry files are excluded. The resulting byte totals are logical communication volume, not measured network traffic.

## Deterministic resumption

Every checkpoint records the global model, next round, best validation score and round, selection-generator state, global Python, NumPy, CPU Torch, and CUDA random states, split and partition identities, model-initialization identity, configuration identity, Git commit, communication totals, and completed client and round records. Automatic resumption skips completed compatible executions, continues compatible interrupted executions, and rejects identity mismatches without overwriting evidence.

## Evidence collected

The centralized prerequisite is the [centralized summary](../results/centralized/centralized_summary.md). Its SHD LIF `independent_evaluation` accuracy is contextual evidence, not a reproduction target or acceptance threshold.

Local deterministic software verification covers partition integrity, paired client selection, isolated LIF client training, weighted aggregation, communication accounting, checkpoint creation, resumption, official-test isolation, completion assessment, and result aggregation. No federated scientific GPU result has been generated or claimed.

## Prospective Roihu evidence

Each array task reserves one GH200 and executes one configuration/seed pair. Job-level `nvidia-smi` telemetry samples supported utilization, memory, power, and temperature fields every two seconds. This telemetry describes the allocated GPU job; it is not per-client energy, neural-model energy, or neuromorphic energy. Unsupported fields and the sampling command are recorded explicitly.

## Unresolved scientific limitations

- Six scientific GPU executions have not yet been run.
- Three seeds limit inferential strength.
- The effect of client heterogeneity on convergence and per-client fairness remains unmeasured.
- Logical communication volume is not observed network traffic.
- Job-level GPU telemetry cannot attribute energy to individual clients.
- SSC evidence is absent from this reference.

## Conclusions permitted by the evidence

The repository now defines a deterministic and testable FedAvg reference protocol tied to the centralized SHD LIF evidence. No conclusion about federated accuracy, convergence, participation effects, communication efficiency, or resource behaviour is permitted until all six executions pass acceptance and aggregation checks.

## Commands after review

Submit:

~~~bash
bash scripts/slurm/submit_roihu_federated.sh \
  --work-dir "/scratch/$CSC_PROJECT/$USER/hpc-snn" \
  --max-parallel 1
~~~

Monitor:

~~~bash
squeue --job <JOB_ID> --array -o "%.18i %.9P %.28j %.2t %.10M %.10l %R"
~~~

Aggregate:

~~~bash
fedapfa-summarize-federated \
  --manifest experiments/federated_baselines/manifest.yaml \
  --runs-root "/scratch/$CSC_PROJECT/$USER/hpc-snn/runs/federated" \
  --output-dir "/scratch/$CSC_PROJECT/$USER/hpc-snn/results/federated"
~~~
