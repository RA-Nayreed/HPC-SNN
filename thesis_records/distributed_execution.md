# Completed single-node distributed FedAvg evaluation

## Scientific status

This record documents descriptive evidence from two committed collections. The physical-device collection is `valid` with 24 expected and 24 completed tasks. The one-GPU capacity collection is `valid` with nine expected and nine completed tasks. Both use execution commit [`4491078710d01d22e9517e62c8de0f3554c4b3f2`](../results/distributed_evaluation/provenance/execution-commit.txt), seeds 7, 17, and 27, zero resumptions, exactly one official-test access per run, and no validation findings.

Physical-device arrays `250928`, `250929`, and `250946` and capacity array `250950` are entirely `COMPLETED` with exit code `0:0`. Here, `valid` means that the collection passed its declared consistency gates; it does not make every treatment numerically equivalent. Similarly, `not_claimed` and `equivalence_not_established` are scientific-status outcomes rather than execution failures.

## Research questions

1. Does exclusive one-process-per-GPU execution preserve the scientific structure, selected checkpoint, parameters, and official-test metrics of each paired one-GPU reference?
2. How much do two or four physical GPUs reduce communication-round runtime for SHD and SSC, and how much do two GPUs reduce runtime for CIFAR-10?
3. How do utilization, process load imbalance, per-process CUDA allocation, and scheduler-accounted GPU-hours change with parallel width?
4. On one GH200, do two or four CUDA MPS client processes increase throughput, and are those treatments numerically transparent relative to one exclusive process?

## Execution architecture

All runs used one Roihu node and synchronous FedAvg. Exclusive execution mapped one distributed process to each physical GH200 and used NCCL. The capacity treatments mapped two or four client processes to one physical GH200, used Gloo for CPU control and detached state movement, and used CUDA MPS for client computation.

Rank zero selected clients, assigned selected position `i` to process `i % process_count`, and owned the authoritative global model. Other ranks trained their assigned clients and returned detached CPU updates. In plain language, rank zero selected clients, restored updates to the established order, aggregated once, selected checkpoints, and evaluated the official test collection. It alone performed workload validation, wrote shared metrics and checkpoints, and constructed the official-test dataset. Nonzero ranks constructed neither validation nor official-test datasets.

SHD and SSC used example-count aggregation. CIFAR-10 used equal weights for its two selected clients. Rank zero checked client identity, selected position, shape, dtype, finiteness, and model identity before restoring the selected-client order and invoking aggregation once.

Client seeds were independent of device count, process rank, topology, and completion order. They derived from scientific seed streams, communication round, and client identity, not device index, arrival order, or process placement. The evidence confirms identical client-selection order and client-training seeds across paired topology treatments.

## Dataset protocols

- **SHD:** the established 256/256 LIF FedAvg protocol used 20 clients, ten selected per round, 100 rounds, one local epoch, local batch 32, Adam at `0.001`, gradient clipping 1, and example-count aggregation. A seed-specific stratified validation split was removed before client partitioning and selected the best-validation checkpoint. The official test collection was accessed once afterward.
- **SSC:** all 75,466 official training examples were partitioned across 20 clients with label-Dirichlet alpha 0.5; ten clients were selected per round. The 128/128 LIF model used 700-to-140 channel reduction, 10 ms bins, one local epoch, local batch 256, and example-count aggregation. All 9,981 official validation examples selected the checkpoint, then all 20,382 official test examples were evaluated once.
- **CIFAR-10:** the non-IID alpha-0.5 S-VGG9 BNTT protocol used ten clients, two selected per round, five local epochs, 20 timesteps, and uniform aggregation. It had no internal validation collection, selected round 100, and evaluated all 10,000 official test examples once afterward.

SHD, SSC, and CIFAR-10 were summarized separately. No cross-dataset pooling was performed.

## Experimental matrix and pairing

| Collection | Workload | Physical GPUs | Client processes per GPU | Seeds | Tasks |
|---|---|---|---:|---|---:|
| Physical devices | SHD | 1, 2, 4 | 1 | 7, 17, 27 | 9 |
| Physical devices | SSC | 1, 2, 4 | 1 | 7, 17, 27 | 9 |
| Physical devices | CIFAR-10 | 1, 2 | 1 | 7, 17, 27 | 6 |
| One-GPU capacity | SHD | 1 | 1, 2, 4 | 7, 17, 27 | 9 |

Each treatment was paired by workload and seed to the one-GPU, one-process run in its own collection. Total runtime is the sum of communication-round total durations. Paired speedup is reference runtime divided by treatment runtime for the same seed; parallel efficiency is speedup divided by process count. Values below are means ± sample standard deviations across the three paired seeds.

## Physical multi-GPU results

| Workload | GPUs | Test accuracy | Total runtime (s) | Paired speedup | Efficiency | GPU utilization | Load imbalance | Structural | Numerical |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| SHD | 1 | 0.699647 ± 0.029954 | 1561.09 ± 42.5351 | 1.00000 ± 0 | 1.00000 ± 0 | 16.9111% ± 0.564368% | 0 ± 0 | exact | exact |
| SHD | 2 | 0.699647 ± 0.029954 | 1011.02 ± 8.85204 | 1.54400 ± 0.0342813 | 0.771999 ± 0.0171407 | 25.8264% ± 0.825906% | 0.330663 ± 0.0733092 | exact | exact |
| SHD | 4 | 0.699647 ± 0.029954 | 620.950 ± 8.46605 | 2.51471 ± 0.0930752 | 0.628677 ± 0.0232688 | 42.7625% ± 1.25962% | 0.630360 ± 0.0385039 | exact | exact |
| SSC | 1 | 0.426635 ± 0.0128545 | 2812.16 ± 31.2943 | 1.00000 ± 0 | 1.00000 ± 0 | 13.0648% ± 0.256050% | 0 ± 0 | exact | exact |
| SSC | 2 | 0.426635 ± 0.0128545 | 1777.76 ± 44.2429 | 1.58222 ± 0.0227238 | 0.791112 ± 0.0113619 | 25.6943% ± 0.341631% | 0.230179 ± 0.0514786 | exact | exact |
| SSC | 4 | 0.426635 ± 0.0128545 | 1192.56 ± 22.1012 | 2.35830 ± 0.0176213 | 0.589575 ± 0.00440534 | 45.1094% ± 0.473272% | 0.532826 ± 0.0215700 | exact | exact |
| CIFAR-10 | 1 | 0.737100 ± 0.0192486 | 23074.3 ± 739.096 | 1.00000 ± 0 | 1.00000 ± 0 | 32.2526% ± 0.672774% | 0 ± 0 | exact | exact |
| CIFAR-10 | 2 | 0.737100 ± 0.0192486 | 15427.8 ± 641.652 | 1.49625 ± 0.0356666 | 0.748123 ± 0.0178333 | 37.2154% ± 2.88017% | 0.529373 ± 0.0798654 | exact | exact |

Every physical-device comparison had exact structural identity, equal selected checkpoints, zero maximum absolute and relative parameter difference, and equal official-test accuracy and macro-F1. Physical multi-GPU execution reduced latency but did not scale linearly. Efficiency fell as process count increased because validation, aggregation, checkpoint work, and load imbalance remained.

The load-imbalance metric is the largest per-round value of `(maximum process busy time - minimum process busy time) / maximum process busy time`. Ten selected clients cannot be divided evenly over four processes: selected-order round-robin assignment yields three, three, two, and two clients. Client partitions also differ in example count, and client training durations differ even at equal client count. These effects explain why utilization can rise while efficiency declines. The explanation is descriptive; the runs did not isolate each source of imbalance.

### Per-process CUDA allocation

Each process recorded its peak allocated tensor memory. The table reports the treatment mean of the maximum per-process peak in each run, in bytes and binary GiB (`bytes / 2^30`).

| Collection | Workload | GPUs | Processes per GPU | Peak allocation (bytes) | Peak allocation (GiB) |
|---|---|---:|---:|---:|---:|
| Physical devices | SHD | 1 | 1 | 116,934,997.333 ± 48,923.022 | 0.108904203 ± 0.000045563 |
| Physical devices | SHD | 2 | 1 | 116,934,997.333 ± 48,923.022 | 0.108904203 ± 0.000045563 |
| Physical devices | SHD | 4 | 1 | 116,860,928 ± 99,774.337 | 0.108835220 ± 0.000092922 |
| Physical devices | SSC | 1 | 1 | 208,718,336 ± 0 | 0.194384098 ± 0 |
| Physical devices | SSC | 2 | 1 | 208,718,336 ± 0 | 0.194384098 ± 0 |
| Physical devices | SSC | 4 | 1 | 208,718,336 ± 0 | 0.194384098 ± 0 |
| Physical devices | CIFAR-10 | 1 | 1 | 2,205,461,504 ± 0 | 2.053996086 ± 0 |
| Physical devices | CIFAR-10 | 2 | 1 | 2,205,461,504 ± 0 | 2.053996086 ± 0 |
| One-GPU capacity | SHD | 1 | 1 | 116,934,997.333 ± 48,923.022 | 0.108904203 ± 0.000045563 |
| One-GPU capacity | SHD | 1 | 2 | 116,934,485.333 ± 48,923.022 | 0.108903726 ± 0.000045563 |
| One-GPU capacity | SHD | 1 | 4 | 116,859,904 ± 99,774.337 | 0.108834267 ± 0.000092922 |

Allocated tensor memory is not reserved CUDA memory. The JSON records reserved peaks separately. Neither allocated nor reserved CUDA memory is the GPU's HBM capacity, Grace CPU memory, or the memory quantity reserved by Slurm. These quantities must not be substituted for one another.

## One-GPU capacity findings

| Client processes | Test accuracy | Total runtime (s) | GPU utilization | Paired speedup | Efficiency | Load imbalance | Structural | Numerical |
|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 0.699647 ± 0.0299540 | 1515.45 ± 85.4437 | 17.2476% ± 1.06860% | 1.00000 ± 0 | 1.00000 ± 0 | 0 ± 0 | exact | exact reference |
| 2 | 0.699647 ± 0.0214393 | 864.519 ± 14.8104 | 32.5880% ± 1.56503% | 1.75221 ± 0.0714499 | 0.876106 ± 0.0357250 | 0.297964 ± 0.0475917 | exact | difference observed |
| 4 | 0.701561 ± 0.0271646 | 561.249 ± 5.20705 | 44.7365% ± 0.912779% | 2.69982 ± 0.141900 | 0.674955 ± 0.0354750 | 0.639708 ± 0.0479606 | exact | difference observed |

CUDA MPS substantially improved throughput on one GH200: mean runtime fell from 1515.45 seconds with one process to 864.519 seconds with two processes and 561.249 seconds with four. This establishes a capacity improvement for the measured setup, not numerical transparency.

### Numerical MPS limitation

Paired official-test accuracy differences are treatment minus the one-process reference, in percentage points:

| Client processes | Seed 7 | Seed 17 | Seed 27 |
|---:|---:|---:|---:|
| 2 | −2.12014 | +2.16431 | −0.04417 |
| 4 | +0.574205 | +0.397527 | −0.397527 |

Maximum absolute selected-checkpoint parameter differences across the six packed comparisons ranged from approximately 0.389 to 0.516. Maximum relative differences ranged from about 9,458 to 88,350; several comparison parameters were near zero, so those relative values are extremely large and are not independently meaningful. Seed 27 selected a different checkpoint under both packed treatments: round 94 for one process, round 99 for two processes, and round 77 for four processes. Predictions were not stored, so prediction agreement is unavailable. Structural identity remained exact in every packed comparison.

Changed CUDA kernel selection, scheduling, or floating-point accumulation under concurrent MPS execution is a plausible explanation for the differences. The evaluation did not isolate these mechanisms, so that explanation remains an inference.

### Decision for later experiments

- Exclusive one-process-per-GPU execution is the authoritative path for later scientific accuracy comparisons.
- MPS is retained as a separately identified capacity treatment.
- MPS results must not be silently pooled with exclusive execution.
- No preferred packing level is declared solely from these three seeds.

## Resource interpretation

The following means ± sample standard deviations come from committed Slurm accounting, not an estimate based only on round runtime:

| Collection | Workload | GPUs | Processes per GPU | Allocated GPU-hours |
|---|---|---:|---:|---:|
| Physical devices | SHD | 1 | 1 | 0.441481 ± 0.0119713 |
| Physical devices | SHD | 2 | 1 | 0.578704 ± 0.00478982 |
| Physical devices | SHD | 4 | 1 | 0.731852 ± 0.00944989 |
| Physical devices | SSC | 1 | 1 | 0.792130 ± 0.00861260 |
| Physical devices | SSC | 2 | 1 | 1.01222 ± 0.0252946 |
| Physical devices | SSC | 4 | 1 | 1.38296 ± 0.0234214 |
| Physical devices | CIFAR-10 | 1 | 1 | 6.42028 ± 0.205518 |
| Physical devices | CIFAR-10 | 2 | 1 | 8.59519 ± 0.356597 |
| One-GPU capacity | SHD | 1 | 1 | 0.427593 ± 0.0238636 |
| One-GPU capacity | SHD | 1 | 2 | 0.247130 ± 0.00416975 |
| One-GPU capacity | SHD | 1 | 4 | 0.163056 ± 0.00146986 |

Additional physical GPUs reduced elapsed training time, but every multi-GPU efficiency was below one. Reduced latency therefore does not automatically mean reduced GPU consumption; in these records, allocated GPU-hours rose for each physical multi-GPU treatment. Allocated GPU-hours use scheduler elapsed execution time and physical GPU count. Pending queue time is not included in allocated execution time, and round runtime alone is not an allocation or billing record. Utilization is a sampled activity percentage, not power or energy, so no energy-efficiency conclusion follows from it.

## Interpretation and limitations

The physical-device results provide descriptive evidence that this single-node coordination path preserved exact numerical results while reducing latency for the three measured workloads. The capacity results provide descriptive evidence that concurrent MPS processes increased one-GPU throughput while changing numerical outcomes. Neither observation establishes causal superiority.

Limitations are:

- three seeds;
- one HPC system;
- single-node execution only and no multinode result;
- concurrent independent allocations and possible system-load variation;
- two-second telemetry sampling;
- no direct energy measurement;
- no prediction records for capacity comparisons;
- numerical differences under MPS;
- no statistical-significance claim; and
- no novelty claim from distributed execution alone.

The evidence does not establish reduced physical network traffic, energy efficiency, multinode scalability, a published reproduction, implementation equivalence outside the declared exact comparisons, or thesis novelty. All result comparisons in this record are descriptive.

## Evidence and provenance

The physical-device collection provides the committed [JSON summary](../results/distributed_evaluation/distributed_evaluation_summary.json), [CSV summary](../results/distributed_evaluation/distributed_evaluation_summary.csv), [generated tables](../results/distributed_evaluation/distributed_evaluation_summary.md), [Slurm accounting](../results/distributed_evaluation/provenance/slurm-accounting.txt), [execution commit](../results/distributed_evaluation/provenance/execution-commit.txt), [array IDs](../results/distributed_evaluation/provenance/slurm-array-ids.txt), and [SHA-256 record](../results/distributed_evaluation/provenance/evidence-sha256.txt).

The capacity collection provides the committed [JSON summary](../results/device_capacity_evaluation/distributed_evaluation_summary.json), [CSV summary](../results/device_capacity_evaluation/distributed_evaluation_summary.csv), [generated tables](../results/device_capacity_evaluation/distributed_evaluation_summary.md), [Slurm accounting](../results/device_capacity_evaluation/provenance/slurm-accounting.txt), [execution commit](../results/device_capacity_evaluation/provenance/execution-commit.txt), [array ID](../results/device_capacity_evaluation/provenance/slurm-array-ids.txt), and [SHA-256 record](../results/device_capacity_evaluation/provenance/evidence-sha256.txt).
