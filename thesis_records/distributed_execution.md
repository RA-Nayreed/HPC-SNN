# Distributed execution

## Scientific objective

Evaluate whether synchronous FedAvg preserves each workload’s scientific identity and numerical behavior when selected clients execute concurrently on one, two, or four GPUs within one Roihu node. Separately provide the capability to measure multiple independent client processes on one physical GPU through CUDA MPS without treating unmeasured packing as preferable.

## Workloads and matrix

The canonical `experiments/distributed_evaluation/manifest.yaml` contains eight exclusive-device treatments crossed with seeds 7, 17, and 27: SHD on one, two, and four GPUs; SSC on one, two, and four GPUs; and non-IID alpha-0.5 CIFAR-10 on one and two GPUs. This is exactly 24 tasks: 9 SHD, 9 SSC, and 6 CIFAR-10. CIFAR-10 has no four-GPU treatment because only two clients participate per round.

SHD retains the established 256/256 LIF 20/10 reference, one local epoch, local batch 32, Adam at `0.001`, clipping 1, example-count aggregation, derived validation, and best-validation checkpoint selection. SSC is an independent evaluation using the complete 75,466-example official training collection for a 20-client label-Dirichlet alpha-0.5 partition, ten selected clients, one local epoch, local batch 256, the established 128/128 LIF model, 700-to-140 channel reduction, 10 ms bins, example-count aggregation, the complete 9,981-example official validation collection for selection, and the 20,382-example official test once afterward. CIFAR-10 reuses the active paper-reported non-IID S-VGG9 BNTT configuration with ten clients, two selected, five local epochs, 20 timesteps, uniform aggregation, no internal validation, round-100 selection, and one later official-test evaluation.

Scientific identity is equal across device-count treatments within a workload and distinct across workloads. Each workload’s one-GPU distributed path defines the timing and numerical reference; earlier sequential runtimes and other datasets are not valid speedup denominators.

## Coordination and determinism

Rank 0 alone selects clients, owns the authoritative global state, restores detached CPU client results to selected-client order, calls the existing configured FedAvg implementation once, validates when required, writes shared records and checkpoints, and constructs the official-test dataset after selection. Nonzero ranks construct neither validation nor official-test datasets. Every rank may train assigned clients. Assignment is `selected_position % process_count`.

Client Python, NumPy, CPU Torch, CUDA, loader, dropout, and model randomness derive from the scientific seed streams, communication round, and client ID. Process rank, device index, device slot, process arrival, and device count do not enter the seed. The next round begins only after every expected client result has been checked and the current aggregate is durable.

Client results preserve client and round identity, selected position, process/device mapping, population and presented examples, batches, training seed, update norm, spike statistics, training and data-wait durations, allocated and reserved memory, and detached CPU tensors. Missing, duplicate, unexpected, non-finite, shape-incompatible, or dtype-incompatible results abort the round. Logical communication remains the existing client download/upload quantity; process movement is separate execution data movement.

## Exclusive devices and CUDA MPS

Exclusive execution requires one process per physical GPU, NCCL control and tensor communication, and `cuda_process_service: none`. Supported physical-device counts are one, two, and four, subject to the selected-client count.

Same-GPU packing is represented separately under `experiments/device_capacity_evaluation/` and is absent from the 24-task matrix. It supports one, two, or four physical GPUs and one, two, or four client processes per device, with `process_count = device_count * client_processes_per_device`. Total processes cannot exceed selected clients. Mapping is `device_index = process_rank % device_count` and `device_slot = process_rank // device_count`. Packing requires Gloo for CPU control and detached CPU state movement plus CUDA MPS for client computation; several NCCL ranks cannot share one GPU.

The Roihu script creates unique MPS pipe and log directories in the job-local area, starts and verifies the control daemon, applies `floor(100 / client_processes_per_device)` as the active-thread percentage, launches the configured torchrun process count, stops MPS for every exit path, and archives logs. This percentage does not guarantee equal GPU time. No MPS topology is scientifically preferred without device-capacity measurements.

## Measurements and resumption

Every round records physical devices, processes per device, total processes, process mapping, client-to-process and client-to-device assignments, client duration and data wait, process/device busy duration, parallel client wall time, summed client time, aggregation, validation, checkpoint, total round duration, clients and examples per process/device, estimated idle time, load imbalance, and allocated/reserved memory per process. Process resident host memory is captured immediately before and after workload construction together with the signed difference for every execution attempt; only stable rank-to-device fields participate in resumable topology identity. Available Roihu telemetry contributes finite physical-device utilization sample counts and aggregate and per-device minima, means, and maxima. Allocation history is retained across resumption; utilization remains scoped to its latest execution attempt and is excluded from treatment statistics for resumed runs. Device utilization, logical communication, internal movement, Slurm allocation, and busy time remain separate quantities.

Optional PyTorch profiling is disabled for ordinary treatments. When explicitly enabled for named rounds, rank-specific traces cover CPU and CUDA operations, copies, synchronization, client-model construction, forward, backward, optimizer, aggregation, and validation. Official-test evaluation is not profiled by default. Profiler overhead prevents interpreting traced rounds as ordinary runtime.

Checkpoint compatibility includes dataset, complete scientific identity, device count, client processes per device, process count, process mapping, control backend, CUDA process service, assignment and aggregation order, Git state, split, partition, and initialization. A failure during client work does not advance `last.pt`; resumption repeats that complete round.

The summarizer groups workload before treatment. It checks exact scientific, split, partition, selection, client-seed, example-count, aggregation, logical-communication, checkpoint-selection, and official-test-access identities. It measures maximum absolute and relative selected-checkpoint parameter differences, selected-round equality, accuracy and macro-F1 differences, and prediction agreement when predictions are recorded. Accuracy similarity alone is not numerical equivalence, and no numerical tolerance is invented.

## Evidence collected

Local CPU/Gloo tests verify one- and two-process synchronous execution, simultaneous client entry, selected-order restoration, one aggregation call, exact aggregate identity, round-atomic interruption and resumption, failure without checkpoint advancement, coordinator-only validation/test behavior, and unchanged logical communication. CPU tests also verify SHD, SSC, and CIFAR workload routing; SSC split isolation; the 24-task matrix; MPS process/device mapping and backend rules; topology-bound checkpoints; grouped Slurm commands; and MPS cleanup structure.

No one-, two-, or four-GPU CUDA/NCCL scientific execution and no CUDA MPS capacity execution has been run or submitted for this collection. No distributed result summary exists in the repository.

## Unresolved scientific limitations

CUDA numerical differences, run-to-run variability, GH200 load imbalance, data-pipeline limits, repeated small timestep kernels, NCCL and Gloo movement cost, MPS interference, device utilization, speedup, parallel efficiency, memory capacity, allocated GPU-hours, and Slurm allocation effects remain unmeasured. Three seeds support descriptive paired reporting but do not establish statistical significance. There is no utilization, energy, causality, or implementation-equivalence claim.

## Conclusions permitted by the evidence

The repository contains one dataset-independent synchronous coordination path, a validated 24-task exclusive-device matrix, and separately validated MPS capacity controls. Local tests establish coordination invariants on CPU. They do not establish CUDA/NCCL/MPS numerical equivalence, acceleration, utilization, device capacity, energy efficiency, or a thesis result.
