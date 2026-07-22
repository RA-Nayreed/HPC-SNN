# Non-IID severity and measured-energy evaluation

## Status

This record defines an unexecuted prospective collection. Its implementation is locally verified, but scientific execution has not occurred and no Week 7 result exists. It makes no heterogeneity, energy-efficiency, accuracy, adoption, universality, significance, novelty, or production-readiness claim. Weeks 1–6 and their committed evidence remain unchanged; existing `results/` artifacts are not evidence for this collection.

## Question and matrix

The collection asks how client-data heterogeneity affects runtime, population/event/load imbalance, official-test performance, logical communication, and measured GPU energy for the fixed SHD 256/256 and SSC 128/128 LIF workloads. It crosses two datasets, seeds 37/47/57, and deterministic IID plus label-Dirichlet alpha 1.0, 0.5, and 0.1, for 24 executions in six four-GPU Slurm allocations.

Each dataset/seed allocation executes four fresh distributed processes in rotated order: IID/1.0/0.5/0.1 at seed 37, 1.0/0.1/IID/0.5 at seed 47, and 0.5/IID/0.1/1.0 at seed 57. Every treatment uses one node, four physical GPUs, four processes, `example_count_longest_processing_time`, `flat_ordered`, sample-count FedAvg, and no MPS. Scientific initialization and client-selection streams are identical within dataset/seed; only partition treatment and its provenance change.

## Partition and evidence gates

All eligible training indices must appear exactly once. Validation and test labels cannot influence partitions. Evidence retains total eligible examples, each client population and presented count, class counts, represented classes, entropy, partition hash and seed, plus retry or deterministic repair actions. Extreme alpha cannot silently remove a client or example; an unsatisfied minimum-population rule fails with an exact explanation.

The versioned resolved-config whitelist permits only distribution/alpha, partition identity/provenance, rotated order, and treatment/execution/output identity differences. Resume compatibility binds those fields together with scientific seed, initialization, selection stream, scheduler, aggregation, topology, process/device mapping, measurement, calibration, and Git state. Every treatment and attempt has independent process, random, loader, checkpoint, telemetry, and accounting state.

## Measurement and analysis

A topology-compatible 100 ms four-GPU calibration is required. Per-node telemetry, UUID validation, boundary interpolation, per-device integration, phase separation, failure handling, and allocation reconciliation follow the [prospective protocol](../docs/experimental_protocol.md#measured-energy-calibration-and-accounting). Allocation elapsed time and billed GPU-hours are stored once; four internal treatment durations and derived exposures are not billed four times.

The summary reports three-seed mean and sample standard deviation separately for each dataset/treatment. Accuracy and macro-F1 differences are paired to IID by dataset and seed. It also reports validation metric, checkpoint round, runtime, client wall time, population/event/load imbalance, gross, idle-adjusted and client energy, energy per round and accepted client, utilization, peak memory, and logical communication. Negative differences and non-monotonic trends are observations, not evidence failures. Eight deterministic figures are generated only by the explicit command in the [reproducibility guide](../docs/reproducibility.md#week-7-prospective-execution).

## Limitations

The design is descriptive and runs no significance test. Datasets are never pooled. Logical communication is not physical network traffic, and measured joules are not allocated GPU-hours. Frozen Week 5 models are transfer diagnostics only. CPU-only local checks do not validate CUDA, NCCL, NVML, or GH200 performance. Scientific conclusions require complete accepted Roihu execution and accounting evidence.
