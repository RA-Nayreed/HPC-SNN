# Scheduling evaluation

## Status and objective

The scheduling experiment completed successfully. The committed collection is `valid`, evidence is complete, all validation-findings lists are empty, and all 18 executions passed the declared acceptance checks. The final decision is `event_structure_scheduler_not_adopted`. This is a valid negative scientific result from completed executions.

The objective was to test whether a frozen event-structure predicted-cost model could reduce synchronous round time by assigning clients already selected by the established federated protocol. Scheduler treatment changed assignment and timing, not client selection or the scientific workload.

## Experiment matrix

SHD and SSC were evaluated independently; datasets were not pooled. Each dataset used seeds 37, 47, and 57 and three scheduler treatments.

| Dataset | Seeds | Treatments per seed | Executions | Slurm allocations |
|---|---|---:|---:|---:|
| SHD | 37, 47, 57 | 3 | 9 | 3 |
| SSC | 37, 47, 57 | 3 | 9 | 3 |
| Total | — | — | 18 | 6 |

The compared treatments were:

- `round_robin`: round-robin assignment;
- `example_count_longest_processing_time`: example-count longest-processing-time assignment;
- `event_structure_longest_processing_time`: frozen event-structure predicted-cost longest-processing-time assignment.

## Hardware and allocation layout

Each dataset/seed combination occupied one `gpumedium` allocation on one Roihu node with four physical NVIDIA GH200 120GB GPUs, 288 CPU cores, four NCCL processes, one process per GPU, and no CUDA MPS. The three treatments ran sequentially within that allocation, with treatment order rotated across seeds. Four distinct allocation GPU UUIDs and the rank-to-device mapping were recorded before workload execution.

Slurm array `301305` completed successfully. Every corresponding task was `COMPLETED` with exit code `0:0`.

| Array task | Raw job ID | Dataset | Seed | Allocation elapsed (s) | Allocation GPU-hours |
|---|---:|---|---:|---:|---:|
| `301305_0` | `301306` | SHD | 37 | 1,885 | 2.094444 |
| `301305_1` | `301307` | SHD | 47 | 1,858 | 2.064444 |
| `301305_2` | `301308` | SHD | 57 | 1,846 | 2.051111 |
| `301305_3` | `301309` | SSC | 37 | 3,698 | 4.108889 |
| `301305_4` | `301310` | SSC | 47 | 3,660 | 4.066667 |
| `301305_5` | `301305` | SSC | 57 | 3,804 | 4.226667 |
| Total | — | — | — | 16,751 | 18.612222 |

Allocation elapsed time and GPU-hours are Slurm accounting values for the complete allocation. They are recorded once and are not attributed separately to each sequential treatment. `internal_treatment_duration_seconds` is the enclosing duration of one treatment used in allocation reconciliation. `derived_treatment_gpu_exposure_hours` is that internal duration multiplied by four GPUs and divided by 3,600; it is descriptive exposure, not separately billed accounting. All six allocation reconciliations were within the declared two-second tolerance.

## Scientific invariants

The paired evidence preserves dataset and preprocessing, split and partition, client population, selected clients and order, 100 rounds, ten of twenty clients per round, one local epoch, model initialization, optimizer and learning-rate policy, example-count FedAvg weights and denominators, validation, checkpoint selection, and isolated official-test evaluation. The aggregation topology remained `flat_ordered`.

Execution structure and model-state structure matched in all 12 paired scheduler comparisons. Mathematical equivalence and exact numerical identity were both established: maximum absolute and relative parameter differences were zero; parameters, predictions, checkpoint selections, and official-test results were identical within every dataset/seed comparison. Scheduler choice therefore changed assignment and timing while retaining the required scientific equivalence.

## Timing and telemetry results

Values below are arithmetic means across the three seeds for each dataset and treatment. Total runtime is the reported scientific workload duration; internal treatment duration is the larger allocation-reconciliation interval. Paired speedup and runtime reduction use `round_robin` as the within-dataset, within-seed reference.

| Dataset | Treatment | Total runtime (s) | Internal duration (s) | Derived GPU exposure (h) | Paired speedup | Runtime reduction | Scheduler overhead | Mean GPU utilization |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| SHD | `round_robin` | 612.886 | 620.837 | 0.689819 | reference | reference | 0.00880% | 43.0092% |
| SHD | `example_count_longest_processing_time` | 566.865 | 575.242 | 0.639158 | 1.08135 | 7.4718% | 0.2388% | 38.7162% |
| SHD | `event_structure_longest_processing_time` | 579.134 | 591.677 | 0.657419 | 1.05986 | 5.4419% | 1.7155% | 38.9008% |
| SSC | `round_robin` | 1,188.777 | 1,206.485 | 1.340539 | reference | reference | 0.00460% | 45.8212% |
| SSC | `example_count_longest_processing_time` | 1,130.676 | 1,148.630 | 1.276256 | 1.05128 | 4.8675% | 0.1560% | 42.7800% |
| SSC | `event_structure_longest_processing_time` | 1,254.672 | 1,276.985 | 1.418872 | 0.947381 | −5.5675% | 8.8030% | 45.6465% |

Scheduler-decision overhead is the time spent making assignments and is distinct from aggregation time. The reported GPU-utilization values are measured telemetry associated with each treatment; they do not measure energy. Aggregation timing and logical payload accounting remain separate fields in the summary. Logical bytes describe modeled payload movement, not physical network traffic.

For SHD, event-structure scheduling improved the three-seed mean. Seed 37 was 0.6805% slower than its paired round-robin reference, while seeds 47 and 57 improved by 9.8691% and 7.1371%. Mean official-test accuracy was 0.705536 and was identical across the three scheduler strategies.

For SSC, all three event-structure comparisons were slower than their paired round-robin references, by 4.8943%, 7.2394%, and 4.5687% for seeds 37, 47, and 57. Mean official-test accuracy was 0.437723 and was identical across scheduler strategies.

## Acceptance and decision gates

| Evidence or decision gate | Result |
|---|---|
| Collection valid; evidence complete; execution and measurement complete | passed |
| Provenance complete; official-test isolation preserved | passed |
| Execution-protocol, scheduler, aggregation, and scientific equivalence | passed |
| Exact dataset/seed pair coverage | passed |
| Permitted pre-execution information only | passed |
| Predictions and checkpoints identical | passed |
| SHD mean runtime improvement at least 5% | passed |
| SSC mean runtime improvement at least 5% | not satisfied |
| Event-structure overhead below 1% | not satisfied |
| At least two improved seeds in each dataset | not satisfied |
| No dataset/seed pair more than 2% slower | not satisfied |
| No slower than example-count scheduling in each dataset | not satisfied |

Event-structure scheduling helped SHD on average, but it did not generalize successfully to SSC, exceeded the declared decision-overhead threshold, and did not consistently outperform the simpler example-count strategy. The paired per-dataset and per-seed adoption conditions were not satisfied. The evidence-based decision is therefore `event_structure_scheduler_not_adopted`.

## Limitations

The measurements cover three seeds, two audio datasets, one single-node four-GH200 topology, and one frozen cost model. They are descriptive rather than inferential. Treatment order was rotated, but treatments still ran sequentially within shared allocations. Derived treatment GPU exposure is not billed GPU accounting; logical bytes are not physical network counters; GPU utilization is not energy measurement. The evidence does not support claims beyond the tested scheduler, workloads, hardware, and adoption gates.

## Evidence and provenance

The authoritative evidence is the committed [summary Markdown](../results/scheduling_evaluation/scheduling_evaluation_summary.md), [summary JSON](../results/scheduling_evaluation/scheduling_evaluation_summary.json), [summary CSV](../results/scheduling_evaluation/scheduling_evaluation_summary.csv), [acceptance record](../results/scheduling_evaluation/scheduling_evaluation_acceptance.json), [hypothesis-decision record](../results/scheduling_evaluation/scheduling_evaluation_decision.json), [validation findings](../results/scheduling_evaluation/scheduling_evaluation_validation_findings.json), [provenance record](../results/scheduling_evaluation/scheduling_evaluation_provenance.json), [Slurm accounting](../results/scheduling_evaluation/slurm-accounting.txt), and [job-ID map](../results/scheduling_evaluation/slurm-job-id-map.txt). Generated figures include the [paired runtime comparison](../results/scheduling_evaluation/figures/paired_runtime_reduction_by_dataset_and_seed.png), [SHD overhead](../results/scheduling_evaluation/figures/shd_scheduler_overhead_fraction.png), and [SSC overhead](../results/scheduling_evaluation/figures/ssc_scheduler_overhead_fraction.png).

Scientific execution commit: `d06ddcc775cc400c2188a6f104c2665041137ea7`.

Evidence archive commit: `d3f93227eb9f45d315747105268ba416e58ec0a1`.
