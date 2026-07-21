# Hierarchical-reduction evaluation

## Status and objective

The two-node hierarchical-reduction experiment completed successfully. The committed collection is `valid`, evidence is complete, all validation-findings lists are empty, and all 12 executions passed the collection acceptance checks. The final decision is `node_hierarchical_reduction_not_retained`. This is a valid negative scientific result from completed executions.

The objective was to determine whether node-local sufficient-statistics grouping reduced logical inter-node update movement and retained acceptable runtime and numerical behavior on a fixed two-node allocation. This was an aggregation execution strategy for established FedAvg, not hierarchical federated learning.

## Experiment matrix

SHD and SSC were evaluated independently; datasets were not pooled. Each dataset used seeds 37, 47, and 57 and two aggregation treatments.

| Dataset | Seeds | Treatments per seed | Executions | Slurm allocations |
|---|---|---:|---:|---:|
| SHD | 37, 47, 57 | 2 | 6 | 3 |
| SSC | 37, 47, 57 | 2 | 6 | 3 |
| Total | — | — | 12 | 6 |

The comparison was between:

- `flat_ordered`: the established path, which gathers weighted terms at global rank zero in selected-client order;
- `node_hierarchical`: the same weighted contributions grouped within each node before the node-level results are combined.

## Hardware and allocation layout

Each dataset/seed combination occupied one `gpumedium` allocation on two Roihu nodes. Each node supplied two physical NVIDIA GH200 120GB GPUs and 144 CPU cores, for four GPUs, 288 CPU cores, four NCCL ranks, one process per GPU, and no CUDA MPS. Both treatments used the same four physical GPUs and ran sequentially within the allocation, with treatment order rotated across seeds. The evidence records two distinct GPU UUIDs per node, four distinct UUIDs globally, and the node-major rank mapping.

Slurm array `301313` completed successfully. Every corresponding task was `COMPLETED` with exit code `0:0`.

| Array task | Raw job ID | Dataset | Seed | Allocation elapsed (s) | Allocation GPU-hours |
|---|---:|---|---:|---:|---:|
| `301313_0` | `301314` | SHD | 37 | 1,406 | 1.562222 |
| `301313_1` | `301315` | SHD | 47 | 1,354 | 1.504444 |
| `301313_2` | `301316` | SHD | 57 | 1,383 | 1.536667 |
| `301313_3` | `301317` | SSC | 37 | 2,696 | 2.995556 |
| `301313_4` | `301336` | SSC | 47 | 2,717 | 3.018889 |
| `301313_5` | `301313` | SSC | 57 | 2,741 | 3.045556 |
| Total | — | — | — | 12,297 | 13.663333 |

Allocation elapsed time and GPU-hours are Slurm accounting values for the complete allocation. They are recorded once and are not attributed separately to both sequential treatments. `internal_treatment_duration_seconds` is the enclosing duration of one treatment used in allocation reconciliation. `derived_treatment_gpu_exposure_hours` is that duration multiplied by four GPUs and divided by 3,600; it is descriptive exposure, not separately billed accounting. All six allocation reconciliations were within the declared two-second tolerance.

## Workload invariants and aggregation difference

The paired evidence preserves dataset and preprocessing, split and partition, client population, selected clients and order, 100 rounds, ten of twenty clients per round, one local epoch, model initialization, optimizer and learning-rate policy, example-count FedAvg, validation, official-test isolation, event-structure scheduling, four-GPU process topology, and one process per GPU. Every client update was included once, and aggregation weights and denominators were correct.

Structural workload identity was preserved in all six dataset/seed pairs. The changed factor was the aggregation grouping order. Both treatments formed globally normalized weighted terms and used the same aggregation policy, but `node_hierarchical` first grouped terms within each node. The first recorded divergence in every pair was `aggregation_grouping_first_divergence`.

## Timing, communication, and telemetry results

Values below are arithmetic means across three seeds for each dataset and topology. Total runtime is the reported scientific workload duration; internal treatment duration is the larger allocation-reconciliation interval. Paired speedup and runtime reduction compare `node_hierarchical` with the paired `flat_ordered` reference.

| Dataset | Topology | Total runtime (s) | Internal duration (s) | Derived GPU exposure (h) | Paired speedup | Runtime reduction | Aggregation time (s) | Logical inter-node bytes | Scheduler overhead | Mean GPU utilization |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SHD | `flat_ordered` | 651.002 | 664.040 | 0.737823 | reference | reference | 27.1995 | 230,038,848 | 1.5947% | 39.7570% |
| SHD | `node_hierarchical` | 641.737 | 654.920 | 0.727689 | 1.01439 | 1.41365% | 12.9944 | 85,622,400 | 1.6133% | 39.4254% |
| SSC | `flat_ordered` | 1,286.500 | 1,308.277 | 1.453641 | reference | reference | 1.32386 | 84,089,400 | 8.5234% | 46.0298% |
| SSC | `node_hierarchical` | 1,315.953 | 1,338.047 | 1.486718 | 0.977602 | −2.2934% | 3.36221 | 31,260,000 | 8.3989% | 45.5533% |

For SHD, hierarchical grouping reduced the mean logical inter-node count by about 62.8%, decreased aggregation time from 27.1995 to 12.9944 seconds, and produced a small mean total-runtime improvement of 1.41365%. For SSC, it also reduced mean logical inter-node bytes by about 62.8%, but aggregation time increased from 1.32386 to 3.36221 seconds and total runtime was 2.2934% slower.

Logical communication bytes count modeled payload movement under the recorded topology; they are not measured physical network traffic. Aggregation time is measured separately from scheduler-decision overhead. The scheduler was held constant and was not the hierarchy treatment. Mean GPU utilization is measured telemetry, but it is not an energy measurement and does not convert the logical-byte result into a physical-network result.

## Numerical-difference results

Mathematical equivalence under the declared tolerances and exact numerical identity were not established for any hierarchical pair. Exact numerical identity refers to bitwise-identical parameters and is stricter than structural workload identity. Accuracy similarity alone would not establish either property.

| Dataset | Structural identity | Mathematical equivalence | Bitwise parameter identity | Mean maximum absolute parameter difference | Prediction identity | Checkpoint identity | First divergence |
|---|---|---|---|---:|---|---|---|
| SHD | established in 3/3 pairs | not established in 3/3 | not established in 3/3 | 0.219044 | differed in 3/3 | differed in 3/3 | aggregation grouping |
| SSC | established in 3/3 pairs | not established in 3/3 | not established in 3/3 | 0.398351 | differed in 3/3 | agreed in 3/3 | aggregation grouping |

Predictions and checkpoint selections were therefore not consistently identical for SHD. SSC selected the same checkpoint in all three pairs, but its official predictions differed in all three. These outcomes are consistent with the recorded first divergence at aggregation grouping; the evidence does not assign the difference to a changed scientific workload.

## Acceptance and decision gates

| Evidence or decision gate | Result |
|---|---|
| Collection valid; evidence complete; execution and measurement complete | passed |
| Provenance complete; execution-protocol and scheduler equivalence | passed |
| Official-test isolation preserved | passed |
| Every update included once; weights and denominators correct | passed |
| Logical inter-node movement reduced | passed |
| Structural and mathematical equivalence | not satisfied |
| Parameter differences within tolerance | not satisfied |
| Official predictions agree | not satisfied |
| Selected checkpoints agree across all pairs | not satisfied |
| No material runtime regression | not satisfied |

Hierarchical grouping materially reduced logical inter-node bytes, but the SHD runtime improvement was small, SSC became slower, and aggregation grouping changed floating-point results. The numerical-equivalence requirements were not satisfied. The evidence-based decision is therefore `node_hierarchical_reduction_not_retained`, and `flat_ordered` remains the established aggregation path.

## Limitations

The measurements cover three seeds, two audio datasets, two nodes, and four GH200 GPUs. They are descriptive rather than inferential and do not establish behavior at other node counts or on other hardware. Logical byte counts are derived communication accounting rather than physical network counters. GPU utilization is measured telemetry, but device energy was not evaluated. The observed numerical differences prevent using this collection as evidence of mathematical equivalence or exact identity, even where checkpoint selection or aggregate accuracy was similar.

## Evidence and provenance

The authoritative evidence is the committed [summary Markdown](../results/hierarchical_reduction_evaluation/hierarchical_reduction_evaluation_summary.md), [summary JSON](../results/hierarchical_reduction_evaluation/hierarchical_reduction_evaluation_summary.json), [summary CSV](../results/hierarchical_reduction_evaluation/hierarchical_reduction_evaluation_summary.csv), [acceptance record](../results/hierarchical_reduction_evaluation/hierarchical_reduction_evaluation_acceptance.json), [hypothesis-decision record](../results/hierarchical_reduction_evaluation/hierarchical_reduction_evaluation_decision.json), [validation findings](../results/hierarchical_reduction_evaluation/hierarchical_reduction_evaluation_validation_findings.json), [provenance record](../results/hierarchical_reduction_evaluation/hierarchical_reduction_evaluation_provenance.json), [Slurm accounting](../results/hierarchical_reduction_evaluation/slurm-accounting.txt), and [job-ID map](../results/hierarchical_reduction_evaluation/slurm-job-id-map.txt). Generated figures include the [paired runtime comparison](../results/hierarchical_reduction_evaluation/figures/hierarchical_paired_runtime_difference.png), [maximum parameter difference](../results/hierarchical_reduction_evaluation/figures/hierarchical_maximum_parameter_difference.png), [SHD logical movement](../results/hierarchical_reduction_evaluation/figures/shd_logical_inter_node_movement.png), and [SSC logical movement](../results/hierarchical_reduction_evaluation/figures/ssc_logical_inter_node_movement.png).

Scientific execution commit: `d06ddcc775cc400c2188a6f104c2665041137ea7`.

Evidence archive commit: `d3f93227eb9f45d315747105268ba416e58ec0a1`.
