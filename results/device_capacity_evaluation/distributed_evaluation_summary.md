# Single-node distributed FedAvg evaluation

Status: **valid**

Each workload uses its own one-GPU distributed execution as the timing and numerical reference.

## SHD

| GPUs | Client processes/GPU | Seeds | Test accuracy | Macro-F1 | Peak allocation (bytes) | GPU utilization (%) | Load imbalance | Structural | Numerical |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 1 | 3/3 | 0.699647 ± 0.029954 | 0.696571 ± 0.0301767 | 1.16935e+08 ± 48923 | 17.2476 ± 1.0686 | 0 ± 0 | exact | exact |
| 1 | 2 | 3/3 | 0.699647 ± 0.0214393 | 0.694716 ± 0.0201077 | 1.16934e+08 ± 48923 | 32.588 ± 1.56503 | 0.297964 ± 0.0475917 | exact | unavailable_or_difference_observed |
| 1 | 4 | 3/3 | 0.701561 ± 0.0271646 | 0.700135 ± 0.0255618 | 1.1686e+08 ± 99774.3 | 44.7365 ± 0.912779 | 0.639708 ± 0.0479606 | exact | unavailable_or_difference_observed |

| GPUs | Client processes/GPU | Total runtime (s) | Round time (s) | Client wall time (s) | Aggregation (s) | Validation (s) | Speedup | Efficiency |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 1515.45 ± 85.4437 | 15.1545 ± 0.854437 | 1400.96 ± 76.7282 | 4.22374 ± 0.150878 | 102.839 ± 8.9681 | 1 ± 0 | 1 ± 0 |
| 1 | 2 | 864.519 ± 14.8104 | 8.64519 ± 0.148104 | 754.675 ± 11.1059 | 2.37101 ± 0.0604805 | 100.624 ± 4.47412 | 1.75221 ± 0.0714499 | 0.876106 ± 0.035725 |
| 1 | 4 | 561.249 ± 5.20705 | 5.61249 ± 0.0520705 | 453.07 ± 1.54042 | 2.01539 ± 0.0269887 | 99.4348 ± 4.04286 | 2.69982 ± 0.1419 | 0.674955 ± 0.035475 |

Total runtime is the sum of recorded communication-round total durations. Speedup and parallel efficiency are paired within each workload. Profiling traces, when enabled, add overhead and are not ordinary runtime evidence. Internal execution movement is separate from logical federated communication.
