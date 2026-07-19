# Single-node distributed FedAvg evaluation

Status: **valid**

Each workload uses its own one-GPU distributed execution as the timing and numerical reference.

## SHD

| GPUs | Client processes/GPU | Seeds | Test accuracy | Macro-F1 | Peak allocation (bytes) | GPU utilization (%) | Load imbalance | Structural | Numerical |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 1 | 3/3 | 0.699647 ± 0.029954 | 0.696571 ± 0.0301767 | 1.16935e+08 ± 48923 | 16.9111 ± 0.564368 | 0 ± 0 | exact | exact |
| 2 | 1 | 3/3 | 0.699647 ± 0.029954 | 0.696571 ± 0.0301767 | 1.16935e+08 ± 48923 | 25.8264 ± 0.825906 | 0.330663 ± 0.0733092 | exact | exact |
| 4 | 1 | 3/3 | 0.699647 ± 0.029954 | 0.696571 ± 0.0301767 | 1.16861e+08 ± 99774.3 | 42.7625 ± 1.25962 | 0.63036 ± 0.0385039 | exact | exact |

| GPUs | Client processes/GPU | Total runtime (s) | Round time (s) | Client wall time (s) | Aggregation (s) | Validation (s) | Speedup | Efficiency |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 1561.09 ± 42.5351 | 15.6109 ± 0.425351 | 1443.54 ± 39.5169 | 4.22558 ± 0.0823837 | 105.842 ± 4.22549 | 1 ± 0 | 1 ± 0 |
| 2 | 1 | 1011.02 ± 8.85204 | 10.1102 ± 0.0885204 | 884.192 ± 12.9664 | 4.1819 ± 0.110672 | 114.839 ± 6.7909 | 1.544 ± 0.0342813 | 0.771999 ± 0.0171407 |
| 4 | 1 | 620.95 ± 8.46605 | 6.2095 ± 0.0846605 | 496.858 ± 6.01688 | 4.28002 ± 0.188848 | 111.639 ± 2.69447 | 2.51471 ± 0.0930752 | 0.628677 ± 0.0232688 |

## SSC

| GPUs | Client processes/GPU | Seeds | Test accuracy | Macro-F1 | Peak allocation (bytes) | GPU utilization (%) | Load imbalance | Structural | Numerical |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 1 | 3/3 | 0.426635 ± 0.0128545 | 0.387055 ± 0.0147739 | 2.08718e+08 ± 0 | 13.0648 ± 0.25605 | 0 ± 0 | exact | exact |
| 2 | 1 | 3/3 | 0.426635 ± 0.0128545 | 0.387055 ± 0.0147739 | 2.08718e+08 ± 0 | 25.6943 ± 0.341631 | 0.230179 ± 0.0514786 | exact | exact |
| 4 | 1 | 3/3 | 0.426635 ± 0.0128545 | 0.387055 ± 0.0147739 | 2.08718e+08 ± 0 | 45.1094 ± 0.473272 | 0.532826 ± 0.02157 | exact | exact |

| GPUs | Client processes/GPU | Total runtime (s) | Round time (s) | Client wall time (s) | Aggregation (s) | Validation (s) | Speedup | Efficiency |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 2812.16 ± 31.2943 | 28.1216 ± 0.312943 | 2447.23 ± 28.0459 | 1.67258 ± 0.0627198 | 355.556 ± 3.59127 | 1 ± 0 | 1 ± 0 |
| 2 | 1 | 1777.76 ± 44.2429 | 17.7776 ± 0.442429 | 1403.9 ± 43.2581 | 1.59651 ± 0.263451 | 364.049 ± 1.98358 | 1.58222 ± 0.0227238 | 0.791112 ± 0.0113619 |
| 4 | 1 | 1192.56 ± 22.1012 | 11.9256 ± 0.221012 | 820.282 ± 19.6539 | 1.65061 ± 0.0272001 | 361.925 ± 3.38151 | 2.3583 ± 0.0176213 | 0.589575 ± 0.00440534 |

## CIFAR10

| GPUs | Client processes/GPU | Seeds | Test accuracy | Macro-F1 | Peak allocation (bytes) | GPU utilization (%) | Load imbalance | Structural | Numerical |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 1 | 3/3 | 0.7371 ± 0.0192486 | 0.735136 ± 0.0174057 | 2.20546e+09 ± 0 | 32.2526 ± 0.672774 | 0 ± 0 | exact | exact |
| 2 | 1 | 3/3 | 0.7371 ± 0.0192486 | 0.735136 ± 0.0174057 | 2.20546e+09 ± 0 | 37.2154 ± 2.88017 | 0.529373 ± 0.0798654 | exact | exact |

| GPUs | Client processes/GPU | Total runtime (s) | Round time (s) | Client wall time (s) | Aggregation (s) | Validation (s) | Speedup | Efficiency |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 23074.3 ± 739.096 | 230.743 ± 7.39096 | 23053.6 ± 739.103 | 6.21229 ± 0.0348419 | 0 ± 0 | 1 ± 0 | 1 ± 0 |
| 2 | 1 | 15427.8 ± 641.652 | 154.278 ± 6.41652 | 15405 ± 641.091 | 6.92578 ± 0.310199 | 0 ± 0 | 1.49625 ± 0.0356666 | 0.748123 ± 0.0178333 |

Total runtime is the sum of recorded communication-round total durations. Speedup and parallel efficiency are paired within each workload. Profiling traces, when enabled, add overhead and are not ordinary runtime evidence. Internal execution movement is separate from logical federated communication.
