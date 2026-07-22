# System scaling and measured-energy evaluation

## Status

This record defines an unexecuted prospective collection. Its implementation is locally verified, but scientific execution has not occurred and no Week 7 result exists. It makes no scaling, energy-efficiency, accuracy, numerical-equivalence, adoption, universality, significance, novelty, or production-readiness claim. Weeks 1–6 and their committed evidence remain unchanged; existing `results/` artifacts are not evidence for this collection.

## Question and matrix

The collection asks how physical execution layout affects runtime, load balance, official-test performance, logical communication, measured GPU energy, and numerical identity for the fixed SHD 256/256 and SSC 128/128 LIF workloads. It crosses two datasets, seeds 37/47/57, and `one_node_one_gpu`, `one_node_two_gpu`, `one_node_four_gpu`, and `two_nodes_four_gpus`, for 24 executions in 24 Slurm allocations. Label-Dirichlet alpha 0.5 is fixed.

Every execution uses 20 clients, ten selected clients, 100 rounds, one local epoch, Adam at 0.001, sample-count FedAvg through `flat_ordered`, `example_count_longest_processing_time`, one process per physical GPU, and no MPS. Dataset splitting, eligible training indices, initialization, selection, client seeds, selected order, weights, validation, checkpoint selection, and one rank-zero official-test evaluation are invariant within each dataset/seed comparison.

## Identity and evidence gates

The one-GPU execution is the reference. Evidence retains selected clients and seeds, ordered updates and weights, checkpoint round and tensors, predictions, accuracy, and macro-F1. Structural identity, exact tensors, bounded differences, prediction identity, checkpoint identity, and metric identity are classified independently. A numerical difference remains an executable outcome but cannot be called exact equivalence.

The versioned resolved-config whitelist permits only topology, mapping, rendezvous, treatment/execution/output identity, and allocation-description differences. Resume compatibility additionally binds distribution, partition, initialization, selection stream, scheduler, aggregation, measurement, calibration, Git state, and topology. Attempts retain their own telemetry and accounting.

## Measurement and analysis

One 100 ms NVML sampler and file per node cover every canonical physical GPU UUID. Energy is boundary-interpolated and integrated per GPU before summation; same-device client overlaps fail, while cross-device concurrency is valid. Complete treatment energy remains distinct from client, distribution, collection, aggregation, validation, official-test, checkpoint, idle, interrupted, and unattributed energy. Every topology requires a matching calibration with ten alternating pairs, at most 2% median overhead, identical updates, adequate coverage, no sampling errors, and no official-test access.

The summary reports three-seed mean and sample standard deviation by dataset/topology, paired speedup and parallel efficiency, energy ratio, derived energy-delay product, runtime and phase timing, official metrics, utilization, memory, load, communication, inter-node movement, allocation GPU-hours, derived exposure, and numerical classifications. Datasets are never pooled and monotonic improvement is not an acceptance condition. The eight deterministic figures are defined in the [protocol](../docs/experimental_protocol.md#analysis-and-acceptance) and generated only by the explicit command in the [reproducibility guide](../docs/reproducibility.md#week-7-prospective-execution).

## Limitations

Logical communication is not measured network traffic. Integrated device energy is not Slurm billing, and derived treatment GPU exposure is not separately billed. Frozen Week 5 predictions are transfer diagnostics, not universal predictors and not scheduling inputs. CPU-only local checks do not validate CUDA, NCCL, multinode behavior, NVML, or GH200 performance. A complete negative or non-monotonic result remains valid evidence.
