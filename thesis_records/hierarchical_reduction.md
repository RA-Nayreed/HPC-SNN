# Node-hierarchical reduction

## Systems question

This prospective evaluation asks whether node-local sufficient-statistics reduction decreases logical inter-node update movement on a fixed two-node/two-GPU-per-node allocation. It is an execution strategy for established FedAvg, not hierarchical federated learning.

The flat path gathers updates at global rank zero and the node-hierarchical path groups those same contributions by node. Both use the `established_flat_weighted_terms_v1` tensor policy inherited literally from pre-Week-6 FedAvg: uniform or example-count weights are normalized as Python floats before tensor arithmetic; every floating or complex input is cast to float64 and added in selected-client order with its normalized weight as `alpha`; and the completed accumulator is cast back to the original dtype without a post-sum division. The historical float64 conversion of complex inputs, including its imaginary-component behavior, is deliberately unchanged. Integral and Boolean buffers require exact agreement across every client and preserve their input dtype. Both paths enforce nonempty input, state-key, shape, dtype, normalized-weight, contribution-coverage, finite-value, and nonfloating-equality validation. Hierarchical reduction forms the same weighted terms within each node and changes only their grouping. Logical movement is not physical network measurement.

Fully resolved paired configurations are compared by a versioned whitelist. Hierarchical pairs may differ only in aggregation topology, treatment/order identity, and run/output identity; all named dataset, split, partition, client, round, model, initialization, optimizer, training, seed, FedAvg, validation, checkpoint, official-test, GPU, process, and process-per-GPU invariants must compare equal.

Each dataset/seed allocation contains two sequential treatments. Allocation elapsed time and billed GPU-hours are recorded once, while each treatment reports an internal duration and clearly derived four-GPU exposure. The two durations plus initialization, between-treatment overhead, remaining overhead, and the declared at-most-two-second reconciliation error must equal the allocation elapsed time.

The Roihu request is `gpumedium`, two nodes, two Slurm tasks with one task per node, two GH200 GPUs and 144 CPU cores per task/node, one distributed agent and two workers per node, four global workers, 36 hours, and no CUDA MPS. A two-node preflight requires two nonempty distinct UUIDs on each node and exactly four distinct UUIDs globally. Before workload construction, every NCCL rank proves exact node-major rank/local-rank/device-index membership in that allocation UUID set.

## Numerical classification

Each SHD/SSC seed-37/47/57 pair will report structural identity, mathematical equivalence under declared tolerances, bitwise parameter identity, maximum absolute and relative parameter difference, checkpoint selection identity, prediction identity, and official-test metric differences. Floating reduction grouping may prevent bitwise identity; accuracy similarity alone is insufficient.

## Status

No hierarchical-reduction scientific execution or result is created by the implementation work. Retention remains undecided until exactly 12 Roihu executions pass contribution, normalized-weight, topology, numerical, prediction, checkpoint, movement, runtime, official-test, and provenance gates.
