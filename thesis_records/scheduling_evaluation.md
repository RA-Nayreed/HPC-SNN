# Scheduling evaluation

## Scientific question

This prospective evaluation asks whether the frozen event-structure wall-time model can reduce synchronous round time by assigning clients already selected by the established protocol. It is not a client-selection experiment.

The declared matrix contains SHD and SSC, seeds 37/47/57, and round-robin, example-count longest-processing-time, and event-structure longest-processing-time assignment. All paired scientific inputs, client seeds, updates, selected-order FedAvg, validation, checkpoint selection, and one-time official-test access must remain identical.

## Model and metadata

The deployable model is `results/resource_measurement/client_cost_model.json`, verified against SHA-256 `78ad111c6997999e017b7b29c07fadfdab86a11a83c128b06e2b8924d4a1471c`. It is a ridge event-structure model targeting client wall time. The accepted source collection contains 6,000 unique client rows: 4,000 seed-7/17 fitting rows and 2,000 untouched seed-27 evaluation rows. The immutable evaluation-provenance artifact has SHA-256 `1488d3542e5b0770ab750b9d3812aeeac86bc091e3a11954e1f8a6a73ba9f924`. Its recorded row identities prove zero overlap between seed 27 and the coefficient-fitting or seed-7/17 client-grouped selection rows. Seed 27 did not affect normalization, coefficient fitting, regression-family selection, feature selection, or hyperparameter selection. It was used only for post-freeze evaluation and the post-freeze adoption decision. Diagnostic-oracle artifacts and unpermitted features are rejected.

The simulation computes ten workload fields centrally from training data. A real client would disclose those values to the coordinator. They contain no labels but may reveal example volume, sequence shape, padding, and event density. Raw events are not scheduler metadata. No privacy-preserving claim is made.

## Pairing, execution, and accounting

Manifest validation compares fully resolved paired configurations against a versioned whitelist. Scheduling pairs may differ only in scheduler strategy, treatment/order identity, and run/output identity. The validator separately proves equality of dataset and preprocessing, split, partition, client population and participation, selected-client count, rounds, model and initialization, optimizer and learning-rate policy, local training and loader rules, client seed derivation, FedAvg weighting, validation and checkpoint selection, official-test policy and one access, and the four-GPU/four-process/one-process-per-GPU topology.

Each dataset/seed allocation runs three treatments sequentially. Slurm elapsed time and billed GPU-hours therefore exist once at allocation scope. Each treatment retains only its internal wall duration and a derived four-GPU exposure. Reconciliation requires the sum of the three internal durations, initialization, between-treatment overhead, remaining overhead, and a declared at-most-two-second reconciliation error to equal allocation elapsed time. Derived exposure is not separately billed accounting.

The Roihu request is `gpumedium`, one node, one Slurm task, four GH200 GPUs, four workers, one worker per GPU, 288 CPU cores, 36 hours, and no CUDA MPS. Before any workload is constructed, the launcher requires four nonempty distinct allocation UUIDs; all NCCL ranks collectively prove exact node-major rank/local-rank/device-index mapping to that same set. Missing, duplicate, or unexpected UUID mappings fail the execution.

Adoption is evaluated on six explicit dataset/seed pairs and never by pooling SHD with SSC. For each dataset, the 5% threshold is the arithmetic mean of its three paired seed-level runtime reductions. At least two of three reductions must be positive separately for SHD and SSC, and every individual dataset/seed reduction must be at least -2%. The event-structure dataset means must also be no worse than the corresponding example-count means; the remaining equivalence, overhead, prediction, checkpoint, and permitted-information gates are conjunctive.

## Status

No scheduling scientific execution or result is created by the implementation work. Adoption remains undecided until exactly 18 Roihu executions pass completion, measurement, equivalence, official-test isolation, provenance validation, and every performance condition.
