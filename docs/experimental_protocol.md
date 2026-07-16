# Experimental protocol

## Centralized evaluation matrix

The canonical manifest is `experiments/centralized/manifest.yaml`. It contains six mandatory `scientific_evaluation` configurations crossed with seeds 7, 17, and 27, producing 18 independent single-GPU tasks:

| Experiment | Dataset and model | Attention | Protocol |
|---|---|---|---|
| shd_lif_independent_evaluation | SHD 256/256 LIF | none | independent_evaluation |
| shd_dcls_published_protocol | SHD 256/256 DCLS | none | published_protocol |
| shd_pfa_equation_independent_evaluation | SHD 256/256 LIF | equation | independent_evaluation |
| shd_pfa_public_published_protocol | SHD 256/256 LIF | public_behavior | published_protocol |
| ssc_lif_128_independent_evaluation | SSC 128/128 LIF | none | independent_evaluation |
| ssc_pfa_equation_128_independent_evaluation | SSC 128/128 LIF | equation | independent_evaluation |

Every canonical configuration uses CUDA, all examples required by its protocol, null train, validation, and test batch limits, eight data-loader workers, persistent workers, and an explicit early-stopping setting. The SSC 512-neuron model is outside the current evaluation scope.

## Split policies

For SHD, `independent_evaluation` creates a seed-specific stratified validation split from the official training file. Checkpoint selection uses only that derived validation split. The official SHD test file is constructed and evaluated only after checkpoint selection.

For SSC, `independent_evaluation` uses the official training and validation files for fitting and checkpoint selection. The official SSC test file is constructed and evaluated only after checkpoint selection.

For SHD, `published_protocol` reproduces source evaluation behaviour by using the official test file for model selection and evaluating the selected checkpoint on that file again. This monitoring is recorded explicitly in `acceptance.json`. It is a paper-compatible reproduction measurement, not an unbiased held-out estimate.

Results from `independent_evaluation` and `published_protocol` are never pooled. The centralized summarizer groups by manifest experiment and verifies each resolved protocol.

## PfA variants and model boundary

Equation PfA and pinned public-code behaviour are separate treatments until their variance-versus-squared-deviation discrepancy is resolved. DCLS and PfA are also separate: no verified source in the repository specifies a combined DCLS+PfA architecture, so `dcls_shd` rejects PfA.

## Terminal conditions

Independent evaluations permit validation early stopping with patience 20. Published-protocol configurations use fixed 150-epoch training with `early_stop_patience: null`. An evaluation is terminal only after every configured epoch or a documented valid early-stop condition. Interruption is not completion; `--resume-auto` continues from `checkpoints/last.pt`.

## Collected evidence

All 18 mandatory executions completed for the six experiments and seeds 7, 17, and 27. The generated [centralized summary](../results/centralized/centralized_summary.md) reports the aggregated measurements while preserving the protocol distinctions defined above. Observed results do not alter the split, selection, or acceptance rules in this methods document.

## Federated SHD LIF reference

SHD is the sole dataset in the FedAvg baseline because it has a validated centralized LIF comparison and permits federated correctness to be examined without changing the dataset, model, and aggregation algorithm together. SSC remains reserved for cross-dataset generalization and resource evaluation. The ordinary two-hidden-layer LIF model is used before any attention mechanism. DCLS, PfA, adaptive attention, hierarchical aggregation, multiple GPUs, and multiple nodes are excluded. This baseline is reference infrastructure and is not a novelty claim.

The manifest `experiments/federated_baselines/manifest.yaml` contains two treatments: 20 clients with label-wise Dirichlet alpha 0.5 and either 10 selected clients (50%) or 5 selected clients (25%) per round. Seeds 7, 17, and 27 produce six scientific executions. Each execution uses 100 rounds, one local epoch, batch size 32, Adam learning rate 0.001, zero weight decay, gradient clipping at 1, and no learning-rate scheduler or early stopping.

Both treatments use the official `shd_train.h5` and `shd_test.h5` files, 20 classes, 10 ms temporal integration, and deterministic reduction from 700 cochlear channels to 140 inputs. The model has two hidden LIF layers with 256 neurons each, tau 10.05, threshold 1, subtractive detached reset, an ATan surrogate with alpha 5, dropout 0.4, and no batch normalization. The global model starts from seed-specific random initialization rather than a centralized checkpoint.

### Split and partition isolation

The shared SHD `independent_evaluation` splitter creates the seed-specific stratified 10% validation split. Validation indices are removed before client partitioning. The Dirichlet procedure operates only on the remaining training indices, assigns every eligible index exactly once, rejects any construction with a client below 32 examples, and uses a finite deterministic resampling limit. Validation and official-test examples are never assigned to clients.

The partition artifact records exact client indices, client sizes, per-client class counts, size statistics, label entropy, construction attempts, dataset identity, split identity, and integrity checks. For a given seed, the two participation treatments must have identical split, partition, and model-initialization identities.

### Client selection and optimization

A separate selection generator produces one deterministic permutation of the 20 clients for each round. The 25% treatment uses the first five clients and the 50% treatment uses the first ten, making the lower-participation selection a subset of the higher-participation selection. Client data ordering and dropout use separate per-round, per-client seeds and cannot advance the selection generator.

Every selected client receives an isolated copy of the current global model. A new Adam optimizer is created for that selection; optimizer state is not retained across selections and is not aggregated. The global model begins from random initialization associated with the scientific seed and never from a centralized checkpoint. SNN state is reset between batches, clients, validation, and final test evaluation.

### Federated Averaging

For selected clients `k`, the server computes:

`w_next = sum(n_k * w_k) / sum(n_k)`

where `n_k` is the number of training examples owned by client `k`. Aggregation weights must be finite, nonnegative, and sum to one. Empty updates, incompatible state dictionaries, non-finite values, and unsafe non-floating state differences are rejected.

### Selection and official-test access

Global validation accuracy selects `checkpoints/best.pt`, while `checkpoints/last.pt` records every completed round for resumption. The official SHD test dataset is not constructed or accessed during communication rounds. It is constructed only after all 100 rounds and evaluated exactly once using the selected global checkpoint. The centralized SHD LIF accuracy is reported only as contextual evidence and does not determine completion or scientific status.

### Communication definition

Logical communication includes one global-model download and one client-model upload for every selected client. Bytes equal each communicated tensor's element count multiplied by its element size. Optimizer state, dataset transfer, checkpoint I/O, and telemetry are excluded. This value is a deterministic accounting quantity, not measured network traffic.

No compression, quantization, sparsification, or event-based communication method is part of this baseline. With the model and round count fixed, doubling selected clients doubles the logical communication count. Logical communication is neither physical network measurement nor energy consumption.

### Completion rules

Completion requires all 100 rounds, valid partition integrity, 20 clients, the configured number of distinct selected clients in each round, finite records, valid aggregation weights, both nonempty checkpoints, nonempty client and round logs, complete and nonduplicated assignment of eligible training indices, validation and test isolation, exactly one official-test evaluation after model selection, Git and configuration provenance, scientific identities, and consistent logical communication accounting. Accuracy is not a completion condition. A missing verified reproduction target yields `scientific_status: not_claimed`, not execution failure.

### Collected federated evidence

All six mandatory scientific executions completed and passed aggregation validation for both treatments and seeds 7, 17, and 27. The [federated summary](../results/federated/federated_summary.md) records the measurements, and the [federated scientific record](../thesis_records/federated_baseline.md) separates evidence, interpretation, and limitations. Observed results do not alter the partition, selection, test-isolation, or acceptance rules in this methods document.
