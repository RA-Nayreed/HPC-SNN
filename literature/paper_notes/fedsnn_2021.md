# Venkatesha et al. (2021) Fed-SNN source audit

Primary paper: Y. Venkatesha, Y. Kim, L. Tassiulas, and P. Panda, “Federated Learning with Spiking Neural Networks,” *IEEE Transactions on Signal Processing* 69 (2021), DOI 10.1109/TSP.2021.3121632, [arXiv:2106.06579 rendered as HTML](https://ar5iv.labs.arxiv.org/html/2106.06579).

Authors’ repository: [Intelligent-Computing-Lab-Panda/FedSNN](https://github.com/Intelligent-Computing-Lab-Panda/FedSNN). The comparison is pinned to commit [`1ab26154b064119850bc9f84c55304b5b45f7df6`](https://github.com/Intelligent-Computing-Lab-Panda/FedSNN/tree/1ab26154b064119850bc9f84c55304b5b45f7df6), dated 6 October 2021. Branch movement after this audit cannot change the cited source.

Inspected at that commit:

- `README.md`
- `models/vgg_spiking_bntt.py`
- `main_fed.py`
- `models/Update.py`
- `models/Fed.py`
- `models/test.py`
- `test_cifar10.sh`
- `utils/sampling.py`
- `utils/options.py`

The eleven-page arXiv paper and every source file above were inspected on 18 July 2026. Repository source, paper statements, and project interpretations are kept distinct below.

## Corrected source behavior

The released CIFAR-10 transform is `ToTensor()` followed by channel mean and standard deviation 0.5. For a unit-interval tensor `x`, the project implements the equivalent signed representation

`x_signed = 2x - 1`, with `x_signed` in `[-1, 1]`.

The released spike generator is represented as

`s_t = 1(2U_t <= |x_signed|) sign(x_signed)`, where `U_t` is uniform on `[0, 1)`.

The project creates all temporal random values with a required explicit `torch.Generator`. The signed encoder is exclusive to the corrected CIFAR-10 S-VGG9 path; SHD and SSC encoding are unchanged.

The released S-VGG9 has seven convolutional layers with channels `[64, 64, 128, 128, 256, 256, 256]`, average pooling after convolutions 2, 4, and 7, a 1024-unit first linear layer, and a ten-class output layer. Convolution and linear biases are absent. Every convolution and linear weight is initialized once with Xavier uniform gain 2. Each BNTT layer has timestep-specific scale, no additive bias, momentum 0.1, and epsilon `1e-4`; its scale starts at one.

The hidden membrane rule is

`u_t = 0.95 u_(t-1) + BNTT_t(W s_t)`, followed by threshold-one spiking and subtractive reset.

The output layer accumulates voltage without leak and returns the temporal mean

`z = (1 / T) sum_t W_out s_t`, with `T = 20` from the released command.

The division is part of the model forward calculation, so cross-entropy gradients pass through the temporal mean.

The custom `TemporalBatchNorm` implementation was compared numerically with PyTorch `BatchNorm2d` and `BatchNorm1d` configured with fixed momentum 0.1, epsilon `1e-4`, affine scale, and no bias. Training outputs, stored running means and variances, and evaluation outputs were exactly equal in the CPU comparison. Its timestep scales are independent, so no structural replacement was needed.

Local source training uses batch size 32, shuffling, and dropped incomplete batches. Source `Fed.py` averages every selected client state with weight `1/P`; the corrected configuration therefore uses exact uniform weights. This includes floating BNTT running means and variances. SHD retains example-count weighting.

The source non-IID partition draws a Dirichlet vector separately for each class, masks clients whose current size is not below `N/clients`, normalizes the masked proportions, splits by cumulative integer boundaries, and retries until every client has at least ten examples. The project records this as `fedsnn_balanced_label_dirichlet`, separate from the existing unbalanced `label_dirichlet` implementation. The released IID index assignment is recorded separately as `fedsnn_random_iid`.

## Paper and released-source conflicts

| Setting | Paper | Released program | Corrected project treatment |
|---|---|---|---|
| Active Table I rows | Reports CIFAR-10 SNN 10/2 IID 76.44% and non-IID 73.94% | Example command is IID only | Two active treatment identities; example command is not active |
| Local epochs | 5 | Example command uses 2 | 5 in both active Table I treatments |
| Repetitions | 3 | Example command expresses one run | Seeds 7, 17, and 27 |
| Distribution | Reports IID and alpha-0.5 non-IID rows | Example command passes `--iid` | IID and source-balanced alpha-0.5 treatments |
| Timesteps | Not stated | Command uses 20 | 20, classified as released-source evidence |
| Aggregation | Equation 1 weights by client example count | `Fed.py` uses equal selected-client averaging | Uniform, classified as released-source evidence; conflict disclosed |
| Momentum | 0.95 | Command omits `--momentum` and inherits source default 0.9 | 0.95, classified as paper evidence |
| Weight decay | SNN value not stated | Command does not override source default `1e-4` | `1e-4`, classified as source-derived and not paper-explicit |
| Learning-rate boundaries | Reduction factor 5 after 40, 60, and 80 | Command inherits fractional boundaries 0.33 and 0.66, resolving to 33 and 66 for 100 rounds | 40, 60, and 80, classified as paper evidence |
| Training collection | Uses all 50,000 standard training examples | Uses the standard training collection | All 50,000 indices are partitioned; no internal validation split |
| Evaluation data | Calls the standard 10,000-image test split validation and reports after round 100 | Evaluates that split during training and at termination | Official test is never monitored during training and is evaluated once after round 100 |
| Model selection | Final reported validation follows round 100; no held-validation selection rule is documented | No separate held-validation checkpoint rule | Explicit `final_round`; `last.pt` is selected without calling it best |
| Local work | Algorithm describes client work, and the implementation trains all clients for model-deviation diagnostics | Only selected states enter aggregation | Project trains selected clients only; diagnostic-only unselected work is omitted |

The effective configured learning rate is 0.1 through round 40, 0.02 for rounds 41–60, 0.004 for rounds 61–80, and 0.0008 from round 81. This matches the source’s “reduce after the boundary has completed” placement when the paper boundaries are supplied.

## Corrected experiment identities

`cifar10_fedsnn_paper_reported_iid_evaluation` represents the Table I IID row. It uses ten clients, two selected clients, 100 rounds, five local epochs, even IID assignment with nearly equal class proportions, batch size 32, SGD learning rate 0.1, momentum 0.95, weight decay `1e-4`, 20 timesteps, uniform aggregation, and the corrected source representation.

`cifar10_fedsnn_paper_reported_noniid_evaluation` differs only in identity and distribution-specific fields: it uses the source-compatible balanced label-Dirichlet partition with alpha 0.5 and minimum client size ten. The model, optimizer, local work, rounds, schedule, aggregation, and evaluation policy are otherwise identical.

Both treatments partition every one of the 50,000 standard training indices exactly once and use zero internal validation examples. No empty validation loader is constructed and no per-round validation runs. The final-round checkpoint is selected after round 100, then the complete official 10,000-example test collection is evaluated exactly once. That collection corresponds to what the paper calls validation. This project’s one-time access is stricter than released-source monitoring, so the result cannot be called procedure-for-procedure equivalent.

Seeds 7, 17, and 27 produce exactly six corrected federated tasks. Interrupted executions resume from compatible `last.pt` checkpoints. A best-validation checkpoint is not required or created, and unavailable validation metrics remain null.

The released `test_cifar10.sh` example remains useful source evidence: it requests IID, two local epochs, and 20 timesteps, while leaving source defaults such as momentum 0.9 and weight decay `1e-4` unmodified. It is not a Table I reproduction configuration and is not present in an active collection.

## Scientific status and retained prior evidence

Table I reports 76.44% for the CIFAR-10 SNN 10/2 IID setting and 73.94% for its non-IID setting. Both are stored only as descriptive references. Neither has a tolerance or can trigger acceptance. Reports calculate signed and absolute percentage-point differences while retaining scientific status `equivalence_not_established`.

The committed [corrected summary](../../results/fedsnn_paper_evaluation/published_fedsnn_summary.md) records all six executions as complete:

| Distribution | Seed 7 | Seed 17 | Seed 27 | Mean ± sample SD | Paper reference | Mean signed difference |
|---|---:|---:|---:|---:|---:|---:|
| IID | 81.50% | 82.16% | 81.55% | 81.7367% ± 0.3675 pp | 76.44% | +5.2967 pp |
| Label-Dirichlet non-IID, alpha 0.5 | 72.01% | 75.80% | 73.32% | 73.7100% ± 1.9249 pp | 73.94% | -0.2300 pp |

Mean macro-F1 is 81.7070% for IID and 73.5136% for non-IID, closely following accuracy and providing no evidence of class collapse. The corrected implementation therefore learns successfully. The non-IID mean closely agrees with the paper’s descriptive value, whereas the IID mean is stable across the three seeds but exceeds its reference by about 5.30 percentage points. The descriptive IID-to-non-IID reduction is 8.0267 percentage points.

This is strong protocol-aligned validation, not established implementation equivalence or an exact reproduction pass. Three seeds do not support a statistical-significance claim. The evidence also does not support novelty, causality, or energy-efficiency claims, and the paper/source discrepancies above remain unresolved.

The prior independent implementation completed computationally and produced 18.23–26.79% official-test accuracy. It was scientifically incompatible because it used unit-interval inputs, crop and flip augmentation, unsigned Poisson encoding, a summed readout, default initialization, 25 timesteps, BNTT epsilon `1e-5`, retained incomplete local batches, unbalanced Dirichlet sampling, and example-count aggregation. Its configuration remains under `experiments/published_fedsnn/history`, and its generated runs remain under `runs/published_fedsnn` without modification.

The corrected `fedsnn_paper_evaluation` evidence is the active Fed-SNN reference. The unsuccessful prior evidence remains byte-for-byte preserved, is not pooled or averaged with the corrected runs, and must not be presented as the active baseline. Corrected experiment names and the `runs/fedsnn_paper_evaluation` root prevent automatic resumption from loading prior unsuccessful checkpoints or the superseded, unexecuted released-command identity. Checkpoint compatibility also hashes the resolved aggregation and selection policies and verifies them explicitly.
