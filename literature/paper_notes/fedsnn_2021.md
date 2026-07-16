# Venkatesha et al. (2021) Fed-SNN protocol record

Primary source: S. Venkatesha, Y. Kim, L. Tassiulas, and P. Panda, “Federated Learning with Spiking Neural Networks,” *IEEE Transactions on Signal Processing* 69 (2021), DOI 10.1109/TSP.2021.3121632, [arXiv:2106.06579](https://arxiv.org/abs/2106.06579).

The complete eleven-page primary paper was inspected before implementation. The implementation is independent and is not described as an exact reproduction because several required model and preprocessing values are not reported.

## Explicit primary-paper settings

| Setting | Value | Source location | Classification |
|---|---|---|---|
| Dataset | CIFAR-10 | Section IV-A, page 5 | Explicit |
| Architecture family | VGG9 / S-VGG9 | Section IV-A, page 5 | Explicit |
| Architecture depth | Seven convolutional and two linear layers | Section IV-A, page 5 | Explicit |
| Pooling type | Average pooling | Section IV-A, page 5 | Explicit; exact locations are not stated |
| Input encoding | Poisson-distributed spike trains from image intensities | Section II-A, page 2, and Section IV-A, page 5 | Explicit |
| Temporal normalization | Batch Normalization Through Time (BNTT) | Section IV-A, page 5 | Explicit |
| Total clients | 10 | Section IV-A, page 5, and the 10/2 setting in Figure 5 / Table I, page 7 | Explicit |
| Participating clients | 2 | 10/2 setting in Figure 5 / Table I, page 7 | Explicit |
| Non-IID partition | Dirichlet with alpha 0.5 | Section IV-C and Figure 5, pages 6–7 | Explicit |
| Local epochs | 5 | Training setup in Section IV-A, page 5 | Explicit |
| Local batch size | 32 | Training setup in Section IV-A, page 5 | Explicit |
| Communication rounds | 100 | Training setup in Section IV-A, page 5 | Explicit |
| Optimizer | SGD | Training setup in Section IV-A, page 5 | Explicit |
| SNN learning rate | 0.1 | Training setup in Section IV-A, page 5 | Explicit |
| Momentum | 0.95 | Training setup in Section IV-A, page 5 | Explicit |
| Learning-rate reduction factor | 5 | Training setup in Section IV-A, page 5 | Explicit |
| Learning-rate reduction locations | 40, 60, and 80, described as epochs | Training setup in Section IV-A, page 5 | Explicit locations; communication-round use is an interpretation |
| Surrogate derivative | Triangular form proportional to `max(0, 1 - abs((u-v)/v))` | Equation (4), Section II-A, page 3 | Explicit form; scale is not numerical |
| Classification readout | Continuous output membrane accumulation with leak one | Section II-A, equations (1)–(3), pages 2–3 | Explicit |

Table I reports 73.94% for one non-IID CIFAR-10 SNN 10/2 entry. That number is not configured as a reference target because the paper does not report enough architecture, neuron, and preprocessing detail to establish that this implementation is precisely the same setting. `reference_test_accuracy` and tolerance therefore remain null, and scientific status remains `not_claimed`.

## Explicit implementation interpretations

| Required field | Configured value | Evidence and rationale | Classification |
|---|---|---|---|
| Learning-rate schedule axis | Apply reductions after communication rounds 40, 60, and 80 | The primary paper says “epochs,” while its federated procedure is indexed by communication round. The implementation records the round-boundary interpretation explicitly. | Interpretation |
| Timesteps | 25 | The primary paper does not state a value. The cited BNTT method reports 25 timesteps for CIFAR-10 in its experimental setup. | Assumption informed by cited method |
| Leak | 0.95 | Not reported numerically in the primary paper or the inspected BNTT method description. | Assumption |
| Firing threshold | 1.0 | The primary paper defines a threshold symbol but gives no numerical value. | Assumption |
| Surrogate scale | 0.3 | The primary paper leaves the scale symbolic. The cited BNTT method reports a damping factor of 0.3. | Assumption informed by cited method |
| Convolution channels | 64, 64, 128, 128, 256, 256, 256 | Exact channel widths are not reported. | Assumption |
| Pooling locations | After convolution layers 2, 4, and 7 | Average pooling is reported, but placement is not. | Assumption |
| First linear width | 1024 | The two-linear-layer depth is reported, but hidden width is not. | Assumption |
| Input normalization | Scale unsigned image values to [0, 1], without channel standardization | Input normalization is not reported; unit-interval intensities preserve the Poisson-rate interpretation. | Assumption |
| Data augmentation | Random 32×32 crop with padding 4 and horizontal flip probability 0.5 | The primary paper does not report augmentation. The cited BNTT method reports random crop and horizontal flip without numerical crop or flip parameters. | Assumption informed by cited method |
| BNTT affine form | Independent timestep-specific scale and running mean/variance, with no additive bias | The cited BNTT method explicitly describes timestep-specific parameters and removal of beta. | Cited-method interpretation |
| BNTT momentum and epsilon | 0.1 and 1e-5 | Not numerically reported. | Assumption |
| Weight decay | 0 | Not reported. | Assumption |
| Gradient clipping | Disabled | Not reported. | Assumption |

The cited BNTT source inspected for these interpretations is Y. Kim and P. Panda, “Revisiting Batch Normalization for Training Low-Latency Deep Spiking Neural Networks From Scratch,” *Frontiers in Neuroscience* 15 (2021), [DOI 10.3389/fnins.2021.773954](https://doi.org/10.3389/fnins.2021.773954).

## Implementation boundaries

- CIFAR-10 uses the standard torchvision training and test splits.
- The held validation split is selected deterministically from the standard training split.
- The official test split is constructed exactly once and only after validation-based model selection.
- Poisson random values use an explicit per-device generator.
- Every BNTT timestep has separate trainable scale and running statistics.
- The configuration records every interpretation above under `protocol_assumptions` where it affects the scientific execution.
- Completion and scientific status are independent; no target means `completed: true` can coexist only with `scientific_status: not_claimed`.
