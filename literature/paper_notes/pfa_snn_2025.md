# PfA-SNN 2025 publication-protocol note

## Evidence available in this repository

The repository literature notes, references, search records, and reproduction-fetch scripts contain no bibliographic record or verifiable published SHD or SSC accuracy for these configurations. The canonical configurations therefore set `reference_test_accuracy` and `absolute_tolerance` to null. Their scientific status must be `not_claimed` even after execution completion.

The public-code-behaviour implementation is pinned to upstream commit `0898fc22480c86bccd7f6fccb0d43fdfbd579797`. That provenance identifies the behaviour being tested but does not establish a published accuracy target.

## Equation and code discrepancy

For current x, neuron-wise mean m, deviation d = x - m, variance v = mean(d²), threshold θ, and stabilizer λ:

- Equation PfA uses weight θd / (d² + 2λ + 2v).
- The pinned public behaviour uses weight θd / (3d² + 2λ), because its code substitutes element-wise squared deviation where the equation uses neuron-wise variance.

Both then use bias `(θ - weight(x + m)) / 2`, sigmoid attention, and multiplication of current by attention. Both are parameter-free and differentiable, but they are not equivalent. The centralized evaluation therefore tests equation PfA under `independent_evaluation` and public behaviour under the explicitly labelled SHD `published_protocol`.

## Architecture boundary

No repository literature record verifies that the source method requires a combined DCLS+PfA architecture. The evaluation does not infer one by weakening validation: DCLS and PfA remain separate models, and `dcls_shd` continues to reject attention variants.

## Protocol boundary

The SHD `published_protocol` exposes the official test file during model selection. It is useful only as a labelled paper-compatible reproduction behaviour and cannot be combined with SHD or SSC `independent_evaluation` results, whose official test files are evaluated only after model selection.

A verified source may populate a reference accuracy and tolerance. Until such evidence is recorded, null is the only defensible value and no reproduction pass may be claimed.
