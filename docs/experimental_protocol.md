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
