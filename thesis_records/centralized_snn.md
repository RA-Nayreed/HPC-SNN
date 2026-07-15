# Centralized SNN evaluation

## Scientific objective

Establish comparable centralized SHD and SSC baselines for two-layer LIF networks, equation PfA, pinned public-code-behaviour PfA, and the published SHD DCLS procedure while preserving the distinction between unbiased held-out evaluation and publication-protocol reproduction.

## Methods

The canonical collection contains six experiments and three deterministic seeds, 7, 17, and 27, for exactly 18 independent GH200 tasks:

| Experiment | Dataset and model | PfA | Protocol |
|---|---|---|---|
| shd_lif_independent_evaluation | SHD 256/256 LIF | none | independent_evaluation |
| shd_dcls_published_protocol | SHD 256/256 DCLS | none | published_protocol |
| shd_pfa_equation_independent_evaluation | SHD 256/256 LIF | equation | independent_evaluation |
| shd_pfa_public_published_protocol | SHD 256/256 LIF | pinned public behaviour | published_protocol |
| ssc_lif_128_independent_evaluation | SSC 128/128 LIF | none | independent_evaluation |
| ssc_pfa_equation_128_independent_evaluation | SSC 128/128 LIF | equation | independent_evaluation |

The SSC 512-neuron model is outside the current evaluation scope. DCLS+PfA remains prohibited because no verified repository source specifies that combined architecture.

## Experiment protocol

For SHD, `independent_evaluation` creates a seed-specific stratified validation split from the official training file. Checkpoint selection uses only that split, and the official test file is accessed only after model selection. For SSC, the same protocol uses the official validation file for selection and accesses the official test file afterward.

For SHD, `published_protocol` intentionally monitors the official test file during training to reproduce source evaluation behaviour. Its metrics are never pooled with `independent_evaluation` metrics and are never presented as unbiased held-out estimates. Equation PfA and public-code behaviour remain separate because their denominator calculations differ.

Every canonical configuration uses all examples required by its protocol, null batch limits, CUDA, eight data-loader workers, deterministic seeds, and explicit early-stopping settings. Independent evaluations use validation patience 20; published-protocol configurations use fixed 150-epoch training with null patience.

Execution completion requires:

- every configured epoch or a documented valid early stop;
- finite losses, accuracies, and metric records;
- nonempty `best_validation.pt` and `last.pt` checkpoints;
- complete-dataset use without subset or batch limits;
- official test evaluation after model selection;
- the configured model class;
- nonempty `metrics.jsonl` and `training.log`;
- resolved configuration and Git provenance.

Scientific reproduction is evaluated separately from completion. A verified reference accuracy and tolerance are required for `passed` or `failed`. A null reference produces `not_claimed` and never produces a reproduction pass.

## Evidence collected

Verified Roihu facts are aarch64, Python 3.12.12, PyTorch 2.10.0+cu130 with a CUDA 13.0 build, NVIDIA GH200 120GB, successful CUDA forward and backward execution, and a successful DCLS 0.1.1 CPU probe. The official SHD and SSC files and counts have been validated. Automated tests cover configuration validation, manifest expansion, completion gates, resumption, aggregation, protocol separation, and launcher resources.

No results from the 18 canonical tasks have been recorded in this scientific record.

## Unresolved scientific limitations

Repository literature evidence does not provide a verifiable published accuracy for any canonical configuration, so every reference target remains null. The DCLS CUDA training outcome is unknown until canonical execution completes. The equation and public-code PfA discrepancy remains unresolved. The published SHD protocol monitors the official test set and therefore cannot provide an unbiased held-out estimate.

## Conclusions permitted by the evidence

The repository defines a reproducible centralized evaluation procedure with six distinct experiments, three seeds, explicit split policies, truthful completion gates, automatic compatible-run resumption, and protocol-preserving aggregation. No accuracy, reproduction, or DCLS CUDA training conclusion is permitted before canonical results are collected.

## Execution commands

Submit:

~~~bash
bash scripts/slurm/submit_roihu_centralized.sh \
  --work-dir "/scratch/$CSC_PROJECT/$USER/hpc-snn" \
  --max-parallel 4
~~~

Monitor:

~~~bash
squeue --job <JOB_ID> --array -o "%.18i %.9P %.28j %.2t %.10M %.10l %R"
~~~

Aggregate after every mandatory task reaches execution completion:

~~~bash
fedapfa-summarize-centralized \
  --manifest experiments/centralized/manifest.yaml \
  --runs-root "/scratch/$CSC_PROJECT/$USER/hpc-snn/runs/centralized" \
  --output-dir "/scratch/$CSC_PROJECT/$USER/hpc-snn/results/centralized"
~~~
