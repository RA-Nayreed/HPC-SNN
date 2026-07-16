# HPC-SNN

Federated adaptive Parameter-free Attention for spiking neural networks.

The centralized SHD and SSC evaluation is complete. It covers official event files, deterministic 10 ms and 140-channel preprocessing, two-layer LIF models, an SHD DCLS reference, equation PfA, and pinned public-code-behaviour PfA. All 18 required executions completed, and the generated [centralized summary](results/centralized/centralized_summary.md) records the three-seed evidence.

## Centralized evaluation matrix

| Experiment | Dataset and model | Attention | Protocol |
|---|---|---|---|
| shd_lif_independent_evaluation | SHD, 256/256 LIF | none | independent_evaluation |
| shd_dcls_published_protocol | SHD, 256/256 DCLS | none | published_protocol |
| shd_pfa_equation_independent_evaluation | SHD, 256/256 LIF | equation PfA | independent_evaluation |
| shd_pfa_public_published_protocol | SHD, 256/256 LIF | public-code behaviour | published_protocol |
| ssc_lif_128_independent_evaluation | SSC, 128/128 LIF | none | independent_evaluation |
| ssc_pfa_equation_128_independent_evaluation | SSC, 128/128 LIF | equation PfA | independent_evaluation |

Every experiment uses seeds 7, 17, and 27, giving 18 independent one-GPU tasks. The SSC 512-neuron model is outside the current evaluation scope. DCLS and PfA are not combined because no verified source in the repository specifies that architecture.

The protocols are not statistically interchangeable. For SHD, `independent_evaluation` creates a deterministic stratified validation split from the official training set and accesses the official test set only after checkpoint selection. For SSC, it uses the official validation split for selection and accesses the official test set afterward. The SHD `published_protocol` reproduces source evaluation behaviour that monitors the official test split during training; those results are labelled as reproduction measurements and are not unbiased held-out estimates.

## Centralized results

Official-test accuracy is reported as mean ± sample standard deviation across seeds 7, 17, and 27:

| Experiment | Protocol | Official-test accuracy |
|---|---|---:|
| SHD LIF | `independent_evaluation` | 76.3693% ± 2.26431% |
| SHD DCLS | `published_protocol` | 91.3722% ± 0.243267% |
| SHD equation PfA | `independent_evaluation` | 78.4305% ± 1.91990% |
| SHD public-behaviour PfA | `published_protocol` | 84.0253% ± 0.0919464% |
| SSC LIF-128 | `independent_evaluation` | 54.8981% ± 0.198346% |
| SSC equation PfA-128 | `independent_evaluation` | 55.7812% ± 0.570910% |

Every scientific status is `not_claimed`: the executions completed, but no verified literature target and tolerance are available for a reproduction decision. This status is neither a reproduction pass nor an execution failure.

The independent-evaluation centralized results provide the reference measurements for subsequent federated experiments. Published-protocol measurements remain separate because they monitor the official SHD test split during training.

## Local setup and checks

Install CPU PyTorch from the official PyTorch index, then install the project:

~~~bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
python -m pip install -e ".[dev]"
ruff check src tests
python -m pytest -q
~~~

Raw datasets and run outputs remain excluded from Git. Automated tests use synthetic data and do not download SHD or SSC.

## Centralized execution on Roihu

~~~bash
bash scripts/slurm/submit_roihu_centralized.sh \
  --work-dir "/scratch/$CSC_PROJECT/$USER/hpc-snn" \
  --max-parallel 4
~~~

Monitor the returned job ID with:

~~~bash
squeue --job <JOB_ID> --array -o "%.18i %.9P %.28j %.2t %.10M %.10l %R"
~~~

After all 18 tasks complete:

~~~bash
fedapfa-summarize-centralized \
  --manifest experiments/centralized/manifest.yaml \
  --runs-root "/scratch/$CSC_PROJECT/$USER/hpc-snn/runs/centralized" \
  --output-dir "/scratch/$CSC_PROJECT/$USER/hpc-snn/results/centralized"
~~~

Execution completion requires a valid terminal condition, finite metrics, both checkpoints, complete-dataset use without batch limits, official test evaluation, the expected model class, and nonempty metrics and logs. Scientific reproduction is separate: it requires a verified reference accuracy and tolerance. Current literature targets remain null because no value is verifiable from repository records, so completed runs report `scientific_status: not_claimed` rather than a reproduction pass.

## Federated SHD LIF reference

The federated SHD LIF evaluation is complete. It uses SHD so that FedAvg can be assessed against the centralized SHD LIF reference without also changing the dataset or model family. The ordinary 256/256 LIF network is evaluated before attention mechanisms; FedAvg is reference infrastructure, not a novelty claim. SSC remains necessary for cross-dataset generalization and resource evidence but is outside this two-treatment matrix.

Accuracy is mean ± sample standard deviation across seeds 7, 17, and 27:

| Participation | Selected / total clients | Best validation accuracy | Official-test accuracy | Logical communication | Seeds |
|---:|---:|---:|---:|---:|---:|
| 50% | 10 / 20 | 88.6846% ± 0.935984% | 69.9647% ± 2.99540% | 856,224,000 bytes | 7, 17, 27 |
| 25% | 5 / 20 | 84.3137% ± 1.76743% | 67.0936% ± 1.21527% | 428,112,000 bytes | 7, 17, 27 |

The [generated federated summary](results/federated/federated_summary.md) provides the aggregation, and the [federated scientific record](thesis_records/federated_baseline.md) documents the methods, evidence, interpretation, and limitations.

For each seed, the established stratified SHD validation split is removed before client partitioning. The remaining training indices are assigned exactly once by deterministic label-wise Dirichlet sampling. Both participation treatments reuse the same split, partition, initial global parameters, and round-specific client permutation; the five-client selection is the prefix of the ten-client selection. Validation selects the global checkpoint, and the official test dataset is constructed only after all 100 communication rounds.

FedAvg uses sample-count weighting, `w_next = sum(n_k * w_k) / sum(n_k)`. Each selected client receives an isolated global-model copy and a newly created Adam optimizer. Optimizer state stays local and is neither retained nor aggregated. Logical communication counts one tensor-model download and upload per selected client; it excludes optimizer state, dataset transfer, checkpoint I/O, and telemetry and is not measured network traffic.

Within this protocol, selecting ten rather than five clients increased mean official-test accuracy by 2.87102 percentage points and used exactly twice the logical communication. Both federated means remained below the centralized SHD LIF independent-evaluation mean of 76.3693%, by 6.4046 and 9.2756 percentage points respectively. These are descriptive three-seed comparisons, not significance or causality claims.

All six executions passed completion checks. Their scientific status is `not_claimed` because no verified published FedAvg target is configured; this is expected and is not an execution failure or a reproduction claim. Federated and centralized measurements remain visibly separate even where their independent-evaluation rules permit contextual comparison.
