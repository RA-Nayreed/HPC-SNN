# HPC-SNN

Federated adaptive Parameter-free Attention for spiking neural networks.

The canonical centralized evaluation covers SHD and SSC with official event files, deterministic 10 ms and 140-channel preprocessing, two-layer LIF models, an SHD DCLS reference, equation PfA, and pinned public-code-behaviour PfA. Execution records include strict acceptance, interruption-safe resumption, Git provenance, and three-seed aggregation.

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
