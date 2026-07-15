# Roihu centralized evaluation

The verified environment is project_2015302, aarch64, Python 3.12.12, PyTorch 2.10.0+cu130 with a CUDA 13.0 build, and NVIDIA GH200 120GB. CUDA forward and backward execution, the DCLS 0.1.1 CPU probe, 55 non-CUDA tests, Ruff, official SHD and SSC validation, and the reduced-sample SHD LIF CUDA diagnostic have passed. Complete DCLS CUDA training is not yet claimed.

The array script loads `python-pytorch/2.10` before activating:

    /projappl/$CSC_PROJECT/$USER/hpc-snn-venv

It imports `DCLS` with uppercase spelling, confirms CUDA, reserves one GH200 and 72 CPU cores per task on `gpumedium` for at most 36 hours, and launches one configuration and seed per task. Default array concurrency is four. Data must exist under `$WORK_DIR/data/shd` and `$WORK_DIR/data/ssc`. Runs and logs are restricted to `$WORK_DIR/runs/centralized` and `$WORK_DIR/slurm-logs/centralized`.

Submit all 18 tasks:

~~~bash
bash scripts/slurm/submit_roihu_centralized.sh \
  --work-dir "/scratch/$CSC_PROJECT/$USER/hpc-snn" \
  --max-parallel 4
~~~

The wrapper validates that `WORK_DIR` is below `/scratch/$CSC_PROJECT/`, validates the centralized manifest, rejects reduced-sample, sweep, or memorization-validation entries, and prints the submitted job ID.

Monitor the returned ID:

~~~bash
squeue --job <JOB_ID> --array -o "%.18i %.9P %.28j %.2t %.10M %.10l %R"
~~~

Compatible interrupted tasks can be resubmitted with the same command: `--resume-auto` skips completed runs and resumes incomplete runs from `checkpoints/last.pt`.

After all tasks complete:

~~~bash
fedapfa-summarize-centralized \
  --manifest experiments/centralized/manifest.yaml \
  --runs-root "/scratch/$CSC_PROJECT/$USER/hpc-snn/runs/centralized" \
  --output-dir "/scratch/$CSC_PROJECT/$USER/hpc-snn/results/centralized"
~~~

The summary command exits nonzero for missing, duplicate, invalid, or incomplete mandatory runs. Null literature targets are reported as `not_claimed` and do not make aggregation fail. The SSC 512-neuron model is outside the current evaluation scope.
