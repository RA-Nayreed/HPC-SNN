# Roihu centralized evaluation

The verified environment is project_2015302 on aarch64 with Python 3.12.12 and PyTorch 2.10.0+cu130 built for CUDA 13.0. Each reserved NVIDIA GH200 couples a 72-core Grace CPU with 120 GiB of CPU memory and a Hopper GPU with approximately 96 GiB of HBM. Roihu exposes up to 217 GiB of combined allocatable memory associated with one reserved GH200; this must not be interpreted as 120 GiB of GPU HBM. See CSC's [Roihu system description](https://docs.csc.fi/computing/systems-roihu/) and [partition limits](https://docs.csc.fi/computing/running/batch-job-partitions/).

CUDA forward and backward execution, the DCLS 0.1.1 CPU probe, official SHD and SSC validation, and the repository checks succeeded. More importantly, all 18 centralized array tasks completed successfully on GH200 GPUs, including all three DCLS executions. The aggregated evidence is available in the [centralized summary](../../results/centralized/centralized_summary.md).

## Resource accounting

Roihu charges GPU jobs according to the number of reserved GPUs. The 72 CPU cores and associated memory included with each GPU reservation are not billed separately. CSC currently assigns 200 billing units per GPU-hour on Roihu GPU partitions; consult the official [CSC billing documentation](https://docs.csc.fi/computing/hpc-billing/) for the applicable accounting rules.

The centralized array used 6.7775 allocated GPU-hours, approximately 1,355.5 GPU billing units. Scheduler accounting measures reserved execution resources. It is not an energy measurement and does not provide historical GPU utilization, GPU-memory telemetry, or GPU energy consumption.

## Installation

The array script loads `python-pytorch/2.10` before activating:

    /projappl/$CSC_PROJECT/$USER/hpc-snn-venv

It imports `DCLS` with uppercase spelling and confirms CUDA availability before training. Data must exist under `$WORK_DIR/data/shd` and `$WORK_DIR/data/ssc`. Runs and logs are restricted to `$WORK_DIR/runs/centralized` and `$WORK_DIR/slurm-logs/centralized`.

## Submission

The launcher reserves one GH200 and 72 CPU cores per task on `gpumedium` for at most 36 hours and assigns one configuration and seed to each array task. Default array concurrency is four.

Submit all 18 tasks:

~~~bash
bash scripts/slurm/submit_roihu_centralized.sh \
  --work-dir "/scratch/$CSC_PROJECT/$USER/hpc-snn" \
  --max-parallel 4
~~~

The wrapper validates that `WORK_DIR` is below `/scratch/$CSC_PROJECT/`, validates the centralized manifest, rejects reduced-sample, sweep, or memorization-validation entries, and prints the submitted job ID.

## Monitoring and resumption

Monitor the returned ID:

~~~bash
squeue --job <JOB_ID> --array -o "%.18i %.9P %.28j %.2t %.10M %.10l %R"
~~~

Compatible interrupted tasks can be resubmitted with the same submission command: `--resume-auto` skips completed runs and resumes incomplete runs from `checkpoints/last.pt`.

## Aggregation

~~~bash
fedapfa-summarize-centralized \
  --manifest experiments/centralized/manifest.yaml \
  --runs-root "/scratch/$CSC_PROJECT/$USER/hpc-snn/runs/centralized" \
  --output-dir "/scratch/$CSC_PROJECT/$USER/hpc-snn/results/centralized"
~~~

The summary command exits nonzero for missing, duplicate, invalid, or incomplete mandatory runs. Null literature targets are reported as `not_claimed` and do not make aggregation fail. The SSC 512-neuron model is outside the current evaluation scope.
