# Roihu centralized evaluation

The verified environment is project_2015302 on aarch64 with Python 3.12.12 and PyTorch 2.10.0+cu130 built for CUDA 13.0. Each reserved NVIDIA GH200 couples a 72-core Grace CPU with 120 GiB of CPU memory and a Hopper GPU with approximately 96 GiB of HBM. Roihu exposes up to 217 GiB of combined allocatable memory associated with one reserved GH200; this must not be interpreted as 120 GiB of GPU HBM. See CSC's [Roihu system description](https://docs.csc.fi/computing/systems-roihu/) and [partition limits](https://docs.csc.fi/computing/running/batch-job-partitions/).

CUDA forward and backward execution, the DCLS 0.1.1 CPU probe, official SHD and SSC validation, and the repository checks succeeded. More importantly, all 18 centralized array tasks completed on GH200 GPUs with exit code `0:0`, including all three DCLS executions. The aggregated evidence is available in the [centralized summary](../../results/centralized/centralized_summary.md).

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

## Federated SHD LIF execution

The FedAvg launcher uses `gpumedium`, one GH200, one Slurm task, and 72 CPU cores for each independent configuration/seed pair. The canonical manifest expands to six tasks: two participation treatments crossed with seeds 7, 17, and 27. Default array concurrency is one and may be set from one through six after resource review.

Data are read from `$WORK_DIR/data/shd`. Federated runs, generated summaries, job-level telemetry, and Slurm logs are written under `$WORK_DIR/runs/federated`, `$WORK_DIR/results/federated`, `$WORK_DIR/telemetry/federated`, and `$WORK_DIR/slurm-logs/federated`. The launcher does not download data and exits if CUDA or either official SHD file is unavailable.

Submit after human review:

~~~bash
bash scripts/slurm/submit_roihu_federated.sh \
  --work-dir "/scratch/$CSC_PROJECT/$USER/hpc-snn" \
  --max-parallel 1
~~~

Monitor the returned job ID:

~~~bash
squeue --job <JOB_ID> --array -o "%.18i %.9P %.28j %.2t %.10M %.10l %R"
~~~

Compatible interrupted executions may be submitted with the same command. The array invokes `--resume-auto`, which skips completed compatible records and resumes compatible records from `checkpoints/last.pt`.

Aggregate after all six tasks complete:

~~~bash
fedapfa-summarize-federated \
  --manifest experiments/federated_baselines/manifest.yaml \
  --runs-root "/scratch/$CSC_PROJECT/$USER/hpc-snn/runs/federated" \
  --output-dir "/scratch/$CSC_PROJECT/$USER/hpc-snn/results/federated"
~~~

### Job-level GPU telemetry

The array tests each requested `nvidia-smi` field, records unsupported fields explicitly, and samples supported timestamp, identity, utilization, memory, power, and temperature fields every two seconds. It records the sampling command and interval, starts collection before training, and stops the background process on success, failure, or signal while preserving collected CSV data.

This telemetry describes the allocated GPU job. It does not isolate individual client work, does not measure communication traffic, and must not be labelled per-client, neural-model, or neuromorphic energy. No federated GPU telemetry or scientific accuracy evidence has yet been collected.
