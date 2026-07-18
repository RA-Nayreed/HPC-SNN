# Roihu scientific execution

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

The federated baseline ran as Slurm array `189464` from Git commit `29ad1558dff52b856ee35b6ce2f538ec2006594a`. All six independent tasks completed on `gpumedium` with exit code `0:0`, using one GH200 reservation per task. The [federated summary](../../results/federated/federated_summary.md) contains the valid three-seed aggregation, and the [federated scientific record](../../thesis_records/federated_baseline.md) documents its interpretation and limitations.

The requested 36-hour limit was a scheduler constraint, not the actual runtime. The 72 CPU cores and memory attached to each GH200 reservation are scheduler resources and must not be reported as separate FedAvg algorithmic resources.

The canonical reproduction command is:

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

Aggregate compatible accepted records with:

~~~bash
fedapfa-summarize-federated \
  --manifest experiments/federated_baselines/manifest.yaml \
  --runs-root "/scratch/$CSC_PROJECT/$USER/hpc-snn/runs/federated" \
  --output-dir "/scratch/$CSC_PROJECT/$USER/hpc-snn/results/federated"
~~~

### Job-level GPU telemetry

The array tests each requested `nvidia-smi` field, records unsupported fields explicitly, and samples supported timestamp, identity, utilization, memory, power, and temperature fields every two seconds. It records the sampling command and interval, starts collection before training, and stops the background process on success, failure, or signal while preserving collected CSV data.

This telemetry describes the allocated GPU job. It does not isolate individual client work, does not measure communication traffic, and must not be labelled per-client, neural-model, or neuromorphic energy. The committed federated evidence does not report device-utilization or energy estimates. Logical communication is tensor accounting rather than physical network measurement. Successful GH200 execution therefore does not demonstrate low-energy SNN operation or FPGA-equivalent or neuromorphic energy efficiency.

## Corrected CIFAR-10 Fed-SNN execution and evidence

The corrected six-task execution completed successfully. The [Slurm accounting record](../../results/fedsnn_paper_evaluation/provenance/slurm-accounting.txt) contains tasks `236880_0` through `236880_5`, all `COMPLETED` with exit code `0:0`. Each task used one GH200, and every scientific execution reached round 100. CIFAR-10 was already present at `$WORK_DIR/data/cifar10`; neither launcher nor trainer downloaded it.

| Distribution | Seed accuracies | Mean ± sample SD | Paper reference | Mean signed difference |
|---|---|---:|---:|---:|
| IID | 81.50%, 82.16%, 81.55% | 81.7367% ± 0.3675 pp | 76.44% | +5.2967 pp |
| Label-Dirichlet non-IID, alpha 0.5 | 72.01%, 75.80%, 73.32% | 73.7100% ± 1.9249 pp | 73.94% | -0.2300 pp |

The [committed summary](../../results/fedsnn_paper_evaluation/published_fedsnn_summary.md) is the active operational reference. Its scientific status is `equivalence_not_established`. The IID-to-non-IID mean reduction is 8.0267 percentage points; this is descriptive and is not a statistical-significance or causal claim.

Re-run the corrected federated matrix only when an independent execution is required:

~~~bash
bash scripts/slurm/submit_roihu_published_fedsnn.sh \
  --work-dir "/scratch/$CSC_PROJECT/$USER/hpc-snn" \
  --max-parallel 1
~~~

Runs, results, telemetry, and Slurm output are isolated under `$WORK_DIR/{runs,results,telemetry,slurm-logs}/fedsnn_paper_evaluation`. Compatible interrupted tasks resume from their own `last.pt`; prior `runs/published_fedsnn`, `runs/fedsnn_corrected`, and superseded experiment identities are incompatible. The two active tasks use all 50,000 training examples, no internal validation loader, final-round selection, and one official-test evaluation after round 100.

Monitor and inspect accounting:

~~~bash
squeue --job <JOB_ID> --array -o "%.18i %.9P %.28j %.2t %.10M %.10l %R"
sacct -j <JOB_ID> --array \
  --format=JobID,State,ExitCode,Elapsed,Start,End,AllocTRES
~~~

Regenerate the summary from a complete compatible six-run collection with:

~~~bash
fedapfa-summarize-published-fedsnn \
  --manifest experiments/published_fedsnn/manifest.yaml \
  --runs-root "/scratch/$CSC_PROJECT/$USER/hpc-snn/runs/fedsnn_paper_evaluation" \
  --output-dir "/scratch/$CSC_PROJECT/$USER/hpc-snn/results/fedsnn_paper_evaluation"
~~~

The centralized learning verification remains available as a separate check from a one-GPU Roihu allocation after loading `python-pytorch/2.10` and activating the established environment:

~~~bash
python3 -m fedapfa.cli.train_centralized \
  experiments/published_fedsnn/cifar10/centralized_learning_verification.yaml \
  --data-root "/scratch/$CSC_PROJECT/$USER/hpc-snn/data/cifar10" \
  --output-root "/scratch/$CSC_PROJECT/$USER/hpc-snn/runs/fedsnn_centralized_verification" \
  --device cuda --resume-auto
~~~

The centralized configuration is separate from the completed six-task Fed-SNN evidence and must not be pooled with it.
