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

## Single-node distributed FedAvg execution

The completed [physical-device collection](../../results/distributed_evaluation/distributed_evaluation_summary.md) used arrays `250928`, `250929`, and `250946`; the completed [one-GPU capacity collection](../../results/device_capacity_evaluation/distributed_evaluation_summary.md) used array `250950`. All 33 tasks were `COMPLETED` with exit code `0:0` at execution commit `4491078710d01d22e9517e62c8de0f3554c4b3f2`.

### Allocation layouts

The authoritative comparisons used one process per physical GH200 and NCCL:

| Physical GH200 GPUs | Distributed processes | CPU cores | Slurm memory | Recorded `gres` |
|---:|---:|---:|---:|---|
| 1 | 1 | 72 | 217086M | `gpu:gh200=1` |
| 2 | 2 | 144 | 434172M | `gpu:gh200=2` |
| 4 | 4 | 288 | 868344M | `gpu:gh200=4` |

The capacity treatment reserved one physical GH200 and 72 CPU cores, then ran one, two, or four client processes. The one-process row used NCCL without a CUDA process service. The two- and four-process rows used Gloo for CPU control and detached state movement plus CUDA MPS for GPU work. Those rows measure one-GPU process capacity and remain separate from authoritative scientific-accuracy comparisons.

The launcher requires Roihu's writable job-local `TMPDIR` for MPS. It creates `CUDA_MPS_PIPE_DIRECTORY` and `CUDA_MPS_LOG_DIRECTORY` below `$TMPDIR/fedapfa_mps/<array_task>`, sets the active-thread percentage to integer `100 / client_processes_per_device`, verifies the MPS daemon, and archives logs under `$WORK_DIR/mps-logs/device_capacity_evaluation`. MPS pipes and daemon logs must not be placed in a shared repository or scratch path.

### Python and distributed launcher

The project interpreter is mandatory:

    /projappl/$CSC_PROJECT/$USER/hpc-snn-venv/bin/python3

The batch script loads `python-pytorch/2.10`, activates the project environment, verifies the locations of PyTorch and `fedapfa`, and launches:

    "$FEDAPFA_VENV/bin/python3" -m torch.distributed.run

Calling the module-provided `torchrun` executable directly is incompatible with this project environment because that executable's interpreter path belongs to the module installation. Activating the project environment does not rewrite its interpreter path, so workers are not guaranteed to import the project installation and its dependency set. Invoking `torch.distributed.run` through the project Python keeps the launcher and workers on the required interpreter.

### Scheduler association limit

The execution campaign observed a 40-GPU association limit. When the association already held that aggregate number of GH200 resources, further jobs remained pending with reason `AssocGrpGRES`. This reason means that the Slurm association group's generic-resource allocation limit is currently reached; it is not a task failure. A pending request has not begun allocated execution and is not execution billing. The scheduler can release it when association usage falls below the limit.

### Resource terms

- A **physical GPU** is one allocated GH200 accelerator recorded as `gres/gpu:gh200`; it is not the same as a distributed process.
- A **process** is a PyTorch distributed rank. Exclusive runs use one rank per GPU, whereas capacity runs place two or four ranks on one GPU through MPS.
- **CPU cores** are Grace CPU cores attached to the allocation. The records contain 72 cores per physical GH200; process count does not change that allocation.
- **GPU HBM** is accelerator memory, approximately 96 GiB visible on these GH200 GPUs. It is distinct from per-process allocated tensors and reserved CUDA memory.
- **Grace memory** is CPU memory associated with the Grace component, described by Roihu as 120 GiB per reserved GH200.
- **Slurm memory** is the scheduler allocation shown above. The one-GPU value of 217086M is not GPU HBM and must not be reported as such.

Allocated tensor peaks, reserved CUDA peaks, device HBM, Grace CPU memory, and Slurm memory answer different questions. Likewise, round runtime, Slurm elapsed execution time, pending time, allocated GPU-hours, utilization, and energy are separate quantities. The committed accounting includes execution time only; the two-second utilization telemetry is not a direct energy measurement.

Exact submission, monitoring, accounting transformation, summary generation, and evidence-verification commands are in the [reproducibility guide](../../docs/reproducibility.md#completed-distributed-execution-reproduction). Scientific interpretation and limitations are in the [distributed record](../../thesis_records/distributed_execution.md).

## Client resource measurement allocation

The resource campaign uses one gpumedium allocation for all six tasks, run sequentially when uninterrupted:

| Resource | Allocation or process layout |
|---|---|
| Node | 1 |
| Physical GH200 | 1 |
| Distributed processes | 1 |
| Client processes per GPU | 1 |
| CPU cores | 72 |
| Slurm memory | 217086M |
| Control backend | NCCL |
| CUDA MPS | disabled |
| Time limit | 24 hours |

This is the authoritative one-process-per-GPU path. It is distinct from the earlier one-GPU capacity treatment that used two or four MPS client processes with Gloo control. MPS settings are rejected for resource measurement, and no capacity row can enter the client-cost table.

The runner loads python-pytorch/2.10, activates /projappl/$CSC_PROJECT/$USER/hpc-snn-venv, and sets python_bin to that environment's exact bin/python3. It verifies imports for PyTorch, fedapfa, and the maintained pynvml binding and prints Python, PyTorch, project, CUDA, driver, and NVML information. The environment setup installs nvidia-ml-py, whose import name is pynvml; imports remain safe on login hosts where the NVML shared library is unavailable, but requested measurement fails clearly on a compute node without NVML.

When distributed initialization is needed, the runner uses:

    "$python_bin" -m torch.distributed.run --nproc-per-node=1

It never invokes the module-provided torchrun executable. That executable is tied to the module installation's interpreter path, whereas the project and its NVML dependency are installed in the project environment. Direct use can therefore start workers outside the required environment.

NVML resolution uses the UUID of the allocated visible GPU. CUDA index zero is not assumed to equal NVML physical index zero. Exactly one visible CUDA device and one resolved UUID are required. This collection does not use CUDA MPS, so MPS pipe and log variables must be absent; TMPDIR MPS directories apply only to separately identified capacity measurements.

The observed Roihu association limit remains 40 GH200 resources. AssocGrpGRES means the association group's generic-resource limit is occupied; it is a pending reason rather than execution failure. Pending allocations have not begun allocated execution and are not execution billing.

Resource terms remain distinct:

- one physical GH200 is an accelerator allocation, not a process count;
- the distributed process is the single PyTorch rank;
- 72 CPU cores are Grace CPU resources attached to the reservation;
- GPU HBM is accelerator memory and differs from allocated and reserved CUDA memory;
- Grace memory is CPU memory associated with the Grace component;
- 217086M is Slurm memory and is not GPU HBM;
- NVML watts and integrated joules are device measurements;
- Slurm elapsed time and allocated GPU-hours are accounting quantities;
- pending time is outside allocated execution time.

Utilization alone does not establish energy efficiency, and allocated GPU-hours cannot substitute for device energy. The two 30-second idle intervals establish an attempt-specific power reference; they are device measurements rather than scheduler accounting.

Submit and monitor with:

~~~bash
bash scripts/slurm/submit_roihu_resource_measurement.sh \
  --work-dir "/scratch/$CSC_PROJECT/$USER/hpc-snn"
squeue --job <JOB_ID> \
  -o "%.18i %.9P %.28j %.12T %.10M %.10l %R"
~~~

The launcher verifies existing SHD and SSC files and never downloads data. It runs a passing training-only SHD calibration before iterating the exact six-task manifest. Exact accounting, fitting, summary, and verification commands are in the [reproducibility guide](../../docs/reproducibility.md#client-resource-measurement-and-cost-estimation). Scientific status and limitations are in the [resource record](../../thesis_records/resource_measurement.md); no resource result is available until Roihu records and aggregation are committed.

## CIFAR-10 Fed-SNN execution and evidence

The six-task execution completed successfully. The [Slurm accounting record](../../results/fedsnn_paper_evaluation/provenance/slurm-accounting.txt) contains tasks `236880_0` through `236880_5`, all `COMPLETED` with exit code `0:0`. Each task used one GH200, and every scientific execution reached round 100. CIFAR-10 was already present at `$WORK_DIR/data/cifar10`; neither launcher nor trainer downloaded it.

| Distribution | Seed accuracies | Mean ± sample SD | Paper reference | Mean signed difference |
|---|---|---:|---:|---:|
| IID | 81.50%, 82.16%, 81.55% | 81.7367% ± 0.3675 pp | 76.44% | +5.2967 pp |
| Label-Dirichlet non-IID, alpha 0.5 | 72.01%, 75.80%, 73.32% | 73.7100% ± 1.9249 pp | 73.94% | -0.2300 pp |

The [committed summary](../../results/fedsnn_paper_evaluation/published_fedsnn_summary.md) is the active operational reference. Its scientific status is `equivalence_not_established`. The IID-to-non-IID mean reduction is 8.0267 percentage points; this is descriptive and is not a statistical-significance or causal claim.

Re-run the federated matrix only when an independent execution is required:

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
