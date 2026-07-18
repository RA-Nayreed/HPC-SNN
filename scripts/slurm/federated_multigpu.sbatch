#!/usr/bin/env bash
#SBATCH --job-name=fedavg-distributed
#SBATCH --partition=gpumedium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=36:00:00

set -euo pipefail

: "${CSC_PROJECT:?CSC_PROJECT must be set}"
: "${USER:?USER must be set}"
: "${WORK_DIR:?WORK_DIR must be exported by the submission wrapper}"
: "${REPO_ROOT:?REPO_ROOT must be exported by the submission wrapper}"
: "${DISTRIBUTED_MANIFEST:?DISTRIBUTED_MANIFEST must be exported by the submission wrapper}"
: "${EXPECTED_DEVICE_COUNT:?EXPECTED_DEVICE_COUNT must be exported by the submission wrapper}"
: "${SLURM_ARRAY_TASK_ID:?this script must run as a Slurm array task}"

WORK_DIR="$(realpath -m "${WORK_DIR}")"
REPO_ROOT="$(realpath -m "${REPO_ROOT}")"
scratch_prefix="/scratch/${CSC_PROJECT}/${USER}/"
case "${WORK_DIR}/" in
    "${scratch_prefix}"*) ;;
    *) echo "WORK_DIR must be under ${scratch_prefix}" >&2; exit 2 ;;
esac
[[ -d "${REPO_ROOT}/.git" ]] || { echo "REPO_ROOT is not a Git checkout: ${REPO_ROOT}" >&2; exit 2; }
[[ -f "${DISTRIBUTED_MANIFEST}" ]] || { echo "missing manifest: ${DISTRIBUTED_MANIFEST}" >&2; exit 2; }
[[ "${EXPECTED_DEVICE_COUNT}" =~ ^(1|2|4)$ ]] || { echo "expected device count is incompatible" >&2; exit 2; }

module purge
module load python-pytorch/2.10
venv="${FEDAPFA_VENV:-/projappl/${CSC_PROJECT}/${USER}/hpc-snn-venv}"
[[ -f "${venv}/bin/activate" ]] || { echo "missing Roihu virtual environment: ${venv}" >&2; exit 2; }
python_bin="${venv}/bin/python3"
[[ -x "${python_bin}" ]] || { echo "project Python interpreter is not executable: ${python_bin}" >&2; exit 2; }
source "${venv}/bin/activate"
cd "${REPO_ROOT}"

"${python_bin}" - <<'PY'
import sys

import fedapfa
import torch

print(f"python_executable={sys.executable}")
print(f"torch_location={torch.__file__}")
print(f"fedapfa_location={fedapfa.__file__}")
print(f"torch_cuda_build={torch.version.cuda}")
print(f"cuda_visible={torch.cuda.is_available()}")
PY

"${python_bin}" -m fedapfa.cli.scientific_manifest validate --manifest "${DISTRIBUTED_MANIFEST}"
task="$("${python_bin}" -m fedapfa.cli.scientific_manifest task \
    --manifest "${DISTRIBUTED_MANIFEST}" --index "${SLURM_ARRAY_TASK_ID}")"
IFS=$'\t' read -r config seed dataset mode protocol experiment participation device_count \
    client_processes_per_device process_count control_backend cuda_process_service configured_output_root <<< "${task}"
[[ "${mode}" == "scientific_evaluation" ]] || { echo "distributed task has incompatible mode" >&2; exit 2; }
[[ "${device_count}" == "${EXPECTED_DEVICE_COUNT}" ]] || { echo "task device count differs from allocation group" >&2; exit 2; }
[[ "${process_count}" -le 10 ]] || { echo "task process count exceeds selected-client capacity" >&2; exit 2; }
[[ "$((device_count * client_processes_per_device))" == "${process_count}" ]] || {
    echo "task process topology is incompatible" >&2; exit 2;
}

allocated_devices="$("${python_bin}" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable; distributed execution requires GH200 GPUs")
print(torch.cuda.device_count())
PY
)"
[[ "${allocated_devices}" == "${device_count}" ]] || { echo "visible CUDA devices differ from allocation" >&2; exit 2; }

case "${dataset}" in
    shd)
        data_root="${WORK_DIR}/data/shd"
        [[ -f "${data_root}/shd_train.h5" && -f "${data_root}/shd_test.h5" ]] || {
            echo "SHD data files are unavailable" >&2; exit 2;
        }
        ;;
    ssc)
        data_root="${WORK_DIR}/data/ssc"
        [[ -f "${data_root}/ssc_train.h5" && -f "${data_root}/ssc_valid.h5" && -f "${data_root}/ssc_test.h5" ]] || {
            echo "SSC data files are unavailable" >&2; exit 2;
        }
        ;;
    cifar10)
        data_root="${WORK_DIR}/data/cifar10"
        [[ -f "${data_root}/cifar-10-batches-py/data_batch_1" ]] || {
            echo "CIFAR-10 training files are unavailable" >&2; exit 2;
        }
        ;;
    *) echo "distributed task dataset is incompatible" >&2; exit 2 ;;
esac

runs_root="${WORK_DIR}/${configured_output_root}"
telemetry_root="${WORK_DIR}/telemetry/${configured_output_root#runs/}"
mps_archive_root="${WORK_DIR}/mps-logs/${configured_output_root#runs/}"
mkdir -p "${runs_root}" "${telemetry_root}" "${mps_archive_root}"
job_label="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}_${SLURM_ARRAY_TASK_ID}"
telemetry_csv="${telemetry_root}/${job_label}.csv"
export FEDAPFA_GPU_TELEMETRY="${telemetry_csv}"
nvidia-smi --query-gpu=timestamp,index,uuid,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu \
    --format=csv,nounits > "${telemetry_csv}"
(
    while true; do
        nvidia-smi --query-gpu=timestamp,index,uuid,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu \
            --format=csv,noheader,nounits >> "${telemetry_csv}" || true
        sleep 2
    done
) &
telemetry_pid=$!
mps_started=0
mps_job_root=""
mps_archive="${mps_archive_root}/${job_label}"

stop_execution_services() {
    status=$?
    trap - EXIT
    if kill -0 "${telemetry_pid}" 2>/dev/null; then
        kill "${telemetry_pid}" 2>/dev/null || true
        wait "${telemetry_pid}" 2>/dev/null || true
    fi
    if [[ "${mps_started}" == "1" ]]; then
        printf 'quit\n' | nvidia-cuda-mps-control >/dev/null 2>&1 || true
        mkdir -p "${mps_archive}"
        cp -a "${CUDA_MPS_LOG_DIRECTORY}/." "${mps_archive}/" 2>/dev/null || true
    fi
    exit "${status}"
}
trap stop_execution_services EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [[ "${cuda_process_service}" == "mps" ]]; then
    [[ "${control_backend}" == "gloo" && "${client_processes_per_device}" -gt 1 ]] || {
        echo "MPS execution requires Gloo and multiple client processes per device" >&2; exit 2;
    }
    : "${TMPDIR:?MPS execution requires Roihu TMPDIR as the job-local directory}"
    [[ -d "${TMPDIR}" ]] || { echo "Roihu TMPDIR is not an existing directory: ${TMPDIR}" >&2; exit 2; }
    [[ -w "${TMPDIR}" ]] || { echo "Roihu TMPDIR is not writable: ${TMPDIR}" >&2; exit 2; }
    command -v nvidia-cuda-mps-control >/dev/null || { echo "CUDA MPS control command is unavailable" >&2; exit 2; }
    mps_job_root="${TMPDIR}/fedapfa_mps/${job_label}"
    export CUDA_MPS_PIPE_DIRECTORY="${mps_job_root}/pipe"
    export CUDA_MPS_LOG_DIRECTORY="${mps_job_root}/log"
    mkdir -p "${CUDA_MPS_PIPE_DIRECTORY}" "${CUDA_MPS_LOG_DIRECTORY}"
    export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="$((100 / client_processes_per_device))"
    nvidia-cuda-mps-control -d
    mps_started=1
    printf 'get_server_list\n' | nvidia-cuda-mps-control >/dev/null || {
        echo "CUDA MPS control daemon did not respond" >&2; exit 2;
    }
    export FEDAPFA_MPS_ARCHIVE="${mps_archive}"
    export FEDAPFA_MPS_ACTIVE_THREAD_PERCENTAGE="${CUDA_MPS_ACTIVE_THREAD_PERCENTAGE}"
elif [[ "${cuda_process_service}" != "none" || "${control_backend}" != "nccl" ]]; then
    echo "exclusive execution requires NCCL and CUDA process service none" >&2
    exit 2
fi

export OMP_NUM_THREADS="$((SLURM_CPUS_PER_TASK / process_count))"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_ASYNC_ERROR_HANDLING=1
echo "array=${SLURM_ARRAY_TASK_ID} experiment=${experiment} seed=${seed} dataset=${dataset} devices=${device_count} processes=${process_count} service=${cuda_process_service}"
srun "${python_bin}" -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc-per-node="${process_count}" \
    -m fedapfa.cli.train_federated_distributed "${config}" \
    --seed "${seed}" --data-root "${data_root}" --output-root "${runs_root}" --resume-auto
