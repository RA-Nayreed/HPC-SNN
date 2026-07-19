#!/usr/bin/env bash
#SBATCH --job-name=fedavg-resource
#SBATCH --partition=gpumedium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=72
#SBATCH --mem=217086M
#SBATCH --time=24:00:00

set -euo pipefail

: "${CSC_PROJECT:?CSC_PROJECT must be set}"
: "${USER:?USER must be set}"
: "${WORK_DIR:?WORK_DIR must be exported by the submission wrapper}"
: "${REPO_ROOT:?REPO_ROOT must be exported by the submission wrapper}"
: "${RESOURCE_MANIFEST:?RESOURCE_MANIFEST must be exported by the submission wrapper}"
: "${SLURM_JOB_ID:?this script must run inside a Slurm allocation}"

WORK_DIR="$(realpath -m "${WORK_DIR}")"
REPO_ROOT="$(realpath -m "${REPO_ROOT}")"
scratch_prefix="/scratch/${CSC_PROJECT}/${USER}/"
case "${WORK_DIR}/" in
    "${scratch_prefix}"*) ;;
    *) echo "WORK_DIR must be under ${scratch_prefix}" >&2; exit 2 ;;
esac
[[ -d "${REPO_ROOT}/.git" ]] || { echo "REPO_ROOT is not a Git checkout" >&2; exit 2; }
[[ -f "${RESOURCE_MANIFEST}" ]] || { echo "resource manifest is unavailable" >&2; exit 2; }

for variable in CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY CUDA_MPS_ACTIVE_THREAD_PERCENTAGE \
    FEDAPFA_MPS_ARCHIVE FEDAPFA_MPS_ACTIVE_THREAD_PERCENTAGE; do
    [[ -z "${!variable:-}" ]] || { echo "MPS setting ${variable} is incompatible" >&2; exit 2; }
done

module purge
module load python-pytorch/2.10
venv="${FEDAPFA_VENV:-/projappl/${CSC_PROJECT}/${USER}/hpc-snn-venv}"
[[ -f "${venv}/bin/activate" ]] || { echo "project environment is unavailable" >&2; exit 2; }
source "${venv}/bin/activate"
python_bin="${venv}/bin/python3"
[[ -x "${python_bin}" ]] || { echo "project Python interpreter is not executable" >&2; exit 2; }
cd "${REPO_ROOT}"

"${python_bin}" - <<'PY'
import sys

import fedapfa
import pynvml
import torch

pynvml.nvmlInit()
try:
    print(f"python_executable={sys.executable}")
    print(f"python_version={sys.version.split()[0]}")
    print(f"torch_version={torch.__version__}")
    print(f"torch_location={torch.__file__}")
    print(f"fedapfa_location={fedapfa.__file__}")
    print(f"cuda_build={torch.version.cuda}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"nvml_driver={pynvml.nvmlSystemGetDriverVersion()}")
    print(f"nvml_device_count={pynvml.nvmlDeviceGetCount()}")
finally:
    pynvml.nvmlShutdown()
PY

visible_devices="$("${python_bin}" - <<'PY'
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
)"
[[ "${visible_devices}" == "1" ]] || { echo "exactly one visible CUDA device is required" >&2; exit 2; }
gpu_uuid="$(nvidia-smi --query-gpu=uuid --format=csv,noheader | sed -n '1p')"
[[ "${gpu_uuid}" == GPU-* ]] || { echo "GPU UUID could not be resolved" >&2; exit 2; }
export FEDAPFA_GPU_UUID="${gpu_uuid}"

[[ -f "${WORK_DIR}/data/shd/shd_train.h5" && -f "${WORK_DIR}/data/shd/shd_test.h5" ]] || {
    echo "SHD files are unavailable; this job never downloads data" >&2; exit 2;
}
[[ -f "${WORK_DIR}/data/ssc/ssc_train.h5" && -f "${WORK_DIR}/data/ssc/ssc_valid.h5" \
    && -f "${WORK_DIR}/data/ssc/ssc_test.h5" ]] || {
    echo "SSC files are unavailable; this job never downloads data" >&2; exit 2;
}

"${python_bin}" -m fedapfa.cli.scientific_manifest validate --manifest "${RESOURCE_MANIFEST}"
task_count="$("${python_bin}" -m fedapfa.cli.scientific_manifest count --manifest "${RESOURCE_MANIFEST}")"
[[ "${task_count}" == "6" ]] || { echo "resource manifest must contain six tasks" >&2; exit 2; }

runs_root="${WORK_DIR}/runs/resource_measurement"
result_root="${WORK_DIR}/results/resource_measurement"
calibration_root="${result_root}/calibration/${SLURM_JOB_ID}"
provenance_root="${runs_root}/allocation-provenance"
mkdir -p "${runs_root}" "${result_root}" "${calibration_root}" "${provenance_root}"
printf '%s\n' "${SLURM_JOB_ID}" > "${provenance_root}/slurm-allocation-id.txt"
git rev-parse HEAD > "${provenance_root}/execution-commit.txt"
printf '%s\n' "${gpu_uuid}" > "${provenance_root}/gpu-uuid.txt"

calibration_config="${REPO_ROOT}/experiments/resource_measurement/shd/lif_client_resource.yaml"
"${python_bin}" -m fedapfa.cli.calibrate_resource_measurement "${calibration_config}" \
    --data-root "${WORK_DIR}/data/shd" --result-root "${calibration_root}" --repetitions 10
calibration_artifact="${calibration_root}/instrumentation_calibration.json"
"${python_bin}" -c 'import json,sys; value=json.load(open(sys.argv[1], encoding="utf-8")); raise SystemExit(0 if value.get("passed") else 2)' \
    "${calibration_artifact}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_ASYNC_ERROR_HANDLING=1

terminate_job() {
    status=$?
    trap - EXIT
    exit "${status}"
}
trap terminate_job EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

for ((index = 0; index < task_count; index++)); do
    task="$("${python_bin}" -m fedapfa.cli.scientific_manifest task \
        --manifest "${RESOURCE_MANIFEST}" --index "${index}")"
    IFS=$'\t' read -r config seed dataset mode protocol experiment participation device_count \
        client_processes_per_device process_count control_backend cuda_process_service configured_output_root <<< "${task}"
    [[ "${mode}" == "scientific_evaluation" && "${device_count}" == "1" \
        && "${client_processes_per_device}" == "1" && "${process_count}" == "1" \
        && "${control_backend}" == "nccl" && "${cuda_process_service}" == "none" ]] || {
        echo "resource task topology is incompatible" >&2; exit 2;
    }
    case "${dataset}" in
        shd) data_root="${WORK_DIR}/data/shd" ;;
        ssc) data_root="${WORK_DIR}/data/ssc" ;;
        *) echo "resource task dataset is incompatible" >&2; exit 2 ;;
    esac
    echo "resource_task=${index} experiment=${experiment} seed=${seed} dataset=${dataset} gpu_uuid=${gpu_uuid}"
    srun "${python_bin}" -m torch.distributed.run \
        --standalone --nnodes=1 --nproc-per-node=1 \
        -m fedapfa.cli.train_resource_measurement "${config}" \
        --seed "${seed}" --data-root "${data_root}" --output-root "${runs_root}" \
        --calibration "${calibration_artifact}" --resume-auto
done
