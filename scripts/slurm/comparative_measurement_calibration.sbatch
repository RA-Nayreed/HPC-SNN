#!/usr/bin/env bash
#SBATCH --job-name=snn-energy-calibration
#SBATCH --partition=gpumedium
#SBATCH --time=04:00:00

set -euo pipefail
: "${CSC_PROJECT:?CSC_PROJECT must be set}"
: "${USER:?USER must be set}"
: "${WORK_DIR:?WORK_DIR must be exported}"
: "${REPO_ROOT:?REPO_ROOT must be exported}"
: "${SCALING_TOPOLOGY:?SCALING_TOPOLOGY must be exported}"
: "${CALIBRATION_CONFIG:?CALIBRATION_CONFIG must be exported}"

module purge
module load python-pytorch/2.10
venv="${FEDAPFA_VENV:-/projappl/${CSC_PROJECT}/${USER}/hpc-snn-venv}"
python_bin="${venv}/bin/python3"
[[ -x "${python_bin}" ]] || { echo "project Python interpreter is unavailable" >&2; exit 2; }
export EVALUATION_PYTHON="${python_bin}"
cd "${REPO_ROOT}"
"${python_bin}" -c 'import fedapfa, torch; assert torch.cuda.is_available(); assert torch.version.cuda; assert torch.distributed.is_nccl_available()'
"${python_bin}" -m fedapfa.cli.validate_run "${CALIBRATION_CONFIG}"
[[ -f "${WORK_DIR}/data/shd/shd_train.h5" && -f "${WORK_DIR}/data/shd/shd_test.h5" ]] || {
    echo "local SHD data are unavailable" >&2
    exit 2
}

case "${SCALING_TOPOLOGY}" in
    one_node_one_gpu) expected_nodes=1; expected_per_node=1; expected_gpus=1; expected_processes=1 ;;
    one_node_two_gpu) expected_nodes=1; expected_per_node=2; expected_gpus=2; expected_processes=2 ;;
    one_node_four_gpu) expected_nodes=1; expected_per_node=4; expected_gpus=4; expected_processes=4 ;;
    two_nodes_four_gpus) expected_nodes=2; expected_per_node=2; expected_gpus=4; expected_processes=4 ;;
    *) echo "unsupported calibration topology" >&2; exit 2 ;;
esac
[[ "${SLURM_JOB_NUM_NODES}" == "${expected_nodes}" && "${SLURM_NTASKS}" == "${expected_nodes}" ]] || {
    echo "calibration allocation layout differs" >&2
    exit 2
}
[[ -d "${TMPDIR}" && -w "${TMPDIR}" ]] || { echo "TMPDIR is not writable" >&2; exit 2; }
mapfile -t allocated_nodes < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
[[ "${#allocated_nodes[@]}" == "${expected_nodes}" ]] || exit 2
uuid_dir="${WORK_DIR}/telemetry/system_scaling_energy_evaluation/calibration-uuid/${SLURM_JOB_ID}_${SCALING_TOPOLOGY}"
mkdir -p "${uuid_dir}"
export FEDAPFA_GPU_UUID_PREFLIGHT_DIR="${uuid_dir}"
export EXPECTED_GPUS_PER_NODE="${expected_per_node}"
srun --nodes="${expected_nodes}" --ntasks="${expected_nodes}" --ntasks-per-node=1 --kill-on-bad-exit=1 bash -c '
    set -euo pipefail
    mapfile -t raw_uuids < <(nvidia-smi --query-gpu=uuid --format=csv,noheader,nounits)
    [[ "${#raw_uuids[@]}" == "${EXPECTED_GPUS_PER_NODE}" ]] || exit 2
    raw_text="$(IFS=,; printf "%s" "${raw_uuids[*]}")"
    canonical="$(${EVALUATION_PYTHON} -c "
import sys
from fedapfa.distributed.process_context import allocated_gpu_uuids
print(\",\".join(allocated_gpu_uuids(sys.argv[1], expected_count=int(sys.argv[2]))))
" "${raw_text}" "${EXPECTED_GPUS_PER_NODE}")"
    IFS=, read -r -a canonical_values <<< "${canonical}"
    printf "%s\n" "${raw_uuids[@]}" > "${FEDAPFA_GPU_UUID_PREFLIGHT_DIR}/${SLURM_PROCID}.raw.txt"
    printf "%s\n" "${canonical_values[@]}" > "${FEDAPFA_GPU_UUID_PREFLIGHT_DIR}/${SLURM_PROCID}.txt"
'
raw_files=()
canonical_files=()
for ((node=0; node<expected_nodes; node++)); do
    raw_files+=("${uuid_dir}/${node}.raw.txt")
    canonical_files+=("${uuid_dir}/${node}.txt")
done
mapfile -t raw_uuids < <(sed -n 'p' "${raw_files[@]}")
mapfile -t canonical_uuids < <(sed -n 'p' "${canonical_files[@]}")
[[ "${#raw_uuids[@]}" == "${expected_gpus}" && "${#canonical_uuids[@]}" == "${expected_gpus}" ]] || exit 2
export FEDAPFA_ALLOCATED_GPU_UUIDS_RAW="$(IFS=,; printf '%s' "${raw_uuids[*]}")"
export FEDAPFA_ALLOCATED_GPU_UUIDS="$(IFS=,; printf '%s' "${canonical_uuids[*]}")"
unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY CUDA_MPS_ACTIVE_THREAD_PERCENTAGE
unset NCCL_ASYNC_ERROR_HANDLING
export OMP_NUM_THREADS=72
export MKL_NUM_THREADS=72
export OPENBLAS_NUM_THREADS=72
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
result_root="${WORK_DIR}/calibration/system_scaling_energy_evaluation/${SCALING_TOPOLOGY}"

if [[ "${expected_nodes}" == 1 ]]; then
    srun --nodes=1 --ntasks=1 --ntasks-per-node=1 --kill-on-bad-exit=1 \
        "${python_bin}" -m torch.distributed.run --standalone --nnodes=1 \
        --nproc-per-node="${expected_processes}" -m fedapfa.cli.calibrate_comparative_measurement \
        "${CALIBRATION_CONFIG}" --seed 37 --data-root "${WORK_DIR}/data/shd" \
        --result-root "${result_root}" --repetitions 10
else
    export CALIBRATION_MASTER_ADDR="${allocated_nodes[0]}"
    export CALIBRATION_MASTER_PORT="$((16000 + SLURM_JOB_ID % 39000))"
    export CALIBRATION_RESULT_ROOT="${result_root}"
    srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --kill-on-bad-exit=1 bash -c '
        set -euo pipefail
        visible="$(${EVALUATION_PYTHON} -c "import torch; print(torch.cuda.device_count())")"
        [[ "${visible}" == 2 ]] || exit 2
        "${EVALUATION_PYTHON}" -m torch.distributed.run \
            --nnodes=2 --nproc-per-node=2 --node-rank="${SLURM_PROCID}" \
            --rdzv-backend=c10d --rdzv-endpoint="${CALIBRATION_MASTER_ADDR}:${CALIBRATION_MASTER_PORT}" \
            --rdzv-id="calibration_${SLURM_JOB_ID}" \
            -m fedapfa.cli.calibrate_comparative_measurement "${CALIBRATION_CONFIG}" \
            --seed 37 --data-root "${WORK_DIR}/data/shd" \
            --result-root "${CALIBRATION_RESULT_ROOT}" --repetitions 10
    '
fi
