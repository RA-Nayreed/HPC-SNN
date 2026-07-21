#!/usr/bin/env bash
#SBATCH --job-name=snn-scheduling
#SBATCH --partition=gpumedium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:gh200:4
#SBATCH --cpus-per-task=288
#SBATCH --time=36:00:00

set -euo pipefail
: "${CSC_PROJECT:?CSC_PROJECT must be set}"
: "${USER:?USER must be set}"
: "${WORK_DIR:?WORK_DIR must be exported}"
: "${REPO_ROOT:?REPO_ROOT must be exported}"
: "${EVALUATION_MANIFEST:?EVALUATION_MANIFEST must be exported}"
: "${SLURM_ARRAY_TASK_ID:?Slurm array task is required}"
module purge
module load python-pytorch/2.10
venv="${FEDAPFA_VENV:-/projappl/${CSC_PROJECT}/${USER}/hpc-snn-venv}"
python_bin="${venv}/bin/python3"
[[ -x "${python_bin}" ]] || { echo "project Python interpreter is unavailable" >&2; exit 2; }
cd "${REPO_ROOT}"
"${python_bin}" -c 'import fedapfa, torch; assert torch.distributed.is_nccl_available()'
"${python_bin}" -m fedapfa.cli.scientific_manifest validate --manifest "${EVALUATION_MANIFEST}"
allocation="$("${python_bin}" -m fedapfa.cli.scientific_manifest allocation-task \
    --manifest "${EVALUATION_MANIFEST}" --index "${SLURM_ARRAY_TASK_ID}")"
IFS=$'\t' read -r dataset seed treatment_1 config_1 treatment_2 config_2 treatment_3 config_3 <<< "${allocation}"
case "${dataset}" in
    shd) data_root="${WORK_DIR}/data/shd"; [[ -f "${data_root}/shd_train.h5" && -f "${data_root}/shd_test.h5" ]] ;;
    ssc) data_root="${WORK_DIR}/data/ssc"; [[ -f "${data_root}/ssc_train.h5" && -f "${data_root}/ssc_valid.h5" && -f "${data_root}/ssc_test.h5" ]] ;;
    *) echo "unsupported evaluation dataset" >&2; exit 2 ;;
esac || { echo "local ${dataset^^} data are unavailable" >&2; exit 2; }
visible="$(${python_bin} -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.device_count())')"
[[ "${visible}" == 4 ]] || { echo "scheduling allocation requires four visible CUDA devices" >&2; exit 2; }
mapfile -t allocated_gpu_uuids < <(nvidia-smi --query-gpu=uuid --format=csv,noheader,nounits)
[[ "${#allocated_gpu_uuids[@]}" == 4 ]] || { echo "scheduling allocation must expose exactly four GPU UUIDs" >&2; exit 2; }
for uuid in "${allocated_gpu_uuids[@]}"; do
    [[ -n "${uuid//[[:space:]]/}" ]] || { echo "scheduling allocation contains an empty GPU UUID" >&2; exit 2; }
done
[[ "$(printf '%s\n' "${allocated_gpu_uuids[@]}" | sort -u | wc -l)" == 4 ]] || {
    echo "scheduling allocation GPU UUIDs must be nonempty and distinct" >&2
    exit 2
}
export FEDAPFA_ALLOCATED_GPU_UUIDS="$(IFS=,; printf '%s' "${allocated_gpu_uuids[*]}")"
unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY CUDA_MPS_ACTIVE_THREAD_PERCENTAGE
export OMP_NUM_THREADS=72 NCCL_ASYNC_ERROR_HANDLING=1 NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export FEDAPFA_ALLOCATION_EXECUTION_ORDER="${treatment_1},${treatment_2},${treatment_3}"
treatments=("${treatment_1}" "${treatment_2}" "${treatment_3}")
configs=("${config_1}" "${config_2}" "${config_3}")
for position in 0 1 2; do
    treatment="${treatments[position]}"
    config="${configs[position]}"
    export FEDAPFA_TREATMENT_POSITION="$((position + 1))"
    telemetry="${WORK_DIR}/telemetry/scheduling_evaluation/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${treatment}.csv"
    export FEDAPFA_GPU_TELEMETRY="${telemetry}"
    mkdir -p "$(dirname "${telemetry}")"
    nvidia-smi --query-gpu=timestamp,index,uuid,name,utilization.gpu,utilization.memory --format=csv,nounits > "${telemetry}"
    (while true; do nvidia-smi --query-gpu=timestamp,index,uuid,name,utilization.gpu,utilization.memory --format=csv,noheader,nounits >> "${telemetry}" || true; sleep 2; done) &
    telemetry_pid=$!
    stop_telemetry() { kill "${telemetry_pid}" 2>/dev/null || true; wait "${telemetry_pid}" 2>/dev/null || true; }
    trap stop_telemetry EXIT INT TERM
    echo "allocation_task=${SLURM_ARRAY_TASK_ID} dataset=${dataset} seed=${seed} treatment=${treatment} position=$((position + 1))"
    srun --kill-on-bad-exit=1 "${python_bin}" -m torch.distributed.run --standalone \
        --nnodes=1 --nproc-per-node=4 -m fedapfa.cli.train_system_evaluation "${config}" \
        --seed "${seed}" --data-root "${data_root}" \
        --output-root "${WORK_DIR}/runs/scheduling_evaluation" --resume-auto
    stop_telemetry
    trap - EXIT INT TERM
done
