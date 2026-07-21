#!/usr/bin/env bash
#SBATCH --job-name=snn-hierarchy
#SBATCH --partition=gpumedium
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:gh200:2
#SBATCH --cpus-per-task=144
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
IFS=$'\t' read -r dataset seed treatment_1 config_1 treatment_2 config_2 <<< "${allocation}"
case "${dataset}" in
    shd) data_root="${WORK_DIR}/data/shd"; [[ -f "${data_root}/shd_train.h5" && -f "${data_root}/shd_test.h5" ]] ;;
    ssc) data_root="${WORK_DIR}/data/ssc"; [[ -f "${data_root}/ssc_train.h5" && -f "${data_root}/ssc_valid.h5" && -f "${data_root}/ssc_test.h5" ]] ;;
    *) echo "unsupported evaluation dataset" >&2; exit 2 ;;
esac || { echo "local ${dataset^^} data are unavailable" >&2; exit 2; }
mapfile -t allocated_nodes < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
[[ "${#allocated_nodes[@]}" == 2 ]] || { echo "hierarchical allocation requires exactly two nodes" >&2; exit 2; }
master_addr="${allocated_nodes[0]}"
master_port="$((15000 + SLURM_JOB_ID % 40000))"
gpu_uuid_preflight_dir="${WORK_DIR}/telemetry/hierarchical_reduction_evaluation/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_gpu_uuid_preflight"
mkdir -p "${gpu_uuid_preflight_dir}"
export FEDAPFA_GPU_UUID_PREFLIGHT_DIR="${gpu_uuid_preflight_dir}"
srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --kill-on-bad-exit=1 bash -c '
    set -euo pipefail
    [[ "${SLURM_NTASKS}" == 2 && "${SLURM_PROCID}" =~ ^[01]$ ]] || exit 2
    mapfile -t node_gpu_uuids < <(nvidia-smi --query-gpu=uuid --format=csv,noheader,nounits)
    [[ "${#node_gpu_uuids[@]}" == 2 ]] || { echo "each hierarchy node must expose exactly two GPU UUIDs" >&2; exit 2; }
    for uuid in "${node_gpu_uuids[@]}"; do
        [[ -n "${uuid//[[:space:]]/}" ]] || { echo "hierarchy node contains an empty GPU UUID" >&2; exit 2; }
    done
    [[ "$(printf "%s\n" "${node_gpu_uuids[@]}" | sort -u | wc -l)" == 2 ]] || {
        echo "hierarchy node GPU UUIDs must be nonempty and distinct" >&2
        exit 2
    }
    printf "%s\n" "${node_gpu_uuids[@]}" > "${FEDAPFA_GPU_UUID_PREFLIGHT_DIR}/${SLURM_PROCID}.txt"
'
[[ -f "${gpu_uuid_preflight_dir}/0.txt" && -f "${gpu_uuid_preflight_dir}/1.txt" ]] || {
    echo "hierarchy GPU UUID preflight is incomplete" >&2
    exit 2
}
mapfile -t allocated_gpu_uuids < <(sort -u "${gpu_uuid_preflight_dir}/0.txt" "${gpu_uuid_preflight_dir}/1.txt")
[[ "${#allocated_gpu_uuids[@]}" == 4 ]] || {
    echo "hierarchy allocation must contain exactly four distinct GPU UUIDs" >&2
    exit 2
}
export FEDAPFA_ALLOCATED_GPU_UUIDS="$(IFS=,; printf '%s' "${allocated_gpu_uuids[*]}")"
unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY CUDA_MPS_ACTIVE_THREAD_PERCENTAGE
export OMP_NUM_THREADS=72 NCCL_ASYNC_ERROR_HANDLING=1 NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export FEDAPFA_ALLOCATION_EXECUTION_ORDER="${treatment_1},${treatment_2}"
treatments=("${treatment_1}" "${treatment_2}")
configs=("${config_1}" "${config_2}")
for position in 0 1; do
    export FEDAPFA_TREATMENT_POSITION="$((position + 1))"
    export EVALUATION_TREATMENT="${treatments[position]}"
    export EVALUATION_SEED="${seed}"
    export EVALUATION_CONFIG="${configs[position]}"
    export EVALUATION_DATA_ROOT="${data_root}"
    export EVALUATION_OUTPUT_ROOT="${WORK_DIR}/runs/hierarchical_reduction_evaluation"
    export EVALUATION_MASTER_ADDR="${master_addr}"
    export EVALUATION_MASTER_PORT="${master_port}"
    export EVALUATION_PYTHON="${python_bin}"
    export FEDAPFA_GPU_TELEMETRY="${WORK_DIR}/telemetry/hierarchical_reduction_evaluation/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${EVALUATION_TREATMENT}.csv"
    mkdir -p "$(dirname "${FEDAPFA_GPU_TELEMETRY}")"
    : > "${FEDAPFA_GPU_TELEMETRY}"
    echo "allocation_task=${SLURM_ARRAY_TASK_ID} dataset=${dataset} seed=${seed} treatment=${EVALUATION_TREATMENT} position=$((position + 1)) rendezvous=${master_addr}:${master_port}"
    srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --kill-on-bad-exit=1 bash -c '
        set -euo pipefail
        [[ "${SLURM_NTASKS}" == 2 && "${SLURM_PROCID}" =~ ^[01]$ ]] || exit 2
        visible="$(${EVALUATION_PYTHON} -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.device_count())")"
        [[ "${visible}" == 2 ]] || { echo "each hierarchy node requires two visible CUDA devices" >&2; exit 2; }
        node_offset=$((SLURM_PROCID * 2))
        (while true; do
            nvidia-smi --query-gpu=timestamp,index,uuid,name,utilization.gpu,utilization.memory --format=csv,noheader,nounits |
                awk -F, -v offset="${node_offset}" "BEGIN{OFS=\",\"} {gsub(/ /,\"\",\$2); \$2=\$2+offset; print}"
            sleep 2
        done) >> "${FEDAPFA_GPU_TELEMETRY}" &
        telemetry_pid=$!
        stop_telemetry() { kill "${telemetry_pid}" 2>/dev/null || true; wait "${telemetry_pid}" 2>/dev/null || true; }
        trap stop_telemetry EXIT INT TERM
        "${EVALUATION_PYTHON}" -m torch.distributed.run \
            --nnodes=2 --nproc-per-node=2 --node-rank="${SLURM_PROCID}" \
            --rdzv-backend=c10d --rdzv-endpoint="${EVALUATION_MASTER_ADDR}:${EVALUATION_MASTER_PORT}" \
            --rdzv-id="${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${FEDAPFA_TREATMENT_POSITION}" \
            -m fedapfa.cli.train_system_evaluation "${EVALUATION_CONFIG}" \
            --seed "${EVALUATION_SEED}" --data-root "${EVALUATION_DATA_ROOT}" \
            --output-root "${EVALUATION_OUTPUT_ROOT}" --resume-auto
    '
done
