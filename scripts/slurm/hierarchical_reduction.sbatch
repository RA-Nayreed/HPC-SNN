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
export EVALUATION_PYTHON="${python_bin}"
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
    raw_node_gpu_uuids="$(IFS=,; printf "%s" "${node_gpu_uuids[*]}")"
    canonical_node_gpu_uuids="$("${EVALUATION_PYTHON}" -c "
import sys
from fedapfa.distributed.process_context import allocated_gpu_uuids
print(\",\".join(allocated_gpu_uuids(sys.argv[1], expected_count=2)))
" "${raw_node_gpu_uuids}")"
    IFS=, read -r -a canonical_values <<< "${canonical_node_gpu_uuids}"
    printf "%s\n" "${node_gpu_uuids[@]}" > "${FEDAPFA_GPU_UUID_PREFLIGHT_DIR}/${SLURM_PROCID}.raw.txt"
    printf "%s\n" "${canonical_values[@]}" > "${FEDAPFA_GPU_UUID_PREFLIGHT_DIR}/${SLURM_PROCID}.txt"
'
[[ -f "${gpu_uuid_preflight_dir}/0.txt" && -f "${gpu_uuid_preflight_dir}/1.txt" && \
   -f "${gpu_uuid_preflight_dir}/0.raw.txt" && -f "${gpu_uuid_preflight_dir}/1.raw.txt" ]] || {
    echo "hierarchy GPU UUID preflight is incomplete" >&2
    exit 2
}
mapfile -t allocated_gpu_uuids < <(sed -n 'p' "${gpu_uuid_preflight_dir}/0.txt" "${gpu_uuid_preflight_dir}/1.txt")
[[ "${#allocated_gpu_uuids[@]}" == 4 ]] || {
    echo "hierarchy allocation must contain exactly four distinct GPU UUIDs" >&2
    exit 2
}
mapfile -t raw_allocated_gpu_uuids < <(sed -n 'p' "${gpu_uuid_preflight_dir}/0.raw.txt" "${gpu_uuid_preflight_dir}/1.raw.txt")
export FEDAPFA_ALLOCATED_GPU_UUIDS_RAW="$(IFS=,; printf '%s' "${raw_allocated_gpu_uuids[*]}")"
canonical_allocation="$(IFS=,; printf '%s' "${allocated_gpu_uuids[*]}")"
export FEDAPFA_ALLOCATED_GPU_UUIDS="$("${python_bin}" -c '
import sys
from fedapfa.distributed.process_context import allocated_gpu_uuids
print(",".join(allocated_gpu_uuids(sys.argv[1], expected_count=4)))
' "${canonical_allocation}")"
unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY CUDA_MPS_ACTIVE_THREAD_PERCENTAGE
unset NCCL_ASYNC_ERROR_HANDLING
export OMP_NUM_THREADS=72 TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
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
    telemetry_base="${WORK_DIR}/telemetry/hierarchical_reduction_evaluation/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${EVALUATION_TREATMENT}"
    export FEDAPFA_GPU_TELEMETRY="${telemetry_base}.csv"
    telemetry_node_files=("${telemetry_base}.node-0.csv" "${telemetry_base}.node-1.csv")
    export FEDAPFA_GPU_TELEMETRY_NODE_FILES="$(IFS=,; printf '%s' "${telemetry_node_files[*]}")"
    mkdir -p "$(dirname "${FEDAPFA_GPU_TELEMETRY}")"
    : > "${FEDAPFA_GPU_TELEMETRY}"
    echo "allocation_task=${SLURM_ARRAY_TASK_ID} dataset=${dataset} seed=${seed} treatment=${EVALUATION_TREATMENT} position=$((position + 1)) rendezvous=${master_addr}:${master_port}"
    srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --kill-on-bad-exit=1 bash -c '
        set -euo pipefail
        [[ "${SLURM_NTASKS}" == 2 && "${SLURM_PROCID}" =~ ^[01]$ ]] || exit 2
        visible="$(${EVALUATION_PYTHON} -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.device_count())")"
        [[ "${visible}" == 2 ]] || { echo "each hierarchy node requires two visible CUDA devices" >&2; exit 2; }
        IFS=, read -r -a telemetry_node_files <<< "${FEDAPFA_GPU_TELEMETRY_NODE_FILES}"
        [[ "${#telemetry_node_files[@]}" == 2 ]] || { echo "hierarchy telemetry node paths are incomplete" >&2; exit 2; }
        node_telemetry="${telemetry_node_files[SLURM_PROCID]}"
        node_offset=$((SLURM_PROCID * 2))
        printf "%s\n" "timestamp,index,uuid,name,utilization.gpu,utilization.memory,node_rank" > "${node_telemetry}"
        sample_telemetry() {
            nvidia-smi --query-gpu=timestamp,index,uuid,name,utilization.gpu,utilization.memory --format=csv,noheader,nounits |
                awk -F, -v offset="${node_offset}" -v node_rank="${SLURM_PROCID}" \
                    "BEGIN{OFS=\",\"} {for(i=1;i<=NF;i++){gsub(/^[ \\t]+|[ \\t]+$/,\"\",\$i)}; \$2=\$2+offset; print \$0,node_rank}"
        }
        sample_telemetry >> "${node_telemetry}"
        (while true; do sleep 2; sample_telemetry; done) >> "${node_telemetry}" &
        telemetry_pid=$!
        stop_telemetry() {
            if [[ -n "${telemetry_pid:-}" ]] && kill -0 "${telemetry_pid}" 2>/dev/null; then
                kill "${telemetry_pid}" 2>/dev/null || true
                wait "${telemetry_pid}" 2>/dev/null || true
            fi
            telemetry_pid=""
        }
        trap stop_telemetry EXIT
        trap "exit 130" INT
        trap "exit 143" TERM
        "${EVALUATION_PYTHON}" -m torch.distributed.run \
            --nnodes=2 --nproc-per-node=2 --node-rank="${SLURM_PROCID}" \
            --rdzv-backend=c10d --rdzv-endpoint="${EVALUATION_MASTER_ADDR}:${EVALUATION_MASTER_PORT}" \
            --rdzv-id="${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${FEDAPFA_TREATMENT_POSITION}" \
            -m fedapfa.cli.train_system_evaluation "${EVALUATION_CONFIG}" \
            --seed "${EVALUATION_SEED}" --data-root "${EVALUATION_DATA_ROOT}" \
            --output-root "${EVALUATION_OUTPUT_ROOT}" --resume-auto
    '
    "${python_bin}" -m fedapfa.cli.merge_gpu_telemetry \
        --node-file "${telemetry_node_files[0]}" \
        --node-file "${telemetry_node_files[1]}" \
        --output "${FEDAPFA_GPU_TELEMETRY}" \
        --allocated-gpu-uuids "${FEDAPFA_ALLOCATED_GPU_UUIDS}"
done
