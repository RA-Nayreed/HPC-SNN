#!/usr/bin/env bash
#SBATCH --job-name=snn-scaling-energy
#SBATCH --partition=gpumedium
#SBATCH --time=36:00:00

set -euo pipefail
: "${CSC_PROJECT:?CSC_PROJECT must be set}"
: "${USER:?USER must be set}"
: "${WORK_DIR:?WORK_DIR must be exported}"
: "${REPO_ROOT:?REPO_ROOT must be exported}"
: "${EVALUATION_MANIFEST:?EVALUATION_MANIFEST must be exported}"
: "${SCALING_TOPOLOGY:?SCALING_TOPOLOGY must be exported}"
: "${SCALING_ALLOCATION_INDICES:?SCALING_ALLOCATION_INDICES must be exported}"
: "${SLURM_ARRAY_TASK_ID:?Slurm array task is required}"

module purge
module load python-pytorch/2.10
venv="${FEDAPFA_VENV:-/projappl/${CSC_PROJECT}/${USER}/hpc-snn-venv}"
python_bin="${venv}/bin/python3"
[[ -x "${python_bin}" ]] || { echo "project Python interpreter is unavailable" >&2; exit 2; }
export EVALUATION_PYTHON="${python_bin}"
cd "${REPO_ROOT}"
"${python_bin}" -c 'import fedapfa, torch; assert torch.cuda.is_available(); assert torch.version.cuda; assert torch.distributed.is_nccl_available()'
"${python_bin}" -m fedapfa.cli.scientific_manifest validate --manifest "${EVALUATION_MANIFEST}"

[[ "${SCALING_ALLOCATION_INDICES}" =~ ^[0-9]+(:[0-9]+){5}$ && "${SLURM_ARRAY_TASK_ID}" =~ ^[0-5]$ ]] || {
    echo "scaling allocation-index map is incompatible" >&2
    exit 2
}
IFS=: read -r -a allocation_indices <<< "${SCALING_ALLOCATION_INDICES}"
[[ "${#allocation_indices[@]}" == 6 ]] || {
    echo "scaling allocation-index map is incompatible" >&2
    exit 2
}
declare -A observed_allocation_indices=()
for index in "${allocation_indices[@]}"; do
    canonical_index="$((10#${index}))"
    [[ -z "${observed_allocation_indices[${canonical_index}]+x}" ]] || {
        echo "scaling allocation-index map is incompatible" >&2
        exit 2
    }
    observed_allocation_indices["${canonical_index}"]=1
done
manifest_index="${allocation_indices[SLURM_ARRAY_TASK_ID]}"
allocation="$("${python_bin}" -m fedapfa.cli.scientific_manifest allocation-task \
    --manifest "${EVALUATION_MANIFEST}" --index "${manifest_index}")"
IFS=$'\t' read -r allocation_index dataset seed treatment config <<< "${allocation}"
[[ "${allocation_index}" == "${manifest_index}" && "${treatment}" == "${SCALING_TOPOLOGY}" ]] || {
    echo "scaling manifest allocation differs from the submitted topology" >&2
    exit 2
}

case "${dataset}" in
    shd)
        data_root="${WORK_DIR}/data/shd"
        [[ -f "${data_root}/shd_train.h5" && -f "${data_root}/shd_test.h5" ]]
        ;;
    ssc)
        data_root="${WORK_DIR}/data/ssc"
        [[ -f "${data_root}/ssc_train.h5" && -f "${data_root}/ssc_valid.h5" && -f "${data_root}/ssc_test.h5" ]]
        ;;
    *) echo "unsupported comparative dataset" >&2; exit 2 ;;
esac || { echo "local ${dataset^^} data are unavailable" >&2; exit 2; }

case "${SCALING_TOPOLOGY}" in
    one_node_one_gpu) expected_nodes=1; expected_per_node=1; expected_gpus=1; expected_processes=1 ;;
    one_node_two_gpu) expected_nodes=1; expected_per_node=2; expected_gpus=2; expected_processes=2 ;;
    one_node_four_gpu) expected_nodes=1; expected_per_node=4; expected_gpus=4; expected_processes=4 ;;
    two_nodes_four_gpus) expected_nodes=2; expected_per_node=2; expected_gpus=4; expected_processes=4 ;;
    *) echo "unsupported scaling topology" >&2; exit 2 ;;
esac
[[ "${SLURM_JOB_NUM_NODES}" == "${expected_nodes}" ]] || { echo "Slurm node layout differs" >&2; exit 2; }
[[ "${SLURM_NTASKS}" == "${expected_nodes}" ]] || { echo "Slurm node-side launcher count differs" >&2; exit 2; }
[[ -d "${TMPDIR}" && -w "${TMPDIR}" ]] || { echo "TMPDIR is not writable" >&2; exit 2; }
calibration="${WORK_DIR}/calibration/system_scaling_energy_evaluation/${SCALING_TOPOLOGY}/instrumentation_calibration.json"
[[ -f "${calibration}" ]] || { echo "topology-compatible calibration is unavailable: ${calibration}" >&2; exit 2; }

mapfile -t allocated_nodes < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
[[ "${#allocated_nodes[@]}" == "${expected_nodes}" ]] || { echo "allocated node coverage differs" >&2; exit 2; }
uuid_dir="${WORK_DIR}/telemetry/system_scaling_energy_evaluation/uuid-preflight/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${SCALING_TOPOLOGY}"
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
[[ "${#raw_uuids[@]}" == "${expected_gpus}" && "${#canonical_uuids[@]}" == "${expected_gpus}" ]] || {
    echo "allocation GPU UUID preflight is incomplete" >&2
    exit 2
}
export FEDAPFA_ALLOCATED_GPU_UUIDS_RAW="$(IFS=,; printf '%s' "${raw_uuids[*]}")"
export FEDAPFA_ALLOCATED_GPU_UUIDS="$(IFS=,; printf '%s' "${canonical_uuids[*]}")"
unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY CUDA_MPS_ACTIVE_THREAD_PERCENTAGE
unset NCCL_ASYNC_ERROR_HANDLING
export OMP_NUM_THREADS=72
export MKL_NUM_THREADS=72
export OPENBLAS_NUM_THREADS=72
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export FEDAPFA_ALLOCATION_EXECUTION_ORDER="${treatment}"
export FEDAPFA_TREATMENT_POSITION=1

output_root="${WORK_DIR}/runs/system_scaling_energy_evaluation"
mkdir -p "${output_root}"
timing_root="${output_root}/allocation_timing"
mkdir -p "${timing_root}"
timing_path="${timing_root}/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.json"
"${python_bin}" -m fedapfa.cli.record_comparative_allocation_timing start \
    --path "${timing_path}" --allocation-index "${allocation_index}" \
    --collection system_scaling_energy_evaluation --dataset "${dataset}" \
    --seed "${seed}" --order "${treatment}"
"${python_bin}" -m fedapfa.cli.record_comparative_allocation_timing begin-treatment \
    --path "${timing_path}" --allocation-index "${allocation_index}" \
    --treatment "${treatment}" --position 1
echo "allocation_index=${allocation_index} dataset=${dataset} seed=${seed} topology=${treatment} nodes=${expected_nodes} gpus=${expected_gpus} processes=${expected_processes}"
if [[ "${expected_nodes}" == 1 ]]; then
    visible="$("${python_bin}" -c 'import torch; print(torch.cuda.device_count())')"
    [[ "${visible}" == "${expected_per_node}" ]] || { echo "visible CUDA count differs" >&2; exit 2; }
    srun --nodes=1 --ntasks=1 --ntasks-per-node=1 --kill-on-bad-exit=1 \
        "${python_bin}" -m torch.distributed.run --standalone --nnodes=1 \
        --nproc-per-node="${expected_processes}" -m fedapfa.cli.train_comparative_evaluation \
        "${config}" --seed "${seed}" --data-root "${data_root}" --output-root "${output_root}" \
        --calibration "${calibration}" --manifest "${EVALUATION_MANIFEST}" --resume-auto
else
    master_addr="${allocated_nodes[0]}"
    master_port="$((15000 + SLURM_JOB_ID % 40000))"
    export EVALUATION_MASTER_ADDR="${master_addr}"
    export EVALUATION_MASTER_PORT="${master_port}"
    export EVALUATION_CONFIG="${config}"
    export EVALUATION_SEED="${seed}"
    export EVALUATION_DATA_ROOT="${data_root}"
    export EVALUATION_OUTPUT_ROOT="${output_root}"
    export EVALUATION_CALIBRATION="${calibration}"
    srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --kill-on-bad-exit=1 bash -c '
        set -euo pipefail
        visible="$(${EVALUATION_PYTHON} -c "import torch; print(torch.cuda.device_count())")"
        [[ "${visible}" == 2 ]] || exit 2
        "${EVALUATION_PYTHON}" -m torch.distributed.run \
            --nnodes=2 --nproc-per-node=2 --node-rank="${SLURM_PROCID}" \
            --rdzv-backend=c10d --rdzv-endpoint="${EVALUATION_MASTER_ADDR}:${EVALUATION_MASTER_PORT}" \
            --rdzv-id="${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}" \
            -m fedapfa.cli.train_comparative_evaluation "${EVALUATION_CONFIG}" \
            --seed "${EVALUATION_SEED}" --data-root "${EVALUATION_DATA_ROOT}" \
            --output-root "${EVALUATION_OUTPUT_ROOT}" --calibration "${EVALUATION_CALIBRATION}" --manifest "${EVALUATION_MANIFEST}" --resume-auto
    '
fi
"${python_bin}" -m fedapfa.cli.record_comparative_allocation_timing end-treatment \
    --path "${timing_path}" --allocation-index "${allocation_index}" \
    --treatment "${treatment}" --position 1
"${python_bin}" -m fedapfa.cli.record_comparative_allocation_timing complete \
    --path "${timing_path}" --allocation-index "${allocation_index}"
