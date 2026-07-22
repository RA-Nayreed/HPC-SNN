#!/usr/bin/env bash
#SBATCH --job-name=snn-noniid-energy
#SBATCH --partition=gpumedium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:gh200:4
#SBATCH --cpus-per-task=288
#SBATCH --mem=868344M
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
"${python_bin}" -c 'import fedapfa, torch; assert torch.cuda.is_available(); assert torch.version.cuda; assert torch.distributed.is_nccl_available()'
"${python_bin}" -m fedapfa.cli.scientific_manifest validate --manifest "${EVALUATION_MANIFEST}"
allocation="$("${python_bin}" -m fedapfa.cli.scientific_manifest allocation-task \
    --manifest "${EVALUATION_MANIFEST}" --index "${SLURM_ARRAY_TASK_ID}")"
IFS=$'\t' read -r allocation_index dataset seed \
    treatment_1 config_1 treatment_2 config_2 treatment_3 config_3 treatment_4 config_4 <<< "${allocation}"
[[ "${allocation_index}" == "${SLURM_ARRAY_TASK_ID}" ]] || { echo "allocation index differs" >&2; exit 2; }
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
[[ "${SLURM_JOB_NUM_NODES}" == 1 && "${SLURM_NTASKS}" == 1 ]] || exit 2
[[ -d "${TMPDIR}" && -w "${TMPDIR}" ]] || { echo "TMPDIR is not writable" >&2; exit 2; }
visible="$("${python_bin}" -c 'import torch; print(torch.cuda.device_count())')"
[[ "${visible}" == 4 ]] || { echo "non-IID allocation requires four visible physical GPUs" >&2; exit 2; }
mapfile -t raw_uuids < <(nvidia-smi --query-gpu=uuid --format=csv,noheader,nounits)
[[ "${#raw_uuids[@]}" == 4 ]] || { echo "non-IID allocation UUID coverage differs" >&2; exit 2; }
export FEDAPFA_ALLOCATED_GPU_UUIDS_RAW="$(IFS=,; printf '%s' "${raw_uuids[*]}")"
export FEDAPFA_ALLOCATED_GPU_UUIDS="$("${python_bin}" -c '
import sys
from fedapfa.distributed.process_context import allocated_gpu_uuids
print(",".join(allocated_gpu_uuids(sys.argv[1], expected_count=4)))
' "${FEDAPFA_ALLOCATED_GPU_UUIDS_RAW}")"
calibration="${WORK_DIR}/calibration/system_scaling_energy_evaluation/one_node_four_gpu/instrumentation_calibration.json"
[[ -f "${calibration}" ]] || { echo "four-GPU calibration is unavailable" >&2; exit 2; }
unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY CUDA_MPS_ACTIVE_THREAD_PERCENTAGE
unset NCCL_ASYNC_ERROR_HANDLING
export OMP_NUM_THREADS=72
export MKL_NUM_THREADS=72
export OPENBLAS_NUM_THREADS=72
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export FEDAPFA_ALLOCATION_EXECUTION_ORDER="${treatment_1},${treatment_2},${treatment_3},${treatment_4}"
treatments=("${treatment_1}" "${treatment_2}" "${treatment_3}" "${treatment_4}")
configs=("${config_1}" "${config_2}" "${config_3}" "${config_4}")
output_root="${WORK_DIR}/runs/non_iid_energy_evaluation"
mkdir -p "${output_root}"
timing_root="${output_root}/allocation_timing"
mkdir -p "${timing_root}"
timing_path="${timing_root}/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.json"
"${python_bin}" -m fedapfa.cli.record_comparative_allocation_timing start \
    --path "${timing_path}" --allocation-index "${allocation_index}" \
    --collection non_iid_energy_evaluation --dataset "${dataset}" \
    --seed "${seed}" --order "${FEDAPFA_ALLOCATION_EXECUTION_ORDER}"
for position in 0 1 2 3; do
    export FEDAPFA_TREATMENT_POSITION="$((position + 1))"
    "${python_bin}" -m fedapfa.cli.record_comparative_allocation_timing begin-treatment \
        --path "${timing_path}" --allocation-index "${allocation_index}" \
        --treatment "${treatments[position]}" --position "$((position + 1))"
    echo "allocation_index=${allocation_index} dataset=${dataset} seed=${seed} treatment=${treatments[position]} position=$((position + 1))"
    srun --nodes=1 --ntasks=1 --ntasks-per-node=1 --kill-on-bad-exit=1 \
        "${python_bin}" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=4 \
        -m fedapfa.cli.train_comparative_evaluation "${configs[position]}" \
        --seed "${seed}" --data-root "${data_root}" --output-root "${output_root}" \
        --calibration "${calibration}" --manifest "${EVALUATION_MANIFEST}" --resume-auto
    "${python_bin}" -m fedapfa.cli.record_comparative_allocation_timing end-treatment \
        --path "${timing_path}" --allocation-index "${allocation_index}" \
        --treatment "${treatments[position]}" --position "$((position + 1))"
done
"${python_bin}" -m fedapfa.cli.record_comparative_allocation_timing complete \
    --path "${timing_path}" --allocation-index "${allocation_index}"
