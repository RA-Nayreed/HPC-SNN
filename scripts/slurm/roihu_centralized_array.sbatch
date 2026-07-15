#!/usr/bin/env bash
#SBATCH --job-name=hpc-snn-centralized
#SBATCH --partition=gpumedium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=36:00:00

set -euo pipefail

: "${CSC_PROJECT:?CSC_PROJECT must be set}"
: "${USER:?USER must be set}"
: "${WORK_DIR:?WORK_DIR must be exported by the submission wrapper}"
: "${CENTRALIZED_MANIFEST:?CENTRALIZED_MANIFEST must be exported by the submission wrapper}"
: "${SLURM_ARRAY_TASK_ID:?this script must run as a Slurm array task}"

WORK_DIR="$(realpath -m "${WORK_DIR}")"
scratch_prefix="/scratch/${CSC_PROJECT}/"
case "${WORK_DIR}/" in
    "${scratch_prefix}"*) ;;
    *) echo "WORK_DIR must be under ${scratch_prefix}" >&2; exit 2 ;;
esac

module purge
module load python-pytorch/2.10
venv="/projappl/${CSC_PROJECT}/${USER}/hpc-snn-venv"
if [[ ! -f "${venv}/bin/activate" ]]; then
    echo "missing Roihu virtual environment: ${venv}" >&2
    exit 2
fi
source "${venv}/bin/activate"

python3 -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print(torch.cuda.get_device_name(0))'
python3 -c 'import DCLS; print("DCLS import passed")'
python3 -m fedapfa.cli.centralized_manifest validate --manifest "${CENTRALIZED_MANIFEST}"

task_line="$(python3 -m fedapfa.cli.centralized_manifest task --manifest "${CENTRALIZED_MANIFEST}" --index "${SLURM_ARRAY_TASK_ID}")"
IFS=$'\t' read -r config_path seed dataset mode protocol experiment <<< "${task_line}"
if [[ "${mode}" != "scientific_evaluation" ]]; then
    echo "manifest task ${SLURM_ARRAY_TASK_ID} is not scientific_evaluation mode" >&2
    exit 2
fi
task_label="${experiment} ${config_path}"
if [[ "${task_label,,}" =~ reduced[_-]?sample|sweep|memorization ]]; then
    echo "manifest task contains a forbidden reduced-sample label: ${task_label}" >&2
    exit 2
fi
case "${dataset}" in
    shd) data_root="${WORK_DIR}/data/shd" ;;
    ssc) data_root="${WORK_DIR}/data/ssc" ;;
    *) echo "unsupported centralized dataset: ${dataset}" >&2; exit 2 ;;
esac
runs_root="${WORK_DIR}/runs/centralized"
mkdir -p "${runs_root}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "task=${SLURM_ARRAY_TASK_ID} experiment=${experiment} seed=${seed} protocol=${protocol}"
srun python3 -m fedapfa.cli.train_centralized "${config_path}" \
    --seed "${seed}" \
    --data-root "${data_root}" \
    --output-root "${runs_root}" \
    --resume-auto
