#!/usr/bin/env bash
#SBATCH --job-name=shd-heterogeneity
#SBATCH --partition=gpumedium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=36:00:00

set -euo pipefail

: "${CSC_PROJECT:?CSC_PROJECT must be set}"
: "${USER:?USER must be set}"
: "${WORK:?WORK must be exported by the submission wrapper}"
: "${REPO_ROOT:?REPO_ROOT must be exported by the submission wrapper}"
: "${MANIFEST:?MANIFEST must be exported by the submission wrapper}"
: "${SLURM_ARRAY_TASK_ID:?this script requires a Slurm array index}"
WORK="$(realpath -m "${WORK}")"
scratch_prefix="/scratch/${CSC_PROJECT}/${USER}/"
case "${WORK}/" in
    "${scratch_prefix}"*) ;;
    *) echo "WORK must be under ${scratch_prefix}" >&2; exit 2 ;;
esac

module purge
module load python-pytorch/2.10
venv="${FEDAPFA_VENV:-/projappl/${CSC_PROJECT}/${USER}/hpc-snn-venv}"
[[ -f "${venv}/bin/activate" ]] || { echo "missing virtual environment: ${venv}" >&2; exit 2; }
source "${venv}/bin/activate"
cd "${REPO_ROOT}"

python3 - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable; one GH200 is required")
print(f"CUDA device: {torch.cuda.get_device_name(0)}")
PY
nvidia-smi
python3 -m fedapfa.cli.scientific_manifest validate --manifest "${MANIFEST}"
task_count="$(python3 -m fedapfa.cli.scientific_manifest count --manifest "${MANIFEST}")"
[[ "${task_count}" == "9" ]] || { echo "heterogeneity manifest must contain 9 tasks" >&2; exit 2; }
task="$(python3 -m fedapfa.cli.scientific_manifest task --manifest "${MANIFEST}" --index "${SLURM_ARRAY_TASK_ID}")"
IFS=$'\t' read -r config seed dataset mode protocol experiment participation <<< "${task}"
[[ "${dataset}" == "shd" && "${mode}" == "scientific_evaluation" && "${protocol}" == "independent_evaluation" ]] || {
    echo "incompatible heterogeneity task" >&2; exit 2;
}
[[ "${participation}" == "0.5" ]] || { echo "participation must be 0.50" >&2; exit 2; }

data_root="${WORK}/data/shd"
runs_root="${WORK}/runs/heterogeneity"
telemetry_root="${WORK}/telemetry/heterogeneity"
mkdir -p "${runs_root}" "${telemetry_root}"
[[ -f "${data_root}/shd_train.h5" && -f "${data_root}/shd_test.h5" ]] || {
    echo "SHD files are unavailable under ${data_root}" >&2; exit 2;
}
job_label="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}_${SLURM_ARRAY_TASK_ID}"
telemetry="${telemetry_root}/${job_label}.csv"
nvidia-smi --query-gpu=timestamp,index,uuid,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu \
    --format=csv,nounits > "${telemetry}"
(
    while true; do
        nvidia-smi --query-gpu=timestamp,index,uuid,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu \
            --format=csv,noheader,nounits >> "${telemetry}" || true
        sleep 2
    done
) &
telemetry_pid=$!
trap 'kill "${telemetry_pid}" 2>/dev/null || true; wait "${telemetry_pid}" 2>/dev/null || true' EXIT

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
echo "array=${SLURM_ARRAY_TASK_ID} experiment=${experiment} seed=${seed}"
srun python3 -m fedapfa.cli.train_federated "${config}" \
    --seed "${seed}" --data-root "${data_root}" --output-root "${runs_root}" --resume-auto
