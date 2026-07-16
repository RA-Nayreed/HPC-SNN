#!/usr/bin/env bash
#SBATCH --job-name=published-fedsnn
#SBATCH --partition=gpumedium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=72:00:00

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
[[ "${task_count}" == "3" ]] || { echo "published protocol manifest must contain 3 tasks" >&2; exit 2; }
task="$(python3 -m fedapfa.cli.scientific_manifest task --manifest "${MANIFEST}" --index "${SLURM_ARRAY_TASK_ID}")"
IFS=$'\t' read -r config seed dataset mode protocol experiment participation <<< "${task}"
[[ "${dataset}" == "cifar10" && "${mode}" == "scientific_evaluation" && "${protocol}" == "published_protocol" ]] || {
    echo "incompatible published protocol task" >&2; exit 2;
}
data_root="${WORK}/data/cifar10"
runs_root="${WORK}/runs/published_fedsnn"
telemetry_root="${WORK}/telemetry/published_fedsnn"
mkdir -p "${runs_root}" "${telemetry_root}"
[[ -f "${data_root}/cifar-10-batches-py/data_batch_1" ]] || { echo "CIFAR-10 training files are unavailable" >&2; exit 2; }
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
echo "array=${SLURM_ARRAY_TASK_ID} experiment=${experiment} seed=${seed} participation=${participation}"
srun python3 -m fedapfa.cli.train_federated "${config}" \
    --seed "${seed}" --data-root "${data_root}" --output-root "${runs_root}" --resume-auto
