#!/usr/bin/env bash
#SBATCH --job-name=shd-context
#SBATCH --partition=gpumedium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=08:00:00

set -euo pipefail

: "${CSC_PROJECT:?CSC_PROJECT must be set}"
: "${USER:?USER must be set}"
: "${WORK:?WORK must be exported by the submission wrapper}"
: "${REPO_ROOT:?REPO_ROOT must be exported by the submission wrapper}"
: "${MANIFEST:?MANIFEST must be exported by the submission wrapper}"
: "${FEDERATED_RUNS_ROOT:?FEDERATED_RUNS_ROOT must be exported by the submission wrapper}"
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
context_count="$(python3 -m fedapfa.cli.scientific_manifest context-count --manifest "${MANIFEST}")"
[[ "${context_count}" == "3" ]] || { echo "context selection must contain 3 tasks" >&2; exit 2; }
task="$(python3 -m fedapfa.cli.scientific_manifest context-task --manifest "${MANIFEST}" --index "${SLURM_ARRAY_TASK_ID}")"
IFS=$'\t' read -r seed experiment source_path <<< "${task}"

data_root="${WORK}/data/shd"
context_root="${WORK}/diagnostics/heterogeneity/federated_baseline"
telemetry_root="${WORK}/telemetry/heterogeneity/context"
mkdir -p "${context_root}" "${telemetry_root}"
[[ -f "${data_root}/shd_train.h5" ]] || { echo "SHD training file is unavailable" >&2; exit 2; }
job_label="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}_${SLURM_ARRAY_TASK_ID}"
telemetry="${telemetry_root}/${job_label}.csv"
nvidia-smi --query-gpu=timestamp,index,uuid,name,utilization.gpu,memory.used,power.draw,temperature.gpu \
    --format=csv,nounits > "${telemetry}"
(
    while true; do
        nvidia-smi --query-gpu=timestamp,index,uuid,name,utilization.gpu,memory.used,power.draw,temperature.gpu \
            --format=csv,noheader,nounits >> "${telemetry}" || true
        sleep 2
    done
) &
telemetry_pid=$!
trap 'kill "${telemetry_pid}" 2>/dev/null || true; wait "${telemetry_pid}" 2>/dev/null || true' EXIT

echo "array=${SLURM_ARRAY_TASK_ID} experiment=${experiment} seed=${seed} source=${source_path}"
srun fedapfa-derive-heterogeneity-context \
    --runs-root "${FEDERATED_RUNS_ROOT}" \
    --federated-summary "${REPO_ROOT}/results/federated/federated_summary.json" \
    --data-root "${data_root}" --output-root "${context_root}" --seed "${seed}"
