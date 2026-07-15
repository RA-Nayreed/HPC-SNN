#!/usr/bin/env bash
#SBATCH --job-name=hpc-snn-fedavg
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
: "${REPO_ROOT:?REPO_ROOT must be exported by the submission wrapper}"
: "${FEDERATED_MANIFEST:?FEDERATED_MANIFEST must be exported by the submission wrapper}"
: "${SLURM_ARRAY_TASK_ID:?this script must run as a Slurm array task}"

WORK_DIR="$(realpath -m "${WORK_DIR}")"
REPO_ROOT="$(realpath -m "${REPO_ROOT}")"
scratch_prefix="/scratch/${CSC_PROJECT}/"
case "${WORK_DIR}/" in
    "${scratch_prefix}"*) ;;
    *) echo "WORK_DIR must be under ${scratch_prefix}" >&2; exit 2 ;;
esac
[[ -d "${REPO_ROOT}/.git" ]] || { echo "REPO_ROOT is not a Git checkout: ${REPO_ROOT}" >&2; exit 2; }
[[ -f "${FEDERATED_MANIFEST}" ]] || { echo "missing federated manifest: ${FEDERATED_MANIFEST}" >&2; exit 2; }

module purge
module load python-pytorch/2.10
venv="${FEDAPFA_VENV:-/projappl/${CSC_PROJECT}/${USER}/hpc-snn-venv}"
[[ -f "${venv}/bin/activate" ]] || { echo "missing Roihu virtual environment: ${venv}" >&2; exit 2; }
source "${venv}/bin/activate"

python3 - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable; federated execution requires one GH200")
print(f"CUDA device: {torch.cuda.get_device_name(0)}")
PY
python3 -m fedapfa.cli.federated_manifest validate --manifest "${FEDERATED_MANIFEST}"

task_line="$(python3 -m fedapfa.cli.federated_manifest task \
    --manifest "${FEDERATED_MANIFEST}" --index "${SLURM_ARRAY_TASK_ID}")"
IFS=$'\t' read -r config_path seed dataset mode protocol experiment participation <<< "${task_line}"
[[ "${dataset}" == "shd" ]] || { echo "unsupported federated dataset: ${dataset}" >&2; exit 2; }
[[ "${mode}" == "scientific_evaluation" ]] || { echo "manifest task is not a scientific evaluation" >&2; exit 2; }
[[ "${protocol}" == "independent_evaluation" ]] || { echo "manifest task has an incompatible protocol" >&2; exit 2; }

data_root="${WORK_DIR}/data/shd"
runs_root="${WORK_DIR}/runs/federated"
telemetry_root="${WORK_DIR}/telemetry/federated"
mkdir -p "${runs_root}" "${telemetry_root}"
[[ -f "${data_root}/shd_train.h5" ]] || { echo "missing SHD training file: ${data_root}/shd_train.h5" >&2; exit 2; }
[[ -f "${data_root}/shd_test.h5" ]] || { echo "missing SHD test file: ${data_root}/shd_test.h5" >&2; exit 2; }

job_label="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}_${SLURM_ARRAY_TASK_ID}"
telemetry_csv="${telemetry_root}/${job_label}.csv"
telemetry_command="${telemetry_root}/${job_label}-command.txt"
telemetry_unsupported="${telemetry_root}/${job_label}-unsupported-fields.txt"
telemetry_status="${telemetry_root}/${job_label}-status.txt"
sampling_interval_seconds=2
requested_fields=(timestamp index uuid name utilization.gpu utilization.memory memory.used memory.total power.draw temperature.gpu)
supported_fields=()
unsupported_fields=()
for field in "${requested_fields[@]}"; do
    if nvidia-smi --query-gpu="${field}" --format=csv,noheader,nounits >/dev/null 2>&1; then
        supported_fields+=("${field}")
    else
        unsupported_fields+=("${field}")
    fi
done
if ((${#supported_fields[@]} == 0)); then
    echo "nvidia-smi exposes none of the requested telemetry fields" >&2
    exit 2
fi
query="$(IFS=,; echo "${supported_fields[*]}")"
printf 'interval_seconds=%s\ncommand=nvidia-smi --query-gpu=%s --format=csv,nounits\n' \
    "${sampling_interval_seconds}" "${query}" > "${telemetry_command}"
if ((${#unsupported_fields[@]} == 0)); then
    echo "none" > "${telemetry_unsupported}"
else
    printf '%s\n' "${unsupported_fields[@]}" > "${telemetry_unsupported}"
fi
nvidia-smi --query-gpu="${query}" --format=csv,nounits > "${telemetry_csv}"

telemetry_pid=""
stop_telemetry() {
    if [[ -n "${telemetry_pid}" ]] && kill -0 "${telemetry_pid}" 2>/dev/null; then
        kill "${telemetry_pid}" 2>/dev/null || true
        wait "${telemetry_pid}" 2>/dev/null || true
    fi
}
trap stop_telemetry EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
(
    while true; do
        if ! nvidia-smi --query-gpu="${query}" --format=csv,noheader,nounits >> "${telemetry_csv}" 2>> "${telemetry_status}"; then
            echo "telemetry sample failed at $(date --iso-8601=seconds)" >> "${telemetry_status}"
        fi
        sleep "${sampling_interval_seconds}"
    done
) &
telemetry_pid=$!

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
cd "${REPO_ROOT}"
echo "task=${SLURM_ARRAY_TASK_ID} experiment=${experiment} seed=${seed} participation=${participation}"
srun python3 -m fedapfa.cli.train_federated "${config_path}" \
    --seed "${seed}" \
    --data-root "${data_root}" \
    --output-root "${runs_root}" \
    --resume-auto
