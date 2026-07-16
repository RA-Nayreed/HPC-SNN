#!/usr/bin/env bash
set -euo pipefail

usage() { echo "Usage: $0 --work-dir PATH --federated-runs-root PATH [--max-parallel N]" >&2; }
work_dir=""
federated_runs_root=""
max_parallel=1
while (($#)); do
    case "$1" in
        --work-dir) work_dir="${2:?}"; shift 2 ;;
        --federated-runs-root) federated_runs_root="${2:?}"; shift 2 ;;
        --max-parallel) max_parallel="${2:?}"; shift 2 ;;
        *) usage; exit 2 ;;
    esac
done
: "${CSC_PROJECT:?CSC_PROJECT must be set}"
: "${USER:?USER must be set}"
[[ -n "${work_dir}" && -n "${federated_runs_root}" ]] || { usage; exit 2; }
[[ "${max_parallel}" =~ ^[1-9][0-9]*$ ]] || { echo "invalid parallel task count" >&2; exit 2; }
work_dir="$(realpath -m "${work_dir}")"
scratch_prefix="/scratch/${CSC_PROJECT}/${USER}/"
case "${work_dir}/" in
    "${scratch_prefix}"*) ;;
    *) echo "work directory must be under ${scratch_prefix}" >&2; exit 2 ;;
esac
federated_runs_root="$(realpath -m "${federated_runs_root}")"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
manifest="${repo_root}/experiments/heterogeneity_evaluation/manifest.yaml"
module load python-pytorch/2.10
venv="${FEDAPFA_VENV:-/projappl/${CSC_PROJECT}/${USER}/hpc-snn-venv}"
[[ -f "${venv}/bin/activate" ]] || { echo "missing virtual environment: ${venv}" >&2; exit 2; }
source "${venv}/bin/activate"
python3 -m fedapfa.cli.scientific_manifest validate --manifest "${manifest}"
training_count="$(python3 -m fedapfa.cli.scientific_manifest count --manifest "${manifest}")"
context_count="$(python3 -m fedapfa.cli.scientific_manifest context-count --manifest "${manifest}")"
[[ "${training_count}" == "9" && "${context_count}" == "3" ]] || { echo "manifest task counts are incompatible" >&2; exit 2; }

mkdir -p "${work_dir}/runs/heterogeneity" "${work_dir}/diagnostics/heterogeneity/federated_baseline" \
    "${work_dir}/results/heterogeneity" "${work_dir}/slurm-logs/heterogeneity" \
    "${work_dir}/telemetry/heterogeneity"
training_job="$(sbatch --parsable --account="${CSC_PROJECT}" --chdir="${repo_root}" \
    --array="0-8%${max_parallel}" \
    --output="${work_dir}/slurm-logs/heterogeneity/training_%A_%a.out" \
    --error="${work_dir}/slurm-logs/heterogeneity/training_%A_%a.err" \
    --export="ALL,WORK=${work_dir},REPO_ROOT=${repo_root},MANIFEST=${manifest}" \
    "${script_dir}/heterogeneity_array.sbatch")"
training_job="${training_job%%;*}"
echo "Heterogeneity training job ID: ${training_job}"
context_job="$(sbatch --parsable --account="${CSC_PROJECT}" --chdir="${repo_root}" \
    --array="0-2%${max_parallel}" \
    --output="${work_dir}/slurm-logs/heterogeneity/context_%A_%a.out" \
    --error="${work_dir}/slurm-logs/heterogeneity/context_%A_%a.err" \
    --export="ALL,WORK=${work_dir},REPO_ROOT=${repo_root},MANIFEST=${manifest},FEDERATED_RUNS_ROOT=${federated_runs_root}" \
    "${script_dir}/heterogeneity_context_array.sbatch")"
context_job="${context_job%%;*}"
echo "Heterogeneity context job ID: ${context_job}"
