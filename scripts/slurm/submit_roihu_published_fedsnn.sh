#!/usr/bin/env bash
set -euo pipefail
usage() { echo "Usage: $0 --work-dir PATH [--max-parallel N]" >&2; }
work_dir=""
max_parallel=1
while (($#)); do
    case "$1" in
        --work-dir) work_dir="${2:?}"; shift 2 ;;
        --max-parallel) max_parallel="${2:?}"; shift 2 ;;
        *) usage; exit 2 ;;
    esac
done
: "${CSC_PROJECT:?CSC_PROJECT must be set}"
: "${USER:?USER must be set}"
[[ -n "${work_dir}" ]] || { usage; exit 2; }
[[ "${max_parallel}" =~ ^[1-9][0-9]*$ ]] || { echo "invalid parallel task count" >&2; exit 2; }
work_dir="$(realpath -m "${work_dir}")"
scratch_prefix="/scratch/${CSC_PROJECT}/${USER}/"
case "${work_dir}/" in
    "${scratch_prefix}"*) ;;
    *) echo "work directory must be under ${scratch_prefix}" >&2; exit 2 ;;
esac
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
manifest="${repo_root}/experiments/published_fedsnn/manifest.yaml"
module load python-pytorch/2.10
venv="${FEDAPFA_VENV:-/projappl/${CSC_PROJECT}/${USER}/hpc-snn-venv}"
[[ -f "${venv}/bin/activate" ]] || { echo "missing virtual environment: ${venv}" >&2; exit 2; }
source "${venv}/bin/activate"
python3 -m fedapfa.cli.scientific_manifest validate --manifest "${manifest}"
task_count="$(python3 -m fedapfa.cli.scientific_manifest count --manifest "${manifest}")"
[[ "${task_count}" == "3" ]] || { echo "published protocol manifest must contain 3 tasks" >&2; exit 2; }
mkdir -p "${work_dir}/runs/published_fedsnn" "${work_dir}/results/published_fedsnn" \
    "${work_dir}/slurm-logs/published_fedsnn" "${work_dir}/telemetry/published_fedsnn"
job_id="$(sbatch --parsable --account="${CSC_PROJECT}" --chdir="${repo_root}" \
    --array="0-2%${max_parallel}" \
    --output="${work_dir}/slurm-logs/published_fedsnn/%A_%a.out" \
    --error="${work_dir}/slurm-logs/published_fedsnn/%A_%a.err" \
    --export="ALL,WORK=${work_dir},REPO_ROOT=${repo_root},MANIFEST=${manifest}" \
    "${script_dir}/published_fedsnn_array.sbatch")"
job_id="${job_id%%;*}"
echo "Published Fed-SNN job ID: ${job_id}"
