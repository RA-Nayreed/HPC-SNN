#!/usr/bin/env bash
set -euo pipefail

usage() { echo "Usage: $0 --work-dir PATH [--max-parallel N]" >&2; }
work_dir=""
max_parallel=""
while (($#)); do
    case "$1" in
        --work-dir) [[ $# -ge 2 ]] || { usage; exit 2; }; work_dir="$2"; shift 2 ;;
        --max-parallel) [[ $# -ge 2 ]] || { usage; exit 2; }; max_parallel="$2"; shift 2 ;;
        *) usage; exit 2 ;;
    esac
done
: "${CSC_PROJECT:?CSC_PROJECT must be set}"
: "${USER:?USER must be set}"
[[ -n "${work_dir}" ]] || { usage; exit 2; }
[[ -z "${max_parallel}" || "${max_parallel}" =~ ^[1-9][0-9]*$ ]] || {
    echo "--max-parallel must be a positive integer" >&2
    exit 2
}
work_dir="$(realpath -m "${work_dir}")"
case "${work_dir}/" in
    "/scratch/${CSC_PROJECT}/${USER}/"*) ;;
    *) echo "work directory must be below /scratch/${CSC_PROJECT}/${USER}" >&2; exit 2 ;;
esac
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
manifest="${repo_root}/experiments/non_iid_energy_evaluation/manifest.yaml"
logs="${work_dir}/slurm-logs/non_iid_energy_evaluation"
mkdir -p "${logs}" "${work_dir}/runs/non_iid_energy_evaluation" \
    "${work_dir}/generated-evidence/non_iid_energy_evaluation" \
    "${work_dir}/telemetry/non_iid_energy_evaluation"
module load python-pytorch/2.10
venv="${FEDAPFA_VENV:-/projappl/${CSC_PROJECT}/${USER}/hpc-snn-venv}"
python_bin="${venv}/bin/python3"
[[ -x "${python_bin}" ]] || { echo "project Python interpreter is unavailable" >&2; exit 2; }
"${python_bin}" -c 'import fedapfa, torch'
"${python_bin}" -m fedapfa.cli.scientific_manifest validate --manifest "${manifest}"
task_count="$("${python_bin}" -m fedapfa.cli.scientific_manifest count --manifest "${manifest}")"
allocation_count="$("${python_bin}" -m fedapfa.cli.scientific_manifest allocation-count --manifest "${manifest}")"
[[ "${task_count}" == 24 && "${allocation_count}" == 6 ]] || {
    echo "non-IID matrix must contain 24 executions in six allocations" >&2
    exit 2
}
array="0-5"
if [[ -n "${max_parallel}" ]]; then
    array+="%${max_parallel}"
fi
job_id="$(sbatch --parsable --account="${CSC_PROJECT}" --chdir="${repo_root}" \
    --array="${array}" --output="${logs}/%A_%a.out" --error="${logs}/%A_%a.err" \
    --export="ALL,WORK_DIR=${work_dir},REPO_ROOT=${repo_root},EVALUATION_MANIFEST=${manifest}" \
    "${script_dir}/non_iid_energy.sbatch")"
job_id="${job_id%%;*}"
[[ "${job_id}" =~ ^[0-9]+$ ]] || { echo "Slurm returned an invalid job identifier" >&2; exit 2; }
echo "non_iid_energy_job_id=${job_id}"
echo "scientific_execution_count=24"
echo "slurm_allocation_count=6"
echo "executions_per_allocation=4"
echo "partition=gpumedium"
