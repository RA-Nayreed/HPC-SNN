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
    echo "--max-parallel must be a positive integer" >&2; exit 2;
}
work_dir="$(realpath -m "${work_dir}")"
case "${work_dir}/" in
    "/scratch/${CSC_PROJECT}/${USER}/"*) ;;
    *) echo "work directory must be below /scratch/${CSC_PROJECT}/${USER}" >&2; exit 2 ;;
esac
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
manifest="${repo_root}/experiments/hierarchical_reduction_evaluation/manifest.yaml"
logs="${work_dir}/slurm-logs/hierarchical_reduction_evaluation"
mkdir -p "${logs}" "${work_dir}/runs/hierarchical_reduction_evaluation" \
    "${work_dir}/generated-evidence/hierarchical_reduction_evaluation" \
    "${work_dir}/telemetry/hierarchical_reduction_evaluation"
module load python-pytorch/2.10
venv="${FEDAPFA_VENV:-/projappl/${CSC_PROJECT}/${USER}/hpc-snn-venv}"
python_bin="${venv}/bin/python3"
[[ -x "${python_bin}" ]] || { echo "project Python interpreter is unavailable" >&2; exit 2; }
"${python_bin}" -c 'import fedapfa, torch'
"${python_bin}" -m fedapfa.cli.scientific_manifest validate --manifest "${manifest}"
scientific_count="$("${python_bin}" -m fedapfa.cli.scientific_manifest count --manifest "${manifest}")"
allocation_count="$("${python_bin}" -m fedapfa.cli.scientific_manifest allocation-count --manifest "${manifest}")"
[[ "${scientific_count}" == 12 && "${allocation_count}" == 6 ]] || {
    echo "hierarchical matrix must contain 12 executions in six allocations" >&2; exit 2;
}
for ((index=0; index<allocation_count; index++)); do
    "${python_bin}" -m fedapfa.cli.scientific_manifest allocation-task --manifest "${manifest}" --index "${index}"
done
array="0-5"
simultaneous_allocations=6
if [[ -n "${max_parallel}" ]]; then
    array+="%${max_parallel}"
    ((max_parallel < simultaneous_allocations)) && simultaneous_allocations="${max_parallel}"
fi
echo "allocation_task_count=6"
echo "scientific_execution_count=12"
echo "gpus_per_allocation=4"
echo "nodes_per_allocation=2"
echo "maximum_possible_simultaneous_gpu_demand=$((simultaneous_allocations * 4))"
echo "partition=gpumedium"
job_id="$(sbatch --parsable --account="${CSC_PROJECT}" --chdir="${repo_root}" \
    --array="${array}" --output="${logs}/%A_%a.out" --error="${logs}/%A_%a.err" \
    --export="ALL,WORK_DIR=${work_dir},REPO_ROOT=${repo_root},EVALUATION_MANIFEST=${manifest}" \
    "${script_dir}/hierarchical_reduction.sbatch")"
job_id="${job_id%%;*}"
[[ "${job_id}" =~ ^[0-9]+$ ]] || { echo "Slurm returned an invalid job identifier" >&2; exit 2; }
echo "hierarchical_reduction_job_id=${job_id}"
