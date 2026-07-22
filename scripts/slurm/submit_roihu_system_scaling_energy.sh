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
manifest="${repo_root}/experiments/system_scaling_energy_evaluation/manifest.yaml"
module load python-pytorch/2.10
venv="${FEDAPFA_VENV:-/projappl/${CSC_PROJECT}/${USER}/hpc-snn-venv}"
python_bin="${venv}/bin/python3"
[[ -x "${python_bin}" ]] || { echo "project Python interpreter is unavailable" >&2; exit 2; }
"${python_bin}" -c 'import fedapfa, torch'
"${python_bin}" -m fedapfa.cli.scientific_manifest validate --manifest "${manifest}"
task_count="$("${python_bin}" -m fedapfa.cli.scientific_manifest count --manifest "${manifest}")"
allocation_count="$("${python_bin}" -m fedapfa.cli.scientific_manifest allocation-count --manifest "${manifest}")"
[[ "${task_count}" == 24 && "${allocation_count}" == 24 ]] || {
    echo "system scaling matrix must contain 24 executions in 24 allocations" >&2
    exit 2
}
mkdir -p "${work_dir}/slurm-logs/system_scaling_energy_evaluation" \
    "${work_dir}/runs/system_scaling_energy_evaluation" \
    "${work_dir}/generated-evidence/system_scaling_energy_evaluation" \
    "${work_dir}/telemetry/system_scaling_energy_evaluation"
array="0-5"
if [[ -n "${max_parallel}" ]]; then
    array+="%${max_parallel}"
fi
topologies=(one_node_one_gpu one_node_two_gpu one_node_four_gpu two_nodes_four_gpus)
indices=("0,1,2,12,13,14" "3,4,5,15,16,17" "6,7,8,18,19,20" "9,10,11,21,22,23")
nodes=(1 1 1 2)
gpus_per_node=(1 2 4 2)
cpus=(72 144 288 144)
memory=(217086M 434172M 868344M 434172M)
job_ids=()
for position in 0 1 2 3; do
    topology="${topologies[position]}"
    logs="${work_dir}/slurm-logs/system_scaling_energy_evaluation/${topology}"
    mkdir -p "${logs}"
    job_id="$(sbatch --parsable --account="${CSC_PROJECT}" --chdir="${repo_root}" \
        --array="${array}" --nodes="${nodes[position]}" --ntasks="${nodes[position]}" \
        --ntasks-per-node=1 --gres="gpu:gh200:${gpus_per_node[position]}" \
        --cpus-per-task="${cpus[position]}" --mem="${memory[position]}" \
        --output="${logs}/%A_%a.out" --error="${logs}/%A_%a.err" \
        --export="ALL,WORK_DIR=${work_dir},REPO_ROOT=${repo_root},EVALUATION_MANIFEST=${manifest},SCALING_TOPOLOGY=${topology},SCALING_ALLOCATION_INDICES=${indices[position]}" \
        "${script_dir}/system_scaling_energy.sbatch")"
    job_id="${job_id%%;*}"
    [[ "${job_id}" =~ ^[0-9]+$ ]] || { echo "Slurm returned an invalid job identifier" >&2; exit 2; }
    job_ids+=("${job_id}")
    echo "${topology}_job_id=${job_id}"
done
echo "scientific_execution_count=24"
echo "slurm_allocation_count=24"
echo "partition=gpumedium"
echo "submission_job_ids=$(IFS=,; printf '%s' "${job_ids[*]}")"
