#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 --work-dir PATH [--collection distributed_evaluation|device_capacity_evaluation] [--datasets shd,ssc,cifar10] [--device-counts 1,2,4] [--max-parallel N]" >&2
}
work_dir=""
collection="distributed_evaluation"
datasets="shd,ssc,cifar10"
device_counts="1,2,4"
max_parallel=1
while (($#)); do
    case "$1" in
        --work-dir) [[ $# -ge 2 ]] || { usage; exit 2; }; work_dir="$2"; shift 2 ;;
        --collection) [[ $# -ge 2 ]] || { usage; exit 2; }; collection="$2"; shift 2 ;;
        --datasets) [[ $# -ge 2 ]] || { usage; exit 2; }; datasets="$2"; shift 2 ;;
        --device-counts) [[ $# -ge 2 ]] || { usage; exit 2; }; device_counts="$2"; shift 2 ;;
        --max-parallel) [[ $# -ge 2 ]] || { usage; exit 2; }; max_parallel="$2"; shift 2 ;;
        *) usage; exit 2 ;;
    esac
done

: "${CSC_PROJECT:?CSC_PROJECT must be set}"
: "${USER:?USER must be set}"
[[ -n "${work_dir}" ]] || { usage; exit 2; }
[[ "${max_parallel}" =~ ^[1-9][0-9]*$ ]] || { echo "--max-parallel must be positive" >&2; exit 2; }
[[ "${datasets}" =~ ^(shd|ssc|cifar10)(,(shd|ssc|cifar10))*$ ]] || { echo "--datasets is incompatible" >&2; exit 2; }
[[ "${device_counts}" =~ ^(1|2|4)(,(1|2|4))*$ ]] || { echo "--device-counts is incompatible" >&2; exit 2; }
work_dir="$(realpath -m "${work_dir}")"
scratch_prefix="/scratch/${CSC_PROJECT}/${USER}/"
case "${work_dir}/" in
    "${scratch_prefix}"*) ;;
    *) echo "work directory must be under ${scratch_prefix}" >&2; exit 2 ;;
esac

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
case "${collection}" in
    distributed_evaluation) expected_task_count=24 ;;
    device_capacity_evaluation) expected_task_count=9 ;;
    *) echo "--collection is incompatible" >&2; exit 2 ;;
esac
manifest="${repo_root}/experiments/${collection}/manifest.yaml"
array_script="${script_dir}/federated_multigpu.sbatch"
logs_root="${work_dir}/slurm-logs/${collection}"
mkdir -p "${logs_root}" "${work_dir}/runs/${collection}" \
    "${work_dir}/results/${collection}" "${work_dir}/telemetry/${collection}"

module load python-pytorch/2.10
venv="${FEDAPFA_VENV:-/projappl/${CSC_PROJECT}/${USER}/hpc-snn-venv}"
[[ -f "${venv}/bin/activate" ]] || { echo "missing Roihu virtual environment: ${venv}" >&2; exit 2; }
source "${venv}/bin/activate"
python3 -m fedapfa.cli.scientific_manifest validate --manifest "${manifest}"
task_count="$(python3 -m fedapfa.cli.scientific_manifest count --manifest "${manifest}")"
[[ "${task_count}" == "${expected_task_count}" ]] || {
    echo "selected manifest has an incompatible task count" >&2; exit 2;
}

dataset_selected() { [[ ",${datasets}," == *",$1,"* ]]; }
device_selected() { [[ ",${device_counts}," == *",$1,"* ]]; }
declare -A indices=( [1]="" [2]="" [4]="" )
for ((index = 0; index < task_count; index++)); do
    task="$(python3 -m fedapfa.cli.scientific_manifest task --manifest "${manifest}" --index "${index}")"
    IFS=$'\t' read -r _ _ dataset _ _ _ _ device_count _ _ _ _ _ <<< "${task}"
    if dataset_selected "${dataset}" && device_selected "${device_count}"; then
        if [[ -n "${indices[${device_count}]}" ]]; then
            indices[${device_count}]+=","
        fi
        indices[${device_count}]+="${index}"
    fi
done

submitted=0
for device_count in 1 2 4; do
    [[ -n "${indices[${device_count}]}" ]] || continue
    group_logs="${logs_root}/${device_count}_gpu"
    mkdir -p "${group_logs}"
    job_id="$(sbatch --parsable \
        --account="${CSC_PROJECT}" --chdir="${repo_root}" \
        --gres="gpu:gh200:${device_count}" --cpus-per-task="$((72 * device_count))" \
        --array="${indices[${device_count}]}%${max_parallel}" \
        --output="${group_logs}/%A_%a.out" --error="${group_logs}/%A_%a.err" \
        --export="ALL,WORK_DIR=${work_dir},REPO_ROOT=${repo_root},DISTRIBUTED_MANIFEST=${manifest},EXPECTED_DEVICE_COUNT=${device_count}" \
        "${array_script}")"
    job_id="${job_id%%;*}"
    case "${device_count}" in
        1) echo "one_gpu_job_id=${job_id}" ;;
        2) echo "two_gpu_job_id=${job_id}" ;;
        4) echo "four_gpu_job_id=${job_id}" ;;
    esac
    submitted=$((submitted + 1))
done
[[ "${submitted}" -gt 0 ]] || { echo "requested filters select no manifest tasks" >&2; exit 2; }
