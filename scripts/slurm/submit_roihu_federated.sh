#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 --work-dir PATH [--max-parallel N]" >&2
}

work_dir=""
max_parallel=1
while (($#)); do
    case "$1" in
        --work-dir)
            [[ $# -ge 2 ]] || { usage; exit 2; }
            work_dir="$2"
            shift 2
            ;;
        --max-parallel)
            [[ $# -ge 2 ]] || { usage; exit 2; }
            max_parallel="$2"
            shift 2
            ;;
        *)
            usage
            exit 2
            ;;
    esac
done

: "${CSC_PROJECT:?CSC_PROJECT must be set}"
: "${USER:?USER must be set}"
[[ -n "${work_dir}" ]] || { usage; exit 2; }
[[ "${max_parallel}" =~ ^[1-6]$ ]] || { echo "--max-parallel must be an integer from 1 through 6" >&2; exit 2; }

work_dir="$(realpath -m "${work_dir}")"
scratch_prefix="/scratch/${CSC_PROJECT}/"
case "${work_dir}/" in
    "${scratch_prefix}"*) ;;
    *) echo "WORK_DIR must be under ${scratch_prefix}" >&2; exit 2 ;;
esac

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
manifest="${repo_root}/experiments/federated_baselines/manifest.yaml"
array_script="${script_dir}/fedavg_single_gpu.sbatch"
logs_root="${work_dir}/slurm-logs/federated"
mkdir -p "${logs_root}" "${work_dir}/runs/federated" "${work_dir}/results/federated" \
    "${work_dir}/telemetry/federated"

module load python-pytorch/2.10
venv="${FEDAPFA_VENV:-/projappl/${CSC_PROJECT}/${USER}/hpc-snn-venv}"
[[ -f "${venv}/bin/activate" ]] || { echo "missing Roihu virtual environment: ${venv}" >&2; exit 2; }
source "${venv}/bin/activate"
python3 -m fedapfa.cli.federated_manifest validate --manifest "${manifest}"
task_count="$(python3 -m fedapfa.cli.federated_manifest count --manifest "${manifest}")"
[[ "${task_count}" == "6" ]] || { echo "federated manifest must expand to six tasks" >&2; exit 2; }

job_id="$(sbatch --parsable \
    --account="${CSC_PROJECT}" \
    --chdir="${repo_root}" \
    --array="0-$((task_count - 1))%${max_parallel}" \
    --output="${logs_root}/%A_%a.out" \
    --error="${logs_root}/%A_%a.err" \
    --export="ALL,WORK_DIR=${work_dir},REPO_ROOT=${repo_root},FEDERATED_MANIFEST=${manifest}" \
    "${array_script}")"
job_id="${job_id%%;*}"
echo "Submitted job ID: ${job_id}"
