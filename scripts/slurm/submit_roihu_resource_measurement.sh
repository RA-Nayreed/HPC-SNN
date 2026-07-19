#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 --work-dir PATH" >&2
}

work_dir=""
while (($#)); do
    case "$1" in
        --work-dir) [[ $# -ge 2 ]] || { usage; exit 2; }; work_dir="$2"; shift 2 ;;
        *) usage; exit 2 ;;
    esac
done

: "${CSC_PROJECT:?CSC_PROJECT must be set}"
: "${USER:?USER must be set}"
[[ -n "${work_dir}" ]] || { usage; exit 2; }
work_dir="$(realpath -m "${work_dir}")"
scratch_prefix="/scratch/${CSC_PROJECT}/${USER}/"
case "${work_dir}/" in
    "${scratch_prefix}"*) ;;
    *) echo "work directory must be under ${scratch_prefix}" >&2; exit 2 ;;
esac

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
manifest="${repo_root}/experiments/resource_measurement/manifest.yaml"
batch_script="${script_dir}/resource_measurement.sbatch"
logs_root="${work_dir}/slurm-logs/resource_measurement"
mkdir -p "${logs_root}" "${work_dir}/runs/resource_measurement" "${work_dir}/results/resource_measurement"

module load python-pytorch/2.10
venv="${FEDAPFA_VENV:-/projappl/${CSC_PROJECT}/${USER}/hpc-snn-venv}"
[[ -x "${venv}/bin/python3" ]] || { echo "project Python interpreter is unavailable" >&2; exit 2; }
python_bin="${venv}/bin/python3"
"${python_bin}" -m fedapfa.cli.scientific_manifest validate --manifest "${manifest}" >/dev/null
task_count="$("${python_bin}" -m fedapfa.cli.scientific_manifest count --manifest "${manifest}")"
[[ "${task_count}" == "6" ]] || { echo "resource manifest must contain six tasks" >&2; exit 2; }

job_id="$(sbatch --parsable \
    --account="${CSC_PROJECT}" --chdir="${repo_root}" \
    --output="${logs_root}/%j.out" --error="${logs_root}/%j.err" \
    --export="ALL,WORK_DIR=${work_dir},REPO_ROOT=${repo_root},RESOURCE_MANIFEST=${manifest}" \
    "${batch_script}")"
job_id="${job_id%%;*}"
[[ "${job_id}" =~ ^[0-9]+$ ]] || { echo "Slurm returned an invalid job identifier" >&2; exit 2; }
printf '%s\n' "${job_id}"
