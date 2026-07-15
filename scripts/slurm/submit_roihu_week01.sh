#!/usr/bin/env bash
set -euo pipefail
partition=""; config=""; work_dir=""; duration=""; dry_run=0; allow_smoke=0
while (($#)); do
  case "$1" in
    --partition) partition="${2:?}"; shift 2;;
    --config) config="${2:?}"; shift 2;;
    --work-dir) work_dir="${2:?}"; shift 2;;
    --time) duration="${2:?}"; shift 2;;
    --dry-run) dry_run=1; shift;;
    --allow-smoke-gpumedium) allow_smoke=1; shift;;
    *) echo "unknown argument: $1" >&2; exit 2;;
  esac
done
: "${CSC_PROJECT:?set CSC_PROJECT to the CSC project account}"
[[ "$partition" == gputest || "$partition" == gpumedium ]] || { echo "partition must be gputest or gpumedium" >&2; exit 2; }
[[ "$work_dir" == "/scratch/$CSC_PROJECT/"* ]] || { echo "--work-dir must be under /scratch/$CSC_PROJECT" >&2; exit 2; }
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
[[ -f "$config" ]] || { echo "configuration does not exist: $config" >&2; exit 2; }
config="$(realpath "$config")"
read -r mode dataset < <(PYTHONPATH="$repo/src" python3 -c 'import sys; from fedapfa.configuration import load_config; c=load_config(sys.argv[1]); print(c["mode"],c["dataset"]["name"])' "$config")
[[ "$partition" != gputest || "$mode" != full ]] || { echo "full training is forbidden on gputest" >&2; exit 2; }
[[ "$partition" != gpumedium || "$mode" != smoke || "$allow_smoke" == 1 ]] || { echo "smoke runs require --allow-smoke-gpumedium on gpumedium" >&2; exit 2; }
if [[ "$partition" == gputest ]]; then
  duration="00:15:00"; script="$repo/scripts/slurm/roihu_week01_gputest.sbatch"
else
  duration="${duration:-36:00:00}"; script="$repo/scripts/slurm/roihu_week01_gpumedium.sbatch"
  if [[ ! "$duration" =~ ^([0-9]+):([0-9]{2}):([0-9]{2})$ ]]; then
    echo "--time must use HH:MM:SS" >&2; exit 2
  fi
  hours="${BASH_REMATCH[1]}"; minutes="${BASH_REMATCH[2]}"; seconds="${BASH_REMATCH[3]}"
  ((10#$minutes < 60 && 10#$seconds < 60 && 10#$hours * 3600 + 10#$minutes * 60 + 10#$seconds <= 36 * 3600)) || {
    echo "gpumedium time exceeds 36 hours or is invalid" >&2; exit 2
  }
fi
scratch="$work_dir"; logs="$scratch/slurm-logs"; mkdir -p "$logs" "$scratch/runs"
data_root="$scratch/data/$dataset"; output_root="$scratch/runs"
command=(sbatch --account="$CSC_PROJECT" --partition="$partition" --time="$duration" --chdir="$repo" --output="$logs/%x-%j.out" --export="ALL,CONFIG=$config,DATA_ROOT=$data_root,OUTPUT_ROOT=$output_root" "$script")
printf 'Final command:'; printf ' %q' "${command[@]}"; printf '\n'
((dry_run)) || "${command[@]}"
