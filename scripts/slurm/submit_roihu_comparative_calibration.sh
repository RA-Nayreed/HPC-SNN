#!/usr/bin/env bash

set -euo pipefail
: "${CSC_PROJECT:?CSC_PROJECT must be set}"
: "${USER:?USER must be set}"
: "${WORK_DIR:?WORK_DIR must be exported}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export REPO_ROOT="${repo_root}"
script="${repo_root}/scripts/slurm/comparative_measurement_calibration.sbatch"

topologies=(one_node_one_gpu one_node_two_gpu one_node_four_gpu two_nodes_four_gpus)
nodes=(1 1 1 2)
gpus=(1 2 4 2)
cpus=(72 144 288 144)
memory=(217086M 434172M 868344M 434172M)
configs=(
    experiments/system_scaling_energy_evaluation/shd/lif_one_node_one_gpu.yaml
    experiments/system_scaling_energy_evaluation/shd/lif_one_node_two_gpu.yaml
    experiments/system_scaling_energy_evaluation/shd/lif_one_node_four_gpu.yaml
    experiments/system_scaling_energy_evaluation/shd/lif_two_nodes_four_gpus.yaml
)

for index in 0 1 2 3; do
    sbatch \
        --account="${CSC_PROJECT}" \
        --nodes="${nodes[index]}" \
        --ntasks="${nodes[index]}" \
        --ntasks-per-node=1 \
        --gres="gpu:gh200:${gpus[index]}" \
        --cpus-per-task="${cpus[index]}" \
        --mem="${memory[index]}" \
        --export=ALL,REPO_ROOT="${repo_root}",SCALING_TOPOLOGY="${topologies[index]}",CALIBRATION_CONFIG="${repo_root}/${configs[index]}" \
        "${script}"
done
