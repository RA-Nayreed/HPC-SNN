#!/usr/bin/env bash
#SBATCH --job-name=hpc-snn-week01
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/week01-%j.out
# Set account, partition, time, and modules for this site before submission.
set -euo pipefail
CONFIG="${1:?supply an experiment YAML path}"
hostname
nvidia-smi
python --version
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
fedapfa-train-centralized "$CONFIG"
