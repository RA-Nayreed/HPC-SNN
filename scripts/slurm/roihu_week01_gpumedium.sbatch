#!/usr/bin/env bash
#SBATCH --partition=gpumedium
#SBATCH --time=36:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:gh200:1
set -euo pipefail
: "${CONFIG:?CONFIG is required}" "${DATA_ROOT:?DATA_ROOT is required}" "${OUTPUT_ROOT:?OUTPUT_ROOT is required}"
module purge
module load python-pytorch/2.10
VENV="${VENV:-/projappl/${CSC_PROJECT}/${USER}/hpc-snn-venv}"
source "$VENV/bin/activate"

echo "Python: $(command -v python3)"
python3 -c "import torch, fedapfa, DCLS; print(torch.__version__, fedapfa.__file__, DCLS.__file__)"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
hostname
uname -m
module list
nvidia-smi
python3 --version
python3 - <<'PY'
import torch
print("PyTorch",torch.__version__)
print("CUDA runtime",torch.version.cuda)
print("CUDA available",torch.cuda.is_available())
if not torch.cuda.is_available(): raise SystemExit("CUDA is unavailable inside the allocation")
if torch.cuda.device_count()!=1: raise SystemExit(f"expected exactly one visible GPU, found {torch.cuda.device_count()}")
print("GPU",torch.cuda.get_device_name(0))
PY
python3 - <<'PY' "$CONFIG"
import sys
from fedapfa.configuration import load_config
if load_config(sys.argv[1])["device"]!="cuda": raise SystemExit("Roihu GPU jobs require device=cuda")
PY
srun python3 -m fedapfa.cli.train_centralized "$CONFIG" --data-root "$DATA_ROOT" --output-root "$OUTPUT_ROOT" --device cuda
