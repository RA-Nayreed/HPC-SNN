#!/usr/bin/env bash
set -euo pipefail
: "${CSC_PROJECT:?set CSC_PROJECT}"; repo="${1:-$(pwd)}"; venv="${2:-/projappl/$CSC_PROJECT/hpc-snn-venv}"
module purge
module load python-pytorch/2.10
python3 -m venv --system-site-packages "$venv"
source "$venv/bin/activate"
python3 -m pip install --upgrade pip
python3 -m pip install --no-deps -e "$repo"
python3 -m pip install "numpy>=1.24" "h5py>=3.9" "pyyaml>=6" "matplotlib" "scipy" "tqdm" \
  "nvidia-ml-py>=12.535.133" "pytest>=8" "ruff>=0.6"
python3 -m pip install --no-deps "spikingjelly==0.0.0.0.14"
if [[ "${INSTALL_DCLS:-0}" == 1 ]]; then
  python3 -m pip install --no-deps "dcls==0.1.1"
  python3 - <<'PY'
import torch
from DCLS.construct.modules import Dcls1d
layer = Dcls1d(2, 3, kernel_count=1, dilated_kernel_size=5, padding=2, bias=False)
x = torch.randn(2, 2, 8, requires_grad=True)
layer(x).sum().backward()
assert layer.P.grad is not None and torch.isfinite(layer.P.grad).all()
print("DCLS 0.1.1 forward/backward probe passed")
PY
fi
python3 - <<'PY'
import platform,h5py,matplotlib,numpy,pynvml,scipy,torch,torchvision,tqdm,yaml
import spikingjelly
print(platform.machine(),torch.__version__,h5py.__version__,numpy.__version__,yaml.__version__)
print("nvml_binding", getattr(pynvml, "__version__", "available"))
print((torch.tensor([1.0])+1).item())
PY
