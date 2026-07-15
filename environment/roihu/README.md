# Roihu centralized execution

Load `python-pytorch/2.10`; it already supplies CUDA and cuDNN. Keep shared environments under `/projappl/$CSC_PROJECT` and datasets, checkpoints, runs, and logs under `/scratch/$CSC_PROJECT/$USER`. GPU diagnostics are performed only inside the submitted allocation.

DCLS is optional. Set `INSTALL_DCLS=1` when running the setup script to install the pinned `dcls==0.1.1` source distribution without replacing CSC PyTorch and run an immediate CPU forward/backward probe. The package passed that probe on local x86_64 CPU with PyTorch 2.13, but Grace aarch64 with CSC PyTorch 2.10/CUDA 13 remains unverified until this probe and a smoke run execute inside Roihu.

```bash
scripts/roihu/setup_environment.sh "$PWD"
export ROIHU_WORKDIR="/scratch/$CSC_PROJECT/$USER"
scripts/slurm/submit_roihu_week01.sh --partition gputest --config experiments/week01_pfa_reproduction/02_plain_lif_shd.yaml --work-dir "$ROIHU_WORKDIR" --dry-run
```
