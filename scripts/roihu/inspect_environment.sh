#!/usr/bin/env bash
set -euo pipefail
csc-workspaces
sinfo --summarize
scontrol show partition gputest
scontrol show partition gpumedium
module avail python-pytorch
module list
python3 --version
python3 -c 'import platform; print(platform.machine())'
