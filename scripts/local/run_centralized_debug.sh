#!/usr/bin/env bash
set -euo pipefail
fedapfa-train-centralized experiments/week01_pfa_reproduction/02_plain_lif_shd.yaml --device cpu "$@"
