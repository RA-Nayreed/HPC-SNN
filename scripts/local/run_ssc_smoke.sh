#!/usr/bin/env bash
set -euo pipefail
fedapfa-train-centralized experiments/week01_pfa_reproduction/09_ssc_smoke.yaml --device cpu "$@"
