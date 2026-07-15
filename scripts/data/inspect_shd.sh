#!/usr/bin/env bash
set -euo pipefail
root="${1:-data/raw/shd}"
fedapfa-inspect-data "$root/shd_train.h5"
fedapfa-inspect-data "$root/shd_test.h5"
