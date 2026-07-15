#!/usr/bin/env bash
set -euo pipefail
root="${1:-data/raw/ssc}"
fedapfa-inspect-data "$root/ssc_train.h5"
fedapfa-inspect-data "$root/ssc_valid.h5"
fedapfa-inspect-data "$root/ssc_test.h5"
