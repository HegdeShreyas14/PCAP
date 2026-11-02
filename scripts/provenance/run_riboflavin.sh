#!/usr/bin/env bash
# Riboflavin structure study: main run plus batch sensitivity. No timings.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
L=results/runlogs/riboflavin_study.log; : > "$L"
mkdir -p results/riboflavin_sensitivity; O=results/riboflavin_sensitivity
$PY src/riboflavin_study.py >> "$L" 2>&1
$PY src/riboflavin_study.py --sub-batches 22 --exact-ps --out-prefix $O/all_batchsub >> "$L" 2>&1
$PY src/riboflavin_study.py --subset pt24 --sub-batches 22 --exact-ps --out-prefix $O/pt24 >> "$L" 2>&1
$PY src/riboflavin_study.py --subset pt48 --sub-batches 22 --exact-ps --out-prefix $O/pt48 >> "$L" 2>&1
$PY src/riboflavin_study.py --subset within_batch --sub-batches 22 --exact-ps --out-prefix $O/within_batch >> "$L" 2>&1
echo "ALL DONE" >> "$L"
