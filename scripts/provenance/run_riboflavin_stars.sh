#!/usr/bin/env bash
# Riboflavin StARS at p=200 and 500, after the structure study. No timings.
set -euo pipefail
cd "$(dirname "$0")/../.."
scripts/provenance/run_riboflavin.sh
.venv/bin/python src/riboflavin_stars.py --ps 200 500 > results/runlogs/riboflavin_stars.log 2>&1
echo "ALL DONE" >> results/runlogs/riboflavin_stars.log
