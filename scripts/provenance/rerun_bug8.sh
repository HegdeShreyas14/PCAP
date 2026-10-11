#!/usr/bin/env bash
# Regenerate the RQ1 pair and percolation after the bug-8 sampler fix.
# Strictly sequential (runbook: never run two sweeps concurrently).
# Old datasets stay recoverable from git history (commit 96f4249).
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
L=results/runlogs
mkdir -p "$L"

if [ -n "$(git status --porcelain --untracked-files=no -- src)" ]; then
  echo "src/ has uncommitted changes; commit first so the datasets are attributable." >&2
  exit 1
fi
echo "commit $(git rev-parse HEAD)  started $(date -Is)" | tee "$L/rerun_bug8.log"

$PY tests/test_sampling_determinism.py 2>&1 | tee -a "$L/rerun_bug8.log"

step() {  # name, command...
  local name=$1; shift
  echo "[$(date -Is)] START $name" | tee -a "$L/rerun_bug8.log"
  "$@" > "$L/$name.log" 2>&1
  echo "[$(date -Is)] DONE  $name" | tee -a "$L/rerun_bug8.log"
}

step rq1_primary_1t $PY src/sweep.py --machine primary --blas-threads 1 --out results/rq1_primary_1t.csv
step rq1_primary_8t $PY src/sweep.py --machine primary --blas-threads 8 --out results/rq1_primary_8t.csv
step rq1_compare    $PY src/analyze.py --compare results/rq1_primary_1t.csv results/rq1_primary_8t.csv
grep "pairing check" "$L/rq1_compare.log" | tee -a "$L/rerun_bug8.log"
step percolation    $PY src/percolation.py --seeds 0 1 2 --blas-threads 8 --out results/percolation.csv

echo "[$(date -Is)] ALL DONE" | tee -a "$L/rerun_bug8.log"
