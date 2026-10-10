#!/usr/bin/env bash
# Regenerate RQ2/RQ3 on the bug-8-fixed sampler, with the same commands that
# produced the originals (reconstructed from their CSVs). Strictly sequential.
# Restarts the stopped trade_plat container on exit, success or failure.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
L=results/runlogs
mkdir -p "$L"
trap 'docker start trade_plat-matching-engine-1 >/dev/null 2>&1 && echo "[$(date -Is)] restarted trade_plat container" >> "$L/rerun_rq2_rq3.log"' EXIT

if [ -n "$(git status --porcelain --untracked-files=no -- src)" ]; then
  echo "src/ has uncommitted changes; commit first." >&2; exit 1
fi
echo "commit $(git rev-parse HEAD)  started $(date -Is)" | tee "$L/rerun_rq2_rq3.log"
$PY tests/test_sampling_determinism.py 2>&1 | tee -a "$L/rerun_rq2_rq3.log"

step() {
  local name=$1; shift
  echo "[$(date -Is)] START $name" | tee -a "$L/rerun_rq2_rq3.log"
  "$@" > "$L/$name.log" 2>&1
  echo "[$(date -Is)] DONE  $name" | tee -a "$L/rerun_rq2_rq3.log"
}

step batched_cpu     $PY src/bench_batched.py --backends numpy --out results/batched_cpu.csv
step batched_cpu_gpu $PY src/bench_batched.py --backends numpy torch --out results/batched_cpu_gpu.csv
step gpu_crossover   $PY src/gpu_crossover.py --p 1200 2000 --regime many_tiny --out results/gpu_crossover.csv
step gpu_mem_2400    $PY src/gpu_crossover.py --p 2400 --regime many_tiny --lam 0.08 --out results/gpu_mem_2400.csv
step gpu_mem_3200    $PY src/gpu_crossover.py --p 3200 --regime many_tiny --lam 0.06 --out results/gpu_mem_3200.csv
echo "[$(date -Is)] ALL DONE" | tee -a "$L/rerun_rq2_rq3.log"
