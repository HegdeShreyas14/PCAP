#!/usr/bin/env bash
# Remaining runs after the 2026-10-11 reboot (NVIDIA driver updated mid-session):
#   1. lambda boundary: lambda 0.01-0.04 for cells whose best-F1 lambda sat at
#      the grid edge, per seed, at 1 and 8 threads (new files only)
#   2. GPU crossover seeds 1-2, float64 and float32
#   3. seed-0 GPU crossover rerun on the new driver (compare with committed data)
#   4. run-package smoke test, then ENVIRONMENT.md refresh
# Strictly sequential; stops the author's unrelated container and restarts it on exit.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python; L=results/runlogs; LOG=$L/run_after_reboot.log
nvidia-smi >/dev/null 2>&1 || { echo "nvidia-smi fails (driver/library mismatch?): reboot first" >&2; exit 1; }
if [ -n "$(git status --porcelain --untracked-files=no -- src)" ]; then echo "commit src/ first" >&2; exit 1; fi
trap 'docker start trade_plat-matching-engine-1 >/dev/null 2>&1 && echo "[$(date -Is)] restarted trade_plat container" >> "$LOG"' EXIT
docker stop trade_plat-matching-engine-1 >/dev/null 2>&1 || true
sleep 15
{ echo "recorded $(date -Is) before run_after_reboot.sh"; echo "power_profile $(powerprofilesctl get)"
  echo "governor $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)"
  echo "stopped: docker trade_plat-matching-engine-1 (restarted on exit)"
  nvidia-smi --query-gpu=name,driver_version,temperature.gpu,utilization.gpu,memory.used --format=csv
  echo "loadavg $(cat /proc/loadavg)"; } > "$L/machine_state_after_reboot.txt"
echo "commit $(git rev-parse HEAD) started $(date -Is)" | tee "$LOG"
FAILED=()
step() { local n=$1; shift; echo "[$(date -Is)] START $n" | tee -a "$LOG"
         if "$@" > "$L/$n.log" 2>&1; then echo "[$(date -Is)] DONE  $n" | tee -a "$LOG"
         else echo "[$(date -Is)] FAILED $n" | tee -a "$LOG"; FAILED+=("$n"); fi; }
step determinism $PY tests/test_sampling_determinism.py

# 1. lambda boundary
mkdir -p results/lambda_boundary
for s in 0 1 2; do
  if [ $s = 0 ]; then SRC=results/rq1_primary_8t.csv; else SRC=results/seeds/rq1_primary_8t_seed$s.csv; fi
  while read -r reg p ratio; do
    for t in 1 8; do
      n=lam_ext_seed${s}_${reg}_p${p}_r${ratio}_${t}t
      step "$n" $PY src/sweep.py --machine primary --seed $s --blas-threads $t --regimes "$reg" --ps "$p" \
          --ratios "$ratio" --lambdas 0.01 0.02 0.03 0.04 --out "results/lambda_boundary/$n.csv"
    done
  done < <($PY scripts/provenance/edge_cells.py "$SRC")
done

# 2. GPU seeds 1-2
for s in 1 2; do
  step crossover_fp64_seed$s $PY src/gpu_crossover.py --p 1200 2000 --regime many_tiny --seed $s --out results/seeds/gpu_crossover_seed$s.csv
  step crossover_fp32_seed$s $PY src/gpu_crossover.py --p 1200 2000 --regime many_tiny --seed $s --dtype float32 --out results/seeds/gpu_crossover_fp32_seed$s.csv
done

# 3. seed-0 rerun on the new driver
mkdir -p results/driver_recheck
step recheck_fp64_seed0 $PY src/gpu_crossover.py --p 1200 2000 --regime many_tiny --seed 0 --out results/driver_recheck/gpu_crossover_seed0.csv
step recheck_fp32_seed0 $PY src/gpu_crossover.py --p 1200 2000 --regime many_tiny --seed 0 --dtype float32 --out results/driver_recheck/gpu_crossover_fp32_seed0.csv

# 4. run-package smoke test (output not committed) and environment refresh
step package_smoke scripts/run_package.sh --machine primary-smoke --tier smoke --yes
step capture_environment $PY scripts/capture_environment.py

if [ ${#FAILED[@]} -gt 0 ]; then echo "[$(date -Is)] FINISHED WITH FAILURES: ${FAILED[*]}" | tee -a "$LOG"
else echo "[$(date -Is)] ALL DONE" | tee -a "$LOG"; fi
