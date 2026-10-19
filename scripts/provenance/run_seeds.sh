#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python; L=results/runlogs; LOG=$L/run_seeds.log
trap 'docker start trade_plat-matching-engine-1 >/dev/null 2>&1 && echo "[$(date -Is)] restarted trade_plat container" >> "$LOG"' EXIT
if [ -n "$(git status --porcelain --untracked-files=no -- src)" ]; then echo "commit src/ first" >&2; exit 1; fi
docker stop trade_plat-matching-engine-1 >/dev/null 2>&1 || true
sleep 15
{ echo "recorded $(date -Is) before run_seeds.sh"; echo "power_profile $(powerprofilesctl get)"; echo "governor $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)"; echo "stopped: docker trade_plat-matching-engine-1 (restarted on exit)"; nvidia-smi --query-gpu=name,driver_version,temperature.gpu,utilization.gpu,memory.used --format=csv; echo "loadavg $(cat /proc/loadavg)"; } > "$L/machine_state_seeds.txt"
echo "commit $(git rev-parse HEAD) started $(date -Is)" | tee "$LOG"
$PY tests/test_sampling_determinism.py | tee -a "$LOG"
step() { local n=$1; shift; echo "[$(date -Is)] START $n" | tee -a "$LOG"; "$@" > "$L/$n.log" 2>&1; echo "[$(date -Is)] DONE  $n" | tee -a "$LOG"; }
for s in 1 2; do
  step rq1_1t_seed$s $PY src/sweep.py --machine primary --seed $s --blas-threads 1 --out results/seeds/rq1_primary_1t_seed$s.csv
  step rq1_8t_seed$s $PY src/sweep.py --machine primary --seed $s --blas-threads 8 --out results/seeds/rq1_primary_8t_seed$s.csv
  step rq1_compare_seed$s $PY src/analyze.py --compare results/seeds/rq1_primary_1t_seed$s.csv results/seeds/rq1_primary_8t_seed$s.csv
  grep "pairing check" "$L/rq1_compare_seed$s.log" | tee -a "$LOG"
done
for s in 1 2; do
  step crossover_fp64_seed$s $PY src/gpu_crossover.py --p 1200 2000 --regime many_tiny --seed $s --out results/seeds/gpu_crossover_seed$s.csv
  step crossover_fp32_seed$s $PY src/gpu_crossover.py --p 1200 2000 --regime many_tiny --seed $s --dtype float32 --out results/seeds/gpu_crossover_fp32_seed$s.csv
done
echo "[$(date -Is)] ALL DONE" | tee -a "$LOG"
