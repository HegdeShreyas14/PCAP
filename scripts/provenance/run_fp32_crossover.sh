#!/usr/bin/env bash
# float32 GPU crossover + VRAM anchors (same points as the float64 runs).
# Only run after fp32_diagnostic.py passes. Restarts the stopped container on exit.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python; L=results/runlogs
trap 'docker start trade_plat-matching-engine-1 >/dev/null 2>&1 && echo "[$(date -Is)] restarted trade_plat container" >> "$L/run_fp32_crossover.log"' EXIT
if [ -n "$(git status --porcelain --untracked-files=no -- src)" ]; then echo "commit src/ first" >&2; exit 1; fi
docker stop trade_plat-matching-engine-1 >/dev/null 2>&1 || true
sleep 15
{ echo "recorded $(date -Is) before run_fp32_crossover.sh"; echo "power_profile $(powerprofilesctl get)"; echo "governor $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)"; echo "stopped: docker trade_plat-matching-engine-1 (restarted on exit)"; nvidia-smi --query-gpu=name,driver_version,temperature.gpu,utilization.gpu,memory.used --format=csv; echo "loadavg $(cat /proc/loadavg)"; } > "$L/machine_state_fp32_crossover.txt"
echo "commit $(git rev-parse HEAD) started $(date -Is)" | tee "$L/run_fp32_crossover.log"
step() { local n=$1; shift; echo "[$(date -Is)] START $n" | tee -a "$L/run_fp32_crossover.log"; "$@" > "$L/$n.log" 2>&1; echo "[$(date -Is)] DONE  $n" | tee -a "$L/run_fp32_crossover.log"; }
step gpu_crossover_fp32 $PY src/gpu_crossover.py --p 1200 2000 --regime many_tiny --dtype float32 --out results/gpu_crossover_fp32.csv
step gpu_mem_2400_fp32  $PY src/gpu_crossover.py --p 2400 --regime many_tiny --lam 0.08 --dtype float32 --out results/gpu_mem_2400_fp32.csv
step gpu_mem_3200_fp32  $PY src/gpu_crossover.py --p 3200 --regime many_tiny --lam 0.06 --dtype float32 --out results/gpu_mem_3200_fp32.csv
echo "[$(date -Is)] ALL DONE" | tee -a "$L/run_fp32_crossover.log"
