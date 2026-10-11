#!/usr/bin/env bash
# Like-for-like float32 CPU vs GPU timing. Restarts the stopped container on exit.
set -euo pipefail
cd "$(dirname "$0")/../.."
L=results/runlogs
trap 'docker start trade_plat-matching-engine-1 >/dev/null 2>&1 && echo "[$(date -Is)] restarted trade_plat container" >> "$L/fp32_cpu_baseline.log"' EXIT
if [ -n "$(git status --porcelain --untracked-files=no -- src)" ]; then echo "commit src/ first" >&2; exit 1; fi
docker stop trade_plat-matching-engine-1 >/dev/null 2>&1 || true
sleep 15
{ echo "recorded $(date -Is) before run_fp32_cpu_baseline.sh"; echo "power_profile $(powerprofilesctl get)"; echo "governor $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)"; echo "stopped: docker trade_plat-matching-engine-1 (restarted on exit)"; nvidia-smi --query-gpu=name,driver_version,temperature.gpu,utilization.gpu,memory.used --format=csv; echo "loadavg $(cat /proc/loadavg)"; } > "$L/machine_state_fp32_cpu_baseline.txt"
echo "commit $(git rev-parse HEAD) started $(date -Is)" > "$L/fp32_cpu_baseline.log"
.venv/bin/python src/fp32_cpu_baseline.py --blas-threads 8 >> "$L/fp32_cpu_baseline.log" 2>&1
echo "[$(date -Is)] done" >> "$L/fp32_cpu_baseline.log"
