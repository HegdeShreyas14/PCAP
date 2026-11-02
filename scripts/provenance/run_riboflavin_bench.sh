#!/usr/bin/env bash
# Riboflavin timings: quiet machine, container stopped, CPU use logged throughout.
set -uo pipefail
cd "$(dirname "$0")/../.."
L=results/runlogs; LOG=$L/riboflavin_bench.log; CPU=$L/riboflavin_bench_cpu.log
MON=""
cleanup() { [ -n "$MON" ] && kill "$MON" 2>/dev/null; docker start trade_plat-matching-engine-1 >/dev/null 2>&1 && echo "[$(date -Is)] restarted trade_plat container" >> "$LOG"; }
trap cleanup EXIT
: > "$LOG"; : > "$CPU"
docker stop trade_plat-matching-engine-1 >/dev/null 2>&1 || true
busy() { read -r _ a b c d e f g h _ < /proc/stat; t1=$((a+b+c+d+e+f+g+h)); i1=$((d+e)); sleep "$1"; read -r _ a b c d e f g h _ < /proc/stat; t2=$((a+b+c+d+e+f+g+h)); i2=$((d+e)); echo $(( ($(nproc)*100*((t2-t1)-(i2-i1)))/(t2-t1) )); }
quiet=0
for _ in $(seq 1 90); do
  b=$(busy 10); echo "[$(date -Is)] waiting: busy ${b}% of one core" >> "$LOG"
  if [ "$b" -lt 150 ]; then quiet=$((quiet+1)); else quiet=0; fi
  [ "$quiet" -ge 6 ] && break
done
[ "$quiet" -ge 6 ] || { echo "[$(date -Is)] ABORT: machine never went quiet" >> "$LOG"; exit 3; }
{ echo "recorded $(date -Is) before run_riboflavin_bench.sh"; echo "power_profile $(powerprofilesctl get)"; echo "governor $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)"; echo "kernel $(uname -r)"; echo "stopped: docker trade_plat-matching-engine-1 (restarted on exit)"; nvidia-smi --query-gpu=name,driver_version,temperature.gpu,utilization.gpu,memory.used --format=csv; echo "loadavg $(cat /proc/loadavg)"; echo "docker running: $(docker ps --format '{{.Names}}')"; } > "$L/machine_state_riboflavin_bench.txt"
( while :; do echo "$(date -Is) $(ps -eo %cpu,comm --sort=-%cpu --no-headers | head -4 | tr -s ' ' | tr '\n' ';')" >> "$CPU"; sleep 30; done ) &
MON=$!
.venv/bin/python src/riboflavin_bench.py --blas-threads 8 >> "$LOG" 2>&1
s=$?; echo "[$(date -Is)] finished status $s" >> "$LOG"; exit $s
