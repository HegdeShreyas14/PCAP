#!/usr/bin/env bash
# Cross-platform run package: reproduce the paper's experiments on another
# machine and bundle the results for comparison with the primary machine.
#
#   scripts/run_package.sh --machine NAME --tier smoke|standard|full [--yes]
#
#   smoke     ~10 min   setup check: determinism test, quick sweep, tiny GPU run
#   standard  ~3-5 h    RQ1 1t/8t pair, profiling, percolation, GPU crossover in
#                       float64 and float32, float32 diagnostic and CPU baseline
#   full      ~8-10 h   standard + seeds 1-2, batched CPU/GPU, VRAM anchors,
#                       screening exactness
#
# Output: results/machines/NAME/ and results/machines/NAME.tar.gz (+ .sha256).
# Send the .tar.gz back. Times are for the primary machine; slower machines
# take longer. GPU steps are skipped when CUDA is unavailable.
set -euo pipefail
cd "$(dirname "$0")/.."

MACHINE="" TIER="" YES=0
while [ $# -gt 0 ]; do
  case $1 in
    --machine) MACHINE=$2; shift 2 ;;
    --tier) TIER=$2; shift 2 ;;
    --yes) YES=1; shift ;;
    *) echo "unknown argument $1" >&2; exit 1 ;;
  esac
done
[[ "$MACHINE" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "--machine NAME (letters, digits, - or _) is required" >&2; exit 1; }
[[ "$TIER" =~ ^(smoke|standard|full)$ ]] || { echo "--tier smoke|standard|full is required" >&2; exit 1; }

PY=${PY:-.venv/bin/python}
[ -x "$PY" ] || { echo "$PY not found: run scripts/setup_env.sh first" >&2; exit 1; }
OUT=results/machines/$MACHINE
mkdir -p "$OUT/logs"
LOG=$OUT/logs/run_package.log

# ---------------------------------------------------------------- preflight
if git rev-parse --git-dir >/dev/null 2>&1; then
  if [ -n "$(git status --porcelain --untracked-files=no -- src)" ]; then
    echo "src/ has local changes; results would not match any commit. Revert them first." >&2; exit 1
  fi
else
  echo "!! not a git clone: provenance will lack a commit hash. Prefer 'git clone'." >&2
fi

PHYS=$("$PY" -c "import sys; sys.path.insert(0,'src'); from provenance import system_snapshot as s; print(s()['cpu_physical_cores'] or 8)")
THREADS=$(( PHYS < 8 ? PHYS : 8 ))      # never more BLAS threads than physical cores
HAS_GPU=$("$PY" -c "import torch; print(int(torch.cuda.is_available()))" 2>/dev/null || echo 0)

echo "== preflight =="
echo "machine label     : $MACHINE   tier: $TIER"
echo "physical cores    : $PHYS  -> multithreaded runs use $THREADS BLAS threads"
echo "CUDA GPU          : $([ "$HAS_GPU" = 1 ] && echo yes || echo 'no (GPU steps skipped)')"
echo "power profile     : $(powerprofilesctl get 2>/dev/null || echo unknown)"
echo "AC power          : $(cat /sys/class/power_supply/*/online 2>/dev/null | tr '\n' ' ' || echo unknown)"
echo "load average      : $(cut -d' ' -f1-3 /proc/loadavg 2>/dev/null || uptime)"
echo "top CPU processes :"; ps -eo pcpu,comm --sort=-pcpu 2>/dev/null | sed -n 2,6p | sed 's/^/    /'
command -v docker >/dev/null && { echo "docker containers :"; docker ps --format '    {{.Names}} ({{.Image}})' 2>/dev/null || true; }
cat <<'EOF'

Timed results are only valid on a quiet machine:
  - plugged in, performance power profile
  - browsers, IDEs, chat apps and unrelated containers/VMs closed
  - do not use the machine until the run finishes
EOF
if [ $YES -eq 0 ]; then
  read -r -p "Continue? [y/N] " a; [[ "$a" =~ ^[Yy]$ ]] || exit 1
fi

{ echo "recorded $(date -Is) before run_package.sh --tier $TIER"
  echo "machine_label $MACHINE"; echo "blas_threads_multi $THREADS"; echo "cuda_gpu $HAS_GPU"
  echo "power_profile $(powerprofilesctl get 2>/dev/null || echo unknown)"
  echo "governor $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo unknown)"
  echo "loadavg $(cat /proc/loadavg 2>/dev/null || uptime)"
  command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,driver_version,temperature.gpu,utilization.gpu,memory.used --format=csv
  echo "top processes:"; ps -eo pcpu,comm --sort=-pcpu 2>/dev/null | head -8
  command -v docker >/dev/null && { echo "docker:"; docker ps --format '{{.Names}}' 2>/dev/null || true; }
} > "$OUT/logs/machine_state.txt"

echo "commit $(git rev-parse HEAD 2>/dev/null || echo none) machine $MACHINE tier $TIER started $(date -Is)" | tee "$LOG"
"$PY" - "$OUT" <<'PYEOF'
import json, subprocess, sys
sys.path.insert(0, "src")
from provenance import run_metadata
out = sys.argv[1]
meta = run_metadata()
try:
    import torch
    meta["torch"] = {"version": torch.__version__, "cuda": torch.version.cuda,
                     "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
except ImportError:
    meta["torch"] = None
json.dump(meta, open(f"{out}/system.json", "w"), indent=2, default=str)
open(f"{out}/requirements-freeze.txt", "w").write(
    subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout)
PYEOF

# A failed step is logged and the run continues (later steps may depend on it
# and fail too); the summary at the end lists every failure.
FAILED=()
step() { local n=$1; shift; echo "[$(date -Is)] START $n" | tee -a "$LOG"
         if "$@" > "$OUT/logs/$n.log" 2>&1; then echo "[$(date -Is)] DONE  $n" | tee -a "$LOG"
         else echo "[$(date -Is)] FAILED $n (see logs/$n.log)" | tee -a "$LOG"; FAILED+=("$n"); fi; }
gpu() { if [ "$HAS_GPU" = 1 ]; then step "$@"
        else echo "[$(date -Is)] SKIP  $1 (no CUDA)" | tee -a "$LOG"; fi; }

step determinism "$PY" tests/test_sampling_determinism.py

if [ "$TIER" = smoke ]; then
  step sweep_quick "$PY" src/sweep.py --quick --machine "$MACHINE" --blas-threads 1 --out "$OUT/sweep_quick.csv"
  gpu  gpu_tiny_fp64 "$PY" src/gpu_crossover.py --p 300 --lam 0.1 0.3 --trials 2 --blas-threads "$THREADS" --out "$OUT/gpu_tiny_fp64.csv"
  gpu  gpu_tiny_fp32 "$PY" src/gpu_crossover.py --p 300 --lam 0.1 0.3 --trials 2 --blas-threads "$THREADS" --dtype float32 --out "$OUT/gpu_tiny_fp32.csv"
else
  step rq1_1t   "$PY" src/sweep.py --machine "$MACHINE" --blas-threads 1 --out "$OUT/rq1_1t.csv"
  step rq1_multi "$PY" src/sweep.py --machine "$MACHINE" --blas-threads "$THREADS" --out "$OUT/rq1_${THREADS}t.csv"
  step rq1_compare "$PY" src/analyze.py --compare "$OUT/rq1_1t.csv" "$OUT/rq1_${THREADS}t.csv"
  step profile_small "$PY" src/profile_block.py --regime imbalanced --p 200 --ratio 10 --lam 0.15
  step profile_large "$PY" src/profile_block.py --regime balanced --p 800 --ratio 10 --lam 0.10
  step percolation "$PY" src/percolation.py --seeds 0 1 2 --blas-threads "$THREADS" --out "$OUT/percolation.csv"
  gpu  crossover_fp64 "$PY" src/gpu_crossover.py --p 1200 2000 --regime many_tiny --blas-threads "$THREADS" --out "$OUT/gpu_crossover.csv"
  gpu  crossover_fp32 "$PY" src/gpu_crossover.py --p 1200 2000 --regime many_tiny --blas-threads "$THREADS" --dtype float32 --out "$OUT/gpu_crossover_fp32.csv"
  gpu  fp32_diagnostic "$PY" src/fp32_diagnostic.py --blas-threads "$THREADS" --crossover-csv "$OUT/gpu_crossover.csv" --crossover-env "$OUT/gpu_crossover_env.json" --out "$OUT/fp32_diagnostic.csv"
  gpu  fp32_cpu_baseline "$PY" src/fp32_cpu_baseline.py --blas-threads "$THREADS" --crossover-csv "$OUT/gpu_crossover_fp32.csv" --out "$OUT/fp32_cpu_baseline.csv"
  if [ "$TIER" = full ]; then
    for s in 1 2; do
      step rq1_1t_seed$s "$PY" src/sweep.py --machine "$MACHINE" --seed $s --blas-threads 1 --out "$OUT/rq1_1t_seed$s.csv"
      step rq1_multi_seed$s "$PY" src/sweep.py --machine "$MACHINE" --seed $s --blas-threads "$THREADS" --out "$OUT/rq1_${THREADS}t_seed$s.csv"
    done
    step batched_cpu "$PY" src/bench_batched.py --backends numpy --blas-threads "$THREADS" --machine "$MACHINE" --ref-csv "$OUT/rq1_${THREADS}t.csv" --out "$OUT/batched_cpu.csv"
    gpu  batched_cpu_gpu "$PY" src/bench_batched.py --backends numpy torch --blas-threads "$THREADS" --machine "$MACHINE" --ref-csv "$OUT/rq1_${THREADS}t.csv" --out "$OUT/batched_cpu_gpu.csv"
    for d in float64 float32; do
      gpu mem_2400_$d "$PY" src/gpu_crossover.py --p 2400 --regime many_tiny --lam 0.08 --blas-threads "$THREADS" --dtype $d --out "$OUT/gpu_mem_2400_$d.csv"
      gpu mem_3200_$d "$PY" src/gpu_crossover.py --p 3200 --regime many_tiny --lam 0.06 --blas-threads "$THREADS" --dtype $d --out "$OUT/gpu_mem_3200_$d.csv"
    done
    step screening "$PY" src/screening.py --blas-threads "$THREADS" --rq1 "$OUT/rq1_${THREADS}t.csv" --perc "$OUT/percolation.csv" --out-exact "$OUT/screening_exactness.csv" --out-mech "$OUT/screening_cross_block.csv"
  fi
fi

if [ ${#FAILED[@]} -gt 0 ]; then
  echo "[$(date -Is)] FINISHED WITH FAILURES: ${FAILED[*]}" | tee -a "$LOG"
else
  echo "[$(date -Is)] ALL DONE" | tee -a "$LOG"
fi
tar -czf "results/machines/$MACHINE.tar.gz" -C results/machines "$MACHINE"
( cd results/machines && sha256sum "$MACHINE.tar.gz" > "$MACHINE.tar.gz.sha256" )
echo
echo "Finished. Send these two files back:"
echo "  results/machines/$MACHINE.tar.gz"
echo "  results/machines/$MACHINE.tar.gz.sha256"
