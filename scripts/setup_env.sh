#!/usr/bin/env bash
# Build the project venv from the package lock on a new machine.
#
#   scripts/setup_env.sh            # creates .venv
#   VENV=/path/to/venv scripts/setup_env.sh
#
# Everything except PyTorch is installed at the exact locked versions. PyTorch
# 2.14.0 is installed for the machine: the CUDA 13 build if an NVIDIA driver is
# present, otherwise the CPU build. GPU experiments are skipped automatically
# on CPU-only machines; everything else still runs.
set -euo pipefail
cd "$(dirname "$0")/.."
VENV=${VENV:-.venv}
LOCK=results/environment/requirements-lock.txt
TORCH_VERSION=2.14.0

command -v python3.12 >/dev/null || { echo "python3.12 is required (the lock is built for CPython 3.12)." >&2; exit 1; }
[ -d "$VENV" ] && { echo "$VENV already exists; remove it or set VENV=... to build elsewhere." >&2; exit 1; }

python3.12 -m venv "$VENV"
PY="$VENV/bin/python"
"$PY" -m pip install -q --upgrade pip

# core: everything in the lock except torch and its CUDA wheels
grep -v -E '^(#|torch==|triton==|nvidia-|cuda-)' "$LOCK" > /tmp/core-lock.$$.txt
echo "installing locked core packages..."
"$PY" -m pip install -q -r /tmp/core-lock.$$.txt
rm -f /tmp/core-lock.$$.txt

if command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null 2>&1; then
  echo "NVIDIA driver found ($(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)); installing torch $TORCH_VERSION (CUDA 13.0, ~3 GB)..."
  if ! "$PY" -m pip install -q "torch==$TORCH_VERSION" --index-url https://download.pytorch.org/whl/cu130; then
    echo "CUDA 13.0 build failed (driver too old?). Falling back to the CPU build; GPU experiments will be skipped." >&2
    "$PY" -m pip install -q "torch==$TORCH_VERSION" --index-url https://download.pytorch.org/whl/cpu
  fi
else
  echo "no NVIDIA GPU detected; installing the CPU build of torch $TORCH_VERSION"
  "$PY" -m pip install -q "torch==$TORCH_VERSION" --index-url https://download.pytorch.org/whl/cpu
fi

"$PY" - <<'PYEOF'
import numpy, scipy, gglasso, torch
print(f"numpy {numpy.__version__}  scipy {scipy.__version__}  gglasso {gglasso.__version__}  "
      f"torch {torch.__version__}  CUDA available: {torch.cuda.is_available()}")
PYEOF
"$PY" tests/test_sampling_determinism.py
echo "environment ready: $VENV"
