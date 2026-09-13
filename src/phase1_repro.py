"""
phase1_repro.py — run once per machine.

Confirms GGLasso installs and runs, reproduces a small worked example, and
prints the environment fingerprint that must be recorded before any timing
result from that machine counts.

Paste the ENVIRONMENT block into notes/PROGRESS.md (or your machine notes).

    python src/phase1_repro.py
"""

import os
import platform
import sys
import time

# Pin BLAS before numpy is imported, same as sweep.py. Reported timings here
# are illustrative only, but the thread count belongs in the fingerprint.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

import gglasso
from gglasso.helper.data_generation import group_power_network, sample_covariance_matrix
from gglasso.solver.single_admm_solver import ADMM_SGL, block_SGL, get_connected_components

# ---------------------------------------------------------------------------
# 1. Environment fingerprint
# ---------------------------------------------------------------------------
print("=" * 70)
print("ENVIRONMENT")
print("=" * 70)
print(f"Python         : {sys.version.split()[0]}")
print(f"Platform       : {platform.platform()}")
print(f"Processor      : {platform.processor() or platform.machine()}")
print(f"Logical cores  : {os.cpu_count()}")
print(f"BLAS threads   : {os.environ.get('OMP_NUM_THREADS')}")
print(f"NumPy          : {np.__version__}")
print(f"GGLasso        : {getattr(gglasso, '__version__', 'unknown')}")
try:
    import torch
    print(f"PyTorch        : {torch.__version__}")
    print(f"CUDA available : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU            : {torch.cuda.get_device_name(0)}")
        print(f"CUDA version   : {torch.version.cuda}")
        props = torch.cuda.get_device_properties(0)
        print(f"VRAM           : {props.total_memory / 1024**3:.1f} GB")
except ImportError:
    print("PyTorch        : not installed (needed for Phase 3 only)")

print()
print("Also record by hand (not detectable from Python):")
print("  - GPU power mode / performance profile")
print("  - RAM installed")
print("  - mains power vs battery during timed runs")

# ---------------------------------------------------------------------------
# 2. Worked example: confirm the install actually solves something
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("WORKED EXAMPLE: block-sparse precision matrix, p=100, 10 blocks")
print("=" * 70)

p, K, N, nblocks = 100, 1, 200, 10
Sigma, Theta = group_power_network(p, K, M=nblocks, seed=42)
S, _ = sample_covariance_matrix(Sigma, N, seed=42)
S = S[0]

# Both solvers get IDENTICAL tolerances. block_SGL's defaults are looser than
# ADMM_SGL's, and unequal settings manufacture a fake speedup.
tol, rtol = 1e-7, 1e-5

# lambda must be large enough that the graph actually fractures. At small
# lambda the whole graph is one component and block_SGL has nothing to exploit.
lambda1 = 0.35
Omega_0 = np.eye(p)

t0 = time.perf_counter()
sol_admm, info_admm = ADMM_SGL(S, lambda1, Omega_0, tol=tol, rtol=rtol, measure=True)
t_admm = time.perf_counter() - t0

# NOTE (gglasso 0.3.0): block_SGL returns `sol` only, not (sol, info), despite
# its docstring. No per-block iteration count is available.
t0 = time.perf_counter()
sol_block = block_SGL(S, lambda1, Omega_0, tol=tol, rtol=rtol)
t_block = time.perf_counter() - t0

numC, allC = get_connected_components(S, lambda1)

# info dict has keys status/runtime/residual — there is no 'iter' key.
iters = len(info_admm["residual"]) if isinstance(info_admm, dict) and "residual" in info_admm else "?"

print(f"ADMM_SGL   : {t_admm:.4f}s, {iters} iterations")
print(f"block_SGL  : {t_block:.4f}s")
print(f"Detected blocks at lambda={lambda1}: {numC} (ground truth: {nblocks})")
print("  (detected != true is expected: S is a finite-sample covariance)")

theta_admm = sol_admm["Theta"]
theta_block = sol_block["Theta"]
print(f"Max |ADMM - block| elementwise: {np.max(np.abs(theta_admm - theta_block)):.2e}")
print(f"Solvers agree (allclose, atol=1e-3): "
      f"{np.allclose(theta_admm, theta_block, atol=1e-3)}")

print()
print("Phase 1 complete. Record the ENVIRONMENT block above for this machine.")