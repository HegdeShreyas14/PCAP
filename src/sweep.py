"""
sweep.py — RQ1 experimental harness (v2).

Compares CPU ADMM_SGL against CPU block_SGL across:
    - problem size p
    - block-size distribution regime (balanced / imbalanced / many_tiny / one_giant)
    - regularization lambda

Design commitments
------------------
1. NOTHING is logged, measured, or introspected inside the timing path.
   Iteration counts and peak memory come from separate diagnostic runs.
   Solver stdout is suppressed during timing (block_SGL prints once per
   block; at ~100 blocks that is ~100 console writes inside the measured
   region, and it biases the block solver specifically).

2. Run-to-run variance is real. On one machine, an identical config with an
   identical seed produced speedups of 4.98x, 5.18x and 6.04x across three
   repeats of median-of-5. We therefore record the full trial vector, not
   just a median, so any reported speedup can carry a spread. Do not quote a
   single speedup number without one.

3. true_n_blocks != detected_n_blocks. The ground truth we generate and the
   components GGLasso finds at a given lambda are different things: S is a
   finite-sample covariance, so the estimated graph differs from the
   population graph. Performance is governed by DETECTED blocks, since those
   are what actually execute independently. Both are recorded.

4. The schema is deliberately wide. Re-running 128 configs to recover a
   column we forgot is expensive; storing an extra float is not.

Usage
-----
    python src/sweep.py --quick                        # ~1 min smoke test
    python src/sweep.py --machine primary \
                        --out results/rq1_primary.csv  # full sweep
"""

import argparse
import contextlib
import csv
import io
import json
import os
import platform
import statistics
import sys
import time
import tracemalloc
from datetime import datetime

# ---------------------------------------------------------------------------
# BLAS threading must be pinned BEFORE numpy is imported, or the setting is
# ignored.
#
# Why this matters: NumPy dispatches eigendecomposition to a threaded BLAS.
# ADMM_SGL does one large eigendecomposition per iteration, which a threaded
# BLAS parallelises well. block_SGL does many small ones, which it does not --
# small matrices cannot saturate the threads. So on a multi-core machine the
# CPU baseline speeds up more than the block solver does, and the measured
# "algorithmic speedup" silently depends on how many cores BLAS grabbed.
#
# Leaving this uncontrolled makes results incomparable across our three
# machines, which is precisely the cross-platform claim we are trying to make.
# Default to 1 thread so the algorithmic comparison is clean; use
# --blas-threads N to measure the threaded case deliberately.
# ---------------------------------------------------------------------------
def _set_blas_threads(n):
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(n)


_PRE_ARGS = argparse.ArgumentParser(add_help=False)
_PRE_ARGS.add_argument("--blas-threads", type=int, default=1)
_pre, _ = _PRE_ARGS.parse_known_args()
_set_blas_threads(_pre.blas_threads)

import numpy as np

import gglasso
from gglasso.solver.single_admm_solver import ADMM_SGL, block_SGL, get_connected_components

from blockgen import make_instance, REGIMES


# --------------------------------------------------------------------------
# environment
# --------------------------------------------------------------------------

def environment():
    env = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "logical_cores": os.cpu_count(),
        "blas_threads": os.environ.get("OMP_NUM_THREADS", "unset"),
        "numpy": np.__version__,
        "gglasso": getattr(gglasso, "__version__", "unknown"),
    }
    try:
        import torch
        env["torch"] = torch.__version__
        env["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name(0)
            env["cuda"] = torch.version.cuda
        else:
            env["gpu"] = None
    except ImportError:
        env["torch"] = None
        env["cuda_available"] = False
        env["gpu"] = None
    return env


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def glasso_objective(Theta, S, lam):
    """
    -log det(Theta) + tr(S Theta) + lam * ||Theta||_{1,od}

    Off-diagonal penalty only, matching GGLasso's prox_od_1norm, which copies
    the diagonal through untouched. Returns inf for a non-PD iterate so a
    diverged solve is comparable instead of crashing the sweep.
    """
    sign, logdet = np.linalg.slogdet(Theta)
    if sign <= 0:
        return float("inf")
    off_diag = np.abs(Theta).sum() - np.abs(np.diag(Theta)).sum()
    return -logdet + np.trace(S @ Theta) + lam * off_diag


def support_agreement(A, B, tol=1e-6):
    """Fraction of entries where the two solutions agree on zero/nonzero."""
    return float(((np.abs(A) > tol) == (np.abs(B) > tol)).mean())


def support_recovery(Theta_hat, Theta_true, tol=1e-6):
    """
    Precision / recall / F1 of the recovered off-diagonal support against the
    TRUE precision matrix.

    Why this is here
    ----------------
    Runtime alone can mislead. In testing, the fastest configuration
    (imbalanced, p=200, lambda=0.25) reached 5.9x by detecting 116 components
    of which 79 were singletons, with a largest block of 16 -- against a true
    structure of [180, 8, 8, 2, 2]. The speedup came from over-fragmenting the
    graph, not from recovering its real structure.

    A large lambda will always look fast, because it shatters the problem into
    trivial pieces. Without a statistical quality measure alongside the timing,
    we cannot tell "GPU acceleration helps" from "we over-regularized until
    there was nothing left to compute". These columns let us restrict the
    performance claims to the lambda range where the estimate is still useful.
    """
    p = Theta_hat.shape[0]
    off = ~np.eye(p, dtype=bool)
    est = (np.abs(Theta_hat) > tol) & off
    tru = (np.abs(Theta_true) > tol) & off

    tp = float((est & tru).sum())
    fp = float((est & ~tru).sum())
    fn = float((~est & tru).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    if precision + recall > 0 and np.isfinite(precision) and np.isfinite(recall):
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = float("nan")
    return {"support_precision": precision,
            "support_recall": recall,
            "support_f1": f1}


def sparsity(Theta, tol=1e-6):
    """Fraction of off-diagonal entries that are zero."""
    p = Theta.shape[0]
    off = np.abs(Theta) > tol
    np.fill_diagonal(off, False)
    n_off = p * (p - 1)
    return float(1.0 - off.sum() / n_off) if n_off else float("nan")


def size_stats(sizes, prefix):
    """min/max/mean/std/median of a block-size list, prefixed for the CSV."""
    if not sizes:
        return {f"{prefix}_{k}": float("nan")
                for k in ("min", "max", "mean", "std", "median")}
    a = np.asarray(sizes, dtype=float)
    return {
        f"{prefix}_min": float(a.min()),
        f"{prefix}_max": float(a.max()),
        f"{prefix}_mean": float(a.mean()),
        f"{prefix}_std": float(a.std()),
        f"{prefix}_median": float(np.median(a)),
    }


# --------------------------------------------------------------------------
# timing: nothing but the solver call inside the measured region
# --------------------------------------------------------------------------

@contextlib.contextmanager
def _silence():
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def timed(fn, warmup, trials):
    """Warm up (discarded), then time `trials` runs. Returns (times, last_result)."""
    with _silence():
        for _ in range(warmup):
            fn()
        times = []
        result = None
        for _ in range(trials):
            t0 = time.perf_counter()
            result = fn()
            times.append(time.perf_counter() - t0)
    return times, result


def timing_summary(times, prefix):
    return {
        f"{prefix}_median": statistics.median(times),
        f"{prefix}_mean": statistics.fmean(times),
        f"{prefix}_min": min(times),
        f"{prefix}_max": max(times),
        f"{prefix}_std": statistics.stdev(times) if len(times) > 1 else 0.0,
        f"{prefix}_trials_json": json.dumps([round(t, 6) for t in times]),
    }


# --------------------------------------------------------------------------
# diagnostics: run OUTSIDE the timing path
# --------------------------------------------------------------------------

def diagnostics(S, lam, Omega_0, common):
    """
    Separate, untimed runs to collect iteration counts and peak memory.

    Kept out of the timing path on purpose: measure=True adds bookkeeping and
    tracemalloc adds substantial allocation-tracking overhead, either of which
    would contaminate the runtime numbers.

    NOTE (gglasso 0.3.0): block_SGL returns `sol` only, not (sol, info),
    despite its docstring. There is no per-block iteration count available, so
    block_iterations is recorded as NaN. Verified against the installed source.
    """
    out = {}
    with _silence():
        res = ADMM_SGL(S, lam, Omega_0, **{**common, "measure": True})
        info_admm = res[1] if isinstance(res, tuple) and len(res) > 1 else {}
        # gglasso 0.3.0's info dict has keys: status, runtime, residual.
        # There is no 'iter' key; the iteration count is the length of the
        # recorded residual history.
        if isinstance(info_admm, dict) and "residual" in info_admm:
            out["admm_iterations"] = int(len(info_admm["residual"]))
            out["admm_status"] = str(info_admm.get("status", ""))
        else:
            out["admm_iterations"] = -1
            out["admm_status"] = ""
        out["block_iterations"] = float("nan")  # unavailable in gglasso 0.3.0

        tracemalloc.start()
        ADMM_SGL(S, lam, Omega_0, **common)
        _, peak_admm = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        tracemalloc.start()
        block_SGL(S, lam, Omega_0, **common)
        _, peak_block = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    out["admm_peak_mem_mb"] = peak_admm / 1024 / 1024
    out["block_peak_mem_mb"] = peak_block / 1024 / 1024
    return out


# --------------------------------------------------------------------------
# one configuration
# --------------------------------------------------------------------------

def run_config(regime, p, n_samples, lam, seed, warmup, trials,
               tol, rtol, max_iter, machine, env):
    S, Theta_true, true_blocks, true_sizes, styles = make_instance(
        regime, p, n_samples, seed=seed)
    S = S[0] if S.ndim == 3 else S
    Omega_0 = np.eye(p)

    numC, allC = get_connected_components(S, lam)
    detected_sizes = sorted((len(c) for c in allC), reverse=True)

    common = dict(tol=tol, rtol=rtol, max_iter=max_iter, verbose=False, measure=False)

    # --- timed runs: solver call only ---
    admm_times, sol_admm = timed(lambda: ADMM_SGL(S, lam, Omega_0, **common), warmup, trials)
    theta_admm = sol_admm[0]["Theta"] if isinstance(sol_admm, tuple) else sol_admm["Theta"]

    block_times, sol_block = timed(lambda: block_SGL(S, lam, Omega_0, **common), warmup, trials)
    theta_block = sol_block["Theta"]

    # --- untimed diagnostics ---
    diag = diagnostics(S, lam, Omega_0, common)

    t_admm = statistics.median(admm_times)
    t_block = statistics.median(block_times)
    speedup = t_admm / t_block if t_block > 0 else float("nan")

    # conservative bounds from the trial vectors, so figures can show a band
    speedup_lo = min(admm_times) / max(block_times) if max(block_times) > 0 else float("nan")
    speedup_hi = max(admm_times) / min(block_times) if min(block_times) > 0 else float("nan")

    obj_admm = glasso_objective(theta_admm, S, lam)
    obj_block = glasso_objective(theta_block, S, lam)
    max_abs = float(np.max(np.abs(theta_admm - theta_block)))
    denom = max(float(np.max(np.abs(theta_admm))), 1e-12)

    row = {
        # provenance
        "machine": machine,
        "cpu": env.get("processor", ""),
        "gpu": env.get("gpu") or "none",
        "gglasso": env.get("gglasso", ""),
        "numpy": env.get("numpy", ""),
        "logical_cores": env.get("logical_cores", ""),
        "blas_threads": env.get("blas_threads", ""),
        # configuration
        "regime": regime,
        "p": p,
        "n_samples": n_samples,
        "lambda": lam,
        "seed": seed,
        "true_sparsity": sparsity(Theta_true),
        # ground-truth block structure
        "true_n_blocks": len(true_sizes),
        "true_sizes_json": json.dumps(true_sizes),
        # which network model actually generated each block. powerlaw can fail
        # and fall back to Erdos-Renyi depending on networkx version, so two
        # machines running identical code may not draw from the same model.
        # If gen_n_erdos differs across machines, the cross-machine comparison
        # is invalid and must be re-run with an explicit style.
        "gen_style_requested": "powerlaw",
        "gen_n_powerlaw": sum(1 for s in styles if s == "powerlaw"),
        "gen_n_erdos": sum(1 for s in styles if s == "erdos"),
        "gen_n_direct": sum(1 for s in styles if s == "direct"),
        # detected block structure (this is what governs performance)
        "detected_n_blocks": numC,
        "detected_singletons": sum(1 for s in detected_sizes if s == 1),
        "detected_sizes_json": json.dumps(detected_sizes[:200]),
        # runtimes
        **timing_summary(admm_times, "t_admm"),
        **timing_summary(block_times, "t_block"),
        # RQ1 headline
        "algorithmic_speedup": speedup,
        "speedup_lo": speedup_lo,
        "speedup_hi": speedup_hi,
        # diagnostics (untimed)
        **diag,
        # correctness
        "obj_admm": obj_admm,
        "obj_block": obj_block,
        "obj_rel_gap": abs(obj_admm - obj_block) / max(abs(obj_admm), 1e-12),
        "max_abs_error": max_abs,
        "max_rel_error": max_abs / denom,
        "support_agreement": support_agreement(theta_admm, theta_block),
        # statistical quality of the estimate against ground truth: guards
        # against reporting speedups from an over-regularized, useless fit
        **support_recovery(theta_block, Theta_true),
        # protocol
        "trials": trials,
        "warmup": warmup,
        "tol": tol,
        "rtol": rtol,
        "max_iter": max_iter,
    }
    row.update(size_stats(true_sizes, "true_block"))
    row.update(size_stats(detected_sizes, "detected_block"))
    return row


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/rq1_results.csv")
    ap.add_argument("--machine", default="unlabelled",
                    help="primary | secondary | tertiary")
    ap.add_argument("--trials", type=int, default=9,
                    help="timed trials per config; >=9 given observed variance")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tol", type=float, default=1e-7)
    ap.add_argument("--rtol", type=float, default=1e-5)
    ap.add_argument("--max-iter", type=int, default=1000)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="resume from the per-row JSONL checkpoint for --out")
    ap.add_argument("--blas-threads", type=int, default=1,
                    help="BLAS threads (applied before numpy import). "
                         "1 = clean algorithmic comparison; >1 = threaded CPU baseline.")
    args = ap.parse_args()

    if args.quick:
        p_values, lambdas = [100, 200], [0.15, 0.25]
        ratios = [2, 10]
        regimes, trials, warmup = ["balanced", "imbalanced"], 3, 1
    else:
        p_values = [100, 200, 400, 800]
        lambdas = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
        # n/p ratio matters a great deal: at n=2p the estimated graph is so
        # noisy that the fast (high-lambda) regime is also badly
        # over-regularized. By n=10p, peak-F1 and ~4x speedup coincide.
        # Sweeping this is what lets us report speedup at a statistically
        # defensible operating point rather than at whatever lambda is fastest.
        ratios = [2, 5, 10]
        regimes, trials, warmup = REGIMES, args.trials, args.warmup

    env = environment()
    print("Environment:")
    for k, v in env.items():
        print(f"  {k:16s} {v}")
    print(f"\nMachine label: {args.machine}   trials={trials} warmup={warmup}\n")

    # Derive the env path safely: str.replace would silently collide onto the
    # CSV path if --out has no .csv suffix (e.g. --out results/run1).
    if args.out.endswith(".csv"):
        env_path = args.out[:-4] + "_env.json"
    else:
        env_path = args.out + "_env.json"
    with open(env_path, "w") as f:
        json.dump({**env, "machine": args.machine,
                   "trials": trials, "warmup": warmup,
                   "tol": args.tol, "rtol": args.rtol}, f, indent=2)

    checkpoint_path = args.out + ".partial.jsonl"
    rows = []
    completed = set()
    if args.resume and os.path.exists(checkpoint_path):
        with open(checkpoint_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                rows.append(row)
                completed.add((row["regime"], int(row["p"]),
                               int(row["n_samples"]), float(row["lambda"])))
        print(f"Resuming from {len(rows)} checkpointed configurations in "
              f"{checkpoint_path}\n")
    elif os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    total = len(regimes) * len(p_values) * len(ratios) * len(lambdas)
    i = 0
    for regime in regimes:
        for p in p_values:
            for ratio in ratios:
                for lam in lambdas:
                    i += 1
                    key = (regime, p, ratio * p, float(lam))
                    if key in completed:
                        print(f"[{i:3d}/{total}] {regime:11s} p={p:4d} n/p={ratio:2d} "
                              f"lam={lam:.2f} ... checkpointed")
                        continue
                    print(f"[{i:3d}/{total}] {regime:11s} p={p:4d} n/p={ratio:2d} "
                          f"lam={lam:.2f} ... ", end="", flush=True)
                    try:
                        row = run_config(regime, p, ratio * p, lam, args.seed,
                                         warmup, trials, args.tol, args.rtol,
                                         args.max_iter, args.machine, env)
                        rows.append(row)
                        with open(checkpoint_path, "a", encoding="utf-8") as checkpoint:
                            checkpoint.write(json.dumps(row) + "\n")
                            checkpoint.flush()
                        print(f"blocks={row['detected_n_blocks']:4d} "
                              f"speedup={row['algorithmic_speedup']:.2f}x "
                              f"[{row['speedup_lo']:.2f}-{row['speedup_hi']:.2f}] "
                              f"F1={row['support_f1']:.3f}")
                    except Exception as e:
                        print(f"FAILED: {type(e).__name__}: {e}")

    if not rows:
        print("No results produced.")
        return

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows ({len(rows[0])} columns) to {args.out}")
    print(f"Environment written to {env_path}")
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    # RQ1 readout: speedup against DETECTED block count, bucketed.
    # F1 is shown alongside because a large lambda always looks fast: it
    # shatters the graph into trivial pieces. Speedup in a bucket where F1 has
    # collapsed is not a useful operating point.
    print("\nSpeedup vs detected block count (all regimes pooled):")
    buckets = [(1, 1), (2, 9), (10, 49), (50, 99), (100, 199), (200, 10**9)]
    for lo, hi in buckets:
        sub = [r for r in rows
               if lo <= r["detected_n_blocks"] <= hi
               and np.isfinite(r["algorithmic_speedup"])]
        if sub:
            sp = [r["algorithmic_speedup"] for r in sub]
            f1 = [r["support_f1"] for r in sub if np.isfinite(r["support_f1"])]
            label = f"{lo}" if lo == hi else f"{lo}-{hi if hi < 10**9 else '+'}"
            f1s = f"{statistics.median(f1):.3f}" if f1 else "n/a"
            print(f"  blocks {label:>8s}: median {statistics.median(sp):5.2f}x  "
                  f"range {min(sp):.2f}-{max(sp):.2f}  "
                  f"median F1 {f1s}  n={len(sub)}")

    print("\nBy regime:")
    for regime in regimes:
        sub = [r["algorithmic_speedup"] for r in rows
               if r["regime"] == regime and np.isfinite(r["algorithmic_speedup"])]
        if sub:
            print(f"  {regime:12s} median {statistics.median(sub):5.2f}x  "
                  f"range {min(sub):.2f}-{max(sub):.2f}  n={len(sub)}")

    # The defensible headline: speedup at the lambda that maximises estimate
    # quality, not at whatever lambda happens to be fastest.
    print("\nSpeedup at the best-F1 lambda (per regime / p / sample ratio):")
    keys = sorted({(r["regime"], r["p"], r["n_samples"] // r["p"]) for r in rows})
    for regime, p, ratio in keys:
        sub = [r for r in rows
               if r["regime"] == regime and r["p"] == p
               and r["n_samples"] // r["p"] == ratio
               and np.isfinite(r["support_f1"])]
        if not sub:
            continue
        best = max(sub, key=lambda r: r["support_f1"])
        print(f"  {regime:11s} p={p:4d} n/p={ratio:2d}: "
              f"best F1 {best['support_f1']:.3f} at lam={best['lambda']:.2f} "
              f"-> {best['algorithmic_speedup']:.2f}x "
              f"({best['detected_n_blocks']} blocks)")


if __name__ == "__main__":
    main()
