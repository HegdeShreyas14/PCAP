"""
gpu_crossover.py — trace GPU-batched vs CPU-batched speedup as a function of
the LARGEST DETECTED BLOCK SIZE, not p.

Two modes
---------
1. CROSSOVER BRACKETING (default): sweep lambda finely at a fixed p, so
   s_max is directly observed rather than hoped for via a p choice. Use this
   at 1-2 p anchors (e.g. 1200 and 2000) to trace speedup vs s_max directly,
   and to check whether the crossover (in terms of s_max) holds steady across
   p or drifts -- drift would indicate overhead depends on something beyond
   s_max (see below).

2. MEMORY-WALL ANCHORS (--lam explicit): at large p (2400, 3200, ...) you are
   not trying to bracket a precise crossover, you are checking scalability
   and VRAM behaviour, where detection noise in s_max matters far less. Pass
   one best-known lambda directly instead of scanning.

Why not just sweep p with one lambda per point (as originally proposed)
-------------------------------------------------------------------------
p is a noisy proxy for s_max. Our own data already shows this: imbalanced
p=400 produced a SMALLER largest block (43) than p=200 (59), at a single
best-F1-lambda instance each. Picking p=1000 and hoping its largest block
lands near a target of 900-1100 is aiming through the same noisy proxy that
produced an unreliable extrapolation in the first place (see below).

On the "T = T_fixed + T_GPU(s_max)" model
------------------------------------------
This is a reasonable HYPOTHESIS, not a derived law. Fitting it from exactly
two points (many_tiny p=400/800, s_max=299/693) and evaluating at a THIRD
point we already had (p=200, s_max=14) predicts 0.73s; the measured value was
0.20s -- 3.7x off. Overhead is not a clean function of s_max alone; it also
depends on how many distinct block-size GROUPS exist, since each group pays
its own per-iteration launch/sync cost independent of arithmetic size. This
script logs n_groups precisely so that can be checked against the real data
instead of assumed.

Timing breakdown
-----------------
Wall-clock torch time is decomposed into t_h2d (host->device transfer),
t_compute (the ADMM loop, GPU-only), and t_d2h (device->host), accumulated
across groups. This is what turns "torch is slower" into "here is exactly
where the time goes" -- e.g. whether the small-p penalty is transfer-bound or
dispatch-bound informs a completely different fix.

CPU baseline labeling
----------------------
Every row carries `cpu_baseline_label`. Above the point where the unbatched
gglasso reference becomes too expensive to run (see RUNBOOK, p>1600), the
CPU baseline is the batched-numpy path, not the original reference. It is
validated to ~1e-14 against the reference at every size tested so far, but it
is a different artifact and must be labeled "CPU-best (batched numpy)" in any
table or figure, never bare "CPU" -- conflating the two would misrepresent
which baseline a given number was measured against.

Usage
-----
    # crossover bracketing
    python src/gpu_crossover.py --p 1200 --regime many_tiny
    python src/gpu_crossover.py --p 1200 2000 --regime many_tiny imbalanced

    # memory-wall anchor at a single known-good lambda
    python src/gpu_crossover.py --p 3200 --regime many_tiny --lam 0.08
"""

import argparse
import contextlib
import csv
import io
import os
import time
import warnings

warnings.filterwarnings("ignore")

# Pin BLAS before numpy import, same convention as sweep.py / percolation.py.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import numpy as np

from blockgen import make_instance
from batched_admm import batched_block_SGL, get_backend
from gglasso.solver.single_admm_solver import get_connected_components, block_SGL


@contextlib.contextmanager
def _silence():
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def timed_call(fn, warmup=2, trials=5):
    for _ in range(warmup):
        fn()
    times = [None] * trials
    result = None
    for i in range(trials):
        t0 = time.perf_counter()
        result = fn()
        times[i] = time.perf_counter() - t0
    times.sort()
    return times[len(times) // 2], result  # median


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", nargs="+", type=int, default=[1200])
    ap.add_argument("--regime", nargs="+", default=["many_tiny"])
    ap.add_argument("--ratio", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lam-min", type=float, default=0.03)
    ap.add_argument("--lam-max", type=float, default=0.40)
    ap.add_argument("--lam-steps", type=int, default=18)
    ap.add_argument("--lam", nargs="+", type=float, default=None,
                    help="explicit lambda value(s) instead of a linspace scan. "
                         "Use this for large-p memory-wall anchors where "
                         "bracketing a crossover is not the goal and a single "
                         "best-known lambda is enough (e.g. --lam 0.10). "
                         "Overrides --lam-min/--lam-max/--lam-steps.")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--ref-every", type=int, default=4,
                    help="run the slow block_SGL reference only every Nth "
                         "lambda point, as an algorithmic-gain sanity anchor "
                         "-- it is O(s_max^3) and not the quantity this "
                         "script is measuring")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/gpu_crossover.csv")
    ap.add_argument("--check-correctness", action="store_true", default=True)
    args = ap.parse_args()

    try:
        import torch
        has_gpu = torch.cuda.is_available()
    except ImportError:
        torch = None
        has_gpu = False
    if not has_gpu:
        print("!! CUDA not available -- torch backend will fail. Aborting.")
        return

    lams = np.asarray(args.lam) if args.lam is not None else np.linspace(
        args.lam_min, args.lam_max, args.lam_steps)
    rows = []

    print(f"{'regime':11s} {'p':>5s} {'lam':>5s} {'blks':>5s} {'grps':>5s} "
          f"{'s_max':>6s} {'numpy':>8s} {'torch':>8s} {'ratio':>7s} {'maxdiff':>9s}")
    print("-" * 92)

    for regime in args.regime:
        for p in args.p:
            S, Theta, blocks, sizes, styles = make_instance(
                regime, p, args.ratio * p, seed=args.seed)
            S = S[0] if S.ndim == 3 else S
            Omega_0 = np.eye(p)

            for i, lam in enumerate(lams):
                lam = float(lam)
                numC, allC = get_connected_components(S, lam)
                det_sizes = sorted((len(c) for c in allC), reverse=True)
                s_max = det_sizes[0] if det_sizes else 0

                # torch and numpy batched: what this script actually measures
                torch.cuda.reset_peak_memory_stats()
                t_np, (sol_np, info_np) = timed_call(
                    lambda: batched_block_SGL(S, lam, backend="numpy",
                                              return_info=True),
                    trials=args.trials)
                t_gpu, (sol_gpu, info_gpu) = timed_call(
                    lambda: batched_block_SGL(S, lam, backend="torch",
                                              device=args.device,
                                              return_info=True),
                    trials=args.trials)
                peak_mem_mb = torch.cuda.max_memory_allocated() / 1024**2

                maxdiff = float(np.max(np.abs(sol_np["Theta"] - sol_gpu["Theta"])))
                n_groups = info_np["n_groups"]

                # Structural context, same definitions as analyze.py: F_max is
                # the share of cubic work in the largest block; cubic_gain is
                # the arithmetic-reduction proxy, NOT a predicted speedup (see
                # analyze.py docstring -- profiling already showed this proxy
                # overweights the largest block relative to measured wall-clock).
                cubic_tot = sum(float(x) ** 3 for x in det_sizes) if det_sizes else float("nan")
                f_max = (float(s_max) ** 3 / cubic_tot) if cubic_tot else float("nan")
                cubic_gain = (float(p) ** 3 / cubic_tot) if cubic_tot else float("nan")

                # Sparse reference anchor only, since it is O(s_max^3) and this
                # script's question is numpy-batched vs torch-batched, not the
                # unbatched algorithmic gain (already characterized in
                # sweep.py). t_ref stays NaN on skipped points -- it is a
                # spot-check column, not a swept quantity.
                t_ref = float("nan")
                if i % args.ref_every == 0:
                    with _silence():
                        t_ref, _ = timed_call(
                            lambda: block_SGL(S, lam, Omega_0, tol=1e-7,
                                              rtol=1e-5, max_iter=1000,
                                              verbose=False),
                            warmup=1, trials=3)

                ratio_gpu_np = t_np / t_gpu if t_gpu > 0 else float("nan")
                flag = "" if maxdiff < 1e-8 else "  !! CORRECTNESS"
                print(f"{regime:11s} {p:5d} {lam:5.2f} {numC:5d} {n_groups:5d} "
                      f"{s_max:6d} {t_np:7.4f}s {t_gpu:7.4f}s {ratio_gpu_np:6.2f}x "
                      f"{maxdiff:9.2e}{flag}")
                print(f"    torch breakdown: h2d={info_gpu['t_h2d']*1000:.1f}ms  "
                      f"compute={info_gpu['t_compute']*1000:.1f}ms  "
                      f"d2h={info_gpu['t_d2h']*1000:.1f}ms  "
                      f"F_max={100*f_max:.1f}%  peak_vram={peak_mem_mb:.1f}MB")

                rows.append(dict(regime=regime, p=p, lam=lam, n_blocks=numC,
                                 n_groups=n_groups, s_max=s_max,
                                 cubic_gain=cubic_gain, f_max=f_max,
                                 t_numpy=t_np, t_torch=t_gpu,
                                 t_torch_h2d=info_gpu["t_h2d"],
                                 t_torch_compute=info_gpu["t_compute"],
                                 t_torch_d2h=info_gpu["t_d2h"],
                                 max_conv_spread=info_gpu["max_conv_spread"],
                                 ratio_torch_over_numpy=ratio_gpu_np,
                                 peak_gpu_mem_mb=peak_mem_mb,
                                 cpu_baseline_label=info_np["cpu_baseline_label"],
                                 max_diff=maxdiff, t_ref=t_ref))

    if rows:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {len(rows)} rows to {args.out}")

    # crude crossover readout: smallest s_max where torch beats numpy
    print("\nCrossover (first s_max where torch/numpy ratio exceeds 1.0):")
    for regime in args.regime:
        for p in args.p:
            sub = sorted(
                [r for r in rows if r["regime"] == regime and r["p"] == p],
                key=lambda r: r["s_max"])
            won = [r for r in sub if r["ratio_torch_over_numpy"] > 1.0]
            if won:
                print(f"  {regime} p={p}: s_max={won[0]['s_max']} "
                      f"(n_groups={won[0]['n_groups']}, "
                      f"lambda={won[0]['lam']:.2f})")
            else:
                best = max(sub, key=lambda r: r["ratio_torch_over_numpy"],
                          default=None)
                if best:
                    print(f"  {regime} p={p}: no crossover in range. "
                          f"Best: {best['ratio_torch_over_numpy']:.2f}x "
                          f"at s_max={best['s_max']}")


if __name__ == "__main__":
    main()