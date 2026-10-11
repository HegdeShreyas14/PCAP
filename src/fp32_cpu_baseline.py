"""
fp32_cpu_baseline.py — like-for-like float32 comparison for the GPU result.

The float32 GPU crossover (gpu_crossover_fp32.csv) is measured against the
float64 CPU baseline. A float32 CPU solver is faster too, so the fair GPU
speedup is t_numpy32 / t_torch32. This times, in one session and under the
crossover protocol (2 warm-up + 5 timed, median):

    numpy float64  (reference baseline)
    numpy float32
    torch float32

at every lambda of the p=1200/2000 crossover scan plus the two VRAM anchors.

    python src/fp32_cpu_baseline.py --blas-threads 8
"""

import argparse
import csv
import json
import os
import time

_PRE = argparse.ArgumentParser(add_help=False)
_PRE.add_argument("--blas-threads", type=int, default=8)
_pre, _ = _PRE.parse_known_args()
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = str(_pre.blas_threads)

import numpy as np

from blockgen import make_instance
from batched_admm import batched_block_SGL
from provenance import write_env


def timed(fn, warmup, trials):
    for _ in range(warmup):
        fn()
    times, result = [], None
    for _ in range(trials):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times)), result


def main():
    ap = argparse.ArgumentParser(parents=[_PRE])
    ap.add_argument("--crossover-csv", default="results/gpu_crossover_fp32.csv")
    ap.add_argument("--anchors", nargs="+", default=["2400:0.08", "3200:0.06"])
    ap.add_argument("--regime", default="many_tiny")
    ap.add_argument("--ratio", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--out", default="results/fp32_cpu_baseline.csv")
    args = ap.parse_args()
    print(f"environment -> {write_env(args.out, args=args)}")

    with open(args.crossover_csv) as f:
        points = [(int(r["p"]), float(r["lam"])) for r in csv.DictReader(f)]
    points += [(int(a.split(":")[0]), float(a.split(":")[1])) for a in args.anchors]

    rows, cache = [], {}
    for p, lam in points:
        if p not in cache:
            cache[p] = make_instance(args.regime, p, args.ratio * p, seed=args.seed)[0]
        S = cache[p]
        run = lambda backend, dtype: batched_block_SGL(
            S, lam, backend=backend, dtype=dtype, return_info=True)
        t64, (s64, i64) = timed(lambda: run("numpy", "float64"), args.warmup, args.trials)
        tc32, (sc32, ic32) = timed(lambda: run("numpy", "float32"), args.warmup, args.trials)
        tg32, (sg32, ig32) = timed(lambda: run("torch", "float32"), args.warmup, args.trials)
        scale = np.abs(s64["Theta"]).max()
        sizes = sorted(i64["group_sizes"], reverse=True)
        r = dict(regime=args.regime, p=p, lam=lam, n_blocks=i64["n_blocks"],
                 n_groups=i64["n_groups"], s_max=max(sizes) if sizes else 1,
                 t_numpy64=t64, t_numpy32=tc32, t_torch32=tg32,
                 cpu_fp32_over_fp64=t64 / tc32,                 # CPU gain from float32
                 gpu_speedup_vs_cpu64=t64 / tg32,               # as in gpu_crossover_fp32
                 gpu_speedup_vs_cpu32=tc32 / tg32,              # like-for-like
                 rel_diff_numpy32=float(np.abs(sc32["Theta"] - s64["Theta"]).max() / scale),
                 rel_diff_torch32=float(np.abs(sg32["Theta"] - s64["Theta"]).max() / scale),
                 not_converged_numpy32=ic32["n_not_converged"],
                 not_converged_torch32=ig32["n_not_converged"],
                 max_iters_numpy64=i64["max_iters"], max_iters_numpy32=ic32["max_iters"],
                 max_iters_torch32=ig32["max_iters"])
        rows.append(r)
        print(f"p={p:5d} lam={lam:.3f} s_max={r['s_max']:5d}  np64 {t64:.3f}s  np32 {tc32:.3f}s  "
              f"torch32 {tg32:.3f}s  | GPU vs CPU32 {r['gpu_speedup_vs_cpu32']:.2f}x  "
              f"(vs CPU64 {r['gpu_speedup_vs_cpu64']:.2f}x)  "
              f"nc np32={r['not_converged_numpy32']} gpu32={r['not_converged_torch32']}", flush=True)

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
