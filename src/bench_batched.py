"""
bench_batched.py — CPU benchmark of batched_block_SGL against gglasso's block_SGL.

Uses sweep.py's measurement discipline, because the point is a number that can
go in the paper:

  * BLAS threads pinned BEFORE numpy is imported (an unpinned run is not
    reproducible, and ADMM/block respond to threading very differently)
  * warm-up runs discarded, median of N timed trials, full trial vector kept
  * solver stdout suppressed inside the timed region (block_SGL prints one line
    per block, which penalises the reference specifically)
  * identical tol/rtol/max_iter/rho across every backend
  * correctness checked per config against block_SGL, not assumed

Lambda defaults to each config's best-F1 value taken from an existing sweep CSV,
so the comparison sits at the statistically defensible operating point rather
than at whatever lambda happens to be fastest.

    python src/bench_batched.py --blas-threads 8
    python src/bench_batched.py --blas-threads 8 --backends numpy torch
"""

import argparse
import os

# --- BLAS threads must be pinned before numpy is imported ------------------
def _set_blas_threads(n):
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(n)

_PRE = argparse.ArgumentParser(add_help=False)
_PRE.add_argument("--blas-threads", type=int, default=8)
_pre, _ = _PRE.parse_known_args()
_set_blas_threads(_pre.blas_threads)

import contextlib
import csv
import io
import json
import statistics
import sys
import time
from datetime import datetime

import numpy as np

from gglasso.solver.single_admm_solver import block_SGL, get_connected_components
from blockgen import make_instance
from batched_admm import batched_block_SGL


@contextlib.contextmanager
def _silence():
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def timed(fn, warmup, trials):
    """Warm up (discarded), then time `trials` runs. Returns (times, last result)."""
    with _silence():
        for _ in range(warmup):
            fn()
        times, out = [], None
        for _ in range(trials):
            t0 = time.perf_counter()
            out = fn()
            times.append(time.perf_counter() - t0)
    return times, out


def best_f1_lambdas(path):
    """best-F1 lambda per (regime, p, n/p) from a sweep CSV."""
    best = {}
    try:
        with open(path) as f:
            for r in csv.DictReader(f):
                try:
                    f1 = float(r["support_f1"])
                except (ValueError, KeyError):
                    continue
                if f1 != f1:
                    continue
                k = (r["regime"], int(r["p"]), int(r["n_samples"]) // int(r["p"]))
                if k not in best or f1 > best[k][0]:
                    best[k] = (f1, float(r["lambda"]))
    except FileNotFoundError:
        pass
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regimes", nargs="+", default=["imbalanced", "many_tiny"])
    ap.add_argument("--ps", nargs="+", type=int, default=[200, 400, 800])
    ap.add_argument("--ratios", nargs="+", type=int, default=[10])
    ap.add_argument("--backends", nargs="+", default=["numpy"],
                    help="batched backends to test: numpy and/or torch")
    ap.add_argument("--lam", type=float, default=None,
                    help="override; default = best-F1 lambda from --ref-csv")
    ap.add_argument("--ref-csv", default="results/rq1_primary_8t.csv")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trials", type=int, default=9)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--tol", type=float, default=1e-7)
    ap.add_argument("--rtol", type=float, default=1e-5)
    ap.add_argument("--max-iter", type=int, default=1000)
    ap.add_argument("--machine", default="primary")
    ap.add_argument("--out", default="results/batched_cpu.csv")
    ap.add_argument("--blas-threads", type=int, default=8)
    args = ap.parse_args()

    lam_map = best_f1_lambdas(args.ref_csv)
    print(f"BLAS threads: {os.environ['OMP_NUM_THREADS']}   "
          f"trials={args.trials} warmup={args.warmup}   seed={args.seed}")
    print(f"lambda: {'override ' + str(args.lam) if args.lam else 'best-F1 from ' + args.ref_csv}")
    print(f"backends: block_SGL (ref) + batched {args.backends}\n")

    hdr = (f"{'regime':11s} {'p':>4s} {'n/p':>4s} {'lam':>5s} {'blks':>5s} "
           f"{'grps':>5s} {'ref_s':>8s}")
    for b in args.backends:
        hdr += f" {('bat_' + b)[:9]:>9s} {('x vs ref'):>9s} {'maxdiff':>9s}"
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for regime in args.regimes:
        for p in args.ps:
            for ratio in args.ratios:
                lam = args.lam or lam_map.get((regime, p, ratio), (None, 0.15))[1]
                S, Theta_true, _, true_sizes, _ = make_instance(
                    regime, p, ratio * p, seed=args.seed)
                S = S[0] if S.ndim == 3 else S
                Omega_0 = np.eye(p)
                common = dict(tol=args.tol, rtol=args.rtol,
                              max_iter=args.max_iter, verbose=False)

                numC, allC = get_connected_components(S, lam)

                ref_times, ref_sol = timed(
                    lambda: block_SGL(S, lam, Omega_0, **common),
                    args.warmup, args.trials)
                t_ref = statistics.median(ref_times)
                Theta_ref = ref_sol["Theta"]

                row = {"machine": args.machine, "blas_threads": args.blas_threads,
                       "regime": regime, "p": p, "n_over_p": ratio, "lambda": lam,
                       "seed": args.seed, "detected_n_blocks": numC,
                       "true_n_blocks": len(true_sizes),
                       "t_ref_median": t_ref, "t_ref_min": min(ref_times),
                       "t_ref_json": json.dumps([round(t, 6) for t in ref_times]),
                       "trials": args.trials, "warmup": args.warmup,
                       "tol": args.tol, "rtol": args.rtol}

                line = (f"{regime:11s} {p:4d} {ratio:4d} {lam:5.2f} {numC:5d}")
                n_groups = None
                for bname in args.backends:
                    try:
                        bt, (bsol, binfo) = timed(
                            lambda: batched_block_SGL(
                                S, lam, backend=bname, tol=args.tol,
                                rtol=args.rtol, max_iter=args.max_iter,
                                return_info=True),
                            args.warmup, args.trials)
                        t_b = statistics.median(bt)
                        d = float(np.max(np.abs(Theta_ref - bsol["Theta"])))
                        n_groups = binfo["n_groups"]
                        row[f"t_{bname}_median"] = t_b
                        row[f"t_{bname}_min"] = min(bt)
                        row[f"t_{bname}_json"] = json.dumps([round(t, 6) for t in bt])
                        row[f"speedup_{bname}"] = t_ref / t_b if t_b > 0 else float("nan")
                        row[f"maxdiff_{bname}"] = d
                        row["n_groups"] = n_groups
                        row["n_singletons"] = binfo["n_singletons"]
                        row["group_sizes_json"] = json.dumps(binfo["group_sizes"])
                    except Exception as e:
                        row[f"t_{bname}_median"] = float("nan")
                        row[f"speedup_{bname}"] = float("nan")
                        row[f"maxdiff_{bname}"] = float("nan")
                        print(f"  [{bname} FAILED: {type(e).__name__}: {e}]")

                line += f" {(n_groups if n_groups is not None else -1):5d} {t_ref:8.4f}"
                for bname in args.backends:
                    tb = row.get(f"t_{bname}_median", float("nan"))
                    sp = row.get(f"speedup_{bname}", float("nan"))
                    dd = row.get(f"maxdiff_{bname}", float("nan"))
                    line += f" {tb:9.4f} {sp:8.2f}x {dd:9.1e}"
                print(line)
                rows.append(row)

    if rows:
        keys = sorted({k for r in rows for k in r})
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {len(rows)} rows to {args.out}")

    print("\nmaxdiff is vs block_SGL at identical tol/rtol/max_iter/rho. Anything "
          "above ~1e-10 means the batched path is NOT solving the same problem.")
    print("grps = distinct block SIZES, i.e. number of batched calls. Batching "
          "only pays when blocks share sizes; many groups means little batching.")


if __name__ == "__main__":
    main()
