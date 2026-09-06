"""
profile_block.py — attribute block_SGL runtime for ONE configuration.

Splits a single block_SGL call into:

    eigh        time in np.linalg.eigh / eigvalsh   (the decomposition arithmetic)
    detect      time in get_connected_components    (building the block structure)
    assemble    time in scipy block_diag            (stitching blocks back together)
    glue        everything else: the Python for-loop over components, per-block
                S[ix_(C,C)] slicing, ADMM_SGL setup + per-block mask allocation,
                and the inverse-permutation indexing

Why this exists
---------------
analyze.py TABLE 4 says the largest detected block usually holds ~99% of the
cubic work, so inter-block parallelism looks scarce. But "cubic work" is an
arithmetic model. What actually runs is hundreds of small ADMM_SGL calls, each
with fixed Python overhead. If the fragmented regime is dominated by that
overhead rather than by eigh, then:

  * a GPU that only accelerates the eigendecomposition kernel will not help it,
  * but batching the many small blocks into one call will,

and the Phase 3 design is a hybrid — batch the fragments, accelerate the giant —
backed by measurement instead of assumption.

Run the two poles and compare the `eigh` fraction:

    python src/profile_block.py --regime imbalanced --p 200 --ratio 10 --lam 0.15
    python src/profile_block.py --regime balanced   --p 800 --ratio 10 --lam 0.10
"""

import argparse
import os

# --- BLAS threads must be pinned before numpy is imported ------------------
_PRE = argparse.ArgumentParser(add_help=False)
_PRE.add_argument("--blas-threads", type=int, default=1)
_pre, _ = _PRE.parse_known_args()
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = str(_pre.blas_threads)

import contextlib
import cProfile
import io
import pstats
import statistics
import time

import numpy as np

import gglasso.solver.single_admm_solver as sas
from gglasso.solver.single_admm_solver import block_SGL, get_connected_components

from blockgen import make_instance


# --------------------------------------------------------------------------
# instrumentation: wrap the calls we want to attribute time to
# --------------------------------------------------------------------------

class Tally:
    def reset(self):
        self.eigh_n = 0
        self.eigh_t = 0.0
        self.eigh_times = []          # per-call seconds
        self.eigh_dims = []           # per-call matrix dimension
        self.detect_t = 0.0
        self.detect_n = 0
        self.assemble_t = 0.0
        self.assemble_n = 0


T = Tally()
T.reset()

_orig_eigh = np.linalg.eigh
_orig_eigvalsh = np.linalg.eigvalsh
_orig_ccomp = sas.get_connected_components
_orig_block_diag = sas.block_diag


def _timed_eigh(a, *args, **kw):
    t0 = time.perf_counter()
    out = _orig_eigh(a, *args, **kw)
    dt = time.perf_counter() - t0
    T.eigh_n += 1
    T.eigh_t += dt
    T.eigh_times.append(dt)
    T.eigh_dims.append(int(np.asarray(a).shape[-1]))
    return out


def _timed_eigvalsh(a, *args, **kw):
    t0 = time.perf_counter()
    out = _orig_eigvalsh(a, *args, **kw)
    dt = time.perf_counter() - t0
    T.eigh_n += 1
    T.eigh_t += dt
    T.eigh_times.append(dt)
    T.eigh_dims.append(int(np.asarray(a).shape[-1]))
    return out


def _timed_ccomp(*args, **kw):
    t0 = time.perf_counter()
    out = _orig_ccomp(*args, **kw)
    T.detect_t += time.perf_counter() - t0
    T.detect_n += 1
    return out


def _timed_block_diag(*args, **kw):
    t0 = time.perf_counter()
    out = _orig_block_diag(*args, **kw)
    T.assemble_t += time.perf_counter() - t0
    T.assemble_n += 1
    return out


def _patch():
    np.linalg.eigh = _timed_eigh
    np.linalg.eigvalsh = _timed_eigvalsh
    sas.get_connected_components = _timed_ccomp
    sas.block_diag = _timed_block_diag


def _unpatch():
    np.linalg.eigh = _orig_eigh
    np.linalg.eigvalsh = _orig_eigvalsh
    sas.get_connected_components = _orig_ccomp
    sas.block_diag = _orig_block_diag


# --------------------------------------------------------------------------

def ms(x):
    return f"{x * 1e3:.2f} ms"


def pct(x, whole):
    return f"{100 * x / whole:5.1f}%" if whole > 0 else "  n/a"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--regime", required=True)
    ap.add_argument("--p", type=int, required=True)
    ap.add_argument("--ratio", type=int, default=10, help="n_samples = ratio * p")
    ap.add_argument("--lam", type=float, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--tol", type=float, default=1e-7)
    ap.add_argument("--rtol", type=float, default=1e-5)
    ap.add_argument("--max-iter", type=int, default=1000)
    ap.add_argument("--blas-threads", type=int, default=1)  # consumed by pre-parser
    args = ap.parse_args()

    S, Theta_true, _, true_sizes, _ = make_instance(
        args.regime, args.p, args.ratio * args.p, seed=args.seed)
    S = S[0] if S.ndim == 3 else S
    Omega_0 = np.eye(args.p)
    common = dict(tol=args.tol, rtol=args.rtol, max_iter=args.max_iter,
                  verbose=False, measure=False)

    # detected structure (untimed, for context)
    numC, allC = get_connected_components(S, args.lam)
    det = sorted((len(c) for c in allC), reverse=True)
    n_single = sum(1 for s in det if s == 1)
    cubic = sum(float(s) ** 3 for s in det)
    f_max = (det[0] ** 3 / cubic) if cubic else float("nan")

    print(f"config      : {args.regime}  p={args.p}  n/p={args.ratio}  "
          f"lam={args.lam}  seed={args.seed}  blas_threads={os.environ['OMP_NUM_THREADS']}")
    print(f"true blocks : {len(true_sizes)}  (max {max(true_sizes)})")
    print(f"detected    : {numC} blocks, {n_single} singletons, "
          f"largest {det[0]}  ->  F_max(cubic) = {f_max:.3f}")
    print(f"solver      : tol={args.tol} rtol={args.rtol} max_iter={args.max_iter}  "
          f"repeats={args.repeats} (warmup {args.warmup})")
    print("-" * 78)

    _patch()
    _sink = io.StringIO()  # gglasso prints one "ADMM terminated" line per block
    try:
        with contextlib.redirect_stdout(_sink):
            for _ in range(args.warmup):
                block_SGL(S, args.lam, Omega_0, **common)

            walls, per = [], []
            for _ in range(args.repeats):
                T.reset()
                t0 = time.perf_counter()
                block_SGL(S, args.lam, Omega_0, **common)
                wall = time.perf_counter() - t0
                walls.append(wall)
                per.append((wall, T.eigh_n, T.eigh_t, T.detect_t, T.assemble_t,
                            list(T.eigh_times), list(T.eigh_dims)))

            # profile once for a function-level table
            T.reset()
            pr = cProfile.Profile()
            pr.enable()
            block_SGL(S, args.lam, Omega_0, **common)
            pr.disable()
    finally:
        _unpatch()

    # pick the median-wall repeat as representative
    per.sort(key=lambda r: r[0])
    wall, eigh_n, eigh_t, detect_t, assemble_t, eigh_times, eigh_dims = per[len(per) // 2]
    glue = wall - eigh_t - detect_t - assemble_t

    print(f"wall (median of {args.repeats})   {ms(wall)}     "
          f"[min {ms(min(walls))}, max {ms(max(walls))}]")
    print()
    print(f"  eigh        {ms(eigh_t):>12}   {pct(eigh_t, wall)}   {eigh_n} calls")
    print(f"  detect      {ms(detect_t):>12}   {pct(detect_t, wall)}")
    print(f"  assemble    {ms(assemble_t):>12}   {pct(assemble_t, wall)}   (block_diag only)")
    print(f"  glue        {ms(glue):>12}   {pct(glue, wall)}   "
          f"(loop + slicing + ADMM_SGL setup + inverse-perm indexing)")
    print()

    if eigh_times:
        et = sorted(eigh_times)
        biggest = et[-1]
        print(f"  eigh calls  : mean {ms(statistics.fmean(et))}  "
              f"median {ms(et[len(et) // 2])}  "
              f"p90 {ms(et[int(0.9 * len(et)) - 1])}  max {ms(biggest)}")
        print(f"  largest call: {ms(biggest)}  = {pct(biggest, eigh_t).strip()} of all eigh time  "
              f"(dim {max(eigh_dims)})")
        # dimension histogram
        buckets = [(1, 2), (3, 8), (9, 20), (21, 50), (51, 150), (151, 10 ** 9)]
        hist = []
        for lo, hi in buckets:
            k = sum(1 for d in eigh_dims if lo <= d <= hi)
            if k:
                hist.append(f"{lo}-{hi if hi < 10**9 else ''}:{k}")
        print(f"  eigh dims   : " + "  ".join(hist))
    print("-" * 78)

    # verdict — the Phase 3 question is "is there a giant kernel to accelerate,
    # or is it death by a thousand small calls + Python overhead?"
    eigh_frac = eigh_t / wall if wall else float("nan")
    conc = (max(eigh_times) / eigh_t) if eigh_times and eigh_t else float("nan")
    mean_call = statistics.fmean(eigh_times) if eigh_times else float("nan")
    if conc > 0.5:
        v = ("ARITHMETIC-BOUND, CONCENTRATED — one eigh call is "
             f"{conc:.0%} of all eigh time.\n            Accelerating that single "
             "kernel is the win; the fragments are noise.")
    elif eigh_frac < 0.5:
        v = ("GLUE / DISPATCH-BOUND — eigh is a minority of wall and no single "
             "call dominates.\n            A faster eig kernel cannot move this; "
             "the win is BATCHING the per-block work.")
    else:
        v = ("OVERHEAD-BOUND, NO DOMINANT KERNEL — eigh is the plurality of wall "
             f"but spread over {eigh_n} sub-ms calls\n            (largest {conc:.0%}). "
             "Batching amortizes both the eig calls and the per-iteration Python cost.")
    print(f"verdict     : {v}")
    print(f"              eigh {eigh_frac:.0%} of wall | largest eigh {conc:.1%} of eigh "
          f"| mean call {ms(mean_call)} over {eigh_n} calls")
    print("-" * 78)

    print("cProfile (top 12 by cumulative time):")
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(12)
    for line in s.getvalue().splitlines():
        if line.strip():
            print("  " + line)


if __name__ == "__main__":
    main()
