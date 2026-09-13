"""
percolation.py — locate the percolation threshold and compare it to the
statistically optimal lambda.

The question
------------
analyze.py's Table 4 reports that the largest detected block typically holds
~99% of the cubic arithmetic, so inter-block headroom H = 1 - F_max is near
zero and there is little for a GPU to parallelise across blocks. This script
asks why, and whether it is avoidable.

The mechanism
-------------
Thresholding the empirical covariance at lambda and taking connected
components is a percolation process. Below a critical lambda the graph has a
giant component holding most variables; above it the giant shatters into
comparable fragments. Because block cost scales as s^3, a surviving giant
holds essentially all the work no matter how many crumbs surround it.

IMPORTANT: F_max is a CUBIC-WORK fraction, not a node fraction. The two
diverge sharply. A graph can be shattered node-wise -- hundreds of components
-- while one moderate block still holds 99% of the s^3 work. The output
reports both (`F_max` and `max_frac_of_p`) because the work fraction is what
bounds parallel speedup and the node fraction is what looks impressive.

The tension
-----------
The statistically optimal lambda falls as p grows. The percolation threshold
does not fall as fast. Above some p the optimal lambda sits BELOW the
threshold, the giant survives at every usable lambda, and inter-block
parallelism is unavailable at any defensible operating point. That crossing,
if it exists, limits the approach rather than the implementation.

Usage
-----
    python src/percolation.py --seeds 0 1 2 --out results/percolation.csv
    python src/percolation.py --no-f1            # structure only; NO verdict
"""

import argparse
import contextlib
import csv
import io
import json
import os
from pathlib import Path
import statistics
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# BLAS threading must be pinned BEFORE numpy is imported or the setting is
# ignored. Same pre-parser as sweep.py: an unpinned run inherits whatever the
# ambient thread count is, which is not reproducible and, on a 16-core chip
# asked for 32 threads, is pathological.
# ---------------------------------------------------------------------------
def _set_blas_threads(n):
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(n)


_PRE = argparse.ArgumentParser(add_help=False)
_PRE.add_argument("--blas-threads", type=int, default=1)
_pre, _ = _PRE.parse_known_args()
_set_blas_threads(_pre.blas_threads)

import numpy as np

from blockgen import make_instance, REGIMES
from gglasso.solver.single_admm_solver import get_connected_components, block_SGL


@contextlib.contextmanager
def _silence():
    """block_SGL prints one line per block; at 200+ blocks over a 25-point
    grid that is tens of thousands of lines."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def cubic_summary(p, sizes):
    tot = sum(float(s) ** 3 for s in sizes)
    mx = float(max(sizes)) ** 3 if sizes else float("nan")
    return {
        "n_blocks": len(sizes),
        "max_block": max(sizes) if sizes else 0,
        "max_frac_of_p": (max(sizes) / p) if sizes else float("nan"),
        "F_max": (mx / tot) if tot > 0 else float("nan"),
        "headroom": (1 - mx / tot) if tot > 0 else float("nan"),
        "cubic_gain": (p ** 3 / tot) if tot > 0 else float("nan"),
        "ceiling": (p ** 3 / mx) if mx == mx and mx > 0 else float("nan"),
    }


def structure_at(S, p, lam):
    numC, allC = get_connected_components(S, lam)
    sizes = sorted((len(c) for c in allC), reverse=True)
    return cubic_summary(p, sizes)


def support_f1(Theta_hat, Theta_true, tol=1e-6):
    p = Theta_hat.shape[0]
    off = ~np.eye(p, dtype=bool)
    est = (np.abs(Theta_hat) > tol) & off
    tru = (np.abs(Theta_true) > tol) & off
    tp = float((est & tru).sum()); fp = float((est & ~tru).sum()); fn = float((~est & tru).sum())
    pr = tp / (tp + fp) if tp + fp else float("nan")
    rc = tp / (tp + fn) if tp + fn else float("nan")
    return (2 * pr * rc / (pr + rc)) if pr == pr and rc == rc and pr + rc > 0 else float("nan")


def percolation_lambda(records, threshold):
    """Smallest lambda where F_max drops below `threshold`. nan if never."""
    for r in sorted(records, key=lambda x: x["lambda"]):
        if r["F_max"] == r["F_max"] and r["F_max"] < threshold:
            return r["lambda"]
    return float("nan")


def f(x, w=9, d=3):
    return (" " * max(0, w - 3) + "n/a") if x != x else f"{x:{w}.{d}f}"


def main():
    ap = argparse.ArgumentParser(parents=[_PRE])
    ap.add_argument("--regimes", nargs="+", default=REGIMES)
    ap.add_argument("--ps", nargs="+", type=int, default=[100, 200, 400, 800])
    ap.add_argument("--ratios", nargs="+", type=int, default=[2, 5, 10])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2],
                    help="percolation thresholds fluctuate at finite size; "
                         "3-5 seeds are needed before quoting a threshold")
    ap.add_argument("--lam-min", type=float, default=0.02)
    ap.add_argument("--lam-max", type=float, default=0.50)
    ap.add_argument("--lam-steps", type=int, default=25)
    ap.add_argument("--f-max-threshold", type=float, default=0.90)
    # Solver settings default to sweep.py's so lam*(F1) is directly comparable
    # with the best-F1 lambda in rq1_primary_*.csv. Do not change one without
    # the other.
    ap.add_argument("--tol", type=float, default=1e-7)
    ap.add_argument("--rtol", type=float, default=1e-5)
    ap.add_argument("--max-iter", type=int, default=1000)
    ap.add_argument("--out", default="results/percolation.csv")
    ap.add_argument("--no-f1", action="store_true",
                    help="structure only, no solving: FAST but yields NO verdict")
    ap.add_argument("--resume", action="store_true",
                    help="resume lambda records from OUT.partial.jsonl")
    args = ap.parse_args()

    if args.no_f1:
        print("!! --no-f1: lambda*(F1) cannot be computed, so every verdict will be")
        print("!! 'indeterminate'. Use this only to inspect structure. The paper")
        print("!! figure requires a full run.\n")

    lams = np.linspace(args.lam_min, args.lam_max, args.lam_steps)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(f"{args.out}.partial.jsonl")
    rows = []
    if args.resume and checkpoint.exists():
        with checkpoint.open() as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        if not args.no_f1:
            rows = [r for r in rows if r.get("f1_computed", True)]
        print(f"Resuming from {len(rows)} checkpointed lambda records")

    print(f"BLAS threads: {os.environ.get('OMP_NUM_THREADS')}   "
          f"seeds: {args.seeds}   tol={args.tol} rtol={args.rtol}\n")
    print(f"{'regime':12s} {'p':>5s} {'n/p':>4s} {'lam_perc':>16s} {'lam*(F1)':>16s} "
          f"{'F_max@lam*':>11s} {'oracle':>8s} {'verdict':>22s}")
    print("-" * 108)

    for regime in args.regimes:
        for p in args.ps:
            for ratio in args.ratios:
                perc_by_seed, star_by_seed, fmax_by_seed = [], [], []
                oracle_fmax = float("nan")

                for seed in args.seeds:
                    S, Theta, blocks, sizes, styles = make_instance(
                        regime, p, ratio * p, seed=seed)
                    S = S[0] if S.ndim == 3 else S
                    Omega_0 = np.eye(p)

                    # Oracle: F_max the TRUE block structure would give if
                    # detection were perfect. If oracle F_max is low but the
                    # detected F_max is high, detection is merging true blocks
                    # (finite-sample noise), not the ground truth lacking
                    # parallelism. That distinction matters before blaming the
                    # generator.
                    if seed == args.seeds[0]:
                        oracle_fmax = cubic_summary(p, sorted(sizes, reverse=True))["F_max"]

                    recs = [r for r in rows
                            if r["regime"] == regime and int(r["p"]) == p
                            and int(r["n_over_p"]) == ratio and int(r["seed"]) == seed
                            and (args.no_f1 or r.get("f1_computed", True))]
                    completed_lams = {float(r["lambda"]) for r in recs}
                    for lam in lams:
                        if float(lam) in completed_lams:
                            continue
                        st = structure_at(S, p, float(lam))
                        st.update(regime=regime, p=p, n_over_p=ratio, seed=seed,
                                  blas_threads=os.environ.get("OMP_NUM_THREADS"),
                                  tol=args.tol, rtol=args.rtol,
                                  oracle_F_max=oracle_fmax,
                                  true_n_blocks=len(sizes),
                                  f1_computed=not args.no_f1,
                                  **{"lambda": float(lam)})
                        if not args.no_f1:
                            try:
                                with _silence():
                                    sol = block_SGL(S, float(lam), Omega_0,
                                                    tol=args.tol, rtol=args.rtol,
                                                    max_iter=args.max_iter,
                                                    verbose=False, measure=False)
                                st["support_f1"] = support_f1(sol["Theta"], Theta)
                            except Exception:
                                st["support_f1"] = float("nan")
                        else:
                            st["support_f1"] = float("nan")
                        recs.append(st)
                        rows.append(st)
                        with checkpoint.open("a") as handle:
                            handle.write(json.dumps(st) + "\n")
                            handle.flush()

                    perc_by_seed.append(percolation_lambda(recs, args.f_max_threshold))
                    usable = [r for r in recs if r["support_f1"] == r["support_f1"]]
                    if usable:
                        best = max(usable, key=lambda r: r["support_f1"])
                        star_by_seed.append(best["lambda"])
                        fmax_by_seed.append(best["F_max"])

                def agg(vals):
                    good = [v for v in vals if v == v]
                    if not good:
                        return float("nan"), float("nan")
                    return (statistics.median(good),
                            (max(good) - min(good)) if len(good) > 1 else 0.0)

                lp, lp_spread = agg(perc_by_seed)
                ls, ls_spread = agg(star_by_seed)
                fm, _ = agg(fmax_by_seed)
                n_nan_perc = sum(1 for v in perc_by_seed if v != v)

                if ls != ls:
                    verdict = "indeterminate"
                elif lp != lp:
                    # never percolates on this grid: the giant survives at
                    # every lambda tested, which is the strongest form of the
                    # negative result
                    verdict = "NEVER percolates"
                elif ls < lp:
                    verdict = "NO inter-block work"
                elif abs(ls - lp) < 1e-12:
                    verdict = "at threshold (tie)"
                else:
                    verdict = "headroom available"

                note = f" [{n_nan_perc}/{len(args.seeds)} seeds n/a]" if n_nan_perc else ""
                print(f"{regime:12s} {p:5d} {ratio:4d} "
                      f"{f(lp)}±{f(lp_spread,5,3)} {f(ls)}±{f(ls_spread,5,3)} "
                      f"{f(100*fm,10,1)}% {f(100*oracle_fmax,7,1)}% {verdict:>22s}{note}")

    if rows:
        keys = sorted({k for r in rows for k in r})
        with out_path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        checkpoint.unlink(missing_ok=True)
        print(f"\nWrote {len(rows)} rows to {args.out}")

    print("""
Reading this table
------------------
lam_perc     median over seeds of the smallest lambda where the giant component
             breaks (F_max < threshold), with min-max spread. A large spread
             means the threshold is not resolved and more seeds are needed.
lam*(F1)     lambda maximising support-F1: the defensible operating point.
F_max@lam*   share of CUBIC work in the largest block at that lambda. This is
             a work fraction, not a node fraction -- a graph can be shattered
             into hundreds of components and still have F_max near 1.
oracle       F_max the TRUE block structure would give under perfect detection.
             If oracle is LOW but F_max@lam* is HIGH, finite-sample noise is
             merging true blocks. If oracle is also HIGH, the ground truth
             itself has no inter-block parallelism and that is a property of
             the regime, not a detection failure.

verdicts
  headroom available   lam* sits above the percolation threshold: real
                       inter-block work exists at a usable lambda.
  NO inter-block work  lam* sits below it: at every lambda worth using the
                       giant survives.
  NEVER percolates     the giant survives across the whole lambda grid.
  at threshold (tie)   lam* == lam_perc; treat as marginal, not as headroom.

If the verdict flips from 'headroom available' to 'NO inter-block work' as p
grows, that crossing bounds the block-parallel approach itself.""")


if __name__ == "__main__":
    main()
