"""
screening.py — why the detected blocks are not the true blocks.

Two checks behind Finding 6:

1. Exactness. Thresholding |S| at lambda and taking connected components gives
   EXACTLY the vertex partition of the graphical lasso solution at that lambda
   (Witten, Friedman & Simon 2011; Mazumder & Hastie 2012). We verify it on our
   instances by solving the full problem without decomposition and comparing
   partitions. If it holds, "detection" is not a noisy step that loses blocks:
   the estimator itself has merged them.

2. Mechanism. At each best-F1 lambda, count above-threshold entries of S that
   connect two different TRUE blocks, and how many true blocks the giant
   component fuses. A few spurious cross-block entries are enough to merge
   blocks, so oracle and detected F_max diverge.

    python src/screening.py --blas-threads 8
"""

import argparse
import contextlib
import csv
import io
import os

_PRE = argparse.ArgumentParser(add_help=False)
_PRE.add_argument("--blas-threads", type=int, default=8)
_pre, _ = _PRE.parse_known_args()
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = str(_pre.blas_threads)

import numpy as np
from scipy.sparse.csgraph import connected_components

from blockgen import make_instance
from provenance import write_env
from gglasso.solver.single_admm_solver import ADMM_SGL, get_connected_components


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def best_f1(rows, keys):
    best = {}
    for r in rows:
        k = tuple(r[c] for c in keys)
        if r["support_f1"] in ("", "nan"):
            continue
        if k not in best or float(r["support_f1"]) > float(best[k]["support_f1"]):
            best[k] = r
    return list(best.values())


def partition(labels):
    groups = {}
    for i, l in enumerate(labels):
        groups.setdefault(int(l), []).append(i)
    return sorted(tuple(g) for g in groups.values())


def write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {path}")


def exactness(args):
    """Full-problem solve vs thresholded-S components at RQ1 best-F1 points."""
    rows = best_f1(read_csv(args.rq1), ("regime", "p", "n_samples"))
    rows = [r for r in rows if int(r["p"]) <= args.max_p]
    out = []
    for r in sorted(rows, key=lambda r: (r["regime"], int(r["p"]), int(r["n_samples"]))):
        p, n, lam = int(r["p"]), int(r["n_samples"]), float(r["lambda"])
        S, *_ = make_instance(r["regime"], p, n, seed=int(r["seed"]))
        nS, comps = get_connected_components(S, lam)
        labS = np.empty(p, int)
        for i, c in enumerate(comps):
            labS[c] = i
        with contextlib.redirect_stdout(io.StringIO()):
            sol, _ = ADMM_SGL(S, lam, np.eye(p), tol=1e-10, rtol=1e-8,
                              max_iter=5000, verbose=False)
        A = (np.abs(sol["Theta"]) > 0).astype(int)   # prox output: exact zeros
        np.fill_diagonal(A, 1)
        nT, labT = connected_components(A, directed=False)
        same = partition(labT) == partition(labS)
        out.append(dict(regime=r["regime"], p=p, n_over_p=n // p, seed=r["seed"],
                        **{"lambda": lam}, n_components_threshold=nS,
                        n_components_solution=nT, identical_partition=same))
        print(f"  {r['regime']:11s} p={p:4d} n/p={n // p:2d} lam={lam:.2f}  "
              f"{nS:4d} vs {nT:4d}  {'identical' if same else 'DIFFERENT'}")
    write(args.out_exact, out)
    print(f"identical partitions: {sum(o['identical_partition'] for o in out)}/{len(out)}")


def mechanism(args):
    """Cross-true-block entries at percolation best-F1 points (low-oracle regimes)."""
    rows = best_f1(read_csv(args.perc), ("regime", "p", "n_over_p", "seed"))
    rows = [r for r in rows if r["regime"] in ("balanced", "many_tiny")]
    out = []
    for r in sorted(rows, key=lambda r: (r["regime"], int(r["p"]), int(r["n_over_p"]), r["seed"])):
        p, ratio, lam = int(r["p"]), int(r["n_over_p"]), float(r["lambda"])
        S, _, blocks, sizes, _ = make_instance(r["regime"], p, ratio * p, seed=int(r["seed"]))
        lab = np.empty(p, int)
        for i, b in enumerate(blocks):
            lab[b] = i
        iu = np.triu_indices(p, 1)
        above = np.abs(S[iu]) > lam
        cross = above & (lab[iu[0]] != lab[iu[1]])
        _, comps = get_connected_components(S, lam)
        giant = max(comps, key=len)
        out.append(dict(regime=r["regime"], p=p, n_over_p=ratio, seed=r["seed"],
                        **{"lambda": lam}, above_threshold=int(above.sum()),
                        cross_block=int(cross.sum()),
                        cross_frac=float(cross.sum() / max(above.sum(), 1)),
                        true_blocks=len(sizes), giant_size=len(giant),
                        true_blocks_in_giant=len(set(lab[giant])),
                        F_max=float(r["F_max"]), oracle_F_max=float(r["oracle_F_max"])))
    write(args.out_mech, out)


def main():
    ap = argparse.ArgumentParser(parents=[_PRE])
    ap.add_argument("--rq1", default="results/rq1_primary_8t.csv")
    ap.add_argument("--perc", default="results/percolation.csv")
    ap.add_argument("--max-p", type=int, default=400,
                    help="full unscreened solves at tol 1e-10 get slow beyond this")
    ap.add_argument("--out-exact", default="results/screening_exactness.csv")
    ap.add_argument("--out-mech", default="results/screening_cross_block.csv")
    args = ap.parse_args()
    write_env(args.out_exact, args=args)
    write_env(args.out_mech, args=args)
    exactness(args)
    mechanism(args)


if __name__ == "__main__":
    main()
