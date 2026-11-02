"""riboflavin_stars.py — choose lambda on the riboflavin data by StARS (Liu,
Roeder & Wasserman 2010): the smallest lambda, walking down the grid, whose
edge instability stays at or below beta. Subsamples are random samples or whole
fermentation batches, since the samples are not independent. No timings.

    python src/riboflavin_stars.py
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

from provenance import write_env
from riboflavin_study import apply_subset, structure
from gglasso.solver.single_admm_solver import block_SGL


def support(S, lam, tol, rtol, max_iter):
    p = S.shape[0]
    with contextlib.redirect_stdout(io.StringIO()):
        sol = block_SGL(S, float(lam), np.eye(p), tol=tol, rtol=rtol,
                        max_iter=max_iter, verbose=False)
    A = sol["Theta"] != 0
    np.fill_diagonal(A, False)
    return A


def main():
    ap = argparse.ArgumentParser(parents=[_PRE])
    ap.add_argument("--data", default="data/riboflavin/riboflavin.npz")
    ap.add_argument("--ps", nargs="+", type=int, default=[200, 500, 1000])
    ap.add_argument("--modes", nargs="+", default=["random", "batch"])
    ap.add_argument("--subsamples", type=int, default=20)
    ap.add_argument("--sub-n", type=int, default=56)
    ap.add_argument("--sub-batches", type=int, default=22)
    ap.add_argument("--beta", type=float, default=0.05)
    ap.add_argument("--lam-min", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tol", type=float, default=1e-7)
    ap.add_argument("--rtol", type=float, default=1e-5)
    ap.add_argument("--max-iter", type=int, default=1000)
    ap.add_argument("--out", default="results/riboflavin_stars.csv")
    args = ap.parse_args()
    write_env(args.out, args=args)

    d = np.load(args.data)
    X_all, batch, _ = apply_subset(d["X"], d["samples"].astype(str), "all")
    order = np.argsort(-X_all.var(0), kind="stable")
    lams = np.arange(49, 0, -1) * 0.02                  # high to low, step 0.02
    lams = lams[lams >= args.lam_min - 1e-9]

    rows = []
    rng = np.random.default_rng(args.seed)
    for p in args.ps:
        X = X_all[:, order[:p]]
        S = np.corrcoef(X, rowvar=False)
        pairs = p * (p - 1)                             # ordered, as A is symmetric
        for mode in args.modes:
            subs = []
            for _ in range(args.subsamples):
                if mode == "batch":
                    ids = rng.choice(np.unique(batch), args.sub_batches, replace=False)
                    subs.append(np.flatnonzero(np.isin(batch, ids)))
                else:
                    subs.append(rng.choice(len(X), args.sub_n, replace=False))
            chosen, sup = None, 0.0
            for lam in lams:
                theta = np.zeros((p, p))
                for idx in subs:
                    theta += support(np.corrcoef(X[idx], rowvar=False), lam,
                                     args.tol, args.rtol, args.max_iter)
                theta /= len(subs)
                D = float((2 * theta * (1 - theta)).sum() / pairs)
                sup = max(sup, D)
                _, nb, smax, fmax = structure(S, lam)
                stop = sup > args.beta
                if not stop:
                    chosen = float(lam)
                rows.append(dict(p=p, mode=mode, **{"lambda": round(float(lam), 4)},
                                 instability=D, sup_instability=sup, within_beta=not stop,
                                 n_blocks=nb, s_max=smax, F_max=fmax))
                print(f"p={p} {mode} lam={lam:.2f} D={D:.4f} blocks={nb} s_max={smax} "
                      f"F_max={fmax:.3f}", flush=True)
                if stop:
                    break
            for r in rows:
                if r["p"] == p and r["mode"] == mode:
                    r["selected"] = chosen is not None and abs(r["lambda"] - chosen) < 1e-9
            print(f"p={p} {mode}: StARS lambda = {chosen}", flush=True)

            with open(args.out, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(r for r in rows if "selected" in r)


if __name__ == "__main__":
    main()
