"""riboflavin_study.py — does the giant component appear on real data?

Component structure of the graphical lasso on the riboflavin data (71 samples,
top-variance genes, sample correlation as S), against a permutation null that
shuffles each gene independently, plus subsample stability, a check of screening
exactness and a check of sample independence. No timings.

    python scripts/fetch_riboflavin.py          # first, see that script
    python src/riboflavin_study.py
"""

import argparse
import contextlib
import csv
import hashlib
import io
import os
import re

_PRE = argparse.ArgumentParser(add_help=False)
_PRE.add_argument("--blas-threads", type=int, default=8)
_pre, _ = _PRE.parse_known_args()
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = str(_pre.blas_threads)

import numpy as np
from scipy.sparse.csgraph import connected_components

from provenance import write_env
from gglasso.solver.single_admm_solver import ADMM_SGL, get_connected_components


def structure(S, lam):
    """Labels from GGLasso's screening, plus block count, s_max and F_max."""
    n, comps = get_connected_components(S, lam)
    labels = np.empty(S.shape[0], int)
    for i, c in enumerate(comps):
        labels[c] = i
    sizes = np.bincount(labels).astype(float)
    return labels, n, int(sizes.max()), float(sizes.max() ** 3 / (sizes ** 3).sum())


def ari(a, b):
    """Adjusted Rand index between two labelings."""
    table = np.zeros((a.max() + 1, b.max() + 1))
    np.add.at(table, (a, b), 1)
    comb = lambda x: x * (x - 1) / 2
    both, rows, cols = comb(table).sum(), comb(table.sum(1)).sum(), comb(table.sum(0)).sum()
    expected = rows * cols / comb(float(len(a)))
    top = (rows + cols) / 2 - expected
    return 1.0 if top == 0 else float((both - expected) / top)


def permuted(X, rng, within=None):
    """Shuffle each gene across samples: keeps marginals, removes dependence.
    With within (batch labels), shuffle inside each batch only.
    """
    if within is None:
        return rng.permuted(X, axis=0)
    out = X.copy()
    for b in np.unique(within):
        out[within == b] = rng.permuted(X[within == b], axis=0)
    return out


def apply_subset(X, names, subset):
    """Sample names carry fermentation batch and time (Fbat107PT24). Returns
    the data, each row's batch, and the labels the null must respect.
    """
    batch = np.array([re.search(r"Fbat(\d+)", n).group(1) for n in names])
    hour = np.array([re.search(r"PT_?(\d+)", n).group(1) for n in names])
    if subset == "all":
        return X, batch, None
    if subset == "within_batch":
        R = X.copy()
        for b in np.unique(batch):
            R[batch == b] -= R[batch == b].mean(0)
        return R, batch, batch
    keep = hour == subset[2:]                       # pt24, pt48: one row per batch
    return X[keep], batch[keep], None


def quant(m, q):
    return np.quantile(m, q, axis=0)


def curves(X, lams, args, rng, batch, within):
    p = X.shape[1]
    S = np.corrcoef(X, rowvar=False)
    real = [structure(S, lam) for lam in lams]

    null = np.zeros((args.perms, len(lams), 3))
    for b in range(args.perms):
        Sb = np.corrcoef(permuted(X, rng, within), rowvar=False)
        for j, lam in enumerate(lams):
            null[b, j] = structure(Sb, lam)[1:]

    sub = np.zeros((args.subsamples, len(lams), 4))
    for b in range(args.subsamples):
        if args.sub_batches:                        # resample whole batches
            ids = rng.choice(np.unique(batch), args.sub_batches, replace=False)
            rows = np.isin(batch, ids)
        else:
            rows = rng.choice(len(X), args.sub_n, replace=False)
        Sb = np.corrcoef(X[rows], rowvar=False)
        for j, lam in enumerate(lams):
            lab, nb, smax, fmax = structure(Sb, lam)
            sub[b, j] = nb, smax, fmax, ari(lab, real[j][0])

    return [dict(
        p=p, **{"lambda": round(float(lam), 4)},
        n_blocks=real[j][1], s_max=real[j][2], F_max=real[j][3],
        null_blocks_med=np.median(null[:, j, 0]), null_smax_med=np.median(null[:, j, 1]),
        null_Fmax_med=np.median(null[:, j, 2]),
        null_Fmax_lo=quant(null[:, j, 2], .025), null_Fmax_hi=quant(null[:, j, 2], .975),
        sub_Fmax_med=np.median(sub[:, j, 2]),
        sub_Fmax_lo=quant(sub[:, j, 2], .025), sub_Fmax_hi=quant(sub[:, j, 2], .975),
        sub_ari_med=np.median(sub[:, j, 3]),
        sub_ari_lo=quant(sub[:, j, 3], .025), sub_ari_hi=quant(sub[:, j, 3], .975),
    ) for j, lam in enumerate(lams)]


def exactness(X, lams, args):
    """Full unscreened solve vs thresholded-S components, across the transition."""
    p = X.shape[1]
    S = np.corrcoef(X, rowvar=False)
    inside = [lam for lam in lams if 1 < structure(S, lam)[1] < p]
    picks = [inside[i] for i in np.unique(np.linspace(0, len(inside) - 1, args.exact_lams).astype(int))]
    out = []
    for lam in picks:
        labS, nS, *_ = structure(S, lam)
        with contextlib.redirect_stdout(io.StringIO()):
            sol, info = ADMM_SGL(S, float(lam), np.eye(p), tol=1e-10, rtol=1e-8,
                                 max_iter=args.exact_iter, verbose=False, measure=False)
        A = (np.abs(sol["Theta"]) > 0).astype(int)
        np.fill_diagonal(A, 1)
        nT, labT = connected_components(A, directed=False)
        same = ari(labS, labT) == 1.0
        out.append(dict(p=p, **{"lambda": round(float(lam), 4)}, n_components_threshold=nS,
                        n_components_solution=nT, identical_partition=same,
                        status=info.get("status", "") if isinstance(info, dict) else ""))
        print(f"  exact p={p} lam={lam:.2f}  {nS} vs {nT}  "
              f"{'identical' if same else 'DIFFERENT'}  {out[-1]['status']}", flush=True)
    return out


def independence(X, rng, perms, within=None):
    """Sample-by-sample correlation on standardised genes: independent samples
    look like the permuted null.
    """
    def stats(Z):
        Z = (Z - Z.mean(0)) / Z.std(0)
        C = np.corrcoef(Z)
        np.fill_diagonal(C, -1)
        ev = np.linalg.eigvalsh(C + 2 * np.eye(len(C)))[::-1]   # undo the -1 diagonal
        return np.median(C.max(1)), C.max(), ev[0] / ev.sum()

    real = stats(X)
    null = np.array([stats(permuted(X, rng, within)) for _ in range(perms)])
    C = np.corrcoef((X - X.mean(0)) / X.std(0))
    np.fill_diagonal(C, -1)
    pairs = int((np.triu(C, 1) > null[:, 1].max()).sum())
    return dict(p=X.shape[1], median_nearest_corr=real[0], null_median_nearest_corr=null[:, 0].mean(),
                max_pair_corr=real[1], null_max_pair_corr=null[:, 1].max(),
                pc1_share=real[2], null_pc1_share=null[:, 2].mean(),
                pairs_above_null_max=pairs)


def write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {path}", flush=True)


def main():
    ap = argparse.ArgumentParser(parents=[_PRE])
    ap.add_argument("--data", default="data/riboflavin/riboflavin.npz")
    ap.add_argument("--ps", nargs="+", type=int, default=[200, 500, 1000])
    ap.add_argument("--perms", type=int, default=100)
    ap.add_argument("--subsamples", type=int, default=50)
    ap.add_argument("--sub-n", type=int, default=56)
    ap.add_argument("--sub-batches", type=int, default=0,
                    help="subsample this many whole batches instead of --sub-n samples")
    ap.add_argument("--subset", default="all", choices=["all", "pt24", "pt48", "within_batch"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exact-ps", nargs="*", type=int, default=[200, 500])
    ap.add_argument("--exact-lams", type=int, default=6)
    ap.add_argument("--exact-iter", type=int, default=5000)
    ap.add_argument("--out-prefix", default="results/riboflavin")
    args = ap.parse_args()

    d = np.load(args.data)
    X_raw = d["X"]
    digest = hashlib.sha256(X_raw.tobytes()).hexdigest()
    X_all, batch, within = apply_subset(X_raw, d["samples"].astype(str), args.subset)
    print(f"subset {args.subset}: {X_all.shape[0]} samples, {len(np.unique(batch))} batches", flush=True)
    write_env(args.out_prefix + "_curve.csv", args=args, extra={"X_sha256": digest})
    order = np.argsort(-X_raw.var(0), kind="stable")   # same genes in every subset
    lams = np.arange(1, 50) * 0.02

    rng = np.random.default_rng(args.seed)
    rows, exact, indep = [], [], []
    for p in args.ps:
        X = X_all[:, order[:p]]
        rows += curves(X, lams, args, rng, batch, within)
        indep.append(independence(X, rng, args.perms, within))
        print(f"p={p}: curves done; {indep[-1]}", flush=True)
    write(args.out_prefix + "_curve.csv", rows)
    write(args.out_prefix + "_independence.csv", indep)
    for p in args.exact_ps:
        exact += exactness(X_all[:, order[:p]], lams, args)
        write(args.out_prefix + "_exactness.csv", exact)


if __name__ == "__main__":
    main()
