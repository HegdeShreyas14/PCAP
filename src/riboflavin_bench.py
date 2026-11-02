"""riboflavin_bench.py — the four solve paths on the riboflavin data at the StARS
lambda and one grid step either side. Settings match the synthetic sweep.

    python src/riboflavin_bench.py --blas-threads 8
"""

import argparse
import contextlib
import csv
import io
import json
import os
import time

_PRE = argparse.ArgumentParser(add_help=False)
_PRE.add_argument("--blas-threads", type=int, default=8)
_pre, _ = _PRE.parse_known_args()
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = str(_pre.blas_threads)

import numpy as np

from batched_admm import batched_block_SGL
from provenance import write_env
from riboflavin_study import structure
from gglasso.solver.single_admm_solver import ADMM_SGL, block_SGL


def timed(fn, warmup, trials):
    with contextlib.redirect_stdout(io.StringIO()):
        for _ in range(warmup):
            fn()
        times, out = [], None
        for _ in range(trials):
            t0 = time.perf_counter()
            out = fn()
            times.append(time.perf_counter() - t0)
    return times, out


def objective(Theta, S, lam):
    sign, logdet = np.linalg.slogdet(Theta)
    if sign <= 0:
        return float("inf")
    return float(-logdet + np.trace(S @ Theta)
                 + lam * (np.abs(Theta).sum() - np.abs(np.diag(Theta)).sum()))


def main():
    ap = argparse.ArgumentParser(parents=[_PRE])
    ap.add_argument("--data", default="data/riboflavin/riboflavin.npz")
    ap.add_argument("--stars", default="results/riboflavin_stars.csv")
    ap.add_argument("--ps", nargs="+", type=int, default=[500, 1000])
    ap.add_argument("--step", type=float, default=0.02)
    ap.add_argument("--lam", nargs="*", default=["1000:0.10"],
                    help="p:lambda for sizes with no StARS run; recorded as lam_source=assigned")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--trials", type=int, default=9)
    ap.add_argument("--out", default="results/riboflavin_bench.csv")
    args = ap.parse_args()
    write_env(args.out, args=args)

    with open(args.stars) as f:
        chosen = {int(r["p"]): float(r["lambda"]) for r in csv.DictReader(f)
                  if r["selected"] == "True" and r["mode"] == "random"}
    assigned = {int(a.split(":")[0]): float(a.split(":")[1]) for a in args.lam}
    chosen.update(assigned)
    X_all = np.load(args.data)["X"]
    order = np.argsort(-X_all.var(0), kind="stable")
    kw = dict(tol=1e-7, rtol=1e-5, max_iter=1000)

    rows = []
    for p in args.ps:
        S = np.corrcoef(X_all[:, order[:p]], rowvar=False)
        Omega_0 = np.eye(p)
        for offset in (0, -1, 1):
            lam = round(chosen[p] + offset * args.step, 4)
            _, nb, smax, fmax = structure(S, lam)
            paths = dict(
                ADMM_SGL=lambda: ADMM_SGL(S, lam, Omega_0, verbose=False, **kw),
                block_SGL=lambda: block_SGL(S, lam, Omega_0, verbose=False, **kw),
                batched_numpy=lambda: batched_block_SGL(S, lam, backend="numpy", return_info=True, **kw),
                batched_torch=lambda: batched_block_SGL(S, lam, backend="torch", return_info=True, **kw),
            )
            ref = t_ref = None
            for name, fn in paths.items():
                times, out = timed(fn, args.warmup, args.trials)
                info = out[1] if isinstance(out, tuple) else {}
                sol = out[0] if isinstance(out, tuple) else out
                t = float(np.median(times))
                if ref is None:
                    ref, t_ref = sol["Theta"], t
                r = dict(p=p, lam_source="assigned" if p in assigned else "stars", lam_offset=offset, **{"lambda": lam}, n_blocks=nb, s_max=smax,
                         F_max=fmax, path=name, t_median=t, t_min=min(times), t_max=max(times),
                         speedup_vs_admm=t_ref / t,
                         theta_rel_diff=float(np.abs(sol["Theta"] - ref).max() / np.abs(ref).max()),
                         obj_rel_gap=abs(objective(sol["Theta"], S, lam) - objective(ref, S, lam))
                         / abs(objective(ref, S, lam)),
                         status=info.get("status", ""),
                         not_converged=info.get("n_not_converged", ""),
                         times_json=json.dumps([round(x, 6) for x in times]))
                rows.append(r)
                print(f"p={p} lam={lam:.2f} blocks={nb} s_max={smax} {name:14s} {t:9.3f}s "
                      f"[{min(times):.3f},{max(times):.3f}] x{r['speedup_vs_admm']:.2f} "
                      f"obj_gap={r['obj_rel_gap']:.1e} {r['status']} nc={r['not_converged']}", flush=True)
                with open(args.out, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    w.writeheader()
                    w.writerows(rows)


if __name__ == "__main__":
    main()
