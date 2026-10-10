"""
fp32_diagnostic.py — how much of the GPU result is the cost of FP64?

Consumer GPUs run float64 at a small fraction of their float32 rate, and the
batched torch solver defaults to float64. This diagnostic re-runs four
existing crossover points with three variants under the crossover protocol
(same instance, same lambda, 2 warm-up + 5 timed, median):

    numpy float64   CPU baseline (batched)
    torch float64   GPU reference
    torch float32   diagnostic

and judges float32 against PRE-REGISTERED criteria (CRITERIA below, committed
before any float32 result existed). float32 is not held to the float64
solver's ~1e-14 agreement: its machine epsilon is ~1.2e-7. It is compared to
the float64 result and to a high-accuracy float64 solve, because the float64
result itself only satisfies the solver tolerance (rtol 1e-5), which sets
the natural error scale.

    python src/fp32_diagnostic.py --blas-threads 8
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

from blockgen import make_instance
from batched_admm import batched_block_SGL
from provenance import write_env

# ---------------------------------------------------------------------------
# Pre-registered acceptance criteria for float32 (fixed before running).
#   Solver settings are NOT relaxed for float32: same tol/rtol/max_iter.
# ---------------------------------------------------------------------------
CRITERIA = {
    # every block problem meets the reference stopping test within max_iter
    "all_converged": True,
    # |obj32 - obj64| / |obj64|, objective evaluated in float64 on the same S
    "max_obj_rel_diff": 1e-6,
    # max|Theta32 - Theta64| / max|Theta64|
    "max_theta_rel_err": 1e-3,
    # float32's distance to the high-accuracy solution may be at most this
    # multiple of float64's own distance at the same stopping tolerance
    "max_err_ratio_vs_fp64": 10.0,
    # fraction of off-diagonal entries whose zero/nonzero status differs
    "max_support_disagreement": 1e-3,
    # support flips where the float64 entry is clearly nonzero (|.| > 1e-4):
    # a flip there is a real structural change, not a near-zero coin toss
    "max_nonmarginal_flips": 0,
    # |F1_32 - F1_64| against the true precision matrix
    "max_f1_abs_diff": 0.005,
}
MARGIN = 1e-4


def glasso_objective(Theta, S, lam):
    """-log det + tr(S Theta) + lam * off-diagonal L1 (gglasso's problem)."""
    sign, logdet = np.linalg.slogdet(Theta)
    if sign <= 0:
        return float("inf")
    off = np.abs(Theta).sum() - np.abs(np.diag(Theta)).sum()
    return float(-logdet + np.trace(S @ Theta) + lam * off)


def support_f1(Theta_hat, Theta_true):
    off = ~np.eye(Theta_hat.shape[0], dtype=bool)
    est, tru = (Theta_hat != 0) & off, (np.abs(Theta_true) > 1e-6) & off
    tp, fp, fn = (est & tru).sum(), (est & ~tru).sum(), (~est & tru).sum()
    return float(2 * tp / (2 * tp + fp + fn)) if tp else 0.0


def timed(fn, warmup, trials):
    for _ in range(warmup):
        fn()
    times, result = [], None
    for _ in range(trials):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times)), times, result


def pick_configs(path, p, targets):
    """Existing crossover points at this p nearest each target s_max."""
    with open(path) as f:
        rows = [r for r in csv.DictReader(f) if int(r["p"]) == p]
    picked = []
    for t in targets:
        r = min(rows, key=lambda r: abs(int(r["s_max"]) - t))
        if r not in picked:
            picked.append(r)
    return picked


def main():
    ap = argparse.ArgumentParser(parents=[_PRE])
    ap.add_argument("--crossover-csv", default="results/gpu_crossover.csv")
    ap.add_argument("--crossover-env", default="results/gpu_crossover_env.json")
    ap.add_argument("--p", type=int, default=2000)
    ap.add_argument("--targets", nargs="+", type=int, default=[100, 500, 1100, 2000])
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--out", default="results/fp32_diagnostic.csv")
    args = ap.parse_args()

    # same instance definition the crossover used
    xargs = json.load(open(args.crossover_env))["args"]
    regime = xargs["regime"][0]
    ratio, seed = int(xargs["ratio"]), int(xargs["seed"])
    tol, rtol, max_iter = 1e-7, 1e-5, 1000          # crossover/sweep settings

    print(f"environment -> {write_env(args.out, args=args, extra={'criteria': CRITERIA})}")
    print("pre-registered criteria:", json.dumps(CRITERIA))

    S, Theta_true, *_ = make_instance(regime, args.p, ratio * args.p, seed=seed)
    configs = pick_configs(args.crossover_csv, args.p, args.targets)
    rows = []
    for c in configs:
        lam = float(c["lam"])
        run = lambda backend, dtype, **kw: batched_block_SGL(
            S, lam, backend=backend, dtype=dtype, tol=kw.get("tol", tol),
            rtol=kw.get("rtol", rtol), max_iter=kw.get("max_iter", max_iter),
            return_info=True)

        t64c, tr64c, (sol64c, inf64c) = timed(lambda: run("numpy", "float64"), args.warmup, args.trials)
        t64g, tr64g, (sol64g, inf64g) = timed(lambda: run("torch", "float64"), args.warmup, args.trials)
        t32g, tr32g, (sol32g, inf32g) = timed(lambda: run("torch", "float32"), args.warmup, args.trials)
        # same instance as the committed crossover row?
        assert inf64c["n_blocks"] == int(c["n_blocks"]), "instance differs from crossover CSV"

        # high-accuracy float64 solution (untimed) sets the error scale
        solx, infx = run("numpy", "float64", tol=1e-10, rtol=1e-8, max_iter=20000)

        T64, T64g, T32, Tx = (sol64c["Theta"], sol64g["Theta"], sol32g["Theta"], solx["Theta"])
        scale = np.abs(T64).max()
        off = ~np.eye(args.p, dtype=bool)
        z64, z32 = (T64 != 0) & off, (T32 != 0) & off
        flips = z64 != z32
        nonmarginal = flips & ((np.abs(T64) > MARGIN) | (np.abs(T32) > MARGIN))
        obj64, obj32 = glasso_objective(T64, S, lam), glasso_objective(T32, S, lam)
        err64_x = np.abs(T64 - Tx).max() / scale
        err32_x = np.abs(T32 - Tx).max() / scale
        f1_64, f1_32 = support_f1(T64, Theta_true), support_f1(T32, Theta_true)

        r = dict(
            regime=regime, p=args.p, n_over_p=ratio, seed=seed, lam=lam,
            n_blocks=inf64c["n_blocks"], n_groups=inf64c["n_groups"],
            s_max=int(c["s_max"]),
            t_numpy64=t64c, t_torch64=t64g, t_torch32=t32g,
            gpu_speedup_fp64=t64c / t64g, gpu_speedup_fp32=t64c / t32g,
            fp32_over_fp64_gpu_speedup=t64g / t32g,
            t_numpy64_json=json.dumps([round(x, 6) for x in tr64c]),
            t_torch64_json=json.dumps([round(x, 6) for x in tr64g]),
            t_torch32_json=json.dumps([round(x, 6) for x in tr32g]),
            torch64_vs_numpy64_maxdiff=float(np.abs(T64g - T64).max()),
            max_iters_fp64=inf64g["max_iters"], max_iters_fp32=inf32g["max_iters"],
            not_converged_fp64=inf64g["n_not_converged"],
            not_converged_fp32=inf32g["n_not_converged"],
            not_converged_highacc=infx["n_not_converged"],
            theta_abs_err=float(np.abs(T32 - T64).max()),
            theta_rel_err=float(np.abs(T32 - T64).max() / scale),
            err_fp64_vs_highacc=float(err64_x), err_fp32_vs_highacc=float(err32_x),
            err_ratio=float(err32_x / err64_x) if err64_x > 0 else float("inf"),
            obj_fp64=obj64, obj_fp32=obj32,
            obj_rel_diff=abs(obj32 - obj64) / abs(obj64),
            support_disagreement=float(flips.sum() / off.sum()),
            support_flips=int(flips.sum()), nonmarginal_flips=int(nonmarginal.sum()),
            f1_fp64=f1_64, f1_fp32=f1_32, f1_abs_diff=abs(f1_32 - f1_64),
        )
        checks = {
            "all_converged": r["not_converged_fp32"] == 0,
            "max_obj_rel_diff": r["obj_rel_diff"] <= CRITERIA["max_obj_rel_diff"],
            "max_theta_rel_err": r["theta_rel_err"] <= CRITERIA["max_theta_rel_err"],
            "max_err_ratio_vs_fp64": r["err_ratio"] <= CRITERIA["max_err_ratio_vs_fp64"],
            "max_support_disagreement": r["support_disagreement"] <= CRITERIA["max_support_disagreement"],
            "max_nonmarginal_flips": r["nonmarginal_flips"] <= CRITERIA["max_nonmarginal_flips"],
            "max_f1_abs_diff": r["f1_abs_diff"] <= CRITERIA["max_f1_abs_diff"],
        }
        r.update({f"pass_{k}": v for k, v in checks.items()})
        r["pass_all"] = all(checks.values())
        rows.append(r)
        print(f"s_max={r['s_max']:5d} lam={lam:.3f}  numpy64 {t64c:.3f}s  torch64 {t64g:.3f}s  "
              f"torch32 {t32g:.3f}s  | GPU speedup fp64 {r['gpu_speedup_fp64']:.2f}x "
              f"fp32 {r['gpu_speedup_fp32']:.2f}x  | {'PASS' if r['pass_all'] else 'FAIL: ' + ', '.join(k for k, v in checks.items() if not v)}")

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
