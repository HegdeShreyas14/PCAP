"""gpu_crossover.py — trace GPU-batched vs CPU-batched speedup as a function of
the largest detected block size, not p.
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
# Explicit, not setdefault: an inherited OMP_NUM_THREADS used to win silently.
_PRE = argparse.ArgumentParser(add_help=False)
_PRE.add_argument("--blas-threads", type=int, default=8)
_pre, _ = _PRE.parse_known_args()
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = str(_pre.blas_threads)

import numpy as np

from blockgen import make_instance
from provenance import write_env
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
    ap = argparse.ArgumentParser(parents=[_PRE])
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
    ap.add_argument("--dtype", default="float64", choices=["float64", "float32"],
                    help="torch solver precision. float64 is the reference; "
                         "float32 was validated in fp32_diagnostic.py. The numpy "
                         "CPU baseline is always float64.")
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
    print(f"environment -> {write_env(args.out, args=args, extra={'torch': torch.__version__, 'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0), 'vram_total_mib': torch.cuda.get_device_properties(0).total_memory // 2**20})}")

    print(f"{'regime':11s} {'p':>5s} {'lam':>5s} {'blks':>5s} {'grps':>5s} "
          f"{'s_max':>6s} {'numpy':>8s} {'torch':>8s} {'gpu_x':>7s} {'maxdiff':>9s}")
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
                                              device=args.device, dtype=args.dtype,
                                              return_info=True),
                    trials=args.trials)
                peak_mem_mb = torch.cuda.max_memory_allocated() / 1024**2

                maxdiff = float(np.max(np.abs(sol_np["Theta"] - sol_gpu["Theta"])))
                max_rel_diff = maxdiff / float(np.max(np.abs(sol_np["Theta"])))
                n_groups = info_np["n_groups"]

                # F_max and cubic_gain, same definitions as analyze.py.
                cubic_tot = sum(float(x) ** 3 for x in det_sizes) if det_sizes else float("nan")
                f_max = (float(s_max) ** 3 / cubic_tot) if cubic_tot else float("nan")
                cubic_gain = (float(p) ** 3 / cubic_tot) if cubic_tot else float("nan")

                t_ref = float("nan")
                if i % args.ref_every == 0:
                    with _silence():
                        t_ref, _ = timed_call(
                            lambda: block_SGL(S, lam, Omega_0, tol=1e-7,
                                              rtol=1e-5, max_iter=1000,
                                              verbose=False),
                            warmup=1, trials=3)

                ratio_gpu_np = t_np / t_gpu if t_gpu > 0 else float("nan")
                # float64 must match the numpy path to round-off; float32 is held
                # to the bound fixed in advance in fp32_diagnostic.py
                ok = maxdiff < 1e-8 if args.dtype == "float64" else max_rel_diff <= 1e-3
                ok = ok and info_gpu["n_not_converged"] == 0
                flag = "" if ok else "  !! CORRECTNESS/CONVERGENCE"
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
                                 max_group_k=info_gpu["max_group_k"],
                                 mem_group_k=info_gpu["mem_group_k"],
                                 mem_group_s=info_gpu["mem_group_s"],
                                 mem_group_ks2=info_gpu["mem_group_ks2"],
                                 gpu_speedup=ratio_gpu_np,          # t_numpy / t_torch: >1 means GPU faster
                                 peak_gpu_mem_mb=peak_mem_mb,
                                 cpu_baseline_label=info_np["cpu_baseline_label"],
                                 max_diff=maxdiff, t_ref=t_ref,
                                 dtype=args.dtype, max_rel_diff=max_rel_diff,
                                 not_converged_torch=info_gpu["n_not_converged"],
                                 max_iters_torch=info_gpu["max_iters"],
                                 max_iters_numpy=info_np["max_iters"]))

    if rows:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {len(rows)} rows to {args.out}")

    # crude crossover readout: smallest s_max where torch beats numpy
    print("\nCrossover (first s_max where GPU speedup t_numpy/t_torch exceeds 1.0):")
    for regime in args.regime:
        for p in args.p:
            sub = sorted(
                [r for r in rows if r["regime"] == regime and r["p"] == p],
                key=lambda r: r["s_max"])
            won = [r for r in sub if r["gpu_speedup"] > 1.0]
            if won:
                print(f"  {regime} p={p}: s_max={won[0]['s_max']} "
                      f"(n_groups={won[0]['n_groups']}, "
                      f"lambda={won[0]['lam']:.2f})")
            else:
                best = max(sub, key=lambda r: r["gpu_speedup"],
                          default=None)
                if best:
                    print(f"  {regime} p={p}: no crossover in range. "
                          f"Best: {best['gpu_speedup']:.2f}x "
                          f"at s_max={best['s_max']}")


if __name__ == "__main__":
    main()