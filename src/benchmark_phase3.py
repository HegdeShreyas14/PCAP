"""Benchmark the four Phase-3 solver paths on identical generated instances."""

import argparse
import contextlib
import csv
import io
import json
import os
from pathlib import Path
import platform
import statistics
import time

# Thread variables must be set before importing NumPy.
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--blas-threads", type=int, default=8)
_known, _ = _pre.parse_known_args()
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = str(_known.blas_threads)

import numpy as np
from gglasso.solver.single_admm_solver import ADMM_SGL, block_SGL, get_connected_components

from batched_sgl import batched_block_sgl
from blockgen import REGIMES, make_instance


@contextlib.contextmanager
def _silence():
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def _theta(value):
    if isinstance(value, tuple):
        value = value[0]
    return value["Theta"]


def _timed(fn, warmup, trials, cuda=False):
    if cuda:
        import torch
    with _silence():
        for _ in range(warmup):
            fn()
        values, last = [], None
        for _ in range(trials):
            if cuda:
                torch.cuda.synchronize()
            started = time.perf_counter()
            last = fn()
            if cuda:
                torch.cuda.synchronize()
            values.append(time.perf_counter() - started)
    return values, last


def _objective(theta, S, lam):
    sign, logdet = np.linalg.slogdet(theta)
    if sign <= 0:
        return float("inf")
    penalty = np.abs(theta).sum() - np.abs(np.diag(theta)).sum()
    return float(-logdet + np.trace(S @ theta) + lam * penalty)


def _f1(theta, truth, threshold=1e-6):
    off = ~np.eye(theta.shape[0], dtype=bool)
    est, actual = (np.abs(theta) > threshold) & off, (np.abs(truth) > threshold) & off
    tp, fp, fn = (est & actual).sum(), (est & ~actual).sum(), (~est & actual).sum()
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    return (2 * precision * recall / (precision + recall)
            if precision == precision and recall == recall and precision + recall else float("nan"))


def _stats(times, prefix):
    return {
        f"{prefix}_median": statistics.median(times),
        f"{prefix}_min": min(times),
        f"{prefix}_max": max(times),
        f"{prefix}_trials_json": json.dumps([round(v, 7) for v in times]),
    }


def main():
    ap = argparse.ArgumentParser(parents=[_pre])
    ap.add_argument("--out", default="results/phase3_results.csv")
    ap.add_argument("--machine", default=platform.node())
    ap.add_argument("--regimes", nargs="+", default=REGIMES)
    ap.add_argument("--ps", nargs="+", type=int, default=[100, 200, 400])
    ap.add_argument("--ratios", nargs="+", type=int, default=[2, 10])
    ap.add_argument("--lambdas", nargs="+", type=float,
                    default=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trials", type=int, default=9)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--tol", type=float, default=1e-7)
    ap.add_argument("--rtol", type=float, default=1e-5)
    ap.add_argument("--max-iter", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--padding-ratio", type=float, default=1.5)
    ap.add_argument("--gpu-dtype", choices=["float32", "float64"], default="float64")
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="resume completed configurations from OUT.partial.jsonl")
    args = ap.parse_args()

    import torch
    cuda = torch.cuda.is_available() and not args.no_gpu
    if args.quick:
        # At p=100 both regimes are [20] * 5, so p=200 is the smallest useful
        # smoke comparison between balanced [40] * 5 and many_tiny [20] * 10.
        regimes, ps, ratios, lambdas = ["balanced", "many_tiny"], [200], [10], [0.15, 0.25]
        trials, warmup = 3, 1
    else:
        regimes, ps, ratios, lambdas = args.regimes, args.ps, args.ratios, args.lambdas
        trials, warmup = args.trials, args.warmup

    common = dict(tol=args.tol, rtol=args.rtol, max_iter=args.max_iter)
    batch_common = dict(**common, batch_size=args.batch_size,
                        padding_ratio=args.padding_ratio)
    configs = [(r, p, ratio, lam) for r in regimes for p in ps
               for ratio in ratios for lam in lambdas]
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(f"{args.out}.partial.jsonl")
    rows = []
    completed = set()
    if args.resume and checkpoint.exists():
        with checkpoint.open() as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    rows.append(row)
                    completed.add((row["regime"], int(row["p"]),
                                   int(row["n_over_p"]), float(row["lambda_"])))
        print(f"Resuming from {len(rows)} checkpointed configurations")
    print(f"CUDA: {cuda} ({torch.cuda.get_device_name(0) if cuda else 'not used'})")
    print(f"Running {len(configs)} paired configurations; {trials} trials each")

    for index, (regime, p, ratio, lam) in enumerate(configs, 1):
        if (regime, p, ratio, lam) in completed:
            print(f"[{index:3d}/{len(configs)}] {regime:11s} p={p:4d} n/p={ratio:2d} "
                  f"lam={lam:.2f} ... checkpointed")
            continue
        S, truth, _, _, styles = make_instance(regime, p, ratio * p, seed=args.seed)
        S = S[0] if S.ndim == 3 else S
        initial = np.eye(p)
        n_components, components = get_connected_components(S, lam)
        sizes = sorted((len(c) for c in components), reverse=True)
        solvers = {
            "admm": (lambda: ADMM_SGL(S, lam, initial, verbose=False,
                                       measure=False, **common), False),
            "block": (lambda: block_SGL(S, lam, initial, verbose=False,
                                         measure=False, **common), False),
            "batch_numpy": (lambda: batched_block_sgl(
                S, lam, initial, backend="numpy", **batch_common), False),
            "batch_torch_cpu": (lambda: batched_block_sgl(
                S, lam, initial, backend="torch-cpu", **batch_common), False),
        }
        if cuda:
            solvers["batch_cuda"] = (lambda: batched_block_sgl(
                S, lam, initial, backend="cuda", dtype=args.gpu_dtype,
                **batch_common), True)

        row = dict(machine=args.machine, cpu=platform.processor(),
                   gpu=torch.cuda.get_device_name(0) if cuda else "none",
                   torch=torch.__version__, numpy=np.__version__,
                   blas_threads=args.blas_threads, regime=regime, p=p,
                   n_over_p=ratio, n_samples=ratio * p, lambda_=lam,
                   seed=args.seed, detected_n_blocks=n_components,
                   detected_max_block=max(sizes),
                   detected_sizes_json=json.dumps(sizes),
                   gen_styles_json=json.dumps(styles), gpu_dtype=args.gpu_dtype,
                   trials=trials, warmup=warmup)
        outputs = {}
        for name, (fn, uses_cuda) in solvers.items():
            values, output = _timed(fn, warmup, trials, cuda=uses_cuda)
            outputs[name] = _theta(output)
            row.update(_stats(values, f"t_{name}"))

        reference = outputs["block"]
        ref_obj = _objective(reference, S, lam)
        row["support_f1"] = _f1(reference, truth)
        for name, theta in outputs.items():
            row[f"{name}_speedup_vs_admm"] = row["t_admm_median"] / row[f"t_{name}_median"]
            row[f"{name}_max_abs_vs_block"] = float(np.max(np.abs(theta - reference)))
            obj = _objective(theta, S, lam)
            row[f"{name}_obj_rel_gap_vs_block"] = abs(obj - ref_obj) / max(abs(ref_obj), 1e-12)
        rows.append(row)
        with checkpoint.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
            handle.flush()
        gpu_text = (f" cuda={row['batch_cuda_speedup_vs_admm']:.2f}x"
                    if cuda else "")
        print(f"[{index:3d}/{len(configs)}] {regime:11s} p={p:4d} n/p={ratio:2d} "
              f"lam={lam:.2f} blocks={n_components:3d} F1={row['support_f1']:.3f} "
              f"block={row['block_speedup_vs_admm']:.2f}x "
              f"batchCPU={row['batch_numpy_speedup_vs_admm']:.2f}x{gpu_text}")

    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    checkpoint.unlink(missing_ok=True)
    print(f"Wrote {len(rows)} rows to {path}")


if __name__ == "__main__":
    main()
