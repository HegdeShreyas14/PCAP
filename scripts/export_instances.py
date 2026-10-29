"""Export the synthetic instances used in the experiments to .npz files.

Each file holds X (raw samples, n x p), S (sample covariance, the solver
input), Sigma, Theta (true precision matrix) and block (true block label per
variable). index.csv lists every file with S fingerprints that match the
S_trace / S_fro / S_sum columns in results/.

    python scripts/export_instances.py                      # RQ1 grid, seed 0
    python scripts/export_instances.py --seeds 0 1 2 --ps 100 200
"""

import argparse
import csv
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from blockgen import REGIMES, generate_block_precision, make_instance, regime_block_sizes  # noqa: E402


def samples(Sigma, n, seed):
    """The exact draw inside blockgen.sample_covariance."""
    rng = np.random.default_rng(seed + 90000)
    return rng.standard_normal((n, Sigma.shape[0])) @ np.linalg.cholesky(Sigma).T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regimes", nargs="+", default=REGIMES)
    ap.add_argument("--ps", nargs="+", type=int, default=[100, 200, 400, 800])
    ap.add_argument("--ratios", nargs="+", type=int, default=[2, 5, 10])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "synthetic"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows = []
    for seed in args.seeds:
        for regime in args.regimes:
            for p in args.ps:
                sizes = regime_block_sizes(regime, p)
                Sigma, Theta, blocks, _ = generate_block_precision(sizes, seed=seed)
                label = np.empty(p, dtype=np.int32)
                for i, b in enumerate(blocks):
                    label[b] = i
                for ratio in args.ratios:
                    n = ratio * p
                    X = samples(Sigma, n, seed)
                    S = np.cov(X, rowvar=False, bias=True)
                    S_ref = make_instance(regime, p, n, seed=seed)[0]
                    assert np.array_equal(S, S_ref), "export differs from the experiments' S"
                    name = f"{regime}_p{p}_n{n}_seed{seed}.npz"
                    np.savez_compressed(os.path.join(args.out, name), X=X, S=S, Sigma=Sigma,
                                        Theta=Theta, block=label)
                    rows.append(dict(file=name, regime=regime, p=p, n=n, n_over_p=ratio, seed=seed,
                                     true_blocks=len(sizes), true_block_sizes=" ".join(map(str, sizes)),
                                     S_trace=float(np.trace(S)), S_fro=float(np.linalg.norm(S)),
                                     S_sum=float(S.sum())))
                    print(f"wrote {name}", flush=True)

    with open(os.path.join(args.out, "index.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} instances -> {args.out}")


if __name__ == "__main__":
    main()
