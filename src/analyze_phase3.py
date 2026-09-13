"""Create the final Phase-3 summary table and speedup figure from a CSV."""

import argparse
import csv
import os
from pathlib import Path

# Keep Matplotlib's cache inside the writable project directory.
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib"))
import matplotlib.pyplot as plt


SOLVERS = [
    ("block", "GGLasso block_SGL"),
    ("batch_numpy", "Batched NumPy CPU"),
    ("batch_torch_cpu", "Batched PyTorch CPU"),
    ("batch_cuda", "Batched CUDA"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    with open(args.csv, newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit("input CSV has no data rows")
    for row in rows:
        for key in ("p", "n_over_p", "lambda_", "support_f1"):
            row[key] = float(row[key])

    # A performance operating point is selected only by estimate quality.
    best = []
    keys = sorted({(r["regime"], int(r["p"]), int(r["n_over_p"])) for r in rows})
    for key in keys:
        candidates = [r for r in rows
                      if (r["regime"], int(r["p"]), int(r["n_over_p"])) == key]
        best.append(max(candidates, key=lambda r: r["support_f1"]))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "phase3_best_f1.csv"
    fields = ["regime", "p", "n_over_p", "lambda_", "support_f1",
              "detected_n_blocks", "detected_max_block"]
    available = [(key, label) for key, label in SOLVERS
                 if f"{key}_speedup_vs_admm" in best[0]]
    fields += [f"{key}_speedup_vs_admm" for key, _ in available]
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(best)

    labels = [f"{r['regime']}\np={int(r['p'])}, n/p={int(r['n_over_p'])}"
              for r in best]
    x = list(range(len(best)))
    width = 0.8 / max(len(available), 1)
    fig_width = max(9, 1.25 * len(best))
    fig, ax = plt.subplots(figsize=(fig_width, 5.5))
    for j, (key, label) in enumerate(available):
        values = [float(r[f"{key}_speedup_vs_admm"]) for r in best]
        offset = (j - (len(available) - 1) / 2) * width
        ax.bar([v + offset for v in x], values, width=width, label=label)
    ax.axhline(1.0, color="black", linewidth=0.8)
    ax.set_ylabel("Speedup versus full ADMM (×)")
    ax.set_title("Phase 3 performance at each configuration's best-F1 lambda")
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    figure_path = out_dir / "phase3_speedup.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    print(f"Wrote {summary_path}")
    print(f"Wrote {figure_path}")
    print("\nBest-F1 operating points:")
    for row in best:
        metrics = "  ".join(
            f"{label}={float(row[f'{key}_speedup_vs_admm']):.2f}x"
            for key, label in available)
        print(f"  {row['regime']:12s} p={int(row['p']):4d} "
              f"n/p={int(row['n_over_p']):2d} lambda={row['lambda_']:.2f} "
              f"F1={row['support_f1']:.3f}  {metrics}")


if __name__ == "__main__":
    main()
