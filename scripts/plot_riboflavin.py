"""Riboflavin figure: F_max vs lambda, real data against the permutation null,
plus the batch sensitivity curves. Reads results/riboflavin*.
"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from make_graph import COL  # noqa: E402

RES = os.path.join(ROOT, "results")
SENS = os.path.join(RES, "riboflavin_sensitivity")
RUNS = [("all samples (n=71)", os.path.join(RES, "riboflavin_curve.csv"), COL["blue"], "-"),
        ("24 h only (n=28)", os.path.join(SENS, "pt24_curve.csv"), COL["orange"], "--"),
        ("batch means removed", os.path.join(SENS, "within_batch_curve.csv"), COL["green"], "-.")]


def load(path, p):
    with open(path) as f:
        return [r for r in csv.DictReader(f) if int(r["p"]) == p]


def col(rows, k):
    return [float(r[k]) for r in rows]


out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "figs", "fig6_riboflavin.pdf")
ps = (200, 500, 1000)
fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.3), sharey=True)
for ax, p in zip(axes, ps):
    base = load(RUNS[0][1], p)
    ax.fill_between(col(base, "lambda"), col(base, "null_Fmax_lo"), col(base, "null_Fmax_hi"),
                    color=COL["grey"], alpha=0.35, lw=0, label="permuted, 95% band (n=71)")
    for label, path, color, ls in RUNS:
        rows = load(path, p)
        ax.plot(col(rows, "lambda"), col(rows, "F_max"), ls, color=color, lw=1.6, label=label)
    ax.set_title(f"p = {p}", fontsize=9)
    ax.set_xlabel("λ (correlation threshold)", fontsize=8)
    ax.set_xlim(0.2, 1.0)
    ax.tick_params(labelsize=7.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[0].set_ylabel("$F_{max}$ (largest-component\nwork share)", fontsize=8)
handles, labels = axes[0].get_legend_handles_labels()
fig.tight_layout(rect=(0, 0.1, 1, 1))
fig.legend(handles, labels, fontsize=7, frameon=False, ncol=4, loc="lower center")
fig.savefig(out, bbox_inches="tight")
fig.savefig(os.path.splitext(out)[0] + ".png", dpi=170, bbox_inches="tight")
print("wrote", out)
