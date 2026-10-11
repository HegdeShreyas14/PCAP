#!/usr/bin/env python3
"""
make_figures.py -- generate all paper figures directly from results/*.csv.

Nothing here is hardcoded. Every number plotted is read from the CSVs the
harness produced, so re-running a sweep and re-running this script keeps the
paper's figures in sync with the data automatically. If a required CSV is
missing, that figure is skipped with a warning rather than silently plotting
stale or invented values.

Usage
-----
    python make_figures.py                          # reads ./results, writes ./figs
    python make_figures.py --results DIR --out DIR
    python make_figures.py --format png             # for slides instead of the paper

Figures produced
----------------
    fig1_rq1_threading           RQ1: 1t vs 8t speedup and retention by p
    fig2_rq1_fragmentation_trap  RQ1: speedup vs F1 across detected-block buckets
    fig3_rq1_oracle_gap          RQ1: detected vs oracle F_max (percolation.csv)
    fig4_rq1_percolation         RQ1: F_max vs lambda, showing the transition
    fig5_rq2_crossover           RQ2: torch/numpy ratio vs s_max, per p series
    fig6_rq2_timing_breakdown    RQ2: h2d / compute / d2h split vs s_max
    fig7_rq3_vram_model          RQ3: fitted VRAM law + held-out anchors
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------
# style
# --------------------------------------------------------------------------

COL = {
    "blue":   "#4C72B0",
    "orange": "#DD8452",
    "green":  "#55A868",
    "red":    "#C44E52",
    "purple": "#8172B3",
    "grey":   "#8C8C8C",
}

def set_style():
    plt.rcParams.update({
        "font.size": 9,
        "font.family": "serif",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.axisbelow": True,
        "figure.dpi": 200,
        "savefig.bbox": "tight",
    })


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def load_csv(path):
    """Read a CSV into a list of dicts. Returns None (with a warning) if absent."""
    if not os.path.exists(path):
        print(f"  [skip] {os.path.basename(path)} not found", file=sys.stderr)
        return None
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print(f"  [skip] {os.path.basename(path)} is empty", file=sys.stderr)
        return None
    return rows


def fnum(row, key, default=float("nan")):
    """Float-convert a CSV field, tolerating blanks and missing columns."""
    v = row.get(key, "")
    if v is None or v == "":
        return default
    try:
        return float(v)
    except ValueError:
        return default


def inum(row, key, default=0):
    try:
        return int(float(row.get(key, default)))
    except (ValueError, TypeError):
        return default


def gpu_speedup(row):
    """t_numpy / t_torch (>1 = GPU faster). Older CSVs call this column
    ratio_torch_over_numpy despite holding the same quantity; read either."""
    v = fnum(row, "gpu_speedup")
    return v if v == v else fnum(row, "ratio_torch_over_numpy")


def median(xs):
    xs = [x for x in xs if x == x]          # drop NaN
    return float(np.median(xs)) if xs else float("nan")


def best_f1_rows(rows, by_seed=True):
    """
    One row per group: the lambda maximising support_f1.

    by_seed=True keys on (regime, p, n/p, seed) so each seed contributes its
    own best-lambda operating point. This matters: percolation.py runs
    multiple seeds precisely so structural quantities can be summarised
    across them, and grouping without `seed` would instead cherry-pick a
    single global-best (seed, lambda) row per cell and silently discard the
    rest.

    by_seed=False reproduces the coarser grouping, for callers that genuinely
    want one row per (regime, p, n/p).
    """
    groups = defaultdict(list)
    for r in rows:
        p = inum(r, "p")
        n = inum(r, "n_samples", 0) or inum(r, "n_over_p", 0) * p
        ratio = (n // p) if p else 0
        key = (r.get("regime", ""), p, ratio)
        if by_seed:
            key = key + (r.get("seed", ""),)
        groups[key].append(r)
    out = []
    for key, sub in groups.items():
        scored = [r for r in sub if fnum(r, "support_f1") == fnum(r, "support_f1")]
        if scored:
            out.append(max(scored, key=lambda r: fnum(r, "support_f1")))
    return out


# --------------------------------------------------------------------------
# Fig 1: RQ1 regime speedup, 1t vs 8t
# --------------------------------------------------------------------------

def fig_rq1_threading(res, out, ext):
    """
    Fig 1: what an 8-thread BLAS baseline does to the block-decomposition win.

    (a) median speedup at the best-F1 lambda, 1t vs 8t, by p.
    (b) retention (8t speedup / 1t speedup) per configuration, by p. Hollow
        markers are `ns`: the two runs' trial bands overlap, so the difference
        is not resolvable.

    Grouped by p, not regime: with properly paired runs (bug 8) the threading
    effect is set by p -- ADMM's single p x p eigendecomposition only becomes
    large enough to use 8 cores from p ~ 400 -- and is flat across regimes.
    The operating point is chosen from the 1t run and looked up at the same
    lambda in the 8t run, as in analyze.py --compare.
    """
    r1 = load_csv(os.path.join(res, "rq1_primary_1t.csv"))
    r8 = load_csv(os.path.join(res, "rq1_primary_8t.csv"))
    if r1 is None or r8 is None:
        return

    key = lambda r: (r["regime"], inum(r, "p"), inum(r, "n_samples"),
                     round(fnum(r, "lambda"), 4))
    by8 = {key(r): r for r in r8}
    pts = []                                    # (p, s1, s8, ns)
    for b in best_f1_rows(r1, by_seed=False):
        o = by8.get(key(b))
        if o is None:
            continue
        ns = not (fnum(o, "speedup_hi") < fnum(b, "speedup_lo")
                  or fnum(o, "speedup_lo") > fnum(b, "speedup_hi"))
        pts.append((inum(b, "p"), fnum(b, "algorithmic_speedup"),
                    fnum(o, "algorithmic_speedup"), ns))
    ps = sorted({q[0] for q in pts})
    if not ps:
        print("  [skip] fig1: no paired best-F1 rows", file=sys.stderr)
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.6))
    x = np.arange(len(ps))

    m1 = [median([q[1] for q in pts if q[0] == p]) for p in ps]
    m8 = [median([q[2] for q in pts if q[0] == p]) for p in ps]
    ax1.plot(x, m1, "-o", color=COL["blue"], lw=2, ms=5, label="1 BLAS thread")
    ax1.plot(x, m8, "-s", color=COL["orange"], lw=2, ms=5, label="8 BLAS threads")
    ax1.set_xticks(x); ax1.set_xticklabels([str(p) for p in ps])
    ax1.set_xlabel("p")
    ax1.set_ylabel("Median speedup at best-F1 λ\n(block_SGL vs. ADMM_SGL)", fontsize=8)
    ax1.set_ylim(0, max(m1 + m8) * 1.25)
    ax1.legend(fontsize=7.5, frameon=False, loc="upper left")
    ax1.set_title("(a)", fontsize=9, loc="left")

    rng = np.random.default_rng(0)             # deterministic jitter
    ret = defaultdict(list)
    for p, s1, s8, ns in pts:
        r = 100 * s8 / s1
        ret[p].append(r)
        xi = ps.index(p) + rng.uniform(-0.15, 0.15)
        ax2.plot(xi, r, "o", ms=5, mec=COL["blue"], mew=1,
                 mfc="none" if ns else COL["blue"], alpha=0.85)
    med = [median(ret[p]) for p in ps]
    ax2.hlines(med, x - 0.3, x + 0.3, color="black", lw=2, zorder=3)
    for xi, m in zip(x, med):
        ax2.text(xi + 0.33, m, f"{m:.0f}%", va="center", fontsize=7)
    ax2.axhline(100, color=COL["grey"], lw=1, ls="--", zorder=0)
    ax2.set_xticks(x); ax2.set_xticklabels([str(p) for p in ps])
    ax2.set_xlim(-0.5, len(ps) - 0.2)
    ax2.set_xlabel("p")
    ax2.set_ylabel("Retained at 8 threads (%)", fontsize=8)
    ax2.set_title("(b)", fontsize=9, loc="left")
    ax2.plot([], [], "o", mec=COL["blue"], mfc=COL["blue"], label="resolvable")
    ax2.plot([], [], "o", mec=COL["blue"], mfc="none", label="ns (bands overlap)")
    ax2.plot([], [], "-", color="black", lw=2, label="median")
    ax2.legend(fontsize=7, frameon=False, loc="lower left")
    save(fig, out, "fig1_rq1_threading", ext)

    print("    retention at best-F1 by p: " +
          ", ".join(f"p={p} {m:.0f}% (ns {sum(q[3] for q in pts if q[0] == p)}/"
                    f"{sum(1 for q in pts if q[0] == p)})" for p, m in zip(ps, med)))


# --------------------------------------------------------------------------
# Fig 2: over-fragmentation trap
# --------------------------------------------------------------------------

BUCKETS = [(1, 1, "1"), (2, 9, "2-9"), (10, 49, "10-49"),
           (50, 99, "50-99"), (100, 199, "100-199"), (200, None, "200+")]


def fig_rq1_fragmentation(res, out, ext, source="rq1_primary_1t.csv"):
    rows = load_csv(os.path.join(res, source))
    if rows is None:
        return

    labels, speed, f1s, ns = [], [], [], []
    for lo, hi, lab in BUCKETS:
        sub = [r for r in rows
               if lo <= inum(r, "detected_n_blocks") <= (hi if hi else 10**9)]
        if not sub:
            continue
        labels.append(lab)
        speed.append(median([fnum(r, "algorithmic_speedup") for r in sub]))
        f1s.append(median([fnum(r, "support_f1") for r in sub]))
        ns.append(len(sub))

    # Two stacked panels sharing the bucket axis, not twin y-axes.
    x = np.arange(len(labels))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.4, 3.2), sharex=True)
    ax1.plot(x, speed, "-o", ms=4, color=COL["blue"])
    ax1.set_yscale("log")
    ax1.set_ylabel("Median speedup", fontsize=8)
    ax2.plot(x, f1s, "-s", ms=4, color=COL["red"])
    ax2.set_ylabel("Median support F1", fontsize=8)
    ax2.set_ylim(0, max(f1s) * 1.2)
    ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=7.5)
    ax2.set_xlabel("Detected block count", fontsize=8)
    save(fig, out, "fig2_rq1_fragmentation_trap", ext)

    print("    buckets: " + ", ".join(
        f"{l}: {s:.2f}x/F1 {f:.3f} (n={n})"
        for l, s, f, n in zip(labels, speed, f1s, ns)))


# --------------------------------------------------------------------------
# Fig 3: detected vs oracle F_max
# --------------------------------------------------------------------------

def fig_rq1_oracle_gap(res, out, ext):
    rows = load_csv(os.path.join(res, "percolation.csv"))
    if rows is None:
        return

    # Two-stage aggregation. Seeds are replicates OF A CELL, not independent
    # observations, so we take the median across seeds within each
    # (regime, p, n/p) cell first, then the median across cells. Pooling every
    # (cell, seed) result into one median instead would weight regimes by how
    # many cells they happen to have and differs materially here -- up to
    # 1.1pp on imbalanced (99.3% pooled vs 98.3% two-stage).
    best = best_f1_rows(rows, by_seed=True)
    percell_det, percell_orc = defaultdict(list), defaultdict(list)
    for r in best:
        p = inum(r, "p")
        n = inum(r, "n_samples", 0) or inum(r, "n_over_p", 0) * p
        cell = (r.get("regime", ""), p, (n // p) if p else 0)
        percell_det[cell].append(fnum(r, "F_max"))
        percell_orc[cell].append(fnum(r, "oracle_F_max"))

    det, orc, nseeds = defaultdict(list), defaultdict(list), defaultdict(set)
    for cell, vals in percell_det.items():
        det[cell[0]].append(median(vals))
        nseeds[cell[0]].add(len(vals))
    for cell, vals in percell_orc.items():
        orc[cell[0]].append(median(vals))

    regimes = sorted(det)
    dv = [100 * median(det[k]) for k in regimes]
    ov = [100 * median(orc[k]) for k in regimes]
    x = np.arange(len(regimes)); w = 0.35

    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    ax.bar(x - w/2, dv, w, label="Detected (at best-F1 $\\lambda$)", color=COL["red"])
    ax.bar(x + w/2, ov, w, label="Oracle (planted structure)", color=COL["green"])
    ax.set_xticks(x); ax.set_xticklabels(regimes, fontsize=7.5)
    ax.set_ylabel("$F_{\\max}$: share of cubic work\nin largest block (%)", fontsize=8)
    ax.set_ylim(0, 118)
    ax.legend(fontsize=6.8, frameon=False, loc="lower right")
    save(fig, out, "fig3_rq1_oracle_gap", ext)

    for k, d, o in zip(regimes, dv, ov):
        spread = (100 * min(det[k]), 100 * max(det[k]))
        print(f"    {k:11s} detected {d:5.1f}% (cell range {spread[0]:.1f}-"
              f"{spread[1]:.1f}%)   oracle {o:5.1f}%   "
              f"[{len(det[k])} cells x {sorted(nseeds[k])} seeds]")


# --------------------------------------------------------------------------
# Fig 4: percolation transition
# --------------------------------------------------------------------------

def fig_rq1_percolation(res, out, ext, regime="balanced", p=400, ratio=10):
    rows = load_csv(os.path.join(res, "percolation.csv"))
    if rows is None:
        return

    sub = [r for r in rows
           if r.get("regime") == regime
           and inum(r, "p") == p
           and inum(r, "n_over_p") == ratio]
    if not sub:
        print(f"  [skip] fig4: no rows for {regime} p={p} n/p={ratio}", file=sys.stderr)
        return

    # average F_max across seeds at each lambda
    by_lam = defaultdict(list)
    by_lam_f1 = defaultdict(list)
    for r in sub:
        by_lam[round(fnum(r, "lambda"), 4)].append(fnum(r, "F_max"))
        by_lam_f1[round(fnum(r, "lambda"), 4)].append(fnum(r, "support_f1"))
    lams = sorted(by_lam)
    fmax = [100 * median(by_lam[l]) for l in lams]
    f1 = [median(by_lam_f1[l]) for l in lams]

    # Two stacked panels sharing lambda, not twin y-axes.
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.4, 3.2), sharex=True)
    ax1.plot(lams, fmax, "-o", ms=3.5, color=COL["red"])
    ax1.set_ylabel("$F_{\\max}$ (%)", fontsize=8)
    ax1.set_ylim(0, 105)
    ax2.plot(lams, f1, "-s", ms=3.5, color=COL["blue"])
    ax2.set_ylabel("Support F1", fontsize=8)
    ax2.set_xlabel("$\\lambda$ (regularization strength)", fontsize=8)

    # mark the best-F1 lambda (the operating point a practitioner would choose)
    if any(v == v for v in f1):
        bi = int(np.nanargmax(f1))
        for ax in (ax1, ax2):
            ax.axvline(lams[bi], color=COL["grey"], ls="--", lw=0.9)
        ax1.text(lams[bi], 50, " best-F1 $\\lambda$", fontsize=6.5,
                 color=COL["grey"], rotation=90, va="center")
    ax1.set_title(f"{regime}, $p$={p}, $n/p$={ratio} (median of seeds)", fontsize=8)
    save(fig, out, "fig4_rq1_percolation", ext)


# --------------------------------------------------------------------------
# Fig 5: GPU crossover
# --------------------------------------------------------------------------

def fig_rq2_crossover(res, out, ext):
    """
    GPU speedup over the batched float64 CPU solver vs largest block size,
    float64 and (if present) float32 GPU. Colour = precision, marker = p.
    float32 points that did not converge within max_iter are drawn hollow:
    they are not valid results, and hiding them would overstate float32.
    """
    series = {}
    for dt, fn, anchor_fns in (
            ("float64", "gpu_crossover.csv", ("gpu_mem_2400.csv", "gpu_mem_3200.csv")),
            ("float32", "gpu_crossover_fp32.csv", ("gpu_mem_2400_fp32.csv", "gpu_mem_3200_fp32.csv"))):
        rows = load_csv(os.path.join(res, fn)) if os.path.exists(os.path.join(res, fn)) else None
        if rows is None:
            continue
        anchors = []
        for afn in anchor_fns:
            if os.path.exists(os.path.join(res, afn)):
                anchors.extend(load_csv(os.path.join(res, afn)) or [])
        series[dt] = (rows, anchors)
    if "float64" not in series:
        print("  [skip] fig5: gpu_crossover.csv not found", file=sys.stderr)
        return

    colour = {"float64": COL["blue"], "float32": COL["orange"]}
    marker = {1200: "o", 2000: "s"}
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    hollow_drawn = False
    for dt, (rows, anchors) in series.items():
        by_p = defaultdict(list)
        for r in rows:
            by_p[inum(r, "p")].append(r)
        for p in sorted(by_p):
            sub = sorted(by_p[p], key=lambda r: fnum(r, "s_max"))
            xs = [fnum(r, "s_max") for r in sub]
            ys = [gpu_speedup(r) for r in sub]
            ax.plot(xs, ys, "-", lw=1.1, color=colour[dt], alpha=0.8)
            for r, x0, y0 in zip(sub, xs, ys):
                bad = inum(r, "not_converged_torch", 0) > 0
                hollow_drawn |= bad
                ax.plot(x0, y0, marker.get(p, "D"), ms=4,
                        mec=colour[dt], mfc="none" if bad else colour[dt], mew=1)
            ax.plot([], [], "-" + marker.get(p, "D"), color=colour[dt], ms=4,
                    label=f"{dt}, $p={p}$")
        if anchors:
            ax.plot([fnum(r, "s_max") for r in anchors], [gpu_speedup(r) for r in anchors],
                    "^", ms=7, color=colour[dt], linestyle="none")
    ax.plot([], [], "^", ms=6, color=COL["grey"], linestyle="none",
            label="anchors ($p$=2400, 3200)")
    if hollow_drawn:
        ax.plot([], [], "o", ms=4, mec=COL["grey"], mfc="none",
                linestyle="none", label="float32 not converged")
    ax.set_yscale("log")
    ax.axhline(1.0, color=COL["grey"], lw=0.8, ls="--")
    ax.set_xlabel("$s_{\\max}$ (largest detected block)", fontsize=8)
    ax.set_ylabel("GPU speedup over batched CPU (float64)", fontsize=8)
    ax.legend(fontsize=6, frameon=False, loc="lower right")
    save(fig, out, "fig5_rq2_crossover", ext)

    for dt, (rows, _) in series.items():
        by_p = defaultdict(list)
        for r in rows:
            by_p[inum(r, "p")].append(r)
        for p in sorted(by_p):
            sub = sorted(by_p[p], key=lambda r: fnum(r, "s_max"))
            won = [r for r in sub if gpu_speedup(r) > 1.0]
            if won:
                print(f"    {dt} p={p}: crosses 1.0 at s_max={inum(won[0], 's_max')} "
                      f"(lambda={fnum(won[0], 'lam'):.2f})")


# --------------------------------------------------------------------------
# Fig 6: GPU timing breakdown (h2d / compute / d2h)
# --------------------------------------------------------------------------

def fig_rq2_timing_breakdown(res, out, ext):
    rows = load_csv(os.path.join(res, "gpu_crossover.csv"))
    if rows is None:
        return
    if "t_torch_compute" not in rows[0]:
        print("  [skip] fig6: no timing-breakdown columns", file=sys.stderr)
        return

    sub = sorted(rows, key=lambda r: fnum(r, "s_max"))
    s = np.array([fnum(r, "s_max") for r in sub])
    h2d = np.array([fnum(r, "t_torch_h2d") for r in sub]) * 1000
    comp = np.array([fnum(r, "t_torch_compute") for r in sub]) * 1000
    d2h = np.array([fnum(r, "t_torch_d2h") for r in sub]) * 1000
    total = h2d + comp + d2h
    transfer_pct = 100 * (h2d + d2h) / np.where(total > 0, total, np.nan)

    fig, ax1 = plt.subplots(figsize=(3.4, 2.5))
    ax1.plot(s, comp, "o", ms=3.5, color=COL["blue"], label="GPU compute")
    ax1.plot(s, h2d + d2h, "s", ms=3.5, color=COL["orange"], label="H2D + D2H transfer")
    ax1.set_yscale("log")
    ax1.set_xlabel("$s_{\\max}$ (largest detected block)", fontsize=8)
    ax1.set_ylabel("Time (ms, log scale)", fontsize=8)
    ax1.legend(fontsize=6.8, frameon=False, loc="upper left")
    save(fig, out, "fig6_rq2_timing_breakdown", ext)

    finite = transfer_pct[np.isfinite(transfer_pct)]
    if finite.size:
        print(f"    transfer share of GPU wall time: "
              f"median {np.median(finite):.2f}%, max {finite.max():.2f}%")


# --------------------------------------------------------------------------
# Fig 7: RQ3 VRAM model, fitted + held-out anchors
# --------------------------------------------------------------------------

def fig_rq3_vram(res, out, ext, fit_below=2000, device_mib=None):
    rows = load_csv(os.path.join(res, "gpu_crossover.csv"))
    if rows is None:
        return
    if device_mib is None:
        # the device the data was measured on, as recorded by the run itself
        env = os.path.join(res, "gpu_crossover_env.json")
        try:
            device_mib = float(json.load(open(env))["vram_total_mib"])
        except (OSError, KeyError, ValueError):
            print(f"  [skip] fig7: no vram_total_mib in {env}; pass --device-mib",
                  file=sys.stderr)
            return

    anchors = []
    for fn in ("gpu_mem_2400.csv", "gpu_mem_3200.csv"):
        a = load_csv(os.path.join(res, fn))
        if a:
            anchors.extend(a)

    # fit on crossover rows only, excluding tiny s_max where the context
    # floor dominates and the quadratic term is unidentifiable
    fit = [r for r in rows
           if 100 < fnum(r, "s_max") <= fit_below
           and fnum(r, "peak_gpu_mem_mb") == fnum(r, "peak_gpu_mem_mb")]
    if len(fit) < 3:
        print("  [skip] fig7: too few rows to fit", file=sys.stderr)
        return

    s = np.array([fnum(r, "s_max") for r in fit])
    v = np.array([fnum(r, "peak_gpu_mem_mb") for r in fit])
    A = np.column_stack([np.ones_like(s), s**2])
    beta, *_ = np.linalg.lstsq(A, v, rcond=None)
    pred = A @ beta
    ss_res = float(((v - pred) ** 2).sum())
    ss_tot = float(((v - v.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    wall = np.sqrt(max(device_mib - beta[0], 0) / beta[1]) if beta[1] > 0 else float("nan")

    data_hi = max([fnum(r, "s_max") for r in anchors], default=s.max())
    # Main panel spans all the way to the wall. Showing only the data range
    # with a device-limit line at the top would visually imply the curve is
    # approaching the limit, when in fact the crossing is >2x beyond any
    # measured point -- the opposite of this section's finding.
    grid_hi = wall * 1.08 if wall == wall else data_hi * 1.2
    grid = np.linspace(0, grid_hi, 400)
    model = beta[0] + beta[1] * grid**2

    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    ax.plot(grid, model, "-", lw=1.2, color=COL["blue"], label="Fitted model")
    ax.plot(s, v, "o", ms=3, color=COL["blue"], label="Fitted points")
    if anchors:
        sa = [fnum(r, "s_max") for r in anchors]
        va = [fnum(r, "peak_gpu_mem_mb") for r in anchors]
        ax.plot(sa, va, "^", ms=7, color=COL["red"], linestyle="none",
                label="Held-out anchors")
    ax.axhline(device_mib, color=COL["grey"], lw=0.9, ls=":")
    ax.set_xlim(0, grid_hi)
    ax.set_ylim(0, device_mib * 1.08)
    ax.set_xlabel("$s_{\\max}$ (largest detected block)", fontsize=8)
    ax.set_ylabel("Peak GPU memory (MB)", fontsize=8)
    ax.legend(fontsize=6.2, frameon=False, loc="upper left")
    # Label the limit line on the RIGHT, below it: the legend occupies the
    # upper-left and the line sits near the top of the axes.
    ax.text(grid_hi * 0.98, device_mib * 0.955,
            f"{device_mib:.0f} MiB device limit", fontsize=6.2,
            color=COL["grey"], ha="right", va="top")
    if wall == wall:
        ax.annotate(f"wall at $s_{{\\max}}\\approx${wall:.0f}",
                    xy=(wall, device_mib), xytext=(wall * 0.52, device_mib * 0.66),
                    fontsize=6.2, color=COL["grey"],
                    arrowprops=dict(arrowstyle="->", color=COL["grey"], lw=0.7))

    # Inset: the measured regime at readable scale, where the held-out
    # confirmation actually lives.
    axin = ax.inset_axes([0.46, 0.13, 0.50, 0.34])
    gi = np.linspace(0, data_hi * 1.10, 200)
    axin.plot(gi, beta[0] + beta[1] * gi**2, "-", lw=1.0, color=COL["blue"])
    axin.plot(s, v, "o", ms=2.5, color=COL["blue"])
    if anchors:
        axin.plot(sa, va, "^", ms=5.5, color=COL["red"], linestyle="none")
    axin.set_xlim(0, data_hi * 1.10)
    axin.set_ylim(0, max(list(v) + (va if anchors else [0])) * 1.18)
    axin.tick_params(labelsize=5.5, length=2)
    axin.set_title("measured range", fontsize=5.8, pad=2)
    axin.grid(alpha=0.25)

    save(fig, out, "fig7_rq3_vram_model", ext)

    if anchors:
        for x0, y0 in zip(sa, va):
            p0 = beta[0] + beta[1] * x0**2
            print(f"    anchor s_max={x0:.0f}: predicted {p0:.1f} MB, "
                  f"measured {y0:.1f} MB, error {100*(y0-p0)/p0:+.1f}%")
    print(f"    fit R2={r2:.4f} on n={len(fit)} rows (s_max<={fit_below}); "
          f"device limit reached at s_max~{wall:.0f} "
          f"({wall/data_hi:.1f}x beyond the largest measured point)")


# --------------------------------------------------------------------------

def save(fig, out, name, ext):
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f"{name}.{ext}")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results", help="directory holding the CSVs")
    ap.add_argument("--out", default="figs", help="directory to write figures to")
    ap.add_argument("--format", default="pdf", choices=["pdf", "png", "svg"],
                    help="pdf for LaTeX, png for slides")
    ap.add_argument("--perc-regime", default="balanced")
    ap.add_argument("--perc-p", type=int, default=400)
    ap.add_argument("--perc-ratio", type=int, default=10)
    ap.add_argument("--device-mib", type=float, default=None,
                    help="GPU memory (MiB) for the RQ3 limit line; default: vram_total_mib from gpu_crossover_env.json")
    ap.add_argument("--figs", nargs="+", type=int, default=list(range(1, 8)),
                    help="which figures to write (the paper uses 1-5)")
    args = ap.parse_args()

    set_style()
    res, out, ext = args.results, args.out, args.format
    print(f"reading {os.path.abspath(res)} -> writing {os.path.abspath(out)}\n")

    for fn, label in [
        (lambda: fig_rq1_threading(res, out, ext), "fig1 RQ1 threading retention by p"),
        (lambda: fig_rq1_fragmentation(res, out, ext), "fig2 RQ1 fragmentation trap"),
        (lambda: fig_rq1_oracle_gap(res, out, ext), "fig3 RQ1 oracle gap"),
        (lambda: fig_rq1_percolation(res, out, ext, args.perc_regime,
                                     args.perc_p, args.perc_ratio),
         "fig4 RQ1 percolation"),
        (lambda: fig_rq2_crossover(res, out, ext), "fig5 RQ2 crossover"),
        (lambda: fig_rq2_timing_breakdown(res, out, ext), "fig6 RQ2 timing breakdown"),
        (lambda: fig_rq3_vram(res, out, ext, device_mib=args.device_mib),
         "fig7 RQ3 VRAM model"),
    ]:
        if int(label.split()[0][3:]) not in args.figs:
            continue
        print(label)
        try:
            fn()
        except Exception as e:
            print(f"  [error] {type(e).__name__}: {e}", file=sys.stderr)
        print()


if __name__ == "__main__":
    main()