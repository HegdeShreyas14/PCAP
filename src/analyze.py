"""
analyze.py — turn sweep CSVs into the numbers that go in the paper.

The central quantity is not raw speedup. It is the fraction of the
*theoretically available* speedup that block_SGL actually realizes, and the
ceiling that the largest connected component places on any block-parallel
scheme.

Definitions
-----------
    theoretical = p^3 / sum_i s_i^3     arithmetic gain available from the
                                        decomposition (ADMM's per-iteration
                                        cost is dominated by an O(p^3) eigh)

    realized    = observed / theoretical   near 1 => arithmetic-bound;
                                           near 0 => overhead-bound

    critical    = p^3 / max_i(s_i)^3    ceiling on ANY block-parallel scheme.
                                        The blocks are independent, so with
                                        unlimited hardware the runtime is set
                                        by the largest block alone.

    max_share   = max_i(s_i)^3 / sum_i s_i^3    fraction of block arithmetic
                                                in the single largest block

Note on mean block size: connected components partition the p variables, so
mean block size is exactly p / detected_n_blocks. We do NOT compute it from
detected_sizes_json, which sweep.py truncates to the 200 largest blocks --
that truncation inflates the mean by 15-60% on highly fragmented configs.
The truncation does not affect `theoretical`, `critical` or `max_share`: the
dropped blocks are all singletons (1^3 is negligible in sum s_i^3) and the
largest block is always retained by the descending sort.

Usage
-----
    python src/analyze.py results/rq1_primary_8t.csv
    python src/analyze.py --compare results/rq1_primary_1t.csv results/rq1_primary_8t.csv

Passing several files WITHOUT --compare analyses each separately, because
pooling runs from different thread settings would silently interleave their
operating points.
"""

import argparse
import csv
import json
import statistics
import sys


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def theoretical_speedup(p, detected_sizes):
    denom = sum(float(s) ** 3 for s in detected_sizes)
    return (float(p) ** 3) / denom if denom > 0 else float("nan")


def critical_path(p, detected_sizes):
    """
    Ceiling on any block-parallel scheme: Amdahl applied to the decomposition.

    Measured on real sweeps the largest block holds 78-100% of the total block
    work, so block COUNT is a poor predictor of available parallelism and
    largest-block SIZE is the one that matters.
    """
    if not detected_sizes:
        return float("nan")
    mx = float(max(detected_sizes)) ** 3
    return (float(p) ** 3) / mx if mx > 0 else float("nan")


def max_block_share(detected_sizes):
    tot = sum(float(s) ** 3 for s in detected_sizes)
    if tot <= 0:
        return float("nan")
    return (float(max(detected_sizes)) ** 3) / tot


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

REQUIRED = ("regime", "p", "lambda", "detected_n_blocks", "algorithmic_speedup",
            "support_f1", "n_samples", "detected_sizes_json")


def load_one(path):
    """Load one CSV. Reports skipped rows rather than swallowing them."""
    rows, skipped, reasons = [], 0, {}
    with open(path) as f:
        reader = csv.DictReader(f)
        missing = [c for c in REQUIRED if c not in (reader.fieldnames or [])]
        if missing:
            print(f"  !! {path}: missing columns {missing} — schema mismatch, skipping file")
            return []
        for r in reader:
            try:
                r["p"] = int(r["p"])
                r["lambda"] = float(r["lambda"])
                r["detected_n_blocks"] = int(r["detected_n_blocks"])
                r["algorithmic_speedup"] = float(r["algorithmic_speedup"])
                r["support_f1"] = float(r["support_f1"])
                r["n_over_p"] = int(r["n_samples"]) // r["p"]
                sizes = json.loads(r["detected_sizes_json"])
                r["_sizes"] = sizes
                r["theoretical"] = theoretical_speedup(r["p"], sizes)
                r["critical"] = critical_path(r["p"], sizes)
                r["max_share"] = max_block_share(sizes)
                r["max_block"] = max(sizes) if sizes else 0
                # exact: components partition the variables. Do not derive
                # this from _sizes, which is truncated to 200 entries.
                r["mean_block"] = (r["p"] / r["detected_n_blocks"]
                                   if r["detected_n_blocks"] else float("nan"))
                r["realized"] = (r["algorithmic_speedup"] / r["theoretical"]
                                 if r["theoretical"] > 0 else float("nan"))
                r["_src"] = path
                rows.append(r)
            except (ValueError, KeyError, json.JSONDecodeError) as e:
                skipped += 1
                reasons[type(e).__name__] = reasons.get(type(e).__name__, 0) + 1
    if skipped:
        print(f"  !! {path}: skipped {skipped} unparseable rows {reasons}")
    return rows


def isnan(x):
    return x != x


def fmt(x, w=6, d=2):
    try:
        if isnan(x):
            return " " * max(0, w - 3) + "n/a"
        return f"{x:{w}.{d}f}"
    except (TypeError, ValueError):
        return " " * max(0, w - 3) + "n/a"


def best_f1_rows(rows):
    """One row per (regime, p, n/p): the lambda maximising support_f1."""
    out = []
    for key in sorted({(r["regime"], r["p"], r["n_over_p"]) for r in rows}):
        sub = [r for r in rows
               if (r["regime"], r["p"], r["n_over_p"]) == key and not isnan(r["support_f1"])]
        if sub:
            out.append(max(sub, key=lambda r: r["support_f1"]))
    return out


# --------------------------------------------------------------------------
# single-file report
# --------------------------------------------------------------------------

def report(rows, label):
    print()
    print("#" * 92)
    print(f"# {label}   ({len(rows)} rows)")
    print("#" * 92)

    best = best_f1_rows(rows)

    print()
    print("TABLE 1 — speedup at the best-F1 lambda (the defensible operating point)")
    print("-" * 92)
    print(f"{'regime':12s} {'p':>5s} {'n/p':>4s} {'lam':>5s} {'blks':>5s} "
          f"{'F1':>6s} {'obs':>7s} {'theory':>9s} {'realized':>9s}")
    for b in best:
        print(f"{b['regime']:12s} {b['p']:5d} {b['n_over_p']:4d} {b['lambda']:5.2f} "
              f"{b['detected_n_blocks']:5d} {fmt(b['support_f1'])} "
              f"{fmt(b['algorithmic_speedup'],6)}x {fmt(b['theoretical'],8,1)}x "
              f"{fmt(b['realized'],9,3)}")

    print()
    print("TABLE 2 — the over-fragmentation trap: speed rises as quality collapses")
    print("-" * 92)
    print(f"{'detected blocks':>16s} {'median F1':>10s} {'median obs':>11s} "
          f"{'median realized':>16s} {'n':>5s}")
    buckets = [(1, 1), (2, 9), (10, 49), (50, 99), (100, 199), (200, None)]
    for lo, hi in buckets:
        sub = [r for r in rows
               if r["detected_n_blocks"] >= lo
               and (hi is None or r["detected_n_blocks"] <= hi)]
        if not sub:
            continue
        f1 = [r["support_f1"] for r in sub if not isnan(r["support_f1"])]
        obs = [r["algorithmic_speedup"] for r in sub]
        rea = [r["realized"] for r in sub if not isnan(r["realized"])]
        label_b = f"{lo}" if hi == lo else (f"{lo}+" if hi is None else f"{lo}-{hi}")
        print(f"{label_b:>16s} {fmt(statistics.median(f1) if f1 else float('nan'),10,3)} "
              f"{fmt(statistics.median(obs),10)}x "
              f"{fmt(statistics.median(rea) if rea else float('nan'),16,3)} {len(sub):5d}")
    print("\nHigh speedup in a bucket where F1 has collapsed is not a usable operating")
    print("point. Report Table 1, not the peak of this table.")

    print()
    print("TABLE 3 — realized fraction by mean block size (= p / detected_n_blocks)")
    print("-" * 92)
    print(f"{'mean block size':>16s} {'median theory':>14s} {'median obs':>11s} "
          f"{'realized':>10s} {'n':>5s}")
    size_buckets = [(0, 5), (5, 10), (10, 25), (25, 50), (50, 150), (150, None)]
    for lo, hi in size_buckets:
        sub = [r for r in rows
               if not isnan(r["mean_block"]) and r["mean_block"] >= lo
               and (hi is None or r["mean_block"] < hi)]
        if not sub:
            continue
        th = [r["theoretical"] for r in sub if not isnan(r["theoretical"])]
        obs = [r["algorithmic_speedup"] for r in sub]
        rea = [r["realized"] for r in sub if not isnan(r["realized"])]
        label_b = f"{lo}+" if hi is None else f"{lo}-{hi}"
        print(f"{label_b:>16s} {fmt(statistics.median(th) if th else float('nan'),13,1)}x "
              f"{fmt(statistics.median(obs),10)}x "
              f"{fmt(statistics.median(rea) if rea else float('nan'),10,3)} {len(sub):5d}")
    print("\nRealized fraction falling as blocks shrink means per-block overhead, not")
    print("linear algebra, is the binding constraint in that regime.")

    print()
    print("TABLE 4 — critical path: share of block work in the LARGEST block")
    print("-" * 92)
    print(f"{'regime':12s} {'p':>5s} {'n/p':>4s} {'blks':>5s} {'max':>5s} "
          f"{'max share':>10s} {'obs':>7s} {'ceiling':>9s}")
    for b in best:
        print(f"{b['regime']:12s} {b['p']:5d} {b['n_over_p']:4d} "
              f"{b['detected_n_blocks']:5d} {b['max_block']:5d} "
              f"{fmt(100*b['max_share'],9,1)}% {fmt(b['algorithmic_speedup'],6)}x "
              f"{fmt(b['critical'],8,1)}x")
    shares = [b["max_share"] for b in best if not isnan(b["max_share"])]
    if shares:
        print(f"\nMedian share of block arithmetic in the single largest block: "
              f"{100*statistics.median(shares):.1f}%")
    print("""
Amdahl applied to the decomposition. Blocks are independent, so with unlimited
parallel hardware the runtime is set by the largest block alone (the 'ceiling'
column). Where the largest block holds most of the work, batching the remaining
blocks onto a GPU cannot approach that ceiling, however many there are.""")

    cand = [b for b in best
            if not isnan(b["max_share"]) and b["max_share"] < 0.9
            and b["detected_n_blocks"] >= 10]
    print()
    print("PHASE 3 TARGETING — configs where block-parallelism has real headroom")
    print("-" * 92)
    if not cand:
        print("None: in every best-F1 config the largest block holds >=90% of the work.")
        print("Batching across blocks is not the right GPU strategy here. Report this")
        print("as a finding and target the largest block's dense eigendecomposition.")
    else:
        for b in sorted(cand, key=lambda r: r["max_share"])[:10]:
            print(f"  {b['regime']:11s} p={b['p']:4d} n/p={b['n_over_p']:2d} "
                  f"lam={b['lambda']:.2f}: {b['detected_n_blocks']:4d} blocks, "
                  f"largest {b['max_block']:4d} ({100*b['max_share']:.1f}% of work), "
                  f"obs {b['algorithmic_speedup']:.2f}x, ceiling {b['critical']:.1f}x")


# --------------------------------------------------------------------------
# compare mode
# --------------------------------------------------------------------------

def compare(rows_a, label_a, rows_b, label_b):
    """
    Join two runs on (regime, p, n/p, lambda) and show the crossover.

    Operating points are chosen from run A, then the SAME configuration is
    looked up in run B. Picking each run's own best-F1 lambda independently
    would compare different lambdas and is not a crossover.
    """
    key = lambda r: (r["regime"], r["p"], r["n_over_p"], round(r["lambda"], 4))
    A = {key(r): r for r in rows_a}
    B = {key(r): r for r in rows_b}
    shared = A.keys() & B.keys()
    print()
    print("#" * 92)
    print(f"# CROSSOVER: {label_a}  ->  {label_b}")
    print(f"# {len(shared)} configurations present in both")
    print("#" * 92)

    print()
    print("Regime medians (all shared configs)")
    print("-" * 92)
    print(f"{'regime':12s} {label_a[:10]:>10s} {label_b[:10]:>10s} {'retained':>10s}")
    for rg in sorted({k[0] for k in shared}):
        sa = [A[k]["algorithmic_speedup"] for k in shared if k[0] == rg]
        sb = [B[k]["algorithmic_speedup"] for k in shared if k[0] == rg]
        if not sa:
            continue
        ma, mb = statistics.median(sa), statistics.median(sb)
        print(f"{rg:12s} {ma:9.2f}x {mb:9.2f}x {100*mb/ma:9.0f}%")

    print()
    print("At run-A's best-F1 lambda, same lambda in both")
    print("-" * 92)
    print(f"{'regime':12s} {'p':>5s} {'n/p':>4s} {'lam':>5s} {'F1':>6s} "
          f"{label_a[:8]:>9s} {label_b[:8]:>9s} {'retained':>9s}  flags")
    for b in best_f1_rows([A[k] for k in shared]):
        k = key(b)
        if k not in B:
            continue
        other = B[k]
        sa, sb = b["algorithmic_speedup"], other["algorithmic_speedup"]

        flags = []
        # Overlapping trial bands mean the difference is not resolvable by
        # this measurement, whatever the ratio says. This is why 'retained'
        # can exceed 100%: it is noise, not the threaded run being faster.
        try:
            alo, ahi = float(b["speedup_lo"]), float(b["speedup_hi"])
            blo, bhi = float(other["speedup_lo"]), float(other["speedup_hi"])
            if alo <= bhi and blo <= ahi:
                flags.append("ns")
        except (KeyError, ValueError):
            pass
        if b["support_f1"] < 0.6:
            flags.append("junk-F1")

        tag = ("  " + ", ".join(flags)) if flags else ""
        print(f"{b['regime']:12s} {b['p']:5d} {b['n_over_p']:4d} {b['lambda']:5.2f} "
              f"{fmt(b['support_f1'])} {sa:8.2f}x {sb:8.2f}x "
              f"{100*sb/sa if sa else float('nan'):8.0f}%{tag}")

    print("\nflags:  ns = trial bands of the two runs overlap; the difference is not")
    print("             resolvable by this measurement. Do not read these as a")
    print("             threading effect, in either direction.")
    print("        junk-F1 = estimate quality has collapsed; not a usable operating point.")


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+")
    ap.add_argument("--compare", action="store_true",
                    help="join exactly two CSVs on (regime,p,n/p,lambda) and "
                         "show the crossover instead of separate reports")
    args = ap.parse_args()

    loaded = [(path, load_one(path)) for path in args.csvs]
    loaded = [(p, r) for p, r in loaded if r]
    if not loaded:
        print("No usable rows found.")
        sys.exit(1)

    if args.compare:
        if len(loaded) != 2:
            print(f"--compare needs exactly two usable CSVs, got {len(loaded)}.")
            sys.exit(1)
        (pa, ra), (pb, rb) = loaded
        compare(ra, pa.split("/")[-1], rb, pb.split("/")[-1])
        return

    # Separate report per file. Pooling runs from different thread settings
    # would interleave their operating points and produce a table that is
    # neither run.
    for path, rows in loaded:
        report(rows, path)
    if len(loaded) > 1:
        print()
        print("Reported each file separately. For a 1t-vs-8t crossover use --compare.")


if __name__ == "__main__":
    main()