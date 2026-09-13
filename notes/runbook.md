# Runbook

How to run everything. `notes/PROGRESS.md` is the source of truth for what has
been done and what was found; this file is how to do it.

---

## Setup (per machine, once)

```bash
git clone <repo> && cd <repo>
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python src/phase1_repro.py         # paste ENVIRONMENT block into PROGRESS.md
```

PyTorch is deliberately **not** in requirements.txt — the correct wheel depends
on your CUDA version. Get the command from https://pytorch.org, then verify:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Needed for Phase 3 only. RQ1 is CPU-only.

---

## Before ANY timed run

- Mains power. Battery mode throttles.
- Fix the CPU/GPU power profile and write down which.
- Close browsers, IDEs, chat apps. A background tab is worth more variance
  than the effect being measured.
- Don't use the machine while it runs.
- **Never run two sweeps concurrently**, or one sweep alongside percolation.
  They compete for cores and you end up measuring contention.

---

## BLAS threads: always pass `--blas-threads` explicitly

NumPy sends eigendecomposition to a threaded BLAS. `ADMM_SGL` does one large
eigh that threads well; `block_SGL` does many small ones that don't. So the
thread count changes the measured speedup, and an unpinned run is not
reproducible.

Threads are pinned before numpy is imported (a pre-parser) — setting the env
var afterwards is ignored.

Calibrated on the primary machine (Ryzen 9 8945HX, 16 physical / 32 logical):

| threads | ADMM p=800 | verdict |
|---|---|---|
| 1 | 2.20s | algorithmic baseline |
| 4 | 1.02s | ~2x faster |
| 8 | 1.12s | same as 4 — **use this** |
| 16 | 1.95s | regresses (SMT contention + thermal throttle) |
| 32 | — | pathological: produced 1150–1442x nonsense, cost ~3h |

**Do not pass more threads than physical cores.**

---

## RQ1 sweeps

```bash
nohup python src/sweep.py --machine primary --blas-threads 1 \
      --out results/rq1_primary_1t.csv > logs/1t.log 2>&1 &

# only after that finishes:
nohup python src/sweep.py --machine primary --blas-threads 8 \
      --out results/rq1_primary_8t.csv > logs/8t.log 2>&1 &
```

384 configs (4 regimes x 4 p x 3 n/p ratios x 8 lambda), 2 warm-up + 9 timed
runs of two solvers each, plus untimed diagnostics. Roughly 77 min (1t) and
42 min (8t) on the primary machine.

Report the **8-thread** numbers. The 1-thread run is the "no BLAS parallelism"
algorithmic upper bound and overstates the win by 1.2–2.7x at large p.

Both runs must use the **same seed** so the crossover is a paired comparison —
instance noise (~2x on speedup) cancels within a pair but not across.

---

## Analysis

```bash
# paired threading crossover
python src/analyze.py --compare results/rq1_primary_1t.csv results/rq1_primary_8t.csv

# full tables for one run
python src/analyze.py results/rq1_primary_8t.csv
```

Passing several CSVs *without* `--compare` reports each separately, on purpose:
pooling runs from different thread settings interleaves their operating points.

`--compare` holds lambda fixed across both runs (run A's best-F1 lambda applied
to both). Picking each run's own best lambda would compare different
configurations, which is not a crossover.

Flags in the output: `ns` = the two runs' trial bands overlap, so the
difference isn't resolvable — this is why "retained" can exceed 100%.
`junk-F1` = estimate quality has collapsed; not a usable operating point.

**Quote F_max from `results/percolation.csv`, not from Table 4 of a single
sweep.** Single-instance F_max is close to a coin flip near the percolation
threshold (measured 36% vs 97% across draws at the same lambda). The
percolation run is multi-seed and owns that claim.

---

## Percolation (structural claim)

```bash
nohup python src/percolation.py --seeds 0 1 2 --blas-threads 8 \
      --out results/percolation.csv > logs/perc.log 2>&1 &
```

Already done (3600 rows, seeds 0/1/2, fixed generator). Re-run only if the
generator changes again.

`--no-f1` is structure-only and much faster, but yields **no verdict** —
lambda*(F1) needs the solves.

---

## Profiling

Profile two classes, not one. The contrast is the point.

```bash
# high headroom: where inter-block work actually exists
python src/profile_block.py --regime imbalanced --p 200 --ratio 10 --lam 0.15

# control: largest-block dominated
python src/profile_block.py --regime balanced --p 800 --ratio 10 --lam 0.10
```

Result (see PROGRESS.md finding 4): both are bound by Python per-block glue
(35–65%) plus thousands of sub-millisecond `eigh` calls that are mostly
dispatch overhead. The largest single `eigh` is ~1% of wall-clock. **The
Phase 3 lever is batching, not a faster eigensolver.**

---

## Git

```bash
git add -A && git commit -m "..."
git tag rq1-v3-seeded          # tag the commit that produced each dataset
git push --tags
```

- Commit result CSVs **with** their `_env.json`, in the same commit. They are
  small (~450 KB) and they are the data. If `results/` is in `.gitignore`,
  take it out.
- Three code corrections have already invalidated earlier datasets. Without
  tags you cannot tell which CSV came from which code version.
- Log who did what, dated, in PROGRESS.md. "Contribution of each team member"
  is part of the demo marks and is miserable to reconstruct in October.