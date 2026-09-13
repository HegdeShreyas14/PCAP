# Project Progress Log

**Project:** GPU-Accelerated Sparse Graphical Lasso — parallelising GGLasso's
block-sparse (connected-component) decomposition on consumer GPUs.

**Team:** Abhyuday Gandhi (240905036), Shreyas S Hegde (240905118),
Shrikrishna Bhakhta (240905214)

Newest entries at the top of each section. Update this as you go — it is the
basis for the progress reviews and for the paper's results section.

---

## Status at a glance

| Phase | State |
|---|---|
| Literature survey + methodology | **Submitted.** 10 recent refs; 5 DOIs still unverified |
| Phase 1 — reproduction, environment capture | **Done** |
| RQ1 — CPU baseline sweeps | **Done.** 1t + 8t, 384 configs each, seeded generator, paired |
| Profiling | **Done.** Decides the Phase 3 design |
| Percolation / structural analysis | **Done.** `results/percolation.csv`, 3600 rows, seeds 0/1/2, fixed generator |
| Phase 3 — GPU implementation | **Not started.** Design now settled by profiling |
| Cross-platform (secondary/tertiary machines) | **At risk** — team capacity |

---

## Headline findings

### 1. Block decomposition helps, but only in a bounded regime

Against a fair 8-thread CPU baseline, connected-component decomposition buys
roughly **2–5×** at statistically defensible lambda. Against a single-threaded
baseline it looks better, because 8 threads accelerate ADMM's one large
eigendecomposition while doing nothing for block_SGL's many small ones.

Regime medians over all 384 shared configs (paired, seeded generator):

| regime | 1t | 8t | retained |
|---|---|---|---|
| balanced | 4.46× | 3.57× | 80% |
| imbalanced | 6.68× | 4.95× | 74% |
| many_tiny | 2.39× | 1.82× | 76% |
| one_giant | 7.62× | 6.25× | 82% |

**Retention is uniform at 74–82%.** The contaminated (pre-bug-3) run showed
68–85%, which looked like a regime effect and was instance noise. This is
precisely what the paired comparison was for, and it is worth stating: the
threading penalty does not vary meaningfully by block-size regime.

Report the 8-thread numbers. The 1-thread run is the "no BLAS parallelism"
algorithmic upper bound and overstates the win by **1.1–2.8×** (retained 93%
down to 36%), worst at large p with high n/p, which is exactly where ADMM's
single large eigendecomposition can use all 8 cores:

- balanced p=800 n/p=10: 12.47× → 4.46× (36% retained)
- imbalanced p=800 n/p=10: 10.08× → 3.74× (37%)
- balanced p=400 n/p=5: 6.99× → 3.27× (47%)

`many_tiny` stays flat (1.06–1.33× at p≥400, 77–93% retained): the regime CPU
decomposition cannot help regardless of threading.

Of the 48 best-F1 rows, **16 are flagged `ns`** — the two runs' trial bands
overlap, so the difference is not resolvable. Every `retained > 100%` is
either `ns` or at p=100. The earlier reading that "threading sometimes helps
the block solver" was an artifact of unflagged noise.

When the graph does not fragment at all, block_SGL is *slightly slower* than
ADMM_SGL (0.89–0.99×, median 0.97× over n=49 configs): detection overhead with
no payoff. Keep this in the paper — it is the honest cost side.

### 2. The over-fragmentation trap

Speedup rises as lambda fragments the graph, but estimate quality collapses.
Pooled by detected block count (1t run, 384 configs):

| detected blocks | median speedup | range | median F1 | n |
|---|---|---|---|---|
| 1 | 0.97× | 0.89–0.99 | 0.156 | 49 |
| 2–9 | 1.02× | 0.92–1.17 | 0.512 | 25 |
| **10–49** | **2.23×** | 0.99–4.98 | **0.762** | 71 |
| 50–99 | 3.79× | 1.18–6.56 | 0.734 | 61 |
| 100–199 | 6.92× | 1.38–15.04 | 0.645 | 60 |
| 200+ | **23.82×** | 3.39–51.48 | **0.274** | 118 |

Speedup climbs monotonically while F1 peaks at 0.762 in the 10–49 bucket and
then falls to 0.274. Those two curves crossing is the trap in one figure, and
it is more persuasive than any table. Without the quality guard we would have
reported a 23.8× win on a solution that is mostly noise.

**Always report speedup at the best-F1 lambda, never at the fastest lambda.**

### 3. The cubic-work model does NOT predict wall-clock (important)

The s^3 proxy (`cubic_gain = p^3 / sum s_i^3`, and `F_max` = share of cubic
work in the largest block) says the largest block holds ~99% of "the work", so
Amdahl should cap any block-parallel scheme.

**Profiling contradicts this.** In the control config (balanced p=800, n/p=10,
lambda=0.10, 238 blocks, largest 214) the largest single `eigh` is **1.7% of
eigh time, ~1% of wall-clock**. There is no dominant kernel.

The proxy overweights the largest block because it ignores iteration counts
and per-call dispatch overhead. This is a genuine methodological finding worth
stating in the paper: a reviewer would otherwise assume the model holds.

### 4. Where the time actually goes (the Phase 3 answer)

| | high-headroom<br>imbalanced p=200 n/p=10 λ=0.15 | control<br>balanced p=800 n/p=10 λ=0.10 |
|---|---|---|
| detected | 59 blocks, largest 18 | 238 blocks, largest 214 |
| wall | 29.6 ms | 297 ms |
| `eigh` | 7.6 ms — **26%** | 174 ms — **58%** |
| Python glue (loop, `ix_` slicing, ADMM_SGL setup, inverse-perm) | 19.4 ms — **65%** | 105 ms — **35%** |
| `get_connected_components` | 1.0 ms — 3% | 12 ms — 4% |
| `block_diag` assembly | 1.6 ms — 5% | 5.9 ms — 2% |
| `eigh` calls | 831, mean 0.01 ms | 2173, mean 0.08 ms, max 3.0 ms |

Both configs are bound by Python per-block glue (35–65%) plus thousands of
sub-millisecond `eigh` calls that are mostly dispatch overhead.
`ADMM_stopping_criterion` alone is 0.12–0.35 s cumulative across 4k–10k
`np.linalg.norm` calls — pure per-iteration overhead.

**Phase 3 lever is batching, not a faster eigensolver.** Collapse the hundreds
of independent `ADMM_SGL` calls into a vectorised batched ADMM over
similarly-sized block groups: one batched `eigh`, batched prox, batched
stopping test per group. A GPU eig kernel alone touches at most 26% of runtime
where headroom exists and 58% in the control. "Accelerate the giant block"
only matters above ~214-node blocks, which this sweep never produces.

This is *measured* evidence for a batched design over the original
per-block-GPU proposal.

Supporting measurement: batched CPU `np.linalg.eigh` on stacked arrays gives
3.1× at 5x5, 1.4× at 15x15, and nothing at 50x50+. So batching is worth doing
and its benefit is concentrated exactly where the blocks are small.

### 5. Percolation explains the structure

Thresholding the empirical covariance and taking connected components is a
percolation process. Below a critical lambda a giant component holds most
variables; above it the giant shatters into comparable fragments. Because
block cost scales as s^3, a surviving giant holds nearly all the cubic work
however many crumbs surround it.

Measured at balanced p=400, n/p=10:

| lambda | blocks | largest | F_max |
|---|---|---|---|
| 0.05 | 8 | 392 (98% of p) | 100.0% |
| 0.10 | 77 | 95 (24%) | 50.6% |
| 0.15 | 120 | 29 (7%) | 36.9% |

The statistically optimal lambda falls as p grows (0.20–0.35 at p=100 down to
0.05–0.20 at p=800). Where it falls below the percolation threshold, the giant
survives at every usable lambda. That crossing, if confirmed, bounds the
block-parallel approach itself rather than any implementation.

Note `F_max` is a **cubic-work** fraction, not a node fraction. A graph can be
shattered into hundreds of components and still have F_max near 1.

### 6. Detection loses parallelism the ground truth has

Oracle check (F_max the TRUE block structure would give under perfect
detection) versus what is actually detected at the best-F1 lambda. Computed
per seed from `results/percolation.csv` (3600 rows, seeds 0/1/2, fixed
generator):

| regime | detected F_max (min–max, median) | oracle F_max |
|---|---|---|
| balanced | 50.4–100% (**99.9%**) | 20% |
| many_tiny | 33.7–100% (**99.9%**) | 2.5–20% |
| imbalanced | 60.0–100% (**99.3%**) | 100% |
| one_giant | 60.6–100% (**99.9%**) | 100% |

For balanced and many_tiny the true structure has abundant inter-block
parallelism — oracle F_max 20% and as low as **2.5%** (p=800, 40 true blocks),
so H = 80–97.5% — and **finite-sample noise merges the true blocks into a
giant**, leaving detected F_max at a median of 99.9%. For imbalanced and
one_giant the ground truth genuinely has none, by construction.

The detected-vs-oracle gap is therefore roughly **99.9% against 20%** (and
99.9% against 2.5% at p=800), not the narrower gap quoted in an earlier draft
of this log. Detection is destroying nearly all of the available inter-block
parallelism, not merely some of it.

This reframes the negative result precisely: graphical lasso does not lack
block parallelism — *component detection on a noisy empirical covariance
cannot find it at statistically useful lambda*. Better detection (shrunk or
regularised covariance, stability selection over subsamples, a detection
threshold decoupled from the solver's lambda) is a strong future-work item and
possibly a better lever than GPU work.

Note the wide min–max ranges (33.7–100%). Single-instance F_max is close to a
coin flip near the percolation threshold, which is why this table is quoted
from the multi-seed percolation run rather than from a single-draw sweep.

---

## Measurement discipline learned the hard way

Each of these silently corrupts results and each cost us time:

1. **Console I/O inside the timing path.** `block_SGL` prints one line per
   block; at ~100 blocks that is ~100 writes inside the measured region, and
   it penalises the block solver specifically. Suppressed.

2. **Run-to-run variance is ~20%.** An identical config with an identical seed
   gave 4.98×, 5.18× and 6.04× across three repeats of median-of-5. The
   harness now uses 9 trials and stores full trial vectors plus
   `speedup_lo`/`speedup_hi`. Never quote a speedup without its spread.

3. **BLAS threading confounds everything.** ADMM's one big eigendecomposition
   threads well; block_SGL's many small ones do not. Threads must be pinned
   *before* numpy is imported, and recorded. `--blas-threads 32` on a 16-core
   chip produced 1150–1442× nonsense (thread thrashing, not algorithm) and
   wasted ~3 h. Calibration showed 4–8 threads optimal on this CPU; 16 already
   regresses.

4. **Instance noise exceeds the effect being measured.** At fixed lambda,
   across 6 sample draws: speedup 2.03–4.26× (2.1× ratio), F_max 36%–97%
   (bimodal near threshold), while F1 stayed within 0.011. Structural
   quantities are far more instance-sensitive than statistical ones.
   *Paired* comparisons (same seed both sides) cancel this; absolute levels do
   not.

5. **Diagnostics must sit outside the timing path.** `measure=True` and
   `tracemalloc` both add overhead. Iteration counts and peak memory are
   collected in separate untimed runs.

---

## Bugs found and fixed

| # | Bug | Impact | Status |
|---|---|---|---|
| 1 | A second agent wrote its own `blockgen.py` with different regime definitions | Would have made three machines' CSVs incomparable | Synced; md5s checked on every machine |
| 2 | `nx.random_powerlaw_tree` fallback to Erdos-Renyi was silent and **machine-dependent** (succeeds on networkx 3.6.1, fails elsewhere) | Two machines running identical code could draw from different generative models | Model now returned per block and recorded as `gen_n_powerlaw` / `gen_n_erdos` / `gen_n_direct` |
| 3 | `sample_covariance_matrix` called **without a seed** — Theta reproducible, S was a fresh draw every call | Each lambda point in a sweep used different data; paired 1t→8t crossover contaminated | Fixed (`seed + 90000`); verified deterministic |
| 4 | `pinv` instead of `inv` for Sigma | Would mask a genuine non-PD bug | Fixed |
| 5 | `analyze.py` silently pooled multiple CSVs | Table 1 interleaved 1t and 8t operating points | Per-file reports; `--compare` joins on (regime, p, n/p, lambda) |
| 6 | Mean block size computed from truncated `detected_sizes_json` (200 entries) | Overstated by 15–60% on fragmented configs | Now exact: `p / detected_n_blocks` |
| 7 | `--out` without `.csv` collided the env JSON onto the CSV path | Data loss | Fixed |

Package quirks (not our bugs, but they bite):

- `block_SGL` returns `sol` only, not `(sol, info)`, despite its docstring.
  No per-block iteration count exists in gglasso 0.3.0 — `block_iterations`
  is permanently NaN.
- The penalty is **off-diagonal only** (`prox_od_1norm` copies the diagonal
  through untouched). Any GPU soft-threshold kernel must match this or it
  silently solves a different problem.
- `generate_precision_matrix(p, M)` asserts `M*L == p` — equal-sized blocks
  only. Three of our four regimes are impossible with it. Hence `blockgen.py`.
- `ADMM_SGL`'s info dict has no `iter` key; iteration count is
  `len(info['residual'])`.
- `ADMM_SGL` and `block_SGL` ship with different default tolerances. Unequal
  settings manufacture fake speedups.

---

## Artifacts

| File | What it is |
|---|---|
| `src/phase1_repro.py` | Install check + environment fingerprint. Run once per machine |
| `src/blockgen.py` | Block-sparse ground truth with arbitrary block-size distributions |
| `src/sweep.py` | RQ1 harness: CPU ADMM_SGL vs CPU block_SGL, 66-column CSV |
| `src/analyze.py` | Tables 1–4, `--compare` for the threading crossover |
| `src/profile_block.py` | cProfile with linalg-vs-glue attribution |
| `src/percolation.py` | Percolation threshold vs optimal lambda, multi-seed, oracle diagnostic |
| `results/percolation.csv` | 3600 rows, seeds 0/1/2, fixed generator. Owns the structural claim |
| `results/rq1_primary_1t.csv` | 384 configs, 1 BLAS thread, seeded generator |
| `results/rq1_primary_8t.csv` | 384 configs, 8 BLAS threads, seeded generator. **Report these numbers** |
| `results/archive_prefix_bug3/` | Pre-fix contaminated pair. Kept deliberately |
| `notes/RUNBOOK.md` | How to run everything: thread calibration, thermal/power protocol, git |
| `notes/PROGRESS.md` | **This file — single source of truth.** `findings.md` was retired into it |


---

## Open items

1. ~~Re-run both sweeps~~ **DONE.** 1t (75 min) and 8t on the seeded
   generator; crossover is now a clean paired comparison. Pre-fix pair kept in
   `results/archive_prefix_bug3/` as evidence the problem was found and
   corrected — do not delete it.
2. ~~Percolation run~~ **DONE** (Sep 2, fixed generator). Table 4's F_max is
   quoted from `results/percolation.csv` — multi-seed — not from a single-draw
   sweep. See finding 6.
3. **Phase 3**: implement batched ADMM over similarly-sized block groups.
   Design settled by profiling. Consider a batched-CPU baseline as well as
   GPU, so the paper separates the batching idea from the hardware.
4. **Verify 5 DOIs** — R2, R5, R6, R8, R9. Ten minutes, on a submitted
   deliverable.
5. **Two-column port**: Figure 1 needs `figure*` in the Springer/IEEE
   templates or the four boxes overlap.
6. **Cross-platform** (secondary/tertiary machines) — at risk. The
   memory-boundary half of RQ3 is answerable on the primary machine alone;
   the cross-platform table would be dropped and the scope stated as a
   limitation.

---

## What the paper's story now is

Not "we made GGLasso faster on a GPU". Rather:

> We characterise the parallelism structure of GGLasso's connected-component
> decomposition. Block count is a poor proxy for exploitable parallelism,
> because the largest component dominates cubically and because the lambda
> that fragments the graph is close to the lambda that destroys the estimate.
> Profiling shows the runtime is bound by per-block Python overhead and
> dispatch, not by the eigendecompositions the cubic-work model points at, so
> the effective GPU strategy is batched execution over block groups rather
> than acceleration of individual blocks. We further show that the true block
> structure does contain substantial inter-block parallelism which
> finite-sample component detection fails to recover, identifying detection
> rather than arithmetic as the binding constraint.

That is a stronger and more defensible contribution than a speedup number, and
every part of it is measured.