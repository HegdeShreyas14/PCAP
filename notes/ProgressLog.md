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
| RQ1 — CPU baseline sweeps | **Done, needs re-run** on the fixed generator |
| Profiling | **Done.** Decides the Phase 3 design |
| Percolation / structural analysis | **Running** on the fixed generator |
| Phase 3 — GPU implementation | **Not started.** Design now settled by profiling |
| Cross-platform (secondary/tertiary machines) | **At risk** — team capacity |

---

## Headline findings

### 1. Block decomposition helps, but only in a bounded regime

Against a fair 8-thread CPU baseline, connected-component decomposition buys
roughly **2–5×** at statistically defensible lambda. Against a single-threaded
baseline it looks better (**~4–12×**), because 8 threads accelerate ADMM's one
large eigendecomposition while doing nothing for block_SGL's many small ones.

Report the 8-thread numbers. The 1-thread run is the "no BLAS parallelism"
algorithmic upper bound and overstates the win by 1.2–2.7× at large p.

When the graph does not fragment at all, block_SGL is *slightly slower* than
ADMM_SGL (0.95–0.98×): detection overhead with no payoff. Keep this in the
paper — it is the honest cost side.

### 2. The over-fragmentation trap

Speedup rises as lambda fragments the graph, but estimate quality collapses.
Pooled by detected block count, the 200+ bucket shows median ~23× speedup at
median F1 0.27. Without a quality guard we would have reported a 23× win on a
solution that is mostly noise.

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
detection) versus what is actually detected at the best-F1 lambda:

| regime | oracle F_max | detected F_max |
|---|---|---|
| balanced | 20% | 68–91% |
| many_tiny | 10–20% | 91% |
| imbalanced | 100% | 87–90% |

For balanced and many_tiny the true structure has abundant inter-block
parallelism (H = 80–90%) and **finite-sample noise merges the true blocks into
a giant**. For imbalanced the ground truth genuinely has none, by construction.

This reframes the negative result precisely: graphical lasso does not lack
block parallelism — *component detection on a noisy empirical covariance
cannot find it at statistically useful lambda*. Better detection (shrunk or
regularised covariance, stability selection over subsamples, a detection
threshold decoupled from the solver's lambda) is a strong future-work item and
possibly a better lever than GPU work.

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
| `results/rq1_primary_1t.csv` | 384 configs, 1 BLAS thread — **re-run needed** (bug 3) |
| `results/rq1_primary_8t.csv` | 384 configs, 8 BLAS threads — **re-run needed** (bug 3) |
| `notes/RUNBOOK.md` | How to run everything, thermal/power protocol |
| `notes/findings.md` | Running log |

---

## Open items

1. **Re-run both sweeps** at seed 0 on the fixed generator (~77 + 42 min).
   Paired comparison fixes the crossover; instance noise cancels within a pair.
2. **Percolation run** on the fixed generator — owns the structural claim, so
   Table 4's F_max should be quoted from here (multi-seed), not from a
   single-draw sweep.
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
