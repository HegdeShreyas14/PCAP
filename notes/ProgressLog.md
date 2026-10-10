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
| RQ1 — CPU baseline sweeps | **Done (re-run 2026-10-10, commit ebd6d6d).** 1t + 8t, 384 configs each, pairing verified 384/384 |
| Profiling | **Done.** Decides the Phase 3 design |
| Percolation / structural analysis | **Done (re-run 2026-10-10, commit ebd6d6d).** 3600 rows, seeds 0/1/2; seed 0 matches RQ1 8t on 192/192 points |
| Phase 3 — GPU implementation | **Done; re-measured 2026-10-10 (commit cf112c1).** Batched CPU + torch solver, crossover, VRAM model |
| Cross-platform (secondary/tertiary machines) | **At risk** — team capacity |

---

## Headline findings

### 1. Block decomposition helps, but only in a bounded regime

Against a fair 8-thread CPU baseline, connected-component decomposition buys
roughly **2–5×** at statistically defensible lambda. Against a single-threaded
baseline it looks better, because 8 threads accelerate ADMM's one large
eigendecomposition while doing nothing for block_SGL's many small ones.

Re-run 2026-10-10 after bug 8. Pairing is now verified: 1t and 8t agree on S
and on detected structure in 384/384 configs (`results/runlogs/rq1_compare.log`).

Regime medians over all 384 shared configs:

| regime | 1t | 8t | retained |
|---|---|---|---|
| balanced | 3.83× | 3.40× | 89% |
| imbalanced | 6.74× | 4.83× | 72% |
| many_tiny | 2.08× | 1.66× | 80% |
| one_giant | 7.27× | 5.71× | 79% |

**Retention is governed by p, not by regime.** At the best-F1 lambda:

| p | median retained | range | `ns` |
|---|---|---|---|
| 100 | 96% | 89–110% | 7/12 |
| 200 | 96% | 82–114% | 4/12 |
| 400 | 65% | 59–90% | 0/12 |
| 800 | 57% | 40–91% | 0/12 |

(All configs, not just best-F1: 98% / 92% / 65% / 48% for p = 100 / 200 / 400 / 800.)
At p ≤ 200 threading makes no resolvable difference; from p = 400 the 8-thread
baseline claws back a third to a half of the win, because ADMM's single p×p
eigendecomposition is finally large enough to use 8 cores. This replaces the
earlier "uniform 74–82%" claim, which came from unpaired runs (bug 8).

The one exception at p ≥ 400 is `many_tiny`, which keeps 83–91%: its speedup
is only ~1.3× to begin with, so there is little for the threaded baseline to
take back. The other three regimes lose 40–60% there. Put precisely: the
threaded baseline erodes the win *where a win exists*, and the erosion grows
with p (Fig 1b; the p=800 cluster near 90% is `many_tiny`).

Report the 8-thread numbers. The 1-thread run is the "no BLAS parallelism"
algorithmic upper bound and overstates the win by up to **2.5×** at large p:

- balanced p=800 n/p=10: 10.03× → 4.78× (48% retained)
- imbalanced p=800 n/p=10: 8.37× → 4.40× (53%)
- balanced p=400 n/p=10: 3.56× → 2.14× (60%)

`many_tiny` stays flat (0.99–1.45× at 8t for p≥400, 83–91% retained): the
regime CPU decomposition cannot help regardless of threading.

Of the 48 best-F1 rows, **11 are flagged `ns`**, all at p ≤ 200. Every
`retained > 100%` row is `ns`.

When the graph does not fragment at all, block_SGL is *slightly slower* than
ADMM_SGL (0.93–1.01× at 1t, 0.88–1.03× at 8t, median 0.97× over n=50 configs): detection overhead with
no payoff. Keep this in the paper — it is the honest cost side.

### 2. The over-fragmentation trap

Speedup rises as lambda fragments the graph, but estimate quality collapses.
Pooled by detected block count (1t run, 384 configs):

| detected blocks | median speedup | range | median F1 | n |
|---|---|---|---|---|
| 1 | 0.97× | 0.93–1.01 | 0.152 | 50 |
| 2–9 | 1.03× | 0.98–1.28 | 0.490 | 21 |
| **10–49** | **2.01×** | 1.00–5.61 | **0.750** | 77 |
| 50–99 | 3.66× | 1.27–7.28 | 0.734 | 58 |
| 100–199 | 7.16× | 1.53–15.97 | 0.654 | 60 |
| 200+ | **23.93×** | 2.86–51.61 | **0.286** | 118 |

(8t: same buckets and F1, medians 0.97 / 1.03 / 1.91 / 3.31 / 5.52 / 13.11×.)

Speedup climbs monotonically while F1 peaks at 0.750 in the 10–49 bucket and
then falls to 0.286. Those two curves crossing is the trap in one figure, and
it is more persuasive than any table. Without the quality guard we would have
reported a 23.9× win on a solution that is mostly noise.

**Always report speedup at the best-F1 lambda, never at the fastest lambda.**

### 3. The cubic-work model does NOT predict wall-clock (important)

The s^3 proxy (`cubic_gain = p^3 / sum s_i^3`, and `F_max` = share of cubic
work in the largest block) says the largest block holds ~99% of "the work", so
Amdahl should cap any block-parallel scheme.

**Profiling contradicts this.** In the control config (balanced p=800, n/p=10,
lambda=0.10, 244 blocks, largest 192, cubic F_max 99.6%) the largest single
`eigh` is **1.4% of eigh time, under 1% of wall-clock**. There is no dominant kernel.

The proxy overweights the largest block because it ignores iteration counts
and per-call dispatch overhead. This is a genuine methodological finding worth
stating in the paper: a reviewer would otherwise assume the model holds.

### 4. Where the time actually goes (the Phase 3 answer)

| | high-headroom<br>imbalanced p=200 n/p=10 λ=0.15 | control<br>balanced p=800 n/p=10 λ=0.10 |
|---|---|---|
| detected | 58 blocks, largest 30 | 244 blocks, largest 192 |
| wall | 24.1 ms | 215 ms |
| `eigh` | 7.5 ms — **31%** | 126 ms — **58%** |
| Python glue (loop, `ix_` slicing, ADMM_SGL setup, inverse-perm) | 15.0 ms — **62%** | 76 ms — **35%** |
| `get_connected_components` | 0.6 ms — 3% | 8.0 ms — 4% |
| `block_diag` assembly | 1.0 ms — 4% | 5.3 ms — 2% |
| `eigh` calls | 801, mean 0.01 ms | 2302, mean 0.05 ms, max 1.7 ms |

Re-measured 2026-10-10 on commit ebd6d6d, 1 BLAS thread, median of 7
(`results/runlogs/profile_*.log`).

Both configs are bound by Python per-block glue (35–62%) plus thousands of
sub-millisecond `eigh` calls that are mostly dispatch overhead.
`ADMM_stopping_criterion` alone is 0.12–0.34 s cumulative over the profiled
repeats — pure per-iteration overhead.

**Phase 3 lever is batching, not a faster eigensolver.** Collapse the hundreds
of independent `ADMM_SGL` calls into a vectorised batched ADMM over
similarly-sized block groups: one batched `eigh`, batched prox, batched
stopping test per group. A GPU eig kernel alone touches at most 31% of runtime
where headroom exists and 58% in the control. "Accelerate the giant block"
only matters for blocks well above the ~200-node maximum this sweep produces
(the RQ2 crossover puts it at s_max ≈ 1000–1500).

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

Measured at balanced p=400, n/p=10 (`results/percolation.csv`, seed 0; the
lambda grid is 0.02:0.02:0.50):

| lambda | blocks | largest | F_max |
|---|---|---|---|
| 0.06 | 21 | 354 (89% of p) | 100.0% |
| 0.10 | 79 | 164 (41%) | 99.8% |
| 0.14 | 113 | 66 (17%) | 86.7% |
| 0.16 | 132 | 28 (7%) | 35.6% |

The giant breaks between 0.14 and 0.16. Seed 1 is similar (F_max 54.8% at
0.14); seed 2 never breaks in this range (largest block 225, F_max 100% at
0.16). That seed-to-seed spread is why thresholds are quoted multi-seed.

The statistically optimal lambda falls as p grows (0.20–0.35 at p=100 down to
0.05–0.20 at p=800). Where it falls below the percolation threshold, the giant
survives at every usable lambda. That crossing, if confirmed, bounds the
block-parallel approach itself rather than any implementation.

Note `F_max` is a **cubic-work** fraction, not a node fraction. A graph can be
shattered into hundreds of components and still have F_max near 1.

### 6. The estimator, not detection, merges the true blocks

Oracle F_max (what the TRUE block structure would allow) versus the F_max of
the components at the best-F1 lambda, per seed from `results/percolation.csv`:

| regime | detected F_max (min–max, median) | oracle F_max |
|---|---|---|
| balanced | 51.4–100% (**99.9%**) | 20% |
| many_tiny | 26.2–100% (**99.9%**) | 2.5–20% |
| imbalanced | 43.1–100% (**99.4%**) | 100% |
| one_giant | 58.7–100% (**99.9%**) | 100% |

For balanced and many_tiny the truth has abundant inter-block parallelism
(H = 80–97.5%) and the components have almost none. For imbalanced and
one_giant the truth has none by construction.

**This is not a detection failure.** Thresholding |S| at lambda and taking
connected components gives *exactly* the vertex partition of the graphical
lasso solution at that lambda (Witten, Friedman & Simon 2011; Mazumder & Hastie
2012; gglasso's `block_SGL` cites the former). Verified on our instances:
solving the full problem without decomposition (tol 1e-10) gives an identical
partition in **36/36** best-F1 cells at p ≤ 400
(`results/screening_exactness.csv`). The giant component is a property of the
graphical lasso *estimate* at its statistically best lambda, not of the
screening step.

**Mechanism: a handful of spurious cross-block entries fuses the true blocks**
(`results/screening_cross_block.csv`, 72 best-F1 cells). In balanced, only a
median **3–4%** of above-threshold entries of S join two different true blocks
(e.g. 33–42 of ~1,900 at p=800, n/p=10), yet that fuses all 5 true blocks into
the giant at every p ≥ 200. In many_tiny at p ≥ 400 the spurious share is
~25% and all 20–40 true blocks fuse. The giant fuses ≥ 2 true blocks in
68/72 cells. Because components are connectivity, one spurious entry per
block pair is enough; F1 barely notices (3% of edges) while parallelism
collapses.

**Consequence for "better detection".** Any partition finer than the
thresholded one (a higher threshold, a shrunk covariance, stability
selection) no longer yields the graphical lasso solution at lambda: solving
the pieces independently is then an approximation, or a different estimator.
That is a real trade-off (parallelism vs exactness) and the right framing for
future work, not "detection is broken". The open question becomes: how much
estimation error does a parallelism-preserving partition cost?

Single-instance F_max is close to a coin flip near the percolation threshold
(min–max 26.2–100%), which is why this table is multi-seed.

### 7. RQ2/RQ3: batching wins on CPU; the GPU only wins on one giant block

Re-measured 2026-10-10, commit cf112c1, performance profile, nothing else
running (`results/runlogs/machine_state_rq2_rq3.txt`). Every batched result
matches block_SGL to ≤ 1e-13.

- **Batched CPU** (`batched_cpu.csv`, best-F1 lambda, p = 200–800):
  **1.25–1.69×** over block_SGL. This is the batching idea on its own, with no
  hardware change.
- **Batched GPU at the same configs** (`batched_cpu_gpu.csv`): **0.10–0.79×**,
  i.e. 1.3–10× *slower* than batched CPU. Launch and synchronisation overhead
  per ADMM iteration swamps blocks of this size.
- **Crossover** (`gpu_crossover.csv`, many_tiny, λ scan): at p = 1200 the GPU
  never wins (best 0.95×, at one 1200-node block). At p = 2000 it first wins at
  **s_max ≈ 1950 (1.07×)**. Anchors: 1.00× at s_max 2001 (p = 2400), 1.23× at
  s_max 3006 (p = 3200). Host↔device transfer is ≤ 0.3% of GPU wall time, so
  the loss is compute-side launch overhead, not PCIe.
- **VRAM model** (RQ3): quadratic fit on s_max ≤ 2000 (R² = 1.000) predicts the
  held-out anchors within **+0.1% and +0.2%**. On the 7707 MiB device it hits
  the limit at s_max ≈ 8200, so memory is not the binding constraint in any
  regime measured here.

So the GPU's advantage is not batching: it wins only when one ≳2000-node block
dominates, i.e. when the problem is essentially one large eigendecomposition.
The paper must say this plainly; it inverts the original "batched GPU" pitch.

The pre-fix measurements claimed crossovers at s_max 1035 / 1522 and 1.36× /
1.59× at the anchors. Those were inflated ~30% by a throttled CPU baseline
(bug 9), not by the sampler change.

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

6. **Record the machine state, and check for competing load.** A Docker
   container from another project (2 cores busy, restarts on boot) and an
   unrecorded power profile slowed 8-thread CPU runs by 16% (median per ADMM
   iteration, up to 46%). Single-threaded runs were unaffected. RQ1 speedups
   survived because both CPU solvers slowed together, but every GPU-vs-CPU
   ratio was inflated ~30% (bug 9). `results/runlogs/machine_state*.txt` now
   records profile, governor, load and what was stopped.

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
| 8 | Seeded S still **depended on BLAS thread count**: gglasso's `sample_covariance_matrix` uses `multivariate_normal(method='svd')`, and the SVD of our block-diagonal Sigma (repeated eigenvalues) differs by thread count. Same seed → S differing by 0.1–0.5 per entry at 1t vs 8t | 1t/8t "paired" sweeps were not paired: detected block count matched in only 141/384 configs, and differed in 40 of the 48 best-F1 rows. Separately: `percolation.csv` does not reproduce from any committed code (rerun of 300 rows at 8 threads with HEAD 96f4249: 59/300 block counts match). Its Sep 2 20:18 timestamp, same day as the archived pre-fix sweeps, suggests it predates the bug-3 fix, but git cannot confirm this | Fixed 2026-10-09: Cholesky draw in `blockgen.sample_covariance`; regression test `tests/test_sampling_determinism.py`. **All datasets in `results/` must be regenerated** — Findings 1, 4, 5, 6 numbers are provisional until then |
| 9 | Machine state uncontrolled and unrecorded: competing Docker container (2 cores) and unrecorded power profile during the Sep runs | 8-thread CPU baseline 16% slower per iteration (up to 46%); GPU/CPU ratios inflated ~30%: claimed crossovers at s_max 1035/1522 are really none / ~1950. RQ1 ratios unaffected (0.98 median) | Re-measured 2026-10-10 on a quiet machine, `performance` profile; state recorded in `results/runlogs/machine_state*.txt`. Fig 5 axis label also fixed: it plots T_numpy/T_torch, not T_torch/T_numpy |

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
| `src/provenance.py` | Writes `<out>_env.json`: commit, src-dirty flag, file md5s, versions, args |
| `scripts/rerun_bug8.sh` | Sequential 1t → 8t → pairing check → percolation |
| `tests/test_sampling_determinism.py` | Fails if S depends on BLAS thread count |
| `results/percolation.csv` | 3600 rows, seeds 0/1/2, commit ebd6d6d. Owns the structural claim |
| `results/rq1_primary_1t.csv` | 384 configs, 1 BLAS thread, commit ebd6d6d |
| `results/rq1_primary_8t.csv` | 384 configs, 8 BLAS threads, commit ebd6d6d, paired 384/384 with 1t. **Report these numbers** |
| `results/runlogs/` | Run logs, machine state (performance profile, competing container stopped), pairing check, profiling |
| `results/archive_prefix_bug8/` | Unpaired pre-bug-8 RQ1 pair + irreproducible percolation. Kept deliberately |
| `results/archive_prefix_bug3/` | Pre-bug-3 contaminated pair. Kept deliberately |
| `results/gpu_crossover.csv`, `batched_cpu*.csv`, `gpu_mem_*.csv` | RQ2/RQ3, commit cf112c1, each with `_env.json`. Pre-fix versions in `results/archive_prefix_bug8/` |
| `src/make_graph.py` | All 7 paper figures from `results/`; VRAM limit read from the run's env |
| `ENVIRONMENT.md` | Generated by `scripts/capture_environment.py`: hardware, software, run conditions, per-dataset provenance. Lock + snapshot in `results/environment/` |
| `notes/runbook.md` | How to run everything: thread calibration, thermal/power protocol, git |
| `notes/ProgressLog.md` | **This file — single source of truth.** `findings.md` was retired into it |


---

## Open items

1. ~~Re-run both sweeps~~ **DONE 2026-10-10** (bug 8 re-run, commit ebd6d6d):
   1t 73 min, 8t 46 min, pairing verified 384/384. Earlier pairs kept in
   `results/archive_prefix_bug3/` and `results/archive_prefix_bug8/`.
2. ~~Percolation run~~ **DONE 2026-10-10** (9 min). Finding 6 is quoted from
   it; seed 0 matches the RQ1 8t sweep on 192/192 points.
2a. ~~Figure 1 by regime~~ **DONE**: now `fig1_rq1_threading` — (a) best-F1
   speedup vs p at 1t/8t, (b) per-config retention vs p with `ns` hollow.
2b. ~~RQ2/RQ3 re-run~~ **DONE 2026-10-10** (19 min, commit cf112c1). See
   finding 7: crossover moved from s_max ~1000–1500 to ~1950 (bug 9).
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
> dispatch, not by the eigendecompositions the cubic-work model points at;
> batching the per-block work recovers 1.25-1.7x on CPU, but on a consumer
> GPU launch overhead erases it below ~2000-node blocks, so the GPU only pays
> when one giant block dominates. Finally, the true structure contains
> substantial inter-block parallelism that the graphical lasso estimate itself
> does not have: screening is exact, and a few spurious cross-block entries
> fuse the true blocks at the statistically best lambda. Recovering that
> parallelism requires trading away exactness, which identifies estimation,
> not arithmetic, as the binding constraint.

That is a stronger and more defensible contribution than a speedup number, and
every part of it is measured.