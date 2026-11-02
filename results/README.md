# Results

Every dataset here was produced by committed code; its `<name>_env.json`
records the commit, a dirty-tree flag (always false for these), the md5 of
every source file, package versions, BLAS threads, the full command line and a
system snapshot. Some env files record a commit hash from before a one-time
message-only history rewrite; `environment/commit_map.tsv` maps them.
`ENVIRONMENT.md` at the repository root summarises all of it.

## Datasets used in the paper

| file | what | produced by | paper |
|---|---|---|---|
| `rq1_primary_1t.csv`, `rq1_primary_8t.csv` | RQ1 sweep, 384 configs, seed 0, 1 and 8 BLAS threads; paired (identical instances) | `src/sweep.py` | §V-A–C, Fig. 1–2, Table I |
| `seeds/rq1_primary_{1t,8t}_seed{1,2}.csv` | same sweep, seeds 1 and 2 | `src/sweep.py --seed` | §V-A (spread across instances) |
| `percolation.csv` | structure + F1 over λ = 0.02…0.50, 3 seeds, 3600 rows; also resolves the RQ1 grid-edge cells | `src/percolation.py` | §V-A, §V-D, Fig. 3–4 |
| `screening_exactness.csv` | full unscreened solve vs. thresholded-S components (36 cells) | `src/screening.py` | §V-D |
| `screening_cross_block.csv` | above-threshold entries crossing true blocks (72 cells) | `src/screening.py` | §V-D |
| `batched_cpu.csv`, `batched_cpu_gpu.csv` | batched solver vs. `block_SGL` at RQ1 best-F1 λ | `src/bench_batched.py` | §V-F |
| `gpu_crossover.csv` | GPU (float64) vs. batched CPU over a λ scan, p = 1200/2000 | `src/gpu_crossover.py` | §V-F, Fig. 6 |
| `gpu_crossover_fp32.csv` | same, GPU in float32 | `src/gpu_crossover.py --dtype float32` | §V-F, Fig. 6 |
| `fp32_diagnostic.csv` | float32 vs. float64 at 4 points against pre-registered criteria | `src/fp32_diagnostic.py` | §V-F |
| `fp32_cpu_baseline.csv` | like-for-like float32 CPU vs. float32 GPU | `src/fp32_cpu_baseline.py` | §V-F, Table IV |
| `gpu_mem_{2400,3200}[_fp32].csv` | held-out VRAM anchors (p = 2400, 3200) | `src/gpu_crossover.py --lam` | §V-G, Fig. 6 |
| `riboflavin_curve.csv`, `riboflavin_independence.csv`, `riboflavin_exactness.csv` | riboflavin structure vs. permutation null, subsample stability, sample independence, screening exactness; no timings | `src/riboflavin_study.py` | §V-E, Fig. 5, Table III |
| `riboflavin_sensitivity/` | same, by batch: whole-batch subsamples, 24 h only, 48 h only, batch means removed | `scripts/provenance/run_riboflavin.sh` | §V-E, Fig. 5 |
| `riboflavin_stars.csv` | StARS at p = 200, 500, random and whole-batch subsamples; no timings | `src/riboflavin_stars.py` | §V-E, Table III |

Column naming: in `gpu_crossover.csv` and `gpu_mem_{2400,3200}.csv` the GPU
speedup column is called `ratio_torch_over_numpy` but holds
**t_numpy / t_torch** (>1 = GPU faster). Newer files call it `gpu_speedup`.
The values are not inverted; only the old name is misleading.

## Supporting material

| path | what |
|---|---|
| `runlogs/` | console log of every run, plus `machine_state*.txt` recorded before each batch (power profile, governor, load, stopped processes) |
| `environment/` | `system.json` (hardware/software snapshot), `requirements-lock.txt` (exact packages), `commit_map.tsv` |

## Superseded data — do not use

| path | why it is kept |
|---|---|
| `archive_prefix_bug3/` | RQ1 pair from before the sample covariance was seeded (bug 3) |
| `archive_prefix_bug8/` | RQ1 pair whose 1- and 8-thread runs used different data (bug 8), a percolation file that does not reproduce from any commit, and RQ2/RQ3 data measured on a throttled CPU baseline (bug 9) |

They document problems that were found and fixed (see `notes/ProgressLog.md`).
None of their numbers appear in the paper.
