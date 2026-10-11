# Provenance scripts

The exact batch scripts that produced the committed datasets in `results/`,
kept so every dataset can be traced to the command that made it. They are
specific to the primary machine (for example, they stop the author's unrelated
Docker container for the duration of a run) and are **not** the way to
reproduce the work elsewhere: use `scripts/run_package.sh` (see
`RUN_PACKAGE.md`).

| script | produced |
|---|---|
| `rerun_bug8.sh` | `rq1_primary_{1t,8t}.csv`, `percolation.csv` (after the bug-8 sampler fix) |
| `rerun_rq2_rq3.sh` | `batched_cpu*.csv`, `gpu_crossover.csv`, `gpu_mem_{2400,3200}.csv` |
| `run_fp32_crossover.sh` | `gpu_crossover_fp32.csv`, `gpu_mem_*_fp32.csv` |
| `run_fp32_cpu_baseline.sh` | `fp32_cpu_baseline.csv` |
| `run_seeds.sh` | `seeds/rq1_primary_{1t,8t}_seed{1,2}.csv` (its GPU steps were aborted; see below) |
| `edge_cells.py` | helper: cells whose best-F1 lambda lies at the grid edge |

`run_seeds.sh`'s GPU steps were stopped because an automatic update replaced
the NVIDIA user-space driver while the old kernel module was still loaded.
They were not rerun: the GPU results are reported for seed 0 only, all on one
driver (595.91). The RQ1 grid-edge question was answered from the finer
lambda grid in `percolation.csv` (same instances) instead of new timing runs.

The scripts `cd` to the repository root, so run them from anywhere.
