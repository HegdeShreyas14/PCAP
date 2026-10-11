# Superseded — do not use

- `rq1_primary_{1t,8t}*`: the 1- and 8-thread runs used different sampled data
  because sampling depended on the BLAS thread count (bug 8).
- `percolation.csv`: does not reproduce from any committed code.
- `batched_cpu*`, `gpu_crossover*`, `gpu_mem_*`: measured against a CPU
  baseline slowed by competing load (bug 9); GPU/CPU ratios ~30% too high.

See `notes/ProgressLog.md`. No number from here appears in the paper.
