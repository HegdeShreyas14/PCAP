# Runbook

How to run individual experiments. To reproduce everything at once, use the
run package instead (`RUN_PACKAGE.md`). Findings and their history are in
`notes/ProgressLog.md`; the paper is `paper/main.tex`.

## Setup

```bash
scripts/setup_env.sh                       # .venv from results/environment/requirements-lock.txt
.venv/bin/python tests/test_sampling_determinism.py   # must pass before any run
```

## Before any timed run

- Mains power, `performance` power profile (`powerprofilesctl set performance`).
- No competing load: close browsers and IDEs, stop unrelated Docker containers
  and VMs. Check with `ps -eo pcpu,args --sort=-pcpu | head` and `docker ps`.
- Pause automatic updates for long runs
  (`sudo systemctl stop unattended-upgrades apt-daily.timer apt-daily-upgrade.timer`):
  a driver update once landed mid-run.
- Never run two timed jobs at once. Do not edit `src/` while a run is going:
  every run records whether `src/` was clean.
- Write a `results/runlogs/machine_state_<batch>.txt` first (the run package
  does this automatically).

## Threads

Always pass `--blas-threads` explicitly; it caps every native pool (NumPy's
OpenBLAS, SciPy's OpenBLAS used by GGLasso's numba kernels, OpenMP) before
NumPy is imported. Calibration on the primary machine (Ryzen 9 8945HX, 16
physical cores): ADMM at p=800 takes 2.20 s at 1 thread, 1.02 s at 4, 1.12 s at
8, 1.95 s at 16. Use 1 (algorithmic baseline) and 8 (threaded baseline); never
exceed the physical core count.

## Experiments

```bash
PY=.venv/bin/python
# RQ1: decomposition speedup, 384 configs; run 1t and 8t with the same seed
$PY src/sweep.py --machine primary --blas-threads 1 --seed 0 --out results/rq1_primary_1t.csv
$PY src/sweep.py --machine primary --blas-threads 8 --seed 0 --out results/rq1_primary_8t.csv
$PY src/analyze.py --compare results/rq1_primary_1t.csv results/rq1_primary_8t.csv   # pairing check + crossover

# structure: percolation (3 seeds) and screening exactness/mechanism
$PY src/percolation.py --seeds 0 1 2 --blas-threads 8
$PY src/screening.py --blas-threads 8

# profiling (1 thread)
$PY src/profile_block.py --regime imbalanced --p 200 --ratio 10 --lam 0.15
$PY src/profile_block.py --regime balanced   --p 800 --ratio 10 --lam 0.10

# RQ2/RQ3: batched solver, GPU crossover (float64 reference, float32 diagnostic), VRAM anchors
$PY src/bench_batched.py --backends numpy torch --out results/batched_cpu_gpu.csv
$PY src/gpu_crossover.py --p 1200 2000 --regime many_tiny
$PY src/gpu_crossover.py --p 1200 2000 --regime many_tiny --dtype float32 --out results/gpu_crossover_fp32.csv
$PY src/fp32_diagnostic.py           # pre-registered float32 acceptance criteria
$PY src/fp32_cpu_baseline.py         # like-for-like float32 CPU vs GPU
$PY src/gpu_crossover.py --p 3200 --regime many_tiny --lam 0.06 --out results/gpu_mem_3200.csv
```

Report speedups at the best-F1 lambda, never the fastest lambda. Report the
8-thread baseline; the 1-thread run is the algorithmic upper bound.

## Analysis and figures

```bash
$PY src/analyze.py results/rq1_primary_8t.csv              # tables for one run
$PY src/make_graph.py --out paper/figs                      # all paper figures
$PY scripts/capture_environment.py                          # ENVIRONMENT.md + results/environment/
$PY scripts/compare_machines.py                             # after unpacking run packages
```

## Provenance rules

- New runs write new files; never overwrite a committed dataset in place.
- Commit each CSV with its `_env.json` and run log in the same commit, from a
  clean `src/`, and tag the commit that produced a dataset.
- Superseded data go to `results/archive_*` with a README saying why.
