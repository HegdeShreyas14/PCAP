# Experiment runbook

Run every command from the repository root with the laptop connected to power.
Close compute-heavy applications, allow the machine to cool before each timed
run, and do not run two benchmark commands concurrently.

## 1. Environment and correctness gate

```powershell
.\.venv\Scripts\python.exe src\phase1_repro.py --out results\environment.json
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Do not collect performance data unless all tests pass. A missing CUDA device may
skip the CUDA test on a CPU-only machine; any other skip or failure needs review.

## 2. Corrected RQ1 CPU sweeps

```powershell
.\.venv\Scripts\python.exe src\sweep.py --out results\rq1_primary_1t.csv --machine primary --seed 0 --trials 9 --blas-threads 1 --resume
.\.venv\Scripts\python.exe src\sweep.py --out results\rq1_primary_8t.csv --machine primary --seed 0 --trials 9 --blas-threads 8 --resume
.\.venv\Scripts\python.exe src\analyze.py --compare results\rq1_primary_1t.csv results\rq1_primary_8t.csv
```

The two runs must use the same seed and grid. Check that generator-style counts
match before treating the comparison as paired.

## 3. Percolation analysis

```powershell
.\.venv\Scripts\python.exe src\percolation.py --seeds 0 1 2 --out results\percolation.csv --blas-threads 8 --resume
```

Do not use `--no-f1` for final results; it cannot determine whether structural
headroom exists at a statistically useful lambda.

## 4. Phase-3 benchmark and artifacts

```powershell
.\.venv\Scripts\python.exe src\benchmark_phase3.py --out results\phase3_results.csv --machine primary --seed 0 --trials 9 --blas-threads 8 --resume
.\.venv\Scripts\python.exe src\analyze_phase3.py results\phase3_results.csv --out-dir results
```

Expected final artifacts:

- `results/rq1_primary_1t.csv` and matching environment JSON
- `results/rq1_primary_8t.csv` and matching environment JSON
- `results/percolation.csv`
- `results/phase3_results.csv`
- `results/phase3_best_f1.csv`
- `results/phase3_speedup.png`

## 5. Acceptance checks

- Unit tests pass on NumPy, PyTorch CPU, and CUDA when available.
- `*_max_abs_vs_block` is below `1e-6` for float64 batched solvers.
- Objective relative gaps against `block_SGL` are below `1e-8` for float64.
- No final performance claim uses a lambda selected for speed.
- Trial vectors contain nine timings and comparison bands are reported.
- CUDA availability, GPU model, package versions, seed, and BLAS threads are
  present in the artifacts.

Large CUDA speedups are not an acceptance condition. The valid finding may be
that transfer and launch overhead dominate consumer-GPU execution for the block
sizes detected at the best-F1 operating point.

All long-running commands use an adjacent `.partial.jsonl` checkpoint when
`--resume` is supplied. Re-run the same command after an interruption; the
checkpoint is removed automatically only after the final CSV is written.

## Recorded final validation (primary machine, 2026-09-13)

- RQ1 CSVs: 384 rows each at 1 and 8 BLAS threads.
- Percolation CSV: 3,600 rows.
- Phase-3 CSV: 144 rows; best-F1 summary: 24 rows.
- Six unit tests passed, including the CUDA equivalence test.
- Worst maximum absolute difference from `block_SGL`: 6.46e-14.
- Worst objective relative gap from `block_SGL`: 7.36e-16.
- Every final configuration stores nine timing values per solver.

For a quick user-facing check after setup, run only the environment/correctness
gate and the four smoke commands in the README. A complete reproduction uses
`run_final_experiments.ps1` and will take substantially longer.
