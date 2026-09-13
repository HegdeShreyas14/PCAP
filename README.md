# GPU-Accelerated Sparse Graphical Lasso

This project evaluates connected-component decomposition for the single
graphical lasso and implements batched ADMM backends for NumPy CPUs, PyTorch
CPUs, and CUDA GPUs. Performance is always reported at the regularization
parameter with the best support-F1, preventing over-fragmented but statistically
poor solutions from being presented as acceleration.

**Team:** Abhyuday Gandhi (240905036), Shreyas S Hegde (240905118),
Shrikrishna Bhakhta (240905214)

## What is implemented

- Reproducible block-sparse instances in four structural regimes.
- Paired full-ADMM versus GGLasso `block_SGL` CPU sweeps.
- Multi-seed percolation and oracle-structure analysis.
- Profiling of eigendecomposition, detection, assembly, and glue.
- Batched ADMM using NumPy, PyTorch CPU, or CUDA without coupling blocks.
- Numerical-equivalence tests against GGLasso 0.3.0.
- Four-way Phase-3 benchmarks, best-F1 summary CSVs, and a figure.

## Final result

The primary-machine experiment set is complete: both corrected RQ1 sweeps
(384 rows each), the multi-seed percolation study (3,600 rows), and the full
Phase-3 comparison (144 rows, 24 best-F1 operating points). All six tests pass.

At best-F1 operating points, median speedup versus full 8-thread ADMM is 3.91x
for `block_SGL`, 4.32x for batched NumPy, 2.34x for batched PyTorch CPU, and
0.50x for batched CUDA. CUDA wins at 8/24 points—all at p=400—and peaks at
3.31x. The worst float64 difference from `block_SGL` is 6.46e-14, far below
the 1e-6 acceptance threshold. See [notes/findings.md](notes/findings.md) for
the interpretation and [results/phase3_speedup.png](results/phase3_speedup.png)
for the final figure.

For a complete explanation of the methodology, every final result, limitations,
safe paper claims, and the publication-readiness assessment, see
[notes/RESULTS_AND_PUBLISHABILITY.md](notes/RESULTS_AND_PUBLISHABILITY.md).

## Setup (Windows PowerShell)

Python 3.14 is known to work with the pinned package versions.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Use the PyTorch index matching the installed NVIDIA driver if CUDA 12.8 is not
appropriate. A CPU-only PyTorch installation can run every command with
`--no-gpu` except the CUDA-specific test.

Capture the machine before collecting results:

```powershell
.\.venv\Scripts\python.exe src\phase1_repro.py
```

## Fast verification

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe src\sweep.py --quick --out results\rq1_smoke.csv --machine primary --blas-threads 8
.\.venv\Scripts\python.exe src\percolation.py --regimes balanced --ps 100 --ratios 10 --seeds 0 --lam-steps 4 --out results\percolation_smoke.csv
.\.venv\Scripts\python.exe src\benchmark_phase3.py --quick --out results\phase3_smoke.csv --machine primary --blas-threads 8
.\.venv\Scripts\python.exe src\analyze_phase3.py results\phase3_smoke.csv
```

To re-run the complete final sequence (hours, resumable):

```powershell
.\run_final_experiments.ps1
```

Each long command supports `--resume` and stores an adjacent temporary JSONL
checkpoint until its final CSV is written.

See [notes/RUNBOOK.md](notes/RUNBOOK.md) for final experiment commands,
measurement protocol, expected artifacts, and interpretation rules.

## Source map

| File | Purpose |
|---|---|
| `src/blockgen.py` | Reproducible arbitrary block-size precision matrices |
| `src/sweep.py` | Corrected RQ1 CPU benchmark |
| `src/analyze.py` | RQ1 tables and 1-thread/8-thread crossover |
| `src/percolation.py` | Multi-seed structural threshold and oracle analysis |
| `src/profile_block.py` | Runtime attribution |
| `src/batched_sgl.py` | Batched NumPy, PyTorch CPU, and CUDA solver |
| `src/benchmark_phase3.py` | Paired four-way Phase-3 benchmark |
| `src/analyze_phase3.py` | Best-F1 summary and speedup figure |
| `src/phase1_repro.py` | Machine/environment fingerprint |
| `tests/test_batched_sgl.py` | Generator and solver correctness tests |

`results/`, `.venv/`, and `.matplotlib/` are regenerable and ignored by Git.

## Methodological constraints

- Pin BLAS threads before NumPy is imported.
- Compare solvers on the same generated instance and seed.
- Keep diagnostics and console output outside timed regions.
- Use at least nine trials for paper numbers.
- Select lambda by support-F1, never by runtime.
- Treat GPU results as end-to-end timings, including device transfer and result
  assembly. Tiny problems are expected to be slower on a GPU.
- At `p=100`, `balanced` and `many_tiny` both resolve to `[20] * 5`; do not use
  that dimension to compare those two regimes.
