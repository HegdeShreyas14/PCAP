# GPU-Accelerated Sparse Graphical Lasso
 
Parallelising GGLasso's block-sparse (connected-component) decomposition on
consumer GPUs, and characterising when it helps.
 
**Team:** Abhyuday Gandhi (240905036), Shreyas S Hegde (240905118),
Shrikrishna Bhakhta (240905214)
 
---

## Reproducing the results

- **Environment:** [ENVIRONMENT.md](ENVIRONMENT.md) — hardware, OS, BLAS, GPU
  driver, run conditions, and which commit produced each dataset. Full package
  lock in `results/environment/requirements-lock.txt`.
- **How to run:** [notes/runbook.md](notes/runbook.md).
- **Findings and their history:** [notes/ProgressLog.md](notes/ProgressLog.md),
  including the bugs that invalidated earlier datasets.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r results/environment/requirements-lock.txt \
    --extra-index-url https://download.pytorch.org/whl/cu130
python tests/test_sampling_determinism.py
scripts/rerun_bug8.sh        # RQ1 sweeps + percolation
scripts/rerun_rq2_rq3.sh     # RQ2/RQ3 (CUDA GPU required)
python src/make_graph.py     # figures
```

