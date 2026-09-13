# Primary machine environment

Captured on 2026-09-13 with `src/phase1_repro.py`.

| Item | Value |
|---|---|
| OS | Windows 11 (10.0.26200) |
| CPU | AMD64 Family 25 Model 124, 12 logical cores |
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU, 6141 MiB |
| NVIDIA driver | 573.05 |
| Python | 3.14.2 |
| NumPy | 2.4.4 |
| SciPy | 1.17.1 |
| GGLasso | 0.3.0 |
| PyTorch | 2.11.0+cu128 |
| CUDA available | yes, runtime 12.8 |

The machine-readable record is regenerated at `results/environment.json`.
Each benchmark also records its own BLAS thread count because thread variables
must be set before importing NumPy.
