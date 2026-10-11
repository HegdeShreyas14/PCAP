# Cross-platform run package

Reproduce the paper's experiments on your machine and send the results back.
Structural results (detected blocks, objective values, F1) should match the
primary machine exactly; timings will differ, and comparing them across
machines is the point.

## Requirements

- Linux (macOS works for CPU-only runs; on Windows use WSL2)
- Python 3.12 (`python3.12 --version`)
- ~6 GB free disk (~3 GB of it is PyTorch + CUDA if you have an NVIDIA GPU)
- Optional: NVIDIA GPU with a recent driver (CUDA 13). Without one, the GPU
  experiments are skipped automatically and everything else still runs.

## Steps

```bash
git clone <repo-url> && cd <repo>
git checkout fix/thread-invariant-sampler      # until merged into main
scripts/setup_env.sh                           # builds .venv from the lock, ~10-20 min
scripts/run_package.sh --machine <yourname-laptop> --tier smoke   # ~10 min: check setup
```

If the smoke tier ends with `ALL DONE`, start the real run. It takes hours,
so run it overnight:

```bash
scripts/run_package.sh --machine <yourname-laptop> --tier standard   # ~3-5 h
# or --tier full (~8-10 h): adds seeds 1-2, batched benchmarks, VRAM anchors
```

Then send back the two files it prints:
`results/machines/<name>.tar.gz` and `results/machines/<name>.tar.gz.sha256`.

## Rules for valid timings

The script records these, but it cannot enforce them:

1. Plugged in, with the power profile set to performance
   (`powerprofilesctl set performance` on Ubuntu).
2. Close browsers, IDEs, chat apps, games, and any Docker containers or VMs.
   The preflight lists the busiest processes and running containers. An
   unrelated container once inflated our GPU results by about 30%.
3. Do not use the machine until it finishes. Never run two packages at once.
4. Do not edit `src/`. The script refuses to run on modified code, so that
   every result maps to a commit.

## What the tiers run

| tier | experiments |
|---|---|
| smoke | determinism test, quick sweep, tiny GPU run (float64 + float32) |
| standard | RQ1 1-thread and multi-thread sweeps (384 configs each) and pairing check, profiling, percolation (3 seeds), GPU crossover in float64 and float32, float32 diagnostic, float32 CPU baseline |
| full | standard + RQ1 seeds 1 and 2, batched CPU/GPU benchmark, VRAM anchors (p = 2400, 3200), screening exactness |

Multi-threaded runs use min(8, physical cores) BLAS threads.

## On the primary machine (merging results)

```bash
tar -xzf <name>.tar.gz -C results/machines/ && sha256sum -c <name>.tar.gz.sha256
python scripts/compare_machines.py     # writes results/cross_platform.csv
```
