# Where Does the Block Parallelism Go?

Characterising connected-component decomposition of the graphical lasso
(GGLasso) on multicore CPUs and a consumer GPU: how much block parallelism is
usable at statistically useful regularisation, why little of it survives, and
when batched GPU execution helps.

**Team:** Abhyuday Gandhi (240905036), Shreyas S Hegde (240905118),
Shrikrishna Bhakhta (240905214)

## Repository map

| path | contents |
|---|---|
| `paper/` | manuscript (`main.tex`, IEEE format), references, figures |
| `src/` | experiment harnesses and the batched solver |
| `scripts/` | environment setup, the cross-platform run package, analysis helpers |
| `tests/` | regression test: sampled data must not depend on BLAS thread count |
| `results/` | every dataset with its provenance; see [results/README.md](results/README.md) |
| `ENVIRONMENT.md` | hardware, software, run conditions, and the commit behind each dataset |
| `RUN_PACKAGE.md` | how to reproduce everything on another machine |
| `notes/` | runbook and the project's chronological research log |

## Reproducing

```bash
scripts/setup_env.sh                                       # venv from the package lock
scripts/run_package.sh --machine <name> --tier smoke       # ~10 min check
scripts/run_package.sh --machine <name> --tier standard    # main experiments, ~3-5 h
.venv/bin/python src/make_graph.py --out paper/figs        # figures from results/
```

Details, tiers and the rules for valid timings: [RUN_PACKAGE.md](RUN_PACKAGE.md).
Individual experiments: [notes/runbook.md](notes/runbook.md).
