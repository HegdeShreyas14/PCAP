# Paper

Two versions sharing `refs.bib` and `figs/`:

- `main.tex`: IEEE conference format (`IEEEtran`, built into Overleaf and TeX Live).
- `main_springer.tex`: Springer Nature journal format (`sn-jnl`). The `sn-jnl` class is not
  included: download Springer Nature's LaTeX template package and place `sn-jnl.cls`
  (and its `.bst` files) next to it, or compile on Overleaf from that template.

## Build

1. Regenerate figures from `results/`: `python src/make_graph.py --out paper/figs --figs 1 2 3 4 5`
2. `paper/build.sh` (TeX Live or TinyTeX) writes `paper/main.pdf`; or upload `paper/` to Overleaf.

## Before submission

- Search `main.tex` for `TODO` (author affiliations, repository URL).
- Verify every entry in `refs.bib` and merge the literature-survey references.
- Switch the document class if the venue requires Springer LNCS or another template.
