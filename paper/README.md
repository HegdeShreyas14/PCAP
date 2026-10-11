# Paper

IEEE conference format (`IEEEtran`, built into Overleaf and TeX Live).

## Build

1. Regenerate figures from `results/`: `python src/make_graph.py --out paper/figs --figs 1 2 3 4 5`
2. `paper/build.sh` (TeX Live or TinyTeX) writes `paper/main.pdf`; or upload `paper/` to Overleaf.

## Before submission

- Search `main.tex` for `TODO` (author affiliations, repository URL).
- Verify every entry in `refs.bib` and merge the literature-survey references.
- Switch the document class if the venue requires Springer LNCS or another template.
