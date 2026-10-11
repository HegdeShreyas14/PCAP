#!/usr/bin/env bash
# Build paper/main.pdf (pdflatex + bibtex) in a temporary directory, so no
# aux files land in the repository. Uses TinyTeX if installed in ~/.TinyTeX.
set -euo pipefail
cd "$(dirname "$0")"
[ -d ~/.TinyTeX/bin/x86_64-linux ] && export PATH=~/.TinyTeX/bin/x86_64-linux:$PATH
command -v pdflatex >/dev/null || { echo "pdflatex not found (install TeX Live or TinyTeX, or use Overleaf)" >&2; exit 1; }
B=$(mktemp -d); trap 'rm -rf "$B"' EXIT
cp -r main.tex refs.bib figs "$B"/
( cd "$B" && pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null && bibtex main >/dev/null \
  && pdflatex -interaction=nonstopmode main.tex >/dev/null && pdflatex -interaction=nonstopmode main.tex >/dev/null ) \
  || { echo "build failed; see $B/main.log" >&2; trap - EXIT; exit 1; }
grep -E "undefined|Overfull" "$B/main.log" || true
cp "$B/main.pdf" main.pdf
echo "wrote paper/main.pdf ($(grep -o '[0-9]* pages' "$B/main.log" | tail -1))"
