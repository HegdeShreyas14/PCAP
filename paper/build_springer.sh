#!/usr/bin/env bash
# Build paper/main_springer.pdf with the official Springer Nature template
# (v3.1, December 2024), downloaded into a temporary directory each time.
set -euo pipefail
cd "$(dirname "$0")"
[ -d ~/.TinyTeX/bin/x86_64-linux ] && export PATH=~/.TinyTeX/bin/x86_64-linux:$PATH
command -v pdflatex >/dev/null || { echo "pdflatex not found" >&2; exit 1; }
URL=https://resource-cms.springernature.com/springer-cms/rest/v1/content/18782940/data/v12
B=$(mktemp -d); trap 'rm -rf "$B"' EXIT
curl -sSfL -m 120 -o "$B/sn.zip" "$URL"
unzip -q -j "$B/sn.zip" 'sn-article-template/sn-jnl.cls' 'sn-article-template/bst/*.bst' -d "$B"
cp -r main_springer.tex refs.bib figs "$B"/
( cd "$B" && pdflatex -interaction=nonstopmode -halt-on-error main_springer.tex >/dev/null && bibtex main_springer >/dev/null \
  && pdflatex -interaction=nonstopmode main_springer.tex >/dev/null && pdflatex -interaction=nonstopmode main_springer.tex >/dev/null ) \
  || { echo "build failed; see $B/main_springer.log (TinyTeX may need: tlmgr install sttools threeparttable appendix wrapfig)" >&2; trap - EXIT; exit 1; }
grep -E "undefined|Overfull" "$B/main_springer.log" || true
cp "$B/main_springer.pdf" main_springer.pdf
echo "wrote paper/main_springer.pdf ($(grep -o '[0-9]* pages' "$B/main_springer.log" | tail -1))"
