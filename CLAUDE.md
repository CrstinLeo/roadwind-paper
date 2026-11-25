# Notes for working on this repository

- Language: discuss with the author in Chinese; the manuscript is in English.
- Never commit or print credentials (Trafikverket API key, Lantmäteriet/Geotorget login). Data stay out of git (see .gitignore).
- Standing rule of the author: use only measured or officially published data with a clear source; no lookup-table estimates presented as data.
- Numbers in the manuscript must come from code in `code/` and be re-checked after every rerun; pilot numbers are wrapped in `\prelim{}`.
- The author pulls the manuscript into Overleaf from this repository; keep `main.tex` at the root and do not rename section files without updating `main.tex`.
- Commit small, descriptive changes; do not rewrite history on `main`.
