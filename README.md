# roadwind-paper

Paper: *From weather stations to the roadside: explaining and correcting the change in near-surface wind with height, roughness, site and forest exposure, for vehicle air-drag estimation* (target: Journal of Wind Engineering & Industrial Aerodynamics).

## Layout

| Path | Content |
|---|---|
| `main.tex`, `sections/`, `figures/`, `refs.bib` | the manuscript (Elsevier `elsarticle`, author–year). Overleaf: import this repository via GitHub sync; `main.tex` is the main document. |
| `elsarticle.cls`, `elsarticle-harv.bst` | class and bibliography style, so the project also compiles outside Overleaf |
| `code/` | Python package `smhi_trv_wind`: data download, GIS features, correction chain, evaluation, sensitivity analysis, map. See `code/README.md` (Chinese). |
| `analysis/legacy/` | exploratory scripts of rounds 1–7, kept for provenance; superseded by `code/` |

Orange numbers in the manuscript come from the 12-day pilot (2026-11-13 to 11-25) and will be replaced when the longer record is available.

## Data

No data are stored in this repository. All inputs are open (SMHI open data, Trafikverket Open API, Naturvårdsverket NMD, OpenStreetMap) and are downloaded by `code/scripts/01–03`. The Trafikverket API key is read from the environment (`TRV_API_KEY` or `TRV_API_KEY_FILE`) and must never be committed.

## Working with Overleaf

The manuscript lives at the repository root so that Overleaf's GitHub sync works without settings. Pull in Overleaf (Menu → GitHub → Pull GitHub changes to Overleaf) before editing, and push your edits back so that both sides stay in step.
