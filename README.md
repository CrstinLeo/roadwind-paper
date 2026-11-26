# roadwind-paper

Paper: *Correcting weather-station wind to the roadside with open geodata: consequences for estimating the air-drag energy of heavy trucks* (target: Journal of Wind Engineering & Industrial Aerodynamics, Elsevier).

## Layout

| Path | Content |
|---|---|
| `main.tex`, `sections/`, `figures/`, `refs.bib` | the manuscript (Elsevier `elsarticle`, numbered references as required by JWEIA). Overleaf: import this repository via GitHub sync; `main.tex` is the main document. |
| `highlights.tex` | highlights as a separate editable file (journal requirement); the items live in `sections/highlights_items.tex` |
| `figures/graphical_abstract.pdf` | graphical abstract (encouraged by the journal), made by `code/scripts/17_graphical_abstract.py` |
| `sections/generated/` | every number and table of the paper, written by `code/scripts/15_paper_numbers.py` — do not edit by hand |
| `elsarticle.cls`, `elsarticle-num-names.bst` | class (v3.5) and bibliography style, so the project also compiles outside Overleaf |
| `code/` | Python package `smhi_trv_wind`: data download, GIS features, correction chain, evaluation, sensitivity analysis, drag model. See `code/README.md`. |
| `code/vehicle/` | NOR-XWIND pre-processing: anemometer calibration, segment bins (`code/vehicle/README.md`) |
| `analysis/` | exploratory and data-preparation scripts (`legacy/`: Swedish rounds 1–7; `norway/`: Frost download, Norwegian features, vehicle segments) |

## Reproducing the numbers

See `code/README.md` for the full run order, environment variables and outputs.

## Data

No data are stored in this repository. All inputs are open (MET Norway Frost API, Kartverket, ESA WorldCover, SMHI open data, Trafikverket Open API, Naturvårdsverket NMD, OpenStreetMap, NOR-XWIND on Zenodo). API keys are read from the environment (`TRV_API_KEY` / `TRV_API_KEY_FILE`, Frost client id) and must never be committed.

## Working with Overleaf

The manuscript lives at the repository root so that Overleaf's GitHub sync works without settings. Pull in Overleaf (Menu → GitHub → Pull GitHub changes to Overleaf) before editing, and push your edits back so that both sides stay in step. Overleaf compiles `main.tex` with pdfLaTeX + BibTeX.

## Before submission

Fill in the red `TODO` items in `main.tex` (authors, affiliation, CRediT, competing interests, AI declaration wording, funding, code repository/DOI).
