# roadwind-paper

Paper: *What the wind costs a truck: correcting weather-station wind to the roadside with open geodata for heavy-vehicle air-drag energy* (target: Journal of Wind Engineering & Industrial Aerodynamics, Elsevier).

## Layout

| Path | Content |
|---|---|
| `main.tex`, `sections/`, `figures/`, `refs.bib` | the manuscript (Elsevier `elsarticle`, numbered references as required by JWEIA). Overleaf: import this repository via GitHub sync; `main.tex` is the main document. |
| `highlights.tex` | highlights as a separate editable file (journal requirement); the items live in `sections/highlights_items.tex` |
| `figures/graphical_abstract.pdf` | graphical abstract (encouraged by the journal), made by `code/scripts/17_graphical_abstract.py` |
| `sections/generated/` | every number and table of the paper, written by `code/scripts/15_paper_numbers.py` — do not edit by hand |
| `elsarticle.cls`, `elsarticle-num-names.bst` | class (v3.5) and bibliography style, so the project also compiles outside Overleaf |
| `code/` | Python package `smhi_trv_wind`: data download, GIS features, correction chain, evaluation, sensitivity analysis, drag model. See `code/README.md` (Chinese). |
| `analysis/` | exploratory and data-preparation scripts (`legacy/`: Swedish rounds 1–7; `norway/`: Frost download, Norwegian features, vehicle segments) |

## Reproducing the numbers

Norway (`SWW_DATA` = Norwegian data root, `SWW_OUT` = output folder):
`11_norway.py` → `21_norway_cv_final.py` (final CV with the corrected nearest-station baseline) → `12_vehicle_norway.py` → `16_norway_ladder_terrain.py` → `18_mechanism_data.py` (`SWW_COUNTRY=NO`) → `20_direction_variant.py NO` → `13_norway_figures.py`.

Sweden (`SWW_DATA` = Swedish data root): `04_run_model.py` (pipeline, map), `09_paper_figures.py` (pipeline result `result.pkl` and Swedish figures), `14_final_cv.py` (`SWW_COUNTRY=SE`), `18_mechanism_data.py` (`SWW_COUNTRY=SE`), `20_direction_variant.py SE`, scans `06`–`08`.

Both: `22_energy.py` (air-drag energy, VECTO crosswind correction and Askerdal Model 1) → `15_paper_numbers.py` (`SWW_NO_OUT`, `SWW_SE_OUT`) → `sections/generated/`, then `19_figures_v2.py` (concept, mechanisms, error structure, time series, site factors, vehicle) and `17_graphical_abstract.py`.

## Data

No data are stored in this repository. All inputs are open (MET Norway Frost API, Kartverket, ESA WorldCover, SMHI open data, Trafikverket Open API, Naturvårdsverket NMD, OpenStreetMap, NOR-XWIND on Zenodo). API keys are read from the environment (`TRV_API_KEY` / `TRV_API_KEY_FILE`, Frost client id) and must never be committed.

## Working with Overleaf

The manuscript lives at the repository root so that Overleaf's GitHub sync works without settings. Pull in Overleaf (Menu → GitHub → Pull GitHub changes to Overleaf) before editing, and push your edits back so that both sides stay in step. Overleaf compiles `main.tex` with pdfLaTeX + BibTeX.

## Before submission

Fill in the red `TODO` items in `main.tex` (authors, affiliation, CRediT, competing interests, AI declaration wording, funding, code repository/DOI).
