# data/ — how to obtain the competition rasters

**Status: blocked, and we are not going around it.** The DrivenData data tab requires an
account. Verified 2026-09-27: requesting it returns the `/accounts/login/` page.

The blocker is the login, **not** the network. Measured host by host the same day:
`github.com`, `codeload.github.com`, `pypi.org` and `files.pythonhosted.org` return
HTTP 200 and download normally; `drivendata.org`, `gdr.openei.org`, `usgs.gov`,
`sciencebase.gov`, `dropbox.com` and `s3.amazonaws.com` fail with `curl` exit 35
(`SSL_ERROR_SYSCALL`). It is a host allowlist. An earlier revision of this file said no
URL resolved at all; that was wrong. No credentials will be requested, stored, or worked
around.

## What is needed

| File | Required | What it is | Used for |
|---|---|---|---|
| `training_features.tif` | **yes** | Multi-band predictor stack. EPSG:32611, 100 m, 3292 × 3730. | All five detectors |
| `existing_faults.tif` | **yes** | The known USGS/INGENIOUS faults, rasterised. | Training labels, the free submission core, the evaluation mask |
| `example_submission.tif` | no | The organiser's template. | Grid cross-check only |
| `1m_DEM_links.csv` | no | URLs for 1 m DEM tiles. | Optional input to detector G-5 |
| `GEMS_96647.pdf` | no | The rules PDF. | Nothing is attributed to it here. |

Source of truth (requires a DrivenData account):
<https://www.drivendata.org/competitions/306/competition-doe-gems/data/>

The problem description lists the **layer groups** in `training_features.tif`:

- surface conductivity and depth to conductive base surface
- detrended elevation and the slope of detrended elevation
- dilatation rate, shear strain rate, and the second invariant of the strain rate tensor
- isostatic gravity anomaly and the slope of the isostatic gravity anomaly
- magnetics: reduced-to-pole magnetic anomaly, total magnetic intensity, the vertical and
  horizontal slope of total magnetic intensity, and the top-of-crustal magnetic source depth
  estimate
- density of earthquakes

It does **not** give an ordered index. The exact band count and names are read from the
file's own per-band `description` / `data_category` tags — the same source the official
[reference solution](https://github.com/drivendataorg/gems-prize-reference-solution) reads —
by `scripts/prepare_data.py`.

## How to fill it

```bash
git clone https://github.com/buffedlizard55-lab/GEMSDOE9.git && cd GEMSDOE9
python3 -m pip install -r requirements.txt

# sign in at the data tab in a browser, download, and move the files into ./data/
bash scripts/download_competition_data.sh     # reports what is present and what is missing
python scripts/prepare_data.py                # verify + dump the band inventory
```

`prepare_data.py` writes `data/processed/band_inventory.json`, which is what
`src/gems/pipeline.py` reads. It also:

- checks the grid of every raster against EPSG:32611, 100 m, 3292 × 3730
  and transform (100, 0, 243350, 0, −100, 4508550)
- reports the catalogue pixel count and density
- **checks the organiser's template against the problem description**, which says it
  "predicts total fault absence", and flags it loudly if it is not that

## After the data is in place

```bash
python scripts/validate_holdout.py            # spatially-blocked holdout, prints the gate
python scripts/build_submission.py --holdout-gate
```

The second command refuses to write a model submission unless the corridor arm beats the
catalogue-only baseline by ≥ 0.02 **and** the model arm beats a budget-matched random
control by ≥ 0.02, averaged over folds. That is the price of a submission slot.

## Not a data source

The Dropbox links in the project brief are **not** treated as one. They are unreachable from
this environment, and they are not linked from any DrivenData or DOE page, so their
provenance is unestablished. See `scripts/site_data.py::ACCESS`.

## Free, official external data (optional, none of it on the critical path)

| Source | URL | Licence | Blocks anything? |
|---|---|---|---|
| USGS 3DEP 10 m DEM | <https://apps.nationalmap.gov/3dep/> | Public domain | No — G-5 falls back to the competition's own detrended-elevation layers |
| USGS GeoDAWN (ScienceBase) | <https://www.sciencebase.gov/catalog/item/657e1d85d34e23d3533209f7> | Public domain | No — the competition already ships derived GeoDAWN products |
| USGS Quaternary Fault and Fold Database | <https://www.usgs.gov/programs/earthquake-hazards/faults> | Public domain | No — already in the labels |
| Landsat-8/9 TIRS, ASTER L1T | <https://earthexplorer.usgs.gov/>, <https://search.earthdata.nasa.gov/> | Public domain | No — dropped from the candidate set; the previous revision's H9-4 depended on them and is not carried forward |

The competition permits external data provided the participant holds a licence that allows
use in the challenge and sharing with the sponsor for evaluation. All of the above are public
domain. The constraint is engineering time and download volume, not licensing.
