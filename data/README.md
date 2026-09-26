# Data — How to Obtain

**Blocker:** No DrivenData auth in sandbox → cannot auto-download training_features.tif, labels.tif, sample_submission.tif, 1m_DEM_links.csv from https://www.drivendata.org/competitions/306/competition-doe-gems/data/ (verified redirect to login).

**Workaround:** Run `bash scripts/download_competition_data.sh` on any unrestricted machine with DrivenData account, then `python scripts/prepare_data.py`.

## Official Files (from data tab, login required)

| File | Size | SHA256 prefix | Description | Source |
|---|---|---|---|---|
| `training_features.tif` | 399.5 MB | `4371c82e3b8339b8…` | 19 bands, EPSG:32611, 100 m, 3292×3730 | https://www.drivendata.org/competitions/306/competition-doe-gems/data/ |
| `existing_faults.tif` | 415.8 KB | `7ba308ccdc4418b3…` | Fault labels raster, 60,988 px | same |
| `example_submission.tif` | 1.5 MB | `2176d08e485aa2cd…` | Template, same as labels (irregularity flagged) | same |
| `1m_DEM_links.csv` | ~50 KB | — | 716 URLs for 1 m DEM tiles | same |
| `GEMS_96647.pdf` | 444.5 KB | `50d854b1e0239fe6…` | Official rules | https://docs.nlr.gov/docs/fy26osti/96647.pdf |

## Dropbox Mirrors (from prompt, may be transient)

- https://www.dropbox.com/scl/fi/aemhtutjgcp6tr3tint94/GEMS_96647.pdf?rlkey=rek210cj2smnmzb8n0sla1vmd&st=wz4kofki&dl=0
- https://www.dropbox.com/scl/fi/6rgvnuady818ol8yqgis4/example_submission.tif?rlkey=kbykilvau066xuogoosbf4cq8&st=8junzdyw&dl=0
- https://www.dropbox.com/scl/fi/t7fyt03qdh9egyme0itwo/existing_faults.tif?rlkey=yiao96uluqdkipf0h5vju71jf&st=rnino7ya&dl=0
- https://www.dropbox.com/scl/fi/3vz9o0wwavi26xaeoxlwr/gems-geodawn-numerical-features.tif?rlkey=je8d8fepqfbst9lnwsq9rkplu&st=zj1lag1r&dl=0
- https://www.dropbox.com/scl/fi/ig0mban712ns1atphgphe/Digital-elevation-model-links-JSON.pdf?rlkey=zm77f1vbtt2if8hlruymptnu3&st=srhhir10&dl=0

## External Free Official Sources for New Hypotheses

| Source | URL | What we use | Coverage | Verified |
|---|---|---|---|---|
| USGS GeoDAWN magnetic & radiometric | https://doi.org/10.5066/P93LGLVQ | 7 radiometric bands K, Th, U, TC, U/K, Th/K, U/Th | 99.975% of footprint | Yes, via 8GEMSDOE |
| USGS 3DEP 10 m DEM (1/3 arcsec) | https://apps.nationalmap.gov/3dep/ | 9 topo/scarp bands, SL, ksn | 100% | Yes |
| USGS 3DEP 1 m DEM tiles | https://prd-tnm.s3.amazonaws.com/?list-type=2&prefix=StagedProducts/Elevation/1m/ | High-res scarp, 716 tiles ~130 GB | 100% but heavy | One tile verified live 185,344,605 B |
| Landsat-8/9 TIRS | https://earthexplorer.usgs.gov/ | Thermal anomaly Band 10-11 | 100% | Official |
| ASTER L1T | https://search.earthdata.nasa.gov/ | Clay alteration kaolinite, alunite | 100% | Official |
| INGENIOUS Great Basin | https://doi.org/10.15121/1881483 | Training labels | — | Official |
| USGS Quaternary Faults | https://www.usgs.gov/programs/earthquake-hazards/faults | Training labels | — | Official |
| GDR | https://gdr.openei.org/submissions/1391 | Competition data archive | — | Official |

## After Download

```
bash scripts/download_competition_data.sh
python scripts/prepare_data.py
# → data/processed/footprint_mask.npz with 5,167,373 px footprint
```
