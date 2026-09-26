# GEMSDOE9 — fault-discovery laboratory

> **Read this project brief before every session.** Maximize P(Win): choose experiments by evidence and opportunity cost, not by familiarity. Own the Outcome: verify the complete chain from source to GeoTIFF; failures are data. Our target is the DOE GEMS Prize, not a claim that every detected lineament is a geothermal vent.

## Standing user prompt / acceptance criteria

Review the repository and earlier GEMSDOE, GEMSDOE2, GEMSDOE3, GEMSDOE4, 5GEMSDOE, 6GEMSDOE and 8GEMSDOE results. Explain why 0.1563 keeps recurring, verify duplicate files rather than infer duplication from rounded scores, and **do not recycle the same submission**. Research free official geology and geophysics data; record sources, licences, verification and uncertainties. Before implementation, rank **3–5 new geological hypotheses**, naming the exact layer(s), physical signature/transform, why an uncatalogued fault may be detected, and the difference from previously implemented methods. Spatially block the holdout; **do not use a weekly submission slot until the new idea beats the current best on an equivalent, uncontaminated holdout**. Make an obvious single-band float32 `[0,1]` EPSG:32611 GeoTIFF download/generation path, strict template-grid/footprint checks, a unique filename, and a short submission note. Provide an executive-summary subpage with upload steps. Maintain an easy-to-read GitHub Pages site and current source feed, verified links and limitations; automate wherever possible. Compare to the public leaderboard, aiming to exceed 0.3049, but never invent private-test performance. Use the official [competition problem](https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/), [data tab](https://www.drivendata.org/competitions/306/competition-doe-gems/data/), [leaderboard](https://www.drivendata.org/competitions/306/competition-doe-gems/leaderboard/), [rules PDF](https://docs.nlr.gov/docs/fy26osti/96647.pdf), [reference solution](https://github.com/drivendataorg/gems-prize-reference-solution) and [INGENIOUS](https://gdr.openei.org/submissions/1391). No hallucinations: distinguish measured results, informed hypotheses, and unknowns. Make multiple independent review passes, fix findings, and explain blockers without requesting credentials.

**Core values:** “Maximize P(Win)” means weigh evidence, costs and risk; “Own the Outcome” means verify and act end to end. Neither justifies overstating evidence or spending a submission slot on an unvalidated idea.

## Executive decision (2026-09-26)

**NO SUBMISSION APPROVED.** The new basin-edge candidate scored **0.02947** mean distance-weighted Tversky on a four-quadrant, catalogue-only proxy at 3% emission, vs random seeds **0.16678–0.17137** and gravity-edge-only **0.04219**. It does **not** beat our local best; it cannot be compared numerically with the sibling LOFSO 0.0933 because the truth protocol differs. The experimental, format-validated file is for pipeline testing **only** and is not a recommended weekly upload. See [holdout report](reports/holdout.json) and [hypotheses](knowledge/hypotheses.md). The actual private new-fault labels are unavailable; a local proxy cannot prove a 0.3049 public score.

**Why repeated 0.1563?** In this session, the `ens12-adopted-floor0.1-w0/submission.tif` files retrieved independently from GEMSDOE, 5GEMSDOE and GEMSDOE2 have the **same SHA-256 `7f00890a62878d612fb5eef67a9a364a2df819433dde74b6762ce4fc0fc4fe15`**. The GEMSDOE and 5GEMSDOE `docs/submission_field.bin` files likewise both hash to `b966d47c7c02b1c2de0d06553363bc8cccd887c697f760e63f49db9fd99a0351`. This establishes codebase artifact duplication, **not independently which exact bytes were uploaded**: public leaderboard shows account best only, not submission→file associations. Rounded scores of *different* files can also coincide. [Audit](knowledge/audit.md).

## Reproduce locally

The authenticated competition data tab redirects to login here. Instead, the three user-supplied Dropbox mirrors were previously mirrored into the public sibling [5GEMSDOE bridge manifest](https://github.com/buffedlizard55-lab/5GEMSDOE/blob/main/data/bridge/manifest.json). We independently reconstructed all three raster files from the public sibling GitHub blobs in this session and checked SHA-256 against the manifest. These pins verify **mirror consistency, not official competition checksums**. `data/` is gitignored. To repeat:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
bash scripts/download_competition_data.sh     # public sibling bridge; sha256 verified
.venv/bin/python scripts/gems9.py evaluate    # overwrites reports/holdout.json
.venv/bin/python scripts/gems9.py build-experimental
.venv/bin/python scripts/gems9.py validate
.venv/bin/python -m pytest -q
```

`build-experimental` writes `reports/basin-experimental.tif`, validates it against the sample template, but **does not authorize upload**. A genuine release needs matched, contamination-resistant validation against the best available comparator and preferably independently verified new-fault labels. No DrivenData credential is requested or stored, and this repository does not auto-submit.

## Navigate

- [Site / current decision](docs/index.html) · [Executive submission guide](docs/executive_summary.html)
- [Ranked hypotheses & negative result](knowledge/hypotheses.md) · [data audit / official links](knowledge/sources.md) · [duplicate audit](knowledge/audit.md)
- [Holdout report](reports/holdout.json) · [limitations & next work](knowledge/next_steps.md)

**Important irregularities:** Official description names some layers differently from the actual TIFF band tags (e.g. depth to conductive base vs `depth_to_base_surf`; band 6 `tc` appears mislabeled in sibling work). This experiment verifies and uses only bands **13 and 15**. The official template has **NaN outside** the footprint; the `Predicted values must be in range [0,1]` error can indicate non-finite/invalid scored pixels, but its exact backend cause is **not independently confirmed**. Our validator checks every scored pixel; do not silently replace NaNs outside the template. See [submission guide](docs/executive_summary.html).
