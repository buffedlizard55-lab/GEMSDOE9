#!/usr/bin/env python3
"""
prepare_data.py — verify the competition rasters once they are on disk, and
resolve the one thing this repo previously guessed: the band inventory.

Why this exists
---------------
The problem description lists the *layer groups* in training_features.tif but
not their order, and the previous revision of this repo hard-coded indices 0..18
and labelled them "PROVISIONAL". A provisional index that is wrong is a silent
wrong answer. So this script:

  1. checks every file against the shape the pipeline expects
  2. prints the per-band `description` / `data_category` tags that the GeoTIFF
     itself carries -- the same source the official reference solution reads
     (https://github.com/drivendataorg/gems-prize-reference-solution)
  3. writes data/processed/band_inventory.json, which is what pipeline.py reads
  4. cross-checks the grid against the organiser's example_submission.tif
  5. reports the catalogue statistics the metric algebra needs

It never modifies the source rasters.

Run:  python scripts/prepare_data.py [--data-dir data]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

EXPECTED_SHAPE = (3730, 3292)
EXPECTED_EPSG = 32611
EXPECTED_RES = (100.0, 100.0)
EXPECTED_TRANSFORM = (100.0, 0.0, 243350.0, 0.0, -100.0, 4508550.0)

NEEDED = {
    "training_features.tif": "multi-band predictor stack (required)",
    "existing_faults.tif": "known USGS/INGENIOUS faults: training labels, the free "
                           "submission core, and the evaluation mask (required)",
    "example_submission.tif": "organiser template: grid cross-check (optional)",
    "1m_DEM_links.csv": "URLs for 1 m DEM tiles; optional input to detector G-5",
}


def sha256_file(p: Path, limit: int | None = None) -> str:
    h = hashlib.sha256()
    n = 0
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
            n += len(chunk)
            if limit and n >= limit:
                break
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    args = ap.parse_args()
    d = Path(args.data_dir)

    print("=== prepare_data ===")
    print(f"data dir: {d.resolve()}")
    print("source:   https://www.drivendata.org/competitions/306/"
          "competition-doe-gems/data/  (requires a DrivenData account)\n")

    missing = []
    for name, why in NEEDED.items():
        p = d / name
        if p.exists():
            print(f"  [ok]      {name:26} {p.stat().st_size:>14,} B   {why}")
        else:
            tag = "required" if "required" in why else "optional"
            print(f"  [{tag:8}] {name:26} {'missing':>14}   {why}")
            if tag == "required":
                missing.append(name)
    if missing:
        print(f"\nMISSING REQUIRED FILES: {', '.join(missing)}")
        print("Download them with a DrivenData account and re-run this script.")
        return 1

    import rasterio

    # ---- 1. features ------------------------------------------------------
    print("\n--- training_features.tif ---")
    with rasterio.open(d / "training_features.tif") as src:
        count, h, w = src.count, src.height, src.width
        dtype = src.dtypes[0]
        crs = src.crs
        res = src.res
        tr = (src.transform.a, src.transform.b, src.transform.c,
              src.transform.d, src.transform.e, src.transform.f)
        tags = {i: dict(src.tags(i)) for i in range(1, count + 1)}
        desc = src.descriptions
        print(f"  bands   : {count}")
        print(f"  shape   : {h} x {w}   {'OK' if (h, w) == EXPECTED_SHAPE else 'MISMATCH'}")
        print(f"  dtype   : {dtype}")
        print(f"  crs     : {crs}   {'OK' if crs and crs.to_epsg() == EXPECTED_EPSG else 'CHECK'}")
        print(f"  res     : {res}   {'OK' if tuple(res) == EXPECTED_RES else 'CHECK'}")
        print(f"  transform: {tr}")
        print("            " + ("OK" if all(abs(a - b) < 1e-6
                                           for a, b in zip(tr, EXPECTED_TRANSFORM))
                                          else "MISMATCH vs " + str(EXPECTED_TRANSFORM)))
        print(f"  size    : {(d / 'training_features.tif').stat().st_size:,} bytes")
        print(f"  sha256  : {sha256_file(d / 'training_features.tif', 1 << 24)}... "
              f"(first 16 MiB; full-file hash is slow on 400 MB over a network mount)")

        print("\n  BAND INVENTORY (authoritative: read from the file's own tags)")
        print(f"  {'#':>3}  {'description tag':<44} {'data_category':<28} dtype")
        inventory = []
        for i in range(1, count + 1):
            t = tags.get(i, {})
            name = (t.get("description") or "").strip()
            cat = (t.get("data_category") or "").strip()
            print(f"  {i:>3}  {name[:44]:<44} {cat[:28]:<28} {dtype}")
            inventory.append({"index": i - 1, "description": name,
                              "data_category": cat, "tags": t})
        unnamed = [b["index"] for b in inventory if not b["description"]]
        if unnamed:
            print(f"\n  NOTE: bands without a description tag: {unnamed} "
                  f"(zero-indexed).")
            print("        pipeline.CompetitionData falls back to positional names for these")
            print("        and raises a loud KeyError on any lookup that misses. Confirm")
            print("        what they are by eye before trusting any detector that uses them.")

    # ---- 2. labels --------------------------------------------------------
    print("\n--- existing_faults.tif ---")
    with rasterio.open(d / "existing_faults.tif") as src:
        lab = src.read(1)
        print(f"  shape   : {src.shape}   {'OK' if src.shape == EXPECTED_SHAPE else 'MISMATCH'}")
        print(f"  crs     : {src.crs}   res {src.res}")
        print(f"  dtype   : {lab.dtype}   min {np.nanmin(lab)}  max {np.nanmax(lab)}")
    pos = int((lab >= 1).sum())
    print(f"  catalogue pixels (>=1): {pos:,}  ({pos / lab.size:.4%} of the raster)")
    print(f"  sha256  : {sha256_file(d / 'existing_faults.tif')}")

    # ---- 3. template cross-check ------------------------------------------
    grid_note = "example_submission.tif not present -- skipped"
    if (d / "example_submission.tif").exists():
        print("\n--- example_submission.tif ---")
        with rasterio.open(d / "example_submission.tif") as src:
            ex = src.read(1)
            same_grid = (src.shape == EXPECTED_SHAPE and src.crs and
                         src.crs.to_epsg() == EXPECTED_EPSG)
            print(f"  shape {src.shape}  crs {src.crs}  res {src.res}  "
                  f"{'grid OK' if same_grid else 'GRID MISMATCH'}")
            print(f"  sha256  : {sha256_file(d / 'example_submission.tif')}")
            finite = int(np.isfinite(ex).sum())
            nonfinite = int((~np.isfinite(ex)).sum())
            print(f"  finite {finite:,}   non-finite {nonfinite:,}")
            print(f"  unique values (first 8): {np.unique(ex)[:8].tolist()}")
            if pos and finite == lab.size:
                same = bool(np.array_equal(np.nan_to_num(ex) >= 1, lab >= 1))
                if same:
                    print("  !! This template is NOT 'total fault absence' as the problem")
                    print("     description states -- it matches the label raster. Flag it,")
                    print("     and do not use it as an all-zero baseline.")
            else:
                print("  matches the problem description ('predicts total fault absence') "
                      "if all values are 0")
        grid_note = "checked"

    # ---- 4. persist -------------------------------------------------------
    out_dir = d / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "band_inventory.json").write_text(json.dumps({
        "n_bands": count, "shape": [h, w], "dtype": dtype,
        "epsg": crs.to_epsg() if crs else None,
        "res": list(res), "transform": list(tr),
        "bands": inventory,
        "unnamed_band_indices": unnamed,
    }, indent=2))
    print(f"\nwritten: {out_dir / 'band_inventory.json'}")
    print(f"template: {grid_note}")
    print("\nNext:  python scripts/validate_holdout.py")
    print("       python scripts/build_submission.py --holdout-gate")
    return 0


if __name__ == "__main__":
    sys.exit(main())
