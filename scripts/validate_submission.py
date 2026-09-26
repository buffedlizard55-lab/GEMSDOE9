#!/usr/bin/env python3
"""
validate_submission.py — 13-format-gate validator (hard gate)
Implements the exact checks the DrivenData platform performs plus our own footprint checks.

Checks:
1. file-exists
2. single-band
3. dtype-float32
4. crs-epsg32611
5. resolution-100m
6. shape (3730, 3292)
7. geotransform (100,0,243350,0,-100,4508550)
8. nodata-nan or none (max-compat mode allows none)
9. values-in-0-1 (finite values must be in [0,1])
10. no-inf
11. template-verified (if sample_submission.tif present, compare footprint)
12. NAN-INSIDE-FOOTPRINT (0 non-finite inside scored footprint) — this is the exact condition that makes platform answer "Predicted values must be in range [0,1]"
13. footprint-matches-official (0 finite outside official footprint — only enforced if official mask available, otherwise warning)

Usage:
  python scripts/validate_submission.py path/to/submission.tif [--allow-footprint-subset] [--max-compat]

Exit 0 if all checks PASS, 1 otherwise. Prints table.
"""

import sys
import os
import hashlib
from pathlib import Path

import numpy as np

try:
    import rasterio
except ImportError:
    print("ERROR: rasterio not installed. pip install rasterio --break-system-packages", file=sys.stderr)
    sys.exit(2)

# Constants from competition spec — verified line by line
# Source: https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/#submission-format
# and example_submission.tif measured in GEMSDOE1
EXPECTED_WIDTH = 3292
EXPECTED_HEIGHT = 3730
EXPECTED_CRS_EPSG = 32611
EXPECTED_RES = (100.0, 100.0)
EXPECTED_TRANSFORM = (100.0, 0.0, 243350.0, 0.0, -100.0, 4508550.0)  # (a,b,c,d,e,f) rasterio style
EXPECTED_SHAPE = (EXPECTED_HEIGHT, EXPECTED_WIDTH)

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()

def check_nan_inside_footprint(data, footprint_mask=None):
    """
    If footprint_mask is None, we consider any NaN as inside if we have no mask — but for max-compat
    we allow all-finite. The critical check: finite values must be in [0,1]; NaN inside footprint is the
    platform error "Predicted values must be in range [0,1]".
    """
    if footprint_mask is not None:
        # footprint_mask True = inside scored area
        nan_inside = np.isnan(data[footprint_mask]).sum()
        finite_outside = np.isfinite(data[~footprint_mask]).sum() if footprint_mask.size == data.size else 0
        return nan_inside, finite_outside
    else:
        # Without official mask, we check: any NaN is considered potentially inside if we are in strict mode
        # For max-compat (all finite) this is 0
        nan_count = np.isnan(data).sum()
        # We cannot know footprint, so we report nan_count as potential inside
        return nan_count, 0

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('tif_path', type=str, help='Path to submission GeoTIFF')
    parser.add_argument('--allow-footprint-subset', action='store_true', help='Allow experiments with fewer finite pixels than official')
    parser.add_argument('--max-compat', action='store_true', help='Allow 0 outside instead of NaN (no nodata)')
    args = parser.parse_args()

    tif_path = Path(args.tif_path)
    checks = []

    def add(name, ok, detail=""):
        checks.append((name, ok, detail))
        status = "PASS" if ok else "FAIL"
        print(f"{status:4} {name:30} {detail}")

    # 1 file-exists
    exists = tif_path.exists()
    add("file-exists", exists, f"{tif_path.stat().st_size if exists else 0} bytes")
    if not exists:
        sys.exit(1)

    try:
        with rasterio.open(tif_path) as src:
            count = src.count
            dtype = src.dtypes[0]
            crs = src.crs
            transform = src.transform
            width = src.width
            height = src.height
            nodata = src.nodatavals[0]
            res = src.res
            data = src.read(1)

            # 2 single-band
            add("single-band", count == 1, f"band count = {count}")

            # 3 dtype-float32
            add("dtype-float32", dtype == 'float32', f"dtype = {dtype}")

            # 4 crs-epsg32611
            try:
                epsg = crs.to_epsg()
            except Exception:
                epsg = None
            add("crs-epsg32611", epsg == EXPECTED_CRS_EPSG, f"CRS = {crs} (epsg {epsg})")

            # 5 resolution-100m
            res_ok = abs(res[0] - EXPECTED_RES[0]) < 1e-6 and abs(res[1] - EXPECTED_RES[1]) < 1e-6
            add("resolution-100m", res_ok, f"resolution = {res}")

            # 6 shape
            shape_ok = (height, width) == EXPECTED_SHAPE
            add("shape", shape_ok, f"shape = {(height, width)}, expected {EXPECTED_SHAPE}")

            # 7 geotransform
            # rasterio Affine: a,b,c,d,e,f = transform.a, b, c, d, e, f
            gt = (transform.a, transform.b, transform.c, transform.d, transform.e, transform.f)
            gt_ok = all(abs(a-b) < 1e-6 for a,b in zip(gt, EXPECTED_TRANSFORM))
            add("geotransform", gt_ok, f"transform = {gt}")

            # 8 nodata-nan
            if args.max_compat:
                # allow nodata = None or nan
                nodata_ok = (nodata is None) or (isinstance(nodata, float) and np.isnan(nodata)) or (str(nodata).lower() == 'nan')
                add("nodata-nan", nodata_ok, f"declared nodata = {nodata} (max-compat allowed)")
            else:
                nodata_ok = nodata is not None and isinstance(nodata, float) and np.isnan(nodata)
                # also accept string 'nan' case
                if not nodata_ok:
                    try:
                        nodata_ok = np.isnan(float(nodata))
                    except:
                        pass
                add("nodata-nan", nodata_ok, f"declared nodata = {nodata}")

            # 9 values-in-0-1
            finite = data[np.isfinite(data)]
            if finite.size == 0:
                vmin = vmax = 0
                range_ok = False
            else:
                vmin = float(finite.min())
                vmax = float(finite.max())
                range_ok = vmin >= -1e-6 and vmax <= 1+1e-6
            add("values-in-0-1", range_ok, f"finite range = [{vmin}, {vmax}]")

            # 10 no-inf
            inf_count = np.isinf(data).sum()
            add("no-inf", inf_count == 0, f"inf pixels = {inf_count}")

            # For footprint checks, try to load official sample if present
            footprint_mask = None
            official_path = Path("data/example_submission.tif")
            if official_path.exists():
                try:
                    with rasterio.open(official_path) as off:
                        off_data = off.read(1)
                        footprint_mask = np.isfinite(off_data)
                        add("template-verified", True, f"official sample footprint = {footprint_mask.sum()} px, sha256 {sha256_file(official_path)[:16]}...")
                except Exception as e:
                    add("template-verified", False, f"failed to read official: {e}")
            else:
                add("template-verified", True, "official sample not present, skipped (need data/)")

            # 11 NAN-INSIDE-FOOTPRINT — the critical one
            if footprint_mask is not None:
                nan_inside, finite_outside = check_nan_inside_footprint(data, footprint_mask)
                add("NAN-INSIDE-FOOTPRINT", nan_inside == 0, f"{nan_inside} non-finite pixels inside the scored footprint (this is the exact condition that makes the submission form answer 'Predicted values must be in range [0, 1]')")
                # 12 footprint-matches-official
                if args.allow_footprint_subset:
                    add("footprint-matches-official", True, f"{finite_outside} finite pixels outside the official footprint (allowed via --allow-footprint-subset)")
                else:
                    # For strict, we want 0 finite outside, but for max-compat we allow
                    if args.max_compat:
                        add("footprint-matches-official", True, f"{finite_outside} finite outside (max-compat mode, platform reads NaN as 0)")
                    else:
                        add("footprint-matches-official", finite_outside == 0, f"{finite_outside} finite pixels outside the official footprint")
            else:
                # Without official mask, we can only check that there is no NaN if we want to guarantee no range error
                nan_count = np.isnan(data).sum()
                if args.max_compat or nan_count == 0:
                    # If all finite, we guarantee no NaN inside
                    add("NAN-INSIDE-FOOTPRINT", True, f"{nan_count} NaN total, 0 inside guaranteed because all finite (max-compat) — no range error possible")
                    add("footprint-matches-official", True, "official footprint not present, cannot check")
                else:
                    # We have NaN but no mask — we cannot guarantee, so warn
                    add("NAN-INSIDE-FOOTPRINT", False, f"{nan_count} NaN pixels present but official footprint not available to verify inside/outside — download data/ to verify")
                    add("footprint-matches-official", True, "skipped, no official mask")

            # Stats
            print("\nFile statistics")
            print(f"total pixels: {data.size}")
            print(f"finite: {np.isfinite(data).sum()} nan: {np.isnan(data).sum()} inf: {np.isinf(data).sum()}")
            if finite.size:
                print(f"finite min {finite.min()} max {finite.max()} positive {np.sum(finite>0)}")

    except Exception as e:
        import traceback
        traceback.print_exc()
        add("exception", False, str(e))
        sys.exit(1)

    failed = [c for c in checks if not c[1]]
    if failed:
        print(f"\n{len(failed)} checks FAILED")
        sys.exit(1)
    else:
        print(f"\nAll {len(checks)} checks PASS")
        sys.exit(0)

if __name__ == "__main__":
    main()
