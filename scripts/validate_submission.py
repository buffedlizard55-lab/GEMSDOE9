#!/usr/bin/env python3
"""
validate_submission.py — hard gate for anything we are about to upload.

This is the script that must pass before a single weekly submission slot is
spent.  It re-implements the checks the DrivenData form performs, and adds the
two that actually bit us:

  PLATFORM-RANGE   every one of the 12,279,160 values must be finite and inside
                   [0, 1].  This is the check whose failure produces the error
                   "Predicted values must be in range [0, 1]".
  GDAL-READABLE    rasterio/GDAL must be able to decode every strip.  The
                   previous browser writer shipped files GDAL could not read at
                   all, and the platform reported it as a *value* error, which
                   sent us hunting for a NaN that was never the problem.

Usage
  python scripts/validate_submission.py FILE [--json] [--quiet]

Exit code 0 = safe to upload, 1 = do not upload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

EXPECTED = {
    "width": 3292,
    "height": 3730,
    "epsg": 32611,
    "res": (100.0, 100.0),
    "transform": (100.0, 0.0, 243350.0, 0.0, -100.0, 4508550.0),
    "dtype": "float32",
    "bands": 1,
}

# Hashes of sibling-repo artifacts that already burned submission slots.  If a
# file we produce matches one of these, we have rebuilt a duplicate and the
# leaderboard will return the same score for a different-looking name.
KNOWN_DUPLICATE_SHA_PREFIXES = [
    "7f00890a62878d61",   # GEMSDOE1 / GEMSDOE2 / 5GEMSDOE / 8GEMSDOE base -> 0.1563
    "f347b70daa",         # GEMSDOE3 "Pindrop nodes"                     -> 0.1193
    "37f9d5b855",         # GEMSDOE3 "catalogue-gap target"               -> 0.0830
    "4e03fc9705",         # GEMSDOE3 "dense ridge control"                -> 0.1152
    "33cec71ff0",         # 6GEMSDOE HGB 88ch top-3%                     -> 0.0286
    "237f0063a4",         # GEMSDOE4 lineament + proxy labels            -> 0.0343
    "8ecbdc712da4b83e",   # GEMSDOE9 v1 placeholder (replaced, kept as a
    #                        guard so the retired artifact can never be re-shipped)
]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_checks(path: Path) -> tuple[list[dict], dict]:
    checks: list[dict] = []

    def add(name, ok, detail=""):
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    info: dict = {}
    add("file-exists", path.exists(), str(path))
    if not path.exists():
        return checks, info

    import rasterio

    # --- can GDAL decode it at all? ---------------------------------------
    try:
        with rasterio.open(path) as src:
            data = src.read(1)
        add("GDAL-READABLE", True, f"decoded {data.shape} in full")
    except Exception as e:  # noqa: BLE001
        add("GDAL-READABLE", False, f"{type(e).__name__}: {e}")
        add("PLATFORM-RANGE", False, "not readable, cannot evaluate")
        return checks, info

    with rasterio.open(path) as src:
        info = {
            "count": src.count,
            "dtype": src.dtypes[0],
            "crs": str(src.crs),
            "epsg": src.crs.to_epsg() if src.crs else None,
            "res": tuple(src.res),
            "transform": (src.transform.a, src.transform.b, src.transform.c,
                          src.transform.d, src.transform.e, src.transform.f),
            "nodata": src.nodatavals[0],
            "compression": str(src.compression),
            "shape": (src.height, src.width),
        }

    add("single-band", info["count"] == EXPECTED["bands"], f"count={info['count']}")
    add("dtype-float32", info["dtype"] == EXPECTED["dtype"], f"dtype={info['dtype']}")
    add("crs-epsg32611", info["epsg"] == EXPECTED["epsg"], f"crs={info['crs']}")
    add("resolution-100m",
        abs(info["res"][0] - 100.0) < 1e-6 and abs(info["res"][1] - 100.0) < 1e-6,
        f"res={info['res']}")
    add("shape-3292x3730",
        info["shape"] == (EXPECTED["height"], EXPECTED["width"]),
        f"shape={info['shape']}")
    add("geotransform",
        all(abs(a - b) < 1e-6 for a, b in zip(info["transform"], EXPECTED["transform"])),
        f"transform={info['transform']}")

    # --- THE gate ----------------------------------------------------------
    n_nan = int(np.isnan(data).sum())
    n_inf = int(np.isinf(data).sum())
    finite = data[np.isfinite(data)]
    vmin = float(finite.min()) if finite.size else float("nan")
    vmax = float(finite.max()) if finite.size else float("nan")
    in_range = finite.size > 0 and vmin >= 0.0 and vmax <= 1.0
    add("PLATFORM-RANGE",
        n_nan == 0 and n_inf == 0 and in_range,
        f"nan={n_nan} inf={n_inf} range=[{vmin}, {vmax}] "
        f"(platform rejects if any value is outside [0,1]; NaN is outside [0,1])")

    add("no-nodata-tag", info["nodata"] is None,
        f"nodata={info['nodata']} (all-finite files carry none)")

    # --- content ------------------------------------------------------------
    n_pos = int((data > 0).sum())
    n_one = int((data == 1).sum())
    frac = n_pos / data.size
    add("has-content", 0 < n_pos < data.size, f"{n_pos} px > 0 ({frac:.3%})")
    add("density-sane", 1e-5 <= frac <= 0.5,
        f"positive density {frac:.4%} "
        f"(an all-zero raster scores DTI=0; >50% is a failed mask)")
    # A real check on the SHAPE of the field, not a tautology. Most files have
    # only two distinct values; a genuinely continuous probability field has
    # thousands. Which of the two scores better is NOT settled a priori -- on an
    # identical pixel set, soft beat hard in 19 of 20 phantom cells, because TP_w
    # and FP_w both scale with p and phi usually binds first. So this gate flags
    # a continuous field for review; it does not claim one shape is optimal.
    n_unique = int(np.unique(data).size)
    add("prediction-shape", n_unique <= 64,
        f"{n_unique} distinct values ({n_one} px exactly 1.0) -- a hard 0/1 mask. "
        f"A continuous probability field is legal and may score either way; "
        f"which one wins is a holdout question, not a formatting rule")

    # --- duplicate guard ----------------------------------------------------
    sha = sha256_file(path)
    info["sha256"] = sha
    info["n_positive"] = n_pos
    info["vmin"], info["vmax"] = vmin, vmax
    dup = next((p for p in KNOWN_DUPLICATE_SHA_PREFIXES if sha.startswith(p)), None)
    add("not-a-known-duplicate", dup is None,
        f"sha256={sha[:24]}..." + (f"  MATCHES SPENT SLOT {dup}" if dup else ""))

    return checks, info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tif_path")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    path = Path(args.tif_path)
    checks, info = run_checks(path)

    if args.json:
        print(json.dumps({"path": str(path), "checks": checks, "info": info,
                          "ok": all(c["ok"] for c in checks)}, indent=2, default=str))
    else:
        if not args.quiet:
            print(f"=== validate_submission: {path} ({path.stat().st_size if path.exists() else 0} bytes) ===")
        for c in checks:
            if args.quiet and c["ok"]:
                continue
            print(f"{'PASS' if c['ok'] else 'FAIL':4}  {c['check']:24}  {c['detail']}")
        if not args.quiet:
            passed = sum(c["ok"] for c in checks)
            print(f"\n{passed}/{len(checks)} checks passed")
            if passed != len(checks):
                print("DO NOT UPLOAD. Fix the FAIL lines first.")

    return 0 if all(c["ok"] for c in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
