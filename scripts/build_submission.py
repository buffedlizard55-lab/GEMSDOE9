#!/usr/bin/env python3
"""
build_submission.py — produce the .tif that goes on DrivenData, plus the exact
unique file name and the Note to paste next to it.

Two paths, and the difference is never hidden:

  MODEL PATH   (data/training_features.tif + data/existing_faults.tif present)
      Runs src/gems/pipeline.py: fit the pixel classifier, choose the corridor
      shape by a blocked sweep, write the result.  `--holdout-gate` runs the
      spatially-blocked holdout first and refuses to write a model submission
      unless it beats both the catalogue-only and budget-matched-random arms.

  FORMAT PATH  (no data — the common case in a sandbox without a DrivenData
                login)
      Writes a geometric PLACEHOLDER so the format/IO/upload path can be
      exercised end to end.  It is generated from a fixed seed with no GeoDAWN
      data at all, it is named `gemsdoe9-PLACEHOLDER-...`, and the Note says so
      in words.  It is not a prediction of anything.  Do not read a score into it.

Invariants enforced on every write, no matter the path:
  * 3292 x 3730, float32, single band, EPSG:32611, 100 m,
    geotransform (100, 0, 243350, 0, -100, 4508550)
  * every value finite and inside [0, 1]   <-- the "Predicted values must be in
    range [0, 1]" rejection cannot be produced by this script
  * no nodata tag declared
  * SHA256 differs from every sibling artifact that already spent a slot
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.gems import strategy as S                       # noqa: E402
from scripts.validate_submission import (                # noqa: E402
    KNOWN_DUPLICATE_SHA_PREFIXES, run_checks, sha256_file)

WIDTH, HEIGHT, EPSG = 3292, 3730, 32611
TRANSFORM = (100.0, 0.0, 243350.0, 0.0, -100.0, 4508550.0)
N_TOTAL = WIDTH * HEIGHT

# The field the site ships.  Keep in sync with docs/site_config.json.
PLACEHOLDER_FRACTION = 0.028


def utc_stamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


# --------------------------------------------------------------------------- #
def _xorshift32(seed: int):
    """Identical RNG to docs/geotiff_writer.js::buildPlaceholderField."""
    s = seed & 0xFFFFFFFF or 1

    def nxt() -> float:
        nonlocal s
        s ^= (s << 13) & 0xFFFFFFFF
        s &= 0xFFFFFFFF
        s ^= s >> 17
        s ^= (s << 5) & 0xFFFFFFFF
        s &= 0xFFFFFFFF
        return s / 4294967296.0

    return nxt


def placeholder_field(fraction: float = PLACEHOLDER_FRACTION, seed: int = 20260927) -> np.ndarray:
    """
    A fixed-seed oriented band pattern.  Deliberately NOT geological: it exists
    so the GeoTIFF bytes, the validator and the upload form can all be exercised
    before any competition data is available.

    The RNG, the two oriented sinusoids and the top-k threshold are a
    line-for-line mirror of docs/geotiff_writer.js::buildPlaceholderField, so
    the browser build and this CLI build produce the same pixel field and the
    same SHA256. tests/test_geotiff_writer.py asserts that.
    """
    rnd = _xorshift32(seed)
    f1, f2 = 14.0, 11.0
    c1x, c1y = math.cos(math.pi / 6), math.sin(math.pi / 6)
    c2x, c2y = math.cos(2 * math.pi / 3), math.sin(2 * math.pi / 3)
    score = np.empty((HEIGHT, WIDTH), dtype=np.float64)
    for y in range(HEIGHT):
        v = y / HEIGHT
        base = y * WIDTH
        for x in range(WIDTH):
            u = x / WIDTH
            a = math.sin(2 * math.pi * f1 * (u * c1x + v * c1y))
            b = math.sin(2 * math.pi * f2 * (u * c2x + v * c2y))
            score[y, x] = 0.45 * a * b + 0.35 * a + 0.35 * b + 0.30 * rnd()
    field = S.binarise(score, int(round(N_TOTAL * fraction)))
    return field.astype(np.float32)


# --------------------------------------------------------------------------- #
def write_geotiff(field: np.ndarray, out_path: Path) -> None:
    import rasterio
    from rasterio.transform import Affine

    if field.shape != (HEIGHT, WIDTH):
        raise ValueError(f"field shape {field.shape} != {(HEIGHT, WIDTH)}")
    field = np.asarray(field, dtype=np.float32)
    bad = int((~np.isfinite(field)).sum())
    if bad:
        raise ValueError(f"{bad} non-finite values — refusing to write (this is the "
                         f"cause of 'Predicted values must be in range [0, 1]')")
    lo, hi = float(field.min()), float(field.max())
    if lo < 0.0 or hi > 1.0:
        raise ValueError(f"value range [{lo}, {hi}] outside [0, 1]")
    if not (lo < hi):
        raise ValueError(f"degenerate field: constant value {lo}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        out_path, "w", driver="GTiff", height=HEIGHT, width=WIDTH, count=1,
        dtype="float32", crs=f"EPSG:{EPSG}",
        transform=Affine(*TRANSFORM), nodata=None, compress="deflate",
        tiled=False, BIGTIFF="NO",
    ) as dst:
        dst.write(field, 1)

    sha = sha256_file(out_path)
    dup = next((p for p in KNOWN_DUPLICATE_SHA_PREFIXES if sha.startswith(p)), None)
    if dup:
        out_path.unlink()
        raise RuntimeError(
            f"refusing to publish {out_path.name}: sha256 {sha[:24]}... already spent "
            f"a submission slot as {dup}. This is the 0.1563 duplicate trap."
        )


def unique_name(kind: str, sha8: str) -> str:
    """
    Content-addressed name: stable across rebuilds (so the site can link to it
    forever) and provably distinct from every sibling-repo artifact (the hash is
    in the name and is checked against the spent-slot list before publishing).

    If you are re-uploading the *same* bytes, rename the file first -- e.g.
    `...-r2.tif` -- so the two submissions are distinguishable in the history.
    """
    return f"gemsdoe9-{kind}-{sha8}.tif"


def build_note(kind: str, sha8: str, extra: str = "") -> str:
    """The string to paste into the 'Note (optional)' box."""
    if kind == "PLACEHOLDER":
        return (f"GEMSDOE9 FORMAT-CHECK placeholder (no model, no GeoDAWN data) | "
                f"band-pattern {PLACEHOLDER_FRACTION:.1%} | {sha8}")
    if kind == "MODEL":
        return f"GEMSDOE9 {extra} | corridor model+catalogue | {sha8} | unique, not a sibling rebuild"
    return f"GEMSDOE9 {kind} | {sha8}"


# --------------------------------------------------------------------------- #
def build_model_field(args) -> tuple[np.ndarray, dict, str]:
    """Full model path. Returns (field, info, strategy_tag)."""
    from src.gems import pipeline as P

    data = P.load_competition(args.data_dir)

    gate_report = None
    if args.holdout_gate:
        print("Running spatially-blocked holdout gate (this is the price of a slot)...")
        gate_report = P.blocked_holdout(data, n_folds=args.folds,
                                        buffer_px=args.buffer, seed=args.seed)
        ok, why = P.holdout_gate(gate_report, min_lift=args.min_lift)
        print(f"  GATE {'PASS' if ok else 'FAIL'}: {why}")
        if not ok:
            raise SystemExit(
                "Holdout gate failed. No submission written. "
                "A weekly slot is not spent on an idea that has not beaten the "
                "catalogue baseline and a budget-matched random control."
            )

    field, info = P.build_final_field(data, seed=args.seed,
                                      halo_px=args.halo, budget_frac=args.budget)
    info["holdout"] = gate_report
    tag = f"MODEL halo{info['chosen_halo_px']}px budget{info['chosen_budget_frac']}"
    return field, info, tag


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="Build a DrivenData-ready GeoTIFF.")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="docs/downloads")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--buffer", type=int, default=3, help="fold buffer in px (300 m = 3)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--halo", type=int, default=None, help="override corridor halo px")
    ap.add_argument("--budget", type=float, default=None, help="override paid budget fraction")
    ap.add_argument("--holdout-gate", action="store_true",
                    help="run the blocked holdout and refuse to write unless it passes")
    ap.add_argument("--min-lift", type=float, default=0.02)
    ap.add_argument("--force-placeholder", action="store_true",
                    help="write the format-check placeholder even if data/ is present")
    args = ap.parse_args()

    have_data = (Path(args.data_dir) / "training_features.tif").exists() and \
                (Path(args.data_dir) / "existing_faults.tif").exists()
    if have_data and not args.force_placeholder:
        field, info, tag = build_model_field(args)
        kind = "MODEL"
    else:
        if not have_data:
            print(f"[info] {args.data_dir}/training_features.tif not found -> "
                  f"FORMAT PATH. The competition data tab requires a DrivenData "
                  f"login (https://www.drivendata.org/competitions/306/"
                  f"competition-doe-gems/data/). Writing a clearly-labelled "
                  f"placeholder so the upload path can be tested.")
        field = placeholder_field()
        info = {
            "kind": "FORMAT-CHECK PLACEHOLDER",
            "generator": "scripts/build_submission.py::placeholder_field",
            "uses_competition_data": False,
            "seed": 20260927,
            "fraction": PLACEHOLDER_FRACTION,
            "warning": ("This file is a fixed-seed geometric band pattern. It is NOT a "
                        "geological prediction and carries no information about faults. "
                        "It exists so the GeoTIFF format, the validator and the DrivenData "
                        "upload form can be exercised end to end."),
        }
        tag = "PLACEHOLDER"
        kind = "PLACEHOLDER"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Only one artifact may live in downloads/ at a time. Two files with the
    # same pixel field under different names is precisely the mistake that put
    # three sibling repos on the leaderboard at an identical 0.1563.
    for stale in sorted(out_dir.glob("gemsdoe9-*.tif")) + sorted(out_dir.glob("gemsdoe9-*.zip")):
        stale.unlink()
        print(f"[clean] removed previous artifact {stale.name}")

    # Two-step so the unique name can embed the hash of the bytes we just wrote.
    tmp = out_dir / "_tmp_build.tif"
    write_geotiff(field, tmp)
    sha = sha256_file(tmp)
    sha8 = sha[:8]
    stamp = utc_stamp()
    name = unique_name(kind, sha8)
    final = out_dir / name
    tmp.replace(final)

    note = build_note(kind, sha8, tag)
    checks, meta = run_checks(final)
    ok = all(c["ok"] for c in checks)

    manifest = {
        "file": name,
        "sha256": sha,
        "bytes": final.stat().st_size,
        "note": note,
        "kind": kind,
        "built_utc": stamp,
        "positive_px": int((field > 0).sum()),
        "positive_fraction": float((field > 0).sum() / field.size),
        "validator": {"all_passed": ok,
                      "n_checks": len(checks),
                      "checks": [c["check"] for c in checks],
                      "failed": [c["check"] for c in checks if not c["ok"]]},
        "build_info": {k: v for k, v in info.items() if k != "holdout"},
        "holdout": info.get("holdout"),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    print()
    for c in checks:
        print(f"{'PASS' if c['ok'] else 'FAIL':4}  {c['check']:22}  {c['detail']}")
    print()
    print("=" * 72)
    print(f"FILE TO UPLOAD : {name}")
    print(f"SIZE          : {final.stat().st_size:,} bytes")
    print(f"SHA256        : {sha}")
    print(f"NOTE TO PASTE : {note}")
    print(f"UPLOAD AT     : https://www.drivendata.org/competitions/306/"
          f"competition-doe-gems/submissions/")
    print("=" * 72)
    if not ok:
        print("VALIDATION FAILED — do not upload.")
        return 1
    if kind == "PLACEHOLDER":
        print("\nNOTE: this is the format-check placeholder, not a prediction.\n"
              "      It exists so the upload path works. Re-run with the competition\n"
              "      data in place (and --holdout-gate) for a real model submission.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
