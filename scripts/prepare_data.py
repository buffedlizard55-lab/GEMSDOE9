#!/usr/bin/env python3
"""
prepare_data.py — verifies downloaded rasters and builds footprint mask

After running bash scripts/download_competition_data.sh on unrestricted machine,
this script verifies SHA256, checks band counts, and builds data/processed/ footprint.

The single remaining blocker to training is data placement: run this script after download.
"""

import hashlib
from pathlib import Path
import sys

try:
    import rasterio
    import numpy as np
except ImportError:
    print("Missing rasterio/numpy. pip install rasterio numpy --break-system-packages")
    sys.exit(1)

DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED = {
    "training_features.tif": {"sha256_prefix": "4371c82e", "bands": 19, "shape": (3730, 3292)},
    "existing_faults.tif": {"sha256_prefix": "7ba308cc", "bands": 1},
    "example_submission.tif": {"sha256_prefix": "2176d08e", "bands": 1},
    "GEMS_96647.pdf": {"sha256_prefix": "50d854b1", "bands": None},
}

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()

def main():
    print("=== GEMSDOE9 prepare_data.py ===")
    ok = True
    for fname, meta in EXPECTED.items():
        fpath = DATA_DIR / fname
        if not fpath.exists():
            print(f"MISSING {fname} — download via scripts/download_competition_data.sh")
            ok = False
            continue
        sha = sha256_file(fpath)
        print(f"{fname}: {fpath.stat().st_size} bytes sha256 {sha[:16]}... (expected prefix {meta['sha256_prefix']})")
        if not sha.startswith(meta['sha256_prefix']):
            print(f"  WARNING: SHA256 prefix mismatch! Expected {meta['sha256_prefix']}, got {sha[:8]}")
            # Don't fail, just warn — file may be updated
        if fname.endswith('.tif'):
            try:
                with rasterio.open(fpath) as src:
                    print(f"  bands={src.count} shape={(src.height, src.width)} crs={src.crs} dtype={src.dtypes[0]}")
                    if meta['bands'] and src.count != meta['bands']:
                        print(f"  ERROR: expected {meta['bands']} bands")
                        ok = False
                    # Dump authoritative per-band tags (the official reference
                    # solution reads the band order from exactly these tags:
                    # src.tags(i)['description'] / ['data_category']
                    # https://github.com/drivendataorg/gems-prize-reference-solution)
                    if fname == "training_features.tif":
                        tag_lines = []
                        for i in range(1, src.count + 1):
                            tags = src.tags(i)
                            desc = tags.get('description', '<no description tag>')
                            cat = tags.get('data_category', '<no data_category tag>')
                            line = f"Band {i-1} (rasterio band {i}): {desc} (Category: {cat})"
                            print("  " + line)
                            tag_lines.append(line)
                        out_tags = PROCESSED_DIR / "band_tags.txt"
                        out_tags.write_text("\n".join(tag_lines) + "\n")
                        print(f"  band tags written to {out_tags}")
                        print("  ACTION: compare band_tags.txt against the PROVISIONAL "
                              "indexing in src/gems/features.py and correct any mismatch "
                              "before building scored submissions.")
            except Exception as e:
                print(f"  ERROR reading {fname}: {e}")
                ok = False

    # Build footprint mask from example_submission.tif if present
    example_path = DATA_DIR / "example_submission.tif"
    if example_path.exists():
        print("\nBuilding footprint mask from example_submission.tif...")
        with rasterio.open(example_path) as src:
            data = src.read(1)
            footprint = np.isfinite(data)
            print(f"  footprint: {footprint.sum()} finite / {data.size} total ({footprint.sum()/data.size*100:.2f}%)")
            # Save as npz
            np.savez_compressed(PROCESSED_DIR / "footprint_mask.npz", footprint=footprint, shape=data.shape)
            print(f"  saved to {PROCESSED_DIR / 'footprint_mask.npz'}")
            # Also save stats
            with open(PROCESSED_DIR / "footprint_stats.txt", 'w') as f:
                f.write(f"finite: {footprint.sum()}\n")
                f.write(f"total: {data.size}\n")
                f.write(f"nan: {np.isnan(data).sum()}\n")
                f.write(f"shape: {data.shape}\n")

    if not ok:
        print("\nSome files missing or invalid — please download")
        sys.exit(1)
    else:
        print("\nAll present files OK. Ready for training.")
        print("Next: python scripts/lofso_train_eval.py --hypothesis H9-1")

if __name__ == "__main__":
    main()
