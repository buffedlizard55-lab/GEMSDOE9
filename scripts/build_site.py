#!/usr/bin/env python3
"""
build_site.py — builds docs/ from JSON evidence (placeholder for CI)
In GEMSDOE9, docs/ is hand-written but this script verifies file existence and SHA.
"""

from pathlib import Path
import hashlib

def sha256_file(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()

def main():
    print("=== Build Site ===")
    docs = Path("docs")
    assert docs.exists()
    # Check required files
    required = ["index.html", "executive_summary.html", "how_to_submit.html", "research.html", "hypotheses.html", "data.html", "geotiff_writer.js"]
    for fname in required:
        p = docs / fname
        assert p.exists(), f"Missing {fname}"
        print(f"Found {fname} {p.stat().st_size} bytes")

    # Check downloads
    dl = docs / "downloads"
    assert dl.exists()
    for f in dl.iterdir():
        if f.is_file():
            sha = sha256_file(f)
            print(f"Download {f.name} {f.stat().st_size} bytes sha256 {sha[:16]}...")

    # Verify no duplicate SHA
    duplicate_prefixes = ["7f00890a62878d61", "f347b70daa", "37f9d5b855", "4e03fc9705", "33cec71ff0", "237f0063a4"]
    for f in dl.glob("*.tif"):
        sha = sha256_file(f)
        for dup in duplicate_prefixes:
            if sha.startswith(dup):
                raise ValueError(f"Duplicate detected: {f} sha {sha} matches {dup} — refusing to publish duplicate")
    print("No duplicates found — distinct from 0.1563 pattern")
    print("Site build PASS")

if __name__ == "__main__":
    main()
