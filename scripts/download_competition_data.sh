#!/usr/bin/env bash
set -euo pipefail

# download_competition_data.sh — downloads official competition rasters
# Requires DrivenData login cookie or manual download on unrestricted machine
# Places files into data/
#
# Official sources (login required):
# - https://www.drivendata.org/competitions/306/competition-doe-gems/data/
#   - training_features.tif (19 bands, 399.5 MB, SHA256 4371c82e3b8339b8...)
#   - existing_faults.tif / labels.tif (415.8 KB, SHA256 7ba308ccdc4418b3...)
#   - example_submission.tif (1.5 MB, SHA256 2176d08e485aa2cd...)
#   - 1m_DEM_links.csv (716 URLs)
# - Dropbox mirrors (provided in prompt, may be transient):
#   - https://www.dropbox.com/scl/fi/3vz9o0wwavi26xaeoxlwr/gems-geodawn-numerical-features.tif?rlkey=je8d8fepqfbst9lnwsq9rkplu&st=zj1lag1r&dl=0
#   - https://www.dropbox.com/scl/fi/t7fyt03qdh9egyme0itwo/existing_faults.tif?rlkey=yiao96uluqdkipf0h5vju71jf&st=rnino7ya&dl=0
#   - etc.

DATA_DIR="data"
mkdir -p "$DATA_DIR"

echo "=== GEMSDOE9 Data Download ==="
echo "This script requires manual login to DrivenData on an unrestricted machine."
echo "If you have the files locally, place them in data/ and run python scripts/prepare_data.py"
echo ""

# Check if files already exist
if [[ -f "$DATA_DIR/training_features.tif" ]]; then
    echo "Found $DATA_DIR/training_features.tif — skipping download"
else
    echo "Missing $DATA_DIR/training_features.tif"
    echo "Please download from https://www.drivendata.org/competitions/306/competition-doe-gems/data/"
    echo "Or from Dropbox mirror (if still live):"
    echo "  https://www.dropbox.com/scl/fi/3vz9o0wwavi26xaeoxlwr/gems-geodawn-numerical-features.tif?rlkey=je8d8fepqfbst9lnwsq9rkplu&st=zj1lag1r&dl=1"
    echo ""
    echo "Attempting wget for Dropbox mirrors (may fail in restricted sandbox)..."
    # Try wget if available
    if command -v wget >/dev/null 2>&1; then
        wget -O "$DATA_DIR/training_features.tif" "https://www.dropbox.com/scl/fi/3vz9o0wwavi26xaeoxlwr/gems-geodawn-numerical-features.tif?rlkey=je8d8fepqfbst9lnwsq9rkplu&st=zj1lag1r&dl=1" || echo "wget failed — manual download required"
    fi
fi

if [[ -f "$DATA_DIR/existing_faults.tif" ]]; then
    echo "Found $DATA_DIR/existing_faults.tif"
else
    echo "Missing $DATA_DIR/existing_faults.tif — trying mirror"
    if command -v wget >/dev/null 2>&1; then
        wget -O "$DATA_DIR/existing_faults.tif" "https://www.dropbox.com/scl/fi/t7fyt03qdh9egyme0itwo/existing_faults.tif?rlkey=yiao96uluqdkipf0h5vju71jf&st=rnino7ya&dl=1" || echo "wget failed"
    fi
fi

if [[ -f "$DATA_DIR/example_submission.tif" ]]; then
    echo "Found $DATA_DIR/example_submission.tif"
else
    echo "Missing $DATA_DIR/example_submission.tif — trying mirror"
    if command -v wget >/dev/null 2>&1; then
        wget -O "$DATA_DIR/example_submission.tif" "https://www.dropbox.com/scl/fi/6rgvnuady818ol8yqgis4/example_submission.tif?rlkey=kbykilvau066xuogoosbf4cq8&st=8junzdyw&dl=1" || echo "wget failed"
    fi
fi

if [[ -f "$DATA_DIR/GEMS_96647.pdf" ]]; then
    echo "Found $DATA_DIR/GEMS_96647.pdf"
else
    echo "Missing $DATA_DIR/GEMS_96647.pdf — trying mirror"
    if command -v wget >/dev/null 2>&1; then
        wget -O "$DATA_DIR/GEMS_96647.pdf" "https://www.dropbox.com/scl/fi/aemhtutjgcp6tr3tint94/GEMS_96647.pdf?rlkey=rek210cj2smnmzb8n0sla1vmd&st=wz4kofki&dl=1" || echo "wget failed"
    fi
fi

echo ""
echo "After download, verify SHA256:"
echo "  training_features.tif: 4371c82e3b8339b8... (399.5 MB, 19 bands)"
echo "  existing_faults.tif:   7ba308ccdc4418b3... (415.8 KB)"
echo "  example_submission.tif: 2176d08e485aa2cd... (1.5 MB)"
echo "  GEMS_96647.pdf:        50d854b1e0239fe6... (444.5 KB)"
echo ""
echo "Then run: python scripts/prepare_data.py"
