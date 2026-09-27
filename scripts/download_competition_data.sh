#!/usr/bin/env bash
# download_competition_data.sh — place the competition rasters in data/.
#
# HONESTY NOTE. The previous revision of this script said it "attempts wget for
# Dropbox mirrors". It could not succeed: this environment denies outbound HTTPS
# from the shell, and the Dropbox links are not linked from any DrivenData or
# DOE page, so their provenance is unestablished. There is no verified mirror.
#
# This script therefore does the only useful thing available: it tells you
# exactly what to download, from where, where to put it, and how to prove you
# got it right. It never asks for credentials and never works around a login.
#
# Source of truth (requires a DrivenData account):
#   https://www.drivendata.org/competitions/306/competition-doe-gems/data/
#
# Usage:
#   1. sign in at the URL above in a browser and download the files
#   2. move them into ./data/
#   3. bash scripts/download_competition_data.sh          # verifies what is there
#   4. python scripts/prepare_data.py                     # band inventory + checks

set -euo pipefail
cd "$(dirname "$0")/.."
DATA_DIR="data"
mkdir -p "$DATA_DIR"

URL="https://www.drivendata.org/competitions/306/competition-doe-gems/data/"

cat <<EOF
=== GEMSDOE9 — competition data ===

Authoritative source (needs a DrivenData account):
  $URL

Download these into ./$DATA_DIR/ :

  training_features.tif    REQUIRED
      Multi-band predictor stack: EPSG:32611, 100 m, 3292 x 3730.
      Layer groups per the problem description:
        - surface conductivity, depth to conductive base surface
        - detrended elevation, slope of detrended elevation
        - dilatation rate, shear strain rate, second invariant of the strain
          rate tensor
        - isostatic gravity anomaly, slope of the isostatic gravity anomaly
        - magnetics: reduced-to-pole anomaly, total magnetic intensity,
          vertical and horizontal slope of TMI, top-of-crustal source depth
        - density of earthquakes
      The problem description lists groups, not an ordered index. The exact
      band count and names are read from the file's own per-band description
      tags by scripts/prepare_data.py.

  existing_faults.tif       REQUIRED
      The known USGS/INGENIOUS faults, rasterised. This is simultaneously the
      training labels, the free core of the submission, and the evaluation
      mask in the blocked holdout.

  example_submission.tif    optional but recommended
      The organiser's template. Used only to cross-check the grid. The problem
      description says it "predicts total fault absence"; scripts/prepare_data.py
      checks that and flags it loudly if it is not what the description says.

  1m_DEM_links.csv          optional
      URLs for 1 m DEM tiles. Only detector G-5 can use them, and G-5 falls back
      to the competition's own detrended-elevation layers, so nothing is blocked.

  GEMS_96647.pdf            optional
      The rules PDF. Nothing on the site is attributed to it.

EOF

echo "--- what is already in ./$DATA_DIR/ ---"
found_all=1
for f in training_features.tif existing_faults.tif example_submission.tif 1m_DEM_links.csv; do
  if [[ -f "$DATA_DIR/$f" ]]; then
    printf '  [ok]      %-24s %14s bytes\n' "$f" "$(stat -c%s "$DATA_DIR/$f")"
  else
    printf '  [missing] %-24s\n' "$f"
    case "$f" in
      training_features.tif|existing_faults.tif) found_all=0 ;;
    esac
  fi
done

echo
if [[ "$found_all" -eq 0 ]]; then
  echo "REQUIRED FILES ARE MISSING."
  echo
  echo "There is no automated way to get them from here: the data tab needs a"
  echo "DrivenData account, and this environment has no outbound HTTPS from the"
  echo "shell, so no URL will resolve. No credentials will be requested."
  echo
  echo "Once you have them:"
  echo "  python scripts/prepare_data.py          # verify + dump the band inventory"
  echo "  python scripts/validate_holdout.py       # spatially-blocked holdout + gate"
  echo "  python scripts/build_submission.py --holdout-gate"
  exit 1
fi

echo "Required files present. Next:"
echo "  python scripts/prepare_data.py"
