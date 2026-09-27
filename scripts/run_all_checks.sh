#!/usr/bin/env bash
# run_all_checks.sh — one command, the whole audit. Run before and after any change.
#
#   bash scripts/run_all_checks.sh
#
# Every step is offline: no network, no DrivenData login, no GPU.
# Exits non-zero the moment anything fails, so it is safe to put in CI.

set -euo pipefail
cd "$(dirname "$0")/.."

step() { printf '\n\033[1m=== %s ===\033[0m\n' "$1"; }

step "0/6  dependencies"
python3 - <<'PY'
import importlib, sys
missing = []
for m in ("numpy", "scipy", "rasterio", "sklearn"):
    try:
        mod = importlib.import_module(m)
        print(f"  {m:10} {getattr(mod, '__version__', '?')}")
    except ImportError:
        missing.append(m)
if missing:
    sys.exit("MISSING: " + ", ".join(missing) +
             "\n  pip install -r requirements.txt"
             "\n  (add --break-system-packages on a PEP-668 system)")
PY

step "1/6  test suite (metric, algebra, corridor+budget, masking, G-5 memory, writer, folds, validator)"
python3 tests/test_validation.py

step "2/6  metric-shape experiment (what the submission SHAPE is worth)"
python3 scripts/validate_holdout.py --strategy   # 5 phantoms, ~2 min; this is the artifact the site renders

step "3/6  submission artifact"
python3 scripts/build_submission.py

step "4/6  independent validation of the shipped artifact"
python3 scripts/validate_submission.py docs/downloads/*.tif

step "5/6  site build (renders docs/ from scripts/site_data.py + live artifacts)"
python3 scripts/build_site.py

step "6/6  site freshness"
python3 scripts/build_site.py --check

printf '\n\033[1;32mALL CHECKS PASS\033[0m\n'
printf 'Reminder: 0 of 5 detectors are validated and no model has been trained.\n'
printf 'The data tab needs a DrivenData login. See docs/data.html.\n'
