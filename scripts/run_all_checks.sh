#!/usr/bin/env bash
set -euo pipefail
# run_all_checks.sh — full repo audit, one command. Run before any submission.
cd "$(dirname "$0")/.."

echo "### 1/6 unit + metric + artifact tests"
python tests/test_validation.py

echo
echo "### 2/6 rules quotes"
python scripts/verify_rules_quotes.py

echo
echo "### 3/6 validate shipped submission (max-compat)"
python scripts/validate_submission.py docs/downloads/gemsdoe9_submission.tif --max-compat

echo
echo "### 4/6 validate uniquely-named copy"
NAMED=$(ls docs/downloads/gems9-*.tif | head -1)
python scripts/validate_submission.py "$NAMED" --max-compat

echo
echo "### 5/6 synthetic holdout gate (metric + masking + protocol wiring)"
python scripts/lofso_train_eval.py --synthetic

echo
echo "### 6/6 site build + duplicate guard"
python scripts/build_site.py

echo
echo "ALL CHECKS PASS"
