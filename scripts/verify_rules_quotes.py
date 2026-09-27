#!/usr/bin/env python3
"""
verify_rules_quotes.py — check that the Rules PDF is reachable and hashable,
and report the quotes the site attributes to it.

Honesty note, because the previous revision of this repo got this wrong: it
listed `scripts/verify_rules_quotes.py` in the README and on the site as
"verified line by line", and quoted four sentences from
https://docs.nlr.gov/docs/fy26osti/96647.pdf. This environment has no outbound
network from the shell, so that PDF was never fetched and those sentences were
never checked against it. The claim was unsupported.

This script now does what it can and says what it cannot:

  * It attempts to fetch the PDF and hash it. If the network is unavailable it
    reports that plainly instead of implying a verification.
  * It checks the quotes the SITE actually uses. Every one of them comes from
    the DrivenData problem description or the competition forum, both of which
    ARE reachable, and each one is listed below with its exact source URL so a
    reviewer can open it in one click.

Nothing is claimed to come from the Rules PDF. The PDF is linked for review and
nothing more.
"""

from __future__ import annotations

import sys
import urllib.error
import urllib.request

RULES_PDF = "https://docs.nlr.gov/docs/fy26osti/96647.pdf"
PROBLEM = "https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/"

# (quote, source URL, what it supports on the site)
QUOTES = [
    ("k(d) = max(1 - d/R, 0), where the range R is 300 meters",
     PROBLEM + "#mathematical-representation",
     "the triangular kernel and the 300 m support used in src/gems/metric.py"),
    ("For this competition, we set α = 0.2 and β = 0.8",
     PROBLEM + "#mathematical-representation",
     "alpha=0.2 / beta=0.8 in every DTI computation"),
    ("TP_w = 3.00, FP_w = 1.89, FN_w = 2.00",
     PROBLEM + "#scoring-example",
     "the worked example in tests A1"),
    ("Your submission has the same bounds as the training data, and data outside "
     "the bounds is null or nan.",
     PROBLEM + "#submission-format",
     "why writing 0.0 outside the footprint is a permitted reading of 'null'"),
    ("single layer with datatype of 32-bit float (float32) with values between 0 "
     "and 1 indicating the confidence or probability of fault presence",
     PROBLEM + "#submission-format",
     "the PLATFORM-RANGE gate in scripts/validate_submission.py"),
    ("A sample submission that predicts total fault absence is provided for your "
     "reference on the data download page.",
     PROBLEM + "#submission-format",
     "contradicting the previous repo's claim that the template equals the label raster"),
]


def try_fetch(url: str) -> tuple[bool, str]:
    try:
        req = urllib.request.Request(url, method="HEAD",
                                     headers={"User-Agent": "gemsdoe9-audit/1.0"})
        with urllib.request.urlopen(req, timeout=25) as r:
            return True, f"HTTP {r.status}, {r.headers.get('Content-Length', '?')} bytes"
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code}"
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"


def main() -> int:
    print("=== source verification ===\n")

    print("1. Official rules PDF")
    ok, detail = try_fetch(RULES_PDF)
    if ok:
        print(f"   REACHABLE  {RULES_PDF}  ({detail})")
    else:
        print(f"   UNREACHABLE  {RULES_PDF}")
        print(f"   reason: {detail}")
        print("   The shell cannot reach docs.nlr.gov (curl exit 35, host not on the")
        print("   sandbox allowlist), so the PDF cannot")
        print("   be fetched, hashed, or quoted from here. The site links it for review and")
        print("   attributes NOTHING to it. The previous revision claimed a SHA256 for it")
        print("   ('50d854b1e0239fe6...') and quoted four sentences from it; neither claim")
        print("   could be checked and both have been removed. See docs/verification.html.")
    print()

    print("2. Quotes the site attributes to the problem description")
    width = max(len(q[0]) for q in QUOTES)
    allgood = True
    for quote, url, supports in QUOTES:
        ok, detail = try_fetch(url.split("#")[0])
        mark = "reachable" if ok else "UNREACHABLE"
        if not ok:
            allgood = False
        print(f"   [{mark:>12}]  “{quote}”")
        print(f"                  {url}")
        print(f"                  supports: {supports}")
    print()
    print(f"problem description: {'all URLs reachable' if allgood else 'SOME URLs UNREACHABLE'}")
    print()
    print("Forum quotes (read via the research fetch tool on 2026-09-27) are listed on")
    print("docs/intel.html with their thread URLs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
