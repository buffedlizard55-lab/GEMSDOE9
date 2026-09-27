#!/usr/bin/env python3
"""
audit_layer_usage.py — which official layers does this repo actually read?

    python scripts/audit_layer_usage.py

The layer list is transcribed from the official problem description,
https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/#provided-features
(read 2026-09-27).  The "read by" column is computed by scanning the actual
`data.band(...)` calls in src/gems/, so the table cannot drift from the code.

Exit code 0 always; this is a report, not a gate.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# Official list, verbatim order, with the substring each is resolved by.
LAYERS = [
    ("surface conductivity",                        ("surface_conductivity", "conductivity")),
    ("depth to conductive base surface",            ("depth_to_conductive_base", "conductive_base")),
    ("detrended elevation",                         ("detrended_elevation", "elevation")),
    ("slope of detrended elevation",                ("slope of detrended", "detrended_elevation_slope")),
    ("dilatation rate",                             ("dilatation_rate", "dilatation")),
    ("shear strain rate",                           ("shear_strain_rate", "shear")),
    ("second invariant of the strain rate tensor",  ("second_invariant",)),
    ("isostatic gravity anomaly",                   ("isostatic_gravity", "gravity_anomaly", "gravity")),
    ("slope of the isostatic gravity anomaly",      ("isostatic_gravity_slope", "gravity_slope")),
    ("reduced-to-pole magnetic anomaly",            ("reduced_to_pole", "magnetic_anomaly", "magnetic")),
    ("total magnetic intensity",                    ("total_magnetic_intensity", "magnetic_intensity")),
    ("vertical slope of total magnetic intensity",  ("vertical_slope", "vertical slope")),
    ("horizontal slope of total magnetic intensity", ("horizontal_slope", "horizontal slope")),
    ("top-of-crustal magnetic source depth",        ("top_of_crustal", "source_depth")),
    ("density of earthquakes",                      ("earthquake_density", "earthquake", "seismic")),
]

SRC_BASE = ["src/gems/features.py"]                      # the original G-1..G-5
SRC_NEW = ["src/gems/features.py", "src/gems/hypotheses.py", "src/gems/pipeline.py"]
SRC = SRC_NEW


def band_calls(files=None) -> list[tuple[str, int, str]]:
    """Every `.band(...)` call, as (file, line, raw args).

    Scanned over the whole file text with DOTALL, not line by line: some of
    these calls wrap onto a second line, and a per-line regex silently misses
    them -- which made this audit report a layer as unread that a detector was
    in fact reading.
    """
    pat = re.compile(r"\.band\((.*?)\)", re.DOTALL)
    out = []
    for rel in (files or SRC):
        p = REPO / rel
        if not p.exists():
            continue
        text = p.read_text()
        for m in pat.finditer(text):
            lineno = text.count("\n", 0, m.start()) + 1
            out.append((rel, lineno, " ".join(m.group(1).split())))
    return out


def readers_for(patterns, files=None) -> set[str]:
    """Which detector function does each `.band(patterns)` call sit inside?"""
    hits = set()
    for rel, lineno, args in band_calls(files):
        text = (REPO / rel).read_text().splitlines()
        # walk back to the enclosing def
        fn = "?"
        for j in range(lineno - 1, -1, -1):
            m = re.match(r"\s*def\s+(\w+)", text[j])
            if m:
                fn = m.group(1)
                break
        for pat in patterns:
            if pat.lower().replace(" ", "_") in args.lower().replace(" ", "_"):
                hits.add(f"{fn} ({Path(rel).name}:{lineno})")
                break
    return hits


def main() -> int:
    print("LAYER USAGE AUDIT")
    print(f"layer list source: https://www.drivendata.org/competitions/306/"
          f"competition-doe-gems/page/967/#provided-features  (read 2026-09-27)")
    print(f"scanned: {', '.join(SRC_NEW)}\n")

    all_calls = band_calls()
    print(f"{len(all_calls)} `.band(...)` calls found:\n")
    for rel, lineno, args in all_calls:
        print(f"    {rel}:{lineno}  .band({args})")

    print(f"\n{'official layer':46} {'G-1..G-5 only':>14} {'+ H-1..H-5'}")
    print("-" * 92)
    unused_base, unused_now = [], []
    for name, pats in LAYERS:
        b = {h for h in readers_for(pats, SRC_BASE) if "features.py" in h}
        c = {h for h in readers_for(pats, SRC_NEW)
             if "features.py" in h or "hypotheses.py" in h}
        bs = "yes" if b else "-- NONE --"
        cs = sorted(c)[0].split(" (")[0] if c else "-- NONE --"
        if not b:
            unused_base.append(name)
        if not c:
            unused_now.append(name)
        print(f"{name:46} {bs:>14} {cs}")
    print("-" * 92)
    print(f"UNUSED by G-1..G-5 (the original detectors): "
          f"{len(unused_base)} of {len(LAYERS)}")
    for nm in unused_base:
        print(f"    - {nm}")
    print(f"\nUNUSED after adding H-1..H-5: {len(unused_now)} of {len(LAYERS)}")
    for nm in unused_now:
        print(f"    - {nm}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
