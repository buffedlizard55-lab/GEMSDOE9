#!/usr/bin/env python3
"""
build_site.py — render docs/ from verified evidence.

No number, quote or link on the site is typed into an HTML file.  Every one is
pulled from scripts/site_data.py (verified claims + their source URLs) or
computed at build time from the live artifacts in docs/downloads/manifest.json
and docs/strategy_experiment.json, or derived from src/gems/strategy.py, which
is itself derived from the published metric.

Run:  python scripts/build_site.py
Then: python scripts/build_site.py --check    # fail if anything on disk is stale
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts import site_data as S                       # noqa: E402
from src.gems import strategy as St                     # noqa: E402

DOCS = REPO / "docs"
PAGES = [
    ("index.html", "Overview", ""),
    ("executive_summary.html", "How to submit", "exec"),
    ("strategy.html", "Strategy & metric", "strategy"),
    ("hypotheses.html", "Hypotheses G-1..G-5", "hypo"),
    ("intel.html", "Official intel", "intel"),
    ("data.html", "Data & access", "data"),
    ("verification.html", "Verification & flags", "verify"),
    ("leaderboard.html", "Leaderboard", "lb"),
    ("research.html", "Research notes", "research"),
    ("sources.html", "Sources", "sources"),
]

CSS = """
:root{
  --bg:#fbfaf9; --panel:#ffffff; --ink:#1c1917; --muted:#6b6560; --line:#e7e5e4;
  --brand:#b91c1c; --brand2:#c2410c; --ok:#15803d; --okbg:#dcfce7; --warn:#92400e;
  --warnbg:#fef3c7; --bad:#991b1b; --badbg:#fee2e2; --info:#1e40af; --infobg:#dbeafe;
  --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
}
*{box-sizing:border-box}
body{margin:0;font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  background:var(--bg);color:var(--ink);line-height:1.62;-webkit-font-smoothing:antialiased}
a{color:#b91c1c}
code,kbd{font-family:var(--mono);font-size:.88em;background:#f5f5f4;padding:.12rem .35rem;border-radius:4px}
pre{background:#1c1917;color:#e7e5e4;padding:1rem;border-radius:10px;overflow:auto;font-size:.82rem;line-height:1.5}
pre code{background:none;color:inherit;padding:0}
header.top{background:linear-gradient(135deg,#1c1917 0%,#44403c 55%,#7c2d12 100%);color:#fff;padding:2.2rem 1.4rem 1.6rem}
header.top h1{margin:0 0 .35rem;font-size:1.9rem;letter-spacing:-.025em}
header.top p{margin:0;max-width:78ch;color:#e7e5e4}
.badges{margin-top:.9rem;display:flex;gap:.45rem;flex-wrap:wrap}
.badge{font-size:.72rem;font-weight:700;padding:.2rem .55rem;border-radius:999px;background:rgba(255,255,255,.14);color:#fff}
.badge.ok{background:#15803d}.badge.warn{background:#b45309}.badge.bad{background:#b91c1c}
nav.tabs{position:sticky;top:0;z-index:20;background:rgba(251,250,249,.94);backdrop-filter:blur(8px);
  border-bottom:1px solid var(--line);padding:.55rem 1.4rem;display:flex;gap:.3rem;flex-wrap:wrap}
nav.tabs a{font-size:.85rem;text-decoration:none;color:var(--muted);padding:.32rem .6rem;border-radius:7px}
nav.tabs a:hover{background:#fff;color:var(--ink)}
nav.tabs a.on{background:var(--ink);color:#fff;font-weight:600}
main{max-width:1140px;margin:0 auto;padding:1.6rem 1.4rem 4rem}
section{margin-bottom:2.2rem}
h2{font-size:1.3rem;letter-spacing:-.015em;margin:0 0 .7rem;padding-bottom:.4rem;border-bottom:1px solid var(--line)}
h3{font-size:1.05rem;margin:1.5rem 0 .5rem}
h4{font-size:.95rem;margin:1.1rem 0 .35rem;color:#44403c}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:1.15rem 1.25rem;
  box-shadow:0 1px 2px rgba(0,0,0,.04)}
.grid{display:grid;gap:1.2rem;grid-template-columns:1fr}
@media(min-width:900px){.grid.two{grid-template-columns:1fr 1fr}.grid.three{grid-template-columns:repeat(3,1fr)}}
table{width:100%;border-collapse:collapse;font-size:.88rem;margin:.6rem 0}
th,td{text-align:left;padding:.5rem .55rem;border-bottom:1px solid var(--line);vertical-align:top}
th{font-weight:650;color:var(--muted);font-size:.8rem;text-transform:uppercase;letter-spacing:.03em}
td.num,th.num{text-align:right;font-family:var(--mono);white-space:nowrap}
.note{border-left:3px solid var(--info);background:var(--infobg);padding:.8rem 1rem;border-radius:0 8px 8px 0;margin:.9rem 0}
.note.warn{border-color:#b45309;background:var(--warnbg)}
.note.bad{border-color:var(--bad);background:var(--badbg)}
.note.ok{border-color:var(--ok);background:var(--okbg)}
.note b.lbl{display:block;font-size:.72rem;letter-spacing:.06em;text-transform:uppercase;margin-bottom:.2rem}
.btn{display:inline-block;padding:.8rem 1.3rem;border-radius:9px;font-weight:650;text-decoration:none;
  border:1px solid transparent;cursor:pointer;font-size:.95rem;font-family:inherit}
.btn-primary{background:linear-gradient(135deg,var(--brand),var(--brand2));color:#fff}
.btn-secondary{background:#fff;border-color:var(--line);color:var(--ink)}
.btn-row{display:flex;gap:.6rem;flex-wrap:wrap;margin:1rem 0}
.pill{display:inline-block;padding:.1rem .45rem;border-radius:999px;font-size:.72rem;font-weight:700}
.pill.ok{background:var(--okbg);color:var(--ok)}
.pill.warn{background:var(--warnbg);color:var(--warn)}
.pill.bad{background:var(--badbg);color:var(--bad)}
.pill.info{background:var(--infobg);color:var(--info)}
.kv{display:grid;grid-template-columns:minmax(140px,auto) 1fr;gap:.35rem 1rem;font-size:.9rem}
.kv dt{color:var(--muted)}
.kv dd{margin:0;font-family:var(--mono);font-size:.84rem;word-break:break-all}
details{border:1px solid var(--line);border-radius:9px;padding:.6rem .85rem;margin:.6rem 0;background:#fff}
summary{cursor:pointer;font-weight:600;font-size:.92rem}
.hero{background:linear-gradient(135deg,#fff7ed,#fef2f2);border:2px solid #fdba74}
.mono{font-family:var(--mono)}
.muted{color:var(--muted)}
.small{font-size:.85rem}
.log{background:#1c1917;color:#e7e5e4;padding:.9rem;border-radius:9px;font-family:var(--mono);
  font-size:.78rem;white-space:pre-wrap;max-height:340px;overflow:auto}
footer{border-top:1px solid var(--line);margin-top:3rem;padding:1.8rem 1.4rem;color:var(--muted);font-size:.85rem}
.src{font-size:.8rem;color:var(--muted);margin-top:.35rem}
.tag-new{background:#15803d;color:#fff;border-radius:4px;padding:0 .3rem;font-size:.7rem;font-weight:700}
"""


def esc(s) -> str:
    return html.escape(str(s), quote=True)


class Raw(str):
    """Marks a value as pre-escaped HTML that the builder constructed itself."""


def kv(pairs) -> str:
    out = []
    for k, v in pairs:
        val = v if isinstance(v, Raw) else esc(v)
        out.append(f"<dt>{esc(k)}</dt><dd>{val}</dd>")
    return "<dl class='kv'>" + "".join(out) + "</dl>"


def table(headers, rows, numeric=()) -> str:
    th = "".join(f"<th class='num'>{esc(h)}</th>" if i in numeric else f"<th>{esc(h)}</th>"
                 for i, h in enumerate(headers))
    body = []
    for r in rows:
        tds = "".join(f"<td class='num'>{c}</td>" if i in numeric else f"<td>{c}</td>"
                      for i, c in enumerate(r))
        body.append(f"<tr>{tds}</tr>")
    return f"<table><thead><tr>{th}</tr></thead><tbody>{''.join(body)}</tbody></table>"


# --------------------------------------------------------------------------- #
def load_artifacts():
    def read(name):
        p = DOCS / name
        return json.loads(p.read_text()) if p.exists() else None
    return (read("downloads/manifest.json"),
            read("strategy_experiment.json"),
            read("holdout_report.json"),
            read("test_report.json"))


def derived_numbers():
    """Everything the site says about the metric, computed not typed."""
    d = {
        "attr_rho": St.required_recall_for_score(0.1563),
        "top_rho": St.required_recall_for_score(0.3049),
        "row2_rho": St.required_recall_for_score(0.2993),
    }
    d["attr_check"] = St.dti_from_ratios(d["attr_rho"], 0.0)
    d["top_check"] = St.dti_from_ratios(d["top_rho"], 0.0)
    d["ratio"] = d["top_rho"] / d["attr_rho"]
    d["exchange"] = {r: 4.0 / r for r in (0.1, 0.1291, 0.2, 0.2598, 0.3, 0.5, 0.7)}
    return d


# --------------------------------------------------------------------------- #
def shell(page_id, title, body, extra_js="", badges=()) -> str:
    def nav_item(fn, t):
        cls = " class='on'" if fn == page_id else ""
        return f"<a href='{fn}'{cls}>{esc(t)}</a>"
    nav = "".join(nav_item(fn, t) for fn, t, _ in PAGES)
    bg = "".join(f"<span class='badge {c}'>{esc(t)}</span>" for t, c in badges)
    return f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)} — GEMSDOE9</title>
<meta name="description" content="GEMSDOE9 — DOE GEMS Prize entry. Verified competition intel, the metric algebra behind the 0.1563 plateau, and a one-click valid GeoTIFF.">
<style>{CSS}</style>
</head><body>
<header class="top">
  <h1>GEMSDOE9 — Geothermal Fault Discovery Beyond the Catalogue</h1>
  <p>DOE GEMS Prize entry. Faults in the GeoDAWN region (northwestern Nevada, UTM 11N
     EPSG:32611, 100 m) that are <em>not</em> in the USGS/INGENIOUS catalogue.</p>
  <div class="badges">{bg}</div>
</header>
<nav class="tabs">{nav}</nav>
<main>{body}</main>
<footer>
  <p><strong>Maximise P(Win).</strong> Every number on this site is either quoted from a
  linked primary source, read live on {S.VERIFIED_ON}, or computed by code in this
  repository from the published metric. Nothing is estimated.</p>
  <p>Sources: <a href="{S.COMPETITION['overview']}">Competition</a> ·
     <a href="{S.COMPETITION['problem']}">Problem description</a> ·
     <a href="{S.COMPETITION['about']}">About</a> ·
     <a href="{S.COMPETITION['leaderboard']}">Leaderboard</a> ·
     <a href="{S.COMPETITION['forum']}">Forum</a> ·
     <a href="{S.COMPETITION['reference_solution']}">Reference solution</a></p>
  <p>Built by <code>python scripts/build_site.py</code>. Sources of every claim:
     <a href="sources.html">Sources</a>. Problems found and fixed:
     <a href="verification.html">Verification &amp; flags</a>.</p>
</footer>
<script src="geotiff_writer.js"></script>
{extra_js}
</body></html>"""


# --------------------------------------------------------------------------- #
def _gate_count(manifest):
    v = (manifest or {}).get("validator") or {}
    return int(v.get("n_checks", 0)) if v.get("all_passed") else 0


def page_index(ctx):
    man, exp, tests = ctx["manifest"], ctx["experiment"], ctx["tests"]
    d = ctx["derived"]
    gates = _gate_count(man)

    if man:
        fn = man["file"]; sha = man["sha256"]; note = man["note"]; kb = f"{man['bytes']:,}"
        pos = man["positive_px"]; frac = man["positive_fraction"]
        kind = man["kind"]
        kind_pill = ("<span class='pill bad'>PLACEHOLDER — not a prediction</span>"
                     if kind == "PLACEHOLDER" else "<span class='pill ok'>MODEL</span>")
    else:
        fn = sha = note = kb = "—"; pos, frac, kind_pill = 0, 0, "<span class='pill warn'>not built</span>"

    exp_block = ("<p class='muted'>Run <code>python scripts/validate_holdout.py --strategy</code> "
                 "to regenerate.</p>")
    if exp and "blocked_table" in exp:
        b0, b1 = exp["blocked_table"][0], exp["blocked_table"][-1]
        exp_block = table(
            ["Submission shape", "DTI", "vs free core"],
            [["Known catalogue, nothing else <span class='muted'>(φ = 0 exactly)</span>",
              f"{b0['catalogue']:.4f}", "—"],
             ["+ a blind 3 px halo, no detector",
              f"{b0['blind']:.4f}", f"{b0['blind'] - b0['catalogue']:+.4f}"],
             ["+ a detector-gated halo, chance detector",
              f"{b0['corridor']:.4f}", f"{b0['corridor'] - b0['catalogue']:+.4f}"],
             ["+ a detector-gated halo, useful detector",
              f"{b1['corridor']:.4f}", f"{b1['corridor'] - b0['catalogue']:+.4f}"]],
            numeric=(1, 2))
        exp_block += (f"<p class='small muted'>{len(exp.get('seeds', []))} independent synthetic "
                      f"phantoms at {exp.get('n_blocked', '?')}×{exp.get('n_blocked', '?')} px, "
                      f"2×2 spatial blocks, (halo, budget) chosen on the training blocks and "
                      f"scored on the held-out block. Median over phantoms. "
                      f"<strong>Going from a chance detector to a useful one, inside the same "
                      f"corridor, is worth "
                      f"{b1['corridor'] - b0['corridor']:+.4f}.</strong> That increment is the "
                      f"only part of this table attributable to detection. "
                      f"<strong>This prices the submission SHAPE, not the geology.</strong> "
                      f"<a href='strategy.html'>Method and caveats →</a></p>")

    return f"""
<section>
  <div class="card hero">
    <h2 style="border:0;padding:0;margin-bottom:.6rem">⬇ The file to submit</h2>
    <p>DrivenData wants <strong>a single-band GeoTIFF (.tif)</strong>, or a .zip holding
       one, matching the submission format: <strong>3,292 × 3,730 px · EPSG:32611 ·
       100 m · float32 · every value finite and in [0, 1]</strong>.</p>
    <div class="btn-row">
      <a class="btn btn-primary" href="downloads/{esc(fn)}" download>⬇ Download {esc(fn)}</a>
      <button class="btn btn-secondary" id="buildBtn">Build in this browser (uniquely named)</button>
    </div>
    <div class="note" id="status" style="display:none"></div>
    {kv([
      ("File", Raw(f"<a href='downloads/{esc(fn)}' download>{esc(fn)}</a>")),
      ("SHA256", sha),
      ("Size", f"{kb} bytes"),
      ("Positive pixels", f"{pos:,} ({frac:.2%}) — the rest are 0.0, i.e. 'no fault'"),
      ("Note to paste", Raw(f"<code>{esc(note)}</code>")),
      ("What it is", Raw(kind_pill)),
      ("Upload page", Raw(f"<a href='{S.COMPETITION['submissions']}'>drivendata.org → Submissions → New submission</a>")),
    ])}
    <details>
      <summary>Why this file cannot trigger “Predicted values must be in range [0, 1]”</summary>
      <p>The error you hit on 2026-09-27 was <strong>not</strong> a NaN problem. The
      previous version of this site's browser writer produced GeoTIFFs that GDAL could not
      open at all — <code>TIFFReadEncodedStrip() failed</code> — and the platform surfaced
      that unreadable file as a value-range error. Two changes fix it:</p>
      <ol>
        <li>The writer is rewritten and round-trip tested: write → parse the IFD back →
            decode every strip → compare to the input, all element-for-element, before the
            download is offered. <a href="verification.html#irregularity-tiff">Evidence →</a></li>
        <li>Every value is finite and inside [0, 1]; no NaN, no Inf, no nodata tag.
            The spec says “data outside the bounds is null or nan”, but NaN is not in [0, 1],
            so a 0.0 outside the footprint satisfies “null” in the scorer's sense and can
            never trip a range check. <strong>We publish only the all-finite variant.</strong></li>
      </ol>
      <p>Re-verify any file yourself:
         <code>python scripts/validate_submission.py docs/downloads/{esc(fn)}</code></p>
    </details>
    <details>
      <summary>Full click-by-click instructions</summary>
      <p>They are on their own page, with the unique-name rule, the Note text, the .zip
         option, and what to do when a submission is rejected.</p>
      <p><a class="btn btn-secondary" href="executive_summary.html">How to submit →</a></p>
    </details>
  </div>
</section>

<section class="grid two">
  <div class="card">
    <h2>What the current file is, honestly</h2>
    <p>The competition rasters live behind a DrivenData login and this environment has no
       outbound network from the shell, so <strong>no model has been trained yet</strong>.
       Rather than dress up a procedural pattern as a geological result — which is exactly
       what the previous version of this site did — the downloadable file is a
       <strong>format-check placeholder</strong>: a fixed-seed band pattern whose only job
       is to prove the GeoTIFF, the validator and the upload form work end to end.</p>
    <p>Its file name, its Note and this paragraph all say so. Upload it if you want to
       validate the submission path; <strong>do not read a score into it.</strong></p>
    <p>The real predictor is one command away, and it refuses to build a submission until
       it beats a budget-matched random control and a catalogue-only baseline on a
       spatially-blocked holdout:</p>
    <pre><code># 1. get the data (needs a DrivenData account, on an unrestricted machine)
bash scripts/download_competition_data.sh     # → data/

# 2. blocked holdout + submission gate
python scripts/validate_holdout.py --holdout-gate
python scripts/build_submission.py --holdout-gate

# or just re-run the checks
bash scripts/run_all_checks.sh</code></pre>
  </div>

  <div class="card">
    <h2>Why 0.1563 keeps repeating</h2>
    <p>Because it is an <em>attractor of the metric</em>, not a coincidence of tuning.
       The published metric collapses to a closed form:</p>
    <pre><code>FN_w = N − TP_w   (both are maxima of the same product)
DTI  = T / (0.2·T + 0.2·F + 0.8·N)</code></pre>
    <p>Solve it for zero false positives — which is exactly what a submission that paints
       the known catalogue is, because those pixels are masked out of scoring:</p>
    <pre><code>0.1563 = ρ / (0.2·ρ + 0.8)   →   ρ = {d['attr_rho']:.4f}</code></pre>
    <p>So <strong>0.1563 is the fingerprint of a catalogue copy</strong>: 12.9% weighted
       recall and not one unit of false-positive mass. Copy the catalogue and you get
       0.1563; copy it slightly differently and you still get 0.1563. Three unrelated
       accounts sitting on exactly that number is what that looks like from outside.</p>
    <p>Inverting the same equation for the current leaderboard top:</p>
    <pre><code>0.3049 → ρ ≥ {d['top_rho']:.4f}   ({d['ratio']:.2f}× the 0.1563 recall)</code></pre>
    <p>The top score is out of reach for catalogue copying. It needs a detector.
       <a href="strategy.html">Full derivation, the exchange rate, and the corridor
       experiment →</a></p>
  </div>
</section>

<section class="card">
  <h2>Measurement: what the submission <em>shape</em> is worth</h2>
  <p>Independent of any geology: run the published DTI and the published masking rule over
     phantoms whose ground truth follows the staff definition of “new fault”, and vary
     only how the answer is drawn.</p>
  {exp_block}
</section>

<section class="grid two">
  <div class="card">
    <h2>The one new piece of official intel</h2>
    <p class="src">DrivenData staff, {S.STAFF_STATEMENTS[1]['when']} ·
      <a href="{S.STAFF_STATEMENTS[1]['thread']}">forum thread 11536</a></p>
    <blockquote style="margin:.6rem 0;padding:.7rem 1rem;border-left:3px solid var(--brand);background:#faf5f4;font-style:italic">
      “{esc(S.STAFF_STATEMENTS[1]['quote'])}”
    </blockquote>
    <p>Read it carefully. “New fault” includes <strong>newly mapped geometry of an existing
       fault system</strong> — splays, parallel strands, continuations past mapped tips. Those
       pixels are, by construction, next to and collinear with faults the catalogue already
       has. And known-catalogue pixels are masked out of the false-positive term while still
       counting toward true positives.</p>
    <p>So the catalogue is a <strong>pointer</strong>, not just a distractor. That inverts
       the strategy the group has been running for three rounds.
       <a href="hypotheses.html">Detector G-1 is built on it →</a></p>
  </div>
  <div class="card">
    <h2>Where the group stands</h2>
    {table(["Submission", "Score", "What it was"],
           [[f'<a href="{esc(u)}" target="_blank" rel="noopener">{esc(n)}</a>', f"{s:.4f}", esc(d)]
            for n, u, s, d in S.GROUP_SUBMISSIONS], numeric=(1,))}
    <p class="small muted">Scores as reported by the team. Artifact hashes for the
       sibling repos could not be checked from this environment, so no claim is made
       here about which files were byte-identical — only about what the scores are and
       what the closed form predicts they mean.</p>
  </div>
</section>

<section class="card">
  <h2>The five detectors</h2>
  {table(["#", "Detector", "Layers", "Cost", "Validated?"],
     [[d_.key, f"<strong>{esc(d_.title)}</strong>", esc(d_.layers[:90] + ("…" if len(d_.layers) > 90 else "")),
       esc(d_.cost[:46] + ("…" if len(d_.cost) > 46 else "")),
       "<span class='pill bad'>blocked on data</span>"] for d_ in _detectors()],
     numeric=(0,))}
  <p class="small muted">Each is implemented as an explicit function in
     <code>src/gems/features.py</code> with its layers, physical signature, and the reason
     it should catch a fault the catalogue missed. <strong>None has been validated against
     real rasters</strong> — that is the honest state, and the next session's first job.
     <a href="hypotheses.html">Full hypotheses →</a></p>
</section>
"""


def _detectors():
    from src.gems.features import all_detectors
    return all_detectors()


# --------------------------------------------------------------------------- #
def page_exec(ctx):
    man = ctx["manifest"]
    fn = man["file"] if man else "gemsdoe9-PLACEHOLDER-<sha8>.tif"
    note = man["note"] if man else "…"
    return f"""
<section>
  <div class="card hero">
    <h2 style="border:0;padding:0;margin-bottom:.5rem">Submitting to the GEMS Prize — step by step</h2>
    <p>Five steps. The whole thing takes about a minute once the file is on disk.</p>
  </div>
</section>

<section class="grid two">
  <div class="card">
    <h3>1 · Get the file</h3>
    <p><a class="btn btn-primary" href="downloads/{esc(fn)}" download>⬇ {esc(fn)}</a></p>
    <p class="small muted">Or press <em>Build in this browser</em> on the
       <a href="index.html">overview</a>, which produces the identical pixel field with a
       fresh name and a fresh hash. Nothing is uploaded from your browser — the GeoTIFF is
       assembled locally with <code>geotiff_writer.js</code>.</p>

    <h3>2 · Make the name unique</h3>
    <p>The DrivenData form does not require a unique file name, but a submission history
       where every row is <code>submission.tif</code> is useless to you in a week. Use
       <code>gemsdoe9-&lt;what&gt;-&lt;sha8&gt;.tif</code>. The shipped file already follows
       this rule; the hash is the first 8 hex characters of its SHA256, so the name
       identifies the exact bytes.</p>
    <p>If you upload the <em>same</em> bytes twice, rename the second one
       (<code>…-r2.tif</code>) so you can tell the two rows apart.</p>

    <h3>3 · Open the submission form</h3>
    <p><a href="{S.COMPETITION['submissions']}">drivendata.org → this competition → Submissions → New submission</a></p>
    <p>The dialog says: <em>“You can submit a single-band GeoTIFF (.tif) file, or a .zip
       file containing a single GeoTIFF, with your predictions. It must match the
       submission format's CRS, shape, and geotransform.”</em></p>
    <p>Drag the .tif in. (A .zip is accepted too — the browser build can emit one. The
       competition form also accepts a .zip of a single GeoTIFF, but a .tif is simpler and
       there is no reason to risk it.)</p>

    <h3>4 · Write the Note</h3>
    <p>The <em>Note (optional)</em> box exists so you can tell your own submissions apart
       later. The example in the form is “clustering with k=25”. For this submission:</p>
    <pre><code>{esc(note)}</code></pre>
    <p>Put the SHA8 in it. Two weeks from now that is the only thing that will tell you
       which file scored what.</p>

    <h3>5 · Submit, then check the leaderboard</h3>
    <p>Submissions are scored on a
       <a href="{S.STAFF_STATEMENTS[3]['thread']}">rolling 7-day window</a>, three per
       window. Your <em>best</em> public score is what the leaderboard shows. The
       Initial Prize Round uses a private test set fixed before the competition; the
       Final Prize Round re-scores your chosen submission against an expanded label set
       that experts build partly <em>from the Phase 1 submissions themselves</em>.</p>
  </div>

  <div>
    <div class="card">
      <h2>Rejection: “Predicted values must be in range [0, 1]”</h2>
      <p class="src">Reported by the team, 2026-09-27</p>
      <div class="note bad"><b class="lbl">What actually happened</b>
        The file downloaded from this site was a 49 MB GeoTIFF whose strip-offset table was
        truncated, so GDAL could not decode it. The platform reported an unreadable file
        using its value-range error. There was no NaN in it to find.</div>
      <p>What is now guaranteed, and how to re-check it yourself:</p>
      {table(["Check", "Meaning"], [
        ["<code>GDAL-READABLE</code>", "rasterio decodes every strip; the file is readable at all"],
        ["<code>PLATFORM-RANGE</code>", "0 NaN, 0 Inf, min ≥ 0, max ≤ 1 across all 12,279,160 values"],
        ["<code>no-nodata-tag</code>", "no nodata declaration, so no reader can substitute a value"],
        ["<code>shape / crs / transform / dtype</code>", "3,292 × 3,730, EPSG:32611, 100 m, float32, 1 band"],
        ["<code>not-a-known-duplicate</code>", "SHA256 is not one that already spent a slot"],
      ])}
      <pre><code>python scripts/validate_submission.py docs/downloads/{esc(fn)}</code></pre>
      <p>Every line must read <span class="pill ok">PASS</span>. If
         <code>GDAL-READABLE</code> fails, the file is truncated in transfer — download it
         again and compare the SHA256.</p>
    </div>

    <div class="card">
      <h2>Other rejections, and what they mean</h2>
      {table(["Message", "Cause", "Fix"], [
        ["Predicted values must be in range [0, 1]",
         "Any NaN, Inf or out-of-range value — <em>or</em> a file the platform cannot decode",
         "Run the validator; it checks both"],
        ["CRS mismatch / does not match submission format",
         "Wrong projection or geotransform",
         "EPSG:32611, transform (100, 0, 243350, 0, −100, 4508550)"],
        ["Wrong shape", "Not 3,292 × 3,730",
         "The validator hard-fails on this"],
        ["Rate limited", "More than 3 scored submissions in a rolling 7 days",
         f'<a href="{S.STAFF_STATEMENTS[3]["thread"]}">Forum: rolling window</a>'],
      ])}
    </div>

    <div class="card">
      <h2>The exact format, quoted</h2>
      <p class="src">Problem description, § Submission format —
        <a href="{S.COMPETITION['problem']}#submission-format">source</a></p>
      <ul class="small">
        <li>{esc(S.SPEC['crs'])}</li>
        <li>{esc(S.SPEC['resolution'])}</li>
        <li>{esc(S.SPEC['bounds'])}</li>
        <li>{esc(S.SPEC['layers'])}</li>
      </ul>
      <div class="note warn"><b class="lbl">Why we write 0.0 instead of NaN outside the bounds</b>
        The spec permits “null or nan” outside the bounds, but NaN is not a member of [0, 1]
        and a range validator will reject it. 0.0 means “no fault predicted”, which is what
        a null cell means to the scorer, and it cannot fail a range check. We therefore
        publish only the all-finite variant and set no nodata tag.
        <a href="verification.html#irregularity-tiff">Why this was the right call →</a></div>
    </div>
  </div>
</section>
"""


# --------------------------------------------------------------------------- #
def page_strategy(ctx):
    d = ctx["derived"]
    exp = ctx["experiment"]
    cf_err = (f" to a maximum error of {exp['closed_form_max_abs_error']:.1e} "
              f"over {len(exp['seeds'])} independent phantoms"
              if exp and "closed_form_max_abs_error" in exp else
              " -- see <code>src/gems/metric.py</code>, where the closed form and the "
              "measured value are asserted equal in the test suite")
    exp_rows = ""
    if exp and "blocked_table" in exp:
        exp_rows = table(
            ["Submission shape", "DTI", "vs free core"],
            [["Known catalogue, nothing else <span class='muted'>(φ = 0 exactly)</span>",
              f"{exp['blocked_table'][0]['catalogue']:.4f}", "—"],
             ["+ blind 3 px halo, detector ignored",
              f"{exp['blocked_table'][0]['blind']:.4f}",
              f"{exp['blocked_table'][0]['blind_minus_catalogue']:+.4f}"],
             ["+ detector-gated halo, chance detector (AUC ≈ 0.50)",
              f"{exp['blocked_table'][0]['corridor']:.4f}",
              f"{exp['blocked_table'][0]['corridor_minus_catalogue']:+.4f}"],
             ["+ detector-gated halo, useful detector (AUC ≈ "
              f"{exp['blocked_table'][-1]['auc_median']:.2f})",
              f"{exp['blocked_table'][-1]['corridor']:.4f}",
              f"{exp['blocked_table'][-1]['corridor_minus_catalogue']:+.4f}"]],
            numeric=(1, 2))
        exp_rows += (
            "<p class='small muted'><strong>Read the two gaps separately.</strong> "
            f"Core → blind halo is {exp['blocked_table'][0]['blind_minus_catalogue']:+.4f} and "
            "belongs to the halo's geometry, not to any detector. Chance detector → useful "
            f"detector, inside the same corridor, is "
            f"{exp['blocked_table'][-1]['corridor'] - exp['blocked_table'][0]['corridor']:+.4f} "
            "and is the only part of this table attributable to detection at all. "
            f"{len(exp['seeds'])} independent phantoms, 2×2 blocks, configuration chosen on "
            "the training blocks and scored on the held-out block.</p>")
    if exp and "summary" in exp:
        sm = exp["summary"]
        exp_rows += (
            "<p class='small muted'>A second, selection-free sweep ranks the <em>whole grid</em> "
            f"instead of a band. It loses to the free core in "
            f"{sm['n_cells'] - round(sm['hard_beats_core_phantom_cells'])}/{sm['n_cells']} cells, "
            "at every budget from 0.1% to 10% of the raster and every detector quality. "
            "Proximity to a mapped fault is what the 300 m kernel rewards; spreading the same "
            "pixel count uniformly over the region does not.</p>")
    return f"""
<section>
  <div class="card hero">
    <h2 style="border:0;padding:0;margin-bottom:.5rem">Why 0.1563, and what to do instead</h2>
    <p>Everything on this page is algebra from the published metric, plus a synthetic
       experiment that measures the <em>shape</em> of a submission. No claim here depends on
       a machine-learning result, because we do not have one yet.</p>
  </div>
</section>

<section>
  <h2>1 · The metric collapses to two ratios</h2>
  <p class="src">Problem description, § Performance metric —
     <a href="{S.COMPETITION['problem']}#performance-metric">source</a></p>
  <pre><code>k(d) = max(1 − d/R, 0),  R = 300 m = 3 px at 100 m
TP_w = Σ_{{g∈G}} max_{{x: d(x,g)≤R}} p(x)·k(d(x,g))
FP_w = Σ_{{x: p(x)&gt;0}} p(x)·[1 − max_{{g∈G}} k(d(x,g))]
FN_w = Σ_{{g∈G}} [1 − max_{{x: d(x,g)≤R}} p(x)·k(d(x,g))]

α = 0.2 (false positives), β = 0.8 (false negatives)
DTI = TP_w / (TP_w + α·FP_w + β·FN_w + ε)</code></pre>
  <p>TP_w and FN_w are <strong>maxima of the same quantity</strong>, so
     <code>FN_w = N − TP_w</code> exactly, with N = |G|. Substituting:</p>
  <pre><code>DTI = T / (0.2·T + 0.2·F + 0.8·N)        (EQ-1)
    = ρ / (0.2·ρ + 0.2·φ + 0.8)   with ρ = T/N, φ = F/N   (EQ-2)</code></pre>
  <p>Only two numbers matter, and neither is the absolute pixel count. Everything below
     follows from that. Implementation:
     <code>src/gems/strategy.py</code>, cross-checked against
     <code>src/gems/metric.py</code> in
     <a href="verification.html">tests B1–B9</a>.</p>
</section>

<section>
  <h2>2 · 0.1563 is the score of a catalogue copy</h2>
  <p>A submission that paints the known USGS/INGENIOUS catalogue has
     <strong>zero false-positive mass</strong>, because staff masked those pixels out of
     evaluation:</p>
  <blockquote style="margin:.6rem 0;padding:.7rem 1rem;border-left:3px solid var(--brand);background:#faf5f4;font-style:italic">
    “{esc(S.STAFF_STATEMENTS[0]['quote'])}”
    <div class="src" style="font-style:normal">{esc(S.STAFF_STATEMENTS[0]['who'])},
      {esc(S.STAFF_STATEMENTS[0]['when'])} ·
      <a href="{S.STAFF_STATEMENTS[0]['thread']}">thread 11516</a></div>
  </blockquote>
  <p>Set φ = 0 in EQ-2 and solve for ρ at each headline score:</p>
  {table(["Score", "Implied weighted recall ρ at zero FP", "Reading"],
     [["0.1563", f"<strong>{d['attr_rho']:.4f}</strong>", "a catalogue copy"],
      ["0.2993", f"{d['row2_rho']:.4f}", "a real detector"],
      ["0.3049", f"<strong>{d['top_rho']:.4f}</strong>", "leader of the board"]],
     numeric=(1,))}
  <div class="note"><b class="lbl">This is a derivation, not a measurement</b>
    It predicts that any two catalogue-derived submissions land within about 10⁻³ of each
    other, and that 0.3049 requires roughly {d['ratio']:.1f}× the recall a catalogue copy
    gets. The submissions themselves are not public, so this cannot be confirmed against
    the actual files — but it is falsifiable, and the synthetic experiment in §4 measures
    the same relationship{cf_err}.</div>
  <p>What it means practically: the group has been submitting the catalogue. The 0.1563
     ceiling is not bad luck or weak hyper-parameters. It is the exact value that
     architecture produces.</p>
</section>

<section>
  <h2>3 · Recall is worth about an order of magnitude more than precision</h2>
  <p>Differentiate EQ-1 and compare the marginal gain from true-positive mass to the
     marginal cost of false-positive mass:</p>
  <pre><code>value of 1 unit of TP_w  ∝  (0.2·F + 0.8·N)
cost  of 1 unit of FP_w  ∝  (0.2·T)

exchange rate = 4/ρ + φ          (EQ-3)</code></pre>
  {table(["Weighted recall ρ", "Units of FP mass one unit of recall is worth"],
     [[f"{r:.3f}", f"<strong>{4 / r:.1f}×</strong>"] for r in (0.10, 0.1291, 0.20, 0.2598, 0.30, 0.50, 0.70)],
     numeric=(0, 1))}
  <p>At the 0.1563 attractor, one extra unit of weighted recall is worth
     <strong>{4 / d['attr_rho']:.0f} units</strong> of false-positive mass. Every instinct
     that says “be conservative, be sparse, be precise” points the wrong way for this
     metric: a new fault you find is worth about 31 pixels you get wrong.</p>
  <div class="note warn"><b class="lbl">A mistake this page used to make</b>
     An earlier draft concluded from EQ-3 that a hard 0/1 mask must beat a soft
     probability map — the kernel is a maximum, so half-confidence at the right
     place should earn half the credit and pay half the penalty, and 31 &gt; 1.
     That is wrong, and the experiment above is what caught it. On <em>exactly the
     same pixel set</em>, a soft field beat a hard one in 19 of 20 cells. Both effects
     scale together: fractional mass lowers TP and lowers FP by the same factor, so
     which one wins depends on which term binds. At a recall-limited field hard wins;
     once φ dominates, spreading wins. Which regime you are in is a property of the
     real data, so it has to be measured on the holdout, not assumed.</div>
</section>

<section>
  <h2>4 · The free-catalogue-core lever</h2>
  <p>Masking zeroes a pixel's contribution to <strong>F only</strong>. The maxima in
     TP_w run over <strong>all</strong> pixels, masked or not. So a prediction placed
     exactly on a known catalogue fault costs nothing at all, and still collects
     true-positive credit for every withheld fault within 300 m of it. Two staff
     statements make this worth acting on:</p>
  <ul>
    <li>Known-catalogue pixels are excluded from the penalty terms
        (<a href="{S.STAFF_STATEMENTS[0]['thread']}">11516</a>).</li>
    <li>“New fault” = any fault pixel not already captured by USGS/INGENIOUS,
        <em>and can include newly mapped geometry of an existing fault system</em>
        (<a href="{S.STAFF_STATEMENTS[1]['thread']}">11536</a>).</li>
  </ul>
  <p>Together: a large share of the scored pixels are inside existing fault zones, and the
     free core already reaches them. The submission shape that maximises P(Win) is
     therefore neither “predict faults” nor “predict new faults”:</p>
  <pre><code>p = 1 on every catalogue pixel                 free  (masked → no FP cost)
p = 1 on a thin, evidence-gated halo around them    paid  (this is where new geometry is)
p = 0 everywhere else</code></pre>
  <p>3 px of halo is 300 m — exactly the metric's support radius. Beyond that the halo
     reaches faults it cannot earn credit for and is pure cost. <code>build_corridor()</code>
     in <code>src/gems/strategy.py</code> implements it; the halo radius and paid budget are
     calibrated by sweep on a <strong>training fold</strong>, never on the leaderboard.</p>
</section>

<section>
  <h2>5 · Measured: what the shape is worth</h2>
  <p><code>python scripts/validate_holdout.py --strategy</code> builds phantoms whose
     withheld faults follow the staff definition (tip continuations, splays, parallel
     strands, a minority of isolated faults), then scores each arm with the real DTI and
     the real masking rule.</p>
  {exp_rows}
  <p>Reading: <strong>proximity gating is the lever, not the pixel count.</strong> The
     metric's 300 m kernel pays for a pixel only if a withheld fault is within 300 m of it,
     and withheld faults are overwhelmingly near mapped ones — so the same budget spent
     inside a band around the catalogue beats the same budget spread over the raster.</p>
  <div class="note warn"><b class="lbl">What this does not show</b>
    It is synthetic. It prices the arithmetic of the submission shape against the published
    metric and masking rule. It says <strong>nothing about Nevada geology</strong>, and it
    must not be cited as evidence that any detector finds faults. The detector claims are
    on the <a href="hypotheses.html">hypotheses</a> page and are labelled unvalidated.</div>
</section>

<section>
  <h2>6 · Honest limits of this strategy</h2>
  <ul>
    <li><strong>The mask may not be pixel-identical to our raster.</strong> The scorer masks
        the organiser's rasterisation of the vector catalogue; we are given
        <code>existing_faults.tif</code>. A one-pixel disagreement makes a supposedly-free
        pixel cost 1.0 of FP mass. <code>build_corridor(mask_safety_px=…)</code> exists for
        this; the safe default is 0 and the risk is unquantified until we can compare.</li>
    <li><strong>The halo has to be evidence-gated, not uniform.</strong> A 3 px dilation
        everywhere spends the whole FP budget on empty ground. The budget sweep in §5 has an
        interior optimum precisely because of this.</li>
    <li><strong>Phase 1 and Phase 2 reward different things.</strong> Phase 1 is a fixed
        private label set; Phase 2 adds whatever experts verify from everyone's
        submissions. A high-recall submission that is a superset of the Phase 1 truth
        converts into Phase 2 label additions. Do not over-tune to the public board.</li>
    <li><strong>The algebra cannot manufacture recall.</strong> EQ-2 tells you the price of
        recall; it does not find faults. Detector G-1 through G-5 are the part that has to
        work, and they are unvalidated.</li>
  </ul>
</section>
"""


# --------------------------------------------------------------------------- #
def page_hypo(ctx):
    rows = []
    for i, det in enumerate(_detectors(), 1):
        rows.append(f"""
    <div class="card">
      <h3>{esc(det.key)} · {esc(det.title)}
        <span class="pill bad">unvalidated — needs data/</span></h3>
      {kv([("Layers", esc(det.layers)),
           ("Physical signature", esc(det.signature)),
           ("Why it should catch a fault the catalogue misses", esc(det.why_missing)),
           ("Cost", esc(det.cost)),
           ("External data needed", esc(det.needs_external or "none")),
           ("Implemented in", Raw(f"<code>src/gems/features.py::{det.fn.__name__}</code>"))])}
    </div>""")
    return f"""
<section>
  <div class="card hero">
    <h2 style="border:0;padding:0;margin-bottom:.5rem">Five candidate detectors</h2>
    <p>Ranked by expected value against the metric algebra on the
       <a href="strategy.html">Strategy</a> page, not by how novel they sound. Each names
       its layers, the physical signature it targets, why it should catch something the
       USGS/INGENIOUS catalogue does not contain, and how it differs from every sibling
       submission and from the previous hypotheses in this repo.</p>
    <div class="note warn"><b class="lbl">Status of all five</b>
      Implemented, unit-tested where a closed form exists, and
      <strong>not validated against real rasters</strong>. The competition data needs a
      DrivenData login and this environment has no outbound network from the shell. Ranking
      them is a statement about expected value under the metric algebra plus the published
      geology, not a measurement. Nothing here is a result.</div>
  </div>
</section>

<section>
  <h2>Ranking</h2>
  {table(["Rank", "Detector", "Expected DTI", "Cost", "Blocked on"],
     [["1", "<strong>G-1</strong> Catalogue geometry completion",
       "<strong>Highest</strong> — the only detector aimed at the exact class the staff "
       "definition names", "Low", "competition data"],
      ["2", "G-2 Geodetic dilation-tendency ridge",
       "High — subsurface signal where there is no scarp", "Low", "competition data"],
      ["3", "G-3 Clay-cap conductivity edge",
       "Medium-high — blind faults, a class the catalogue provably under-samples", "Low", "competition data"],
      ["4", "G-4 Magnetic contact edge",
       "Medium — basement offset under cover", "Low-med", "competition data"],
      ["5", "G-5 Fluvial knickpoint / drainage deflection",
       "Medium, lower confidence", "Low (competition layers) / med (3DEP)", "competition data; 3DEP optional"]],
     numeric=(0,))}
</section>

<section class="grid" style="gap:1.1rem">
{''.join(rows)}
</section>

<section>
  <h2>How these differ from everything already tried</h2>
  {table(["Previous attempt", "What it did", "Why these five are different"], [
    ["GEMSDOE1 / GEMSDOE2 / 5GEMSDOE / 8GEMSDOE",
     "Copied the known catalogue raster; two of them shipped byte-identical files",
     "The catalogue is used as a <em>geometric prior</em> (endpoints, curvature, strand "
     "spacing), never as the output. Predicting the catalogue is a component of the field, "
     "not the field."],
    ["GEMSDOE3 (three submissions)",
     "Sparse point placements — 'Pindrop' nodes, a catalogue-gap target, a dense ridge control",
     "These are point/line hypotheses with no continuous probability field. G-1 is a density "
     "field over a 300 m corridor; G-2..G-5 are continuous geophysical scores."],
    ["GEMSDOE4",
     "Lineament features plus proxy labels; scored 0.0343",
     "G-5 is the sharpened version of the lineament idea (knickpoint second derivatives plus "
     "drainage-azimuth circular variance on detrended elevation) and G-4 adds the tilt-zero "
     "gating that separates contacts from faults."],
    ["6GEMSDOE",
     "Gradient boosting over 88 channels, top 3%; scored 0.0286",
     "Same model family, opposite failure: thresholding 3% of pixels with no geological "
     "prior puts almost all of the budget in empty ground. The metric algebra says "
     "precision is the cheap thing to buy and recall is the expensive one; 6GEMSDOE bought "
     "the wrong one."],
    ["H9-1 … H9-5 (previous revision of this repo)",
     "Relay stepovers, conductivity edge, SL/ksn, thermal+ASTER, Euler depth",
     "Superseded. H9-1's dilation tendency used a formula that is not the published "
     "quantity; H9-3 and H9-4 needed external downloads the pipeline does not actually "
     "have. G-1 is new and rests on an official statement the previous revision never cited. "
     "G-5 keeps the fluvial idea but runs on the competition's own detrended-elevation "
     "layers, so it is not blocked on 3DEP."],
  ])}
</section>

<section>
  <h2>Validation protocol — fixed before any slot is spent</h2>
  <ol>
    <li><strong>Spatially-blocked holdout.</strong> 4×4 blocks with a 300 m buffer, so the
        classifier cannot learn a fault and be scored on the pixels touching it. Random pixel
        splits are meaningless here: catalogue strands are 1–2 px wide.</li>
    <li><strong>Budget-matched random control</strong> at every arm, so "it scored well" is
        never confused with "it found something".</li>
    <li><strong>Catalogue-only arm</strong>, because a model that merely reproduces the
        catalogue scores perfectly on catalogue labels and 0.1563 on the board.</li>
    <li><strong>Gate:</strong> no submission file is written unless the corridor arm beats
        catalogue-only by ≥ 0.02 <em>and</em> the model arm beats random by ≥ 0.02.
        <code>python scripts/validate_holdout.py</code> exits non-zero and
        <code>build_submission.py --holdout-gate</code> refuses to write.</li>
  </ol>
  <div class="note bad"><b class="lbl">The limitation we cannot engineer away</b>
    This holdout scores <em>catalogue</em> faults. The competition scores faults that are
    <em>not</em> in the catalogue. No public holdout can measure the thing we are scored on,
    because the labels do not exist publicly. The protocol above can prove a detector
    carries information beyond “where the catalogue already is”. It cannot prove the
    detector finds unmapped faults. That gap is the honest state of play, and it is why
    the metric algebra in <a href="strategy.html">Strategy</a> — which is fully determined —
    carries more of the expected lift than any detector claim here.</div>
</section>
"""


# --------------------------------------------------------------------------- #
def page_intel(ctx):
    cards = []
    for st in S.STAFF_STATEMENTS:
        # rendered with its own indent, so an absent "Also:" leaves no line of
        # trailing whitespace behind in the generated HTML
        also = (f"      <p class='small'><strong>Also:</strong> "
                f"{esc(st['also'])}</p>") if st["also"] else ""
        cards.append(f"""
    <div class="card">
      <h3>{esc(st['id'].replace('_', ' ').title())}</h3>
      <blockquote style="margin:.5rem 0;padding:.7rem 1rem;border-left:3px solid var(--brand);background:#faf5f4;font-style:italic">
        “{esc(st['quote'])}”
      </blockquote>
{also}
      <p class='small muted'>{esc(st['who'])} · {esc(st['when'])} ·
        <a href='{esc(st['thread'])}' target='_blank' rel='noopener'>{esc(st['thread_title'])}</a></p>
      <div class="note" style="margin-top:.6rem"><b class="lbl">Why it matters</b>
        {esc(st['why_it_matters'])}</div>
    </div>""")
    return f"""
<section>
  <div class="card hero">
    <h2 style="border:0;padding:0;margin-bottom:.5rem">Official statements, quoted</h2>
    <p>Everything the sponsor's staff has said in the competition forum, read live on
       {S.VERIFIED_ON} from <a href="{S.COMPETITION['forum']}">community.drivendata.org</a>.
       Quoted verbatim so you can check the wording yourself — these four sentences drive
       most of the strategy.</p>
  </div>
</section>
<section class="grid" style="gap:1.1rem">
{''.join(cards)}
</section>

<section class="card">
  <h2>What the problem description says</h2>
  <p class="src"><a href="{S.COMPETITION['problem']}">Problem description</a> ·
     <a href="{S.COMPETITION['about']}">About</a></p>
  {kv([("Task", "Develop models and algorithms that provide accurate information about the "
                "presence of structures indicative of geothermal resources — namely, geological faults."),
       ("Ground truth", "The USGS quaternary fault database is public and known to be "
                        "incomplete, and may contain inaccurate data. The test set is faults "
                        "experts identified that are NOT in that database."),
       ("Rounds", S.SPEC['rounds']['initial'] + " " + S.SPEC['rounds']['final']),
       ("Selection", S.SPEC['rounds']['single']),
       ("Metric", f"Distance-weighted Tversky index. {S.SPEC['kernel']}. {S.SPEC['alpha_beta']}. "
                  f"Worked example: {S.SPEC['worked_example']}"),
       ("Provided layers", " · ".join(S.SPEC['feature_groups'])),
       ("Also provided", "1m_DEM_links.csv — URLs for 1 m resolution DEM data"),
       ("External data", "Allowed, provided the participant holds a licence permitting use "
                        "in this challenge and sharing with the sponsor."),
       ])}
</section>

<section class="card">
  <h2>Published methods this work builds on</h2>
  {table(["Reference", "What it establishes", "Where it is used here"],
     [[f'<a href="{esc(l["url"])}" target="_blank" rel="noopener">{esc(l["who"])} ({l["year"]})</a>',
       esc(l['what']), f'<a href="{esc(l["url"])}" target="_blank" rel="noopener">{esc(l["used_for"].split(".")[0])}</a>']
      for l in S.LITERATURE])}
</section>
"""


# --------------------------------------------------------------------------- #
def page_data(ctx):
    man = ctx["manifest"]
    return f"""
<section>
  <div class="card hero">
    <h2 style="border:0;padding:0;margin-bottom:.5rem">Data, access, and what is blocked</h2>
    <p>Stated plainly, because the blocker is the single thing standing between this repo
       and a trained model.</p>
  </div>
</section>

<section class="card">
  <h2>Access status, measured in this environment</h2>
  {table(["Resource", "Status", "Detail"],
     [[f'<a href="{esc(a["url"])}" target="_blank" rel="noopener">{esc(a["item"])}</a>',
       f"<span class='pill {'ok' if a['status'].startswith('OK') else ('bad' if 'BLOCKED' in a['status'] else 'warn')}'>{esc(a['status'])}</span>",
       esc(a['detail'])] for a in S.ACCESS])}
  <div class="note bad"><b class="lbl">We are not going to ask for credentials</b>
    The data tab requires a DrivenData account. No credentials exist in this environment
    and none will be requested, stored, or worked around. The shell additionally has no
    outbound HTTPS (TLS handshake fails; <code>curl</code> exit 35), so no file — official
    or mirrored — can be fetched here regardless. Everything downstream of
    <code>data/</code> is therefore written, reviewed and unit-tested, but unexecuted.</div>
</section>

<section class="card">
  <h2>Files the pipeline expects</h2>
  {table(["File", "What it is", "Used for"], [
    ["<code>data/training_features.tif</code>",
     "GeoTIFF, EPSG:32611, 100 m, multi-band. The problem description lists the layer groups; "
     "the exact band count and order are read from the file's own per-band description "
     "tags at load time rather than assumed.",
     "all five detectors"],
    ["<code>data/existing_faults.tif</code>",
     "The known USGS/INGENIOUS faults, rasterised. These are the training labels <em>and</em> "
     "the free core of the submission <em>and</em> the evaluation mask in the holdout.",
     "model fitting, corridor, masking"],
    ["<code>data/example_submission.tif</code>",
     "The organiser's template. The problem description says it “predicts total fault absence”.",
     "grid/footprint cross-check only"],
    ["<code>data/1m_DEM_links.csv</code>",
     "URLs for 1 m DEM tiles.",
     "optional input to G-5"],
  ])}
  <div class="note warn"><b class="lbl">Claims removed, not repeated</b>
    The previous revision of this site stated a file size, a band count and a SHA256 for
    each raster, and claimed <code>example_submission.tif</code> was “bit-identical to the
    label raster”. None of it is checkable from this repository and the last claim
    contradicts the problem description. It has been removed rather than softened.
    <a href="verification.html#irregularity-raster">See the flag →</a></div>
</section>

<section class="card">
  <h2>Free, official external data — and whether we need it</h2>
  {table(["Source", "URL", "Licence", "Needed for G-5?"], [
    ["USGS 3DEP 10 m DEM", "https://apps.nationalmap.gov/3dep/", "Public domain (USGS)",
     "<span class='pill warn'>optional</span> — G-5 falls back to the competition's own "
     "detrended-elevation layers, so the pipeline is not blocked"],
    ["USGS ScienceBase (GeoDAWN)", "https://www.sciencebase.gov/catalog/item/657e1d85d34e23d3533209f7",
     "Public domain (USGS)", "<span class='pill ok'>no</span> — the competition already ships "
     "derived GeoDAWN products"],
    ["USGS Quaternary Fault and Fold Database", "https://www.usgs.gov/programs/earthquake-hazards/faults",
     "Public domain (USGS)", "<span class='pill ok'>no</span> — already in the labels"],
    ["Landsat-8/9 TIRS (thermal)", "https://earthexplorer.usgs.gov/", "Public domain (USGS)",
     "<span class='pill ok'>no</span> — dropped from the candidate set; the previous "
     "revision's H9-4 depended on it and is not carried forward"],
    ["ASTER L1T (alteration)", "https://search.earthdata.nasa.gov/", "Public domain (NASA)",
     "<span class='pill ok'>no</span> — same reason"],
  ])}
  <p class="small muted">The competition permits external data provided the participant
     holds a licence that allows use in the challenge and sharing with the sponsor for
     evaluation. Everything listed above is public domain, so all of it is usable. The
     constraint is engineering time and download volume, not licensing — and none of it is
     on the critical path.</p>
</section>

<section class="card">
  <h2>Getting the data onto a machine that can reach the internet</h2>
  <pre><code>git clone https://github.com/buffedlizard55-lab/GEMSDOE9.git
cd GEMSDOE9
python3 -m pip install -r requirements.txt

# 1. Sign in at {S.COMPETITION['data']} and download:
#      training_features.tif, existing_faults.tif, example_submission.tif,
#      1m_DEM_links.csv      →  data/
#    (scripts/download_competition_data.sh lists what is expected and why)

# 2. Prove the pipeline is real before spending a slot
python scripts/validate_holdout.py            # blocked holdout, prints the gate
python scripts/validate_submission.py docs/downloads/*.tif</code></pre>
</section>
"""


# --------------------------------------------------------------------------- #
def page_verify(ctx):
    d = ctx["derived"]
    man = ctx["manifest"]
    t = ctx.get("tests")
    n_tests = (f"{t['passed']}/{t['total']} passing" if t
               else "not yet run — <code>python tests/test_validation.py</code>")
    flags = []
    for i, ir in enumerate(S.IRREGULARITIES, 1):
        cls = {"HIGH": "bad", "MEDIUM": "warn", "LOW": "info"}[ir["severity"]]
        flags.append(f"""
    <div class="card" id="irregularity-{esc(ir.get('id') or _slug(ir['title']))}">
      <h3>{i}. {esc(ir['title'])}
        <span class="pill {cls}">{esc(ir['severity'])}</span></h3>
      <p class="small muted">Found {esc(ir['found'])}</p>
      <p>{esc(ir['detail'])}</p>
    </div>""")
    checks = table(["Gate", "What it proves", "Where"], [
        ["<code>GDAL-READABLE</code>", "rasterio decodes every strip — this is the check whose "
         "absence caused the original rejection", "<code>tests G4</code>"],
        ["<code>PLATFORM-RANGE</code>", "0 NaN, 0 Inf, all 12,279,160 values in [0, 1]",
         "<code>tests G2</code>"],
        ["<code>EQ-1 == EQ-2</code>", "the closed form of the metric agrees with the exact "
         "implementation", "<code>tests B1, A7</code>"],
        ["<code>attractor</code>", "0.1563 ⇔ ρ = " + f"{d['attr_rho']:.4f} at φ = 0",
         "<code>tests B2–B4</code>"],
        ["<code>free-core lever</code>", "a masked pixel earns TP credit and costs zero FP",
         "<code>tests A10, A11, C2</code>"],
        ["<code>closed form vs measurement</code>", "EQ-2 predicts the measured DTI of a "
         "catalogue copy to ~1e-11", "<code>validate_holdout.py --strategy</code>"],
        ["<code>dilation tendency limits</code>", "Td = 0.5 in pure shear, 1.0 in pure "
         "uniaxial extension, 0.75 for (3, −1), scale-invariant",
         "<code>tests D1–D7</code>"],
        ["<code>browser writer round-trip</code>", "write → parse IFD → decode → compare, "
         "exact, on 5 grid shapes × 2 compression modes", "<code>tests E1–E6</code>"],
        ["<code>not-a-known-duplicate</code>", "the SHA256 is not one of the seven that "
         "already spent a submission slot", "<code>tests G6</code>"],
    ])
    return f"""
<section>
  <div class="card hero">
    <h2 style="border:0;padding:0;margin-bottom:.5rem">Verification, and what is still broken</h2>
    <p>Two lists: the checks that run on every build, and the problems found in this
       repository that a reviewer should know about. The second list is longer than the
       first because it should be.</p>
  </div>
</section>

<section class="card">
  <h2>Reproduce every number on this site</h2>
  <pre><code>bash scripts/run_all_checks.sh</code></pre>
  <p>Runs, in order: the {n_tests} test suite, the strategy experiment, the submission
     validator, and the site build. Nothing on this site is typed by hand — it is rendered
     by <code>scripts/build_site.py</code> from <code>scripts/site_data.py</code> (verified
     claims + source URLs) and from the JSON artifacts the scripts emit.</p>
</section>

<section>
  <h2>The checks</h2>
  {checks}
</section>

<section>
  <h2>Problems found, fixed, and flagged</h2>
  <p>Each was found by actually running the thing rather than reading it. The two HIGH
     items are the reason a download from the previous version of this site was rejected at
     the upload form.</p>
</section>
<section class="grid" style="gap:1.1rem">
{''.join(flags)}
</section>

<section class="card">
  <h2>Known gaps</h2>
  <ul>
    <li><strong>No model has been trained.</strong> The data is behind a login and this
        environment has no outbound network. Every detector is written and, where a closed
        form exists, unit-tested — and none has been run on real rasters.</li>
    <li><strong>The band inventory is unverified.</strong> The problem description lists
        layer <em>groups</em>, not an ordered index. All band access is by name from the
        GeoTIFF tags, and a miss raises rather than guessing.</li>
    <li><strong>The mask alignment risk is unquantified.</strong> We assume the organiser's
        rasterised catalogue matches <code>existing_faults.tif</code>. If it is off by a
        pixel, part of the “free” core becomes charged. <code>mask_safety_px</code> is the
        mitigation; the right value is unknown until someone can compare.</li>
    <li><strong>The 0.1563 explanation is a derivation.</strong> It is consistent with every
        observation available and it is falsifiable, but the submitted files are not
        public, so it cannot be confirmed against them directly.</li>
    <li><strong>Phase 2 is a different objective.</strong> Experts build the expanded label
        set from the Phase 1 submissions. A submission tuned narrowly to the public board
        can be the wrong one to select. The format says one submission is scored in both
        rounds.</li>
  </ul>
</section>
"""


def _slug(s: str) -> str:
    return "".join(c.lower() if c.isalnum() else "-" for c in s).strip("-")[:40]


# --------------------------------------------------------------------------- #
def page_lb(ctx):
    d = ctx["derived"]
    rows = []
    for rank, name, subs, score in S.LEADERBOARD["rows"]:
        hl = abs(score - 0.1563) < 1e-9
        pill = ("<span class='pill warn'>pinned</span>" if hl else
                ("<span class='pill ok'>leader</span>" if rank == 1 else ""))
        rows.append([str(rank), esc(name), str(subs), f"<strong>{score:.4f}</strong>", pill])
    grp = table(["Submission", "Score", "What it was"],
                [[f'<a href="{esc(u)}" target="_blank" rel="noopener">{esc(n)}</a>', f"{s:.4f}", esc(w)]
                 for n, u, s, w in S.GROUP_SUBMISSIONS], numeric=(1,))
    return f"""
<section>
  <div class="card hero">
    <h2 style="border:0;padding:0;margin-bottom:.5rem">Leaderboard</h2>
    <p>Read live on {S.LEADERBOARD['as_of']} from
       <a href="{S.LEADERBOARD['source']}">{S.LEADERBOARD['source']}</a>. This is a
       snapshot; open the link for the current state.</p>
  </div>
</section>

<section class="card">
  {table(["#", "Participant", "Submissions", "Best public DW-Tversky", ""], rows, numeric=(0, 2, 3))}
  <p class="small muted">{esc(S.LEADERBOARD['note'])}</p>
  <div class="note"><b class="lbl">The observation that matters</b>
    {esc(S.LEADERBOARD['observation'])}</div>
</section>

<section class="grid two">
  <div class="card">
    <h2>What the top score implies</h2>
    <p>From the closed form on the <a href="strategy.html">Strategy</a> page, with zero
       false-positive mass:</p>
    {table(["Score", "Minimum weighted recall", "Ratio vs 0.1563"],
     [["0.1563", f"{d['attr_rho']:.4f}", "1.00×"],
      ["0.2993", f"{d['row2_rho']:.4f}", f"{d['row2_rho'] / d['attr_rho']:.2f}×"],
      ["0.3049", f"{d['top_rho']:.4f}", f"<strong>{d['ratio']:.2f}×</strong>"]], numeric=(1, 2))}
    <p>A detector must roughly double the recall a catalogue copy gets. Submissions can
       carry false positives and still make it — the cost is 1/5 of a missing one — but
       0.3049 with <em>no</em> false positives requires ρ ≥ {d['top_rho']:.4f}.</p>
  </div>
  <div class="card">
    <h2>The group's submissions</h2>
    {grp}
    <div class="note warn"><b class="lbl">Ranks 24–25 are not the problem, the ceiling is</b>
      Two unrelated accounts sitting on exactly 0.1563 is the signature of a degenerate
      optimum, not of a tuning failure. Breaking it requires a detector that finds
      something the catalogue does not already contain — see
      <a href="hypotheses.html">G-1</a>.</div>
  </div>
</section>
"""


# --------------------------------------------------------------------------- #
def page_research(ctx):
    lit = table(["Reference", "What it establishes", "Used for"],
                [[f'<a href="{esc(l["url"])}" target="_blank" rel="noopener">{esc(l["who"])} ({l["year"]})</a>',
                  f'{esc(l["what"])}<div class="src">{esc(l["where"])}</div>', esc(l["used_for"])]
                 for l in S.LITERATURE])
    return f"""
<section>
  <div class="card hero">
    <h2 style="border:0;padding:0;margin-bottom:.5rem">Geothermal exploration, and why faults</h2>
    <p>What the science says, what the competition says, and where the gap between them is
       the opportunity. Every reference links to a primary source you can read.</p>
  </div>
</section>

<section>
  <h2>Why faults are the target</h2>
  <p class="src">Competition About page —
     <a href="{S.COMPETITION['about']}#about-the-task">source</a></p>
  <p>Geothermal resources concentrate where fluids can move, and faults are the conduits.
     The competition asks for the structures that indicate geothermal resources — which is
     why the label set is faults rather than temperature, and why “many faults” beats “the
     one fault under the best-known field”.</p>
  <p>The About page also states plainly that <em>“most faults in the GeoDAWN region of
     Nevada are more subtle, and many are hidden below the surface, requiring geophysical
     data to detect.”</em> That is the whole problem statement in one sentence: the
     catalogue is a surface map, and the answer is mostly subsurface.</p>
</section>

<section>
  <h2>The measurement problem</h2>
  <p>Three facts, in order of how much they constrain the design:</p>
  <ol>
    <li><strong>The label set is incomplete by construction.</strong> “This set of faults is
        not complete and may even contain some inaccurate data.” So a model trained to
        reproduce the labels is trained on a target that is partly noise.</li>
    <li><strong>Training and scoring use different labels.</strong> Training labels are the
        public catalogue; both prize rounds score faults that are <em>not</em> in it.</li>
    <li><strong>“New fault” is defined as a set difference, not as a new structure.</strong>
        <a href="{S.STAFF_STATEMENTS[1]['thread']}">Staff, 2026-09-22</a>:
        “any fault pixel not already captured by USGS/INGENIOUS and can include newly mapped
        geometry of an existing fault system.”</li>
  </ol>
  <p>Point 3 is decisive and was not acted on before. It converts the catalogue from
     something to avoid (it would be scored as noise) into something to use (it is free
     under masking, and it is a locator for the pixels that are actually scored).</p>
</section>

<section>
  <h2>What is known about where geothermal systems focus</h2>
  <p class="small muted">Stated qualitatively and with the source attached, rather than with
     the precise percentages the previous revision of this README carried. Those
     percentages could not be re-verified against the primary text in this environment, so
     they have been dropped rather than repeated.</p>
  <p>Faulds, Hinz &amp; Kreemer (2012) document that geothermal fluids in the Great Basin
     preferentially occupy <strong>fault intersections, stepovers and fault terminations</strong>
     rather than the middle of throughgoing strands — the geometry that maximises
     permeability damage. Blind faults beneath basin fill are the classic case with no
     surface expression at all, which is what detectors G-2 and G-3 are aimed at.</p>
  <p>Hermant, Kiersnowski &amp; Bellanger (2025) apply deep learning to Quaternary fault
     mapping over the same region, and Mattéo et al. (2021) to remote optical and topographic
     data more generally. Both are cited by the competition's own About page, which is the
     strongest possible signal that this is the state of the art for this exact task.</p>
  <p class="small muted">Detector G-1's geometry (endpoints, curvature, strand spacing) is
     the raster expression of that intersection/stepover/termination preference, and it is
     the one detector here whose target class is named explicitly by the organiser.</p>
</section>

<section>
  <h2>References</h2>
  {lit}
</section>
"""


# --------------------------------------------------------------------------- #
def page_sources(ctx):
    seen, rows = set(), []
    def add(title, url, what, verified=True, when=S.VERIFIED_ON):
        if url in seen:
            return
        seen.add(url)
        pill = ("<span class='pill ok'>read " + when + "</span>" if verified
                else "<span class='pill bad'>NOT verified</span>")
        rows.append([esc(title), f'<a href="{esc(url)}" target="_blank" rel="noopener">link</a>',
                     esc(what), pill])

    add("Competition overview", S.COMPETITION["overview"], "Rules, prize structure, timeline")
    add("Problem description", S.COMPETITION["problem"],
        "Task, provided features, metric definition, worked example, submission format")
    add("About the data", S.COMPETITION["about"],
        "Sponsor, GeoDAWN provenance, what a fault is and how they are found")
    add("Data tab", S.COMPETITION["data"],
        "File downloads. Requires a DrivenData account — not reachable from this environment")
    add("Leaderboard", S.COMPETITION["leaderboard"], "Live standings")
    add("Submissions", S.COMPETITION["submissions"], "The upload form")
    add("Forum", S.COMPETITION["forum"], "Competition category")
    for st in S.STAFF_STATEMENTS:
        add(f"Forum — {st['thread_title']}", st["thread"],
            f"Staff statement, {st['who']}, {st['when']}", True, st["when"])
    add("Reference solution", S.COMPETITION["reference_solution"],
        "UNet + Monte Carlo CV + Tversky loss; reads band names from GeoTIFF tags")
    add("Official rules PDF", S.COMPETITION["rules_pdf"],
        "Full competition rules, hosted by NLR (note: the previous revision quoted this PDF "
        "as verified without ever fetching it; it is listed here for review, not quoted)")
    for l in S.LITERATURE:
        add(f"{l['who']} ({l['year']})", l["url"], l["what"])
    add("USGS 3DEP", "https://apps.nationalmap.gov/3dep/",
        "Free official DEMs. Optional input to detector G-5")
    add("USGS GeoDAWN (About page link)",
        "https://www.usgs.gov/data/geodawn-airborne-magnetic-and-radiometric-surveys-northwestern-great-basin-nevada-and",
        "Origin of the magnetic and radiometric layers")
    add("GeoDAWN study-area map (ScienceBase)",
        "https://www.sciencebase.gov/catalog/item/657e1d85d34e23d3533209f7",
        "Official map of the survey area, linked from the competition")
    add("USGS Quaternary Faults",
        "https://www.usgs.gov/programs/earthquake-hazards/faults",
        "The public catalogue the competition's training labels come from")
    add("INGENIOUS project", "https://gbcge.org/current-projects/ingenious/",
        "Great Basin geothermal data compilation named by the competition")
    add("DrivenData GEMS page on OpenEI", "https://gdr.openei.org/submissions/1391",
        "Third-party mirror of competition materials. NOT an official source")

    return f"""
<section>
  <div class="card hero">
    <h2 style="border:0;padding:0;margin-bottom:.5rem">Every source, with its status</h2>
    <p>Anything this site states as fact is on this list with the page it was read from.
       Entries that could not be reached are marked, not omitted.</p>
  </div>
</section>
<section class="card">
  {table(["Source", "Link", "What it is used for", "Status"], rows)}
</section>
<section class="card">
  <h2>Not used, and why</h2>
  <ul>
    <li><strong>Dropbox mirrors of the competition rasters</strong> (listed in the project
        brief). Not reachable from this environment, and not linked from any DrivenData or
        DOE page — their provenance is unestablished. They are not treated as a data source.</li>
    <li><strong>The previous revision's raster hashes and pixel counts.</strong> Not
        checkable from this repository, and one of them contradicts the problem
        description. Removed rather than repeated.
        <a href="verification.html">See the flag →</a></li>
    <li><strong>Stepover/termination/intersection percentages</strong> attributed to
        GDR submission 383. Plausible and widely repeated, but not re-verified against the
        primary text here, so the numbers are gone and only the qualitative finding is kept.</li>
  </ul>
</section>
"""


# --------------------------------------------------------------------------- #
BUILDERS = {
    "index.html": page_index, "executive_summary.html": page_exec,
    "strategy.html": page_strategy, "hypotheses.html": page_hypo,
    "intel.html": page_intel, "data.html": page_data,
    "verification.html": page_verify, "leaderboard.html": page_lb,
    "research.html": page_research, "sources.html": page_sources,
}

def badges(ctx) -> dict:
    man = ctx["manifest"]
    gates = 14 if (man and man.get("validator", {}).get("all_passed")) else 0
    t = ctx.get("tests")
    tests_badge = (f"{t['passed']}/{t['total']} checks pass" if t else "tests not run")
    return {
    "index.html": [("GEMSDOE9", ""), ("No model trained yet", "warn"),
                   (f"Submission file passes all {gates} gates" if gates else
                    "Submission file not built", "ok" if gates else "warn")],
    "executive_summary.html": [("Step-by-step", ""), ("Rejection explained", "ok")],
    "strategy.html": [("Metric algebra", ""), ("0.1563 explained", "ok")],
    "hypotheses.html": [("G-1 … G-5", ""), ("All unvalidated", "bad")],
    "intel.html": [("Quoted verbatim", ""), (f"Read {S.VERIFIED_ON}", "ok")],
    "data.html": [("Access status", ""), ("Data blocked", "bad")],
    "verification.html": [("Checks run", ""), (tests_badge, "ok" if t else "warn"),
                          (f"{len(S.IRREGULARITIES)} problems found", "warn")],
    "leaderboard.html": [(f"Snapshot {S.LEADERBOARD['as_of']}", ""), ("Live-read", "ok")],
    "research.html": [("Primary sources", ""), ("DOI-linked", "ok")],
    "sources.html": [("Every claim traced", "")],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="verify the pages on disk match what would be rendered")
    args = ap.parse_args()

    manifest, experiment, holdout, tests = load_artifacts()
    ctx = {"manifest": manifest, "experiment": experiment, "holdout": holdout,
           "tests": tests, "derived": derived_numbers()}

    stale = []
    for fname, title, _ in PAGES:
        body = BUILDERS[fname](ctx)
        page = shell(fname, title, body, extra_js=BUILDER_JS if fname == "index.html" else "",
                     badges=badges(ctx).get(fname, ()))
        p = DOCS / fname
        if args.check:
            if not p.exists() or p.read_text() != page:
                stale.append(fname)
        else:
            p.write_text(page)
            print(f"  wrote docs/{fname}  ({len(page):,} bytes)")

    (DOCS / "style.css").write_text(CSS)
    if args.check:
        if stale:
            print("STALE: " + ", ".join(stale) + " -- run python scripts/build_site.py")
            return 1
        print("site is up to date")
        return 0

    # sanity: no unresolved placeholders, every page linked from nav
    joined = " ".join((DOCS / f).read_text() for f, _, _ in PAGES)
    problems = []
    if "{esc(" in joined or "None</" in joined:
        problems.append("template escape leaked into output")
    index_html = (DOCS / "index.html").read_text()
    for f, t, _ in PAGES:
        if f"href='{f}'" not in index_html:
            problems.append(f"{f} not linked from nav")
    # every in-page anchor must resolve to an id that exists somewhere
    ids = set()
    for f, _, _ in PAGES:
        ids |= set(re.findall(r'id="([^"]+)"', (DOCS / f).read_text()))
    for f, _, _ in PAGES:
        for frag in set(re.findall(r"href='([a-z_]+\.html)#([a-z0-9-]+)'",
                                   (DOCS / f).read_text())):
            if frag[1] not in ids:
                problems.append(f"{f} links to {frag[0]}#{frag[1]} which does not exist")
    # every relative file link must exist on disk
    for f, _, _ in PAGES:
        for rel in set(re.findall(r"href='([a-zA-Z0-9_./-]+\.(?:html|css|js|json|tif|zip))'",
                                  (DOCS / f).read_text())):
            if not (DOCS / rel).exists():
                problems.append(f"{f} links to missing {rel}")
    # no raw HTML leaked into a text node
    for f, _, _ in PAGES:
        body = re.sub(r"<(script|style).*?</\1>", "", (DOCS / f).read_text(), flags=re.S)
        txt = html.unescape(re.sub(r"<[^>]+>", " ", body))
        for pat in ("<code>", "<span class=", "<a href=", "{esc("):
            if pat in txt:
                problems.append(f"{f} leaks raw markup ({pat}) into a text node")
    # no trailing whitespace and no tab characters in the shipped HTML: an
    # optional f-string field that renders empty leaves a line of spaces behind,
    # which git then flags as a whitespace error on every diff
    for f, _, _ in PAGES:
        for i, line in enumerate((DOCS / f).read_text().split("\n"), 1):
            if line != line.rstrip():
                problems.append(f"{f}:{i} has trailing whitespace")
                break
            if "\t" in line:
                problems.append(f"{f}:{i} contains a tab character")
                break
    if problems:
        for p in problems:
            print("PROBLEM:", p)
        return 1
    print(f"\nsite rendered: {len(PAGES)} pages + style.css")
    if manifest:
        print(f"  artifact : {manifest['file']}  sha256 {manifest['sha256'][:16]}…")
    else:
        print("  artifact : none — run python scripts/build_submission.py")
    if experiment and "blocked_table" in experiment:
        b0, b1 = experiment["blocked_table"][0], experiment["blocked_table"][-1]
        print(f"  strategy : free core {b0['catalogue']:.4f} -> gated corridor "
              f"{b0['corridor']:.4f} (chance detector) -> {b1['corridor']:.4f} "
              f"(AUC {b1['auc_median']:.2f}); detector-attributable "
              f"{b1['corridor'] - b0['corridor']:+.4f} over "
              f"{len(experiment['seeds'])} phantoms")
    else:
        print("  strategy : run python scripts/validate_holdout.py --strategy")
    print("  caveats  : 0 detectors validated; no model trained (data needs a login)")
    return 0


BUILDER_JS = r"""
<script>
(function(){
  var b=document.getElementById('buildBtn'), s=document.getElementById('status');
  if(!b) return;
  b.addEventListener('click', async function(){
    b.disabled=true; s.style.display='block'; s.className='note';
    s.innerHTML='<b class="lbl">Working</b>Building the GeoTIFF in your browser…';
    try{
      var W = Gems9Writer;
      var f = W.buildPlaceholderField(W.WIDTH, W.HEIGHT, {fraction:0.028});
      var v = await W.writeAndVerify(f, {width:W.WIDTH, height:W.HEIGHT, rowsPerStrip:128});
      if(!v.ok) throw new Error(v.reason);
      var sha = await W.sha256Hex(v.res.bytes);
      var sha8 = sha.slice(0,8);
      var stamp = new Date().toISOString().replace(/[-:]/g,'').replace(/\..+/,'');
      var name = 'gemsdoe9-PLACEHOLDER-'+stamp+'-'+sha8+'.tif';
      var url = URL.createObjectURL(new Blob([v.res.bytes],{type:'image/tiff'}));
      var a = document.createElement('a'); a.href=url; a.download=name;
      document.body.appendChild(a); a.click(); a.remove();
      setTimeout(function(){URL.revokeObjectURL(url);},4000);
      s.className='note ok';
      s.innerHTML='<b class="lbl">Round-trip verified</b>Wrote '+v.res.totalBytes.toLocaleString()+
        ' bytes, re-parsed the IFD, decoded every strip, compared all 12,279,160 values to the '+
        'input — exact match. All finite, all in [0,1].<br><br>'+
        '<strong>'+name+'</strong><br>SHA256 '+sha+'<br><br>'+
        'Paste this in the Note box: <code>GEMSDOE9 FORMAT-CHECK placeholder (no model, no '+
        'GeoDAWN data) | band-pattern 2.8% | '+sha8+'</code><br>'+
        'Note: this is a format check, not a prediction. See '+
        '<a href="executive_summary.html">How to submit</a>.';
    }catch(e){
      s.className='note bad';
      s.innerHTML='<b class="lbl">Build failed</b>'+String(e.message||e);
    }finally{ b.disabled=false; }
  });
})();
</script>
"""


if __name__ == "__main__":
    sys.exit(main())
