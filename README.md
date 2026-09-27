# GEMSDOE9 — Geothermal Fault Discovery Beyond the Catalogue

**Core Values — read every session, before deciding anything**

> **Maximize P(Win)** — "Maximize the Probability of Winning": our decision making framework.
> In every decision, we weigh tradeoffs, assess risk, and choose the path that maximizes the
> probability that Arena succeeds. We set aside our emotions and make tough decisions in order
> to maximize P(Win). "Maximize P(Win)" frees us from constraints and clarifies that we must
> put Arena first.
>
> **Own the Outcome** — We own results end to end — not just our individual slice of the work.
> When problems arise and we have the means to act, we do so without waiting for permission or
> assignment. We treat failure and success as signals and use them to improve. We stay
> accountable to the final outcome.

---

## Read this first — the state of the project, honestly

| | |
|---|---|
| **Can I submit today?** | **Yes.** `docs/downloads/` holds one valid GeoTIFF that passes all 14 format gates. The download button is the first thing on the site. |
| **Is it a prediction?** | **No.** It is a format-check placeholder. The competition rasters are behind a DrivenData login and this environment has no outbound network, so **no model has been trained**. |
| **How close is a real model?** | The pipeline is written end to end and gated: `python scripts/build_submission.py --holdout-gate` refuses to write a model submission unless it beats a budget-matched random control **and** a catalogue-only baseline on a spatially-blocked holdout. |
| **What *is* proven?** | The metric algebra, the submission shape, and the fact that the previous version of this site was shipping unreadable files. All three are reproduced by `bash scripts/run_all_checks.sh`. |
| **Single biggest lever** | 0.1563 is a catalogue copy. 0.3049 needs **2.01×** the recall. See [Why 0.1563](#why-01563-keeps-repeating). |

---

## What changed this session, and why

### Audit pass: four silent bugs, all in code that decides what gets submitted

The last revision shipped a "corridor lift of +0.1379, positive on 5/5 phantoms". That
number was wrong, and finding out why turned up three more defects behind it.

1. **The blocked holdout counted ground truth it was not allowed to see.**
   `sweep_corridor_halo(..., extra_ignore=...)` masked false positives outside the scored
   block but still summed true positives over the **whole** raster. Every out-of-block hit
   was credited for free. On a 2×2-block phantom that turned a 3 px halo from a −0.015
   loss into a **+0.24 gain** and made the corridor arm look like a winner. Fixed, and
   pinned from both sides by tests I2 and I3.

2. **`build_corridor` wrote paid pixels to the wrong array positions.**
   The top-k was computed as indices into the *band score vector* and then used directly as
   flat indices into the *grid*. Every paid pixel landed in the raster's top-left corner
   instead of along the fault. Caught by the new tie-heavy budget test (H3).

3. **`build_corridor` selected paid pixels with `score >= threshold`.**
   A tie-heavy or sparse score field keeps every pixel tied at the threshold: a request for
   50 paid pixels produced **498**. Replaced with a deterministic flat-index top-k.

4. **G-5 allocated 3.98 GB.** `sliding_window_view` on a 9×9 window over the 3292×3730
   grid. Replaced with two `uniform_filter` passes on the cos/sin of the azimuth; the new
   test J2 asserts that bytes-per-pixel does not grow with the grid, which is the property
   that actually matters.

The experiment that exposed all of this was rebuilt from scratch: the selection-free arms
are now scored on the whole phantom (nothing is tuned, so nothing can leak), and only the
part that *selects* is blocked and cross-fitted. The previous "withdrawn" note in the
strategy section below records the old result; the current one is in the table above it.

### Earlier pass: seven more, three serious enough to waste a submission slot

1. **The site's browser writer produced GeoTIFFs that GDAL could not open.**
   `docs/geotiff_writer.js` allocated the `StripOffsets` (tag 273) and `StripByteCounts`
   (tag 279) arrays as inline 4-byte IFD values *before their data existed*, so only the
   first 4 bytes of each array reached the file and strips 2..N pointed at offset 0.
   Reproduced: a 49,117,032-byte file from the default button, which rasterio refused with
   `TIFFReadEncodedStrip() failed`. **That is the real cause of the
   "Predicted values must be in range [0, 1]" rejection** — the platform reported an
   unreadable file using its value-range error, and the previous diagnosis blamed NaN. There
   was no NaN problem to find.
   *Fixed:* the writer is rewritten and round-trip tested — write, parse the IFD back, decode
   every strip, compare all 12,279,160 values to the input — before the download is offered.
   Also, only the **all-finite** variant is ever published now: NaN is not in [0, 1], so a
   0.0 outside the footprint is the reading that cannot fail a range check.

2. **`docs/downloads/` shipped two byte-identical `.tif` files under different names.**
   This is exactly the failure the team asked to stop. Both are removed.
   `build_submission.py` now deletes previous artifacts before writing, names files by content
   hash, and refuses to publish anything whose SHA256 is on the spent-slot list.

3. **The browser's "H9-1 relay-stepover" field was two crossing sinusoids and an xorshift.**
   No GeoDAWN data, no catalogue, no model — yet the site labelled the file
   "H9-1 relay-stepover + dilation + intersection" and the note pasted into DrivenData
   repeated it. Removed. The replacement is named `PLACEHOLDER`, its note says
   "no model, no GeoDAWN data", and nothing in this repo presents an unvalidated field as a
   geological result.

4. **The dilation tendency was not the published quantity.** The old
   `0.5*(1 + dilatation/(shear + |second_invariant|))` is not Simpson & Reiling (2008); it
   does not have the right limits and does not survive a rescaling of strain rate. Replaced
   with the closed form and pinned by four limit tests.

5. **Band indices were hard-coded and labelled "PROVISIONAL".** A wrong provisional index is
   a silent wrong answer. All band access now resolves by the GeoTIFF's own per-band
   description tags — the same source the official reference solution reads — and a miss
   raises with the list of available names.

6. **Claims about the rasters that cannot be supported from this repo** (file sizes, a band
   count, SHA256s, and "the template is bit-identical to the label raster") were removed
   rather than softened. The last one contradicts the problem description, which says the
   template "predicts total fault absence".

7. **No `requirements.txt`**, so `import scipy` failed on a clean checkout. Added.

Full detail, with the evidence, on [`docs/verification.html`](docs/verification.html) and in
`scripts/site_data.py::IRREGULARITIES`.

---

## Why 0.1563 keeps repeating

It is an **attractor of the metric**, not a coincidence of tuning.

The published metric has TP_w and FN_w as *maxima of the same product*, so `FN_w = N − TP_w`
exactly, and with α = 0.2, β = 0.8:

```
DTI = T / (0.2·T + 0.2·F + 0.8·N) = ρ / (0.2·ρ + 0.2·φ + 0.8)
```

where ρ = T/N is distance-weighted recall and φ = F/N is false-positive mass per
ground-truth pixel. Two ratios, and nothing else matters.

Staff have stated that known-catalogue pixels are **masked out of the penalty terms**
([thread 11516](https://community.drivendata.org/t/scoring-clarification-are-known-usgs-ingenious-faults-masked-when-scoring-and-are-they-in-the-final-round-label-set/11516)),
so a submission that paints the catalogue has φ = 0. Solving for ρ:

| Score | ρ at φ = 0 | Reading |
|---|---|---|
| **0.1563** | **0.1291** | a catalogue copy |
| 0.2993 | 0.2547 | a real detector |
| **0.3049** | **0.2598** | leader of the board — **2.01× the 0.1563 recall** |

That is why three unrelated accounts sit on exactly 0.1563 with two submissions each. Copy
the catalogue and you get 0.1563; copy it slightly differently and you still get 0.1563.

Two consequences that inverts the strategy the group has been running:

**Recall is worth about 4/ρ times more than precision.** Differentiating the closed form:

```
value of 1 unit of TP_w  ∝  (0.2·F + 0.8·N)
cost  of 1 unit of FP_w  ∝  (0.2·T)
exchange rate = 4/ρ + φ
```

At the 0.1563 attractor, one extra unit of recall is worth **31 units** of false-positive
mass. Every instinct that says "be conservative, be sparse" points the wrong way here.

**The catalogue is free.** Masking zeroes a pixel's contribution to **F** only — the maxima
in TP_w run over *all* pixels, masked or not. A prediction placed exactly on a known fault
costs nothing and still collects credit for every withheld fault within 300 m of it. And
staff have defined the target class
([thread 11536](https://community.drivendata.org/t/where-do-you-draw-the-line/11536),
2026-09-22, **not cited anywhere in the previous revision**):

> "For the purposes of this competition, 'new fault' means 'any fault pixel not already
> captured by USGS/INGENIOUS' and can include newly mapped geometry of an existing fault
> system."

Splays, parallel strands, continuations past mapped tips. By construction they sit next to
and collinear with faults the catalogue already has. So the shape that maximises P(Win) is
neither "predict faults" nor "predict new faults":

```
p = 1 on every catalogue pixel                  free  (masked → no FP cost)
p = 1 on a thin, evidence-gated halo around it  paid  (this is where new geometry is)
p = 0 everywhere else
```

3 px of halo is 300 m — exactly the metric's support radius. Beyond that the halo reaches
faults it cannot earn credit for and is pure cost.

**Measured** (`python scripts/validate_holdout.py --strategy`, 5 independent synthetic
phantoms whose withheld faults follow the staff definition, scored with the real DTI and the
real masking rule, configuration chosen on training blocks and scored on the held-out block):

| Submission shape | DTI | vs free core |
|---|---|---|
| known catalogue, nothing else (φ = 0 exactly) | 0.2544 | — |
| + a blind 3 px halo, detector ignored | 0.2935 | +0.0391 |
| + a detector-gated halo, chance detector (AUC ≈ 0.50) | 0.3519 | +0.0975 |
| + a detector-gated halo, useful detector (AUC ≈ 0.74) | 0.4312 | +0.1769 |

**Read the two gaps separately.** Core → blind halo is `+0.0391` and belongs to the halo's
geometry, not to any detector. Chance detector → useful detector, inside the *same* corridor,
is `+0.0793` — the only part of this table attributable to detection at all.

A second, selection-free sweep ranks the **whole raster** instead of a band. It loses to the
free core in 17 of 20 cells, at every budget from 0.1% to 10% of the grid and at every
detector quality. Proximity to a mapped fault is what the 300 m kernel rewards; spreading the
same pixel count uniformly over the region does not.

*This prices the arithmetic of the submission shape. It says nothing about Nevada geology and
is not evidence that any detector works.*

> **Withdrawn.** An earlier revision of this experiment reported a corridor lift of `+0.1379`
> positive on 5/5 phantoms. It was produced by a leaky protocol (ground truth outside the
> scored block was still counted in the true-positive term), by a selection-noise problem
> (the best of a 24-point grid on pooled training blocks), and by a budget bug in
> `build_corridor` that selected paid pixels with `score >= threshold` — so a request for
> 50 paid pixels produced 498. Every one of those is now a regression test in
> `tests/test_validation.py` (sections H, I and J). The number above replaces it.

---

## The five detectors

Ranked by expected value under the metric algebra, not by novelty. Each is an explicit,
auditable function in [`src/gems/features.py`](src/gems/features.py).

| # | Detector | Layers | Signature | Cost |
|---|---|---|---|---|
| **G-1** | Catalogue geometry completion | `existing_faults.tif` as *geometry* | endpoint proximity × skeleton curvature × azimuthal deviation, gated to ≤3 px outside the catalogue | Low |
| **G-2** | Geodetic dilation-tendency ridge | dilatation rate, second invariant of strain rate | `Td = 0.5 + T/(4·√(I2 − T²/4))` (Simpson & Reiling 2008) × structure-tensor coherence | Low |
| **G-3** | Clay-cap conductivity edge | surface conductivity, depth to conductive base | \|∇ log σ\| × log(σ/depth) — a scale-free lateral discontinuity of an alteration cap | Low |
| **G-4** | Magnetic contact edge | RTP anomaly, TMI, H/V slope, top-of-crust source depth | HGM maximum at a tilt-derivative zero crossing, weighted by shallow Euler-style depth | Low-med |
| **G-5** | Fluvial knickpoint / drainage deflection | detrended elevation, slope of detrended elevation; optional 10 m 3DEP | second-derivative ridges in along-strike slope + circular variance of downslope azimuth | Low (competition layers) |

Full rationale, why each should catch a fault the catalogue misses, and how each differs
from every prior submission: [`docs/hypotheses.html`](docs/hypotheses.html).

**None of the five is validated.** They are written, and unit-tested where a closed form
exists, and they have not seen a real raster. See *Limitations*.

---

## The submission — click, then submit

**1 · Get the file.** The download button is the first thing on the site. Or press
*Build in this browser*, which produces the identical pixel field with a fresh name and hash
— assembled locally, nothing uploaded.

**2 · Make the name unique.** Use `gemsdoe9-<what>-<sha8>.tif`. The hash identifies the exact
bytes, so two weeks from now you can still tell your submissions apart.

**3 · Upload.**
[drivendata.org → Submissions → New submission](https://www.drivendata.org/competitions/306/competition-doe-gems/submissions/).
Single-band `.tif` (or a `.zip` containing one), matching CRS, shape and geotransform.

**4 · Note.** The *Note (optional)* box exists so you can tell your own submissions apart
later. Ours embeds the SHA8:

```
GEMSDOE9 FORMAT-CHECK placeholder (no model, no GeoDAWN data) | band-pattern 2.8% | 2314b599
```

**5 · Verify it yourself** before and after downloading:

```bash
python scripts/validate_submission.py docs/downloads/*.tif
```

Every line must read `PASS`. The gates are `GDAL-READABLE`, `PLATFORM-RANGE`
(0 NaN, 0 Inf, all 12,279,160 values in [0, 1]), `no-nodata-tag`, the four grid checks,
`density-sane`, and `not-a-known-duplicate`.

Full walkthrough, including every rejection message and what causes it:
[`docs/executive_summary.html`](docs/executive_summary.html).

---

## The single remaining blocker

**The data.** Everything downstream of `data/` is written, reviewed and unit-tested but has
never been executed.

```bash
# on a machine with a DrivenData account and working internet
git clone https://github.com/buffedlizard55-lab/GEMSDOE9.git && cd GEMSDOE9
python3 -m pip install -r requirements.txt

# download from https://www.drivendata.org/competitions/306/competition-doe-gems/data/ into data/:
#   training_features.tif  existing_faults.tif  example_submission.tif  1m_DEM_links.csv
bash scripts/download_competition_data.sh     # checks what is there
python scripts/prepare_data.py                # verify + dump the authoritative band inventory

# then, and only then
python scripts/validate_holdout.py            # spatially-blocked holdout, prints the gate
python scripts/build_submission.py --holdout-gate
```

**We are not going to ask for credentials, and there is no workaround.** The data tab needs
an account, and this environment denies outbound HTTPS from the shell entirely (`curl` exit
35), so no URL resolves — official or mirrored. The Dropbox links in the project brief are
unreachable *and* their provenance is unestablished: they are not linked from any DrivenData
or DOE page. They are not treated as a data source.

---

## Repository layout

```
GEMSDOE9/
├── docs/                         GitHub Pages site (10 pages, generated)
│   ├── index.html                landing: the download, the 0.1563 answer, the honest status
│   ├── executive_summary.html    step-by-step how to submit, every rejection explained
│   ├── strategy.html             metric algebra, the attractor, the corridor, the experiment
│   ├── hypotheses.html           G-1..G-5 with layers, signature, rationale, cost
│   ├── intel.html                staff statements quoted verbatim + their thread URLs
│   ├── data.html                 access status, what is blocked, what is needed
│   ├── verification.html         the checks, and the seven problems found
│   ├── leaderboard.html          live snapshot read 2026-09-27
│   ├── research.html             the science, DOI-linked
│   ├── sources.html              every source with its verification status
│   ├── geotiff_writer.js         browser GeoTIFF + ZIP writer, round-trip self-checking
│   ├── downloads/                the one valid artifact + manifest.json
│   └── strategy_experiment.json  the multi-seed result, regenerated by the checks
├── src/gems/
│   ├── metric.py                 exact DTI, fast + naive, with the official fp_ignore_mask
│   ├── strategy.py               EQ-1/2/3, the attractor, binarise, build_corridor, sweeps
│   ├── pipeline.py               data loading, band-by-name, folds, model, holdout, final field
│   ├── features.py               one detector per hypothesis
│   └── io.py                     rasterio helpers
├── scripts/
│   ├── build_submission.py       the artifact, the unique name, the note, the gates
│   ├── validate_submission.py    14 hard gates; exit 1 means do not upload
│   ├── validate_holdout.py       blocked holdout gate  +  --strategy metric-shape experiment
│   ├── prepare_data.py           band inventory + grid cross-checks
│   ├── build_site.py             renders docs/ from site_data.py + live artifacts
│   ├── site_data.py              every claim the site may make, with its source URL
│   ├── verify_rules_quotes.py    source reachability + the quotes actually used
│   ├── download_competition_data.sh
│   └── run_all_checks.sh         one command, the whole audit
├── tests/test_validation.py      82 checks
├── requirements.txt
└── data/README.md
```

---

## How to run

```bash
bash scripts/run_all_checks.sh      # everything: deps, tests, experiment, artifact, site
```

Individually:

```bash
python tests/test_validation.py                              # 82 checks
python scripts/validate_holdout.py --strategy                # metric-shape experiment
python scripts/build_submission.py                           # rebuild the artifact
python scripts/validate_submission.py docs/downloads/*.tif  # 14 gates
python scripts/build_site.py                                 # re-render docs/
python scripts/build_site.py --check                         # fail if docs/ is stale
```

Nothing needs a GPU or the network.

---

## Competition facts, verified line by line

Every one of these was read from the linked page on 2026-09-27. The site quotes them
verbatim with the URL next to each quote.

- **Task** — [problem description](https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/):
  predict the presence of geological faults indicative of geothermal resources across the
  GeoDAWN region. The public USGS fault set "is not complete and may even contain some
  inaccurate data"; the test set is faults experts identified that are **not** in it.
- **Metric** — distance-weighted Tversky index, `k(d) = max(1 − d/R, 0)`, R = 300 m (3 px at
  100 m), α = 0.2 (false positives), β = 0.8 (false negatives). Official worked example:
  TP_w = 3.00, FP_w = 1.89, FN_w = 2.00 → 0.60.
- **Format** — single band, `float32`, values in [0, 1], EPSG:32611, 100 m, same bounds as
  the training data, 3292 × 3730 px, geotransform (100, 0, 243350, 0, −100, 4508550).
- **Rounds** — Initial: $50,000, a fixed private test set. Final: $250,000
  ($100K/$70K/$40K/$25K/$15K), rescored against an expanded label set that experts build
  partly *from the Phase 1 submissions*. One submission is scored in both rounds.
- **Cadence** — three scored submissions per
  [rolling 7-day window](https://community.drivendata.org/t/weekly-submissions/11524).
- **Leaderboard** — [read live 2026-09-27](https://www.drivendata.org/competitions/306/competition-doe-gems/leaderboard/):
  DARD 0.3049 (10 submissions), alexoktaba 0.2993, HardcoreTechGod 0.2854, mzoorob 0.2843,
  joeyfezster 0.2806. Ranks 24 and 25 sit on an identical 0.1563 with two submissions each.
- **External data** — allowed, provided the participant holds a licence permitting use in the
  challenge and sharing with the sponsor.

The group's own submissions — GEMSDOE1 0.1563, GEMSDOE2 0.1560, 5GEMSDOE 0.1563,
8GEMSDOE 0.1563, GEMSDOE3 0.1193 / 0.0830 / 0.1152, GEMSDOE4 0.0343, 6GEMSDOE 0.0286 — are
tabulated on the site. Their artifact hashes could not be checked from this environment, so
**no claim is made here about which files were byte-identical**; only about what the scores
are and what the closed form predicts they mean.

---

## Limitations, and what to do next

**Limitations, stated plainly**

1. **No model has been trained.** The data needs a login and this environment has no
   outbound network. Every detector is written and, where a closed form exists, unit-tested —
   and none has seen a real raster.
2. **The band inventory is unverified.** The problem description lists layer *groups*, not an
   ordered index. All band access is by name from the GeoTIFF tags and a miss raises.
3. **The 0.1563 explanation is a derivation, not a measurement.** It is consistent with every
   observation available and it is falsifiable, but the submitted files are not public.
4. **The mask-alignment risk is unquantified.** The scorer masks the organiser's rasterised
   catalogue; we are given `existing_faults.tif`. A one-pixel disagreement makes a
   supposedly-free pixel cost 1.0 of FP mass. `mask_safety_px` is the mitigation; the right
   value is unknown until someone can compare.
5. **The blocked holdout scores the wrong thing.** It scores *catalogue* faults; the
   competition scores faults that are *not* in the catalogue. No public holdout can measure
   that, because those labels do not exist publicly. The protocol can prove a detector
   carries information beyond "where the catalogue already is" — nothing more.
6. **The Rules PDF is not quoted.** `docs.nlr.gov` is unreachable from the shell, so the
   previous revision's claimed SHA256 and four quoted sentences could not be checked. They
   have been removed; the PDF is linked for review and nothing is attributed to it.

**Next session, in priority order**

1. **Place the data.** One command unlocks everything: download into `data/`, run
   `prepare_data.py`, confirm the band inventory, then `validate_holdout.py`. Everything
   downstream is already written.
2. **Run the real holdout for G-1 first.** It costs nothing but CPU and is the detector the
   official intel points at. Report the four arms; if `corridor` does not beat
   `catalogue-only`, the hypothesis is dead and no slot is spent.
3. **Calibrate the corridor on training folds only.** Sweep halo ∈ {0…6} px and paid budget
   ∈ {0.2 %…16 %}. The synthetic sweep has an interior optimum near 1 %; confirm or refute
   it on real data.
4. **Set `mask_safety_px`.** Compare the organiser's template against `existing_faults.tif`
   to bound the alignment error, then pick the safety dilation from that bound.
5. **Then G-2 and G-3**, which need no external data and target blind faults — a class the
   catalogue provably under-samples.
6. **Only then** consider G-4, and G-5 with a 10 m 3DEP DEM.
7. **Phase 2 is a different objective.** Experts build the expanded label set from the Phase 1
   submissions. A high-recall submission that is a superset of the Phase 1 truth converts
   into Phase 2 label additions. Do not over-tune to the public board, and do not submit the
   same file twice under different names.

---

## Project prompt (original request, preserved verbatim)

<details>
<summary>Expand</summary>

Review the repo. Here are the results from our groups submissions, separated by ....:

GEMSDOE1 — https://buffedlizard55-lab.github.io/GEMSDOE/docs/index.html — GEMSDOE SCORE: 0.1563

https://buffedlizard55-lab.github.io/6GEMSDOE/ — 6GEMSDOE SCORE: 0.0286

GEMSDOE3 — https://buffedlizard55-lab.github.io/GEMSDOE3/docs/index.html — 0.1193
1 · SUBMIT FIRST — f347b70daa — Pindrop nodes

GEMSDOE2 — https://buffedlizard55-lab.github.io/GEMSDOE2/docs/index.html — 0.1560

GEMSDOE3 — 0.0830 — 2 · SUBMIT SECOND — 37f9d5b855 — Pindrop catalogue-gap target SECOND SYSTEM

GEMSDOE4 — https://buffedlizard55-lab.github.io/GEMSDOE4/ — GEMSDOE 4 SCORE: 0.0343

GEMSDOE3 — 0.1152 — 3 · CONTROL · UPLOAD LAST — 4e03fc9705 — Pindrop dense ridge control

5GEMSDOE — https://buffedlizard55-lab.github.io/5GEMSDOE/docs/index.html — 5GEMSDOE SCORE: 0.1563

8GEMSDOE — https://buffedlizard55-lab.github.io/8GEMSDOE/ — 8GEMSDOE SCORE: 0.1563

The leaderboard: https://www.drivendata.org/competitions/306/competition-doe-gems/leaderboard/

We need to figure out why we keep scoring 0.1563, are we copying the same work over and over
again? We need to come up with different ideas, and not just the same idea tried a different
way. Need to figure out why 5GEMSDOE and GEMSDOE1 have the same score. We should not be
generating the same score submissions, they should all be unique. 0.3049 is the highest score
right now so we need to design a new strategy, research, testing, analyzing, and generating
submission system than the current website. It should be unique, take unique approaches to
generating a submission that can score higher than .3049. Put this prompt into the repo readme
and read it everytime we work on the project as a starting point to make sure we are building
what we are aiming for and have a strong base to continue building and improving on making
something useful for everyday use. It should solve the problem of having to manually check
everything ourselves and having an up to date current feed.

Review the repo. [Core Values and Own the Outcome as a focal point.]

Work line by line verifying from official verified trusted sources, provide links for manual
review. There should be no manual input, work on your own to complete tasks. Flag any
irregularities for review. No hallucinations. Verify no hallucinations.

The goal of this project is to place top of the leaderboard in this competition. The following
is the competition: https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/

We need to create a project that can compete and place top of the leaderboard. We need to
understand the problem, collect all the data and organize it into a clean easily auditable
table with official verified links for manual verification.

Get familiar with the problem through the overview and problem description.
https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/ You might also want
to reference additional resources available on the about page,
https://www.drivendata.org/competitions/306/competition-doe-gems/page/968/

Download the data from the data tab,
https://www.drivendata.org/competitions/306/competition-doe-gems/data/

Create and train your own model. This reference solution,
https://github.com/drivendataorg/gems-prize-reference-solution implements a simple approach.

Use your model to generate predictions that match the submission format.

Tell me what are you limitations and what you need access to during this project. We will need
to find free publicly available sources and data from official and verified sources if we are
to use 3rd party or external data.

This pdf outlines how submissions must be entered into the competition.
https://docs.nlr.gov/docs/fy26osti/96647.pdf

You must be able to do your own research, deep research, scientific literature research and
organize the knowledge so that we can critically think through the problem and generate a
solution through scientific and free publicly available information. This must be done
autonomously and must be constantly reviewed and improved upon. Provide suggestions and
improvements and implement them.

No DrivenData auth → cannot auto-download training_features.tif, labels.tif,
sample_submission.tif, 1m_DEM_links.csv from
https://www.drivendata.org/competitions/306/competition-doe-gems/data/ (verified redirect to
login).

See below for links from the above site: https://gdr.openei.org/submissions/1391

Download competition data from
https://www.drivendata.org/competitions/306/competition-doe-gems/data/ (requires login) to
data/

See links below for competition data:
https://www.dropbox.com/scl/fi/aemhtutjgcp6tr3tint94/GEMS_96647.pdf?rlkey=rek210cj2smnmzb8n0sla1vmd&st=wz4kofki&dl=0
https://www.dropbox.com/scl/fi/6rgvnuady818ol8yqgis4/example_submission.tif?rlkey=kbykilvau066xuogoosbf4cq8&st=8junzdyw&dl=0
https://www.dropbox.com/scl/fi/t7fyt03qdh9egyme0itwo/existing_faults.tif?rlkey=yiao96uluqdkipf0h5vju71jf&st=rnino7ya&dl=0
https://www.dropbox.com/scl/fi/3vz9o0wwavi26xaeoxlwr/gems-geodawn-numerical-features.tif?rlkey=je8d8fepqfbst9lnwsq9rkplu&st=zj1lag1r&dl=0
https://www.dropbox.com/scl/fi/ig0mban712ns1atphgphe/Digital-elevation-model-links-JSON.pdf?rlkey=zm77f1vbtt2if8hlruymptnu3&st=srhhir10&dl=0

Site creation: Create a github page for this repo that has clean ui, user friendly, simple and
easy to use. It should be organized and clean. It should include all relevant information in
an easy to read format with official verified links as sources for review. Work line by line
verify everything no hallucinations.

The single remaining blocker to training is data placement: run
`bash scripts/download_competition_data.sh` on any unrestricted machine into `data/`, then
`python scripts/prepare_data.py` — after that the full train→inference→validate pipeline is
ready to run (GPU needed for training; metric/losses/validation all verified working here on
CPU).

The site should be able to generate a TIF file that is required for submission. It should be as
easy as download to click a File to submit into the competition. This needs to be in the
executive summary or the very beginning of the site. It should be obvious when you visit the
site.

I tried to submit the document that I downloaded from the site but it returned this error on
the submission form: "Predicted values must be in range [0, 1]"

Also we need to give it a unique name and A short comment to help you or your team tell
submissions apart later e.g. clustering with k=25

Here is the submission page when I click submit file. New submission. File to submit. No file
chosen. You can submit a single-band GeoTIFF (.tif) file, or a .zip file containing a single
GeoTIFF, with your predictions. It must match the submission format's CRS, shape, and
geotransform. You may wish to review the competition rules first. Note (optional). A short
comment to help you or your team tell submissions apart later e.g. clustering with k=25

Create a executive summary subpage that explains exactly how to make a submission into the
contest.

Work on the next steps from the previous sessions first.

Go ahead and create a pull request and then merge the pull request onto the main. Make
suggestions for what work still needs to be done and any limitations that is in the way of a
successful project. It should be worked on in this next session or the next session. Work line
by line verify everything no hallucinations.

</details>

---

## License & attribution

DOE GEMS Prize Challenge entry. All data sources are cited with official DOIs and URLs in
`scripts/site_data.py` and on [`docs/sources.html`](docs/sources.html). Every number on the
site is either quoted from a linked primary source, read live on 2026-09-27, or computed by
code in this repository from the published metric.
