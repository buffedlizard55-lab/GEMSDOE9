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
| **Is it a prediction?** | **No.** It is a format-check placeholder. The competition rasters are behind a DrivenData login, so **no model has been trained**. |
| **Should you spend a submission slot on the new ideas?** | **No.** Five new candidates were written and put through a spatially-blocked holdout. The best cleared the proximity baseline by **+0.053 AUC on 4 of 5 phantoms**, the second best by +0.029. Both are inside the noise. See [New candidates](#new-candidates-h-1--h-5-and-what-the-holdout-said). |
| **What *is* proven?** | The metric algebra, the submission shape, and 99 automated checks. `bash scripts/run_all_checks.sh` reproduces all of it. |
| **Single biggest lever** | 0.1563 is a catalogue copy. 0.3049 needs **2.01×** the recall. See [Why 0.1563](#why-01563-keeps-repeating). |
| **What was actually blocking us** | Not the network. Five of the fifteen layers the organisers ship were read by **no detector at all**. See [The layer audit](#the-layer-audit). |

---

## This session: what was verified, what was wrong, what changed

Every claim below was produced by running something in this checkout on 2026-09-27.
Where a previous revision of this README asserted something that turned out to be
false, the false claim is named.

### 1. The "no outbound network" claim was false, and it was hiding a fix

The previous README and `docs/data.html` both said the environment *"denies outbound
HTTPS from the shell entirely (`curl` exit 35), so no URL resolves — official or
mirrored."* Measured, host by host:

| Host | Result |
|---|---|
| `github.com`, `codeload.github.com` | **HTTP 200** — the official reference-solution tarball (1,319,246 bytes) downloaded |
| `pypi.org`, `files.pythonhosted.org` | **HTTP 200** — `pip install -r requirements.txt` works |
| `drivendata.org`, `gdr.openei.org`, `usgs.gov`, `dropbox.com`, `s3.amazonaws.com` | `curl` exit 35 (`SSL_ERROR_SYSCALL`) |

It is a **host allowlist, not a blackout**. The distinction mattered: downloading the
official reference solution is what exposed bugs 2 and 3 below. And the DrivenData
pages *are* readable through the research fetch tool, which is how every quote on the
site was verified — so "no URL resolves" was wrong in two separate ways.

**The real blocker is unchanged and is not the network:** the data tab redirects to
`/accounts/login/` (verified 2026-09-27). No credentials, no workaround.

### 2. `load_competition` never masked the nodata sentinel — it would have destroyed every gradient detector

The official reference solution does `X_orig[X_orig < -1e38] = np.nan` before anything
else. We did not. Reproduced directly through our own functions:

```
_grad_mag on a patch containing a -3.4e38 footprint edge  ->  max = inf
_structure_coherence on the same patch                    ->  overflow to NaN
```

`inf` propagates into every gradient- and coherence-based feature: G-3, G-4, G-5 and
the pipeline's feature stack. **No submission built from this code on real data would
have been interpretable, and nothing would have raised.** Fixed in
`pipeline.load_competition`, and `_grad_mag` / `_structure_coherence` now nearest-fill
NaN before differentiating so a footprint boundary does not become a fake edge.
Pinned by tests K1–K3.

### 3. The pipeline would not have found the files

Three names are in circulation for the same two rasters, verified against three
official sources:

| Source | Feature raster | Label raster |
|---|---|---|
| [Problem description](https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/) | `training_features.tif` | "vector and raster formats" |
| [Official reference solution](https://github.com/drivendataorg/gems-prize-reference-solution) | `data/numeric_features.tif` | `data/labels.tif` |
| Links circulated with the brief | `gems-geodawn-numerical-features.tif` | `existing_faults.tif` |

`load_competition` hard-coded one pair and raised `FileNotFoundError` on the others.
It now resolves any of them. Pinned by tests K4–K5.

### 4. `ndarray.ptp()` was removed in NumPy 2.0

`requirements.txt` permits `numpy>=1.26`; the installed interpreter has 2.4.6. Five
calls in the new detector code raised `AttributeError` on first run and were being
swallowed by a broad `except`. Replaced with a `_minmax()` helper. Pinned by test K6,
which parses with `ast` so a docstring mentioning `a.ptp()` cannot mask a real call.

### 5. G-4 was recomputing two products the organisers already ship

`g4_magnetic_contact` computed the vertical and horizontal slope of TMI locally and
never read the bands provided. Worse, a local variable named `vertical` held the
gradient magnitude of the *RTP* field, not a vertical derivative. It now prefers the
shipped bands and falls back to a local gradient only if they are absent. Pinned by
test K10, which swaps the bands and asserts the output moves.

---

## The layer audit

The cheapest untried signal was not a new dataset. It was data we already had.
`python scripts/audit_layer_usage.py` scans the real `.band(...)` calls in `src/gems/`
and reports which of the [15 layers the organisers list](https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/#provided-features)
no detector reads. Run on the code as it stood at the start of this session:

**5 of 15 layers were read by no detector at all** — `shear strain rate`, `isostatic
gravity anomaly`, `slope of the isostatic gravity anomaly`, `density of earthquakes`,
`slope of detrended elevation` — and 2 more (`vertical`/`horizontal slope of TMI`) were
recomputed locally instead of read. That is 7 of 15.

After the G-4 fix and H-1…H-3: **0 of 15**. The count is computed, not asserted, so it
cannot drift from the code.

---

## New candidates H-1 … H-5, and what the holdout said

Full detail on [`docs/hypotheses.html`](docs/hypotheses.html); implemented in
[`src/gems/hypotheses.py`](src/gems/hypotheses.py).

| # | Candidate | Layers | Signature | Cost |
|---|---|---|---|---|
| **H-1** | Seismogenic organisation | earthquake density + shear strain rate *(both previously unread)* | seismicity structure-tensor coherence × shear strain rate, catalogue suppressed | Low |
| **H-2** | Strike-integrated scarp step | detrended elevation + its shipped slope layer | max over 8 azimuths of the *asymmetry* of the cross-strike slope, after 400 m along-strike integration | Low |
| **H-3** | Gravity-gradient lineament | isostatic gravity anomaly + its shipped slope *(both previously unread)* | HGM gated by along-strike gradient-azimuth coherence | Low |
| **H-4** | Multi-depth conductance persistence | INGENIOUS conductance, 5 depths 2–200 km — **external** | cross-depth azimuth persistence of the conductance gradient | Med + data |
| **H-5** | Hydrothermal expression alignment | INGENIOUS sinter/tufa + Quaternary vents — **external** | point-lineament detector over discharge points, normalised by local density | Med + data |

H-4 and H-5 are the fix for a failure mode G-3's own docstring admits — *any* lithologic
contact makes a conductivity edge — and the only candidates whose evidence is the
geothermal system itself rather than a proxy for structure. Both raise a `ValueError`
naming their DOI if the data is absent rather than silently degrading into a different
detector (test K9).

### Validated before a slot was spent — and the answer was no

```bash
python scripts/validate_hypotheses.py --n 512 --folds 4 --select
```

A configuration grid is declared in code, the best configuration per detector is chosen
on **one** phantom (seed 11), and the chosen configuration is then scored on **five
phantoms never used for selection**. The phantom contains deliberate decoys — symmetric
ridgelines, a regional monocline, a diffuse seismic swarm, and lithologic contacts with
a density step but no fault — so a win means the operator separates a fault from things
that look like one, not that it inverts its own generator.

| Candidate | mean ΔAUC over proximity (unseen phantoms) | beat baseline |
|---|---|---|
| **H-3** Gravity-gradient lineament | **+0.0525** | 4/5 |
| **H-1** Seismogenic organisation | **+0.0291** | 4/5 |
| G-5 Fluvial knickpoint | +0.0101 | 3/5 |
| **H-2** Scarp step filter | **−0.0773** | 1/5 |
| G-3 Clay-cap conductivity edge | −0.1140 | 0/5 |
| G-1 Catalogue geometry completion | −0.1618 | 0/5 |
| G-4 Magnetic contact edge | −0.2078 | 0/5 |
| G-2 Dilation-tendency ridge | −0.2183 | 0/5 |

**Three things worth more than the table.**

1. **The a-priori ranking was wrong.** H-2 was ranked first on reasoning — no external
   data, one layer, targets the dominant fault type in the region. It came last but one.
   Ranking hypotheses is not validating them.
2. **The ranking does not survive a second phantom.** On the selection phantom H-1
   looked like a clear winner at **+0.102**. On unseen phantoms it collapsed to
   **+0.030** and H-3 overtook it. That is precisely the selection bias that produced
   the withdrawn `+0.1379` corridor lift in the previous revision — now caught by the
   protocol rather than by a reviewer.
3. **Verdict: do not spend a submission slot on any of these.** +0.053 AUC on 4 of 5
   phantoms is small next to the spread the phantoms themselves produce — one seed on
   H-1 is −0.092 — and the headline number moved by **0.014** when a numerical defect in
   the harness was fixed. A candidate whose measured advantage is smaller than the
   effect of fixing a bug in the measuring instrument is not a result.

**And the caveat that limits all of it:** the phantom's physics are *our* forward model.
H-1 wins partly because seismicity was generated as a fault-parallel ribbon, and H-3
partly because every fault was given a gravity step — the most favourable possible
construction for each. A win here is necessary and never sufficient. Nothing in this
experiment is evidence about the GeoDAWN region.

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
an account. That is the blocker, not the network: measured host by host on 2026-09-27,
`github.com`, `codeload.github.com` and `pypi.org` all return HTTP 200, while
`drivendata.org`, `gdr.openei.org`, `usgs.gov` and `dropbox.com` fail with `curl` exit 35.
It is a host allowlist, not a blackout — and the allowlist is what let us download the
official reference solution and find the nodata bug. The Dropbox links in the brief are
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
│   ├── features.py               one detector per hypothesis (G-1..G-5)
│   ├── hypotheses.py             the new candidates (H-1..H-5)
│   └── io.py                     rasterio helpers
├── scripts/
│   ├── build_submission.py       the artifact, the unique name, the note, the gates
│   ├── validate_submission.py    14 hard gates; exit 1 means do not upload
│   ├── validate_holdout.py       blocked holdout gate  +  --strategy metric-shape experiment
│   ├── validate_hypotheses.py    spatially-blocked holdout for H-1..H-5; --select is the gate
│   ├── audit_layer_usage.py      which official layers no detector reads
│   ├── prepare_data.py           band inventory + grid cross-checks
│   ├── build_site.py             renders docs/ from site_data.py + live artifacts
│   ├── site_data.py              every claim the site may make, with its source URL
│   ├── verify_rules_quotes.py    source reachability + the quotes actually used
│   ├── download_competition_data.sh
│   └── run_all_checks.sh         one command, the whole audit
├── tests/test_validation.py      99 checks
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
python tests/test_validation.py                              # 99 checks
python scripts/audit_layer_usage.py                          # which layers no detector reads
python scripts/validate_hypotheses.py --select               # the H-1..H-5 gate
python scripts/validate_holdout.py --strategy                # metric-shape experiment
python scripts/build_submission.py                           # rebuild the artifact
python scripts/validate_submission.py docs/downloads/*.tif  # 14 gates
python scripts/build_site.py                                 # re-render docs/
python scripts/build_site.py --check                         # fail if docs/ is stale
```

Nothing needs a GPU. `pip install -r requirements.txt` is enough — verified on a clean
checkout this session (numpy 2.4.6, scipy 1.17.1, scikit-learn 1.9.1, rasterio 1.4.4).

---

## Verified intel, with the links to check it yourself

Read live on 2026-09-27. Each row is quoted verbatim on
[`docs/intel.html`](docs/intel.html) next to its URL.

| What | Who | Source |
|---|---|---|
| Catalogue pixels are masked out of the penalty terms | chrisk-dd, DrivenData Staff, 2026-09-16 | [thread 11516](https://community.drivendata.org/t/scoring-clarification-are-known-usgs-ingenious-faults-masked-when-scoring-and-are-they-in-the-final-round-label-set/11516) |
| "new fault" = any fault pixel not already captured by USGS/INGENIOUS, **including newly mapped geometry of an existing system** | chrisk-dd, Staff, 2026-09-23 | [thread 11536](https://community.drivendata.org/t/where-do-you-draw-the-line/11536) |
| **The label sources, fault types and coverage will not be disclosed** | chrisk-dd, Staff, 2026-09-23 | [thread 11527](https://community.drivendata.org/t/how-were-the-new-test-faults-identified-data-sources-and-fault-types/11527) |
| Phase 2 — the $250K pool — is scored on a label set built from **expert review of all Phase 1 submissions** | chrisk-dd, Staff, 2026-09-23 | [thread 11527](https://community.drivendata.org/t/how-were-the-new-test-faults-identified-data-sources-and-fault-types/11527) |
| Submission allowance resets on a rolling window | chrisk-dd, Staff, 2026-09-17 | [thread 11524](https://community.drivendata.org/t/weekly-submissions/11524) |
| The label raster has **one** band; the "Band 19" print in the reference notebook is a bug | chrisk-dd, Staff, 2026-09-23 | [thread 11529](https://community.drivendata.org/t/why-does-the-training-fault-labels-file-in-the-data-tab-have-a-single-band-while-the-labels-in-the-reference-solution-repo-have-19-bands/11529) |

**Two of these change strategy and neither was in the previous README.**

- **The label sources are closed.** Any plan that depends on reverse-engineering how the
  experts mapped the test faults is dead. What *is* open is the layer inventory, which is
  exactly where the layer audit found the untried signal.
- **Phase 2 is built from our own submissions.** Recall converts into Phase 2 label
  additions independently of Phase 1 rank. A submission that is a defensible superset of
  plausible new structure is worth more than one tuned to the public board — which is an
  argument for the corridor shape and against over-fitting.

**On the band count.** The reference notebook's feature loop is
`for i in range(1, src.count + 1)`, and its label cell then prints `f"Band {i}"` using the
leaked loop variable. Staff confirm that value is **19**. So `training_features.tif` has
19 bands and the label raster has 1. That is an *inference from a corroborated chain*, not
an official statement, and it is labelled as such: `prepare_data.py` still reads the real
per-band tags and never trusts it.

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

1. **No model has been trained.** The data tab needs a DrivenData login. Every detector is
   written and unit-tested — and none has seen a real raster.
2. **The band inventory is unverified.** The problem description lists layer *groups*, not an
   ordered index. All band access is by name from the GeoTIFF tags and a miss raises. The
   19-band figure is an inference (see above), not an official statement, and the code does
   not depend on it.
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
6. **The H-1…H-5 validation is on synthetic physics we invented.** It killed H-2 and showed
   the ranking is unstable, which is worth knowing. It is not evidence that H-1 or H-3 works
   in Nevada, and the phantom was constructed in each detector's favour.
7. **The Rules PDF is not quoted.** `docs.nlr.gov` is not on the sandbox allowlist
   (`curl` exit 35), so the previous revision's claimed SHA256 and four quoted sentences
   could not be checked. They have been removed; the PDF is linked for review and nothing is
   attributed to it.
8. **H-4 and H-5 have never been run.** They need two small INGENIOUS files
   ([DOI 10.15121/1881483](https://gdr.openei.org/submissions/1391), CC-BY 4.0, 82 kB and
   9.4 MB). The sources are verified to exist and to be free; the shell cannot download
   them. Both raise rather than silently degrade.

**Next session, in priority order**

1. **Place the data.** Still the single unlock: download into `data/`, run
   `prepare_data.py`, confirm the band inventory, then `validate_holdout.py`. Everything
   downstream is already written. Any of the three filename conventions will resolve.
2. **Re-run the layer audit against the real band tags.**
   `python scripts/audit_layer_usage.py` currently reports against the layer *names* in the
   problem description. On real data, confirm each name actually matches a band tag — a name
   that does not match raises, which is the intended behaviour, but it means the detector is
   silently absent until it is fixed.
3. **Run the real holdout for H-3 and H-1 first**, not G-1. On synthetic physics H-3
   (+0.039) and H-1 (+0.030) were the only candidates that beat proximity; G-1 was
   −0.16. **If a detector does not beat distance-to-catalogue on real data, it is dead and
   no slot is spent.** Use `validate_hypotheses.py --select`, not a single phantom.
4. **Fetch the two INGENIOUS files** and run H-4 and H-5 for real. They are the only
   candidates that fix G-3's admitted false-positive problem, they are small, and they are
   the only ones whose evidence is the geothermal system itself.
5. **Calibrate the corridor on training folds only.** Sweep halo ∈ {0…6} px and paid budget
   ∈ {0.2 %…16 %}. The synthetic sweep has an interior optimum near 1 %; confirm or refute
   it on real data.
6. **Set `mask_safety_px`.** Compare the organiser's template against `existing_faults.tif`
   to bound the alignment error, then pick the safety dilation from that bound.
7. **Phase 2 is a different objective.** Experts build the expanded label set from the Phase 1
   submissions ([thread 11527](https://community.drivendata.org/t/how-were-the-new-test-faults-identified-data-sources-and-fault-types/11527),
   staff, verified). A high-recall submission that is a defensible superset of plausible new
   structure converts into Phase 2 label additions. Do not over-tune to the public board,
   and do not submit the same file twice under different names.

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
