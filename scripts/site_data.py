"""
site_data.py — every fact the site is allowed to state, with its source.

This is the single place a claim enters the site.  If it is not in here, it does
not get rendered.  Each entry carries the URL it was read from and the date it
was read, so a reviewer can check every line.  Anything marked
`verified: False` is rendered with a visible UNVERIFIED banner instead of being
dropped -- an unverified claim that is flagged is useful, one that is hidden is
not.

Nothing here is inferred.  Derived quantities (the 0.1563 attractor recall, the
exchange rate) are computed in scripts/build_site.py from src/gems/strategy.py,
which is itself derived from the published metric, and are labelled "derived".
"""

VERIFIED_ON = "2026-09-27"

COMPETITION = {
    "name": "The Geologic Enhanced Mapping System (GEMS) Prize Challenge",
    "url": "https://www.drivendata.org/competitions/306/competition-doe-gems/",
    "overview": "https://www.drivendata.org/competitions/306/competition-doe-gems/",
    "problem": "https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/",
    "about": "https://www.drivendata.org/competitions/306/competition-doe-gems/page/968/",
    "data": "https://www.drivendata.org/competitions/306/competition-doe-gems/data/",
    "leaderboard": "https://www.drivendata.org/competitions/306/competition-doe-gems/leaderboard/",
    "submissions": "https://www.drivendata.org/competitions/306/competition-doe-gems/submissions/",
    "rules_pdf": "https://docs.nlr.gov/docs/fy26osti/96647.pdf",
    "reference_solution": "https://github.com/drivendataorg/gems-prize-reference-solution",
    "forum": "https://community.drivendata.org/c/gems-prize-challenge/111",
}

# --------------------------------------------------------------------------- #
# Staff statements, quoted verbatim, each with the thread it came from.
# --------------------------------------------------------------------------- #
STAFF_STATEMENTS = [
    {
        "id": "masking",
        "thread": "https://community.drivendata.org/t/scoring-clarification-are-known-usgs-ingenious-faults-masked-when-scoring-and-are-they-in-the-final-round-label-set/11516",
        "thread_title": "Scoring clarification: are known USGS/INGENIOUS faults masked when scoring?",
        "who": "chrisk-dd, DrivenData Staff",
        "when": "2026-09-16",
        "quote": "Pixels corresponding to known USGS/INGENIOUS faults are masked / excluded from evaluation, so they do not count towards penalty terms.",
        "also": "Re-evaluation will also mask/exclude the existing USGS/INGENIOUS faults.",
        "why_it_matters":
            "Masking zeroes a pixel's contribution to the FALSE-POSITIVE term only. "
            "The true-positive term takes a maximum over ALL pixels, masked or not. "
            "So a prediction placed on the known catalogue is free, and still earns "
            "credit for every withheld fault within 300 m of it.",
    },
    {
        "id": "new_fault_definition",
        "thread": "https://community.drivendata.org/t/where-do-you-draw-the-line/11536",
        "thread_title": "Where do you draw the line?",
        "who": "chrisk-dd, DrivenData Staff",
        "when": "2026-09-22",
        "quote": "For the purposes of this competition, “new fault” means “any fault pixel not already captured by USGS/INGENIOUS” and can include newly mapped geometry of an existing fault system.",
        "also": "",
        "why_it_matters":
            "This is the single most load-bearing sentence in the forum. It says the "
            "scored set is dominated by continuations past mapped tips, splays and "
            "parallel strands -- i.e. faults that sit next to, and collinear with, "
            "faults the catalogue already contains. The catalogue is a POINTER to the "
            "scored pixels, not just a distractor. It also appeared AFTER the previous "
            "repo revision was written, and it is not referenced anywhere in it.",
    },
    {
        "id": "phase2",
        "thread": "https://community.drivendata.org/t/how-were-the-new-test-faults-identified-data-sources-and-fault-types/11527",
        "thread_title": "How were the new test faults identified? (data sources and fault types)",
        "who": "chrisk-dd, DrivenData Staff",
        "when": "2026-09-23",
        "quote": "Note that the largest prize pool (Phase 2) will use a test set that is updated by expert review of all Phase 1 submissions, so your fault predictions have an impact on final evaluation even if they are not the most performant in Phase 1.",
        "also": ("We're not sharing details about the data sources, fault types, or "
                 "coverage behind the test faults beyond what's in the problem description."),
        "why_it_matters":
            "The $250,000 Final Prize Round ($100K / $70K / $40K / $25K / $15K) is scored "
            "against a label set that experts build partly FROM the Phase 1 submissions. "
            "Recall -- covering plausible new structure, even at the cost of precision -- "
            "converts directly into Phase 2 score, independently of Phase 1 rank.",
    },
    {
        "id": "cadence",
        "thread": "https://community.drivendata.org/t/weekly-submissions/11524",
        "thread_title": "Weekly Submissions",
        "who": "chrisk-dd, DrivenData Staff",
        "when": "2026-09-17",
        "quote": "The submission allowance resets based on a rolling window, not at a specific date and time.",
        "also": "",
        "why_it_matters":
            "Three scored submissions per rolling 7 days. There is no value in holding "
            "a slot for an unvalidated idea, and no penalty for using one early.",
    },
]

# --------------------------------------------------------------------------- #
# Problem description, quoted from the page.
# --------------------------------------------------------------------------- #
SPEC = {
    "crs": "projected coordinate system for UTM zone 11N, EPSG 32611",
    "resolution": "same resolution as the training data (100m)",
    "bounds": "same bounds as the training data, and data outside the bounds is null or nan",
    "layers": ("single layer with datatype of 32-bit float (float32) with values between "
               "0 and 1 indicating the confidence or probability of fault presence, with "
               "higher values indicating higher probability"),
    "template": ("A sample submission that predicts total fault absence is provided for "
                 "your reference on the data download page. You can use this as a template "
                 "to ensure that your submission is correctly formatted."),
    "kernel": "k(d) = max(1 - d/R, 0), where the range R is 300 meters (3 pixels at 100m resolution)",
    "alpha_beta": "For this competition, we set α = 0.2 and β = 0.8",
    "worked_example": "TP_w = 3.00, FP_w = 1.89, FN_w = 2.00  →  TI_w(α=0.2, β=0.8) = 0.60",
    "rounds": {
        "initial": "A fixed private set of new faults labeled by the sponsor's experts before the competition. Top 5 each win $10,000 ($50,000 pool).",
        "final": "An expanded label set: Initial Round labels plus previously-unknown faults that experts verify after reviewing every team's submission. Top 5 win $100K / $70K / $40K / $25K / $15K ($250,000 pool).",
        "single": "Competitors must choose a single submission for scoring across both rounds before the deadline, without knowing their private test set performance.",
    },
    "feature_groups": [
        "Surface conductivity and depth to conductive base surface",
        "Detrended elevation and the slope of detrended elevation",
        "Dilatation rate, shear strain rate, and the second invariant of the strain rate tensor",
        "Isostatic gravity anomaly and the slope of the isostatic gravity anomaly",
        "Magnetics including reduced-to-pole magnetic anomaly, total magnetic intensity, the vertical and horizontal slope of total magnetic intensity, and the top-of-crustal magnetic source depth estimate",
        "Density of earthquakes",
    ],
}

# --------------------------------------------------------------------------- #
# Leaderboard, read live 2026-09-27.
# --------------------------------------------------------------------------- #
LEADERBOARD = {
    "as_of": VERIFIED_ON,
    "source": COMPETITION["leaderboard"],
    "note": "Public leaderboard. Rank shown is by best public DW-Tversky. Submissions count is the participant's total.",
    "rows": [
        (1, "DARD", 10, 0.3049),
        (2, "alexoktaba", 14, 0.2993),
        (3, "HardcoreTechGod", 6, 0.2854),
        (4, "mzoorob", 16, 0.2843),
        (5, "joeyfezster", 13, 0.2806),
        (6, "GrigorSargsyan", 6, 0.2742),
        (7, "xiaofanhu", 2, 0.2367),
        (8, "exposed", 13, 0.2340),
        (9, "hiii12345", 6, 0.2262),
        (10, "tchu", 11, 0.2220),
        (11, "oshbocker", 16, 0.2185),
        (12, "moongrega", 10, 0.2174),
        (13, "ad3002", 12, 0.1994),
        (14, "Batik Shirt Brothers", 11, 0.1956),
        (15, "doegemsDrivendata", 5, 0.1847),
        (16, "ndavis7", 2, 0.1797),
        (17, "nsabaj", 3, 0.1744),
        (18, "jgaines", 12, 0.1694),
        (19, "dmitry_v", 3, 0.1672),
        (20, "hall4jm", 7, 0.1642),
        (21, "VictorCallejas", 1, 0.1629),
        (22, "Scotty77", 5, 0.1590),
        (23, "fishnchips", 7, 0.1587),
        (24, "extradr19", 2, 0.1563),
        (25, "SDCF9", 2, 0.1563),
    ],
    "observation": (
        "Ranks 24 and 25 sit at an identical 0.1563 with two submissions each. "
        "Three separate accounts are pinned at exactly that value to four decimals. "
        "That is not a coincidence of tuning -- see Strategy for the closed form that "
        "produces it."
    ),
}

# --------------------------------------------------------------------------- #
# Group's own past submissions, as reported by the team.
# --------------------------------------------------------------------------- #
GROUP_SUBMISSIONS = [
    # (label, repo/url, score, what the previous repo's own README said it was)
    ("GEMSDOE1", "https://buffedlizard55-lab.github.io/GEMSDOE/docs/index.html", 0.1563,
     "original artifact"),
    ("GEMSDOE2", "https://buffedlizard55-lab.github.io/GEMSDOE2/docs/index.html", 0.1560,
     "recall arm of the same artifact"),
    ("5GEMSDOE", "https://buffedlizard55-lab.github.io/5GEMSDOE/docs/index.html", 0.1563,
     "same artifact, different repo"),
    ("8GEMSDOE", "https://buffedlizard55-lab.github.io/8GEMSDOE/", 0.1563,
     "same artifact plus a catalogue union"),
    ("GEMSDOE3 a", "https://buffedlizard55-lab.github.io/GEMSDOE3/docs/index.html", 0.1193,
     "Pindrop nodes"),
    ("GEMSDOE3 b", "https://buffedlizard55-lab.github.io/GEMSDOE3/docs/index.html", 0.0830,
     "Pindrop catalogue-gap target"),
    ("GEMSDOE3 c", "https://buffedlizard55-lab.github.io/GEMSDOE3/docs/index.html", 0.1152,
     "Pindrop dense ridge control"),
    ("GEMSDOE4", "https://buffedlizard55-lab.github.io/GEMSDOE4/", 0.0343,
     "lineament + proxy labels"),
    ("6GEMSDOE", "https://buffedlizard55-lab.github.io/6GEMSDOE/", 0.0286,
     "gradient boosting over 88 channels, top 3%"),
]

# --------------------------------------------------------------------------- #
# Access status, measured in this environment.
# --------------------------------------------------------------------------- #
ACCESS = [
    {
        "item": "DrivenData competition data tab",
        "url": COMPETITION["data"],
        "status": "BLOCKED (login)",
        "detail": ("Requires a DrivenData account. Verified 2026-09-27: requesting the "
                   "data tab returns the /accounts/login/ page, so the rasters cannot be "
                   "fetched without credentials. No credentials are available here and "
                   "none will be requested or worked around. THIS, and not the network, "
                   "is the blocker."),
    },
    {
        "item": "Shell egress to drivendata.org / usgs.gov / openei.org / dropbox.com",
        "url": "https://gdr.openei.org/submissions/1391",
        "status": "BLOCKED (sandbox allowlist)",
        "detail": ("curl fails with SSL_ERROR_SYSCALL, exit 35. Measured 2026-09-27 on "
                   "14 hosts. This is a host allowlist, NOT a total network blackout -- "
                   "see the next row. The practical effect is the same: no raster can be "
                   "downloaded from this sandbox."),
    },
    {
        "item": "Shell egress to github.com / codeload.github.com / pypi.org",
        "url": "https://github.com/drivendataorg/gems-prize-reference-solution",
        "status": "OK",
        "detail": ("Measured 2026-09-27: HTTP 200. The official reference-solution "
                   "tarball (1,319,246 bytes) was downloaded from codeload.github.com and "
                   "is what exposed the nodata and filename defects fixed this session. "
                   "A previous revision of this site claimed the environment had no "
                   "outbound HTTPS at all; that was wrong and is corrected here."),
    },
    {
        "item": "DrivenData competition pages, forum, leaderboard",
        "url": COMPETITION["overview"],
        "status": "OK",
        "detail": "Reachable through the research fetch tool even though the shell cannot reach them. Every quote on this site was read from these pages on 2026-09-27.",
    },
    {
        "item": "Dropbox mirrors listed in the project brief",
        "url": "https://www.dropbox.com/scl/fi/3vz9o0wwavi26xaeoxlwr/gems-geodawn-numerical-features.tif?rlkey=je8d8fepqfbst9lnwsq9rkplu&st=zj1lag1r&dl=0",
        "status": "BLOCKED + UNVERIFIED PROVENANCE",
        "detail": ("Unreachable from the shell, and not linked from any official "
                   "DrivenData or DOE page. Treat as unverified third-party copies; the "
                   "data tab is the authoritative source."),
    },
    {
        "item": "INGENIOUS regional dataset compilation (DOE Geothermal Data Repository)",
        "url": "https://gdr.openei.org/submissions/1391",
        "status": "FREE / OFFICIAL / NOT YET FETCHED",
        "detail": ("DOI 10.15121/1881483, licence CC-BY 4.0, 116.98 MB across 9 files. "
                   "Read live 2026-09-27. This is the source for hypotheses H-4 "
                   "(multi-depth electrical conductance, DOI 10.5066/P9TWT2LU) and H-5 "
                   "(paleo-geothermal sinter/tufa 82 kB and Quaternary vents 9.4 MB). "
                   "Confirmed reachable through the fetch tool; the shell cannot download "
                   "it. Nothing in the pipeline depends on it."),
    },
    {
        "item": "USGS 3DEP 10 m DEM",
        "url": "https://apps.nationalmap.gov/3dep/",
        "status": "FREE / OFFICIAL / NOT NEEDED",
        "detail": ("Optional input for detector G-5. G-5 falls back to the competition's "
                   "own detrended-elevation layers, so the pipeline is not blocked on it."),
    },
    {
        "item": "USGS GeoDAWN survey (ScienceBase)",
        "url": "https://www.usgs.gov/data/geodawn-airborne-magnetic-and-radiometric-surveys-northwestern-great-basin-nevada-and",
        "status": "FREE / OFFICIAL / REDUNDANT",
        "detail": "The competition already ships derived GeoDAWN products. Use the data tab, not this.",
    },
]

# --------------------------------------------------------------------------- #
# Layer usage audit.  Reproduced by scripts/audit_layer_usage.py, which scans
# the real .band(...) calls in src/gems/ so this table cannot drift from code.
# Layer names are transcribed from the official problem description.
# --------------------------------------------------------------------------- #
LAYER_AUDIT = {
    "source": "https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/#provided-features",
    "read_on": "2026-09-27",
    "reproduce": "python scripts/audit_layer_usage.py",
    "unused_by_original": ["slope of detrended elevation", "shear strain rate",
                           "isostatic gravity anomaly",
                           "slope of the isostatic gravity anomaly",
                           "density of earthquakes"],
    "unused_after_h": [],
    "n_layers": 15,
    "note": ("G-1..G-5 left 5 of the 15 officially listed layers unread by any "
             "detector, and read 2 more (the vertical and horizontal slope of TMI) "
             "by recomputing them locally instead of using the product the "
             "organiser ships. Before this session's G-4 fix the count was 7 of 15. "
             "H-1..H-3 and the G-4 fix take it to 0 of 15."),
}

# --------------------------------------------------------------------------- #
# New candidate hypotheses, and what the spatially-blocked holdout said about
# them.  Reproduced by scripts/validate_hypotheses.py --select.
# --------------------------------------------------------------------------- #
HYPOTHESIS_VALIDATION = {
    "reproduce": "python scripts/validate_hypotheses.py --n 512 --folds 4 --select",
    "artifact": "docs/hypotheses_validation.json",
    "select_seed": 11,
    "eval_seeds": [23, 37, 41, 53, 67],
    "protocol": ("A configuration grid is declared in code, the best configuration per "
                 "detector is chosen on ONE phantom (seed 11), and the chosen "
                 "configuration is then scored on five phantoms that were never used "
                 "for selection. Reporting a chosen configuration's score on the "
                 "phantom it was chosen on is the exact selection bias that produced "
                 "the withdrawn +0.1379 corridor lift."),
    "baseline": "distance to the nearest catalogue pixel, exp(-d/2)",
    "results": [
        # key, name, mean dAUC on unseen phantoms, n wins / n seeds
        ("H-3", "Gravity-gradient lineament [no catalogue suppression]", +0.0525, "4/5"),
        ("H-1", "Seismogenic organisation [suppression tau=2]", +0.0291, "4/5"),
        ("G-5", "Fluvial knickpoint and drainage deflection", +0.0101, "3/5"),
        ("H-2", "Strike-integrated scarp step filter [no suppression]", -0.0773, "1/5"),
        ("G-3", "Clay-cap conductivity edge", -0.1140, "0/5"),
        ("G-1", "Catalogue geometry completion", -0.1618, "0/5"),
        ("G-4", "Magnetic contact edge", -0.2078, "0/5"),
        ("G-2", "Geodetic dilation-tendency ridge", -0.2183, "0/5"),
    ],
    "verdict": ("NO CANDIDATE CLEARS THE BAR. The best result is +0.053 AUC over "
                "proximity on 4 of 5 phantoms, and the second best is +0.029. Both are "
                "small relative to the spread the phantoms themselves produce -- one "
                "seed on H-1 is -0.092 -- and the number moved by 0.014 when a "
                "numerical defect in the harness was fixed, which is a direct measure "
                "of how little separates these candidates. No submission slot should be "
                "spent on any of them on the strength of this experiment."),
    "headline_finding": ("H-1 looked like the clear winner on the selection phantom "
                         "(+0.102 AUC) and collapsed to +0.029 on unseen ones, while "
                         "H-2 -- the top candidate by a-priori reasoning -- is negative "
                         "on both. The ranking does not survive contact with a second "
                         "phantom. That is the protocol working, not failing."),
    "caveat": ("The phantom's physics are OUR forward model. H-1 wins partly because "
               "seismicity was generated as a fault-parallel ribbon, and H-3 partly "
               "because every fault was given a gravity step. Both are the most "
               "favourable possible construction for those detectors. Decoys were "
               "included so a win means something, but a win here is necessary and "
               "never sufficient. Nothing in this output is evidence about the "
               "GeoDAWN region."),
}

# --------------------------------------------------------------------------- #
# Irregularities. Stated plainly rather than smoothed over.
# --------------------------------------------------------------------------- #
IRREGULARITIES = [
    {
        "title": "The previous site shipped GeoTIFFs that GDAL could not read",
        "id": "tiff",
        "severity": "HIGH",
        "found": "2026-09-27, this session",
        "detail": (
            "docs/geotiff_writer.js allocated the StripOffsets (tag 273) and StripByteCounts "
            "(tag 279) arrays as 'inline' 4-byte IFD values before their data existed, so "
            "only the first 4 bytes of each array ever reached the file. Strips 2..N pointed "
            "at offset 0. Reproduced: a 49,117,032-byte file from the site's default "
            "'Build submission.tif' button, which rasterio/GDAL refused to open with "
            "'TIFFReadEncodedStrip() failed'. The DrivenData form reported that as a VALUE "
            "error -- 'Predicted values must be in range [0, 1]' -- which is why the earlier "
            "diagnosis blamed NaN. There was no NaN problem to find. The writer is rewritten, "
            "round-trip tested against an independent parser, and gated by "
            "tests/test_validation.py E1-E6."
        ),
    },
    {
        "title": "The previous site shipped two byte-identical .tif files under different names",
        "id": "duplicate-artifacts",
        "severity": "HIGH",
        "found": "2026-09-27, this session",
        "detail": (
            "docs/downloads/gemsdoe9_submission.tif and "
            "docs/downloads/gems9-20260926T234507Z-8ecbdc71.tif were both 674,937 bytes with "
            "SHA256 8ecbdc712da4b83e.... This is the exact failure the team asked to stop: two "
            "files, one pixel field, indistinguishable scores. Both have been removed; "
            "scripts/build_submission.py now deletes previous artifacts before writing and "
            "refuses to publish any file whose SHA256 is on the spent-slot list."
        ),
    },
    {
        "title": "The browser 'H9-1 relay-stepover' field was a procedural pattern, not a model",
        "id": "placeholder-labelled-as-science",
        "severity": "HIGH",
        "found": "2026-09-27, this session",
        "detail": (
            "generateH91Field() in the old geotiff_writer.js built a fixed-seed field from two "
            "crossing sinusoids plus an xorshift noise term. No GeoDAWN data, no catalogue, no "
            "model. The site nevertheless labelled the resulting file 'H9-1 relay-stepover + "
            "dilation + intersection' and the note pasted into DrivenData repeated it. The "
            "artifact is gone. The replacement is named PLACEHOLDER, its note says 'no model, "
            "no GeoDAWN data', and nothing in this repo presents an unvalidated field as a "
            "geological result."
        ),
    },
    {
        "title": "The previous dilation tendency was not the published quantity",
        "id": "dilation-tendency",
        "severity": "MEDIUM",
        "found": "2026-09-27, this session",
        "detail": (
            "src/gems/features.py computed 0.5*(1 + dilatation/(shear + |second_invariant|)). "
            "The dilation tendency of Simpson & Reiling (2008) is Td = (e1 - e_n)/(e1 - e3), "
            "which is 0.5 in pure simple shear and 1 in pure uniaxial extension, and is "
            "invariant under a uniform scaling of the strain rate. The old formula has none "
            "of those properties. Replaced with the closed form Td = 0.5 + T/(4*sqrt(I2 - T^2/4)) "
            "and pinned by four limit tests."
        ),
    },
    {
        "title": "Claims about the competition rasters that this repo cannot currently support",
        "id": "raster",
        "severity": "MEDIUM",
        "found": "2026-09-27, this session",
        "detail": (
            "The previous revision asserted, as verified, that training_features.tif is "
            "399.5 MB / 19 bands / SHA256 4371c82e..., that existing_faults.tif has 60,988 "
            "pixels, and that example_submission.tif is 'bit-identical to the label raster'. "
            "None of that is checkable from this repository, and the last one contradicts the "
            "problem description, which says the sample submission 'predicts total fault "
            "absence'. These numbers are removed from the site. The band count is read from the "
            "file's own tags at load time instead (pipeline.band_index_by_tag / "
            "CompetitionData.band_names)."
        ),
    },
    {
        "title": "Band ordering was never verified",
        "id": "band-order",
        "severity": "MEDIUM",
        "found": "2026-09-27, this session",
        "detail": (
            "The previous features.py hard-coded band indices 0..18 and labelled them "
            "'PROVISIONAL'. A provisional index that is wrong is a silent wrong answer. All "
            "band access now goes through CompetitionData.band(*substrings), which resolves "
            "by the GeoTIFF's own description tags and raises a KeyError listing the available "
            "names when a lookup misses."
        ),
    },
    {
        "title": "No project Python environment was declared",
        "id": "no-requirements",
        "severity": "LOW",
        "found": "2026-09-27, this session",
        "detail": (
            "src/gems/features.py and src/gems/metric.py imported scipy at module scope with no "
            "requirements file, so `python tests/test_validation.py` failed on a clean checkout. "
            "requirements.txt now pins the five dependencies, and run_all_checks.sh verifies the "
            "imports before running anything."
        ),
    },
]

# --------------------------------------------------------------------------- #
# Scientific literature actually used, with DOIs/URLs.
# --------------------------------------------------------------------------- #
LITERATURE = [
    {
        "who": "Simpson, R. W. & Reiling, T. A.",
        "year": 2008,
        "what": "Dilation tendency, the ratio (e1 - e_n)/(e1 - e3), and the inverse dilation tendency used to separate strike-slip from normal and reverse faulting.",
        "where": "Tectonophysics 456(1-2):41-49",
        "url": "https://doi.org/10.1016/j.tecto.2008.07.009",
        "used_for": "Detector G-2. Supplies the closed form Td = 0.5 + T/(4D) and its limits.",
    },
    {
        "who": "Faulds, J. E., Hinz, M. D. & Kreemer, M. W.",
        "year": 2012,
        "what": "Structural controls of geothermal systems in the Great Basin, including where geothermal fluids focus along fault stepovers, terminations and intersections.",
        "where": "Workshop on Geothermal Reservoir Engineering, Stanford University",
        "url": "https://gdr.openei.org/files/383/Faulds%20et%20al%202012%20GeoNZ%20Paper.pdf",
        "used_for": "Context for targeting stepovers, terminations and intersections. NOTE: the exact percentages quoted in the previous revision of this README are NOT carried forward -- the figure could not be re-verified against the primary text, so the claim is dropped rather than repeated.",
    },
    {
        "who": "Mattéo, L., Manighetti, I., Tarabalka, Y., Gaucel, J.-M., van den Ende, M., Mercier, A., et al.",
        "year": 2021,
        "what": "Automatic fault mapping in remote optical images and topographic data with deep learning.",
        "where": "J. Geophys. Res. Solid Earth 126, e2020JB021269",
        "url": "https://doi.org/10.1029/2020JB021269",
        "used_for": "Cited by the competition's own About page. Baseline for what published automatic fault mapping achieves.",
    },
    {
        "who": "Hermant, B., Kiersnowski, L. & Bellanger, M.",
        "year": 2025,
        "what": "Using deep learning to map Quaternary faults in Western USA.",
        "where": "50th Workshop on Geothermal Reservoir Engineering, Stanford University",
        "url": "https://pangea.stanford.edu/ERE/db/GeoConf/papers/SGW/2025/Hermant.pdf",
        "used_for": "Cited by the competition's About page. Directly relevant: a state-scale Quaternary fault mapping result over the same region.",
    },
    {
        "who": "DrivenData (sponsor documentation)",
        "year": 2026,
        "what": "Official reference solution, linked from the problem description. "
                "One public commit (Jun 16 2026) by Prof. John Lipor: a single "
                "notebook, unet-mc-cv-reference-solution.ipynb — a UNet with Monte "
                "Carlo cross-validation. Verified from the repository page; the "
                "notebook's internals were not read.",
        "where": "github.com/drivendataorg/gems-prize-reference-solution",
        "url": COMPETITION["reference_solution"],
        "used_for": "Confirms a deep UNet is the sponsor's own baseline, and that "
                    "this repo's band-by-tag resolution (pipeline.load_competition) "
                    "is a deliberate alternative to hard-coded band indices — our "
                    "choice, not a claim about the notebook.",
    },
]
