#!/usr/bin/env python3
"""
validate_hypotheses.py — the spatially-blocked holdout for candidates H-1..H-5.

    python scripts/validate_hypotheses.py                 # default, 768x768
    python scripts/validate_hypotheses.py --n 1024 --folds 16
    python scripts/validate_hypotheses.py --dti           # add the DTI corridor arm

WHAT IT CAN AND CANNOT PROVE — READ THIS BEFORE QUOTING A NUMBER
----------------------------------------------------------------
The competition rasters are behind a DrivenData login and cannot be downloaded
from this environment (verified 2026-09-27: the data tab redirects to
/accounts/login/).  So this script does NOT touch Nevada.

It builds a synthetic region whose PHYSICS follow each hypothesis's own forward
model — a normal-fault scarp is a step, a blind fault is a density/conductivity
discontinuity, an active fault has organised seismicity — and then asks three
questions that a synthetic region CAN answer honestly:

  1. Does the operator discriminate a fault from its own false positives?  The
     phantom contains deliberate DECOYS that produce the same raw signature and
     are not faults: symmetric ridgelines and a regional monocline (decoys for
     the elevation step filter), a diffuse seismic swarm (decoy for H-1), and
     lithologic contacts with a density/conductivity step but no fault (decoys
     for H-3 and G-3).  Without decoys this test would only prove that an
     operator inverts its own forward model, which is circular and worthless.

  2. Does it add anything BEYOND "be near a known fault"?  The baseline is
     distance-to-catalogue.  A detector that cannot beat it is worthless
     whatever its AUC, because the corridor already encodes proximity.

  3. Is that stable across space?  The phantom is blocked into a grid and every
     detector is scored on held-out blocks only.

WHAT IT CANNOT PROVE: that any of this is true of the GeoDAWN region.  The
forward model is ours, not the Earth's.  A detector can win every number here
and still be worthless on real data.  This is a filter that kills bad operators
cheaply; it is not evidence.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from gems.metric import distance_weighted_tversky_fast          # noqa: E402
from gems.pipeline import CompetitionData                        # noqa: E402
from gems import features as F                                   # noqa: E402
from gems import hypotheses as H                                 # noqa: E402
from scipy.ndimage import distance_transform_edt, gaussian_filter  # noqa: E402


# --------------------------------------------------------------------------- #
# phantom: a synthetic Basin-and-Range-flavoured region
# --------------------------------------------------------------------------- #
def _segment_field(shape, cx, cy, az_deg, length, width_fn):
    """A smooth field that is a function of signed distance to a line segment."""
    Hh, Ww = shape
    yy, xx = np.mgrid[0:Hh, 0:Ww].astype(np.float64)
    t = np.deg2rad(az_deg)
    ux, uy = np.cos(t), np.sin(t)              # along strike
    dx, dy = xx - cx, yy - cy
    along = dx * ux + dy * uy
    across = -dx * uy + dy * ux
    half = length / 2.0
    inside = np.abs(along) <= half
    # signed distance to the segment: across when beside it, else to an endpoint
    d_seg = np.where(inside, np.abs(across),
                     np.hypot(np.maximum(np.abs(along) - half, 0.0), across))
    signed = np.sign(across + 1e-12) * d_seg
    return width_fn(signed, along, half), inside, across, along


def make_region(n=768, seed=11, n_faults=26):
    """Return (CompetitionData, gt_new, catalogue, notes)."""
    rng = np.random.default_rng(seed)
    sh = (n, n)

    elev = np.zeros(sh)
    eq = np.zeros(sh)
    shear = np.zeros(sh)
    grav = np.zeros(sh)
    cond = np.zeros(sh)
    catalogue = np.zeros(sh, dtype=bool)
    gt_new = np.zeros(sh, dtype=bool)
    notes = {"mapped": 0, "unmapped": 0, "tip_continuations": 0,
             "parallel_strands": 0, "decoy_ridges": 0, "decoy_swarm": 0,
             "decoy_contacts": 0}

    def trace(mask_az, cx, cy, length, thick=0.9):
        """Rasterise a 1-2 px line segment into `mask`."""
        t = np.deg2rad(mask_az)
        s = np.linspace(-length / 2, length / 2, int(length) * 2)
        for v in s:
            r = int(round(cy + v * np.sin(t)))
            c = int(round(cx + v * np.cos(t)))
            for dr in (0, 1):
                for dc in (0, 1):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < n and 0 <= cc < n:
                        mask_az_out[0][rr, cc] = True

    faults = []
    for i in range(n_faults):
        cx = rng.uniform(0.15, 0.85) * n
        cy = rng.uniform(0.15, 0.85) * n
        az = rng.normal(340, 22) % 180.0          # Basin-and-Range-ish, varied
        length = rng.uniform(70, 230)
        step = rng.uniform(0.4, 1.4)              # scarp relief
        active = rng.random() < 0.6               # is it seismogenic?
        blind = rng.random() < 0.35               # buried: no scarp
        mapped = rng.random() < 0.60
        # colluvial burial: an along-strike envelope that fades the scarp,
        # plus a per-fault burial factor (blind faults have almost none)
        env_cycles = float(rng.uniform(1.0, 3.0))
        env_phase = float(rng.uniform(0, 2 * np.pi))
        burial = float(rng.uniform(0.10, 0.35)) if blind else float(rng.uniform(0.5, 1.0))
        faults.append(dict(cx=cx, cy=cy, az=az, length=length, step=step,
                           active=active, blind=blind, mapped=mapped,
                           burial=burial, env_cycles=env_cycles,
                           env_phase=env_phase))

    for f in faults:
        def width(signed, along, half, f=f):
            # a normal-fault scarp: an asymmetric step, smoothed over ~2 px,
            # fading along strike where colluvium has buried it
            w = 1.6
            step_prof = -0.5 * np.tanh(signed / w)
            u = np.clip(np.abs(along) / max(half, 1), 0, 1)
            taper = np.cos(u * np.pi / 2)
            env = np.clip(0.5 + 0.5 * np.sin(f["env_phase"] + u * f["env_cycles"]
                                             * 2 * np.pi), 0.0, 1.0)
            return f["step"] * f["burial"] * env * step_prof * taper  # noqa: E501

        if not f["blind"]:
            contrib, *_ = _segment_field(sh, f["cx"], f["cy"], f["az"], f["length"], width)
            elev += contrib
        # gravity: density step at every fault, blind or not (basement offset)
        def gwidth(signed, along, half, f=f):
            taper = np.cos(np.clip(np.abs(along) / max(half, 1), 0, 1) * np.pi / 2)
            return 6.0 * (-0.5 * np.tanh(signed / 3.0)) * taper
        gc, *_ = _segment_field(sh, f["cx"], f["cy"], f["az"], f["length"], gwidth)
        grav += gc
        # conductivity: blind faults are the conductive-clay-cap ones
        if f["blind"]:
            def cwidth(signed, along, half, f=f):
                taper = np.cos(np.clip(np.abs(along) / max(half, 1), 0, 1) * np.pi / 2)
                return 40.0 * np.exp(-(signed / 3.0) ** 2) * taper
            cc_, *_ = _segment_field(sh, f["cx"], f["cy"], f["az"], f["length"], cwidth)
            cond += cc_
        # seismicity + shear
        if f["active"]:
            def ewidth(signed, along, half, f=f):
                taper = np.cos(np.clip(np.abs(along) / max(half, 1), 0, 1) * np.pi / 2)
                return np.exp(-(signed / 2.2) ** 2) * taper
            ec, *_ = _segment_field(sh, f["cx"], f["cy"], f["az"], f["length"], ewidth)
            eq += ec * rng.uniform(0.5, 1.5)
        def swidth(signed, along, half, f=f):
            taper = np.cos(np.clip(np.abs(along) / max(half, 1), 0, 1) * np.pi / 2)
            return np.exp(-(signed / 2.6) ** 2) * taper
        sc, *_ = _segment_field(sh, f["cx"], f["cy"], f["az"], f["length"], swidth)
        shear += sc

        # rasterise the trace
        mask_az_out = [np.zeros(sh, dtype=bool)]
        trace(f["az"], f["cx"], f["cy"], f["length"])
        tr = mask_az_out[0]
        if f["mapped"]:
            catalogue |= tr
            notes["mapped"] += 1
        else:
            gt_new |= tr
            notes["unmapped"] += 1

    # --- catalogue-gap geometry, per the staff definition (thread 11536) ---
    for f in [f for f in faults if f["mapped"]][: max(2, n_faults // 5)]:
        # continuation past a mapped tip: same azimuth, beyond the mapped end
        t = np.deg2rad(f["az"])
        ext = f["length"] * rng.uniform(0.18, 0.4)
        sx = f["cx"] + np.cos(t) * f["length"] / 2
        sy = f["cy"] + np.sin(t) * f["length"] / 2
        mask_az_out = [np.zeros(sh, dtype=bool)]
        trace(f["az"], sx + np.cos(t) * ext / 2, sy + np.sin(t) * ext / 2, ext)
        gt_new |= mask_az_out[0]
        notes["tip_continuations"] += 1
        def width(signed, along, half):
            taper = np.cos(np.clip(np.abs(along) / max(half, 1), 0, 1) * np.pi / 2)
            return f["step"] * 0.8 * (-0.5 * np.tanh(signed / 1.6)) * taper
        c2, *_ = _segment_field(sh, sx + np.cos(t) * ext / 2,
                                sy + np.sin(t) * ext / 2, f["az"], ext, width)
        elev += c2

    for f in [f for f in faults if f["mapped"]][: max(2, n_faults // 6)]:
        # parallel strand: offset 4-9 px, i.e. outside the 300 m kernel of the
        # mapped trace only at its far end -- a genuinely new fault pixel
        t = np.deg2rad(f["az"])
        off = rng.uniform(4, 9)
        cx2 = f["cx"] - np.sin(t) * off
        cy2 = f["cy"] + np.cos(t) * off
        mask_az_out = [np.zeros(sh, dtype=bool)]
        trace(f["az"], cx2, cy2, f["length"] * 0.8)
        gt_new |= mask_az_out[0]
        notes["parallel_strands"] += 1

    # ---------------- DECOYS: same signature, not a fault ---------------- #
    # (a) symmetric ridgelines.  A ridge has a strong elevation gradient and a
    #     symmetric cross-profile; the step/asymmetry filter must ignore it.
    for _ in range(6):
        cx, cy = rng.uniform(0.1, 0.9) * n, rng.uniform(0.1, 0.9) * n
        az, ln = rng.uniform(0, 180), rng.uniform(80, 200)
        def rwidth(signed, along, half):
            taper = np.cos(np.clip(np.abs(along) / max(half, 1), 0, 1) * np.pi / 2)
            return 1.1 * np.exp(-(signed / 2.5) ** 2) * taper   # symmetric!
        rc, *_ = _segment_field(sh, cx, cy, az, ln, rwidth)
        elev += rc
        notes["decoy_ridges"] += 1
    # (b) a regional monocline: a long smooth tilt, symmetric gradient.
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float64)
    elev += 0.9 * np.tanh((xx * 0.6 + yy * 0.8 - n * 0.5) / (n * 0.18))
    # (c) a diffuse seismic swarm: high earthquake density, NOT linear.
    swx, swy = rng.uniform(0.2, 0.8) * n, rng.uniform(0.2, 0.8) * n
    eq += 2.2 * np.exp(-((xx - swx) ** 2 + (yy - swy) ** 2) / (2 * 14.0 ** 2))
    notes["decoy_swarm"] += 1
    # (d) lithologic contacts: a density/conductivity step with no fault.
    for _ in range(5):
        cx, cy = rng.uniform(0.1, 0.9) * n, rng.uniform(0.1, 0.9) * n
        az, ln = rng.uniform(0, 180), rng.uniform(90, 240)
        def lwidth(signed, along, half):
            taper = np.cos(np.clip(np.abs(along) / max(half, 1), 0, 1) * np.pi / 2)
            return 7.0 * (-0.5 * np.tanh(signed / 3.0)) * taper
        lc, *_ = _segment_field(sh, cx, cy, az, ln, lwidth)
        grav += lc
        # a conductive layer edge with no fault behind it: this is exactly the
        # false positive G-3's own docstring admits it cannot reject.
        def kwidth(signed, along, half):
            taper = np.cos(np.clip(np.abs(along) / max(half, 1), 0, 1) * np.pi / 2)
            return 45.0 * np.exp(-(signed / 3.0) ** 2) * taper
        kc, *_ = _segment_field(sh, cx, cy, az, ln, kwidth)
        cond += kc
        notes["decoy_contacts"] += 1

    # ---------------- noise, smoothing, non-negativity ---------------- #
    def fbm(scale_list, amp):
        out = np.zeros(sh)
        for s, a in zip(scale_list, amp):
            g = rng.normal(0, 1, (max(2, n // s), max(2, n // s)))
            from scipy.ndimage import zoom
            out += a * zoom(g, (n / g.shape[0], n / g.shape[1]), order=1)[:n, :n]
        return out

    elev += fbm([8, 16, 32, 64], [0.05, 0.09, 0.16, 0.28])
    elev = gaussian_filter(elev, 0.8)
    eq = np.clip(gaussian_filter(eq, 1.4) + np.clip(fbm([4, 16], [0.05, 0.12]), 0, None), 0, None)
    shear = np.clip(gaussian_filter(shear, 1.6) + np.clip(fbm([6, 24], [0.06, 0.14]), 0, None), 0, None)
    grav = gaussian_filter(grav, 1.2) + fbm([16, 48], [0.5, 1.2])
    cond = np.clip(gaussian_filter(cond, 1.1) + 60.0 + fbm([12, 40], [6.0, 12.0]), 1.0, None)

    band_names = ["surface_conductivity", "depth_to_conductive_base",
                  "detrended_elevation", "slope_of_detrended_elevation",
                  "dilatation_rate", "shear_strain_rate",
                  "second_invariant_strain_rate", "isostatic_gravity_anomaly",
                  "isostatic_gravity_slope", "reduced_to_pole_magnetic_anomaly",
                  "total_magnetic_intensity", "vertical_slope_tmi",
                  "horizontal_slope_tmi", "top_of_crustal_source_depth",
                  "earthquake_density"]
    # depth_to_conductive_base and the second invariant get plausible fillers so
    # that G-2/G-3 can run; they are not what H-1..H-3 test.
    depth_base = 800.0 + 400.0 * gaussian_filter(rng.normal(0, 1, sh), 12)
    dil = gaussian_filter(rng.normal(0, 1, sh), 8)
    i2 = np.abs(gaussian_filter(rng.normal(0, 1, sh), 8)) + 0.4
    stack = np.stack([cond, depth_base, elev, np.hypot(*np.gradient(elev)),
                      dil, shear, i2, grav, np.hypot(*np.gradient(grav)),
                      fbm([16, 48], [30.0, 60.0]), fbm([16, 48], [200.0, 400.0]),
                      fbm([16], [20.0]), fbm([16], [20.0]),
                      1500.0 + 800.0 * gaussian_filter(rng.normal(0, 1, sh), 14),
                      eq]).astype(np.float32)

    data = CompetitionData(stack=stack, catalogue=catalogue,
                           band_names=band_names, footprint=None, tags={})
    notes["catalogue_px"] = int(catalogue.sum())
    notes["new_px"] = int(gt_new.sum())
    return data, gt_new, catalogue, notes


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
def roc_auc(score, pos):
    """AUC by the rank identity. No extra dependency."""
    s = np.asarray(score, dtype=np.float64).ravel()
    p = np.asarray(pos, dtype=bool).ravel()
    n_pos, n_neg = int(p.sum()), int((~p).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ties
    s_sorted = s[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return float((ranks[p].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def blocked_aucs(score, gt_new, catalogue, blocks=4):
    """Mean AUC over held-out blocks, on pixels OUTSIDE the catalogue.

    Scoring only outside the catalogue is the point: those are the pixels the
    competition actually scores (thread 11516: catalogue pixels are masked).
    """
    Hh, Ww = gt_new.shape
    bh, bw = Hh // blocks, Ww // blocks
    aucs = []
    for bi in range(blocks):
        for bj in range(blocks):
            sl = (slice(bi * bh, (bi + 1) * bh), slice(bj * bw, (bj + 1) * bw))
            cat_b = catalogue[sl]
            gt_b = gt_new[sl] & ~cat_b
            if gt_b.sum() < 8:
                continue
            aucs.append(roc_auc(score[sl][~cat_b], gt_b[~cat_b]))
    aucs = [a for a in aucs if np.isfinite(a)]
    if not aucs:
        return float("nan"), 0
    return float(np.mean(aucs)), len(aucs)


# --------------------------------------------------------------------------- #
def run(n=768, seed=11, folds=4, do_dti=False, quiet=False):
    def _p(*a):
        if not quiet:
            print(*a)

    data, gt_new, catalogue, notes = make_region(n=n, seed=seed)
    _p(f"PHANTOM {n}x{n}  seed={seed}")
    _p(f"  catalogue {notes['catalogue_px']:,} px | new faults {notes['new_px']:,} px")
    _p(f"  composition: {notes['mapped']} mapped faults, {notes['unmapped']} unmapped, "
       f"{notes['tip_continuations']} tip continuations, "
       f"{notes['parallel_strands']} parallel strands")
    _p(f"  decoys: {notes['decoy_ridges']} symmetric ridgelines, "
       f"{notes['decoy_swarm']} diffuse seismic swarm, "
       f"{notes['decoy_contacts']} lithologic contacts (density step, no fault), "
       f"1 regional monocline")
    _p("")

    # distance-to-catalogue is the baseline every detector must beat
    dcat = distance_transform_edt(~catalogue)
    base_score = np.exp(-dcat / 2.0)
    base_auc, nb = blocked_aucs(base_score, gt_new, catalogue, blocks=folds)
    _p(f"BASELINE  distance-to-catalogue  AUC = {base_auc:.4f}  "
       f"({nb} scored blocks)")
    _p("          every detector below must beat this or it adds nothing the")
    _p("          corridor does not already encode.\n")

    rows = []
    cands = [("BASELINE dist-to-catalogue", base_score, None)]
    for det in F.all_detectors():
        try:
            cands.append((f"{det.key} {det.title}", det(data), det.key))
        except Exception as e:                                    # noqa: BLE001
            _p(f"  [skip] {det.key}: {type(e).__name__}: {e}")
    for hyp in H.shipped_hypotheses():
        try:
            cands.append((f"{hyp.key} {hyp.title}", hyp(data), hyp.key))
        except Exception as e:                                    # noqa: BLE001
            _p(f"  [skip] {hyp.key}: {type(e).__name__}: {e}")
    for hyp in H.external_hypotheses():
        _p(f"  [blocked] {hyp.key} {hyp.title}: needs external data "
           f"({hyp.external})")

    _p(f"\n{'candidate':44} {'AUC':>7} {'dAUC':>8} {'blocks':>7}")
    _p("-" * 70)
    for name, score, key in cands:
        auc, nb = blocked_aucs(score, gt_new, catalogue, blocks=folds)
        d = auc - base_auc
        rows.append({"name": name, "key": key, "auc": auc, "delta_auc": d,
                     "blocks": nb})
        flag = "  <-- adds nothing" if d <= 0 else ""
        _p(f"{name:44} {auc:>7.4f} {d:>+8.4f} {nb:>7}{flag}")
    _p("-" * 70)

    best = max((r for r in rows if r["key"]), key=lambda r: r["auc"])
    _p(f"\nBEST NON-BASELINE CANDIDATE: {best['name']}  AUC {best['auc']:.4f} "
       f"(+{best['delta_auc']:.4f} over proximity)")

    dti_rows = []
    if do_dti:
        from gems import strategy as S
        _p("\nDTI ARM — free catalogue core + detector-gated 3 px halo,")
        _p("scored with the real DTI and the real catalogue mask.\n")
        _p(f"{'candidate':44} {'DTI':>7} {'vs core':>8}")
        _p("-" * 62)
        core = catalogue.astype(np.float32)
        m = catalogue  # masked pixels contribute nothing to FP
        core_dti, _ = distance_weighted_tversky_fast(core, gt_new,
                                                     fp_ignore_mask=m)
        _p(f"{'catalogue core only (free)':44} {core_dti:>7.4f} {'--':>8}")
        for name, score, key in cands:
            if key is None:
                continue
            paid = np.zeros_like(score)
            band = (dcat > 0) & (dcat <= 3.0) & (~catalogue)
            budget = max(1, int(0.02 * band.sum()))
            flat = np.where(band, score, -np.inf).ravel()
            idx = np.argpartition(-flat, budget - 1)[:budget]
            paid.ravel()[idx] = 1.0
            sub = np.clip(core + paid, 0, 1)
            dti, _ = distance_weighted_tversky_fast(sub, gt_new, fp_ignore_mask=m)
            dti_rows.append({"name": name, "key": key, "dti": dti,
                             "delta": dti - core_dti})
            _p(f"{name:44} {dti:>7.4f} {dti - core_dti:>+8.4f}")
        _p("-" * 62)

    return {"n": n, "seed": seed, "folds": folds, "notes": notes,
            "baseline_auc": base_auc, "rows": rows, "dti_rows": dti_rows,
            "best": best["name"]}


def candidate_grid(data, catalogue):
    """Every (detector, configuration) arm that will be compared.

    Configurations are enumerated here so that selection happens over an
    explicit, declared grid rather than by editing code until a number improves.
    `select_and_evaluate` picks on one phantom and scores on others.
    """
    arms = [("BASELINE dist-to-catalogue",
             np.exp(-distance_transform_edt(~catalogue) / 2.0), "baseline")]

    for det in F.all_detectors():
        try:
            arms.append((f"{det.key} {det.title}", det(data), det.key))
        except Exception as e:                                    # noqa: BLE001
            arms.append((f"{det.key} [FAILED {type(e).__name__}]", None, det.key))

    # The strength of the catalogue suppression is a quantity we cannot know
    # without real data, so it is a declared axis rather than an assumption.
    # `inf` means "no suppression at all" and is passed straight through to the
    # detector, which owns the suppression -- no undo/redo round trip, because
    # dividing by a suppression factor that tends to zero is numerically wrong
    # exactly where the suppression matters.
    TAU = (2.0, 8.0, float("inf"))
    for hyp in H.shipped_hypotheses():
        for tau in TAU:
            nm = ("no catalogue suppression" if np.isinf(tau)
                  else f"suppression tau={tau:g}")
            try:
                s = hyp(data, suppress_tau=tau)
                arms.append((f"{hyp.key} {hyp.title} [{nm}]", s, hyp.key))
            except Exception as e:                                # noqa: BLE001
                arms.append((f"{hyp.key} [FAILED {type(e).__name__}]",
                             None, hyp.key))

    return arms


def select_and_evaluate(n=512, folds=4, select_seed=11,
                        eval_seeds=(23, 37, 41, 53, 67), quiet=False):
    """Pick each detector's configuration on ONE phantom, score on others.

    Choosing the best configuration and reporting its score on the same phantom
    is selection bias -- it is the exact failure the README's withdrawn
    "+0.1379 corridor lift" was caused by.  So the grid is declared, one
    phantom is used to choose, and five unseen phantoms are used to report.
    """
    def _p(*a):
        if not quiet:
            print(*a)

    d_sel, g_sel, c_sel, _ = make_region(n=n, seed=select_seed)
    grid = candidate_grid(d_sel, c_sel)

    chosen, base_sel = {}, None
    for name, score, key in grid:
        if score is None:
            continue
        auc, _ = blocked_aucs(score, g_sel, c_sel, blocks=folds)
        if key == "baseline":
            base_sel = auc
        cur = chosen.get(key)
        if cur is None or auc > cur[1]:
            chosen[key] = (name, auc, score)

    _p(f"SELECTION PHANTOM seed={select_seed} ({n}x{n}) — configuration chosen here")
    _p(f"  baseline distance-to-catalogue AUC = {base_sel:.4f}")
    for key, (name, auc, _) in sorted(chosen.items()):
        if key == "baseline":
            continue
        _p(f"  chosen {key:4} {name:52} AUC {auc:.4f}")

    _p(f"\nEVALUATION — {len(eval_seeds)} unseen phantoms, spatially blocked")
    _p(f"{'candidate':52} {'dAUC mean':>10} {'per-seed dAUC'}")
    _p("-" * 92)
    results = []
    for key, (name, _sel_auc, _) in sorted(chosen.items()):
        if key == "baseline":
            continue
        deltas, aucs = [], []
        for s in eval_seeds:
            d_ev, g_ev, c_ev, _ = make_region(n=n, seed=s)
            # rebuild ONLY the chosen configuration on this phantom
            sc = _rebuild(name, d_ev)
            if sc is None:
                continue
            b, _nb = blocked_aucs(np.exp(-distance_transform_edt(~c_ev) / 2.0),
                                  g_ev, c_ev, blocks=folds)
            a, _nb2 = blocked_aucs(sc, g_ev, c_ev, blocks=folds)
            aucs.append(a)
            deltas.append(a - b)
        if not deltas:
            continue
        mean = float(np.mean(deltas))
        results.append({"key": key, "name": name, "mean_delta_auc": mean,
                        "per_seed": [round(x, 4) for x in deltas],
                        "mean_auc": float(np.mean(aucs)),
                        "n_wins": int(sum(1 for x in deltas if x > 0)),
                        "n_seeds": len(deltas)})
        _p(f"{name[:52]:52} {mean:>+10.4f} "
           f"{' '.join(f'{x:+.3f}' for x in deltas)}  "
           f"({sum(1 for x in deltas if x > 0)}/{len(deltas)} beat baseline)")
    _p("-" * 92)
    if results:
        best = max(results, key=lambda r: r["mean_delta_auc"])
        _p(f"\nTOP CANDIDATE ON UNSEEN PHANTOMS: {best['name']}")
        _p(f"  mean dAUC over distance-to-catalogue: {best['mean_delta_auc']:+.4f}")
        _p(f"  beat the baseline on {best['n_wins']}/{best['n_seeds']} phantoms")
    return {"select_seed": select_seed, "eval_seeds": list(eval_seeds),
            "baseline_select_auc": base_sel, "results": results,
            "chosen": {k: v[0] for k, v in chosen.items()}}


def _rebuild(name, data):
    """Rebuild a chosen arm on a fresh phantom from its declared name.

    The configuration is recovered from the arm's own label, so the arm that is
    scored on the evaluation phantoms is exactly the arm that was selected --
    no second copy of the logic that could drift from the first.
    """
    m = re.search(r"tau=([0-9.e+-]+)\]", name)
    if m:
        tau = float(m.group(1))
    elif "no catalogue suppression" in name:
        tau = float("inf")
    else:
        tau = 1.5
    key = name.split()[0]
    if key.startswith("H-"):
        for hyp in H.shipped_hypotheses():
            if hyp.key == key:
                return hyp(data, suppress_tau=tau)
        return None
    for det in F.all_detectors():
        if name.startswith(det.key):
            return det(data)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=768)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--folds", type=int, default=4,
                    help="blocks per axis for the spatial holdout")
    ap.add_argument("--dti", action="store_true",
                    help="also score the corridor submission with the real DTI")
    ap.add_argument("--select", action="store_true",
                    help="declared-grid selection on one phantom, evaluation on "
                         "five unseen ones (the protocol that must be used "
                         "before a submission slot is spent)")
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    if args.select:
        rec = select_and_evaluate(n=args.n, folds=args.folds)
    else:
        rec = run(n=args.n, seed=args.seed, folds=args.folds, do_dti=args.dti)

    print("\n" + "=" * 70)
    print("WHAT THIS DOES AND DOES NOT SHOW")
    print("  * The phantom's physics are OUR forward model, not the Earth's.")
    print("  * Decoys are included, so a win means the operator separates a")
    print("    fault from things that look like one -- not that it inverts its")
    print("    own generator.")
    print("  * A win here is necessary, never sufficient. Nothing in this")
    print("    output is evidence about the GeoDAWN region.")
    print("  * No submission slot may be spent on the strength of this file.")
    print("=" * 70)

    if args.out:
        Path(args.out).write_text(json.dumps(rec, indent=2, default=float))
        print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
