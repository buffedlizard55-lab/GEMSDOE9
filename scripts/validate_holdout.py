#!/usr/bin/env python3
"""
validate_holdout.py — the gate that decides whether a submission slot is spent.

    python scripts/validate_holdout.py              # blocked holdout on real data
    python scripts/validate_holdout.py --strategy   # the metric-shape experiment

MODE 1 (default) — SPATIALLY-BLOCKED HOLDOUT ON REAL DATA
    Needs data/training_features.tif + data/existing_faults.tif.  Runs
    src.gems.pipeline.blocked_holdout: buffered 4x4 blocks, whole-block
    holdout, exact DTI kernel, the official catalogue mask, and four arms
    (model, budget-matched random, catalogue-only, corridor).  Then
    pipeline.holdout_gate decides pass/fail.  A failing gate writes nothing.

    Honest limitation, stated in the output and on the site: this scores
    CATALOGUE faults, which the competition does not score. It proves the
    detector carries information beyond "where the catalogue already is". It
    cannot prove the detector finds faults the catalogue missed -- that needs
    withheld labels nobody has. This is why the strategy below, not the
    holdout, carries most of the expected lift.

MODE 2 (--strategy) — METRIC-SHAPE EXPERIMENT
    Runs with no data at all.  Builds a phantom whose ground truth follows the
    staff definition of "new fault" (any fault pixel not already in
    USGS/INGENIOUS, including newly mapped geometry of an existing system),
    then measures, with the real DTI and the real mask:

        catalogue-copy      -> reproduces the 0.1563 fingerprint
        corridor            -> what the free-core + paid-halo shape buys
        soft vs binary      -> whether probabilities or a 0/1 mask wins
        halo/budget sweep   -> the calibration the model path performs

    WHAT THIS DOES AND DOES NOT SHOW.  It validates the ARITHMETIC of the
    submission shape against the published metric and the published masking
    rule. It says nothing about Nevada geology. A phantom cannot. It is here
    so the design decisions in strategy.py are not taken on faith.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.gems.metric import distance_weighted_tversky_fast   # noqa: E402
from src.gems import strategy as S                            # noqa: E402


# --------------------------------------------------------------------------- #
# phantom
# --------------------------------------------------------------------------- #
# Shared reading notes. One definition, printed by both the single-phantom and
# the multi-phantom path, so the two can never drift apart.
READ_BEFORE_QUOTING = """  READ BEFORE QUOTING ANY NUMBER FROM THIS TABLE:
   * The free catalogue core has phi = 0 EXACTLY. It is not a baseline to beat,
     it is the floor every paid pixel has to earn past.
   * `hard` vs `soft` is the only pair that shares a pixel set exactly, so it is
     the only comparison here that isolates the SHAPE of the submission. Full
     mass wins only while the field is recall-limited; once phi dominates,
     spreading the same mass over the same pixels is worth more. Which regime
     you are in is a property of the real data, not of this table.
   * The halo's SIGN IS NOT STABLE. At 768x768 a blind 3 px halo loses to the
     free core; at 512x512 it wins. The halo is 3 px either way -- 300 m, the
     metric radius -- but the window changes the catalogue-to-target ratio, so it
     changes what the ring covers. That instability is the whole argument for
     calibrating on the REAL blocked holdout instead of arguing a halo from a
     phantom.
   * Part A and Part B use DIFFERENT phantom sizes on purpose: Part A is 20
     metric evaluations per cell and can afford 768; Part B is a 20-point grid
     x 3 training regions x 4 folds and cannot. Do not compare a Part A number
     to a Part B number.
   * Synthetic, on fault geometry I invented. This prices the shape of a
     submission against the published metric. It is NOT evidence that any
     detector works on Nevada data."""


def _segments(shape, segs):
    out = np.zeros(shape, dtype=bool)
    H, W = shape
    for (y0, x0, y1, x1) in segs:
        n = int(max(abs(y1 - y0), abs(x1 - x0))) + 1
        for t in range(n):
            u = t / max(n - 1, 1)
            y = int(round(y0 + u * (y1 - y0)))
            x = int(round(x0 + u * (x1 - x0)))
            if 0 <= y < H and 0 <= x < W:
                out[y, x] = True
    return out


def make_phantom(n=768, seed=7):
    """
    Catalogue strands + 'new' faults generated the way the competition says the
    scored set is built: tips extended, splays branched off, parallel strands
    added beside existing ones, plus a minority of genuinely isolated faults.
    """
    rng = np.random.default_rng(seed)
    cat, new = [], []

    for _ in range(22):                      # long NW-SE strands (Walker-Lane-like)
        y0 = int(rng.integers(40, n - 40)); x0 = int(rng.integers(20, n - 120))
        L = int(rng.integers(150, 400))
        a = np.deg2rad(rng.normal(-55, 8))
        cat.append((y0, x0, int(y0 + L * np.sin(a)), int(x0 + L * np.cos(a))))
    for _ in range(7):                       # NE-striking cross faults
        y0 = int(rng.integers(60, n - 60)); x0 = int(rng.integers(30, n - 200))
        L = int(rng.integers(100, 260))
        a = np.deg2rad(rng.normal(35, 10))
        cat.append((y0, x0, int(y0 + L * np.sin(a)), int(x0 + L * np.cos(a))))

    for (y0, x0, y1, x1) in cat:
        if rng.random() < 0.85:              # class A: continuation past a tip
            L = np.hypot(y1 - y0, x1 - x0) or 1.0
            e = int(rng.integers(15, 70))
            new.append((y1, x1, int(y1 + (y1 - y0) / L * e), int(x1 + (x1 - x0) / L * e)))
        if rng.random() < 0.55:              # class B: splay at a random point
            t = rng.uniform(0.2, 0.8)
            ya, xa = y0 + t * (y1 - y0), x0 + t * (x1 - x0)
            a = np.arctan2(y1 - y0, x1 - x0) + np.deg2rad(rng.normal(0, 22))
            L = int(rng.integers(30, 110))
            new.append((int(ya), int(xa), int(ya + L * np.sin(a)), int(xa + L * np.cos(a))))
        if rng.random() < 0.45:              # class C: parallel strand 2-4 px away
            a = np.arctan2(y1 - y0, x1 - x0) + np.deg2rad(rng.normal(0, 8))
            off = int(rng.integers(2, 5))
            nx, ny = -np.sin(a) * off, np.cos(a) * off
            new.append((int(y0 + ny), int(x0 + nx), int(y1 + ny), int(x1 + nx)))
    for _ in range(6):                       # class D: genuinely isolated
        y0 = int(rng.integers(30, n - 30)); x0 = int(rng.integers(30, n - 120))
        L = int(rng.integers(30, 90))
        a = np.deg2rad(rng.normal(-55, 25))
        new.append((y0, x0, int(y0 + L * np.sin(a)), int(x0 + L * np.cos(a))))

    catalogue = _segments((n, n), cat)
    gt = _segments((n, n), new) & ~catalogue
    return catalogue, gt


def _dilate(mask, iterations):
    """3x3 (8-connected) binary dilation. Matches strategy._dilate exactly."""
    out = np.asarray(mask).astype(bool).copy()
    for _ in range(int(iterations)):
        acc = out.copy()
        acc[1:, :] |= out[:-1, :]
        acc[:-1, :] |= out[1:, :]
        acc[:, 1:] |= out[:, :-1]
        acc[:, :-1] |= out[:, 1:]
        out = acc
    return out


# --------------------------------------------------------------------------- #
# a detector with a KNOWN, SWEEPPABLE quality
# --------------------------------------------------------------------------- #
def make_detector(catalogue, gt, rng, mu):
    """
    score = mu * (smoothed truth + 0.3 * smoothed catalogue) + N(0,1)

    `mu` is the only free parameter and it directly controls the ROC AUC, which
    we then MEASURE rather than assume.  This matters: the honest version of
    "does a detector help?" has to sweep detector quality, because the answer
    depends on it. A single arbitrary detector proves nothing either way.
    """
    from scipy.ndimage import gaussian_filter
    s = gaussian_filter(gt.astype(np.float32), 2.0)
    s = s / (float(s.max()) + 1e-9)
    c = gaussian_filter(catalogue.astype(np.float32), 3.0)
    c = c / (float(c.max()) + 1e-9)
    return (mu * (s + 0.3 * c) + rng.normal(0, 1, s.shape)).astype(np.float32)


def roc_auc(score, gt):
    """AUC of `score` against `gt`, by the rank identity. No extra dependency."""
    flat = np.asarray(score, dtype=np.float64).reshape(-1)
    pos = np.asarray(gt, dtype=bool).reshape(-1)
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(flat, kind="stable")
    ranks = np.empty(flat.size, dtype=np.float64)
    sr = flat[order]
    i = 0
    r = 1
    while i < sr.size:                       # average ranks within ties
        j = i
        while j + 1 < sr.size and sr[j + 1] == sr[i]:
            j += 1
        ranks[order[i:j + 1]] = (r + r + (j - i)) / 2.0
        r += (j - i + 1)
        i = j + 1
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


# --------------------------------------------------------------------------- #
def _arms_on_grid(score, catalogue, gt, budget_frac, n_all, rng):
    """
    Selection-FREE arms. Nothing here is chosen using gt, so all of it can be
    evaluated on the whole phantom at once -- no blocks, no leakage, ~100x
    cheaper than a 4-fold protocol, and far less noisy.

    Returns a dict of arm name -> (DTI, components).
    """
    m = distance_weighted_tversky_fast
    cat_field = np.where(catalogue, 1.0, 0.0).astype(np.float32)
    band = _dilate(catalogue, 3) & ~catalogue
    # rank once; deterministic top-k over the whole grid
    sflat = np.clip(score, 0, 1).astype(np.float64).reshape(-1)
    k = int(round(budget_frac * n_all))
    top = np.argsort(-sflat, kind="stable")[:k]
    # hard: p = 1 on the same pixels
    hard = cat_field.copy()
    hard.reshape(-1)[top] = 1.0
    # soft: p = the detector's confidence on exactly the same pixels
    if k:
        z = sflat[top]
        conf = 1.0 / (1.0 + np.exp(-(z - z.mean()) / (z.std() + 1e-9)))
    else:
        conf = np.zeros(0, np.float64)
    soft = np.zeros((n_all and catalogue.shape), np.float32)
    soft.reshape(-1)[top] = conf.astype(np.float32)
    soft = np.maximum(soft, cat_field)
    # budget-matched random control, same pixel count, same region
    rnd = np.zeros(catalogue.shape, np.float32)
    if k:
        rnd.reshape(-1)[rng.choice(n_all, size=k, replace=False)] = 1.0
    out = {}
    for name, f in (("catalogue", cat_field), ("hard", hard), ("soft", soft),
                    ("random", rnd)):
        d, t = m(f, gt, fp_ignore_mask=catalogue)
        out[name] = (float(d), t)
    return out


def run_strategy(n=768, seed=7, folds=2, quiet=False,
                 mus=(0.0, 0.6, 1.2, 2.0),
                 budget_fracs=(0.001, 0.003, 0.01, 0.03, 0.10),
                 n_blocked=512, mus_blocked=(0.0, 2.0)):
    """
    What is the submission SHAPE worth? Two separate questions, kept separate
    because only one of them involves choosing anything.

    PART A -- selection-free, evaluated on the whole phantom.
        catalogue  the known catalogue and nothing else. phi = 0 exactly, so
                   this is an upper bound that nothing can pass without a
                   detector that genuinely adds recall.
        hard       p = 1 on the top-k detector pixels, plus the free core.
        soft       p = a calibrated-looking fractional confidence on exactly
                   the same pixels, plus the free core.
        random     k random pixels, no core. The detector control.

        hard vs soft is the controlled experiment: identical pixel set,
        identical recall, identical detector ranking. Only the VALUE put on
        each pixel differs. Sweeping k over two decades finds the crossover.

    PART B -- selection, blocked 2x2, calibrated on training blocks only.
        corridor  catalogue + a detector-gated halo, (halo, budget) chosen on
                  the TRAINING blocks with a selection-noise guard, scored on
                  the block that was not used to choose it.
        blind     catalogue + a 3 px dilation, detector ignored.

    Part A needs no blocking because nothing is selected; Part B needs it
    because something is. Mixing them in one protocol is what produced the
    earlier, invalid "+0.14 lift".
    """
    _p = (lambda *a, **k: None) if quiet else print
    rng = np.random.default_rng(seed)
    catalogue, gt = make_phantom(n, seed)
    n_all = gt.size
    m = distance_weighted_tversky_fast
    cat_field = np.where(catalogue, 1.0, 0.0).astype(np.float32)
    blind = np.where(_dilate(catalogue, 3), 1.0, 0.0).astype(np.float32)

    _p("=" * 92)
    _p("METRIC-SHAPE EXPERIMENT 1 -- what is the submission shape worth?")
    _p("=" * 92)
    _p(f"phantom {n}x{n} ({n_all:,} px) | catalogue {int(catalogue.sum()):,} px "
          f"({catalogue.mean():.3%}) | withheld new faults {int(gt.sum()):,} px "
          f"({gt.mean():.3%})")
    _p("Ground truth built to the staff definition of 'new fault' (thread 11536):")
    _p("tip continuations, splays, parallel strands, plus isolated faults.\n")
    _p("PART A -- selection-free, scored on the whole phantom. Nothing is tuned,")
    _p("so nothing can leak. `hard` and `soft` share one pixel set exactly, and")
    _p("they rank the WHOLE GRID: they are the UNGATED control, and Part B is")
    _p("what proximity-gating buys on top of them.\n")
    _p(f"{'mu':>5} {'AUC':>6} {'budget':>8} | {'catalogue':>9} {'soft':>7} "
          f"{'hard':>7} {'hard-soft':>10} | {'random':>7} {'hard-random':>12}")
    _p("-" * 92)

    n_A, catA, gtA = n, catalogue, gt
    rows: list[list[dict]] = []
    for mu in mus:
        score = make_detector(catalogue, gt, rng, mu)
        auc = roc_auc(score, gt)
        rows.append([])
        for bf in budget_fracs:
            a = _arms_on_grid(score, catalogue, gt, bf, n_all, rng)
            d_hs = a["hard"][0] - a["soft"][0]
            d_hr = a["hard"][0] - a["random"][0]
            rows[-1].append({
                "mu": mu, "auc": float(auc), "budget_frac": bf,
                "n_paid": int(round(bf * n_all)),
                "catalogue": a["catalogue"][0], "soft": a["soft"][0],
                "hard": a["hard"][0], "random": a["random"][0],
                "hard_minus_soft": float(d_hs),
                "hard_minus_random": float(d_hr),
                "catalogue_TP_w": a["catalogue"][1]["TP_w"],
                "hard_TP_w": a["hard"][1]["TP_w"],
                "hard_FP_w": a["hard"][1]["FP_w"],
                "soft_FP_w": a["soft"][1]["FP_w"],
            })
            _p(f"{mu:>5.2f} {auc:>6.3f} {bf:>8.3f} | {a['catalogue'][0]:>9.4f} "
                  f"{a['soft'][0]:>7.4f} {a['hard'][0]:>7.4f} {d_hs:>+10.4f} | "
                  f"{a['random'][0]:>7.4f} {d_hr:>+12.4f}")
    _p("-" * 92)
    flat = [r for grp in rows for r in grp]
    cat_med = float(np.median([r["catalogue"] for r in flat]))
    beat = [r for r in flat if r["hard"] > r["catalogue"]]
    _p(f"  catalogue core alone               {cat_med:.4f}  (phi = 0 exactly)")
    _p(f"  hard beats soft on identical px    "
          f"{sum(1 for r in flat if r['hard_minus_soft'] > 0)}/{len(flat)} cells, "
          f"median {np.median([r['hard_minus_soft'] for r in flat]):+.4f}")
    _p(f"  hard beats a same-size random fill {sum(1 for r in flat if r['hard_minus_random'] > 0)}"
          f"/{len(flat)} cells, median {np.median([r['hard_minus_random'] for r in flat]):+.4f}")
    _p(f"  hard beats the free core           {len(beat)}/{len(flat)} cells")
    if beat:
        w = min(beat, key=lambda r: r["hard_minus_soft"])
        _p(f"     smallest paying budget in the sweep: {w['budget_frac']:.3%} "
          f"of the grid ({w['n_paid']:,} px), hard {w['hard']:.4f} vs core "
          f"{w['catalogue']:.4f}")
    _p()

    # ---- PART B: the only part that selects, so the only part blocked -------
    # Part B is the expensive half (a 20-point grid x 3 training regions x 4
    # folds x every detector quality) and its answer is a sign, not a precise
    # number, so it runs on its own smaller phantom. Stated in the output rather
    # than quietly assumed.
    n, mus_b = n_blocked, list(mus_blocked)
    catalogue, gt = make_phantom(n, seed)
    n_all = gt.size
    cat_field = np.where(catalogue, 1.0, 0.0).astype(np.float32)
    blind = np.where(_dilate(catalogue, 3), 1.0, 0.0).astype(np.float32)
    _p(f"PART B -- selection, blocked. Separate {n}x{n} phantom; (halo, budget) "
          f"is chosen on the TRAINING")
    _p("blocks of each fold with a cross-fitted selection guard, then scored on")
    _p("the held-out block. Out-of-block pixels are masked, not scored.\n")
    _p(f"{'mu':>5} {'AUC':>6} | {'catalogue':>9} {'blind3px':>8} {'corridor':>8} "
          f"{'corr-core':>10} {'spend':>6} | one fold's choice")
    _p("-" * 92)
    brows = []
    bs = n // folds
    block_mask = []
    for fy in range(folds):
        for fx in range(folds):
            bm = np.zeros((n, n), bool)
            bm[fy * bs:(fy + 1) * bs, fx * bs:(fx + 1) * bs] = True
            block_mask.append(bm)
    for mu in mus_b:
        score = make_detector(catalogue, gt, rng, mu)
        auc = roc_auc(score, gt)
        acc = {k: [] for k in ("catalogue", "blind", "corridor")}
        chosen, n_spent = [], 0
        for i in range(len(block_mask)):
            val = block_mask[i]
            train = ~val
            if (gt & val).sum() < 50 or (gt & train).sum() < 50:
                continue
            # every OTHER block is its own training region, so the guard has to
            # clear the bar in all of them separately
            regions = [block_mask[j] for j in range(len(block_mask)) if j != i]
            sel = S.select_corridor_config(
                score, catalogue, gt, regions, halos=(0, 1, 2, 3),
                budgets=(0.0, 0.002, 0.01, 0.04, None), metric=m)
            h_c, b_c = sel["halo_px"], sel["budget_frac"]
            n_spent += int(sel["accepted"])
            chosen.append((h_c, "all-band" if b_c is None else f"{b_c:.2%}",
                           "spent" if sel["accepted"] else "declined"))
            corr = S.build_corridor(
                score, catalogue, halo_px=h_c,
                budget_px=None if b_c is None else int(round(b_c * n_all)))

            def ev(f, _v=val):
                f = np.where(_v, f, 0.0).astype(np.float32)
                return m(f, gt & _v,
                         fp_ignore_mask=(~_v) | (catalogue & _v))[0]

            acc["catalogue"].append(ev(cat_field))
            acc["blind"].append(ev(blind))
            acc["corridor"].append(ev(corr))
        mean = {k: float(np.mean(v)) for k, v in acc.items() if v}
        d = mean["corridor"] - mean["catalogue"]
        brows.append({"mu": mu, "auc": float(auc), **mean,
                      "corridor_minus_catalogue": d,
                      "blind_minus_catalogue": mean["blind"] - mean["catalogue"],
                      "folds_spending": n_spent, "n_folds": len(chosen),
                      "configs_chosen": chosen})
        _p(f"{mu:>5.2f} {auc:>6.3f} | {mean['catalogue']:>9.4f} {mean['blind']:>8.4f} "
              f"{mean['corridor']:>8.4f} {d:>+10.4f} {n_spent}/{len(chosen):>5} | "
              f"{chosen[0] if chosen else '-'}")
    _p("-" * 92)
    _p(f"  blind 3 px halo, no detector: {np.median([r['blind'] for r in brows]):.4f}"
        f" against a {np.median([r['catalogue'] for r in brows]):.4f} core.")
    n_decl = sum(r["n_folds"] - r["folds_spending"] for r in brows)
    n_all_c = sum(r["n_folds"] for r in brows)
    _p(f"  cross-fitted guard declined to spend in {n_decl}/{n_all_c} folds.")
    _p(f"  calibrated corridor vs core: median of per-fold differences "
          f"{np.median([r['corridor_minus_catalogue'] for r in brows]):+.4f}")
    _p()
    _p(READ_BEFORE_QUOTING)

    out = {
        "n": n_A, "n_blocked": n, "seed": seed, "folds": folds,
        "n_catalogue_px": int(catA.sum()), "n_gt_px": int(gtA.sum()),
        "rows": rows, "blocked": brows,
        "caveat": ("Synthetic phantom, detector quality swept rather than assumed. "
                   "Part A is selection-free and scored on the whole phantom; Part B "
                   "is the only part that selects and is therefore the only part "
                   "blocked. Measures the ARITHMETIC of the submission shape against "
                   "the published metric and masking rule. Says nothing about Nevada "
                   "geology and is not evidence that any detector works."),
    }
    return out


def run_strategy_multiseed(seeds, n=768,
                           mus=(0.0, 0.6, 1.2, 2.0),
                           budget_fracs=(0.001, 0.003, 0.01, 0.03, 0.10),
                           n_blocked=512, mus_blocked=(0.0, 2.0)):
    """Aggregate the same experiment over independent phantoms."""
    print("=" * 92)
    print(f"METRIC-SHAPE EXPERIMENT 1 -- {len(seeds)} phantoms x {len(mus)} detector "
          f"qualities, {n}x{n}")
    print("=" * 92)
    recs = [run_strategy(n=n, seed=sd, quiet=True, mus=mus,
                         budget_fracs=budget_fracs, n_blocked=n_blocked,
                         mus_blocked=mus_blocked) for sd in seeds]
    mus, bfs = list(mus), list(budget_fracs)
    mus_b = list(mus_blocked)

    def at(part, i, j, key):
        return np.array([r[part][i][j][key] for r in recs])

    def a1(part, i, key):
        return np.array([r[part][i][key] for r in recs])

    def bm(i, key):
        return np.array([r["blocked"][i][key] for r in recs])

    print(f"PART A -- selection-free. Median over {len(seeds)} phantoms.\n")
    print(f"{'mu':>5} {'AUC':>6} {'budget':>8} | {'catalogue':>9} {'soft':>7} "
          f"{'hard':>7} {'hard-soft':>10} | {'random':>7} {'hard-random':>12}")
    print("-" * 92)
    table = []
    for i, mu in enumerate(mus):
        auc = float(np.median([r["rows"][i][0]["auc"] for r in recs]))
        for j, bf in enumerate(bfs):
            row = {k: float(np.median(at("rows", i, j, k))) for k in
                   ("catalogue", "soft", "hard", "random")}
            d_hs = float(np.median(at("rows", i, j, "hard_minus_soft")))
            d_hr = float(np.median(at("rows", i, j, "hard_minus_random")))
            frac_beats = float(np.mean(at("rows", i, j, "hard") >
                                       at("rows", i, j, "catalogue")))
            table.append({"mu": mu, "auc_median": auc, "budget_frac": bf, **row,
                          "hard_minus_soft": d_hs, "hard_minus_random": d_hr,
                          "frac_phantoms_hard_beats_core": frac_beats})
            print(f"{mu:>5.2f} {auc:>6.3f} {bf:>8.3f} | {row['catalogue']:>9.4f} "
                  f"{row['soft']:>7.4f} {row['hard']:>7.4f} {d_hs:>+10.4f} | "
                  f"{row['random']:>7.4f} {d_hr:>+12.4f}")
    print("-" * 92)
    n_cells = len(table)
    hs = np.array([r["hard_minus_soft"] for r in table])
    hr = np.array([r["hard_minus_random"] for r in table])
    core_wins = sum(r["frac_phantoms_hard_beats_core"] for r in table)
    print(f"  free catalogue core (phi = 0 exactly)   "
          f"{np.median([r['catalogue'] for r in table]):.4f}")
    print(f"  hard beats soft on an identical pixel set: "
          f"{int((hs > 0).sum())}/{n_cells} cells, median {np.median(hs):+.4f}")
    print(f"  hard beats a same-size random fill:       "
          f"{int((hr > 0).sum())}/{n_cells} cells, median {np.median(hr):+.4f}")
    print(f"  hard beats the free core:                 "
          f"{core_wins:.1f}/{n_cells} phantom-cells")
    print()

    print("PART B -- selection, blocked 2x2, cross-fitted guard.\n")
    print(f"{'mu':>5} {'AUC':>6} | {'catalogue':>9} {'blind3px':>8} {'corridor':>8} "
          f"{'corr-core':>10} {'spend':>7}")
    print("-" * 92)
    btable = []
    for i, mu in enumerate(mus_b):
        g = bm
        row = {k: float(np.median(g(i, k))) for k in ("catalogue", "blind", "corridor")}
        d = float(np.median(g(i, "corridor_minus_catalogue")))
        spent = int(np.sum(g(i, "folds_spending")))
        folds = int(np.sum(g(i, "n_folds")))
        btable.append({"mu": mu,
                       "auc_median": float(np.median(a1("blocked", i, "auc"))),
                       **row, "corridor_minus_catalogue": d,
                       "blind_minus_catalogue": float(np.median(g(i, "blind_minus_catalogue"))),
                       "folds_spending": spent, "n_folds": folds})
        print(f"{mu:>5.2f} {btable[-1]['auc_median']:>6.3f} | {row['catalogue']:>9.4f} "
              f"{row['blind']:>8.4f} {row['corridor']:>8.4f} {d:>+10.4f} "
              f"{spent}/{folds:>5}")
    print("-" * 92)
    if len(btable) >= 2:
        lo, hi = btable[0], btable[-1]
        print(f"  detector-attributable lift inside a FIXED corridor: "
              f"{hi['corridor'] - lo['corridor']:+.4f} "
              f"(AUC {lo['auc_median']:.2f} -> {hi['auc_median']:.2f}, same config "
              f"chosen in both). The gap between the corridor and `blind3px` is")
        print(f"  the halo's geometry; this gap is the DETECTOR's, and it is the "
              f"only number in Part B attributable to detection at all.")
    tot_f = sum(r["n_folds"] for r in btable)
    tot_s = sum(r["folds_spending"] for r in btable)
    print(f"  blind 3 px halo, no detector: "
          f"{np.median([r['blind'] for r in btable]):.4f} against a "
          f"{np.median([r['catalogue'] for r in btable]):.4f} core.")
    print(f"  cross-fitted guard declined to spend in {tot_f - tot_s}/{tot_f} folds.")
    print(f"  calibrated corridor vs core: median "
          f"{np.median([r['corridor_minus_catalogue'] for r in btable]):+.4f}")
    print()
    print(READ_BEFORE_QUOTING)
    print()

    return {
        "experiment": "metric-shape-1",
        "protocol": ("Part A is selection-free and scored on the whole phantom, so "
                     "it needs no blocking and cannot leak. Part B is the only part "
                     "that selects, so it is blocked 2x2 and its (halo, budget) is "
                     "chosen on the training blocks with a cross-fitted guard. "
                     "Out-of-block pixels are masked from BOTH the true-positive and "
                     "false-positive terms."),
        "n": n, "n_blocked": n_blocked, "seeds": list(seeds), "mus": mus,
        "budget_fracs": bfs, "mus_blocked": mus_b,
        "n_catalogue_px": int(recs[0]["n_catalogue_px"]),
        "n_gt_px": int(recs[0]["n_gt_px"]),
        "table": table, "blocked_table": btable,
        "summary": {
            "core_dti_median": float(np.median([r["catalogue"] for r in table])),
            "blind_halo_dti_median": float(np.median([r["blind"] for r in btable])),
            "hard_beats_soft_cells": int((hs > 0).sum()), "n_cells": n_cells,
            "hard_minus_soft_median": float(np.median(hs)),
            "hard_beats_random_cells": int((hr > 0).sum()),
            "hard_minus_random_median": float(np.median(hr)),
            "hard_beats_core_phantom_cells": float(core_wins),
            "guard_declined_folds": int(tot_f - tot_s), "guard_folds": int(tot_f),
            "corridor_minus_core_median": float(
                np.median([r["corridor_minus_catalogue"] for r in btable])),
            "detector_lift_in_fixed_corridor": (
                float(btable[-1]["corridor"] - btable[0]["corridor"])
                if len(btable) >= 2 else None),
            "halo_geometry_lift": (
                float(btable[0]["blind"] - btable[0]["catalogue"])
                if btable else None),
        },
        "per_seed": [{k: recs[i][k] for k in ("seed", "rows", "blocked")}
                     for i in range(len(recs))],
        "caveat": recs[0]["caveat"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", action="store_true", help="run the metric-shape experiment")
    ap.add_argument("--seeds", default="7,11,23,101,777",
                    help="comma-separated phantom seeds for the multi-seed summary")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--buffer", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-lift", type=float, default=0.02)
    ap.add_argument("--n", type=int, default=768)
    args = ap.parse_args()
    if args.strategy:
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
        if len(seeds) == 1:
            out = run_strategy(n=args.n, seed=seeds[0])
            out.update({"experiment": "metric-shape-1", "protocol": out["caveat"],
                        "seeds": [seeds[0]], "table": out["rows"],
                        "blocked_table": out["blocked"], "summary": {}})
        else:
            out = run_strategy_multiseed(seeds, n=args.n)
        dest = Path("docs/strategy_experiment.json")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
        print(f"written: {dest}")
        return 0
    return run_real(args)


if __name__ == "__main__":
    sys.exit(main())
