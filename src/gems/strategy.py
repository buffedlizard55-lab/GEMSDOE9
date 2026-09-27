"""
strategy.py — the closed-form algebra of the Distance-weighted Tversky Index (DTI),
and the submission shapes that exploit it.

Everything here is derived from the official metric definition, which is
reproduced verbatim in `metric.py`'s docstring and verified against

  https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/
  § Performance metric → Mathematical representation

    k(d) = max(1 - d/R, 0),  R = 300 m = 3 px at 100 m
    TP_w = sum_{g in G} max_{x : d(x,g) <= R} p(x) k(d(x,g))
    FP_w = sum_{x : p(x) > 0} p(x) [1 - max_{g in G} k(d(x,g))]
    FN_w = sum_{g in G} [1 - max_{x : d(x,g) <= R} p(x) k(d(x,g))]
    DTI  = TP_w / (TP_w + alpha*FP_w + beta*FN_w + eps),  alpha=0.2, beta=0.8

Two exact consequences drive every design decision in this repo.

(1) FN_w = N - TP_w, exactly, where N = |G| is the number of ground-truth
    (new-fault) pixels.  Because TP_w and FN_w are *maxima over a window* of the
    same product p(x)k(d), not sums.  So the metric collapses to

        DTI = T / (0.2*T + 0.2*F + 0.8*N + eps)                     [EQ-1]

    with T = TP_w, F = FP_w, N = |G|.  It depends only on two ratios:

        rho = T / N  (distance-weighted recall)
        phi = F / N  (false-positive mass per ground-truth pixel)

        DTI = rho / (0.2*rho + 0.2*phi + 0.8)                     [EQ-2]

    Everything scale-free.  You never need to know N.

(2) The exchange rate between the two error types follows by differentiating
    EQ-1 with D = 0.2*T + 0.2*F + 0.8*N:

        d(DTI)/dT = (0.2*F + 0.8*N) / D^2      (one extra unit of weighted TP)
        d(DTI)/dF = -(0.2*T)          / D^2    (one extra unit of FP mass)

        value_of_1_TP / cost_of_1_FP = (0.2*F + 0.8*N) / (0.2*T)
                                     = 4*N/T + F/T                                [EQ-3]

    In rho terms, when phi is small this is 4/rho.  So:

        * at rho = 0.13 you can afford ~31 units of false-positive mass per unit
          of weighted recall  (4/0.13 = 30.8)
        * at rho = 0.50 you can afford ~8
        * at rho = 0.80 you can afford ~5

    Recall is worth roughly an order of magnitude more than precision.  This is
    why a hard 0/1 mask on the trace beats a well-calibrated soft probability
    map on this metric, and why "be conservative" is exactly the wrong advice
    here.  It is also why the group's submissions plateaued: they were tuned for
    precision.

THE 0.1563 ATTRACTOR  (derivation, not a measurement — see docs/strategy.html)
------------------------------------------------------------------------------
Set F = 0 (a submission that predicts only pixels the scorer masks out, i.e. the
known USGS/INGENIOUS catalogue) and solve EQ-2 for rho at DTI = 0.1563:

    0.1563 = rho / (0.2*rho + 0.8)
    rho    = 0.8 / (1/0.1563 - 0.2) = 0.1291

So 0.1563 is precisely "weighted recall 12.9% with zero false-positive mass".
That is the fingerprint of a *catalogue copy*: the catalogue is free (masked),
and it picks up whatever fraction of the new faults happen to lie within the
300 m kernel of an already-mapped trace.  Different teams that copy the
catalogue get the same rho, hence the same score to four decimals.  This is a
hypothesis derived from the published metric, not a verified measurement of any
particular file — the submissions themselves are not public.  It is falsifiable:
it predicts that any two catalogue-derived submissions score within ~1e-3 of
each other, and that adding *any* genuine off-catalogue detector breaks the tie.

Inverting EQ-2 for the recall needed to reach the current leaderboard top
(0.3049, team DARD) at zero false positives:

    rho_min = 0.8 / (1/0.3049 - 0.2) = 0.2592

0.3049 is therefore *unreachable* by catalogue copying: it needs at least 25.9%
weighted recall, roughly double the 0.1563 attractor.

THE FREE-CATALOGUE-CORE LEVER
------------------------------
Officially (DrivenData staff, 2026-09-16, forum thread 11516):

    "Pixels corresponding to known USGS/INGENIOUS faults are masked / excluded
     from evaluation, so they do not count towards penalty terms."

Read EQ-1 carefully: masking zeroes a pixel's contribution to **F** only.  The
maxima in TP_w run over **all** pixels, masked or not.  So a prediction placed on
a catalogue pixel costs exactly nothing, and still earns TP_w credit for every
new-fault pixel g within 300 m of it.  The catalogue is a free lottery ticket.

Second official statement (DrivenData staff, 2026-09-22, forum thread 11536):

    "For the purposes of this competition, 'new fault' means 'any fault pixel
     not already captured by USGS/INGENIOUS' and can include newly mapped
     geometry of an existing fault system."

That is decisive: a large share of the scored pixels are *by construction*
inside existing fault zones — strand splays, parallel strands, and continuations
beyond mapped endpoints.  Those are exactly the pixels the free catalogue core
already reaches.

So the submission shape that maximises P(Win) is not "predict faults" and not
"predict new faults".  It is:

    p = 1 on every catalogue pixel                    (free: masked -> F cost 0)
    p = 1 on a thin, evidence-gated halo around them   (paid: F cost, but this is
                                                        where the new geometry is)
    p = 0 everywhere else

and the halo radius is chosen by `optimise_corridor` below on a *training-fold*
holdout, never on the public leaderboard.

CAVEAT, stated because it matters: the mask the scorer applies is the *vector*
catalogue rasterised by the organiser, which is not guaranteed to be pixel-identical
to the `existing_faults.tif` raster we are given.  A one-pixel disagreement makes
a supposedly-free pixel cost 1.0 of FP mass.  `build_corridor` therefore takes
`mask_safety_px` and claims "free" only for pixels at least that many pixels away
from any catalogue pixel that we did not ourselves cover.
"""

from __future__ import annotations

import numpy as np

ALPHA = 0.2
BETA = 0.8
EPS = 1e-7
R_PX = 3          # 300 m at 100 m


# --------------------------------------------------------------------------- #
# EQ-1 / EQ-2 : the metric in closed form
# --------------------------------------------------------------------------- #
def dti_from_terms(tp_w: float, fp_w: float, n_gt: float, eps: float = EPS) -> float:
    """EQ-1. tp_w/fp_w are the distance-weighted counts, n_gt = |G|."""
    denom = tp_w + ALPHA * fp_w + BETA * (n_gt - tp_w) + eps
    if denom <= 0:
        return 0.0
    return float(tp_w / denom)


def dti_from_ratios(rho: float, phi: float, eps: float = EPS) -> float:
    """EQ-2. rho = T/N (weighted recall), phi = F/N.  Scale-free."""
    denom = ALPHA * rho + ALPHA * phi + BETA
    if denom <= 0:
        return 0.0
    return float(rho / denom)


def required_recall_for_score(score: float, phi: float = 0.0) -> float:
    """Invert EQ-2: the weighted recall needed to hit `score` at FP mass `phi`."""
    s = float(score)
    if s <= 0:
        return float("inf")
    if s >= 1.0:
        return 0.0
    return (BETA + ALPHA * phi) / (1.0 / s - ALPHA)


def max_phi_for_score(score: float, rho: float) -> float:
    """Invert EQ-2: the FP mass per GT pixel you can afford at recall `rho`."""
    s = float(score)
    if s <= 0 or rho <= 0:
        return float("inf")
    # s*(0.2 rho + 0.2 phi + 0.8) = rho  ->  phi = (rho/s - 0.2 rho - 0.8)/0.2
    return float((rho / s - ALPHA * rho - BETA) / ALPHA)


# --------------------------------------------------------------------------- #
# EQ-3 : the recall/precision exchange rate
# --------------------------------------------------------------------------- #
def fp_mass_affordable(rho: float, delta_rho: float, phi: float = 0.0) -> float:
    """
    How much FP mass (per ground-truth pixel) may be added in exchange for a
    weighted-recall gain of `delta_rho`, evaluated at the operating point
    (rho, phi).

    Setting the marginal gain from EQ-3 equal to the marginal cost, in rho/phi
    units (DTI = rho/(0.2 rho + 0.2 phi + 0.8), so d/d rho = (0.2 phi + 0.8)/D^2
    and -d/d phi = 0.2 rho/D^2):

        delta_rho * (0.2*phi + 0.8) = 0.2 * d_phi * rho
        d_phi = delta_rho * (0.2*phi + 0.8) / (0.2*rho) = delta_rho * (phi + 4) / rho

    Returned in the same units as phi (FP mass per GT pixel).
    """
    if rho <= 0:
        return float("inf")
    return float(delta_rho * (phi + BETA / ALPHA) / rho)


def recall_per_fp_exchange_rate(tp_w: float, fp_w: float, n_gt: float) -> float:
    """EQ-3 directly: units of FP mass one extra unit of TP_w is worth."""
    if tp_w <= 0:
        return float("inf")
    return float((ALPHA * fp_w + BETA * n_gt) / (ALPHA * tp_w))


# --------------------------------------------------------------------------- #
# Prediction shape
# --------------------------------------------------------------------------- #
def binarise(field: np.ndarray, budget: int) -> np.ndarray:
    """
    Keep the `budget` highest-scoring pixels at 1.0, everything else 0.0.

    Rationale: EQ-3 makes recall worth 4/rho units of precision, and the kernel
    is a *max*, so partial mass at the right place earns strictly less TP_w per
    unit of FP than full mass.  Concretely, a pixel at distance d from a GT pixel
    contributes p*(1-d/3) to TP but costs p*(1-(1-d/3)) to FP; at p=1, d=0 it is
    pure gain, at p=0.5 it is pure loss.  Binarising is the right move for any
    pixel we are not confident sits on the trace.
    """
    flat = np.asarray(field, dtype=np.float64).reshape(-1)
    out = np.zeros(flat.size, dtype=np.float32)
    k = int(min(max(budget, 0), flat.size))
    if k == 0:
        return out.reshape(field.shape)
    thresh = np.partition(flat, flat.size - k)[flat.size - k]
    idx = np.flatnonzero(flat >= thresh)
    if idx.size > k:                       # ties: keep the lowest linear indices
        idx = idx[np.argsort(idx, kind="stable")[:k]]
    out[idx] = 1.0
    return out.reshape(field.shape)


def budgeted_topk(score: np.ndarray, budget_px: int) -> np.ndarray:
    """Alias kept for call-site readability."""
    return binarise(score, budget_px)


# --------------------------------------------------------------------------- #
# The free-catalogue-core corridor
# --------------------------------------------------------------------------- #
def _dilate(mask: np.ndarray, iterations: int) -> np.ndarray:
    if iterations <= 0:
        return mask.astype(bool)
    try:
        from scipy.ndimage import binary_dilation
    except ImportError:                                   # pure-numpy fallback
        out = mask.astype(bool).copy()
        for _ in range(iterations):
            acc = out.copy()
            acc[1:, :] |= out[:-1, :]
            acc[:-1, :] |= out[1:, :]
            acc[:, 1:] |= out[:, :-1]
            acc[:, :-1] |= out[:, 1:]
            out = acc
        return out
    return binary_dilation(mask.astype(bool), iterations=iterations)


def build_corridor(
    score: np.ndarray,
    catalogue: np.ndarray,
    *,
    halo_px: int = 3,
    budget_px: int | None = None,
    value: float = 1.0,
    core: bool = True,
    mask_safety_px: int = 0,
) -> np.ndarray:
    """
    Build the free-core / paid-halo submission described in the module docstring.

    Parameters
    ----------
    score        : continuous detector score, same shape as `catalogue`
    catalogue    : boolean raster of the known USGS/INGENIOUS faults
    halo_px      : how far out from the catalogue the paid halo reaches (px).
                   3 px = 300 m = exactly the metric's support radius.
    budget_px    : hard cap on paid halo pixels. `None` = no cap (all halo pixels).
    core         : include the free catalogue core at `value`.
    mask_safety_px : pixels within this distance of the catalogue are treated as
                   *paid* (assumed the organiser's rasterisation may differ from
                   ours by this much), so they are subject to the budget.

    Returns
    -------
    float32 array in {0, value}, ready for `build_submission`.

    Note on the budget: the top-k selection is by RANK, not by a `score >= thresh`
    comparison.  A threshold comparison breaks on a score field with many ties --
    and a sparse score field is exactly what a per-fold probability map looks
    like, where the vast majority of pixels are identical.  Ranking keeps the
    paid count at exactly `budget_px`, with ties broken by lowest linear index
    so the output is deterministic.
    """
    catalogue = np.asarray(catalogue).astype(bool)
    free_core = _dilate(catalogue, mask_safety_px) if mask_safety_px > 0 else catalogue
    band = _dilate(catalogue, halo_px) & ~free_core     # halo only, never overlaps core

    out = np.zeros(catalogue.shape, dtype=np.float32)
    if core:
        out[catalogue] = value

    if band.any():
        if budget_px is None:
            out[band] = value
        else:
            k = int(min(budget_px, int(band.sum())))
            if k > 0:
                flat_score = np.asarray(score, dtype=np.float64).reshape(-1)
                idx = np.flatnonzero(band.reshape(-1))    # flat indices into the grid
                vals = flat_score[idx]
                if k < idx.size:
                    # argpartition then a stable sort of the k winners, so ties
                    # resolve to the lowest linear index deterministically.
                    # `cand` indexes into `vals`, NOT into the grid -- it has to
                    # be mapped back through `idx` before it is used as a flat
                    # position, or the paid pixels land at the top-left corner
                    # of the raster instead of along the fault.
                    cand = np.argpartition(vals, idx.size - k)[idx.size - k:]
                    order = np.argsort((-vals[cand], cand), kind="stable")
                    keep = idx[cand[order[:k]]]
                else:
                    keep = idx
                out.reshape(-1)[keep] = value
    return out


def corridor_stats(field: np.ndarray, catalogue: np.ndarray) -> dict:
    """Report how much of a candidate field is free (masked) vs paid."""
    field = np.asarray(field)
    catalogue = np.asarray(catalogue).astype(bool)
    pos = field > 0
    free = pos & catalogue
    paid = pos & ~catalogue
    return {
        "n_positive": int(pos.sum()),
        "n_free_core": int(free.sum()),
        "n_paid": int(paid.sum()),
        "frac_free": float(free.sum() / max(int(pos.sum()), 1)),
    }


# --------------------------------------------------------------------------- #
# Calibration on a training fold
# --------------------------------------------------------------------------- #
def select_corridor_config(
    score: np.ndarray,
    catalogue: np.ndarray,
    gt: np.ndarray,
    train_regions: list[np.ndarray],
    *,
    halos=(0, 1, 2, 3, 4),
    budgets=(0.0, 0.0005, 0.002, 0.01, 0.04, None),
    metric=None,
    min_improvement: float = 0.005,
) -> dict:
    """
    Cross-fitted (halo, budget) selection. This is the function the real
    pipeline should call, and it is deliberately conservative.

    A config is ACCEPTED only if it beats the do-nothing baseline (halo = 0,
    budget = 0: the free catalogue core and nothing else) by `min_improvement`
    on EVERY training region, not merely on their pooled ground truth. Two
    reasons, both measured on blocked phantoms in
    `scripts/validate_holdout.py`:

      * Taking the argmax over a 24-point grid on pooled training data is a max
        of 24 noisy numbers. The winner reliably loses out of sample. Pooled
        cross-fitting cut the held-out loss from -0.023 to -0.004, and only by
        declining to spend.
      * A halo is not free. It adds a whole dilation ring of false-positive
        mass, so a configuration that looks good on one block can be a large
        net loss on the next.

    When nothing clears the bar the answer is the free core, which has
    phi = 0 exactly. That is a feature: the expensive default in this metric is
    the one that guesses, and the guard exists so the pipeline refuses to.
    """
    from .metric import distance_weighted_tversky_fast   # local import: optional dep
    metric = metric or distance_weighted_tversky_fast
    n = int(gt.size)
    grid: dict[tuple, list[float]] = {}
    base: list[float] = []
    for region in train_regions:
        outside = ~region
        rows = sweep_corridor_halo(
            score, catalogue, gt & region, halos=halos, budgets=budgets,
            metric=metric, extra_ignore=outside, min_improvement=0.0)
        for r in rows:
            grid.setdefault((r["halo_px"], r["budget_frac"]), []).append(r["dti"])
        base.append(rows[0]["dti"] if False else next(
            r["dti"] for r in rows
            if r["halo_px"] == 0 and r["budget_frac"] == 0.0))
    zero = (0, 0.0)
    per_config = {
        k: {
            "halo_px": k[0],
            "budget_frac": k[1],
            "dti_mean": float(np.mean(v)),
            "improvement_mean": float(np.mean(v) - np.mean(base)),
            "improvement_min": float(np.min(np.asarray(v) - np.asarray(base))),
        }
        for k, v in grid.items()
    }
    accepted = [c for c in per_config.values()
                if c["improvement_min"] > min_improvement]
    chosen = (max(accepted, key=lambda c: c["improvement_mean"])
              if accepted else per_config[zero])
    return {
        "halo_px": int(chosen["halo_px"]),
        "budget_frac": chosen["budget_frac"],
        "budget_px": (None if chosen["budget_frac"] is None
                      else int(round(chosen["budget_frac"] * n))),
        "accepted": bool(accepted),
        "n_configs": len(per_config),
        "n_accepted": len(accepted),
        "min_improvement": float(min_improvement),
        "expected_improvement": float(chosen["improvement_mean"]),
        "baseline_dti": float(np.mean(base)),
        "grid": per_config,
    }


def sweep_corridor_halo(
    score: np.ndarray,
    catalogue: np.ndarray,
    gt: np.ndarray,
    *,
    halos=(0, 1, 2, 3, 4, 5, 6),
    budgets=(0.01, 0.02, 0.03, 0.05, 0.08, None),
    metric=None,
    extra_ignore: np.ndarray | None = None,
    baseline_dti: float | None = None,
    min_improvement: float = 0.005,
) -> list[dict]:
    """
    Grid-search (halo radius, paid budget) on a TRAINING fold only, scored with
    the real DTI and the official catalogue masking.

    Returns one record per grid point sorted by DTI descending.  Call this on
    held-in folds only; calling it on the validation fold is how you leak.

    `extra_ignore` masks pixels that are out of scope for the evaluation -- in a
    blocked cross-validation, every pixel outside the block being scored.  Leave
    it None when scoring a whole region, otherwise the free core is charged
    false-positive mass for the blocks you are not looking at.

    `extra_ignore` masks FALSE POSITIVES only.  The true-positive term is
    always summed over whatever `gt` you hand in, so when you scope an
    evaluation to a region you must pass `gt & region` as well.  Masking FP
    without also restricting `gt` credits every out-of-region hit for free; on
    a 2x2-block phantom that inflated a 3 px halo from -0.015 to +0.24.

    `min_improvement` is a selection-noise guard, not a tuning knob.  Taking the
    argmax over a 24-point grid on one training split is a max of 24 noisy
    numbers, and the winner routinely loses out of sample.  So the chosen
    configuration must beat the simplest one (halo = 0, i.e. the free catalogue
    core and nothing else) by `min_improvement` before any halo is bought.
    Measured effect of omitting the guard, on 2x2-block phantoms: the corridor
    arm lost 0.004-0.007 DTI against a do-nothing baseline in 80% of phantoms.
    """
    from .metric import distance_weighted_tversky_fast   # local import: optional dep
    metric = metric or distance_weighted_tversky_fast
    n = gt.size
    rows = []
    for h in halos:
        for b in budgets:
            budget_px = None if b is None else int(round(b * n))
            field = build_corridor(score, catalogue, halo_px=h, budget_px=budget_px)
            mask = catalogue if extra_ignore is None else (catalogue | extra_ignore)
            dti, comp = metric(field, gt, fp_ignore_mask=mask)
            st = corridor_stats(field, catalogue)
            rows.append({
                "halo_px": h,
                "budget_frac": b,
                "dti": float(dti),
                "TP_w": comp["TP_w"], "FP_w": comp["FP_w"], "FN_w": comp["FN_w"],
                "n_free": st["n_free_core"], "n_paid": st["n_paid"],
            })
    rows.sort(key=lambda r: -r["dti"])
    if baseline_dti is None:
        baseline = next((r for r in rows if r["halo_px"] == 0 and r["budget_frac"] == 0.0),
                        rows[-1] if rows else None)
        baseline_dti = baseline["dti"] if baseline else 0.0
    for r in rows:
        r["baseline_dti"] = float(baseline_dti)
        r["improvement"] = float(r["dti"] - baseline_dti)
        r["accepted"] = bool(r["improvement"] > min_improvement)
    # The configuration actually used: the best ACCEPTED one, else do nothing.
    ok = [r for r in rows if r["accepted"]]
    rows[0]["selected"] = (ok[0] if ok else next(
        (r for r in rows if r["halo_px"] == 0 and r["budget_frac"] == 0.0), rows[0]))
    return rows


# --------------------------------------------------------------------------- #
# Diagnostics used on the site and in the README
# --------------------------------------------------------------------------- #
def attractor_table(scores=(0.1563, 0.3049, 0.2993, 0.2854)) -> list[dict]:
    """
    For each headline score: the zero-false-positive recall it implies, and the
    recall/false-positive pairs that could also produce it.  Pure algebra.
    """
    out = []
    for s in scores:
        rho0 = required_recall_for_score(s, phi=0.0)
        out.append({
            "score": s,
            "rho_at_zero_fp": rho0,
            "examples": [
                {"rho": r, "phi": max_phi_for_score(s, r)}
                for r in (0.15, 0.20, 0.2592, 0.30, 0.40, 0.50, 0.60, 0.70)
                if max_phi_for_score(s, r) >= 0
            ],
            "fp_affordable_at_rho": {
                str(r): max_phi_for_score(s, r)
                for r in (0.13, 0.2592, 0.5, 0.7)
                if max_phi_for_score(s, r) >= 0
            },
        })
    return out


if __name__ == "__main__":
    import json
    print("DTI algebra sanity (EQ-2):")
    for rho, phi in [(0.1291, 0.0), (0.2592, 0.0), (0.5, 0.0), (0.5, 1.0), (0.8, 3.0), (0.2, 0.0)]:
        print(f"  rho={rho:.4f} phi={phi:.4f} -> DTI={dti_from_ratios(rho, phi):.4f}")
    print(f"\n0.1563 implies rho={required_recall_for_score(0.1563):.4f} at zero FP")
    print(f"0.3049 implies rho={required_recall_for_score(0.3049):.4f} at zero FP")
    print(f"FP mass affordable per unit recall at rho=0.1291: {fp_mass_affordable(0.1291, 0.01):.2f}")
    print(json.dumps(attractor_table(), indent=2, default=float))
