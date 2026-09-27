"""
Distance-weighted Tversky index (DTI) — implements the official competition metric.

Verified line-by-line against the official problem description:
https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/#performance-metric
  - alpha = 0.2 (false-positive penalty), beta = 0.8 (false-negative penalty)
  - triangular kernel k(d) = max(1 - d/R, 0), R = 300 m = 3 px at 100 m resolution
  - TP_w = sum over gt g of max over x with d(x,g)<=R of p(x) * k(d(x,g))
  - FP_w = sum over x with p(x)>0 of p(x) * [1 - max over gt g of k(d(x,g))]
  - FN_w = sum over gt g of [1 - max over x with d(x,g)<=R of p(x) * k(d(x,g))]
  - DTI  = TP_w / (TP_w + alpha*FP_w + beta*FN_w + eps)

Official scoring clarification (DrivenData staff, forum thread 11516, 2026-09-16):
"Pixels corresponding to known USGS/INGENIOUS faults are masked / excluded from
evaluation, so they do not count towards penalty terms." (same in the final round)
https://community.drivendata.org/t/scoring-clarification-are-known-usgs-ingenious-faults-masked-when-scoring-and-are-they-in-the-final-round-label-set/11516
We model this with `fp_ignore_mask`: predicted mass on masked pixels does not
contribute to FP_w (and contributes nothing else, since gt contains only new faults).

Two implementations are provided and cross-tested in tests/test_validation.py:
  - distance_weighted_tversky        : naive per-gt-pixel loop (reference)
  - distance_weighted_tversky_fast   : exact vectorised version (offset max-filter
    + Euclidean distance transform). "Exact" because the image is a pixel grid:
    the set of candidate neighbours of a gt pixel is exactly the integer offsets
    (dy, dx) with sqrt(dy^2+dx^2) <= R, which we enumerate.
"""

import math

import numpy as np
from scipy.ndimage import distance_transform_edt


def triangular_kernel(distance, R=3):
    """Triangular kernel. R in pixels (3 px = 300 m at 100 m resolution)."""
    return np.maximum(1 - np.asarray(distance, dtype=np.float64) / R, 0)


def _validate_inputs(pred, gt):
    pred = np.asarray(pred, dtype=np.float32)
    gt = np.asarray(gt).astype(bool)
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs gt {gt.shape}")
    if pred.ndim != 2:
        raise ValueError("pred/gt must be 2-D arrays")
    return pred, gt


def _fp_term(pred, gt, R, fp_ignore_mask):
    """FP_w via Euclidean distance transform (exact on the pixel grid)."""
    dist_to_gt = distance_transform_edt(~gt)
    k_to_gt = triangular_kernel(dist_to_gt, R=R)
    fp_weight = pred.astype(np.float64) * (1.0 - k_to_gt)
    if fp_ignore_mask is not None:
        fp_weight = np.where(np.asarray(fp_ignore_mask, dtype=bool), 0.0, fp_weight)
    return float(fp_weight.sum())


def _tp_fn_terms(pred, gt, R):
    """For each gt pixel, the max of p(x)*k(d) over integer offsets within R."""
    H, W = gt.shape
    padded = np.pad(pred.astype(np.float64), R, mode="constant", constant_values=0.0)
    best = np.zeros((H, W), dtype=np.float64)
    for dy in range(-R, R + 1):
        for dx in range(-R, R + 1):
            d = math.hypot(dy, dx)
            if d > R:
                continue
            w = 1.0 - d / R
            shifted = padded[R + dy:R + dy + H, R + dx:R + dx + W]
            np.maximum(best, shifted * w, out=best)
    tp_w = float(best[gt].sum())
    fn_w = float((1.0 - best[gt]).sum())
    return tp_w, fn_w, best


def distance_weighted_tversky_fast(pred, gt, alpha=0.2, beta=0.8, R=3, eps=1e-7,
                                   fp_ignore_mask=None):
    """
    Exact, vectorised DTI.

    pred: HxW float in [0,1]
    gt:   HxW binary (0/1) or bool — ground-truth (new) fault pixels
    fp_ignore_mask: optional HxW bool — pixels excluded from evaluation
        (official masking of known USGS/INGENIOUS catalogue faults).
    Returns (DTI, dict(TP_w, FP_w, FN_w)).
    """
    pred, gt = _validate_inputs(pred, gt)
    n_gt = int(gt.sum())
    if n_gt == 0:
        # With no ground truth every unmasked positive pixel is a pure false positive
        fp_weight = pred.astype(np.float64).copy()
        if fp_ignore_mask is not None:
            fp_weight = np.where(np.asarray(fp_ignore_mask, dtype=bool), 0.0, fp_weight)
        FP_w = float(fp_weight.sum())
        return 0.0, dict(TP_w=0.0, FP_w=FP_w, FN_w=0.0)

    FP_w = _fp_term(pred, gt, R, fp_ignore_mask)
    TP_w, FN_w, _ = _tp_fn_terms(pred, gt, R)
    dti = TP_w / (TP_w + alpha * FP_w + beta * FN_w + eps)
    return float(dti), dict(TP_w=TP_w, FP_w=FP_w, FN_w=FN_w)


def distance_weighted_tversky(pred, gt, alpha=0.2, beta=0.8, R=3, eps=1e-7):
    """
    Naive reference implementation (per-gt-pixel 7x7 window loop).
    Kept for cross-validation of the fast path. Same semantics as
    distance_weighted_tversky_fast with fp_ignore_mask=None.
    """
    pred, gt = _validate_inputs(pred, gt)

    if gt.sum() == 0:
        FP_w = float(pred.sum())
        return 0.0, dict(TP_w=0.0, FP_w=FP_w, FN_w=0.0)

    # FP_w: sum p(x) * (1 - k(d(x, G))) using the distance transform
    dist_to_gt = distance_transform_edt(~gt)
    k_to_gt = triangular_kernel(dist_to_gt, R=R)
    FP_w = float(np.sum(pred.astype(np.float64) * (1 - k_to_gt)))

    # TP_w / FN_w: loop over gt pixels, search (2R+1) x (2R+1) window
    H, W = gt.shape
    gt_indices = np.argwhere(gt)
    pad = R
    pred_padded = np.pad(pred.astype(np.float64), pad, mode="constant", constant_values=0)
    yy, xx = np.ogrid[-R:R + 1, -R:R + 1]
    dist_kernel = np.sqrt(xx * xx + yy * yy)
    k_kernel = triangular_kernel(dist_kernel, R=R)

    TP_w = 0.0
    FN_w = 0.0
    for (y, x) in gt_indices:
        yp, xp = y + pad, x + pad
        window = pred_padded[yp - R:yp + R + 1, xp - R:xp + R + 1]
        max_val = float((window * k_kernel).max())
        TP_w += max_val
        FN_w += (1 - max_val)

    dti = TP_w / (TP_w + alpha * FP_w + beta * FN_w + eps)
    return float(dti), dict(TP_w=TP_w, FP_w=FP_w, FN_w=FN_w)


def degenerate_baselines():
    """Closed-form DTI for degenerate strategies vs the public catalogue.

    Numbers c, N from GEMSDOE1 measurements (labels: 60,988 fault px inside a
    5,167,373 px footprint; full raster 12,279,160 px). Used on the
    verification page only.
    """
    c = 60988 / 5167373
    alpha = 0.2
    return {
        "coverage": c,
        "all_zero": 0.0,
        "all_one_everywhere": 60988 / (60988 + 0.2 * (12279160 - 60988)),
        "all_one_inside_footprint_closed_form": c / (c + alpha * (1 - c)),
    }


if __name__ == "__main__":
    # Smoke test
    pred = np.zeros((10, 10), dtype=np.float32)
    gt = np.zeros((10, 10), dtype=bool)
    gt[5, 5] = True
    pred[5, 5] = 1.0
    dti, comp = distance_weighted_tversky_fast(pred, gt)
    print(f"DTI perfect: {dti:.6f} {comp}")
    dti_n, comp_n = distance_weighted_tversky(pred, gt)
    print(f"DTI perfect (naive): {dti_n:.6f} {comp_n}")
    pred[5, 6] = 1.0
    dti, comp = distance_weighted_tversky_fast(pred, gt)
    print(f"DTI off-by-one FP: {dti:.6f} {comp}")
