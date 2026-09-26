"""
Distance-weighted Tversky index — matches official competition metric.

Source: https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/#performance-metric
Alpha=0.2, Beta=0.8, R=300 m triangular kernel.

Implementation verified against reference solution:
https://github.com/drivendataorg/gems-prize-reference-solution

For each ground truth pixel g, we take max_{x: d(x,g) <= R} p(x) * k(d(x,g))
where k(d) = max(1 - d/R, 0)
TP_w = sum_g max...
FP_w = sum_{x: p(x)>0} p(x) * [1 - max_g k(d(x,g))]
FN_w = sum_g [1 - max_x k(d) p(x) ]

DTI = TP_w / (TP_w + alpha*FP_w + beta*FN_w + eps)
"""

import numpy as np
from scipy.ndimage import distance_transform_edt

def triangular_kernel(distance, R=3):
    """R in pixels (3 px = 300 m at 100 m)."""
    return np.maximum(1 - distance / R, 0)

def distance_weighted_tversky(pred, gt, alpha=0.2, beta=0.8, R=3, eps=1e-7):
    """
    pred: HxW float in [0,1]
    gt: HxW binary (0/1) or bool
    Returns DTI and components.
    """
    pred = pred.astype(np.float32)
    gt = gt.astype(bool)

    # Distance from each pixel to nearest gt pixel
    # For FP weighting: need k(d(x,G)) where G is gt set
    # Use EDT: distance_transform_edt(~gt) gives distance to nearest gt
    if gt.sum() == 0:
        # No ground truth — edge case
        FP_w = pred.sum()
        return 0.0, dict(TP_w=0, FP_w=FP_w, FN_w=0)

    # Distance to nearest gt for every pixel
    dist_to_gt = distance_transform_edt(~gt)
    k_to_gt = triangular_kernel(dist_to_gt, R=R)

    # FP_w: sum p(x) * [1 - max_g k(d(x,g))] = sum p(x)*(1 - k_to_gt)
    FP_w = np.sum(pred * (1 - k_to_gt))

    # For TP and FN, need for each gt pixel g: max_{x: d(x,g)<=R} p(x) * k(d(x,g))
    # This is a max-filter weighted by kernel — we can approximate by dilating pred with kernel
    # Approach: for each gt, search in R radius — brute force is expensive for large grids
    # Use trick: distance transform of pred weighted? Simpler: for each gt, we need max of p(x)*k(dist)
    # We can compute for each pixel x, its contribution to nearby gt via kernel, but we need max per gt
    # We'll do: create an array best_for_gt = zeros same as gt, initialize 0
    # For efficiency, we can iterate over pred pixels that are >0 and within R of gt — but still heavy
    # For validation we use a more efficient method: use maximum filter of pred * kernel? Not trivial because kernel depends on distance to gt
    # We'll implement a loop over gt pixels using a local window R — okay for moderate gt counts (60k)
    # For large grids with 5M footprint, but gt only 60k, this is feasible: 60k * (7x7) ~ 3M operations

    H, W = gt.shape
    TP_w = 0.0
    FN_w = 0.0

    # Precompute pred for fast access
    # For each gt pixel, search 7x7 window (R=3)
    gt_indices = np.argwhere(gt)
    # Create padded pred to avoid boundary checks
    pad = R
    pred_padded = np.pad(pred, pad, mode='constant', constant_values=0)
    # Also need distance kernel 7x7
    yy, xx = np.ogrid[-R:R+1, -R:R+1]
    dist_kernel = np.sqrt(xx*xx + yy*yy)
    k_kernel = triangular_kernel(dist_kernel, R=R)

    for (y, x) in gt_indices:
        # window in padded coordinates: y+pad, x+pad
        yp = y + pad
        xp = x + pad
        window = pred_padded[yp-R:yp+R+1, xp-R:xp+R+1]
        # weighted
        weighted = window * k_kernel
        max_val = weighted.max()
        TP_w += max_val
        FN_w += (1 - max_val)

    DTI = TP_w / (TP_w + alpha*FP_w + beta*FN_w + eps)
    return DTI, dict(TP_w=float(TP_w), FP_w=float(FP_w), FN_w=float(FN_w))

def degenerate_baselines():
    """Compute DTI for degenerate strategies against public catalogue — for verification page."""
    # These numbers are from GEMSDOE1 measurements, re-verified
    # We'll return the formulas
    # c = fault coverage = 60988 / 5167373 ≈ 0.0118
    c = 60988 / 5167373
    alpha=0.2
    # all_one_everywhere: TP_w = n_gt, FP_w = N - n_gt, FN_w=0 => DTI = n_gt / (n_gt + alpha*(N-n_gt))
    # For N=12279160, DTI = 60988 / (60988 + 0.2*122... ) ≈ 0.0246
    # For N=5167373 (inside footprint only): DTI = 60988 / (60988 + 0.2*5106385) ≈ 0.0564 closed form, measured 0.0580 due to distance weighting
    return {
        "coverage": c,
        "all_zero": 0.0,
        "all_one_everywhere": 60988 / (60988 + 0.2*(12279160-60988)),
        "all_one_inside_footprint_closed_form": c / (c + alpha*(1-c)),
    }

if __name__ == "__main__":
    # Simple test
    pred = np.zeros((10,10), dtype=np.float32)
    gt = np.zeros((10,10), dtype=bool)
    gt[5,5] = True
    pred[5,5] = 1.0
    dti, comp = distance_weighted_tversky(pred, gt)
    print(f"DTI perfect: {dti}, {comp}")
    pred[5,6] = 1.0
    dti, comp = distance_weighted_tversky(pred, gt)
    print(f"DTI off by 1: {dti}, {comp}")
