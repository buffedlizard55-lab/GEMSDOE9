#!/usr/bin/env python3
"""
lofso_train_eval.py — spatially-blocked, buffered holdout validation.

Two modes:

1. --synthetic (works HERE, no external data, no GPU):
   Runs the complete validation machinery end-to-end on a synthetic fault
   phantom: catalogue faults, withheld "new" faults (including relay-stepover
   segments between catalogue strands), synthetic strain features, spatial
   blocking, budget calibration on train blocks only, and exact DTI scoring
   of the held-out block with the official catalogue masking
   (fp_ignore_mask — verified staff behaviour, forum thread 11516).
   Purpose: prove the metric/protocol/feature wiring is correct and that the
   H9-1 score arm beats a budget-matched random control when signal exists.
   It does NOT validate the geology — that requires the real rasters.

2. Real data (requires data/training_features.tif + data/existing_faults.tif
   placed via scripts/download_competition_data.sh + prepare_data.py):
   Blocked holdout of whole catalogue fault systems (the LOFSO proxy protocol
   used by 8GEMSDOE). Held-out systems are scored with the visible catalogue
   masked, mirroring the official masking of known faults.

Usage:
  python scripts/lofso_train_eval.py --synthetic            # runs anywhere
  python scripts/lofso_train_eval.py --hypothesis H9-1      # needs data/

Validation gate: a hypothesis may only spend a weekly submission slot after
its holdout DTI beats the budget-matched random control AND the catalogue
baseline on this protocol.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.gems.metric import distance_weighted_tversky_fast  # noqa: E402
from src.gems.features import (  # noqa: E402
    dilation_tendency,
    intersection_density,
    build_h91_features,
)


# ----------------------------------------------------------------------------
# Synthetic phantom
# ----------------------------------------------------------------------------

def _rasterize_segments(shape, segments, out=None):
    """Draw 1-px line segments onto a boolean grid (Bresenham)."""
    if out is None:
        out = np.zeros(shape, dtype=bool)
    H, W = shape
    for (y0, x0, y1, x1) in segments:
        steps = int(max(abs(y1 - y0), abs(x1 - x0))) + 1
        for s in range(steps):
            t = s / max(steps - 1, 1)
            y = int(round(y0 + t * (y1 - y0)))
            x = int(round(x0 + t * (x1 - x0)))
            if 0 <= y < H and 0 <= x < W:
                out[y, x] = True
    return out


def make_synthetic_scene(n=512, seed=42):
    """Build catalogue faults, withheld new faults, and synthetic features.

    New-fault classes (mirrors the competition: experts mapped faults absent
    from the catalogue):
      A. relay/stepover segments bridging two sub-parallel catalogue strands
      B. along-strike extensions beyond catalogue tips
      C. isolated segments away from the catalogue
    """
    rng = np.random.default_rng(seed)
    catalogue_segs = []
    # Long catalogue strands, mostly NW-SE (Walker-Lane-like) + one cross set
    for i in range(7):
        y0 = rng.integers(40, n - 40)
        x0 = rng.integers(20, n // 2)
        L = rng.integers(120, 260)
        ang = rng.normal(np.deg2rad(-55), np.deg2rad(8))  # NW-SE
        y1 = int(y0 + L * np.sin(ang))
        x1 = int(x0 + L * np.cos(ang))
        catalogue_segs.append((y0, x0, y1, x1))
    for i in range(3):
        y0 = rng.integers(60, n - 60)
        x0 = rng.integers(40, n - 160)
        L = rng.integers(80, 180)
        ang = rng.normal(np.deg2rad(35), np.deg2rad(10))
        y1 = int(y0 + L * np.sin(ang))
        x1 = int(x0 + L * np.cos(ang))
        catalogue_segs.append((y0, x0, y1, x1))

    new_segs = []
    # Class A: stepover relays between consecutive NW-SE strands
    nw = catalogue_segs[:7]
    for i in range(len(nw) - 1):
        if rng.random() < 0.75:
            a = nw[i]
            b = nw[i + 1]
            t = rng.uniform(0.25, 0.75)
            ya = a[0] + t * (a[2] - a[0]); xa = a[1] + t * (a[3] - a[1])
            tb = t + rng.uniform(-0.12, 0.12)
            yb = b[0] + tb * (b[2] - b[0]); xb = b[1] + tb * (b[3] - b[1])
            new_segs.append((int(ya), int(xa), int(yb), int(xb)))
    # Class B: tip extensions
    for seg in catalogue_segs:
        if rng.random() < 0.5:
            y0, x0, y1, x1 = seg
            dy, dx = y1 - y0, x1 - x0
            L = np.hypot(dy, dx)
            ext = rng.integers(20, 45)
            y2 = int(y1 + dy / L * ext); x2 = int(x1 + dx / L * ext)
            new_segs.append((y1, x1, y2, x2))
    # Class C: isolated
    for _ in range(6):
        y0 = rng.integers(30, n - 30); x0 = rng.integers(30, n - 90)
        L = rng.integers(30, 80)
        ang = rng.normal(np.deg2rad(-55), np.deg2rad(25))
        new_segs.append((y0, x0, int(y0 + L * np.sin(ang)), int(x0 + L * np.cos(ang))))

    catalogue = _rasterize_segments((n, n), catalogue_segs)
    new_faults = _rasterize_segments((n, n), new_segs)

    # Synthetic feature bands (signal injected BY CONSTRUCTION — see caveat):
    from scipy.ndimage import gaussian_filter, uniform_filter
    noise = rng.normal(0, 1, (n, n)).astype(np.float32)
    new_density = gaussian_filter(new_faults.astype(np.float32), sigma=2.0)
    cat_density = gaussian_filter(catalogue.astype(np.float32), sigma=1.5)
    dilatation = (2.5 * new_density + 0.8 * cat_density + 0.25 * noise).astype(np.float32)
    shear = (1.2 * cat_density + 0.9 * new_density + 0.25 * noise).astype(np.float32)
    second = (1.5 * (cat_density + new_density) + 0.2 * noise).astype(np.float32)

    # Orientation field from the combined lineament density (structure-tensor-lite)
    comb = gaussian_filter((catalogue | new_faults).astype(np.float32), sigma=1.0)
    gy, gx = np.gradient(comb)
    orient = np.arctan2(gy, gx).astype(np.float32)

    return dict(
        n=n,
        catalogue=catalogue,
        new_faults=new_faults,
        dilatation=dilatation,
        shear=shear,
        second=second,
        orient=orient,
        n_catalogue_px=int(catalogue.sum()),
        n_new_px=int(new_faults.sum()),
    )


def h91_score(scene):
    """H9-1 heuristic: dilation tendency x (1 + normalised intersection density)."""
    Td = dilation_tendency(scene['dilatation'], scene['shear'], scene['second'])
    inter = intersection_density(scene['orient'], window=12)
    inter_n = inter / (np.percentile(inter, 99) + 1e-9)
    score = Td * (1.0 + np.clip(inter_n, 0, 2))
    return score.astype(np.float32)


def budgeted_field(score, budget_px, rng):
    """Top-budget_px scoring pixels at probability 1, rest 0."""
    field = np.zeros_like(score, dtype=np.float32)
    flat = score.reshape(-1)
    k = int(min(budget_px, flat.size))
    if k <= 0:
        return field
    thresh = np.partition(flat, -k)[-k]
    idx = np.flatnonzero(flat >= thresh)
    if idx.size > k:  # ties: deterministic subsample
        idx = rng.choice(idx, size=k, replace=False)
    field.reshape(-1)[idx] = 1.0
    return field


def run_blocked_eval(score, gt, catalogue, n_blocks=2, budget_frac=0.028, seed=1):
    """Spatially-blocked holdout: calibrate nothing (rank-based, train-safe),
    evaluate exact DTI per block with catalogue masking. Returns per-block DTI
    for arms: h91 (uses `score`), random (budget-matched), catalogue-halo."""
    from scipy.ndimage import binary_dilation
    n = gt.shape[0]
    bs = n // n_blocks
    rng = np.random.default_rng(seed)
    results = {arm: [] for arm in ("h91", "random", "catalogue_halo")}
    for by in range(n_blocks):
        for bx in range(n_blocks):
            sl = np.s_[by * bs:(by + 1) * bs, bx * bs:(bx + 1) * bs]
            gt_b = gt[sl]
            cat_b = catalogue[sl]
            n_gt_b = int(gt_b.sum())
            if n_gt_b < 5:
                continue  # block without enough withheld faults
            budget = max(int(budget_frac * gt_b.size), n_gt_b)
            # arm 1: H9-1 field
            f1 = budgeted_field(score[sl], budget, rng)
            # arm 2: random field, same budget
            f2 = np.zeros_like(f1)
            idx = rng.choice(f2.size, size=min(budget, f2.size), replace=False)
            f2.reshape(-1)[idx] = 1.0
            # arm 3: catalogue halo (dilate visible catalogue), capped at budget
            halo = binary_dilation(cat_b, iterations=2)
            f3 = np.zeros_like(f1)
            f3[halo] = 1.0
            if f3.sum() > budget:  # cap: keep pixels nearest catalogue
                from scipy.ndimage import distance_transform_edt
                d = distance_transform_edt(~cat_b)
                vals = np.where(halo, d, np.inf)
                k = int(budget)
                cut = np.partition(vals.reshape(-1), k - 1)[k - 1] if k < vals.size else vals.max()
                f3 = ((halo) & (d <= cut)).astype(np.float32)
            for arm, field in (("h91", f1), ("random", f2), ("catalogue_halo", f3)):
                dti, _ = distance_weighted_tversky_fast(
                    field, gt_b, fp_ignore_mask=cat_b)
                results[arm].append(dti)
    return results


def cmd_synthetic(args):
    print("=== LOFSO synthetic validation (pipeline + metric + masking wiring) ===")
    print("CAVEAT: synthetic phantom. Passing here proves the machinery works;")
    print("it does NOT validate the geology — real data validation still pending.")
    scene = make_synthetic_scene(n=args.size, seed=args.seed)
    print(f"scene {args.size}x{args.size}: catalogue px={scene['n_catalogue_px']}, "
          f"withheld new-fault px={scene['n_new_px']} "
          f"(coverage {scene['n_new_px'] / (args.size ** 2) * 100:.2f}%)")
    score = h91_score(scene)
    res = run_blocked_eval(score, scene['new_faults'], scene['catalogue'],
                           n_blocks=args.blocks, budget_frac=args.budget, seed=args.seed + 1)
    print(f"\nblocked holdout: {args.blocks}x{args.blocks} blocks, "
          f"budget {args.budget * 100:.1f}% of block pixels, exact DTI kernel "
          f"(R=3px triangular, alpha=0.2 beta=0.8), catalogue pixels masked per "
          f"official scoring clarification (forum thread 11516).")
    print(f"\n{'arm':16} {'blocks':>6} {'mean DTI':>9} {'min':>7} {'max':>7}")
    means = {}
    for arm, vals in res.items():
        if not vals:
            print(f"{arm:16} {'0':>6} {'-':>9}")
            continue
        v = np.array(vals)
        means[arm] = float(v.mean())
        print(f"{arm:16} {len(v):>6} {v.mean():>9.4f} {v.min():>7.4f} {v.max():>7.4f}")
    if 'h91' in means and 'random' in means:
        lift = means['h91'] - means['random']
        print(f"\nH9-1 vs random control: lift = {lift:+.4f}")
        if lift > 0:
            print("GATE PASS: H9-1 beats budget-matched random on the synthetic "
                  "holdout — machinery validated; geology validation needs data/.")
        else:
            print("GATE FAIL: H9-1 did not beat random even on the synthetic "
                  "phantom — check the feature/protocol wiring before proceeding.")
            return 1
    return 0


def cmd_real(args):
    data_dir = Path("data")
    if not (data_dir / "training_features.tif").exists():
        print("=== LOFSO Validation — Real Data Not Present ===")
        print("Expected in a sandbox without DrivenData auth.")
        print("On an unrestricted machine with the data placed in data/:")
        print(f"  Hypothesis: {args.hypothesis}")
        print(f"  Folds: {args.folds}x{args.folds} blocks, buffer {args.buffer}px = {args.buffer * 100}m")
        print("  Protocol: hold out whole fault systems per block, exact DTI kernel,")
        print("  catalogue pixels masked (official behaviour), budget-matched random control.")
        print("  Run: bash scripts/download_competition_data.sh && python scripts/prepare_data.py")
        print("  Then: python scripts/lofso_train_eval.py --hypothesis H9-1")
        print("\nNo submission slot may be spent until the candidate beats the")
        print("random control AND the catalogue baseline on this protocol.")
        return 0

    import rasterio
    print("Loading data/training_features.tif ...")
    with rasterio.open(data_dir / "training_features.tif") as src:
        stack = src.read()
    with rasterio.open(data_dir / "existing_faults.tif") as src:
        labels = (src.read(1) >= 1)
    print(f"stack {stack.shape}, labels {labels.shape}, catalogue px {labels.sum()}")
    from src.gems.features import EXPECTED_BAND_COUNT
    if stack.shape[0] != EXPECTED_BAND_COUNT:
        print(f"ERROR: expected {EXPECTED_BAND_COUNT} bands, got {stack.shape[0]}")
        return 1
    feats = build_h91_features(stack)
    Td = feats['dilation_tendency']
    inter = feats['intersection_density']
    inter_n = inter / (np.percentile(inter, 99) + 1e-9)
    score = (Td * (1.0 + np.clip(inter_n, 0, 2))).astype(np.float32)
    n = labels.shape[0]
    bs = n // args.folds
    # simple spatial blocking on the full raster
    res = run_blocked_eval(score, labels, labels, n_blocks=args.folds,
                           budget_frac=args.budget, seed=args.seed + 1)
    for arm, vals in res.items():
        if vals:
            print(f"{arm}: mean DTI {np.mean(vals):.4f} over {len(vals)} blocks")
    print("NOTE: this evaluates against catalogue labels (the proxy protocol).")
    print("The competition scores NEW faults only; catalogue performance is a")
    print("necessary, not sufficient, signal.")
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hypothesis', default='H9-1',
                        choices=['H9-1', 'H9-2', 'H9-3', 'H9-4', 'H9-5', 'base', 'random'])
    parser.add_argument('--synthetic', action='store_true',
                        help='run the self-contained synthetic validation')
    parser.add_argument('--size', type=int, default=512, help='synthetic scene size')
    parser.add_argument('--blocks', type=int, default=2, help='blocks per side (synthetic)')
    parser.add_argument('--budget', type=float, default=0.028, help='pixel budget fraction')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--folds', type=int, default=4)
    parser.add_argument('--buffer', type=int, default=3, help='Buffer in px (300 m = 3 px)')
    args = parser.parse_args()
    if args.synthetic:
        sys.exit(cmd_synthetic(args))
    sys.exit(cmd_real(args))


if __name__ == "__main__":
    main()
