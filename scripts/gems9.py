#!/usr/bin/env python3
"""GEMS9: preregistered basin-edge candidate and spatial catalogue holdout.

This is NOT a private-new-fault validation. No automated DrivenData upload.
Requires numpy, scipy, rasterio; no label-derived feature engineering.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import rasterio
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
PINS = {
    'training_features.tif': '4371c82e3b8339b807bdffcf4ef59a225520fe2988d521be208ae33743123bc5',
    'labels.tif': '7ba308ccdc4418b31a178f4f1ef21aaa6e152e4028f2f6f64b01f7eb25ae4093',
    'sample_submission.tif': '2176d08e485aa2cd2860ce8df539db4faf4d76163b38a4dd8c30a40454d35cbc',
}
BANDS = {13: 'iso_grav_anom', 15: 'depth_to_base_surf'}


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for b in iter(lambda: stream.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def sources(data):
    for name, expected in PINS.items():
        p = data / name
        if not p.exists() or digest(p) != expected:
            raise ValueError(f'{name}: absent or SHA256 mismatch against bridge manifest; stop')
    with rasterio.open(data / 'sample_submission.tif') as template, \
         rasterio.open(data / 'labels.tif') as labels, \
         rasterio.open(data / 'training_features.tif') as features:
        assert template.count == labels.count == 1 and features.count == 19
        assert all(x.shape == template.shape and x.crs == template.crs and x.transform == template.transform for x in (labels, features))
        assert template.crs.to_epsg() == 32611 and template.res == (100, 100)
        valid = np.isfinite(template.read(1))
        l = labels.read(1)
        assert np.array_equal(valid, l != labels.nodata)
        assert int(valid.sum()) == 5167373 and int((l == 1).sum()) == 60988
        bands = {}
        for idx, name in BANDS.items():
            actual = features.tags(idx).get('band_name', features.descriptions[idx - 1].split(' - ')[0])
            if actual != name:
                raise ValueError(f'band {idx} should be {name}, got {actual}')
            a = features.read(idx)
            bands[name] = np.where((a != features.nodata) & valid & np.isfinite(a), a, np.nan).astype(np.float32)
        return valid, (l == 1) & valid, bands


def smooth(a, valid, sigma):
    # Normalised convolution prevents sentinel/no-data and footprint boundaries
    # from creating fictitious edges. Boundary response is suppressed later.
    ok = valid & np.isfinite(a)
    values = ndi.gaussian_filter(np.where(ok, a, 0).astype(np.float32), sigma)
    weights = ndi.gaussian_filter(ok.astype(np.float32), sigma)
    return values / np.maximum(weights, 1e-6), weights


def normalise_gradient(g, valid):
    # Robust global scale, not fitted to any fault labels.
    scale = np.percentile(g[valid], 95)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('invalid gradient scale')
    return np.minimum(g / scale, 4).astype(np.float32)


def candidate(bands, valid, sigma=2.0):
    """Signed, *anti-parallel* gradient agreement: depth ↑ with gravity ↓.

    Also return single-layer ablations; these are not another upload strategy.
    """
    d, wd = smooth(bands['depth_to_base_surf'], valid, sigma)
    g, wg = smooth(bands['iso_grav_anom'], valid, sigma)
    dy, dx = np.gradient(d)
    gy, gx = np.gradient(g)
    dm = np.hypot(dx, dy)
    gm = np.hypot(gx, gy)
    dn = normalise_gradient(dm, valid & (wd > .99))
    gn = normalise_gradient(gm, valid & (wg > .99))
    cosine = (dx * gx + dy * gy) / np.maximum(dm * gm, 1e-12)
    anti = np.clip(-cosine, 0, 1)
    support = valid & (wd > .99) & (wg > .99)
    score = dn * gn * anti * anti
    return {k: np.where(support, np.nan_to_num(v, nan=0, posinf=0, neginf=0), 0).astype(np.float32)
            for k, v in {'basin_anti_gradient': score, 'depth_edge_only': dn, 'gravity_edge_only': gn}.items()}


def topk(score, eligible, count):
    """Deterministic exact top-k, stable ties; excludes zero-support cells."""
    indices = np.flatnonzero((eligible & np.isfinite(score) & (score > 0)).ravel())
    count = min(count, len(indices))
    result = np.zeros(score.size, dtype=bool)
    if count:
        values = score.ravel()[indices]
        # Stable descending ordering; ties break by index, not rasterio traversal accident.
        order = np.argsort(-values, kind='stable')[:count]
        result[indices[order]] = True
    return result.reshape(score.shape)


def dti(pred, truth):
    """Official R=3px triangular max-pool TP and nearest-truth weighted FP.

    No known-fault mask needed within a held-out block: ALL positive catalogue
    pixels inside that block are pseudo-truth, all other blocks are excluded.
    """
    gt = truth.astype(bool)
    if not gt.any():
        raise ValueError('fold has no positive truth')
    p = np.asarray(pred, dtype=np.float32)
    fp_weight = np.minimum(ndi.distance_transform_edt(~gt) / 3.0, 1.0)
    fp = float(np.sum(p * fp_weight, dtype=np.float64))
    best = np.zeros(p.shape, dtype=np.float32)
    h, w = p.shape
    for y in range(-2, 3):
        for x in range(-2, 3):
            weight = 1.0 - np.hypot(y, x) / 3.0
            if weight <= 0: continue
            sy = slice(max(0, y), h + min(0, y))
            sx = slice(max(0, x), w + min(0, x))
            ty = slice(max(0, -y), h + min(0, -y))
            tx = slice(max(0, -x), w + min(0, -x))
            np.maximum(best[ty, tx], p[sy, sx] * weight, out=best[ty, tx])
    tp = float(np.sum(best[gt], dtype=np.float64))
    fn = float(gt.sum()) - tp
    return tp / (tp + .2 * fp + .8 * fn + 1e-12)


def evaluate(scores, valid, labels, budget=.03):
    """Four pre-fixed disjoint 2x2 geographic folds, 3px edge buffer.

    No labels used in feature building. Catalogue proxy != hidden new faults.
    Ranking budget is fixed at 3% of eligible valid cells *per fold*.
    """
    h, w = valid.shape
    result = {'protocol': '2x2 quadrants, 3px interior collar, exact R=3 DTI; catalogue-only proxy',
              'budget_fraction': budget, 'folds': []}
    for row in range(2):
        for col in range(2):
            y0, y1 = (0, h // 2) if row == 0 else (h // 2, h)
            x0, x1 = (0, w // 2) if col == 0 else (w // 2, w)
            v = valid[y0:y1, x0:x1].copy()
            v[:3] = False; v[-3:] = False; v[:, :3] = False; v[:, -3:] = False
            gt = labels[y0:y1, x0:x1] & v
            k = int(round(budget * int(v.sum())))
            if k < 1 or not gt.any(): raise ValueError('empty geographic fold')
            fold = {'quadrant': [row, col], 'valid_pixels': int(v.sum()), 'truth_pixels': int(gt.sum()),
                    'target_budget': k, 'arms': {}}
            # Avoid truth leakage: all input scores built without catalogue labels.
            for name, full in scores.items():
                selected = topk(full[y0:y1, x0:x1], v, k)
                fold['arms'][name] = {'dti': dti(selected, gt), 'emitted': int(selected.sum())}
            for seed in (17, 23, 41):
                rng = np.random.default_rng(seed + 4*row + col)
                # equivalent exact-k random ranking over the SAME geographic population
                idx = np.flatnonzero(v.ravel())
                selected = np.zeros(v.size, dtype=bool)
                selected[rng.choice(idx, size=k, replace=False)] = True
                fold['arms'][f'random_{seed}'] = {'dti': dti(selected.reshape(v.shape), gt), 'emitted': k}
            result['folds'].append(fold)
    result['mean_dti'] = {key: float(np.mean([f['arms'][key]['dti'] for f in result['folds']]))
                          for key in result['folds'][0]['arms']}
    result['local_best_control'] = max(v for k, v in result['mean_dti'].items() if k != 'basin_anti_gradient')
    result['beats_local_controls'] = result['mean_dti']['basin_anti_gradient'] > result['local_best_control']
    # Historical 8GEMSDOE LOFSO 0.0933 uses a different truth and budget protocol.
    # NEVER assert approval based on numerical comparison to that score.
    result['approved_for_submission'] = False
    result['approval_reason'] = 'Catalogue-only geography holdout does not test hidden expert-new-fault truth or reproduce prior LOFSO controls; no matched current-best proof.'
    return result


def write_prediction(path, score, valid, template_path, budget=.03):
    """Experimental artifact: exact template footprint; no approval implied."""
    output = np.full(valid.shape, np.nan, dtype=np.float32)
    emission = topk(score, valid, int(round(budget * valid.sum())))
    output[valid] = emission[valid].astype(np.float32)
    with rasterio.open(template_path) as template:
        profile = template.profile.copy()
        profile.update(driver='GTiff', count=1, dtype='float32', nodata=np.nan, compress='deflate', predictor=3)
        path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(path, 'w', **profile) as dst:
            dst.write(output, 1)
    validate(path, template_path)
    return int(emission.sum())


def validate(path, template_path):
    with rasterio.open(template_path) as t, rasterio.open(path) as src:
        if src.count != 1 or src.dtypes != ('float32',) or src.crs != t.crs or src.shape != t.shape or src.transform != t.transform or not np.isnan(src.nodata):
            raise ValueError('wrong shape/CRS/transform/bands/dtype/nodata')
        expected = np.isfinite(t.read(1))
        a = src.read(1)
        if not np.array_equal(np.isfinite(a), expected) or not np.all((a[expected] >= 0) & (a[expected] <= 1)):
            raise ValueError('outside footprint must be NaN; all inside must be finite in [0,1]')
    return {'sha256': digest(path), 'positive_pixels': int((a == 1).sum()), 'valid_pixels': int(expected.sum()), 'format_valid': True}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['evaluate', 'build-experimental', 'validate'])
    p.add_argument('--data', type=Path, default=ROOT / 'data')
    p.add_argument('--output', type=Path, default=ROOT / 'reports' / 'basin-experimental.tif')
    p.add_argument('--report', type=Path, default=ROOT / 'reports' / 'holdout.json')
    p.add_argument('--budget', type=float, default=.03)
    a = p.parse_args()
    if not (0 < a.budget <= .1): p.error('budget must be in (0, .1]')
    if a.action == 'validate':
        print(json.dumps(validate(a.output, a.data / 'sample_submission.tif'), indent=2)); return
    valid, labels, bands = sources(a.data)
    scores = candidate(bands, valid)
    if a.action == 'evaluate':
        result = evaluate(scores, valid, labels, a.budget)
        result['sha256_inputs'] = PINS
        a.report.parent.mkdir(parents=True, exist_ok=True)
        a.report.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps({k: result[k] for k in ('mean_dti','beats_local_controls','approved_for_submission')}, indent=2))
    else:
        n = write_prediction(a.output, scores['basin_anti_gradient'], valid, a.data / 'sample_submission.tif', a.budget)
        print(json.dumps({'experimental_only': True, 'emitted_pixels': n, **validate(a.output, a.data / 'sample_submission.tif')}, indent=2))

if __name__ == '__main__': main()
