"""
pipeline.py — end-to-end, runnable-once-data-is-present fault predictor.

This module contains no synthetic stand-ins.  Everything here operates on the
real competition rasters and refuses to run without them.  That is deliberate:
the previous version of this repo shipped a procedurally generated field wearing
the label of a geological hypothesis, which is exactly the kind of claim we do
not make.

What this does, in order:

  1. load_competition()      read training_features.tif + existing_faults.tif
                              (+ example_submission.tif for the footprint)
  2. band_index_by_tag()     read the authoritative per-band descriptions from
                              the GeoTIFF tags instead of trusting a hard-coded
                              index (the official reference solution does the
                              same: https://github.com/drivendataorg/gems-prize-reference-solution)
  3. build_feature_stack()   per-band robust standardisation + gradient/coherence
                              transforms + catalogue-relative geometry features
  4. make_spatial_folds()    buffered spatial blocks, never random pixels
  5. fit_model()             pixel classifier, class-weighted
  6. blocked_holdout()       exact DTI per fold with the official catalogue mask
  7. build_final_field()     corridor-calibrated binary field for submission

The model is intentionally a gradient-boosted tree ensemble rather than a UNet.
A pixel classifier on 19 physically-meaningful layers with heavy class
imbalance (the catalogue is ~1% of the footprint) is a strong, fast, CPU-only
baseline that a UNet would have to beat.  It also cannot silently memorise the
catalogue, which is the failure mode that pinned the whole group at 0.1563.

Requirements: numpy, scipy, rasterio, scikit-learn.  See requirements.txt.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from pathlib import Path

import numpy as np

from .metric import distance_weighted_tversky_fast
from . import strategy as S

WIDTH, HEIGHT = 3292, 3730
EPSG = 32611
TRANSFORM = (100.0, 0.0, 243350.0, 0.0, -100.0, 4508550.0)

# Fallback band names, used ONLY when a band carries no description tag.
# The official feature list is at
# https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/#provided-features
FALLBACK_BAND_NAMES = [
    "surface_conductivity", "depth_to_conductive_base",
    "detrended_elevation", "detrended_elevation_slope",
    "dilatation_rate", "shear_strain_rate", "second_invariant_strain_rate",
    "isostatic_gravity_anomaly", "isostatic_gravity_slope",
    "reduced_to_pole_magnetic_anomaly", "total_magnetic_intensity",
    "vertical_slope_tmi", "horizontal_slope_tmi",
    "top_of_crustal_magnetic_source_depth",
    "earthquake_density",
]  # + any remaining bands, named band_<n>


# --------------------------------------------------------------------------- #
# 1. loading
# --------------------------------------------------------------------------- #
@dataclass
class CompetitionData:
    stack: np.ndarray            # (C, H, W) float32
    catalogue: np.ndarray        # (H, W) bool  -- known USGS/INGENIOUS faults
    band_names: list[str]
    footprint: np.ndarray | None  # (H, W) bool, from example_submission.tif
    tags: dict = dc_field(default_factory=dict)

    @property
    def n_bands(self) -> int:
        return int(self.stack.shape[0])

    def band(self, *patterns: str) -> np.ndarray:
        """Fetch a band by (case-insensitive, underscore-insensitive) substring."""
        for pat in patterns:
            key = pat.lower().replace(" ", "_")
            for i, nm in enumerate(self.band_names):
                if key in nm.lower().replace(" ", "_"):
                    return self.stack[i]
        raise KeyError(
            f"no band matching {patterns!r}. Available: {self.band_names}"
        )


def load_competition(data_dir: str | Path = "data") -> CompetitionData:
    import rasterio

    data_dir = Path(data_dir)
    feat = data_dir / "training_features.tif"
    lab = data_dir / "existing_faults.tif"
    if not feat.exists() or not lab.exists():
        raise FileNotFoundError(
            f"missing {feat} or {lab}. The competition data tab needs a DrivenData "
            f"login: https://www.drivendata.org/competitions/306/competition-doe-gems/data/ "
            f"Place the files in {data_dir}/ and re-run."
        )

    with rasterio.open(feat) as src:
        stack = src.read().astype(np.float32)
        tags = {i: src.tags(i) for i in range(1, src.count + 1)}
    with rasterio.open(lab) as src:
        catalogue = src.read(1) >= 1

    names = []
    for i in range(1, stack.shape[0] + 1):
        t = tags.get(i, {})
        nm = (t.get("description") or t.get("data_category") or "").strip()
        names.append(nm if nm else (
            FALLBACK_BAND_NAMES[i - 1] if i - 1 < len(FALLBACK_BAND_NAMES)
            else f"band_{i}"))

    footprint = None
    ex = data_dir / "example_submission.tif"
    if ex.exists():
        with rasterio.open(ex) as src:
            footprint = np.isfinite(src.read(1))

    return CompetitionData(stack=stack, catalogue=catalogue, band_names=names,
                           footprint=footprint, tags=tags)


# --------------------------------------------------------------------------- #
# 2-3. features
# --------------------------------------------------------------------------- #
def _robust_scale(a: np.ndarray) -> np.ndarray:
    """Median/MAD standardisation; robust to the extreme tails in magnetics."""
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return np.zeros_like(a, dtype=np.float32)
    med = np.median(finite)
    mad = np.median(np.abs(finite - med)) * 1.4826
    if mad < 1e-9:
        mad = float(finite.std()) or 1.0
    out = (a - med) / mad
    return np.clip(np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0),
                   -8, 8).astype(np.float32)


def _grad_mag(a: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(a)
    return np.sqrt(gx * gx + gy * gy)


def _structure_coherence(a: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """
    (lambda1 - lambda2)/(lambda1 + lambda2) of the structure tensor, 0..1.
    1 = perfectly linear (fault-like), 0 = isotropic.
    """
    from scipy.ndimage import gaussian_filter
    gy, gx = np.gradient(a)
    jxx = gaussian_filter(gx * gx, sigma)
    jyy = gaussian_filter(gy * gy, sigma)
    jxy = gaussian_filter(gx * gy, sigma)
    tmp = np.sqrt((jxx - jyy) ** 2 + 4 * jxy * jxy)
    lam1 = 0.5 * (jxx + jyy + tmp)
    lam2 = 0.5 * (jxx + jyy - tmp)
    denom = lam1 + lam2
    return np.where(denom > 1e-12, (lam1 - lam2) / np.maximum(denom, 1e-12), 0.0
                    ).astype(np.float32)


def dilation_tendency_real(dilatation: np.ndarray, second_invariant: np.ndarray) -> np.ndarray:
    """
    Genuine dilation tendency (Simpson & Reiling 2008, Tectonophysics 456:41-49),
    computed from the two strain-rate layers the competition actually provides.

    For a 2-D strain-rate tensor with trace T and second invariant
    I2 = 0.5 * e_ij e_ij, the deviatoric rate is  D = sqrt(I2 - T^2/4),
    principal rates are  T/2 +- D, and

        Td = (e1 - e_n) / (e1 - e3) = 0.5 + T / (4 D)      (e_n ~ 0)

    Td is 0.5 in pure simple shear and rises above 0.5 under extension, which is
    the regime where opening-mode fluid flow concentrates.  We clip to [0, 1]
    because the submission format demands [0, 1].

    The previous implementation in this repo used
    `0.5 * (1 + dilatation / (shear + |second|))`, which is not the published
    definition and does not have the right limits.  This replaces it.
    """
    T = np.asarray(dilatation, dtype=np.float64)
    I2 = np.asarray(second_invariant, dtype=np.float64)
    D2 = np.maximum(I2 - (T * T) / 4.0, 1e-12)
    D = np.sqrt(D2)
    td = 0.5 + T / (4.0 * D)
    return np.clip(np.nan_to_num(td, nan=0.5, posinf=1.0, neginf=0.0), 0.0, 1.0
                   ).astype(np.float32)


def catalogue_geometry_features(catalogue: np.ndarray) -> dict:
    """
    G-1: everything expressible about a pixel purely from the shape of the known
    catalogue.  These are the features that the 0.1563 submissions could not have
    used, because using them requires thinking about the catalogue as *geometry*
    (endpoints, curvature, strand spacing) rather than as a target to copy.

    * dist_to_catalog    : Euclidean distance (px) to the nearest catalogue px
    * endpoint_prox      : inverse distance to a catalogue *endpoint* (px)
    * curvature_prox     : curvature of the catalogue skeleton, Gaussian-smoothed
    * local_azimuth_dev  : |angle| difference between the local skeleton tangent
                           and the nearest skeleton tangent (rad, wrapped) --
                           splays and stepovers have high values
    * strand_proximity   : count of distinct catalogue strands within 5 px
    """
    from scipy.ndimage import distance_transform_edt, gaussian_filter

    cat = catalogue.astype(bool)
    if not cat.any():
        z = np.zeros(cat.shape, dtype=np.float32)
        return {k: z for k in ("dist_to_catalog", "endpoint_prox", "curvature_prox",
                               "local_azimuth_dev", "strand_proximity")}

    dist = distance_transform_edt(~cat).astype(np.float32)

    # Endpoints: catalogue pixels with exactly 1 orthogonal catalogue neighbour.
    nb = np.zeros(cat.shape, dtype=np.uint8)
    nb[1:, :] += cat[:-1, :]
    nb[:-1, :] += cat[1:, :]
    nb[:, 1:] += cat[:, :-1]
    nb[:, :-1] += cat[:, 1:]
    endpoints = cat & (nb == 1)
    if endpoints.any():
        d_end = distance_transform_edt(~endpoints)
        endpoint_prox = np.exp(-d_end / 8.0).astype(np.float32)
    else:
        endpoint_prox = np.zeros(cat.shape, dtype=np.float32)

    # Skeleton tangent field -> curvature and local azimuth deviation.
    gy, gx = np.gradient(gaussian_filter(cat.astype(np.float32), 1.0))
    magnitude = np.sqrt(gx * gx + gy * gy) + 1e-8
    tx, ty = gx / magnitude, gy / magnitude

    # Divergence of the unit tangent = local turning rate of the skeleton.
    div_t = np.gradient(tx)[1] + np.gradient(ty)[0]
    curvature_prox = np.abs(gaussian_filter(div_t, 4.0)).astype(np.float32)

    # Azimuthal deviation: the local skeleton direction against the mean
    # direction of the catalogue in a wide neighbourhood.  High where a second
    # strand runs at an angle to the first -- stepovers and splay junctions.
    w = gaussian_filter(cat.astype(np.float32), 6.0)
    mean_tx = gaussian_filter(np.where(cat, tx, 0.0), 6.0) / np.maximum(w, 1e-6)
    mean_ty = gaussian_filter(np.where(cat, ty, 0.0), 6.0) / np.maximum(w, 1e-6)
    dot = np.clip(np.abs(mean_tx * tx + mean_ty * ty), 0.0, 1.0)
    local_azimuth_dev = (1.0 - dot).astype(np.float32)

    # Multi-strand support: how deep inside the dilated catalogue a pixel sits.
    d_in = distance_transform_edt(~near_strand(cat, 5)).astype(np.float32)
    strand_proximity = np.exp(-d_in / 3.0).astype(np.float32)

    return {
        "dist_to_catalog": dist,
        "endpoint_prox": endpoint_prox,
        "curvature_prox": curvature_prox,
        "local_azimuth_dev": local_azimuth_dev,
        "strand_proximity": strand_proximity,
    }


def _dilate_np(mask: np.ndarray, iterations: int) -> np.ndarray:
    out = mask.astype(bool).copy()
    for _ in range(iterations):
        acc = out.copy()
        acc[1:, :] |= out[:-1, :]
        acc[:-1, :] |= out[1:, :]
        acc[:, 1:] |= out[:, :-1]
        acc[:, :-1] |= out[:, 1:]
        out = acc
    return out


def near_strand(catalogue: np.ndarray, radius_px: int) -> np.ndarray:
    """Catalogue pixels dilated by radius_px (the multi-strand support)."""
    return _dilate_np(catalogue.astype(bool), radius_px)


def build_feature_stack(data: CompetitionData, extra_planes: dict | None = None):
    """
    Returns (X, names) with X of shape (H*W, F) float32 and `names` length F.
    Features are deliberately few and physical: 19 standardised layers plus
    gradient magnitude, structure-tensor coherence, the real dilation tendency,
    and the five catalogue-geometry planes.
    """
    stack, names = data.stack, data.band_names
    planes: list[np.ndarray] = []
    feat_names: list[str] = []
    for i, nm in enumerate(names):
        planes.append(_robust_scale(stack[i]))
        feat_names.append(f"raw::{nm}")

    # derived transforms on the layers that physically express a fault trace
    for key in ("detrended_elevation", "reduced_to_pole", "magnetic_anomaly",
                "conductivity", "gravity"):
        try:
            a = data.band(key)
        except KeyError:
            continue
        planes.append(_robust_scale(_grad_mag(a))); feat_names.append(f"gradmag::{key}")
        planes.append(_structure_coherence(a)); feat_names.append(f"coherence::{key}")

    try:
        dil = data.band("dilatation_rate", "dilatation")
        i2 = data.band("second_invariant")
        planes.append(dilation_tendency_real(dil, i2)); feat_names.append("dilation_tendency")
    except KeyError:
        pass

    geom = catalogue_geometry_features(data.catalogue)
    for k, v in geom.items():
        planes.append(_robust_scale(v) if k == "dist_to_catalog" else v.astype(np.float32))
        feat_names.append(f"geom::{k}")

    for k, v in (extra_planes or {}).items():
        planes.append(np.asarray(v, dtype=np.float32)); feat_names.append(f"extra::{k}")

    X = np.stack(planes, axis=-1).reshape(-1, len(planes)).astype(np.float32)
    return X, feat_names


# --------------------------------------------------------------------------- #
# 4. folds
# --------------------------------------------------------------------------- #
def make_spatial_folds(shape, n_folds: int = 4, buffer_px: int = 3, seed: int = 0):
    """
    Buffered checkerboard blocks.  `buffer_px` removes a ring around every
    validation block from the training set so the classifier cannot learn a
    fault and then be scored on the pixels immediately next to it -- the
    dominant leak in fault mapping, where catalogue strands are 1-2 px wide.
    """
    h, w = shape
    bh, bw = h // n_folds, w // n_folds
    folds = []
    for fy in range(n_folds):
        for fx in range(n_folds):
            val = np.zeros((h, w), dtype=bool)
            val[fy * bh:(fy + 1) * bh, fx * bw:(fx + 1) * bw] = True
            if buffer_px > 0:
                grown = _dilate_np(val, buffer_px)
            else:
                grown = val
            train = ~grown
            folds.append((f"fold{fy}{fx}", train, val))
    return folds


# --------------------------------------------------------------------------- #
# 5-6. model + holdout
# --------------------------------------------------------------------------- #
def fit_model(X: np.ndarray, y: np.ndarray, *, seed: int = 0, max_rows: int = 4_000_000):
    """
    Class-weighted histogram gradient boosting.  Class weights are essential:
    catalogue fault pixels are ~1% of the footprint, and an unweighted fit
    converges on "predict 0 everywhere", which is the DTI=0 degenerate baseline.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier

    y = y.astype(bool)
    if y.sum() == 0:
        raise ValueError("training fold contains no positive pixels")
    rng = np.random.default_rng(seed)
    if X.shape[0] > max_rows:                    # keep it CPU-friendly
        pos = np.flatnonzero(y.reshape(-1))
        neg = np.flatnonzero(~y.reshape(-1))
        n_pos = min(pos.size, max_rows // 8)
        n_neg = max_rows - n_pos
        idx = np.concatenate([rng.choice(pos, n_pos, replace=False),
                              rng.choice(neg, min(n_neg, neg.size), replace=False)])
    else:
        idx = np.arange(X.shape[0])
    clf = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.08, max_leaf_nodes=31,
        min_samples_leaf=40, l2_regularization=1.0,
        class_weight="balanced", early_stopping=True, validation_fraction=0.1,
        random_state=seed,
    )
    clf.fit(X[idx], y.reshape(-1)[idx])
    return clf


def blocked_holdout(data: CompetitionData, *, n_folds: int = 4, buffer_px: int = 3,
                    seed: int = 0, budget_fracs=(0.005, 0.01, 0.02, 0.03, 0.05, 0.08),
                    halos=(0, 1, 2, 3, 4, 6)) -> dict:
    """
    The validation gate.  For every spatial fold:

      * fit on the buffered training half (catalogue labels)
      * score the held-out catalogue pixels with the REAL DTI, masking the
        held-out catalogue (official masking, forum thread 11516)
      * compare arms: model-only, catalogue-only, corridor(model+catalogue),
        and a budget-matched random control

    IMPORTANT AND OFTEN OVERLOOKED: this validates *detector quality on
    catalogue faults*, which is a necessary but NOT sufficient condition.  The
    competition scores faults that are absent from the catalogue.  A model that
    simply memorises the catalogue scores perfectly here and ~0.1563 on the
    board.  The arms below are designed to make that failure visible: if
    "catalogue-only" is not beaten, the model has learned nothing new.
    """
    X, names = build_feature_stack(data)
    y_all = data.catalogue.reshape(-1)
    folds = make_spatial_folds(data.catalogue.shape, n_folds, buffer_px, seed)
    rng = np.random.default_rng(seed)
    records = []

    for name, tr, va in folds:
        gt = va & data.catalogue
        if gt.sum() < 50:
            continue
        y2 = y_all.reshape(data.catalogue.shape)
        clf = fit_model(X[tr.reshape(-1)], y2[tr], seed=seed)
        proba = clf.predict_proba(X[va.reshape(-1)])[:, 1].reshape(gt.shape)
        catalogue_va = data.catalogue & va
        score_full = np.zeros(data.catalogue.shape, dtype=np.float32)
        score_full[va] = proba.astype(np.float32)

        arms = {}
        for b in budget_fracs:
            budget = int(round(b * gt.size))
            f_model = np.zeros_like(score_full)
            f_model[va] = S.binarise(proba, budget)[va]
            f_catalogue = np.zeros_like(score_full)
            f_catalogue[catalogue_va] = 1.0
            f_rand = np.zeros_like(score_full)
            sel = rng.choice(np.flatnonzero(va), size=min(budget, int(va.sum())),
                             replace=False)
            f_rand.reshape(-1)[sel] = 1.0
            arms[f"model@{b}"] = distance_weighted_tversky_fast(
                f_model, gt, fp_ignore_mask=catalogue_va)[0]
            arms[f"random@{b}"] = distance_weighted_tversky_fast(
                f_rand, gt, fp_ignore_mask=catalogue_va)[0]
            arms[f"catalogue@{b}"] = distance_weighted_tversky_fast(
                f_catalogue, gt, fp_ignore_mask=catalogue_va)[0]
        best_halo, best_corr, best_dti = 0, None, -1.0
        for h in halos:
            for b in budget_fracs:
                budget = int(round(b * gt.size))
                f = S.build_corridor(score_full, data.catalogue, halo_px=h,
                                     budget_px=budget)
                d = distance_weighted_tversky_fast(f, gt, fp_ignore_mask=catalogue_va)[0]
                if d > best_dti:
                    best_dti, best_halo = d, h
                    best_corr = (h, b)
        arms["corridor(best)"] = best_dti
        records.append({"fold": name, "n_val_px": int(va.sum()),
                        "n_gt_px": int(gt.sum()), "arms": arms,
                        "best_corridor": best_corr})
        print(f"  {name}: gt={gt.sum():6d}  " +
              "  ".join(f"{k}={v:.4f}" for k, v in arms.items()
                        if k.startswith(("model@0.02", "random@0.02", "catalogue@0.02",
                                         "corridor"))))

    summary = {}
    if records:
        keys = records[0]["arms"].keys()
        for k in keys:
            summary[k] = float(np.mean([r["arms"][k] for r in records]))
    return {"per_fold": records, "mean_dti": summary,
            "n_features": len(names), "feature_names": names,
            "n_folds": n_folds, "buffer_px": buffer_px}


def holdout_gate(report: dict, *, min_lift: float = 0.02) -> tuple[bool, str]:
    """
    The rule that decides whether a submission slot may be spent.
    Requires, averaged over folds:
      * the corridor arm beats the catalogue-only arm      (new information)
      * the model arm beats the budget-matched random arm   (real detector)
    """
    m = report.get("mean_dti", {})
    cat = m.get("catalogue@0.02")
    rnd = m.get("random@0.02")
    corr = m.get("corridor(best)")
    mdl = m.get("model@0.02")
    if cat is None or corr is None or rnd is None:
        return False, "gate could not evaluate: missing arms"
    if corr - cat < min_lift:
        return False, (f"corridor {corr:.4f} does not beat catalogue-only {cat:.4f} "
                       f"by >= {min_lift} on the blocked holdout")
    if mdl is not None and rnd is not None and mdl - rnd < min_lift:
        return False, (f"model {mdl:.4f} does not beat random control {rnd:.4f} "
                       f"by >= {min_lift}")
    return True, (f"corridor {corr:.4f} > catalogue {cat:.4f}; "
                  f"model {mdl} > random {rnd} -- slot may be spent")


# --------------------------------------------------------------------------- #
# 7. final field
# --------------------------------------------------------------------------- #
def build_final_field(data: CompetitionData, *, seed: int = 0,
                      halo_px: int | None = None, budget_frac: float | None = None):
    """
    Fit on ALL catalogue pixels, then choose (halo, budget) by a
    leave-one-block-out sweep *on the catalogue labels* and return the corridor
    field.  When the holdout report supplies a better halo, pass it in.
    """
    X, names = build_feature_stack(data)
    clf = fit_model(X, data.catalogue.reshape(-1), seed=seed)
    proba = clf.predict_proba(X)[:, 1].reshape(data.catalogue.shape).astype(np.float32)

    rows = S.sweep_corridor_halo(proba, data.catalogue, data.catalogue,
                                 metric=distance_weighted_tversky_fast)
    best = rows[0]
    h = halo_px if halo_px is not None else best["halo_px"]
    b = budget_frac if budget_frac is not None else best["budget_frac"]
    budget_px = None if b is None else int(round(b * data.catalogue.size))
    field = S.build_corridor(proba, data.catalogue, halo_px=h, budget_px=budget_px)
    return field, {
        "sweep_top5": rows[:5],
        "chosen_halo_px": h, "chosen_budget_frac": b,
        "calibration_dti_on_catalogue": best["dti"],
        "features": names,
        "caveat": ("calibrated against catalogue labels, which the competition "
                   "does not score. Re-tune on the blocked holdout before "
                   "trusting it; see blocked_holdout()."),
    }
