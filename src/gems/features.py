"""
New feature engineering for GEMSDOE9 — 5 hypotheses H9-1..H9-5

These are derived features not present in previous repos, or implemented differently.

All functions operate on numpy arrays HxW.

Band inventory — verification status (no hallucinations policy):

VERIFIED (official problem description, feature bullet list, checked 2026-09-27):
https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/#provided-features
  The page lists these layer groups for training_features.tif (100 m, EPSG:32611):
  - Surface conductivity and depth to conductive base surface
  - Detrended elevation and the slope of detrended elevation
  - Dilatation rate, shear strain rate, second invariant of the strain rate tensor
  - Isostatic gravity anomaly and the slope of the isostatic gravity anomaly
  - Magnetics: reduced-to-pole magnetic anomaly, total magnetic intensity,
    vertical and horizontal slope of total magnetic intensity,
    top-of-crustal magnetic source depth estimate
  - Density of earthquakes
  (Radiometric layers are implied by the page's figure caption "total radiometric
  counts per second" but are not in the bullet list.)

VERIFIED (GEMSDOE1 measurement on the actual file): 19 bands total.

UNVERIFIED until data/ is placed: the exact per-index band order inside the file.
  The authoritative order lives in the GeoTIFF band tags; the official reference
  solution reads them via src.tags(i)['description'] / ['data_category']
  (https://github.com/drivendataorg/gems-prize-reference-solution).
  scripts/prepare_data.py dumps these tags when the file is present.

PROVISIONAL INDEXING used below (assumes the file follows the page's bullet
order, plus 4 radiometric bands K/Th/U/total-count at the end):
0: surface conductivity                10: total magnetic intensity
1: depth to conductive base            11: vertical slope of TMI
2: detrended elevation                 12: horizontal slope of TMI
3: slope of detrended elevation        13: top-of-crustal magnetic source depth
4: dilatation rate                     14: density of earthquakes
5: shear strain rate                   15-18: radiometric (order unverified)
6: second invariant of strain rate
7: isostatic gravity anomaly
8: slope of isostatic gravity anomaly
9: reduced-to-pole magnetic anomaly

Every consumer of band indices must call assert_band_count() first and the
provisional mapping must be confirmed against the band-tag dump before any
scored submission is built from these features.
"""

import numpy as np
from scipy.ndimage import gaussian_filter, sobel, generic_filter
from scipy.signal import convolve2d

# Provisional band indices (see module docstring — unverified until band tags read)
BAND_SURFACE_CONDUCTIVITY = 0
BAND_DEPTH_CONDUCTIVE_BASE = 1
BAND_DETRENDED_ELEVATION = 2
BAND_DETRENDED_SLOPE = 3
BAND_DILATATION_RATE = 4
BAND_SHEAR_STRAIN_RATE = 5
BAND_SECOND_INVARIANT = 6
BAND_ISOSTATIC_GRAVITY = 7
BAND_ISOSTATIC_GRAVITY_SLOPE = 8
EXPECTED_BAND_COUNT = 19  # verified by GEMSDOE1 measurement of the actual file


def assert_band_count(feature_stack):
    """Hard gate: refuse to build features from a stack whose band count we
    have not verified. 19 bands was measured on the real file in GEMSDOE1.
    Accepts shapes (19, H, W) or (H, W, 19)."""
    if feature_stack.ndim != 3:
        raise ValueError("feature stack must be a 3-D array")
    if EXPECTED_BAND_COUNT not in (feature_stack.shape[0], feature_stack.shape[-1]):
        raise ValueError(
            f"feature stack shape {feature_stack.shape} has no axis of size "
            f"{EXPECTED_BAND_COUNT}; run scripts/prepare_data.py to dump and "
            "verify band tags first")

def dilation_tendency(dilatation, shear, second_inv):
    """
    H9-1: Compute dilation tendency Td from strain rate tensor.
    In extensional regime, Td = (sigma1 - sigma_n)/(sigma1 - sigma3)
    For strain rates, we approximate using dilatation (trace) and shear.
    High dilatation + low shear = high tendency for opening.
    """
    # Normalize
    eps = 1e-6
    # Dilatation positive = extension
    # Second invariant high = high strain
    # Td proxy = (dilatation - min) / (second_inv + eps) clipped
    # More physically: Td = 0.5*(1 + dilatation / (shear + eps))
    Td = 0.5 * (1 + dilatation / (shear + np.abs(second_inv) + eps))
    Td = np.clip(Td, 0, 1)
    return Td.astype(np.float32)

def intersection_density(lineament_orientation, window=20):
    """
    H9-1: Hough line intersections per km².
    lineament_orientation: angle in radians per pixel (from structure tensor)
    We compute density of orientation changes — proxy for intersections.
    Simplified: use gradient of orientation.
    """
    # Orientation gradient magnitude
    gy, gx = np.gradient(lineament_orientation)
    grad_mag = np.sqrt(gx**2 + gy**2)
    # Density via mean filter
    from scipy.ndimage import uniform_filter
    density = uniform_filter(grad_mag, size=window)
    return density.astype(np.float32)

def conductivity_edge(surface_cond, depth_cond):
    """
    H9-2: Laplacian of conductivity + vertical gradient.
    Clay cap edge detection.
    """
    # Vertical gradient proxy: cond / depth
    vert_grad = surface_cond / (depth_cond + 1)
    # Horizontal gradient magnitude
    gy, gx = np.gradient(surface_cond)
    h_grad = np.sqrt(gx**2 + gy**2)
    # Laplacian via second derivative
    lap = np.gradient(gx)[0] + np.gradient(gy)[1]  # approx
    # Edge = |lap| * h_grad * vert_grad
    edge = np.abs(lap) * h_grad * np.clip(vert_grad, 0, 10)
    return edge.astype(np.float32), vert_grad.astype(np.float32), h_grad.astype(np.float32)

def fluvial_sl_ksn(dem, window=5):
    """
    H9-3: Stream Length-Gradient Index SL and normalized channel steepness ksn.
    dem: 10 m DEM resampled to 100 m.
    Simplified implementation: SL = (delta_h / delta_l) * L where L is distance from source.
    We approximate using slope * flow accumulation proxy (inverse of curvature).
    """
    # Slope
    gy, gx = np.gradient(dem)
    slope = np.sqrt(gx**2 + gy**2)
    # Curvature (second derivative)
    gyy, gyx = np.gradient(gy)
    gxy, gxx = np.gradient(gx)
    curvature = gxx + gyy
    # SL proxy: slope * distance — use distance transform from ridges
    # For simplicity, SL anomaly = slope * (1 + |curvature|)
    sl_proxy = slope * (1 + np.abs(curvature))
    # ksn proxy: slope / (drainage_area^theta) — drainage area proxy via 1/(curvature+eps) where convergent
    drainage_proxy = 1.0 / (np.abs(curvature) + 0.1)
    ksn_proxy = slope / (drainage_proxy ** 0.45)
    # Drainage deflection: angle change
    flow_dir = np.arctan2(gy, gx)
    def_angle = np.gradient(flow_dir)[0]  # change in direction
    return sl_proxy.astype(np.float32), ksn_proxy.astype(np.float32), def_angle.astype(np.float32)

def thermal_alteration_proxy(thermal_band, aster_clay):
    """
    H9-4: Thermal anomaly + clay alteration.
    thermal_band: Landsat TIRS Band 10 (100 m)
    aster_clay: ASTER clay index
    Returns combined score.
    """
    # Topo correction: subtract median filter (background)
    from scipy.ndimage import median_filter
    background = median_filter(thermal_band, size=20)
    anomaly = thermal_band - background
    # Z-score
    mean = np.mean(anomaly)
    std = np.std(anomaly) + 1e-6
    z = (anomaly - mean) / std
    thermal_score = np.clip(z, 0, 5) / 5.0
    # Clay score
    clay_score = np.clip(aster_clay, 0, 1)
    combined = thermal_score * 0.6 + clay_score * 0.4
    return combined.astype(np.float32), thermal_score.astype(np.float32), clay_score.astype(np.float32)

def euler_depth_contact(mag_anomaly, grav_anomaly, window=10):
    """
    H9-5: Euler deconvolution depth + gravity-magnetic coincidence.
    Simplified Euler: depth ~ (anomaly / gradient)
    """
    gy_mag, gx_mag = np.gradient(mag_anomaly)
    grad_mag = np.sqrt(gx_mag**2 + gy_mag**2) + 1e-6
    # Euler depth proxy
    euler_depth = np.abs(mag_anomaly) / grad_mag
    euler_depth = np.clip(euler_depth, 0, 5000)  # meters
    # Shallow if <1000 m
    shallow = (euler_depth < 1000).astype(np.float32)
    # Gravity HGM
    gy_grav, gx_grav = np.gradient(grav_anomaly)
    grav_hgm = np.sqrt(gx_grav**2 + gy_grav**2)
    # Coincidence: both high HGM
    mag_hgm = grad_mag
    coincidence = (mag_hgm > np.percentile(mag_hgm, 75)) & (grav_hgm > np.percentile(grav_hgm, 75))
    coincidence = coincidence.astype(np.float32)
    return euler_depth.astype(np.float32), shallow, grav_hgm.astype(np.float32), coincidence

def build_h91_features(feature_stack):
    """
    Build H9-1 feature set from 19-band stack.
    feature_stack: (19, H, W) or (H, W, 19)
    Returns dict of new features.

    NOTE: band indexing is PROVISIONAL (see module docstring) until the
    GeoTIFF band tags have been dumped by scripts/prepare_data.py.
    """
    assert_band_count(feature_stack)
    if feature_stack.shape[0] == EXPECTED_BAND_COUNT:
        # (19, H, W)
        fs = feature_stack
    else:
        fs = np.moveaxis(feature_stack, -1, 0)

    dilatation = fs[BAND_DILATATION_RATE]
    shear = fs[BAND_SHEAR_STRAIN_RATE]
    second = fs[BAND_SECOND_INVARIANT]
    detrended_slope = fs[BAND_DETRENDED_SLOPE]
    grav_slope = fs[BAND_ISOSTATIC_GRAVITY_SLOPE]

    Td = dilation_tendency(dilatation, shear, second)
    # Fake orientation from detrended slope gradient for now — in real pipeline compute structure tensor
    orient = np.arctan2(*np.gradient(detrended_slope))
    inter_density = intersection_density(orient)

    # Combine
    return {
        "dilation_tendency": Td,
        "intersection_density": inter_density,
        "detrended_slope": detrended_slope,
        "grav_slope": grav_slope,
    }
