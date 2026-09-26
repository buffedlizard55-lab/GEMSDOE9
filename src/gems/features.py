"""
New feature engineering for GEMSDOE9 — 5 hypotheses H9-1..H9-5

These are derived features not present in previous repos, or implemented differently.

All functions operate on numpy arrays HxW.

Band mapping (from training_features.tif 19 bands, verified in GEMSDOE1 Data page):
0: surface conductivity
1: depth to conductive base
2: detrended elevation
3: slope of detrended elevation
4: dilatation rate
5: shear strain rate
6: second invariant of strain rate
7: isostatic gravity anomaly
8: slope of isostatic gravity anomaly
9: reduced-to-pole magnetic anomaly
10: total magnetic intensity
11: vertical slope of TMI
12: horizontal slope of TMI
13: top-of-crustal magnetic source depth
14: density of earthquakes
15: total radiometric? (actually band 6 is radiometric total count, not magnetic tilt — bug found in 8GEMSDOE)
16: K radiometric?
17: Th?
18: U? etc — see data.html for full list

We will reference by index but also by name where possible.
"""

import numpy as np
from scipy.ndimage import gaussian_filter, sobel, generic_filter
from scipy.signal import convolve2d

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
    """
    if feature_stack.shape[0] == 19:
        # (19, H, W)
        fs = feature_stack
    else:
        fs = np.moveaxis(feature_stack, -1, 0)

    dilatation = fs[4]
    shear = fs[5]
    second = fs[6]
    detrended_slope = fs[3]
    grav_slope = fs[8]

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
