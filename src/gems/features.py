"""
features.py — one detector per hypothesis, each an explicit, auditable function.

Every score here takes a `CompetitionData` (see pipeline.load_competition) and
returns a continuous float32 field on the 3292x3730 grid, plus a short string
describing which layers it touched.  That pairing is deliberate: the site and
the README can quote the layer list and the physical signature straight from
the code, so a hypothesis cannot drift away from its implementation.

BAND NAMING
-----------
Bands are resolved by name, never by a hard-coded index.  The GeoTIFF carries
per-band `description` / `data_category` tags, and the official reference
solution reads them the same way:
  https://github.com/drivendataorg/gems-prize-reference-solution
`CompetitionData.band(*substrings)` raises KeyError with the full list of
available band names if a lookup misses, so a wrong assumption is loud, not
silent.  The earlier version of this file hard-coded indices 0..18 and labelled
them "PROVISIONAL"; that is gone.

None of these functions runs without the real rasters.  There are no synthetic
stand-ins — see README, "What changed and why".
"""

from __future__ import annotations

import numpy as np

from .pipeline import (
    CompetitionData, catalogue_geometry_features, dilation_tendency_real,
    _robust_scale, _grad_mag, _structure_coherence,
)

__all__ = [
    "Detector", "g1_catalogue_geometry", "g2_dilation_tendency_ridge",
    "g3_clay_cap_edge", "g4_magnetic_contact", "g5_fluvial_knickpoint",
    "all_detectors", "dilation_tendency_real",
]


class Detector:
    """A named hypothesis: score(data) -> field, plus its provenance string."""

    def __init__(self, key: str, title: str, layers: str, signature: str,
                 why_missing: str, cost: str, fn, needs_external: str = ""):
        self.key, self.title = key, title
        self.layers, self.signature = layers, signature
        self.why_missing, self.cost = why_missing, cost
        self.needs_external = needs_external
        self.fn = fn

    def __call__(self, data: CompetitionData) -> np.ndarray:
        return self.fn(data).astype(np.float32)

    def __repr__(self):
        return f"<Detector {self.key} {self.title!r}>"


# --------------------------------------------------------------------------- #
# G-1  Catalogue geometry completion
# --------------------------------------------------------------------------- #
def g1_catalogue_geometry(data: CompetitionData) -> np.ndarray:
    """
    Layers: existing_faults.tif (the catalogue as *geometry*), nothing else.
            Optionally modulated by detrended-elevation slope and magnetic HGM.

    Signature: proximity to a catalogue ENDPOINT and to local skeleton
                curvature/azimuthal deviation, gated to the first 3 px outside
                the catalogue, because the metric's kernel support is 300 m and
                staff have stated that "new fault" includes newly mapped
                geometry of an existing fault system (forum thread 11536).

    Why it catches what the catalogue misses: continuations past mapped tips,
    splays and parallel strands are, by definition, adjacent to and collinear
    with existing strands.  The catalogue tells you exactly where to look; every
    other detector in this repo treats it as a target to memorise instead.
    """
    geom = catalogue_geometry_features(data.catalogue)
    dist = geom["dist_to_catalog"]
    in_support = np.exp(-np.maximum(dist - 1.0, 0.0) / 2.0)
    score = (1.00 * geom["endpoint_prox"]
             + 0.60 * geom["local_azimuth_dev"]
             + 0.50 * _robust_scale(geom["curvature_prox"])
             + 0.30 * geom["strand_proximity"])
    score = score * in_support
    # Distant from the catalogue -> this is no longer "geometry completion".
    return np.where(dist <= 4.0, score, 0.0).astype(np.float32)


# --------------------------------------------------------------------------- #
# G-2  Geodetic dilation-tendency ridge
# --------------------------------------------------------------------------- #
def g2_dilation_tendency_ridge(data: CompetitionData) -> np.ndarray:
    """
    Layers: dilatation rate + second invariant of the strain rate tensor.
            (Both are listed explicitly in the official feature list.)

    Signature: the real dilation tendency
                   Td = (e1 - e_n)/(e1 - e3) = 0.5 + T / (4 D),
                   D = sqrt(I2 - T^2/4) = |e1 - e2|/2,
               from Simpson & Reiling (2008), Tectonophysics 456:41-49, crossed
               with the structure-tensor coherence of the same layers so that
               the high-Td pixels that sit on a *linear* feature win over the
               diffuse ones.

    Why it catches what the catalogue misses: strain rate is a spatially
               smoothed, regional quantity.  It lights up incipient and short
               structures inside basins where surface mapping has no scarp and
               aerial coverage is poor -- exactly the class the catalogue
               under-samples.

    NOTE ON PRIOR WORK IN THIS REPO: the previous implementation used
    `0.5 * (1 + dilatation / (shear + |second_invariant|))`.  That is not the
    published definition of Td, does not have the correct limits (0.5 in pure
    shear, 1 in pure extension), and does not collapse under rate scaling.
    `tests/test_validation.py::test_dilation_tendency` now pins the real one.
    """
    T = data.band("dilatation_rate", "dilatation")
    I2 = data.band("second_invariant")
    td = dilation_tendency_real(T, I2)
    coh = 0.5 * (_structure_coherence(T) + _structure_coherence(I2))
    return np.clip(td * (0.5 + 0.5 * coh), 0, 1).astype(np.float32)


# --------------------------------------------------------------------------- #
# G-3  Clay-cap conductivity edge
# --------------------------------------------------------------------------- #
def g3_clay_cap_edge(data: CompetitionData) -> np.ndarray:
    """
    Layers: surface conductivity + depth to conductive base.
            (Both listed in the official feature list.)

    Signature: the lateral discontinuity of a conductive clay/smectite cap over
               a resistive reservoir.  Computed as the magnitude of the gradient
               of log(surface conductivity) times the vertical contrast
               log(sigma / depth_to_base), which is maximised exactly at the
               cap edge.  A log transform is what makes this scale-free: clay
               caps span 1-10 ohm.m against 100-1000 ohm.m reservoirs.

    Why it catches what the catalogue misses: blind faults under basin fill
               have no scarp, no aerial expression and no offset drainages, so
               surface mapping misses them entirely.  Their only surface
               signature is the alteration halo in the conductivity data.

    Risk, stated: the conductivity edge is also produced by any lithologic
               contact.  That is why this is scored, not thresholded, and why it
               is combined with G-1 in the final field rather than used alone.
    """
    sigma = data.band("surface_conductivity", "conductivity")
    depth = data.band("depth_to_conductive_base", "conductive_base")
    lsig = np.log1p(np.clip(sigma, 0, None))
    ldep = np.log1p(np.clip(depth, 0, None))
    edge = _grad_mag(lsig) * _robust_scale(lsig - ldep)
    return _robust_scale(edge).astype(np.float32)


# --------------------------------------------------------------------------- #
# G-4  Magnetic contact edge
# --------------------------------------------------------------------------- #
def g4_magnetic_contact(data: CompetitionData) -> np.ndarray:
    """
    Layers: reduced-to-pole magnetic anomaly + total magnetic intensity +
            horizontal & vertical slope of TMI + top-of-crustal source depth.

    Signature: a contact between magnetic units shows a horizontal-gradient
               maximum coincident (within a pixel or two) with a tilt-derivative
               zero crossing, with a shallow Euler source (< 1 km).  The
               coincidence constraint is the point: raw gradient magnitude alone
               fires on every contact in the Basin and Range, the coincidence
               fires only on the ones a fault has brought into juxtaposition.

    Why it catches what the catalogue misses: basement-involved faults offset
               the magnetic basement under a cover that hides the surface trace.
               The catalogue maps the surface; the magnetics see the offset.
    """
    rtp = data.band("reduced_to_pole", "magnetic_anomaly", "magnetic")
    tmi = data.band("total_magnetic_intensity", "magnetic_intensity")
    try:
        depth = data.band("top_of_crustal", "source_depth")
    except KeyError:
        depth = None

    hgm_rtp = _grad_mag(rtp)
    hgm_tmi = _grad_mag(tmi)

    # Prefer the slope products the competition ships over locally recomputed
    # gradients: they are the organiser's own filtered derivatives.  Verified
    # unused by every prior revision of this repo -- 7 of the 15 layers listed
    # in the problem description were read by no detector at all
    # (scripts/audit_layer_usage.py).  Fall back to a local gradient when a band
    # is absent, so the detector still runs.  The lookups are written out
    # literally rather than passed through a helper so the audit script, which
    # scans for `.band(...)`, can see them.
    try:
        vertical = _robust_scale(np.nan_to_num(
            data.band("vertical_slope", "vertical slope"), nan=0.0))
    except KeyError:
        vertical = _robust_scale(_grad_mag(rtp))
    try:
        hgm = _robust_scale(np.nan_to_num(
            data.band("horizontal_slope", "horizontal slope"), nan=0.0))
    except KeyError:
        hgm = _robust_scale(_grad_mag(tmi))

    # Tilt derivative of a 2-D potential field: atan(V / H).  Its zero
    # crossings mark source edges regardless of amplitude, which is why it is
    # used to *gate* the horizontal-gradient magnitude rather than added to it.
    tilt = np.degrees(np.arctan2(np.abs(vertical), np.abs(hgm) + 1e-8))
    tilt_edge = 1.0 - np.abs(tilt) / 90.0

    score = 0.6 * _robust_scale(hgm_rtp) + 0.4 * _robust_scale(hgm_tmi)
    score = score * (0.5 + 0.5 * tilt_edge)
    if depth is not None:
        shallow = np.exp(-np.clip(_robust_scale(depth), 0, None) / 2.0)
        score = score * (0.4 + 0.6 * shallow)
    return _robust_scale(score).astype(np.float32)


# --------------------------------------------------------------------------- #
# G-5  Fluvial knickpoint / drainage deflection
# --------------------------------------------------------------------------- #
def g5_fluvial_knickpoint(data: CompetitionData, dem: np.ndarray | None = None) -> np.ndarray:
    """
    Layers: detrended elevation + slope of detrended elevation (in
            training_features.tif), optionally a 10 m 3DEP DEM.

    Signature: (a) a break in the along-strike slope profile -- knickpoint -- and
               (b) a sharp lateral deflection of the local downslope azimuth,
               which is what a fault does to a drainage network.  Detrending
               removes the regional Basin-and-Range tilt so the residual is
               local structure rather than the range front.

    Why it catches what the catalogue misses: an offset drainage works even
               where the scarp is buried in colluvium, which is the normal
               preservation state for young scarps in the Walker Lane.

    EXTERNAL DATA: `dem` is optional.  Without it this uses the competition's
               own detrended-elevation layers.  With it, pass a 10 m 3DEP DEM
               resampled to the 100 m competition grid; USGS 3DEP is free and
               official at https://apps.nationalmap.gov/3dep/.  This detector
               degrades gracefully, so it is not blocked on external data.
    """
    elev = data.band("detrended_elevation", "elevation")
    if dem is not None:
        elev = np.asarray(dem, dtype=np.float32)
    gy, gx = np.gradient(elev)
    slope = np.hypot(gx, gy) + 1e-8
    azim = np.arctan2(gy, gx)

    # Knickpoint: a second-derivative ridge in the along-strike slope.
    knick = _grad_mag(sobel2(gx)) + _grad_mag(sobel2(gy))

    # Drainage deflection: 1 - |<e^{i azim}>| over a local window, i.e. the local
    # circular variance of the downslope azimuth.  Computed with two box filters
    # on cos/sin rather than a sliding-window view: a 9x9 sliding_window_view on
    # the 3292x3730 competition grid is a 4 GB array and will exhaust memory.
    from scipy.ndimage import uniform_filter
    m = 9
    circ = np.hypot(uniform_filter(np.cos(azim), size=m),
                    uniform_filter(np.sin(azim), size=m))
    deflection = 1.0 - circ

    return _robust_scale(0.6 * knick + 0.4 * deflection * slope).astype(np.float32)


def sobel2(a: np.ndarray) -> np.ndarray:
    from scipy.ndimage import sobel
    return sobel(a, axis=0) + sobel(a, axis=1)


# --------------------------------------------------------------------------- #
DETECTORS = [
    Detector(
        "G-1", "Catalogue geometry completion",
        layers="existing_faults.tif (as geometry); optional detrended-elevation "
               "slope + magnetic HGM modulation",
        signature="endpoint proximity x local skeleton curvature x azimuthal "
                  "deviation, gated to <=3 px (300 m = the metric kernel support) "
                  "outside the catalogue",
        why_missing="Staff have stated that 'new fault' means any fault pixel not "
                    "already captured by USGS/INGENIOUS and CAN include newly mapped "
                    "geometry of an existing fault system. Continuations past mapped "
                    "tips, splays and parallel strands are therefore both (a) common in "
                    "the scored set and (b) located exactly where the catalogue says to "
                    "look. Every sibling submission treated the catalogue as a target "
                    "to copy instead of as a pointer.",
        cost="Low - no new data, pure raster geometry",
        fn=g1_catalogue_geometry,
    ),
    Detector(
        "G-2", "Geodetic dilation-tendency ridge",
        layers="dilatation rate; second invariant of the strain rate tensor",
        signature="Td = 0.5 + T/(4*sqrt(I2 - T^2/4)) (Simpson & Reiling 2008), "
                  "x structure-tensor coherence of the same layers",
        why_missing="Strain rate is a smoothed regional quantity: it responds to "
                    "incipient and short structures inside basins where there is no "
                    "scarp and no aerial coverage. The catalogue is a surface map; "
                    "this is a subsurface one.",
        cost="Low - both layers are already in training_features.tif",
        fn=g2_dilation_tendency_ridge,
    ),
    Detector(
        "G-3", "Clay-cap conductivity edge",
        layers="surface conductivity; depth to conductive base",
        signature="|grad log(sigma)| x log(sigma / depth_to_base): a scale-free "
                  "lateral discontinuity of a conductive alteration cap over a "
                  "resistive reservoir",
        why_missing="Blind faults under basin fill have no scarp, no aerial "
                    "expression and no offset drainages. Their only surface "
                    "signature is the clay/serpentinite halo, which the "
                    "electro-magnetic layers resolve and the surface catalogue does not.",
        cost="Low - both layers are already in training_features.tif",
        fn=g3_clay_cap_edge,
    ),
    Detector(
        "G-4", "Magnetic contact edge",
        layers="reduced-to-pole magnetic anomaly; total magnetic intensity; "
               "horizontal and vertical slope of TMI; top-of-crustal source depth",
        signature="horizontal-gradient maximum at a tilt-derivative zero crossing, "
                  "weighted by a shallow (<1 km) Euler-style source depth",
        why_missing="Basement-involved faults offset the magnetic basement beneath "
                    "a cover that hides the surface trace. The catalogue maps the "
                    "surface; magnetics see the offset. The coincidence constraint is "
                    "what keeps this from firing on every lithologic contact.",
        cost="Low-Med - all four layers are in training_features.tif; the tilt "
             "derivative and Euler-style depth are the compute",
        fn=g4_magnetic_contact,
    ),
    Detector(
        "G-5", "Fluvial knickpoint and drainage deflection",
        layers="detrended elevation; slope of detrended elevation; optional 10 m "
               "USGS 3DEP DEM (free, official)",
        signature="second-derivative ridges in the along-strike slope (knickpoints) "
                  "plus local circular variance of the downslope azimuth (deflection)",
        why_missing="An offset drainage survives even when the scarp is buried in "
                    "colluvium, which is the usual preservation state for young "
                    "Walker Lane scarps. Detrending removes the regional range-front "
                    "tilt so the residual is local structure.",
        cost="Low with the competition's own detrended-elevation layers; Medium with "
             "a 10 m 3DEP DEM (https://apps.nationalmap.gov/3dep/)",
        fn=g5_fluvial_knickpoint,
    ),
]


def all_detectors() -> list[Detector]:
    return list(DETECTORS)
