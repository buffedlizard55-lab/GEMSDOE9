"""
hypotheses.py — candidate detectors H-1 .. H-5, the ones this repo had NOT tried.

WHY THESE FIVE
--------------
Audited on 2026-09-27 against the layer list in the official problem description
(https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/#provided-features)
and against every `.band(...)` call in src/gems/features.py.  Result:

    7 of the 15 listed layers were read by NO detector in this repo:
        slope of detrended elevation
        shear strain rate
        isostatic gravity anomaly
        slope of the isostatic gravity anomaly
        vertical slope of total magnetic intensity
        horizontal slope of total magnetic intensity
        density of earthquakes

    scripts/audit_layer_usage.py reproduces that count.

So the cheapest genuinely-new signal is not a new dataset, it is the data
already handed to us that nobody read.  H-1, H-2 and H-3 are built entirely
from those unused layers.  H-4 and H-5 need free official external data and
say so, loudly, rather than silently degrading.

WHAT "VALIDATED" MEANS HERE
---------------------------
Without the competition rasters no claim about Nevada geology can be made.
`scripts/validate_hypotheses.py` therefore does the only thing that is honest:
it builds a phantom whose *physics* follow each hypothesis's own forward model
(a normal-fault scarp is a step, a blind fault is a conductivity discontinuity,
an active fault has organised seismicity), withholds part of the fault set as
"new" per the staff definition, blocks the phantom spatially, and asks whether
the operator recovers the withheld faults better than what the repo already
had.  That validates the OPERATOR.  It cannot validate the geology, and the
output says so.
"""

from __future__ import annotations

import numpy as np

from .pipeline import CompetitionData, _robust_scale, _grad_mag, _structure_coherence

__all__ = [
    "Hypothesis", "h1_seismogenic_organization", "h2_scarp_step_matched_filter",
    "h3_gravity_lineament", "h4_conductance_depth_persistence",
    "h5_hydrothermal_alignment", "shipped_hypotheses", "external_hypotheses",
    "all_hypotheses", "anisotropic_gaussian",
]


class Hypothesis:
    """A named candidate, with the provenance the site renders verbatim."""

    def __init__(self, key, title, layers, signature, why_missing, differs,
                 cost, expected, fn, external=None, external_source=None):
        self.key, self.title = key, title
        self.layers, self.signature = layers, signature
        self.why_missing, self.differs = why_missing, differs
        self.cost, self.expected = cost, expected
        self.fn = fn
        self.external = external            # None, or what it needs
        self.external_source = external_source  # URL + licence + DOI

    def __call__(self, data: CompetitionData, **kw) -> np.ndarray:
        return self.fn(data, **kw).astype(np.float32)

    def __repr__(self):
        return f"<Hypothesis {self.key} {self.title!r}>"


# --------------------------------------------------------------------------- #
# shared operator
# --------------------------------------------------------------------------- #
def anisotropic_gaussian(a, theta_deg, sigma_along, sigma_across):
    """Gaussian smoothing with its major axis along `theta_deg`.

    Used to integrate evidence ALONG a strike direction while leaving the
    cross-strike profile sharp -- the difference between detecting a 1 km-long
    scarp and detecting a pixel.
    """
    from scipy.ndimage import convolve

    t = np.deg2rad(theta_deg)
    ux, uy = np.cos(t), np.sin(t)          # unit vector along strike
    half = max(3, int(np.ceil(3.0 * max(sigma_along, sigma_across))))
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1]
    # coordinates in the rotated frame
    s_along = xx * ux + yy * uy
    s_across = -xx * uy + yy * ux
    k = np.exp(-0.5 * ((s_along / max(sigma_along, 1e-6)) ** 2
                       + (s_across / max(sigma_across, 1e-6)) ** 2))
    k /= k.sum()
    return convolve(np.nan_to_num(np.asarray(a, dtype=np.float64), nan=0.0), k)


def _minmax(a):
    """Min-max normalise to [0, 1].

    Written as a function rather than `a.ptp()` on purpose: ndarray.ptp() was
    REMOVED in NumPy 2.0, and requirements.txt permits numpy>=1.26, so the
    method form silently kills every detector that uses it on a current
    interpreter.  tests/test_validation.py::K3 pins this.
    """
    a = np.asarray(a, dtype=np.float64)
    lo = float(np.nanmin(a)) if np.isfinite(a).any() else 0.0
    hi = float(np.nanmax(a)) if np.isfinite(a).any() else 1.0
    rng = max(hi - lo, 1e-9)
    return np.clip((np.nan_to_num(a, nan=lo) - lo) / rng, 0.0, 1.0)


def _circular_coherence(field, sigma_along=4.0, sigma_across=1.0,
                        azimuths=(0, 45, 90, 135)):
    """Max over azimuth of |<e^{i 2 phi}>| smoothed along that azimuth.

    A straight linear feature has a gradient direction that is *constant along
    its length*.  Averaging the doubled angle over an elongated window and
    taking the resultant length measures exactly that, which is what separates a
    fault from a blob of high gradient.  Doubled angle because a gradient is a
    direction without a sign.
    """
    gy, gx = np.gradient(np.nan_to_num(np.asarray(field, dtype=np.float64)))
    phi = np.arctan2(gy, gx)
    c, s = np.cos(2 * phi), np.sin(2 * phi)
    best = np.zeros(field.shape, dtype=np.float64)
    for th in azimuths:
        mc = anisotropic_gaussian(c, th, sigma_along, sigma_across)
        ms = anisotropic_gaussian(s, th, sigma_along, sigma_across)
        np.maximum(best, np.hypot(mc, ms), out=best)
    return np.clip(best, 0.0, 1.0)


def _suppress_catalogue(score, catalogue, keep_from_px=1.0, tau=1.5):
    """Down-weight pixels already inside the catalogue.

    The scored class is "any fault pixel not already captured by USGS/INGENIOUS"
    (forum thread 11536, DrivenData staff, 2026-09-23).  A detector that spends
    its dynamic range re-finding the catalogue cannot help.  This is a soft
    suppression, not a hard zero, so a genuine parallel strand one pixel away
    still scores.

    `tau=inf` disables suppression entirely.  It is the configuration axis the
    holdout sweeps; without this branch `1 - exp(-d/inf)` would be identically
    zero and the "no suppression" arm would be an all-zero field rather than an
    unsuppressed one.
    """
    from scipy.ndimage import distance_transform_edt

    if tau is None or not np.isfinite(tau):
        return np.asarray(score, dtype=np.float64)
    d = distance_transform_edt(~np.asarray(catalogue, dtype=bool))
    return score * (1.0 - np.exp(-np.maximum(d - keep_from_px, 0.0) / tau))


# --------------------------------------------------------------------------- #
# H-1  Seismogenic organisation
# --------------------------------------------------------------------------- #
def h1_seismogenic_organization(data: CompetitionData,
                                suppress_tau: float = 1.5) -> np.ndarray:
    """
    Layers: density of earthquakes  +  shear strain rate.
            BOTH were read by no detector in this repo before this file.
            Both are in the official feature list.

    Signature: an active fault is (a) seismogenic and (b) linear.  Elevated
               earthquake density that is *spatially organised* -- high
               structure-tensor coherence of the seismicity field -- is a
               structure, whereas a diffuse high-density patch is background or
               a catalogue artefact.  Crossed with the shear strain rate, which
               localises the same deformation independently.  The catalogue is
               then suppressed, so the score is spent on the unmapped part.

    Why it catches what the catalogue misses: the USGS/INGENIOUS set is a
               geomorphic map -- it maps scarps.  A fault that is currently
               slipping but has no preserved scarp (buried under basin fill, or
               eroded) is invisible to it and visible to seismicity.  This is the
               one shipped layer that measures *present* activity rather than
               past surface expression.

    Differs from everything already implemented: G-1..G-5 use catalogue
               geometry, strain dilatation, conductivity, magnetics and
               elevation.  None uses seismicity or shear strain rate at all.
    """
    eq = data.band("earthquake_density", "earthquake", "seismic")
    shear = data.band("shear_strain_rate", "shear")

    eq_n = _robust_scale(np.log1p(np.clip(np.nan_to_num(eq, nan=0.0), 0, None)))
    eq_n = _minmax(eq_n)

    # Organisation of the seismicity field, integrated along candidate strikes.
    org = _circular_coherence(np.log1p(np.clip(np.nan_to_num(eq, nan=0.0), 0, None)))
    shear_n = _robust_scale(shear)
    shear_n = _minmax(shear_n)

    score = eq_n * (0.35 + 0.65 * org) * (0.5 + 0.5 * shear_n)
    return _suppress_catalogue(_robust_scale(score), data.catalogue,
                               tau=suppress_tau).astype(np.float32)


# --------------------------------------------------------------------------- #
# H-2  Strike-integrated scarp step (cross-profile asymmetry)
# --------------------------------------------------------------------------- #
def h2_scarp_step_matched_filter(data: CompetitionData,
                                 azimuths=(0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5),
                                 offset=2.0, suppress_tau: float = 2.0) -> np.ndarray:
    """
    Layer: detrended elevation.

    Signature: a Basin-and-Range normal-fault scarp is not a ridge and not a
               slope -- it is a STEP, steep on the fault-facing side and gentle
               on the bajada side.  The operator therefore compares the
               cross-strike slope at +offset against the slope at -offset,
               after integrating ALONG strike with an anisotropic Gaussian
               (major axis 4 px = 400 m along strike, 1 px across):

                   A(theta) = <|Dn f|>_(+offset along n)  -  <|Dn f|>_(-offset)

               and takes the max of |A| over 8 azimuths.  Along-strike
               integration is what makes a scarp half-buried in colluvium
               detectable at 100 m; the asymmetry test is what stops it firing
               on the regional range-front tilt, which is symmetric.

    Why it catches what the catalogue misses: a scarp is the direct surface
               expression of a young normal fault, and the catalogue-gap class
               the organisers define -- a continuation past a mapped tip, a
               splay, a parallel strand -- is exactly a scarp that someone has
               not traced yet.  Those are collinear with, and adjacent to,
               mapped traces, so they are also where a step filter has the most
               context.

    Differs from G-5: G-5 is an ALONG-strike knickpoint detector (second
               derivative of slope) plus an azimuth-variance term.  This is a
               CROSS-strike step/asymmetry detector.  Different operator,
               different failure mode: G-5 fires on drainage capture, this
               fires on fault faces.
    """
    elev = data.band("detrended_elevation", "elevation")
    e = np.nan_to_num(np.asarray(elev, dtype=np.float64), nan=0.0)

    # The organiser ships a slope-of-detrended-elevation product as well.  A
    # scarp is a slope concentration, so gating by the shipped slope layer
    # suppresses the flat basin floors where the asymmetry statistic is pure
    # noise.  Optional: fall back to a local gradient if the band is absent.
    # (This was the last of the 15 listed layers no detector read.)
    try:
        slope_ref = data.band("slope of detrended", "detrended_elevation_slope",
                              "slope_detrended")
    except KeyError:
        slope_ref = _grad_mag(e)
    slope_ref = _robust_scale(np.nan_to_num(slope_ref, nan=0.0))
    slope_ref = _minmax(slope_ref)

    best = np.zeros(e.shape, dtype=np.float64)
    for th in azimuths:
        t = np.deg2rad(th)
        ux, uy = np.cos(t), np.sin(t)            # along strike
        nx, ny = -uy, ux                         # across strike (unit normal)
        # integrate along strike, keep the cross-strike profile sharp
        sm = anisotropic_gaussian(e, th, 4.0, 1.0)
        gy, gx = np.gradient(sm)
        dn = gx * nx + gy * ny                   # directional derivative across strike
        # shift |dn| by +-offset along the normal and compare: a step is
        # asymmetric, a regional tilt is not.
        sh = int(round(offset))
        sh = max(sh, 1)
        dy = int(round(sh * ny))
        dx = int(round(sh * nx))
        plus = np.roll(np.roll(np.abs(dn), dy, axis=0), dx, axis=1)
        minus = np.roll(np.roll(np.abs(dn), -dy, axis=0), -dx, axis=1)
        np.maximum(best, np.abs(plus - minus), out=best)

    score = _robust_scale(best) * (0.45 + 0.55 * slope_ref)
    return _suppress_catalogue(score, data.catalogue, keep_from_px=1.0,
                               tau=suppress_tau).astype(np.float32)


# --------------------------------------------------------------------------- #
# H-3  Gravity-gradient lineament
# --------------------------------------------------------------------------- #
def h3_gravity_lineament(data: CompetitionData,
                         suppress_tau: float = 1.5) -> np.ndarray:
    """
    Layers: isostatic gravity anomaly  +  slope of the isostatic gravity anomaly.
            BOTH were read by no detector in this repo before this file.

    Signature: a basin-margin normal fault juxtaposes dense basement against
               low-density basin fill, which is a density step that survives
               burial.  The detector takes the horizontal-gradient magnitude of
               the isostatic anomaly and gates it two ways: by the slope layer
               the organiser already computed, and by _circular_coherence --
               the gradient direction must be CONSTANT ALONG the feature, which
               is the property a straight fault has and a lithologic blob does
               not.

    Why it catches what the catalogue misses: gravity sees through basin fill.
               The blind faults under the valleys have no scarp and no aerial
               expression, so the geomorphic catalogue cannot contain them;
               their density offset is still there.

    Differs from G-4: G-4 is magnetics.  In the Great Basin density and
               magnetisation decorrelate -- a lot of the basement is
               non-magnetic granite -- so a magnetic contact edge and a gravity
               step are not redundant, and their intersection is a much smaller
               and much more fault-like set than either alone.
    """
    grav = data.band("isostatic_gravity", "gravity_anomaly", "gravity")
    try:
        gslope = data.band("isostatic_gravity_slope", "gravity_slope")
    except KeyError:
        gslope = None

    hgm = _robust_scale(_grad_mag(grav))
    hgm = _minmax(hgm)
    coh = _circular_coherence(grav)
    score = hgm * (0.4 + 0.6 * coh)
    if gslope is not None:
        g = _robust_scale(gslope)
        g = _minmax(g)
        score = score * (0.5 + 0.5 * g)
    return _suppress_catalogue(_robust_scale(score), data.catalogue,
                               tau=suppress_tau).astype(np.float32)


# --------------------------------------------------------------------------- #
# H-4  Multi-depth conductance persistence   (EXTERNAL DATA)
# --------------------------------------------------------------------------- #
def h4_conductance_depth_persistence(data: CompetitionData,
                                     conductance_stack: np.ndarray | None = None,
                                     ) -> np.ndarray:
    """
    Layers: NOT in training_features.tif.  Needs the INGENIOUS electrical
            conductance maps -- five depth ranges spanning 2 to 200 km.
            Source (free, official, USGS/DOE):
              DOI https://doi.org/10.5066/P9TWT2LU
              listed at https://gdr.openei.org/submissions/1391
              licence: CC-BY 4.0 / US public domain

    Signature: cross-depth AZIMUTH PERSISTENCE of the conductance gradient.  A
               fluid-filled fault is a conductor that is continuous in depth, so
               the direction of the conductance gradient is the same at 2 km and
               at 20 km.  A lithologic or sedimentary contact decorrelates with
               depth.  The statistic is the resultant length of the doubled
               gradient azimuth averaged over the depth slices, times the mean
               gradient magnitude.

    Why it catches what the catalogue misses: blind faults under basin fill.
               And the depth-persistence criterion is precisely what suppresses
               the dominant false positive of G-3, which is that ANY lithologic
               contact produces a conductivity edge -- G-3's own docstring
               admits this.  H-4 is the fix for G-3's failure mode, not a
               variant of it.

    Differs from G-3: G-3 uses one surface layer and a log-ratio.  This uses a
               depth stack and a cross-depth azimuth statistic.  Different data,
               different transform, different false-positive profile.

    Corroboration (not evidence): forum thread 11527 post #6, participant
    moongrega (rank 12, 0.2180): "The conductance gave me something to work
    with".  https://community.drivendata.org/raw/11527
    """
    if conductance_stack is None:
        raise ValueError(
            "H-4 needs the INGENIOUS multi-depth conductance stack "
            "(DOI 10.5066/P9TWT2LU, free, listed at "
            "https://gdr.openei.org/submissions/1391). Pass it as "
            "conductance_stack=(D, H, W). This detector does NOT fall back to "
            "the surface-conductivity layer, because doing so would silently "
            "make it G-3 again."
        )
    st = np.asarray(conductance_stack, dtype=np.float64)
    if st.ndim != 3 or st.shape[1:] != data.catalogue.shape:
        raise ValueError(
            f"conductance_stack must be (D, {data.catalogue.shape[0]}, "
            f"{data.catalogue.shape[1]}); got {st.shape}")

    phis, mags = [], []
    for k in range(st.shape[0]):
        gy, gx = np.gradient(np.nan_to_num(st[k], nan=0.0))
        phis.append(np.arctan2(gy, gx))
        mags.append(np.hypot(gx, gy))
    C = np.mean([np.cos(2 * p) for p in phis], axis=0)
    Ss = np.mean([np.sin(2 * p) for p in phis], axis=0)
    persistence = np.clip(np.hypot(C, Ss), 0, 1)
    mag = _robust_scale(np.mean(mags, axis=0))
    score = mag * (0.3 + 0.7 * persistence)
    return _suppress_catalogue(_robust_scale(score),
                               data.catalogue).astype(np.float32)


# --------------------------------------------------------------------------- #
# H-5  Hydrothermal expression alignment   (EXTERNAL DATA)
# --------------------------------------------------------------------------- #
def _line_kernel(theta_deg, half_len, half_width):
    """Binary structuring element: a rectangle of length 2*half_len+1 along
    `theta_deg` and width 2*half_width+1 across it."""
    t = np.deg2rad(theta_deg)
    ux, uy = np.cos(t), np.sin(t)
    half = half_len
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1]
    along = xx * ux + yy * uy
    across = -xx * uy + yy * ux
    return ((np.abs(along) <= half_len) & (np.abs(across) <= half_width)).astype(np.float64)


def h5_hydrothermal_alignment(data: CompetitionData,
                              expression_xy: np.ndarray | None = None,
                              radius_px: int = 12,
                              azimuths=(0, 15, 30, 45, 60, 75, 90, 105, 120,
                                          135, 150, 165)) -> np.ndarray:
    """
    Layers: NOT in training_features.tif.  Needs the INGENIOUS paleo-geothermal
            features (sinter and tufa deposits) and Quaternary volcanic vents:
              https://gdr.openei.org/files/1391/paleo_geothermal_regional.zip  (82 kB)
              https://gdr.openei.org/files/1391/great_basin_q_volcanics.zip    (9.4 MB)
              DOI https://doi.org/10.15121/1881483, licence CC-BY 4.0

    Signature: a POINT-LINEAMENT detector.  Springs, sinter and vents are
               upflow discharge points and upflow follows faults, so a
               hydrothermal trace is a set of points lying on a line.  The
               points are rasterised, then convolved with an elongated
               rectangle for each of 12 azimuths -- a standard directional line
               detector -- and the best azimuth wins.  The response is divided
               by the local point density over a disc of radius `radius_px/2`,
               so a *cluster* of springs (a discharge zone) cannot beat a
               *line* of them (a fault).  Without that normalisation this
               detector would just be a hot-spring map.

    Why it catches what the catalogue misses: a thermal spring or a sinter
               deposit is direct evidence of fault-controlled fluid upflow, and
               it can exist where the controlling fault has no surface trace at
               all.  This is the only candidate here whose evidence is the
               geothermal system itself rather than a proxy for structure.

    Differs from everything implemented: no detector in this repo uses any
               hydrothermal surface-expression data.  It is also the one that is
               directly about geothermal resources rather than about faults in
               general, which is what the sponsor actually wants.
    """
    from scipy.ndimage import convolve, uniform_filter

    if expression_xy is None:
        raise ValueError(
            "H-5 needs hydrothermal expression points as (N, 2) in pixel "
            "(row, col) coordinates on the competition grid. Sources: "
            "paleo_geothermal_regional.zip and great_basin_q_volcanics.zip, "
            "DOI 10.15121/1881483, CC-BY 4.0, listed at "
            "https://gdr.openei.org/submissions/1391"
        )
    pts = np.asarray(expression_xy, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) == 0:
        raise ValueError(f"expression_xy must be (N, 2) and non-empty; got {pts.shape}")

    H, W = data.catalogue.shape
    rows = np.clip(np.round(pts[:, 0]).astype(int), 0, H - 1)
    cols = np.clip(np.round(pts[:, 1]).astype(int), 0, W - 1)

    pts_raster = np.zeros((H, W), dtype=np.float64)
    np.add.at(pts_raster, (rows, cols), 1.0)

    # local point density over a disc of radius radius_px/2, for the
    # line-vs-cluster normalisation
    r = max(2, int(radius_px) // 2)
    side = 2 * r + 1
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    disc = ((xx * xx + yy * yy) <= r * r).astype(np.float64)
    dens = convolve(pts_raster, disc / disc.sum()) * disc.sum()

    best = np.zeros((H, W), dtype=np.float64)
    for th in azimuths:
        k = _line_kernel(th, int(radius_px), 1.0)
        np.maximum(best, convolve(pts_raster, k), out=best)

    score = best / np.maximum(dens, 1.0)
    return _suppress_catalogue(_robust_scale(score),
                               data.catalogue).astype(np.float32)


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
_UNVERIFIED = ("NEITHER IS VALIDATED ON REAL DATA. Ranked by expected value per "
               "unit of implementation cost, on the reasoning in the docstring, "
               "not on measurement.")

H1 = Hypothesis(
    "H-1", "Seismogenic organisation",
    layers="density of earthquakes; shear strain rate "
           "(both listed in the official feature list, both previously unread "
           "by any detector in this repo)",
    signature="earthquake-density structure-tensor coherence x shear strain "
              "rate, catalogue suppressed",
    why_missing="The catalogue is a geomorphic map: it maps scarps. A fault that "
                "is slipping now but has no preserved scarp is invisible to it. "
                "Seismicity is the one shipped layer that measures present "
                "activity rather than past surface expression.",
    differs="No detector in this repo, G-1..G-5, reads earthquake density or "
            "shear strain rate. Verified by scripts/audit_layer_usage.py.",
    cost="Low - both layers already in training_features.tif",
    expected="Medium. Seismicity products are spatially smoothed, so localisation "
             "at 100 m may be weak. " + _UNVERIFIED,
    fn=h1_seismogenic_organization,
)

H2 = Hypothesis(
    "H-2", "Strike-integrated scarp step (cross-profile asymmetry)",
    layers="detrended elevation",
    signature="max over 8 azimuths of |<|Dn f|>_(+2px) - <|Dn f|>_(-2px)| after "
              "anisotropic along-strike smoothing (400 m along, 100 m across)",
    why_missing="A scarp is the direct surface expression of a young normal "
                "fault. The catalogue-gap class the organisers define -- a "
                "continuation past a mapped tip, a splay, a parallel strand -- "
                "is precisely a scarp nobody has traced yet.",
    differs="G-5 is an ALONG-strike knickpoint detector. This is a CROSS-strike "
            "step/asymmetry detector. Different operator, different failure "
            "mode. Along-strike integration is what neither G-5 nor a plain "
            "gradient does.",
    cost="Low - one layer already in training_features.tif",
    expected="Highest per unit cost: no external data, one layer, and it targets "
             "the dominant fault type in the region. " + _UNVERIFIED,
    fn=h2_scarp_step_matched_filter,
)

H3 = Hypothesis(
    "H-3", "Gravity-gradient lineament",
    layers="isostatic gravity anomaly; slope of the isostatic gravity anomaly "
           "(both listed in the official feature list, both previously unread)",
    signature="horizontal-gradient magnitude of the isostatic anomaly, gated by "
              "the shipped gravity-slope layer and by along-strike gradient-"
              "azimuth coherence",
    why_missing="Gravity sees through basin fill. Blind faults under the valleys "
                "have no scarp and no aerial expression, so a geomorphic "
                "catalogue cannot contain them; their density offset survives.",
    differs="G-4 is magnetics. Density and magnetisation decorrelate in the "
            "Great Basin (much of the basement is non-magnetic granite), so the "
            "two are not redundant.",
    cost="Low - both layers already in training_features.tif",
    expected="Medium. Isostatic anomalies are long-wavelength, so the achievable "
             "spatial precision at 100 m is the open question. " + _UNVERIFIED,
    fn=h3_gravity_lineament,
)

H4 = Hypothesis(
    "H-4", "Multi-depth conductance persistence",
    layers="INGENIOUS electrical conductance, five depth ranges 2-200 km "
           "(NOT in training_features.tif)",
    signature="resultant length of the doubled conductance-gradient azimuth "
              "averaged over depth slices, x mean gradient magnitude",
    why_missing="Blind faults under basin fill. Depth persistence is what "
                "suppresses G-3's admitted dominant false positive -- any "
                "lithologic contact makes a conductivity edge.",
    differs="G-3 uses one surface layer and a log-ratio. This uses a depth stack "
            "and a cross-depth azimuth statistic. It is the fix for G-3's "
            "failure mode, not a variant of it.",
    cost="Medium + a one-time external download",
    expected="Medium-High if the depth slices resolve, but blocked on data. "
             + _UNVERIFIED,
    fn=h4_conductance_depth_persistence,
    external="INGENIOUS electrical conductance maps, 5 depth ranges 2-200 km",
    external_source="DOI 10.5066/P9TWT2LU (free, USGS/DOE), listed at "
                    "https://gdr.openei.org/submissions/1391",
)

H5 = Hypothesis(
    "H-5", "Hydrothermal expression alignment",
    layers="INGENIOUS paleo-geothermal features (sinter, tufa) and Quaternary "
           "volcanic vents (NOT in training_features.tif)",
    signature="point-lineament detector: best of 12 azimuths of the count of "
              "expression points aligned through a pixel within 1.5 px, "
              "normalised by local point density",
    why_missing="A thermal spring or sinter deposit is direct evidence of "
                "fault-controlled fluid upflow and can exist where the "
                "controlling fault has no surface trace at all.",
    differs="Nothing in this repo uses hydrothermal surface-expression data. It "
            "is also the only candidate whose evidence is the geothermal system "
            "itself rather than a proxy for structure.",
    cost="Medium + a one-time external download (the files are small: 82 kB and "
         "9.4 MB)",
    expected="Medium-High and the most sponsor-aligned, but blocked on data. "
             + _UNVERIFIED,
    fn=h5_hydrothermal_alignment,
    external="paleo_geothermal_regional.zip (82 kB) and "
             "great_basin_q_volcanics.zip (9.4 MB)",
    external_source="DOI 10.15121/1881483, CC-BY 4.0, "
                    "https://gdr.openei.org/submissions/1391",
)

ALL = [H1, H2, H3, H4, H5]


def shipped_hypotheses() -> list[Hypothesis]:
    """The ones that run on the competition's own layers only."""
    return [h for h in ALL if h.external is None]


def external_hypotheses() -> list[Hypothesis]:
    return [h for h in ALL if h.external is not None]


def all_hypotheses() -> list[Hypothesis]:
    return list(ALL)
