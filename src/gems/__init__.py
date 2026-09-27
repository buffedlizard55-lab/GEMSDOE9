"""
GEMSDOE9 — package root.

Layout
------
metric.py    the official distance-weighted Tversky index, exact and vectorised
strategy.py  the closed-form algebra of that metric and the submission shapes
             that exploit it
pipeline.py  data loading, band resolution by name, features, spatial folds,
             model fitting, blocked holdout, final field
features.py  one auditable detector function per hypothesis (G-1 .. G-5)
io.py        thin rasterio helpers

Note on the API: the previous release exported `dilation_tendency`,
`conductivity_edge` and `fluvial_sl_ksn` from this package. `dilation_tendency`
was not the published dilation tendency and is gone; use
`features.dilation_tendency_real` (Simpson & Reiling 2008). The other two were
superseded by `g3_clay_cap_edge` and `g5_fluvial_knickpoint`, which operate on a
named-band CompetitionData instead of positional arrays.
"""

from .metric import (
    distance_weighted_tversky,
    distance_weighted_tversky_fast,
    triangular_kernel,
)
from . import strategy
from . import features

__all__ = [
    "distance_weighted_tversky",
    "distance_weighted_tversky_fast",
    "triangular_kernel",
    "strategy",
    "features",
]
