"""IO helpers for GeoTIFF"""

import numpy as np
from pathlib import Path

try:
    import rasterio
except ImportError:
    rasterio = None

def load_footprint_mask(path="data/processed/footprint_mask.npz"):
    p = Path(path)
    if not p.exists():
        return None
    arr = np.load(p)
    return arr['footprint']

def load_raster(path):
    if rasterio is None:
        raise ImportError("rasterio not installed")
    with rasterio.open(path) as src:
        data = src.read(1)
        profile = src.profile
    return data, profile
