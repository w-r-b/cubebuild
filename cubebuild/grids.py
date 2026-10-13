"""Global tier grid definitions.

Convention: EPSG:4326, grid edges aligned to -180/-90 and +180/+90 (edges are
integer multiples of the tier resolution), pixel centers offset by half a tier
(the rasterio edge-aligned convention).
"""

from fractions import Fraction

import numpy as np

TIER_ORDER = ("1deg", "30min", "6min", "3min", "30sec")

_TIER_DEG_FRAC = {
    "1deg": Fraction(1),
    "30min": Fraction(1, 2),
    "6min": Fraction(1, 10),
    "3min": Fraction(1, 20),
    "30sec": Fraction(1, 120),
}

_TIER_CHUNKS = {
    "1deg": (180, 360),
    "30min": (360, 720),
    "6min": (900, 900),
    "3min": (900, 900),
    "30sec": (1080, 1080),
}


def tier_deg(tier: str) -> float:
    """Tier resolution (degrees, floating-point approximation)."""
    return float(_TIER_DEG_FRAC[tier])


def tier_fraction(tier: str) -> Fraction:
    """Tier resolution (degrees, exact rational) — used for exact arithmetic such as conservative-kernel periodic decomposition."""
    return _TIER_DEG_FRAC[tier]


def grid_shape(tier: str) -> tuple[int, int]:
    """Global grid shape (nlat, nlon)."""
    nlat = Fraction(180) / _TIER_DEG_FRAC[tier]
    nlon = Fraction(360) / _TIER_DEG_FRAC[tier]
    if nlat.denominator != 1 or nlon.denominator != 1:
        raise ValueError(f"档位 {tier} 无法整除全球网格")
    return int(nlat), int(nlon)


def tier_centers(tier: str) -> tuple[np.ndarray, np.ndarray]:
    """Pixel-center coordinates (lat, lon) of the tier's global grid, float64."""
    nlat, nlon = grid_shape(tier)
    res = tier_deg(tier)
    lat = -90.0 + (np.arange(nlat, dtype=np.float64) + 0.5) * res
    lon = -180.0 + (np.arange(nlon, dtype=np.float64) + 0.5) * res
    return lat, lon


def tier_edges_lat(tier: str) -> np.ndarray:
    """Latitude edges of the tier's global grid (nlat+1 of them), used for the analytic area formula."""
    nlat, _ = grid_shape(tier)
    res = tier_deg(tier)
    return -90.0 + np.arange(nlat + 1, dtype=np.float64) * res


def tier_chunks(tier: str) -> tuple[int, int]:
    """Tier Zarr chunks, guaranteed not to exceed the grid shape."""
    clat, clon = _TIER_CHUNKS[tier]
    nlat, nlon = grid_shape(tier)
    return (min(clat, nlat), min(clon, nlon))


def aggregation_factor(tier: str, home_tier: str) -> int:
    """Integer aggregation factor from the home tier to the target tier (a non-integer fails explicitly).

    The single factor source for coarse tiers = home-tier block aggregation (mode/sum etc.
    integer-multiple kernels) (shared by vector / points; reused by heatflow point density).
    """
    f = tier_fraction(tier) / tier_fraction(home_tier)
    if f.denominator != 1:
        raise ValueError(f"{tier} 不是主档 {home_tier} 的整数倍聚合")
    return int(f)
