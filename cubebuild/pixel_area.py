"""derived__pixel_area — the pixel-area layer.

Computed exactly per tier by the latitude-analytic formula (without resampling):
    A = R² · Δλ_rad · (sin φ_hi − sin φ_lo)
The Earth radius uses the IUGG equal-area (authalic) radius R = 6371.0071810 km — self-consistent with global area conservation
(the sum of all pixel areas = 4πR², i.e. the total Earth surface area). This convention is recorded in cube_manifest.
"""

import dask.array as da
import numpy as np
import xarray as xr

from .contract import LayerEntry
from .grids import grid_shape, tier_centers, tier_chunks, tier_deg, tier_edges_lat

EARTH_RADIUS_KM = 6371.0071810
FORMULA = "A = R^2 * dlon_rad * (sin(lat_hi) - sin(lat_lo))"


def pixel_area_rows(tier: str) -> np.ndarray:
    """Per-latitude-row pixel area for this tier (km², float64, length nlat).

    Within a latitude row the area is independent of longitude, so the row vector carries the whole layer's information; when written it is broadcast to (nlat, nlon).
    """
    nlat, nlon = grid_shape(tier)
    edges = tier_edges_lat(tier)
    rad = np.deg2rad(edges)
    dlon = np.deg2rad(360.0 / nlon)
    return EARTH_RADIUS_KM**2 * dlon * (np.sin(rad[1:]) - np.sin(rad[:-1]))


def build_pixel_area(entry: LayerEntry, tier: str) -> xr.DataArray:
    """Build this tier's derived__pixel_area DataArray (dask-lazy, broadcasting the row vector block by block)."""
    rows = pixel_area_rows(tier)
    nlat, nlon = grid_shape(tier)
    clat, clon = tier_chunks(tier)

    row_da = da.from_array(rows, chunks=(clat,))
    area = (
        da.broadcast_to(row_da[:, None], shape=(nlat, nlon))
        .rechunk((clat, clon))
        .astype("float32")
    )

    lat, lon = tier_centers(tier)
    return xr.DataArray(
        area,
        dims=("lat", "lon"),
        coords={"lat": lat, "lon": lon},
        name=entry.id,
        attrs={
            "units": entry.unit,
            "long_name": "像元面积（按纬度解析公式逐档精确计算）",
            "source": entry.source,
            "resampling": entry.resampling,
            "earth_radius_km": EARTH_RADIUS_KM,
            "radius_convention": "IUGG authalic (equal-area)",
            "formula": FORMULA,
            "visibility": entry.visibility,
        },
    )
