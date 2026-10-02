"""derived__pixel_area —— 像元面积层。

逐档按纬度解析公式精确计算（不经重采样）：
    A = R² · Δλ_rad · (sin φ_hi − sin φ_lo)
地球半径取 IUGG 等面积（authalic）半径 R = 6371.0071810 km——与全球面积守恒
自洽（全部像元面积之和 = 4πR²，即地球表面总面积）。该约定登记于 cube_manifest。
"""

import dask.array as da
import numpy as np
import xarray as xr

from .contract import LayerEntry
from .grids import grid_shape, tier_centers, tier_chunks, tier_deg, tier_edges_lat

EARTH_RADIUS_KM = 6371.0071810  # IUGG authalic（等面积）半径
FORMULA = "A = R^2 * dlon_rad * (sin(lat_hi) - sin(lat_lo))"


def pixel_area_rows(tier: str) -> np.ndarray:
    """该档每个纬度行的像元面积（km²，float64，长度 nlat）。

    同纬度行内面积与经度无关，行向量即整层信息；写出时广播为 (nlat, nlon)。
    """
    nlat, nlon = grid_shape(tier)
    edges = tier_edges_lat(tier)
    rad = np.deg2rad(edges)
    dlon = np.deg2rad(360.0 / nlon)
    return EARTH_RADIUS_KM**2 * dlon * (np.sin(rad[1:]) - np.sin(rad[:-1]))


def build_pixel_area(entry: LayerEntry, tier: str) -> xr.DataArray:
    """构建该档的 derived__pixel_area DataArray（dask 惰性，逐块广播行向量）。"""
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
