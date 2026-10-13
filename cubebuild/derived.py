"""Three derived layers: land-sea mask / slope / relief.

Deterministic geometric derivation within an allowlist; all three layers are
generated tier by tier (all five tiers):

- derived__landsea_mask: 30″ tier = global ice-surface elevation 0 m contour
  (surface>0 → 1 land, ≤0 → 0 sea; mask ice-surface convention — tile
  selection switches to surface-first, the 62 ice-zone positions take
  bed_crosscheck surface tiles, ice sheets count as land, conventional
  geographic convention); coarse tiers = 30″ mask aggregated by mode toward
  the coarse tier (YAML resampling=mode, contract native 30″ — the mask
  semantics "aggregate by mode toward the coarse tier" is an explicit
  cross-tier exception, coexisting with slope/relief's "recompute per tier,
  no cross-tier"). Convention semantics: the mask is based on ice-surface
  elevation, defined differently from the elevation layer (bare-earth composite
  convention) — there exist sub-ice basin pixels with mask=land but
  elevation<0 (Antarctica/Greenland); mind this when using them together.
- derived__slope: recomputed per tier from that tier's elevation, central-
  difference gradient → arctan, unit degrees; longitude periodic boundary
  (global grid wraps at the dateline), latitude boundary one-sided difference.
- derived__relief: recomputed per tier from that tier's elevation, 3×3 moving-
  window range, unit m; longitude periodic boundary, latitude boundary window
  clipped (polar-edge behavior 2×3).

Horizontal distance spherical analytic (IUGG authalic radius, same convention
as pixel_area): dy = R·Δφ_rad; dx = R·cosφ·Δλ_rad.
"""

import dask.array as da
import numpy as np
import xarray as xr

from .contract import LayerEntry
from .etopo import elevation_array, ice_surface_array
from .grids import tier_centers, tier_chunks, tier_deg
from .kernels import aggregate_mode
from .pixel_area import EARTH_RADIUS_KM

R_M = EARTH_RADIUS_KM * 1000.0
NATIVE_TIER = "30sec"

LAND_FRACTION_BOUNDS = (0.15, 0.45)


def landsea_threshold(z: da.Array) -> da.Array:
    """Elevation → mask: elev > 0 → 1 (land), ≤ 0 → 0 (sea), uint8."""
    return (z > 0).astype(np.uint8)


def mode_downsample(mask: da.Array, factor: int) -> da.Array:
    """Aggregate the mask toward the coarse tier by mode (via aggregate_mode within a block; a block not divisible by factor fails explicitly)."""
    bad = [
        (ax, c)
        for ax, chunks in enumerate(mask.chunks)
        for c in chunks
        if c % factor
    ]
    if bad:
        raise ValueError(
            f"众数聚合 factor={factor} 不整除块结构 {mask.chunks}：{bad[:3]}..."
        )
    return mask.map_blocks(
        aggregate_mode,
        factor=factor,
        dtype=mask.dtype,
        chunks=tuple(tuple(c // factor for c in ch) for ch in mask.chunks),
    )


def slope_core(z: da.Array, lat_deg: np.ndarray, res_deg: float) -> da.Array:
    """Slope core operator (degrees): central-difference gradient → arctan(√(gx²+gy²)).

    - gx: longitude periodic central difference (column 0's left neighbor = column nlon−1, seamless at the dateline);
    - gy: latitude central difference + one-sided difference at the first/last rows (same semantics as np.gradient);
    - horizontal distance spherical analytic (authalic R): dy=R·Δφ, dx=R·cosφ·Δλ.
    z may be chunked arbitrarily (shifts are done via explicit concatenate, independent of chunk alignment).
    """
    z64 = z.astype(np.float64)
    nlat, nlon = z64.shape
    dlat_m = R_M * np.deg2rad(res_deg)
    dlon_m = R_M * np.deg2rad(res_deg) * np.cos(np.deg2rad(lat_deg))  # (nlat,)

    pad = da.concatenate([z64[:, -1:], z64, z64[:, :1]], axis=1)
    gx = (pad[:, 2:] - pad[:, :-2]) / (2.0 * dlon_m[:, None])

    if nlat >= 3:
        gy = da.concatenate(
            [
                (z64[1:2, :] - z64[0:1, :]) / dlat_m,
                (z64[2:, :] - z64[:-2, :]) / (2.0 * dlat_m),
                (z64[-1:, :] - z64[-2:-1, :]) / dlat_m,
            ],
            axis=0,
        )
    elif nlat == 2:
        d = (z64[1:2, :] - z64[0:1, :]) / dlat_m
        gy = da.concatenate([d, d], axis=0)
    else:
        gy = da.zeros_like(z64)

    return np.degrees(np.arctan(np.sqrt(gx * gx + gy * gy)))


def _lon_window3(x: da.Array, ufunc) -> da.Array:
    """Longitude-direction 3-column window (periodic): max/min are separable, applied row-first then column."""
    pad = da.concatenate([x[:, -1:], x, x[:, :1]], axis=1)
    return ufunc(ufunc(pad[:, :-2], pad[:, 1:-1]), pad[:, 2:])


def relief_core(z: da.Array) -> da.Array:
    """Relief core operator (m): 3×3 window max−min.

    max/min are separable: first a latitude 3-row window (boundary clipped — the polar-edge
    window shrinks to 2×3), then a longitude 3-column window (periodic). The semantics of
    the polar row window and the longitude periodicity are pinned bit-exactly by tests.
    """
    nlat = z.shape[0]
    pair_max = da.maximum(z[:-1, :], z[1:, :])
    pair_min = da.minimum(z[:-1, :], z[1:, :])
    if nlat >= 3:
        win_max = da.concatenate(
            [
                pair_max[0:1, :],
                da.maximum(pair_max[:-1, :], pair_max[1:, :]),
                pair_max[-1:, :],
            ],
            axis=0,
        )
        win_min = da.concatenate(
            [
                pair_min[0:1, :],
                da.minimum(pair_min[:-1, :], pair_min[1:, :]),
                pair_min[-1:, :],
            ],
            axis=0,
        )
    elif nlat == 2:
        win_max = da.concatenate([pair_max, pair_max], axis=0)
        win_min = da.concatenate([pair_min, pair_min], axis=0)
    else:
        win_max, win_min = z, z
    return _lon_window3(win_max, da.maximum) - _lon_window3(win_min, da.minimum)


def build_landsea_mask(entry: LayerEntry, tier: str) -> xr.DataArray:
    """Build derived__landsea_mask for the given tier.

    30″ tier = global ice-surface elevation 0 m threshold (mask ice-surface convention,
    tile selection switches to surface-first, ice sheets count as land); coarse tiers =
    30″ mask mode aggregation (ties take the smaller value → sea). The elevation layer
    (bare-earth composite convention) / slope / relief are unaffected.
    """
    mask30 = landsea_threshold(ice_surface_array(NATIVE_TIER))
    if tier == NATIVE_TIER:
        arr = mask30
    else:
        factor = int(round(tier_deg(tier) / tier_deg(NATIVE_TIER)))
        arr = mode_downsample(mask30, factor)
    arr = arr.rechunk(tier_chunks(tier))

    lat, lon = tier_centers(tier)
    return xr.DataArray(
        arr,
        dims=("lat", "lon"),
        coords={"lat": lat, "lon": lon},
        name=entry.id,
        attrs={
            "units": entry.unit,
            "long_name": "陆海掩膜（冰面高程 0m 等值线，冰盖归陆）",
            "source": entry.source,
            "resampling": entry.resampling,
            "derived_from": (
                "etopo2022-bed ice-surface 全域复合（62 冰区 surface 瓦片"
                "于 bed_crosscheck/ + 226 无冰区 surface 瓦片，掩膜冰面口径）"
            ),
            "derivation": (
                "30sec 档 = 全域 ice-surface 高程 0m 阈值（surface>0→1 陆，"
                "≤0→0 海；瓦片选择 surface 优先：冰区 62 位取 bed_crosscheck "
                "surface、无冰区 226 位取源目录 surface）；粗档 = 30sec 掩膜"
                "众数聚合（计数多数，平票取小值→海）；掩膜基于冰面高程，与"
                "高程层（裸地复合口径）定义不同——存在 mask=陆而高程<0 的"
                "冰下盆地像元（南极/格陵兰）"
            ),
            "visibility": entry.visibility,
        },
    )


def build_slope(entry: LayerEntry, tier: str) -> xr.DataArray:
    """Build derived__slope for the given tier (recomputed per tier from that tier's elevation, unit degrees)."""
    z = elevation_array(tier)
    lat, lon = tier_centers(tier)
    arr = slope_core(z, lat, tier_deg(tier)).astype("float32").rechunk(
        tier_chunks(tier)
    )

    return xr.DataArray(
        arr,
        dims=("lat", "lon"),
        coords={"lat": lat, "lon": lon},
        name=entry.id,
        attrs={
            "units": entry.unit,
            "long_name": "坡度（逐档从该档基岩高程重算，每档尺度语义）",
            "source": entry.source,
            "resampling": entry.resampling,
            "derived_from": "topography__bedrock_elevation",
            "derivation": (
                "中央差分梯度 → arctan(√(gx²+gy²))；经度周期边界、"
                "纬度边界单侧差分；水平距离球面解析"
            ),
            "earth_radius_km": EARTH_RADIUS_KM,
            "radius_convention": "IUGG authalic (equal-area)",
            "visibility": entry.visibility,
        },
    )


def build_relief(entry: LayerEntry, tier: str) -> xr.DataArray:
    """Build derived__relief for the given tier (3×3 window range, recomputed per tier, unit m)."""
    z = elevation_array(tier)
    arr = relief_core(z).astype("float32").rechunk(tier_chunks(tier))

    lat, lon = tier_centers(tier)
    return xr.DataArray(
        arr,
        dims=("lat", "lon"),
        coords={"lat": lat, "lon": lon},
        name=entry.id,
        attrs={
            "units": entry.unit,
            "long_name": "局部起伏度（3×3 移动窗口极差，逐档重算）",
            "source": entry.source,
            "resampling": entry.resampling,
            "derived_from": "topography__bedrock_elevation",
            "derivation": (
                "3×3 窗口 max−min；经度周期边界、纬度边界裁剪窗口（极区缘 2×3）"
            ),
            "visibility": entry.visibility,
        },
    )
