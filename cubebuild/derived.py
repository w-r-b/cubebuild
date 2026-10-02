"""派生层三件：陆海掩膜 / 坡度 / 起伏度。

白名单内确定性几何派生，三层全部逐档生成（全五档）：

- derived__landsea_mask：30″ 档 = 全域 ice-surface 高程 0m 等值线
  （surface>0 → 1 陆，≤0 → 0 海；掩膜冰面口径——瓦片
  选择切换为 surface 优先，冰区 62 位取 bed_crosscheck surface 瓦片，
  冰盖归陆，常规地理口径）；粗档 = 30″ 掩膜向粗档众数聚合（YAML
  resampling=mode，契约原生 30″——掩膜语义「向粗档按众数聚合」是
  明示的跨档例外，与坡度/起伏度「逐档重算不跨档」并行不悖）。
  口径语义：掩膜基于冰面高程，与高程层（裸地复合口径）定义不同——
  存在 mask=陆而高程<0 的冰下盆地像元（南极/格陵兰），配对使用注意。
- derived__slope：逐档从该档高程重算，中央差分梯度 → arctan，
  单位度；经度周期边界（全球网格环接日期线）、纬度边界单侧差分。
- derived__relief：逐档从该档高程重算，3×3 移动窗口极差，单位 m；
  经度周期边界、纬度边界裁剪窗口（极区缘行为 2×3）。

水平距离球面解析（IUGG authalic 半径，与 pixel_area 同约定）：
dy = R·Δφ_rad；dx = R·cosφ·Δλ_rad。
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
NATIVE_TIER = "30sec"   # landsea_mask 契约原生档（掩膜众数聚合的源档）

# 陆地占比硬边界（掩膜反转/整体错位等粗错捕捉器）：地球表面陆地约 29%，
# 冰面口径下冰盖归陆，0.15–0.45 对真实值留足裕量又能
# 拦截反转（~0.72）与全错位（~0 或 ~1）。
LAND_FRACTION_BOUNDS = (0.15, 0.45)


def landsea_threshold(z: da.Array) -> da.Array:
    """高程 → 掩膜：elev > 0 → 1（陆），≤ 0 → 0（海），uint8。"""
    return (z > 0).astype(np.uint8)


def mode_downsample(mask: da.Array, factor: int) -> da.Array:
    """掩膜向粗档众数聚合（块内 aggregate_mode；块不整除 factor 即显式失败）。"""
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
    """坡度核心算子（度）：中央差分梯度 → arctan(√(gx²+gy²))。

    - gx：经度周期中央差分（列 0 的左邻 = 列 nlon−1，日期线无缝）；
    - gy：纬度中央差分 + 首/末行单侧差分（np.gradient 同语义）；
    - 水平距离球面解析（authalic R）：dy=R·Δφ，dx=R·cosφ·Δλ。
    z 任意分块皆可（移位由显式 concatenate 完成，不依赖块对齐）。
    """
    z64 = z.astype(np.float64)
    nlat, nlon = z64.shape
    dlat_m = R_M * np.deg2rad(res_deg)
    dlon_m = R_M * np.deg2rad(res_deg) * np.cos(np.deg2rad(lat_deg))  # (nlat,)

    # gx：周期补一列于两侧 → pad[:, j] 与 pad[:, j+2] 恰为中心像元左右邻
    pad = da.concatenate([z64[:, -1:], z64, z64[:, :1]], axis=1)
    gx = (pad[:, 2:] - pad[:, :-2]) / (2.0 * dlon_m[:, None])

    # gy：内部中央差分 + 边界单侧（线性场下两者均解析精确）
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
    else:  # nlat == 1：无纬向邻居
        gy = da.zeros_like(z64)

    return np.degrees(np.arctan(np.sqrt(gx * gx + gy * gy)))


def _lon_window3(x: da.Array, ufunc) -> da.Array:
    """经度方向 3 列窗（周期）：max/min 可分离，先行后列。"""
    pad = da.concatenate([x[:, -1:], x, x[:, :1]], axis=1)
    return ufunc(ufunc(pad[:, :-2], pad[:, 1:-1]), pad[:, 2:])


def relief_core(z: da.Array) -> da.Array:
    """起伏度核心算子（m）：3×3 窗口 max−min。

    max/min 可分离：先纬度 3 行窗（边界裁剪——极区缘窗缩为 2×3），
    再经度 3 列窗（周期）。极区行窗与经度周期的语义由测试逐位钉死。
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
    else:  # nlat == 1：仅经度窗
        win_max, win_min = z, z
    return _lon_window3(win_max, da.maximum) - _lon_window3(win_min, da.minimum)


def build_landsea_mask(entry: LayerEntry, tier: str) -> xr.DataArray:
    """构建该档 derived__landsea_mask。

    30″ 档 = 全域 ice-surface 高程 0m 阈值（掩膜冰面口径，瓦片
    选择切换为 surface 优先，冰盖归陆）；粗档 = 30″ 掩膜众数聚合
    （平票取小值→海）。高程层（裸地复合口径）/坡度/起伏度不受影响。
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
    """构建该档 derived__slope（逐档从该档高程重算，单位度）。"""
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
    """构建该档 derived__relief（3×3 窗口极差，逐档重算，单位 m）。"""
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
