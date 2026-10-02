"""全球档位网格定义。

约定：EPSG:4326，网格边缘对齐 -180/-90 与 +180/+90（边缘为档位分辨率的整数倍），
像元中心偏移半档（rasterio edge-aligned 约定）。
"""

from fractions import Fraction

import numpy as np

TIER_ORDER = ("1deg", "30min", "6min", "3min", "30sec")

# 档位分辨率（度），用 Fraction 精确表示，避免浮点累积误差
_TIER_DEG_FRAC = {
    "1deg": Fraction(1),
    "30min": Fraction(1, 2),
    "6min": Fraction(1, 10),
    "3min": Fraction(1, 20),
    "30sec": Fraction(1, 120),
}

# 各档 Zarr 分块（lat, lon），后续各层统一沿用
_TIER_CHUNKS = {
    "1deg": (180, 360),
    "30min": (360, 720),
    "6min": (900, 900),
    "3min": (900, 900),
    "30sec": (1080, 1080),
}


def tier_deg(tier: str) -> float:
    """档位分辨率（度，浮点近似）。"""
    return float(_TIER_DEG_FRAC[tier])


def tier_fraction(tier: str) -> Fraction:
    """档位分辨率（度，精确有理数）——保守核周期分解等精确算术用。"""
    return _TIER_DEG_FRAC[tier]


def grid_shape(tier: str) -> tuple[int, int]:
    """全球网格形状 (nlat, nlon)。"""
    nlat = Fraction(180) / _TIER_DEG_FRAC[tier]
    nlon = Fraction(360) / _TIER_DEG_FRAC[tier]
    if nlat.denominator != 1 or nlon.denominator != 1:
        raise ValueError(f"档位 {tier} 无法整除全球网格")
    return int(nlat), int(nlon)


def tier_centers(tier: str) -> tuple[np.ndarray, np.ndarray]:
    """档位全球网格的像元中心坐标 (lat, lon)，float64。"""
    nlat, nlon = grid_shape(tier)
    res = tier_deg(tier)
    lat = -90.0 + (np.arange(nlat, dtype=np.float64) + 0.5) * res
    lon = -180.0 + (np.arange(nlon, dtype=np.float64) + 0.5) * res
    return lat, lon


def tier_edges_lat(tier: str) -> np.ndarray:
    """档位全球网格的纬度边缘（nlat+1 个），用于面积解析公式。"""
    nlat, _ = grid_shape(tier)
    res = tier_deg(tier)
    return -90.0 + np.arange(nlat + 1, dtype=np.float64) * res


def tier_chunks(tier: str) -> tuple[int, int]:
    """档位 Zarr 分块，保证不超过网格形状。"""
    clat, clon = _TIER_CHUNKS[tier]
    nlat, nlon = grid_shape(tier)
    return (min(clat, nlat), min(clon, nlon))


def aggregation_factor(tier: str, home_tier: str) -> int:
    """主档 → 目标档的整数聚合因子（非整数即显式失败）。

    粗档 = 主档块内聚合（mode/sum 等整数倍核）的唯一因子来源
    （vector / points 共用；热流点密度复用）。
    """
    f = tier_fraction(tier) / tier_fraction(home_tier)
    if f.denominator != 1:
        raise ValueError(f"{tier} 不是主档 {home_tier} 的整数倍聚合")
    return int(f)
