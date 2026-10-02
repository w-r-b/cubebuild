"""topography__bedrock_elevation —— ETOPO 2022 源适配器 + 面积加权保守核。

源与拼接规则（磁盘实证 + ESSD 论文 MacFerrin et al. 2025, doi:10.5194/essd-17-1835-2025）：
- ETOPO 2022 15s 全球 = 288 片 15°×15° 瓦片（12 纬度带 × 24 经度列）；
- bed 版仅 62 片（格陵兰/北极/南极冰区足迹），其余 226 位由 ice-surface 版补齐
  （无冰区 bed ≡ surface：两者同为裸地/海底 DTM）；
- 拼接规则：bed 优先，surface 补缺 → 全球 bedrock 网格。
- 掩膜冰面口径：surface 优先变体——冰区 62 位取 bed_crosscheck/
  surface 瓦片、无冰区 226 位取源目录 surface 瓦片 → 全域 ice-surface 网格
  （ice_surface_array，仅供 derived__landsea_mask 消费）。

命名约定（论文 §2.2 + ngdc 目录列表实测）：
ETOPO_2022_v1_15s_[N|S]YY[W|E]XXX_{bed|surface}.tif，
YY = 瓦片北边界纬度绝对值。实测修正：−15°..0° 带标签为 N00（北边界 0°
取 N 前缀），非 S00——论文规则推导会得 S00，以目录实测为准。

重采样核：连续变量 = 面积加权保守平均。15s 与全部档位分辨率为整数倍
（30″/3′/6′/30′/1° → 因子 2/12/24/120/240），每个粗像元完全落入单一瓦片，
核 = 精确整数倍块聚合，行权重取纬度解析面积 R²·Δλ·(sin φ_hi − sin φ_lo)。
该核在解析上严格守恒积分量 Σ v·A（保真验证的 <0.1% 判据因此可精确检验）。
"""

from pathlib import Path

import dask.array as da
import numpy as np
import xarray as xr
from dask import delayed

from .contract import LayerEntry
from .grids import grid_shape, tier_centers, tier_deg
from .kernels import aggregate_weighted  # noqa: F401 —— 核库实现，测试经此导入
from .pixel_area import EARTH_RADIUS_KM

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SRC_DIR = REPO_ROOT / "original data/topography/etopo2022-bed"
# 冰区 62 片 surface 瓦片（62/62 构建级校验通过）——掩膜冰面
# 口径的冰区 surface 输入；构建源目录本身零改动（bed 优先口径不受影响）。
BED_CROSSCHECK_DIR = DEFAULT_SRC_DIR / "bed_crosscheck"

# 288 位全球瓦片网格（自北向南 12 带 × 自西向东 24 列；N00 = [−15°, 0°] 带）
LAT_BANDS = ("N90", "N75", "N60", "N45", "N30", "N15",
             "N00", "S15", "S30", "S45", "S60", "S75")
# 列序 = 数组列序（经度 −180 → +180 递增）：W180..W015, E000..E165。
# 排错序（如 E000 在前）会使经度环旋转 180°，仅日期线相邻处出一条假缝
# （其余 15° 边界两侧瓦片在错误排序下仍地理相邻，难以从数据直接察觉）。
LON_COLS = tuple(["W%03d" % d for d in range(180, 0, -15)]
                 + ["E%03d" % d for d in range(0, 180, 15)])

TILE_DEG = 15.0
TILE_SHAPE = (3600, 3600)          # 15s × 15°
SRC_RES_DEG = 15.0 / 3600.0        # 1/240°
DOI = "10.25921/fd45-gt74"
NODATA = -99999.0


class SourceNotReady(Exception):
    """源瓦片未集齐（缺 surface 补齐瓦片），构建期跳过该层并提示。"""

    def __init__(self, missing: list[str]):
        self.missing = missing
        n = len(missing)
        sample = ", ".join(missing[:5]) + ("..." if n > 5 else "")
        super().__init__(
            f"ETOPO 源未集齐：缺 {n} 片 surface 瓦片（如 {sample}）；"
            "下载清单见下载台账"
        )


def label_bounds(label: str) -> tuple[float, float]:
    """瓦片标签 → (北边界纬度, 西边界经度)。如 N60W030 → (60, -30)。"""
    lat_tok, lon_tok = label[:3], label[3:]
    top = int(lat_tok[1:]) * (1 if lat_tok[0] == "N" else -1)
    left = int(lon_tok[1:]) * (-1 if lon_tok[0] == "W" else 1)
    return float(top), float(left)


def all_labels() -> list[str]:
    return [b + c for b in LAT_BANDS for c in LON_COLS]


_ALL_LABELS = frozenset(all_labels())


def scan_tiles(src_dir: str | Path = DEFAULT_SRC_DIR) -> dict[str, Path]:
    """扫描源目录 → {瓦片标签: 文件路径}。bed 优先：同位 bed/surface 并存取 bed。"""
    src_dir = Path(src_dir)
    if not src_dir.is_dir():
        raise FileNotFoundError(f"ETOPO 源目录不存在: {src_dir}")
    chosen = _scan_kind(src_dir, "surface")      # surface 补缺
    chosen.update(_scan_kind(src_dir, "bed"))    # bed 优先（同位覆盖）
    return chosen


def missing_tiles(src_dir: str | Path = DEFAULT_SRC_DIR) -> list[str]:
    """288 位全球网格中未被 bed/surface 占据的瓦片标签。"""
    have = scan_tiles(src_dir)
    return [lb for lb in all_labels() if lb not in have]


def require_full_coverage(src_dir: str | Path = DEFAULT_SRC_DIR) -> dict[str, Path]:
    """全球覆盖校验：缺片即抛 SourceNotReady（列明缺失清单）。"""
    miss = missing_tiles(src_dir)
    if miss:
        raise SourceNotReady(miss)
    return scan_tiles(src_dir)


def scan_surface_tiles(
    src_dir: str | Path = DEFAULT_SRC_DIR,
    ice_surface_dir: str | Path = BED_CROSSCHECK_DIR,
) -> dict[str, Path]:
    """全域 ice-surface 瓦片选择（掩膜冰面口径）。

    瓦片选择规则切换（非数组后处理）：surface 优先——冰区 62 位取
    ice_surface_dir（bed_crosscheck/，构建级校验通过的 surface 瓦片）、
    无冰区 226 位取 src_dir 的既有 surface 瓦片；同位 bed 瓦片被忽略，
    同位两目录皆有 surface 时 ice_surface_dir 优先。组装结果与
    scan_tiles（bed 优先复合）互补独立，互不干扰。
    """
    src_dir, ice_dir = Path(src_dir), Path(ice_surface_dir)
    for d in (src_dir, ice_dir):
        if not d.is_dir():
            raise FileNotFoundError(f"ETOPO 源目录不存在: {d}")
    chosen = _scan_kind(src_dir, "surface")
    chosen.update(_scan_kind(ice_dir, "surface"))
    return chosen


def _scan_kind(src_dir: Path, kind: str) -> dict[str, Path]:
    """扫描单目录 → {瓦片标签: 文件路径}，只收指定 kind（bed/surface）。"""
    chosen: dict[str, Path] = {}
    for f in sorted(src_dir.glob("ETOPO_2022_v1_15s_*_*.tif")):
        stem = f.name.removesuffix(".tif")          # ..._N60W030_surface
        label, _, k = stem.partition("_15s_")[2].rpartition("_")
        if k == kind and label in _ALL_LABELS:
            chosen[label] = f
    return chosen


def require_surface_coverage(
    src_dir: str | Path = DEFAULT_SRC_DIR,
    ice_surface_dir: str | Path = BED_CROSSCHECK_DIR,
) -> dict[str, Path]:
    """全域 ice-surface 覆盖校验：缺片即抛 SourceNotReady（列明缺失清单）。"""
    have = scan_surface_tiles(src_dir, ice_surface_dir)
    miss = [lb for lb in all_labels() if lb not in have]
    if miss:
        raise SourceNotReady(miss)
    return have


def row_areas_km2(lat_top: float, nrows: int, res_deg: float) -> np.ndarray:
    """逐行像元面积（km²）：A = R²·Δλ_rad·(sin φ_hi − sin φ_lo)，Δλ = res_deg。"""
    edges = lat_top - np.arange(nrows + 1, dtype=np.float64) * res_deg
    rad = np.deg2rad(edges)
    dlon = np.deg2rad(res_deg)
    return EARTH_RADIUS_KM**2 * dlon * (np.sin(rad[:-1]) - np.sin(rad[1:]))


def read_tile_validated(path: Path) -> np.ndarray:
    """读瓦片 → float32 数组；含 nodata 像元即显式失败。

    ETOPO 各瓦全球有效（62 bed 片磁盘实证；surface 片到货后同受此校验，
    若含 nodata 构建期即崩，不静默产出偏值）。构建与保真统计共用此读法，
    保证两条路径 nodata 策略一致。
    """
    import rasterio

    with rasterio.open(path) as src:
        if src.shape != TILE_SHAPE:
            raise ValueError(f"{path.name}: 形状 {src.shape} ≠ {TILE_SHAPE}")
        data = src.read(1).astype(np.float32)
        nodata = src.nodata if src.nodata is not None else NODATA
    if np.any(data == nodata):
        raise ValueError(f"{path.name}: 含 nodata 像元，与全球全覆盖预期不符")
    return data


def read_tile_aggregated(path: Path, tier: str, lat_top: float) -> np.ndarray:
    """读单片瓦井聚合到目标档：输出该瓦在该档的子网格块（float32）。

    factor = 档位分辨率 / 15s（整数；15° 瓦在各档均整数分块）。
    输出行序约定：行 0 = 瓦片最南行（全球网格行 0 = −90°）。
    rasterio 读入行 0 = 瓦片北缘，聚合后垂直翻转一次对齐全球约定——
    遗漏翻转会使瓦内纬度梯度反号：积分量偏 ~1e-3 且 15° 边界出现
    假接缝（保真 harness 实测捕获，正是其设计目的）。
    """
    factor = int(round(tier_deg(tier) / SRC_RES_DEG))
    if abs(tier_deg(tier) / SRC_RES_DEG - factor) > 1e-9:
        raise ValueError(f"档位 {tier} 相对 15s 非整数倍，保守核不适用")
    data = read_tile_validated(path)
    w = row_areas_km2(lat_top, TILE_SHAPE[0], SRC_RES_DEG)
    out = aggregate_weighted(data, w, factor)
    return out[::-1].astype(np.float32)


def _assemble_tiles(tiles: dict[str, Path], tier: str) -> da.Array:
    """瓦片选择结果 → 该档全球 dask 数组（块 = 15° 瓦片边界，float32）。

    bed 优先（elevation_array）与 surface 优先（ice_surface_array）共用
    此组装段：行 0 = −90°，LAT_BANDS 自北向南 → 逆序。
    """
    nlat, nlon = grid_shape(tier)
    ot = TILE_SHAPE[0] // int(round(tier_deg(tier) / SRC_RES_DEG))  # 每瓦行数
    on = TILE_SHAPE[1] // int(round(tier_deg(tier) / SRC_RES_DEG))
    if nlat % ot or nlon % on:
        raise ValueError(f"档位 {tier}: 瓦块 ({ot},{on}) 不整除全球网格")

    # 自南向北组装（数组行 0 = -90°；LAT_BANDS 自北向南 → 逆序）
    blocks = []
    for band in reversed(LAT_BANDS):
        row_blocks = []
        for col in LON_COLS:
            label = band + col
            row_top, _ = label_bounds(label)
            blk = delayed(read_tile_aggregated, pure=True)(
                tiles[label], tier, row_top
            )
            row_blocks.append(
                da.from_delayed(blk, shape=(ot, on), dtype=np.float32)
            )
        blocks.append(row_blocks)
    return da.block(blocks)


def elevation_array(tier: str, src_dir: str | Path = DEFAULT_SRC_DIR) -> da.Array:
    """该档基岩高程 dask 数组（bed 优先复合，块 = 15° 瓦片边界，float32）。

    高程层与坡度/起伏度派生层共用此入口：pure delayed 同参同键，
    同档写入时 dask 调度器对共享子图只算一次。陆海掩膜（冰面口径）改走 ice_surface_array，与本入口独立。
    """
    return _assemble_tiles(require_full_coverage(src_dir), tier)


def ice_surface_array(
    tier: str,
    src_dir: str | Path = DEFAULT_SRC_DIR,
    ice_surface_dir: str | Path = BED_CROSSCHECK_DIR,
) -> da.Array:
    """该档全域 ice-surface 高程 dask 数组（掩膜冰面口径）。

    瓦片选择 = surface 优先（冰区 62 位取 bed_crosscheck/、无冰区 226 位
    取构建源目录 surface），组装段与 elevation_array 共用（_assemble_tiles）。
    仅供 derived__landsea_mask 阈值化消费；高程层（裸地复合口径）与
    坡度/起伏度不受影响。
    """
    return _assemble_tiles(
        require_surface_coverage(src_dir, ice_surface_dir), tier
    )


def build_etopo_bedrock(entry: LayerEntry, tier: str) -> xr.DataArray:
    """构建该档 topography__bedrock_elevation（dask 惰性，块 = 15° 瓦片边界）。"""
    arr = elevation_array(tier)

    lat, lon = tier_centers(tier)
    return xr.DataArray(
        arr,
        dims=("lat", "lon"),
        coords={"lat": lat, "lon": lon},
        name=entry.id,
        attrs={
            "units": entry.unit,
            "long_name": "基岩高程（ETOPO 2022 bed 版，冰区基岩/冰下地形）",
            "source": entry.source,
            "doi": DOI,
            "native_res": entry.native_res,
            "resampling": entry.resampling,
            "composite_rule": (
                "62 bed tiles（冰区足迹，bed 优先）+ 226 surface tiles 补齐"
                "（无冰区 bed≡surface，ETOPO 为裸地 DTM）；288 位全球网格"
            ),
            "vertical_datum": "EGM2008 geoid (EPSG:3855)",
            "visibility": entry.visibility,
        },
    )
