"""矢量通道（侧车导出器）：Hasterok GPRV 构造省一级分类。

源（磁盘实证）：
- plates&provinces/shp/global_gprv.shp：914 多边形（834 Polygon +
  80 MultiPolygon），EPSG:4326，穷尽覆盖（3′ 胞心栅格化覆盖率
  0.99991，未覆盖 241 胞 = 省界数字化缝隙）。一级分类 = prov_type
  （论文 Table 1: "dominant tectonic character"），数据实际 15 类
  （官方图例 gprv_types.qml 另含 oceanic plateau/unknown 两类，
  本版数据未出现）。
- 多边形重叠（磁盘实证）：Σ面积 − ∪面积 = 0.66 deg²（0.001%），
  均为省界毗邻处的数字化狭条，非嵌套层级。重叠处理（同 Slab2 惯例：
  显式确定性规则 + 构建期回报总量）：重叠像元取面积较小多边形
  （局部细化特征优先）。
- 极区/日期线（磁盘实证）：环上唯一 |Δlon|>180° 的边为北极盆闭合边
  （−180,90)→(180,90，标准极帽 cut 表示，平面栅格化正确）；无跨
  日期线边，平面胞心栅格化安全。

通道机制（三件之一/之二）：
- 多边形栅格化器 rasterize_polygon_classes：按主档网格胞心落入
  多边形赋类（all_touched=False），向粗档经 kernels.aggregate_mode
  众数聚合（nodata=0 不参与、平票取小、≥1 有效子胞即有值——与保守
  核「W>0 即有值」同语义，惯例）。后续矢量组（板块/陆地类别）
  复用。
- 整型类别层缺测语义：类别编码自 1 起，0 = 无类别（masks.data_missing
  唯一实现），伴生 validity 掩膜两通道一致。

穷尽覆盖型豁免距离场：本通道不产距离层。
许可：Zenodo 存档 10.5281/zenodo.6586972 CC BY 4.0（发布依据，
规避 GitHub 仓库 GPL-3.0 对数据的传染性争议——r1 许可审计）。
"""

import json
from pathlib import Path

import dask.array as da
import numpy as np
import xarray as xr

from .contract import LayerEntry
from .fidelity import INTEGRAL_TOL
from .grids import aggregation_factor, grid_shape, tier_centers, tier_chunks, tier_deg
from .kernels import aggregate_mode

REPO_ROOT = Path(__file__).resolve().parents[1]
GPRV_SRC_DIR = REPO_ROOT / "original data/lithosphere/tectonic-provinces"
GPRV_REL_PATH = "plates&provinces/shp/global_gprv.shp"

HASTEROK_DOI = "10.1016/j.earscirev.2022.104069"   # Hasterok et al. 2022, ESR 231, 104069
ZENODO_DOI = "10.5281/zenodo.6586972"              # CC BY 4.0 存档 v1（ESR 论文数据声明版本；发布许可依据）

HASTEROK_LAYER_ID = "lithosphere__hasterok_province_class"
HOME_TIER = "3min"

# 一级分类编码表（prov_type → uint8 码，1 起；0 = 无类别）。
# 词汇表 = 磁盘实测 15 类按字典序固定编码（确定性）；官方图例另有
# oceanic plateau / unknown 两类本版数据未出现，不占码位。
PROV_TYPE_CODES: dict[str, int] = {
    label: i for i, label in enumerate(sorted([
        "accretionary complex", "back-arc basin", "basin", "craton",
        "foredeep basin", "magmatic province", "narrow rift",
        "oceanic back-arc basin", "oceanic crust", "ophiolite complex",
        "orogenic belt", "passive margin", "shield", "volcanic arc",
        "wide rift",
    ]), start=1)
}

EXPECTED_ROWS = 914          # 已核查 + 源台账自证
NODATA = 0                   # 整型类别层缺测哨兵（编码自 1 起）


def category_encoding() -> dict[str, str]:
    """类别编码表（码 → 语义），attrs/manifest/sidecar 共用唯一来源。"""
    return {str(c): label for label, c in sorted(PROV_TYPE_CODES.items(), key=lambda kv: kv[1])}


# ---------------- 源读取 ----------------

def _iter_rings(geom):
    if geom.geom_type == "Polygon":
        yield geom.exterior
        yield from geom.interiors
    else:
        for p in geom.geoms:
            yield p.exterior
            yield from p.interiors


def load_gprv(src_dir: str | Path = GPRV_SRC_DIR):
    """读 global_gprv.shp → (GeoDataFrame, prov_type 码数组)。

    几何契约（磁盘实证的契约化）：914 行、EPSG:4326、prov_type 词汇
    ⊆ 编码表、无空几何；环边 |Δlon|>180° 仅允许极点闭合边（两端
    |lat|=90）——跨日期线边会使平面栅格化静默错位，读取期即拒绝。
    """
    import geopandas as gpd

    path = Path(src_dir) / GPRV_REL_PATH
    if not path.is_file():
        raise FileNotFoundError(f"GPRV 源文件不存在: {path}")
    gdf = gpd.read_file(path)
    if len(gdf) != EXPECTED_ROWS:
        raise ValueError(f"{GPRV_REL_PATH}: 行数 {len(gdf)} ≠ 期望 {EXPECTED_ROWS}")
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        raise ValueError(f"{GPRV_REL_PATH}: CRS {gdf.crs} ≠ EPSG:4326")
    if gdf.geometry.isna().any() or gdf.geometry.is_empty.any():
        raise ValueError(f"{GPRV_REL_PATH}: 存在空几何")

    unknown = sorted(set(gdf["prov_type"]) - set(PROV_TYPE_CODES))
    if unknown:
        raise ValueError(f"{GPRV_REL_PATH}: prov_type 超出编码表: {unknown}")

    for geom in gdf.geometry:
        for ring in _iter_rings(geom):
            xy = np.asarray(ring.coords)
            # 逐边检查全部 |Δlon|>180° 的边（非仅最大跳变）：每条都
            # 必须是极点闭合边（两端 |lat|=90），否则拒绝——一环内
            # 极点闭合边与较小跨日期线边并存时单查 argmax 会漏检
            for k in np.nonzero(np.abs(np.diff(xy[:, 0])) > 180.0)[0]:
                if not (abs(xy[k][1]) == 90.0 and abs(xy[k + 1][1]) == 90.0):
                    raise ValueError(
                        f"{GPRV_REL_PATH}: 环存在跨日期线边 "
                        f"{tuple(xy[k])}→{tuple(xy[k + 1])}，平面栅格化不安全"
                    )
    codes = gdf["prov_type"].map(PROV_TYPE_CODES).to_numpy(dtype=np.uint8)
    return gdf, codes


# ---------------- 多边形栅格化器（通道机制，后续矢量复用） ----------------

def rasterize_polygon_classes(geoms, codes: np.ndarray, tier: str) -> tuple[np.ndarray, int, int]:
    """类别多边形 → 主档类别网格（纯几何件，源无关）。

    语义：胞心落入多边形即属该类（all_touched=False，与档位网格
    中心偏移半档约定一致）；多边形重叠像元取面积较小者（局部细化
    优先，Slab2「重叠取最浅」同款显式确定性规则）；类外像元 = 0。

    返回 (classes uint8 (nlat, nlon), 重叠像元数, 未覆盖像元数)。
    逐多边形按包围盒开窗烧录（顺序 = 面积降序，小者后烧胜出），
    全程确定性可复跑。
    """
    import rasterio.features as rfeatures
    from rasterio.transform import from_origin

    nlat, nlon = grid_shape(tier)
    res = tier_deg(tier)
    classes = np.zeros((nlat, nlon), dtype=np.uint8)
    covered = np.zeros((nlat, nlon), dtype=np.uint8)   # 覆盖计数（重叠统计）

    # 面积降序（稳定排序）：后烧者胜 → 面积较小多边形赢重叠像元
    order = np.argsort([-g.area for g in geoms], kind="stable")
    for i in order:
        g = geoms[i]
        minx, miny, maxx, maxy = g.bounds
        # 行号自南起（行 0 = −90 侧，与档位网格约定一致）：行 r 跨
        # [−90+r·res, −90+(r+1)·res]；列 c 跨 [−180+c·res, −180+(c+1)·res]
        r0 = max(int(np.floor((miny + 90.0) / res)) - 1, 0)
        r1 = min(int(np.ceil((maxy + 90.0) / res)) + 1, nlat)     # exclusive
        c0 = max(int(np.floor((minx + 180.0) / res)) - 1, 0)
        c1 = min(int(np.ceil((maxx + 180.0) / res)) + 1, nlon)    # exclusive
        if r0 >= r1 or c0 >= c1:
            continue
        # 窗口北边 = 行 r1−1 的北缘；西边 = 列 c0 的西缘
        tr = from_origin(-180.0 + c0 * res, -90.0 + r1 * res, res, res)
        burned = rfeatures.rasterize(
            [(g, 1)], out_shape=(r1 - r0, c1 - c0), transform=tr, fill=0, dtype="uint8"
        ).astype(bool)
        # rasterio 窗口行序北起（local 0 = 行 r1−1），全局网格行 0 = −90
        # 侧——垂直翻转后落位（翻转缺陷由极帽/全网格直核比对测试覆盖）
        burned = burned[::-1]
        win = classes[r0:r1, c0:c1]
        classes[r0:r1, c0:c1] = np.where(burned, codes[i], win)
        covered[r0:r1, c0:c1] += burned

    overlap_cells = int((covered > 1).sum())
    uncovered_cells = int((covered == 0).sum())
    return classes, overlap_cells, uncovered_cells


def _aggregation_factor(tier: str) -> int:
    """主档 3′ → 目标档聚合因子（grids.aggregation_factor 薄包装，模块内惯用名）。"""
    return aggregation_factor(tier, HOME_TIER)


_HOME_CACHE: dict[str, tuple[np.ndarray, int, int]] = {}


def home_tier_classes(src_dir: str | Path = GPRV_SRC_DIR) -> tuple[np.ndarray, int, int]:
    """主档（3′）类别网格（进程内缓存：同层四档只栅格化一次）。"""
    key = f"hasterok:{Path(src_dir).resolve()}"
    cache = _HOME_CACHE.get(key)
    if cache is None:
        gdf, codes = load_gprv(src_dir)
        cache = rasterize_polygon_classes(
            list(gdf.geometry), codes, HOME_TIER
        )
        _HOME_CACHE[key] = cache
    return cache


RASTERIZATION_RULE = (
    "胞心落入多边形即属该类（all_touched=False，网格中心偏移半档约定）；"
    "多边形重叠像元取面积较小多边形（局部细化优先；源重叠 0.66 deg² = "
    "0.001%，省界数字化狭条）；粗档 = 主档 3′ 众数聚合（kernels."
    "aggregate_mode，nodata=0 不参与、平票取小、≥1 有效子胞即有值）"
)


# ---------------- 层构建 ----------------

def build_hasterok_province_class(entry: LayerEntry, tier: str) -> xr.DataArray:
    """构建该档 lithosphere__hasterok_province_class（uint8 类别层）。

    主档 3′ = 栅格化器直出；粗档 = 主档众数聚合（整数因子
    3′→6′/30′/1° = 2/10/20）。缺测 = 0（整型类别层约定），
    validity 伴生掩膜由构建入口按契约自动生成。
    """
    classes, _overlap, _uncovered = home_tier_classes()
    if tier == HOME_TIER:
        values = classes
    else:
        values = aggregate_mode(classes, _aggregation_factor(tier), nodata=NODATA)

    arr = da.from_array(values, chunks=tier_chunks(tier))
    lat, lon = tier_centers(tier)
    attrs = {
        "units": entry.unit,
        "long_name": "构造省一级分类（Hasterok et al. 2022 GPRV，prov_type）",
        "source": entry.source,
        "doi": HASTEROK_DOI,
        "license_note": f"发布依据 Zenodo {ZENODO_DOI}（CC BY 4.0）",
        "native_res": entry.native_res,
        "resampling": entry.resampling,
        "visibility": entry.visibility,
        "category_encoding": json.dumps(
            category_encoding(), ensure_ascii=False, sort_keys=True
        ),
        "rasterization_rule": RASTERIZATION_RULE,
        "source_file": GPRV_REL_PATH,
        # 注意：不可命名 missing_value/_FillValue——xarray zarr 后端按 CF
        # 约定将其当 fill 编码，读回时 0 被掩为 NaN 且 dtype 升 float32
        "nodata_semantics": "0 = 无类别（省界数字化缝隙），类别编码自 1 起",
    }
    return xr.DataArray(
        arr, dims=("lat", "lon"), coords={"lat": lat, "lon": lon},
        name=entry.id, attrs=attrs,
    )


# ---------------- 保真验证（本层自供） ----------------

VECTOR_FIDELITY_LAYERS = {
    HASTEROK_LAYER_ID: {"src_dir": GPRV_SRC_DIR, "unit": "category"},
}


def build_hasterok_fidelity_report(store_dir, layer_id: str, tiers: list[str],
                                   src_dir=None) -> dict:
    """保真报告（类别层判据）：

    ① store 与源重算逐位一致（硬判据，位级——重算 = 栅格化器 + 众数核
       同码路径，捕获写入/存储链路错位）；
    ② 类别众数一致性（硬判据，验收项）：粗档 store 值 == 主档
       store 值的众数聚合——跨档一致性以 store 自身为源，独立于 ①；
    ③ 值域 ⊆ {0} ∪ 编码表（硬判据，编码错误捕获器）；
    ④ 类别分布/覆盖率/重叠与未覆盖总量（报告制）。
    """
    src_dir = src_dir or GPRV_SRC_DIR
    classes, overlap_cells, uncovered_cells = home_tier_classes(src_dir)

    checks: list[dict] = []
    values: dict[str, np.ndarray] = {}    # 逐档 store 值（单遍读取，三组判据复用）
    with xr.open_datatree(store_dir, engine="zarr", chunks={}) as tree:
        for tier in tiers:
            node = tree[f"/{tier}"]
            if layer_id not in node.ds.data_vars:
                raise KeyError(f"store {store_dir} 的 {tier} 档不含 {layer_id}（先 build 再 fidelity）")
            v = node.ds[layer_id].values
            values[tier] = v

            # ① 逐位一致（硬判据）
            expect = classes if tier == HOME_TIER else aggregate_mode(
                classes, _aggregation_factor(tier), nodata=NODATA
            )
            n_bad = int((v != expect).sum())
            checks.append({
                "name": f"{tier}: 与源栅格化重算逐位一致（uint8 位级）",
                "status": "pass" if n_bad == 0 else "FAIL",
                "summary": f"不一致像元 {n_bad}（期望 0）",
            })

        # ② 类别众数一致性（硬判据）：粗档 == 主档 store 值的众数聚合
        # （以 store 自身为源，独立于 ① 的跨档检验）
        for tier in tiers:
            if tier == HOME_TIER:
                continue
            expect = aggregate_mode(
                values[HOME_TIER], _aggregation_factor(tier), nodata=NODATA
            )
            n_bad = int((values[tier] != expect).sum())
            checks.append({
                "name": f"{tier}: 类别众数一致性（= 主档 {HOME_TIER} store 值众数聚合）",
                "status": "pass" if n_bad == 0 else "FAIL",
                "summary": f"不一致像元 {n_bad}（期望 0）",
            })

        # ③ 值域（硬判据）+ ④ 类别分布/覆盖率（报告制）
        valid_codes = np.array(sorted(PROV_TYPE_CODES.values()))
        for tier in tiers:
            v = values[tier]
            bad = int((~np.isin(v, np.append(valid_codes, NODATA))).sum())
            checks.append({
                "name": f"{tier}: 值域 ⊆ {{0}} ∪ 编码表（1..{len(valid_codes)}）",
                "status": "pass" if bad == 0 else "FAIL",
                "summary": f"非法值像元 {bad}（期望 0）",
            })
            counts = np.bincount(v.ravel(), minlength=len(valid_codes) + 1)
            dist = ", ".join(
                f"{label}={counts[c]}" for label, c in
                sorted(PROV_TYPE_CODES.items(), key=lambda kv: kv[1])
            )
            checks.append({
                "name": f"{tier}: 类别分布（报告制）",
                "status": "report",
                "summary": f"覆盖 {counts[1:].sum()}/{v.size}；{dist}",
            })

    nlat, nlon = grid_shape(HOME_TIER)
    checks.append({
        "name": f"{HOME_TIER}: 重叠/未覆盖总量（报告制）",
        "status": "report",
        "summary": (
            f"多边形重叠像元 {overlap_cells}（规则：面积较小者胜）；"
            f"未覆盖像元 {uncovered_cells}/{nlat * nlon}"
            f"（省界数字化缝隙，validity=0）"
        ),
    })

    n_fail = sum(1 for c in checks if c["status"] == "FAIL")
    return {
        "layer": layer_id,
        "store": str(store_dir),
        "integral_tolerance": INTEGRAL_TOL,   # 类别层无积分量；保留报告器统一字段
        "result": "FAIL" if n_fail else "PASS",
        "n_checks": len(checks),
        "checks": checks,
    }
