"""Sediment group: GST1 global sediment thickness + Global Basins dual
channel (internal).

Source (verified on disk):

- GST14_WGS84.XYZ (Bird & Mooney 2026 GST-1, downloaded from birdgeo.com): an
  XYZ point grid of 4151521 data rows = 1441 latitude × 2881 longitude (a
  globally complete grid registered at 0.125° gridline nodes, no missing
  rows—4151521 = 1441×2881 holds exactly). Column-order pitfall (verified on
  disk): the header says "Lon Lat meter",
  but the actual data column order is Lat, Lon, meter (col1 is a constant
  latitude increasing from -90 in steps of 0.125; col2 loops longitude
  -180..180 in steps of 0.125)—take columns by position, never by header
  name; first row measured (-90, -180). The ±180 columns are bit-identical (a
  periodic duplicate copy; drop the +180 column, same as WGM2012); the ±90
  pole rows are constant within the row (a single pole point). Value range
  0..20118 m, no NaN—0 = no sediment is a real value (3.26% of cells), not
  missing. Node→Voronoi cell interpretation: cell = node ±1/16°, pole-row
  latitude clamped to half weight,
  longitude wrapped at the date line.
- evenick2021_global_basins.shp (Evenick 2021 ESR 215:103564): 768 Polygons =
  764 basins (unique Basin UBI) + 4 date-line-crossing basins (Ross / Anadyr /
  Hope / Khatyrka) each split into 2 parts (Split Poly=Yes, the two parts fall
  on either side of the date line, no crossed edge within a part); the
  split-part statistics columns (Max/Mean/Med Sed, Moho, Poly Area) are
  per-part statistics, not basin-level (verified on disk: the two parts of the
  same UBI differ in value)—the sidecar keeps the 768-row source granularity,
  while the build burns the 768-part geometry (same-UBI parts have consistent
  Basin Type, contract-ized).
  EPSG:4326; Basin Type vocabulary of 8 classes (lexicographic encoding 1..8);
  1 invalid geometry
  (Komandorskaya self-intersection, make_valid repair = main component +
  0.0002 deg² sliver,
  area Δ=3.5e-4 deg²); 3 overlapping basin pairs (max 0.0104 deg², at the
  magnitude of a shared digitized boundary); total basin area 16451 deg²
  (25.39% of the globe).

Channels:
- gst1_thickness (the cube's only sediment-thickness layer, kept by the
  restraint principle; the GlobSed /
  CRUST1.0 sediment layers are relegated to validation reference): two tiers
  1°/30′ conservative-kernel downsampling
  (conservative_overlap_mean; 6′, being finer than the native 0.125°, is
  forbidden by hard rule); full source coverage with no NaN → validity is
  constantly 1.
- global_basins_class (internal): home tier 3′ cell-center rasterization
  (burning in descending area order after the make_valid repair) + coarse
  tiers kernels.aggregate_mode majority aggregation (nodata=0 excluded, ties
  take the smaller code, ≥1 valid subcell means the cell has a value).
- global_basins_distance (internal): region-distance semantics—a cell center
  inside any basin polygon → 0; otherwise = spherical great-circle distance to
  the nearest basin boundary segment (km, IUGG
  authalic R, exact per-tier recomputation with hierarchical ring-band
  candidate pruning). The in-domain 0 overwrite
  is done within the distance pyramid cache (per-tier inside masks rasterized
  directly from the vector source, not downsampled across tiers). All four
  tiers have values everywhere (mask-free).

License:
- GST1: source (downloaded from birdgeo.com) has no explicit license, not
  included in the public edition.
- Global Basins: source license CC BY-NC-ND 4.0 (the ND clause restricts
  derivative redistribution) → both
  layers + sidecar visibility=internal, not included in the public edition; the
  paper advises readers to directly
  use the original source (Evenick 2021 supplementary material).
"""

import json
import os
from fractions import Fraction
from pathlib import Path

import dask.array as da
import numpy as np
import xarray as xr

from .contract import LayerEntry
from .faults import distance_grids, independent_min_distance_km, segment_table
from .fidelity import INTEGRAL_TOL, fidelity_checks, hist_add
from .grids import aggregation_factor, tier_centers, tier_chunks, tier_fraction
from .gravmag import RasterSource, gravmag_source_stats
from .kernels import aggregate_mode, conservative_overlap_mean
from .pixel_area import EARTH_RADIUS_KM
from .vector import _iter_rings, rasterize_polygon_classes

REPO_ROOT = Path(__file__).resolve().parents[1]
GST1_SRC_DIR = REPO_ROOT / "original data/sediment/gst1/GST1_WGS84"
XYZ_NAME = "GST14_WGS84.XYZ"
BASINS_SRC_DIR = REPO_ROOT / "original data/sediment/global-basins/shapefile"
BASINS_SHP_NAME = "evenick2021_global_basins.shp"

GST1_DOI = "10.1016/j.tecto.2026.231175"
EVENICK2021_DOI = "10.1016/j.earscirev.2021.103564"
GST1_URL = "http://birdgeo.com/gst-1.htm"

GST1_LAYER_ID = "sediment__gst1_thickness"
BASIN_CLASS_LAYER_ID = "sediment__global_basins_class"
BASIN_DISTANCE_LAYER_ID = "sediment__global_basins_distance"

GST1_TIERS = ("1deg", "30min")
BASINS_HOME_TIER = "3min"
BASINS_DISTANCE_CHAIN = ("1deg", "30min", "6min", "3min")

GST1_EXPECTED = {"shape": (1441, 2881), "spacing": Fraction(1, 8)}

BASIN_TYPE_CODES: dict[str, int] = {
    label: i for i, label in enumerate(sorted([
        "Backarc - Marginal Sea", "Fold and Thrust Belt", "Forearc",
        "Foreland", "Intracratonic", "Passive Margin", "Rift", "Strike-Slip",
    ]), start=1)
}

EXPECTED_POLYGONS = 768
EXPECTED_BASINS = 764
EXPECTED_SPLIT_BASINS = 4
EXPECTED_INVALID = 1
NODATA = 0

GST1_LICENSE_NOTE = (
    "GST-1 源（birdgeo.com 下载）无明示许可；不纳入公开版。"
    f"引用 Bird & Mooney (2026) doi:{GST1_DOI}"
)

BASINS_LICENSE_NOTE = (
    "Evenick (2021) 论文 CC BY-NC-ND 4.0（作者保留版权，ND 条款限制衍生"
    "再分发）——不纳入公开版；论文建议读者直接联用原源"
    f"（doi:{EVENICK2021_DOI}）"
)

GST1_UNIT_BASIS = (
    "源文件首行表头 'Lon      Lat    meter' 第 3 列名 meter（沉积厚度单位 "
    "m）；表头列名序与数据列序不符（数据 col1=Lat col2=Lon，磁盘实证），"
    "单位取第 3 列名 meter，与值域 0..20118 相符"
)


def basin_type_encoding() -> dict[str, str]:
    """Basin-type encoding table (code → semantics), the single source shared
    by attrs/manifest/sidecar."""
    return {
        str(c): label
        for label, c in sorted(BASIN_TYPE_CODES.items(), key=lambda kv: kv[1])
    }



def load_gst1(src_dir: str | Path = GST1_SRC_DIR) -> RasterSource:
    """Read GST14_WGS84.XYZ → normalized source (node→Voronoi cell
    interpretation).

    Contract (contract-ized disk verification): header token = "Lon Lat meter"
    (the column-order pitfall carrier—the data order is Lat, Lon, meter; take
    columns by position); data row count = 1441×2881; first row (lat, lon) =
    (-90, -180); latitude constant within a row, stepping 0.125 between rows
    from -90; longitude within a row -180..180 in steps of 0.125; the ±180
    columns are bit-identical (periodic duplicate); the ±90 pole rows are
    constant within the row. Any source change requires re-reviewing the parse.
    """
    import pandas as pd

    path = Path(src_dir) / XYZ_NAME
    if not path.is_file():
        raise FileNotFoundError(f"GST1 源文件不存在: {path}")
    with open(path, encoding="ascii") as f:
        header_tokens = f.readline().split()
    if header_tokens != ["Lon", "Lat", "meter"]:
        raise ValueError(
            f"{XYZ_NAME}: 表头 {header_tokens} ≠ ['Lon', 'Lat', 'meter']"
            "（源变更须复评；注意表头列名序与数据列序不符——按位置取列）"
        )

    nlat, nlon = GST1_EXPECTED["shape"]
    spacing = GST1_EXPECTED["spacing"]
    df = pd.read_csv(
        path, sep=r"\s+", skiprows=1, header=None,
        names=["col1", "col2", "col3"], dtype=np.float64,
    )
    if len(df) != nlat * nlon:
        raise ValueError(
            f"{XYZ_NAME}: 数据行数 {len(df)} ≠ {nlat}×{nlon}（源变更须复评）"
        )
    lat = df["col1"].to_numpy()
    lon = df["col2"].to_numpy()
    g = df["col3"].to_numpy().reshape(nlat, nlon)
    lat_g = lat.reshape(nlat, nlon)
    lon_g = lon.reshape(nlat, nlon)
    if not (lat_g[0, 0] == -90.0 and lon_g[0, 0] == -180.0):
        raise ValueError(
            f"{XYZ_NAME}: 首行 ({lat_g[0, 0]}, {lon_g[0, 0]}) ≠ 预期 (-90, -180)"
            "（列序或网格变更须复评）"
        )
    if not np.all(lat_g == lat_g[:, :1]):
        raise ValueError(f"{XYZ_NAME}: 行内纬度非常值（列序假设被破坏）")
    expect_lat = -90.0 + np.arange(nlat) * float(spacing)
    if np.max(np.abs(lat_g[:, 0] - expect_lat)) > 1e-9:
        raise ValueError(f"{XYZ_NAME}: 纬度坐标偏离规则网格（首末 {lat_g[0, 0]}, {lat_g[-1, 0]}）")
    expect_lon = -180.0 + np.arange(nlon) * float(spacing)
    if not np.array_equal(lon_g, np.broadcast_to(lon_g[0], lon_g.shape)):
        raise ValueError(f"{XYZ_NAME}: 行内经度模式不一致（内循环假设被破坏）")
    if np.max(np.abs(lon_g[0] - expect_lon)) > 1e-9:
        raise ValueError(f"{XYZ_NAME}: 经度坐标偏离规则网格（首末 {lon_g[0, 0]}, {lon_g[0, -1]}）")
    if not np.array_equal(g[:, 0], g[:, -1]):
        raise ValueError(f"{XYZ_NAME}: ±180 列不同值，周期重复假设被破坏")
    if np.unique(g[0]).size != 1 or np.unique(g[-1]).size != 1:
        raise ValueError(f"{XYZ_NAME}: 极点行行内非常值（极点单点假设被破坏）")

    values = g[:, :-1].astype(np.float32)
    valid = np.isfinite(values)
    lat_edges = -90.0 + (np.arange(nlat + 1, dtype=np.float64) - 0.5) * float(spacing)
    lat_edges[0], lat_edges[-1] = -90.0, 90.0
    return RasterSource(
        values=values,
        valid=valid,
        lat_edges=lat_edges,
        lon_res=spacing,
        attrs={
            "registration_rule": (
                "gridline 节点注册（节点 = 整倍 0.125° 点采样，含 ±90/±180 "
                "端点）→ 节点代表 Voronoi 胞（中心 ±1/16°，经度环接；极点行"
                "纬度钳制半权）；±180 周期重复列弃除；列序陷阱：表头写 "
                "Lon Lat meter、数据实为 Lat Lon meter，按位置取列"
            ),
            "source_file": XYZ_NAME,
            "unit_basis": GST1_UNIT_BASIS,
        },
    )



_SOURCE_CACHE: dict[str, RasterSource] = {}


def _cached_gst1(src_dir: str | Path = GST1_SRC_DIR) -> RasterSource:
    """In-process source cache: the two tiers read the ~116MB source file only
    once."""
    key = f"gst1:{Path(src_dir).resolve()}"
    if key not in _SOURCE_CACHE:
        _SOURCE_CACHE[key] = load_gst1(src_dir)
    return _SOURCE_CACHE[key]


def build_gst1_thickness(entry: LayerEntry, tier: str) -> xr.DataArray:
    """Build this tier's sediment__gst1_thickness (the cube's only
    sediment-thickness layer, float32)."""
    src = _cached_gst1()
    values, _W = conservative_overlap_mean(
        src.values, src.valid, src.lat_edges, src.lon_res, tier_fraction(tier),
        lon_phase=src.lon_phase,
    )
    arr = da.from_array(values, chunks=tier_chunks(tier))
    lat, lon = tier_centers(tier)
    attrs = {
        "units": entry.unit,
        "long_name": "全球沉积总厚度（Bird & Mooney 2026 GST-1，立方唯一沉积厚度层）",
        "source": entry.source,
        "doi": GST1_DOI,
        "source_url": GST1_URL,
        "license_note": GST1_LICENSE_NOTE,
        "native_res": entry.native_res,
        "resampling": entry.resampling,
        "visibility": entry.visibility,
        "earth_radius_km": EARTH_RADIUS_KM,
        "unit_basis": GST1_UNIT_BASIS,
        "nodata_semantics": (
            "无缺测：源全网格有值（0 = 无沉积为真实值，占 3.26%，非缺测），"
            "validity 恒 1；GlobSed/CRUST1.0 沉积层转验证参考（克制原则"
            "筛选留一，层映射表 validation_reference）"
        ),
    }
    attrs.update(src.attrs)
    return xr.DataArray(
        arr, dims=("lat", "lon"), coords={"lat": lat, "lon": lon},
        name=entry.id, attrs=attrs,
    )



def _polygonal_only(geom):
    """Take the areal component of a make_valid result (a GeometryCollection
    drops zero-area line/point remnants).

    Same as pb2002._polygonal_only (copied within the module to avoid
    cross-topic coupling).
    """
    from shapely.ops import unary_union

    if geom.geom_type in ("Polygon", "MultiPolygon"):
        return geom
    parts = [g for g in geom.geoms if g.geom_type in ("Polygon", "MultiPolygon")]
    if not parts:
        raise ValueError("make_valid 后无面状组分——源几何异常")
    return unary_union(parts)


def load_global_basins(src_dir: str | Path = BASINS_SRC_DIR):
    """Read evenick2021_global_basins.shp → (GeoDataFrame, type-code array,
    repaired-geometry list).

    Contract (contract-ized disk verification): 768 rows, EPSG:4326, all
    Polygon, no empty geometry; Basin Type vocabulary ⊆ the 8-class encoding
    table; 764 unique Basin UBI, exactly 4 basins each with 2 split parts and
    consistent type across same-UBI parts (the correctness premise of burning
    per part); ring-edge |Δlon|>180° rejected (the source handles the date line
    by splitting, 0 such edges verified); exactly 1 invalid geometry
    (Komandorskaya self-intersection) repaired to its areal component via
    make_valid. Any source change requires re-reviewing.
    """
    import geopandas as gpd
    from shapely.validation import make_valid

    path = Path(src_dir) / BASINS_SHP_NAME
    if not path.is_file():
        raise FileNotFoundError(f"Global Basins 源文件不存在: {path}")
    gdf = gpd.read_file(path)
    if len(gdf) != EXPECTED_POLYGONS:
        raise ValueError(f"{BASINS_SHP_NAME}: 行数 {len(gdf)} ≠ 期望 {EXPECTED_POLYGONS}")
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        raise ValueError(f"{BASINS_SHP_NAME}: CRS {gdf.crs} ≠ EPSG:4326")
    if gdf.geometry.isna().any() or gdf.geometry.is_empty.any():
        raise ValueError(f"{BASINS_SHP_NAME}: 存在空几何")
    unknown = sorted(set(gdf["Basin Type"]) - set(BASIN_TYPE_CODES))
    if unknown:
        raise ValueError(f"{BASINS_SHP_NAME}: Basin Type 超出编码表: {unknown}")
    n_ubi = gdf["Basin UBI"].nunique()
    if n_ubi != EXPECTED_BASINS:
        raise ValueError(f"{BASINS_SHP_NAME}: 唯一盆地数 {n_ubi} ≠ 期望 {EXPECTED_BASINS}")

    split = [ubi for ubi, grp in gdf.groupby("Basin UBI") if len(grp) > 1]
    if len(split) != EXPECTED_SPLIT_BASINS or any(
        len(gdf[gdf["Basin UBI"] == ubi]) != 2 for ubi in split
    ):
        raise ValueError(
            f"{BASINS_SHP_NAME}: 拆分片盆地形态异常（{len(split)} 个，"
            "期望 4 个 × 各 2 片）"
        )
    for ubi in split:
        grp = gdf[gdf["Basin UBI"] == ubi]
        if grp["Basin Type"].nunique() != 1:
            raise ValueError(f"{BASINS_SHP_NAME}: 盆地 {ubi} 拆分片 Basin Type 不一致")

    for geom in gdf.geometry:
        for ring in _iter_rings(geom):
            xy = np.asarray(ring.coords)
            if np.abs(np.diff(xy[:, 0])).max() > 180.0:
                raise ValueError(
                    f"{BASINS_SHP_NAME}: 环存在跨日期线边（源拆片假设被破坏，"
                    "平面栅格化不安全）"
                )

    invalid_mask = ~gdf.geometry.is_valid
    n_invalid = int(invalid_mask.sum())
    if n_invalid != EXPECTED_INVALID:
        raise ValueError(
            f"{BASINS_SHP_NAME}: 无效几何数 {n_invalid} ≠ 期望 {EXPECTED_INVALID}"
            "（源变更须复评修复策略）"
        )
    geoms = [
        g if ok else _polygonal_only(make_valid(g))
        for g, ok in zip(gdf.geometry, ~invalid_mask)
    ]
    if any(g.is_empty or not g.is_valid for g in geoms):
        raise ValueError(f"{BASINS_SHP_NAME}: 修复后存在空/无效几何")

    codes = gdf["Basin Type"].map(BASIN_TYPE_CODES).to_numpy(dtype=np.uint8)
    return gdf, codes, geoms



_GEOMS_CACHE: dict[str, tuple] = {}
_CLASS_HOME_CACHE: dict[str, tuple] = {}
_CLASS_MODE_CACHE: dict[tuple, np.ndarray] = {}
_INSIDE_CACHE: dict[tuple, np.ndarray] = {}


def _basins_geoms(src_dir: str | Path = BASINS_SRC_DIR):
    """Repaired geometries + type codes (in-process cache)."""
    key = f"basins:{Path(src_dir).resolve()}"
    cache = _GEOMS_CACHE.get(key)
    if cache is None:
        _gdf, codes, geoms = load_global_basins(src_dir)
        cache = (geoms, codes)
        _GEOMS_CACHE[key] = cache
    return cache


def _home_tier_classes(src_dir: str | Path = BASINS_SRC_DIR):
    """Home tier 3′ class grid + overlap/uncovered counts (in-process cache)."""
    key = f"basins:{Path(src_dir).resolve()}"
    cache = _CLASS_HOME_CACHE.get(key)
    if cache is None:
        geoms, codes = _basins_geoms(src_dir)
        cache = rasterize_polygon_classes(geoms, codes, BASINS_HOME_TIER)
        _CLASS_HOME_CACHE[key] = cache
    return cache


def _tier_classes(tier: str, src_dir: str | Path = BASINS_SRC_DIR) -> np.ndarray:
    if tier == BASINS_HOME_TIER:
        return _home_tier_classes(src_dir)[0]
    key = (tier, str(Path(src_dir).resolve()))
    cache = _CLASS_MODE_CACHE.get(key)
    if cache is None:
        home = _home_tier_classes(src_dir)[0]
        cache = aggregate_mode(
            home, aggregation_factor(tier, BASINS_HOME_TIER), nodata=NODATA
        )
        _CLASS_MODE_CACHE[key] = cache
    return cache


BASIN_RASTERIZATION_RULE = (
    "768 源多边形（4 跨日期线盆地各 2 拆分片，片内无跨界边；Komandorskaya "
    "自交 make_valid 修复取面状组分）胞心栅格化（all_touched=False；"
    "面积降序后烧胜——重叠像元取面积较小片；实测盆地间重叠 3 对 "
    "≤0.0104 deg²）；粗档 = 主档 3′ 众数聚合（kernels.aggregate_mode，"
    "nodata=0 不参与、平票取小、≥1 有效子胞即有值）；同 UBI 拆分片 "
    "Basin Type 一致（读取期契约），按片烧录 ≡ 按盆地烧录"
)


def build_global_basins_class(entry: LayerEntry, tier: str) -> xr.DataArray:
    """Build this tier's sediment__global_basins_class (uint8 class layer,
    internal)."""
    values = _tier_classes(tier)
    lat, lon = tier_centers(tier)
    attrs = {
        "units": entry.unit,
        "long_name": "沉积盆地类型（Evenick 2021，764 盆地 8 类，像元内主导类型）",
        "source": entry.source,
        "doi": EVENICK2021_DOI,
        "license_note": BASINS_LICENSE_NOTE,
        "native_res": entry.native_res,
        "resampling": entry.resampling,
        "visibility": entry.visibility,
        "category_encoding": json.dumps(
            basin_type_encoding(), ensure_ascii=False, sort_keys=True
        ),
        "rasterization_rule": BASIN_RASTERIZATION_RULE,
        "source_file": BASINS_SHP_NAME,
        "nodata_semantics": (
            "0 = 无盆地（类别编码自 1 起，盆地外像元；盆地总面积占全球 "
            "25.39%）；validity = 类别非 0（契约 mask: validity 伴生）"
        ),
    }
    arr = da.from_array(values, chunks=tier_chunks(tier))
    return xr.DataArray(
        arr, dims=("lat", "lon"), coords={"lat": lat, "lon": lon},
        name=entry.id, attrs=attrs,
    )



_PYRAMID_CACHE_DIR = REPO_ROOT / "products/.build/global-basins-distance"

_PYRAMID_CACHE: dict[str, dict[str, np.ndarray]] = {}


def _tier_inside(tier: str, src_dir: str | Path = BASINS_SRC_DIR) -> np.ndarray:
    """This tier's inside mask (cell center falls in any basin part;
    rasterized directly from the vector source, exact per tier)."""
    key = (tier, str(Path(src_dir).resolve()))
    cache = _INSIDE_CACHE.get(key)
    if cache is None:
        geoms, _codes = _basins_geoms(src_dir)
        ones = np.ones(len(geoms), dtype=np.uint8)
        cache = rasterize_polygon_classes(geoms, ones, tier)[0] != 0
        _INSIDE_CACHE[key] = cache
    return cache


def _basin_rings(geoms):
    """Repaired geometries → list of boundary rings (the single source shared by
    the distance segment table and fidelity)."""
    return [ring for g in geoms for ring in _iter_rings(g)]


def _distance_pyramid(src_dir: str | Path = BASINS_SRC_DIR) -> dict[str, np.ndarray]:
    """Basin boundary ring segment table → {tier: region-distance float32 global
    grid} (in-process cache).

    The distance engine recomputes boundary distances exactly per tier from the
    vector source (hierarchical ring-band candidate pruning; a lemma guarantees
    no missed candidates); then per tier the inside mask (that tier's cell
    centers rasterized directly from the vector source) overwrites in-domain
    cells to 0—region-distance semantics (distance to the nearest basin, in-domain
    = 0).

    Cache semantics (same as the faults channel): the .npy is a memory guard
    (mmap does not reside in RSS), not a cross-process result cache—
    distance_grids recomputes and rewrites on every call; the real reuse is the
    in-process _PYRAMID_CACHE dict (the four tiers are computed only once,
    shared by build/fidelity in the same process). The overwrite is done in
    place with r+ on the mmap, then reloaded read-only (the overwrite is
    idempotent: 0 → 0), so build-product determinism is unaffected.
    """
    key = f"basins:{Path(src_dir).resolve()}"
    if key not in _PYRAMID_CACHE:
        geoms, _codes = _basins_geoms(src_dir)
        segs, _n_drop = segment_table(_basin_rings(geoms))
        distance_grids(
            segs, BASINS_DISTANCE_CHAIN,
            workers=max(1, os.cpu_count() or 1),
            cache_dir=_PYRAMID_CACHE_DIR,
        )
        out: dict[str, np.ndarray] = {}
        for tier in BASINS_DISTANCE_CHAIN:
            fpath = Path(_PYRAMID_CACHE_DIR) / f"{tier}.npy"
            a = np.load(fpath, mmap_mode="r+")
            a[_tier_inside(tier, src_dir)] = 0.0
            a.flush()
            out[tier] = np.load(fpath, mmap_mode="r")
        _PYRAMID_CACHE[key] = out
    return _PYRAMID_CACHE[key]


BASIN_VALUE_CONVENTION = (
    "像元中心到最近盆地（Evenick 2021 多边形并集）的球面大圆距离（km，"
    "IUGG authalic R=6371.0071810 km，与像元面积层同球）：域内（胞心落入"
    "任一盆地片）= 0，域外 = 到最近盆地边界段的球面大圆距离；逐档从矢量源"
    "精确重算（非跨档降采样）——边界距离候选剪枝用层级环带方案（父档中心"
    "距离环带 ±2×胞半对角线包含子档全体像元的 argmin，数学引理保证不漏"
    "候选），域内 0 由该档 inside 掩膜（胞心直接自矢量源栅格化）覆写；"
    "保真报告以独立航空公式 + 独立点面包含检验抽样全段暴力比对复核"
)


def build_global_basins_distance(entry: LayerEntry, tier: str) -> xr.DataArray:
    """Build this tier's sediment__global_basins_distance (float32, values
    everywhere, internal)."""
    values = _distance_pyramid()[tier]
    lat, lon = tier_centers(tier)
    attrs = {
        "units": entry.unit,
        "long_name": "到最近沉积盆地（Evenick 2021）的球面大圆距离（域内 = 0）",
        "source": entry.source,
        "doi": EVENICK2021_DOI,
        "license_note": BASINS_LICENSE_NOTE,
        "native_res": entry.native_res,
        "resampling": entry.resampling,
        "visibility": entry.visibility,
        "value_convention": BASIN_VALUE_CONVENTION,
        "source_file": BASINS_SHP_NAME,
        "nodata_semantics": "无缺测：距离场全域有值（域内 0 / 域外有限正值；免掩膜）",
    }
    arr = da.from_array(values, chunks=tier_chunks(tier))
    return xr.DataArray(
        arr, dims=("lat", "lon"), coords={"lat": lat, "lon": lon},
        name=entry.id, attrs=attrs,
    )



_GST1_HIST_SPEC = (0.0, 21000.0, 100.0)

SEDIMENT_FIDELITY_LAYERS = {
    GST1_LAYER_ID: {"kind": "raster"},
    BASIN_CLASS_LAYER_ID: {"kind": "class"},
    BASIN_DISTANCE_LAYER_ID: {"kind": "distance"},
}

_SAMPLE_SEED = 20260913
_N_SAMPLES = 120
_DIST_TOL_KM = 1e-3
_DIST_REL_TOL = 3e-7


def _independent_region_distance_km(lat: float, lon: float, segs, tree, geoms) -> float:
    """Independent implementation: shapely point-in-polygon (in-domain → 0) +
    haversine formula brute force over all segments (out-of-domain).

    Deliberately heterogeneous from the main implementation (rasterization vs
    vector containment test; 3D vector engine vs haversine
    formula), for fidelity cross-validation.
    """
    from shapely.geometry import Point

    p = Point(lon, lat)
    for cand in tree.query(p):
        if geoms[int(cand)].contains(p):
            return 0.0
    return independent_min_distance_km(lat, lon, segs)


def build_sediment_fidelity_report(store_dir, layer_id: str, tiers: list[str]) -> dict:
    """Fidelity report.

    gst1_thickness (isomorphic to the gravity-magnetics group): ① store
    bit-for-bit consistent with the recomputation from the source kernel
    (float32 bit-level, NaN-aware); ② valid flag ⟺ source-mask recomputation
    W>0; ③ integral identity Σ v·W = Σ v·w (source statistics computed
    independently, reusing gravmag_source_stats) + quantiles
    (fidelity_checks); ④ coverage (report-only).

    global_basins_class (isomorphic to plate_class): ① store bit-for-bit
    consistent with the source rasterization/majority recomputation; ② coarse
    tier = majority aggregation of home-tier store values; ③ value range ⊆
    encoding table ∪ {0}; ④ class distribution + coverage (report-only; basins
    cover 25.39% of the globe, nonexhaustive coverage).

    global_basins_distance (channel-isomorphic + in-domain semantics): ⑤ finite
    and ≥ 0 everywhere; ⑥ in-domain ⟹ 0 (inside-mask recomputation, hard
    criterion); ⑦ stratified random sampling (uniform + near-field <50 km +
    in-domain) compared against the independent implementation (shapely
    containment + haversine brute force over all segments), tol = max(1e-3 km,
    3e-7·|d|) (hard criterion); ⑧ distance distribution (report-only).
    """
    kind = SEDIMENT_FIDELITY_LAYERS[layer_id]["kind"]
    checks: list[dict] = []

    if kind == "raster":
        src = _cached_gst1()
        src_stats = gravmag_source_stats(src, _GST1_HIST_SPEC)
        with xr.open_datatree(store_dir, engine="zarr", chunks={}) as tree:
            for tier in tiers:
                node = tree[f"/{tier}"]
                if layer_id not in node.ds.data_vars:
                    raise KeyError(
                        f"store {store_dir} 的 {tier} 档不含 {layer_id}（先 build 再 fidelity）"
                    )
                v = node.ds[layer_id].values
                expect, W = conservative_overlap_mean(
                    src.values, src.valid, src.lat_edges, src.lon_res,
                    tier_fraction(tier), lon_phase=src.lon_phase,
                )
                ok = W > 0

                eq = (v == expect) | (np.isnan(v) & np.isnan(expect))
                n_bad = int((~eq).sum())
                checks.append({
                    "name": f"{tier}: 与源核重算逐位一致（float32 位级）",
                    "status": "pass" if n_bad == 0 else "FAIL",
                    "summary": f"不一致像元 {n_bad}（期望 0）",
                })
                mism = int((np.isfinite(v) != ok).sum())
                checks.append({
                    "name": f"{tier}: 有效位与源掩膜复算一致（逐位）",
                    "status": "pass" if mism == 0 else "FAIL",
                    "summary": f"不一致 {mism} 像元（期望 0）",
                })

                tier_stats = {
                    "sum_va": float(np.dot(v[ok].astype(np.float64), W[ok])),
                }
                lo, hi, bw = _GST1_HIST_SPEC
                tier_stats["hist"] = hist_add(
                    np.zeros(int(round((hi - lo) / bw)), dtype=np.uint64),
                    v[np.isfinite(v)], lo=lo, hi=hi, bin_width=bw,
                )
                checks.extend(fidelity_checks(
                    src_stats, tier_stats, tier,
                    hist_spec=_GST1_HIST_SPEC, unit="m",
                ))
                checks.append({
                    "name": f"{tier}: 覆盖率（报告制）",
                    "status": "report",
                    "summary": f"档 {ok.mean():.4f} vs 源（像元计）{src.valid.mean():.4f}",
                })

    elif kind == "class":
        valid_codes = np.array(sorted(BASIN_TYPE_CODES.values()))
        values: dict[str, np.ndarray] = {}
        with xr.open_datatree(store_dir, engine="zarr", chunks={}) as tree:
            for tier in tiers:
                node = tree[f"/{tier}"]
                if layer_id not in node.ds.data_vars:
                    raise KeyError(
                        f"store {store_dir} 的 {tier} 档不含 {layer_id}（先 build 再 fidelity）"
                    )
                values[tier] = node.ds[layer_id].values

        for tier in tiers:
            expect = _tier_classes(tier)
            n_bad = int((values[tier] != expect).sum())
            checks.append({
                "name": f"{tier}: 与源栅格化/众数重算逐位一致（uint8 位级）",
                "status": "pass" if n_bad == 0 else "FAIL",
                "summary": f"不一致像元 {n_bad}（期望 0）",
            })
        for tier in tiers:
            if tier == BASINS_HOME_TIER:
                continue
            expect = aggregate_mode(
                values[BASINS_HOME_TIER],
                aggregation_factor(tier, BASINS_HOME_TIER), nodata=NODATA,
            )
            n_bad = int((values[tier] != expect).sum())
            checks.append({
                "name": (
                    f"{tier}: 类别众数一致性（= 主档 {BASINS_HOME_TIER} store 值众数聚合）"
                ),
                "status": "pass" if n_bad == 0 else "FAIL",
                "summary": f"不一致像元 {n_bad}（期望 0）",
            })
        for tier in tiers:
            v = values[tier]
            bad = int((~np.isin(v, np.append(valid_codes, NODATA))).sum())
            checks.append({
                "name": f"{tier}: 值域 ⊆ 编码表（1..8）∪ {{0}}",
                "status": "pass" if bad == 0 else "FAIL",
                "summary": f"非法值像元 {bad}（期望 0）",
            })
            counts = np.bincount(v.ravel(), minlength=len(valid_codes) + 1)
            inv = {c: lab for lab, c in BASIN_TYPE_CODES.items()}
            dist = ", ".join(
                f"{inv[c]}={counts[c]}" for c in sorted(BASIN_TYPE_CODES.values())
            )
            checks.append({
                "name": f"{tier}: 类别分布 + 覆盖率（报告制）",
                "status": "report",
                "summary": (
                    f"覆盖 {counts[1:].sum()}/{v.size}（{counts[1:].sum() / v.size:.4%}，"
                    f"盆地占全球 ~25.39%）；{dist}"
                ),
            })

    else:  # distance
        geoms, _codes = _basins_geoms()
        segs, n_drop = segment_table(_basin_rings(geoms))
        from shapely.strtree import STRtree

        tree = STRtree(geoms)
        rng = np.random.default_rng(_SAMPLE_SEED)
        with xr.open_datatree(store_dir, engine="zarr", chunks={}) as tree_store:
            for tier in tiers:
                node = tree_store[f"/{tier}"]
                if layer_id not in node.ds.data_vars:
                    raise KeyError(
                        f"store {store_dir} 的 {tier} 档不含 {layer_id}（先 build 再 fidelity）"
                    )
                v = node.ds[layer_id].values
                lat, lon = tier_centers(tier)
                inside = _tier_inside(tier)

                n_bad = int((~np.isfinite(v)).sum() + (v < 0).sum())
                checks.append({
                    "name": f"{tier}: 全域有限且 ≥ 0（免掩膜语义）",
                    "status": "pass" if n_bad == 0 else "FAIL",
                    "summary": f"违法像元 {n_bad}（期望 0）",
                })

                n_bad = int((v[inside] != 0).sum())
                checks.append({
                    "name": f"{tier}: 域内像元距离 = 0（区域距离语义）",
                    "status": "pass" if n_bad == 0 else "FAIL",
                    "summary": f"域内非零像元 {n_bad}（期望 0；域内 {int(inside.sum())} 像元）",
                })

                n_flat = v.size
                n_near = _N_SAMPLES // 3
                near_idx = np.nonzero(((v > 0) & (v < 50.0)).ravel())[0]
                inside_idx = np.nonzero(inside.ravel())[0]
                picks = [int(rng.integers(0, n_flat)) for _ in range(_N_SAMPLES - 2 * n_near)]

                def _take(pool, n):
                    if pool.size >= n:
                        return [int(x) for x in pool[rng.integers(0, pool.size, n)]]
                    return [int(x) for x in pool] + [
                        int(rng.integers(0, n_flat)) for _ in range(n - pool.size)
                    ]

                picks += _take(near_idx, n_near)
                picks += _take(inside_idx, n_near)
                max_abs = 0.0
                max_ratio = 0.0
                worst = None
                for flat in picks:
                    r, c = divmod(flat, v.shape[1])
                    indep = _independent_region_distance_km(
                        float(lat[r]), float(lon[c]), segs, tree, geoms
                    )
                    d = abs(indep - float(v[r, c]))
                    tol = max(_DIST_TOL_KM, _DIST_REL_TOL * abs(indep))
                    ratio = d / tol
                    if ratio > max_ratio:
                        max_ratio = ratio
                        worst = (float(lat[r]), float(lon[c]), indep, float(v[r, c]), d, tol)
                    max_abs = max(max_abs, d)
                ok = max_ratio <= 1.0
                checks.append({
                    "name": (
                        f"{tier}: 抽样 {len(picks)} 像元 vs 独立区域距离"
                        f"（shapely 包含 + 航空公式全段暴力）一致"
                        f"（tol = max(1e-3 km, 3e-7·d)）"
                    ),
                    "status": "pass" if ok else "FAIL",
                    "summary": (
                        f"max|Δ| = {max_abs:.3e} km，max |Δ|/tol = {max_ratio:.3f}"
                        + (f"；最差点 {worst[:2]}（独立 {worst[2]:.6f} vs store "
                           f"{worst[3]:.6f} km，Δ={worst[4]:.2e}/tol={worst[5]:.2e}）"
                           if worst else "")
                    ),
                })

                checks.append({
                    "name": f"{tier}: 距离分布（报告制）",
                    "status": "report",
                    "summary": (
                        f"min {v.min():.3f} / p50 {np.percentile(v, 50):.2f} / "
                        f"mean {v.mean():.2f} / max {v.max():.2f} km；"
                        f"域内(=0) {float((v == 0).mean()):.4%}，"
                        f"<10 km {float((v < 10).mean()):.3%}，"
                        f"<100 km {float((v < 100).mean()):.3%}"
                    ),
                })

        checks.append({
            "name": "段表零长剔除（报告制）",
            "status": "report",
            "summary": f"连续重复顶点段 {n_drop} 条剔除（环闭合首末重复，不参与距离场）",
        })

    n_fail = sum(1 for c in checks if c["status"] == "FAIL")
    return {
        "layer": layer_id,
        "store": str(store_dir),
        "integral_tolerance": INTEGRAL_TOL,
        "result": "FAIL" if n_fail else "PASS",
        "n_checks": len(checks),
        "checks": checks,
    }
