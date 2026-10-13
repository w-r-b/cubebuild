"""Vector channel (sidecar exporter): Hasterok GPRV tectonic-province
first-order classification.

Source (verified on disk):
- plates&provinces/shp/global_gprv.shp: 914 polygons (834 Polygon +
  80 MultiPolygon), EPSG:4326, exhaustive coverage (3′ cell-center
  rasterization coverage
  0.99991; 241 cells uncovered = digitization gaps at province boundaries).
  The first-order classification = prov_type
  (the paper's Table 1: "dominant tectonic character"); the data actually has 15
  classes
  (the official legend gprv_types.qml additionally has oceanic plateau/unknown,
  which do not occur in this version of the data).
- Polygon overlap (verified on disk): Σarea − ∪area = 0.66 deg² (0.001%),
  all digitization slivers at adjacent province boundaries, not a nesting
  hierarchy. Overlap handling (same Slab2 convention:
  an explicit deterministic rule + the total reported at build time): an
  overlapping cell takes the smaller-area polygon
  (locally refined features take priority).
- Poles/date line (verified on disk): the only edge with |Δlon|>180° on a ring
  is the Arctic basin closure edge
  (−180,90)→(180,90, the standard polar-cap cut representation, correctly
  rasterized in the plane); no date-line-crossing
  edges, so planar cell-center rasterization is safe.

Channel mechanism (one of / two of three):
- The polygon rasterizer rasterize_polygon_classes: assigns a class by a home
  tier grid cell center falling in the polygon
  (all_touched=False), and to coarse tiers via kernels.aggregate_mode
  majority aggregation (nodata=0 excluded, ties take the smaller code, ≥1 valid
  subcell means the cell has a value—the same semantics as the conservative
  kernel's "W>0 means value", convention). Later vector groups (plate/land
  class)
  reuse it.
- Integer class layers' missing semantics: class encoding starts from 1, 0 = no
  class (the single implementation masks.data_missing),
  the companion validity mask and the two channels are consistent.

Exhaustive-coverage exemption from the distance field: this channel produces no
distance layer.
License: Zenodo archive 10.5281/zenodo.6586972 CC BY 4.0 (the release basis,
avoiding the controversy over the GitHub repository GPL-3.0's virality over
data—r1 license audit).
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
ZENODO_DOI = "10.5281/zenodo.6586972"

HASTEROK_LAYER_ID = "lithosphere__hasterok_province_class"
HOME_TIER = "3min"

PROV_TYPE_CODES: dict[str, int] = {
    label: i for i, label in enumerate(sorted([
        "accretionary complex", "back-arc basin", "basin", "craton",
        "foredeep basin", "magmatic province", "narrow rift",
        "oceanic back-arc basin", "oceanic crust", "ophiolite complex",
        "orogenic belt", "passive margin", "shield", "volcanic arc",
        "wide rift",
    ]), start=1)
}

EXPECTED_ROWS = 914
NODATA = 0


def category_encoding() -> dict[str, str]:
    """Class encoding table (code → semantics), the single source shared by
    attrs/manifest/sidecar."""
    return {str(c): label for label, c in sorted(PROV_TYPE_CODES.items(), key=lambda kv: kv[1])}



def _iter_rings(geom):
    if geom.geom_type == "Polygon":
        yield geom.exterior
        yield from geom.interiors
    else:
        for p in geom.geoms:
            yield p.exterior
            yield from p.interiors


def load_gprv(src_dir: str | Path = GPRV_SRC_DIR):
    """Read global_gprv.shp → (GeoDataFrame, prov_type code array).

    Geometry contract (contract-ized disk verification): 914 rows, EPSG:4326,
    prov_type vocabulary ⊆ the encoding table, no empty geometry; a ring edge
    with |Δlon|>180° is allowed only as a pole closure edge (both ends
    |lat|=90)—a date-line-crossing edge would silently misplace the planar
    rasterization, rejected at read time.
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
            for k in np.nonzero(np.abs(np.diff(xy[:, 0])) > 180.0)[0]:
                if not (abs(xy[k][1]) == 90.0 and abs(xy[k + 1][1]) == 90.0):
                    raise ValueError(
                        f"{GPRV_REL_PATH}: 环存在跨日期线边 "
                        f"{tuple(xy[k])}→{tuple(xy[k + 1])}，平面栅格化不安全"
                    )
    codes = gdf["prov_type"].map(PROV_TYPE_CODES).to_numpy(dtype=np.uint8)
    return gdf, codes



def rasterize_polygon_classes(geoms, codes: np.ndarray, tier: str) -> tuple[np.ndarray, int, int]:
    """Class polygons → home-tier class grid (pure geometry piece,
    source-agnostic).

    Semantics: a cell center falling in a polygon belongs to that class
    (all_touched=False, consistent with the tier grid's center offset of half a
    tier); an overlapping cell takes the smaller-area polygon (locally refined
    features take priority, the explicit deterministic rule of Slab2's
    "overlap takes the shallowest"); cells outside any class = 0.

    Returns (classes uint8 (nlat, nlon), overlap cell count, uncovered cell
    count).
    Burned polygon by polygon with a bounding-box window (order = descending
    area, so the smaller one burns later and wins), fully deterministic and
    rerunnable.
    """
    import rasterio.features as rfeatures
    from rasterio.transform import from_origin

    nlat, nlon = grid_shape(tier)
    res = tier_deg(tier)
    classes = np.zeros((nlat, nlon), dtype=np.uint8)
    covered = np.zeros((nlat, nlon), dtype=np.uint8)

    order = np.argsort([-g.area for g in geoms], kind="stable")
    for i in order:
        g = geoms[i]
        minx, miny, maxx, maxy = g.bounds
        r0 = max(int(np.floor((miny + 90.0) / res)) - 1, 0)
        r1 = min(int(np.ceil((maxy + 90.0) / res)) + 1, nlat)     # exclusive
        c0 = max(int(np.floor((minx + 180.0) / res)) - 1, 0)
        c1 = min(int(np.ceil((maxx + 180.0) / res)) + 1, nlon)    # exclusive
        if r0 >= r1 or c0 >= c1:
            continue
        tr = from_origin(-180.0 + c0 * res, -90.0 + r1 * res, res, res)
        burned = rfeatures.rasterize(
            [(g, 1)], out_shape=(r1 - r0, c1 - c0), transform=tr, fill=0, dtype="uint8"
        ).astype(bool)
        burned = burned[::-1]
        win = classes[r0:r1, c0:c1]
        classes[r0:r1, c0:c1] = np.where(burned, codes[i], win)
        covered[r0:r1, c0:c1] += burned

    overlap_cells = int((covered > 1).sum())
    uncovered_cells = int((covered == 0).sum())
    return classes, overlap_cells, uncovered_cells


def _aggregation_factor(tier: str) -> int:
    """Home tier 3′ → target tier aggregation factor (thin wrapper around
    grids.aggregation_factor, module-local idiomatic name)."""
    return aggregation_factor(tier, HOME_TIER)


_HOME_CACHE: dict[str, tuple[np.ndarray, int, int]] = {}


def home_tier_classes(src_dir: str | Path = GPRV_SRC_DIR) -> tuple[np.ndarray, int, int]:
    """Home-tier (3′) class grid (in-process cache: the four tiers of the same
    layer are rasterized only once)."""
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



def build_hasterok_province_class(entry: LayerEntry, tier: str) -> xr.DataArray:
    """Build this tier's lithosphere__hasterok_province_class (uint8 class
    layer).

    Home tier 3′ = the rasterizer's direct output; coarse tiers = majority
    aggregation of the home tier (integer factors
    3′→6′/30′/1° = 2/10/20). Missing = 0 (the integer class layer convention),
    and the companion validity mask is generated automatically at the build
    entry point by the contract.
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
        "nodata_semantics": "0 = 无类别（省界数字化缝隙），类别编码自 1 起",
    }
    return xr.DataArray(
        arr, dims=("lat", "lon"), coords={"lat": lat, "lon": lon},
        name=entry.id, attrs=attrs,
    )



VECTOR_FIDELITY_LAYERS = {
    HASTEROK_LAYER_ID: {"src_dir": GPRV_SRC_DIR, "unit": "category"},
}


def build_hasterok_fidelity_report(store_dir, layer_id: str, tiers: list[str],
                                   src_dir=None) -> dict:
    """Fidelity report (class-layer criteria):

    ① store bit-for-bit consistent with the recomputation from source (hard
       criterion, bit-level—recomputation = rasterizer + majority kernel on the
       same code path, catching write/storage-chain misplacement);
    ② class majority consistency (hard criterion, acceptance item): coarse-tier
       store values == majority aggregation of home-tier store values—cross-tier
       consistency using the store itself as source, independent of ①;
    ③ value range ⊆ {0} ∪ the encoding table (hard criterion, encoding-error
       catcher);
    ④ class distribution/coverage/overlap and uncovered totals (report-only).
    """
    src_dir = src_dir or GPRV_SRC_DIR
    classes, overlap_cells, uncovered_cells = home_tier_classes(src_dir)

    checks: list[dict] = []
    values: dict[str, np.ndarray] = {}
    with xr.open_datatree(store_dir, engine="zarr", chunks={}) as tree:
        for tier in tiers:
            node = tree[f"/{tier}"]
            if layer_id not in node.ds.data_vars:
                raise KeyError(f"store {store_dir} 的 {tier} 档不含 {layer_id}（先 build 再 fidelity）")
            v = node.ds[layer_id].values
            values[tier] = v

            expect = classes if tier == HOME_TIER else aggregate_mode(
                classes, _aggregation_factor(tier), nodata=NODATA
            )
            n_bad = int((v != expect).sum())
            checks.append({
                "name": f"{tier}: 与源栅格化重算逐位一致（uint8 位级）",
                "status": "pass" if n_bad == 0 else "FAIL",
                "summary": f"不一致像元 {n_bad}（期望 0）",
            })

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
        "integral_tolerance": INTEGRAL_TOL,
        "result": "FAIL" if n_fail else "PASS",
        "n_checks": len(checks),
        "checks": checks,
    }
