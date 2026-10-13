"""Land-category group: GLiM full lithology / GUM unconsolidated sediments (lithology + thickness classes) /
Mooney 2023 continental-crust tectonic stabilization age.

Sources (disk-verified):
- GLiM (glim-lithology, Hartmann & Moosdorf 2012):
  glim-hartmann2012/glim_wgs84_0point5deg.txt.asc — 0.5° ASCII grid
  720×360, edges aligned to (-180,-90) (self-attested by xllcorner/yllcorner/cellsize),
  center phase bit-identical to the 30′ tier grid (-179.75/89.75-type quarter-degree offset) →
  primary tier 30′ mapped 1:1 (only north-south flip: ASCII row 0 = north side, project grid row 0 = south side).
  Value range {-9999} ∪ {1..16}; source code 15 (nd, 47 cells measured) is merged into 0 = no class.
- GUM (gum-unconsolidated, Börker et al. 2018, PANGAEA.884822 CC BY 3.0):
  GUM_v1.0/GUM_v1.0.shp — 911551 polygons (Polygon/MultiPolygon), EPSG:4326,
  XX lithology 41 classes (paper Table 1a/1b + disk count checked class by class), full-ring scan
  114,781,844 vertices with 0 edges |Δlon|>180° (planar rasterization safe; re-checked at read time by
  filtering bounding-box width >180°). DD thickness information has values for only ~1.4% of polygons (52
  literal forms, mixed m/ft/unitless) → ordered five-class binning (see the THICKNESS_* constants).
  GUM_pyroclastics/GUM_pyroclastics.shp — 20958 rows (Ic consolidated pyroclastics
  20424 + Iy unconsolidated 534; Ic is not part of the unconsolidated-sediment vocabulary and does not participate in rasterization),
  sidecar only (includes pyroclastic-flow features).
- Mooney 2023 (crustal-age-mooney2023, the Gubanov & Mooney July2022 tectonic-province supplement to
  Mooney et al. 2023 ECM1, CC BY 4.0 — stated explicitly by the authors via email):
  Shapefiles/ShapefilesGM/GubanovMooney_July2022.shp — 6 MultiPolygons
  (one each for ARC/PAP/MES/NEO/PAL/MCE), EPSG:4326. Tectonothermal stabilization age
  (tectonothermal stabilization age), not exposed-stratum age. Across all rings only 1
  |Δlon|>180° edge = the pole-closing edge of the MES Antarctic ring (both ends |lat|≈89.98, the source author's
  self-drawn pole cut) — the southernmost cell center of the 3′ tier, -89.975°, is above the cut, so planar rasterization
  is bit-identical to spherical semantics (the read-time contract allows pole-closing edges with |lat|≥89.9).

Channel mechanism:
- Rasterization = cell center falls inside the polygon (all_touched=False), with overlapping pixels assigned to the smaller-area
  polygon (local refinement priority; stable sort by descending area + last-burner-wins). The GUM 30″ primary tier
  uses the windowed rasterizer's dual code (lithology/thickness) + coverage-count extension: a single loop burns each
  polygon via windowed replace. ⚠ Deprecated rasterio MergeAlg.add counting pass: at this source's
  scale (911k complex polygons × 30″ full grid) it empirically triggered glibc heap corruption
  ("malloc(): unaligned tcache chunk detected", reproduced with both uint16/uint32,
  max stack only 3, non-overflow; localized by control experiments B/C/D/D2, the
  replace semantics and windowed path are both safe).
- Coarse tiers = primary-tier mode aggregation (kernels.aggregate_mode, nodata=0 excluded, ties
  take the smaller value, ≥1 valid subcell means a value).
- Land semantics (mask = validity + landsea): categorical layers carry a validity sidecar mask
  (0 = no class); layering it with derived__landsea_mask is done by the consumer
  (the same treatment as for same-source grids).
"""

import json
import re
import warnings
from pathlib import Path

import dask.array as da
import numpy as np
import xarray as xr

from .contract import LayerEntry
from .fidelity import INTEGRAL_TOL
from .grids import aggregation_factor, grid_shape, tier_chunks, tier_deg, tier_centers
from .kernels import aggregate_mode
from .vector import rasterize_polygon_classes

REPO_ROOT = Path(__file__).resolve().parents[1]
NODATA = 0

# =====================================================================
# GLiM（lithosphere__glim_lithology_class）
# =====================================================================

GLIM_SRC_DIR = REPO_ROOT / "original data/lithosphere/glim-lithology"
GLIM_REL_PATH = "glim-hartmann2012/glim_wgs84_0point5deg.txt.asc"
GLIM_LAYER_ID = "lithosphere__glim_lithology_class"
GLIM_HOME_TIER = "30min"

GLIM_DOI = "10.1029/2012GC004370"   # Hartmann & Moosdorf 2012, G3 13, 1197
GLIM_LICENSE_NOTE = (
    "源许可未明示（geo.uni-hamburg.de 资源页）；不纳入公开版"
)

GLIM_HEADER_CONTRACT = {
    "ncols": 720, "nrows": 360, "xllcorner": -180.0, "yllcorner": -90.0,
    "cellsize": 0.5, "nodata_value": -9999.0,
}
GLIM_SOURCE_NODATA = -9999
GLIM_ND_VALUE = 15

GLIM_CLASS_LABELS: dict[int, str] = {
    1: "su — unconsolidated sediments",
    2: "vb — basic volcanic rocks",
    3: "ss — siliciclastic sedimentary rocks",
    4: "pb — basic plutonic rocks",
    5: "sm — mixed sedimentary rocks",
    6: "sc — carbonate sedimentary rocks",
    7: "va — acid volcanic rocks",
    8: "mt — metamorphic rocks",
    9: "pa — acid plutonic rocks",
    10: "vi — intermediate volcanic rocks",
    11: "wb — water bodies",
    12: "py — pyroclastics",
    13: "pi — intermediate plutonic rocks",
    14: "ev — evaporites",
    16: "ig — ice and glaciers",
}


def glim_category_encoding() -> dict[str, str]:
    """GLiM class-encoding table (code → semantics), the single source shared by attrs/manifest/sidecar."""
    return {str(c): label for c, label in sorted(GLIM_CLASS_LABELS.items())}


def load_glim_grid(src_dir: str | Path = GLIM_SRC_DIR) -> np.ndarray:
    """Read the GLiM 0.5° ASCII grid → primary-tier (30′) class grid (360, 720) uint8.

    Row 0 = south side (ASCII row 0 = north side, vertically flipped after reading); -9999 (ocean) and
    source code 15 (nd) → 0 = no class; the remaining 1..16 keep their original codes (including wb water bodies/ig glaciers
    and other source-native classes; land-semantics layering is done by the consumer).
    """
    path = Path(src_dir) / GLIM_REL_PATH
    if not path.is_file():
        raise FileNotFoundError(f"GLiM 源文件不存在: {path}")
    lines = path.read_text(encoding="utf-8").splitlines()
    header: dict[str, float] = {}
    i = 0
    while i < len(lines):
        parts = lines[i].split()
        if len(parts) == 2 and parts[0].lower() in GLIM_HEADER_CONTRACT:
            header[parts[0].lower()] = float(parts[1])
            i += 1
        else:
            break
    if header != GLIM_HEADER_CONTRACT:
        raise ValueError(f"{GLIM_REL_PATH}: 头部契约漂移 {header} ≠ {GLIM_HEADER_CONTRACT}")

    rows = [l.split() for l in lines[i:] if l.strip()]
    if len(rows) != 360 or any(len(r) != 720 for r in rows):
        raise ValueError(f"{GLIM_REL_PATH}: 数据体形状 {len(rows)}×非固定列 ≠ 360×720")
    grid = np.array(rows, dtype=float)
    vocab = set(GLIM_CLASS_LABELS) | {GLIM_ND_VALUE, GLIM_SOURCE_NODATA}
    bad = set(np.unique(grid).astype(int)) - vocab
    if bad:
        raise ValueError(f"{GLIM_REL_PATH}: 值域超出 {{-9999,1..16}}: {sorted(bad)}")

    out = grid[::-1].astype(np.int32)
    out[(out == GLIM_SOURCE_NODATA) | (out == GLIM_ND_VALUE)] = 0
    return out.astype(np.uint8)


_GLIM_CACHE: dict[str, np.ndarray] = {}


def _glim_tier_values(tier: str, src_dir: str | Path = GLIM_SRC_DIR) -> np.ndarray:
    """GLiM class grid for this tier (in-process cache): 30′ mapped directly, 1° = mode aggregation (factor 2)."""
    key = f"glim:{tier}:{Path(src_dir).resolve()}"
    if key not in _GLIM_CACHE:
        home = load_glim_grid(src_dir)
        _GLIM_CACHE[key] = (
            home if tier == GLIM_HOME_TIER
            else aggregate_mode(home, aggregation_factor(tier, GLIM_HOME_TIER), nodata=NODATA)
        )
    return _GLIM_CACHE[key]


GLIM_NODATA_SEMANTICS = (
    "0 = 无类别（源 -9999 海洋 + 源码 15 nd 共 47 胞），类别编码自 1 起"
    "（源值码 1..16 除 15；wb 水体/ig 冰川为源固有类别，陆域语义与 "
    "derived__landsea_mask 组合由消费方完成）"
)


def build_glim_lithology_class(entry: LayerEntry, tier: str) -> xr.DataArray:
    """Build this tier's lithosphere__glim_lithology_class (uint8 categorical layer, two tiers)."""
    values = _glim_tier_values(tier)
    lat, lon = tier_centers(tier)
    attrs = {
        "units": entry.unit,
        "long_name": "全球岩性一级分类（GLiM，Hartmann & Moosdorf 2012，0.5° 源值码）",
        "source": entry.source,
        "doi": GLIM_DOI,
        "license_note": GLIM_LICENSE_NOTE,
        "native_res": entry.native_res,
        "resampling": entry.resampling,
        "visibility": entry.visibility,
        "category_encoding": json.dumps(
            glim_category_encoding(), ensure_ascii=False, sort_keys=True
        ),
        "classification_rule": (
            "源值码 1..16 原位保留（nd=15 → 0）；主档 30′ = 0.5° 源网格直映"
            "（边缘对齐 -180/-90、中心相位一致，仅南北行序翻转）；"
            "1° = 主档众数聚合（kernels.aggregate_mode，nodata=0 不参与、"
            "平票取小、≥1 有效子胞即有值）"
        ),
        "source_file": GLIM_REL_PATH,
        "nodata_semantics": GLIM_NODATA_SEMANTICS,
    }
    arr = da.from_array(values, chunks=tier_chunks(tier))
    return xr.DataArray(
        arr, dims=("lat", "lon"), coords={"lat": lat, "lon": lon},
        name=entry.id, attrs=attrs,
    )


# =====================================================================
# GUM（sediment__gum_lithology_class / sediment__gum_thickness_class）
# =====================================================================

GUM_SRC_DIR = REPO_ROOT / "original data/sediment/gum-unconsolidated"
GUM_MAIN_REL = "Boerker_et_al_GUM_v1.0/GUM_v1.0/GUM_v1.0.shp"
GUM_PYRO_REL = "Boerker_et_al_GUM_v1.0/GUM_pyroclastics/GUM_pyroclastics.shp"
GUM_LITHOLOGY_LAYER_ID = "sediment__gum_lithology_class"
GUM_THICKNESS_LAYER_ID = "sediment__gum_thickness_class"
GUM_HOME_TIER = "30sec"

GUM_DOI = "10.1002/2017GC007273"
GUM_PANGAEA_DOI = "10.1594/PANGAEA.884822"
GUM_LICENSE_NOTE = (
    f"发布依据 PANGAEA {GUM_PANGAEA_DOI}（CC BY 3.0，数据集页自证）；"
    f"论文 DOI {GUM_DOI}（PDF 自证；勘误：早前所记 10.1029/2018GC007637 系笔误）"
)

GUM_EXPECTED_ROWS = 911551
GUM_PYRO_EXPECTED_ROWS = 20958

GUM_LITHOLOGY_LABELS: dict[str, str] = {
    "Ae": "Alluvial – aeolian deposits",
    "Af": "Alluvial – fan deposits",
    "Al": "Alluvial – lacustrine deposits",
    "Ap": "Alluvial – plain deposits",
    "At": "Alluvial – terrace deposits",
    "Au": "Alluvial – undifferentiated",
    "Ca": "Colluvial – alluvial deposits",
    "Cu": "Colluvial – undifferentiated",
    "Du": "Ice",
    "Ea": "Aeolian – loess-like silt deposits",
    "Ed": "Aeolian – dunes",
    "El": "Aeolian – loess deposits",
    "Er": "Aeolian – loess derivates",
    "Eu": "Aeolian – undifferentiated",
    "Gf": "Glacial – fluvioglacial deposits",
    "Gl": "Glacial – glaciolacustrine deposits",
    "Gm": "Glacial – glaciomarine deposits",
    "Gp": "Glacial – proglacial deposits",
    "Gt": "Glacial – till",
    "Gu": "Glacial – undifferentiated",
    "Iy": "Pyroclastics",
    "Lu": "Lacustrine deposits",
    "Mu": "Marine deposits",
    "Op": "Organic – peat deposits",
    "Or": "Organic – reef deposits",
    "Ou": "Organic – undifferentiated",
    "Pg": "Evaporites – gypsum deposits",
    "Pp": "Evaporites – playa deposits",
    "Ps": "Evaporites – salt deposits",
    "Pu": "Evaporites – undifferentiated",
    "Us": "Sediments – undifferentiated",
    "Wl": "Water – lakes",
    "Wu": "Water bodies – undifferentiated",
    "Wr": "Water – rivers",
    "Yb": "Coastal – beach deposits",
    "Yd": "Coastal – delta sediments",
    "Yl": "Coastal – lagoonal sediments",
    "Ym": "Coastal – marsh sediments",
    "Ys": "Coastal – swamp deposits",
    "Yu": "Coastal – undifferentiated",
    "Zu": "Anthropogenic deposits",
}
GUM_LITHOLOGY_CODES: dict[str, int] = {
    xx: i for i, xx in enumerate(sorted(GUM_LITHOLOGY_LABELS), start=1)
}

GUM_PYRO_XX_VOCAB = {"Ic", "Iy"}

FT_TO_M = 0.3048
THICKNESS_BIN_EDGES = (2.0, 10.0, 50.0, 200.0)
THICKNESS_CLASS_LABELS: dict[int, str] = {
    1: "≤ 2 m",
    2: "2–10 m",
    3: "10–50 m",
    4: "50–200 m",
    5: "> 200 m",
}

_THICK_RANGE_RE = re.compile(r"^([<>]?)\s*(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*(m|ft)?$")
_THICK_BOUND_RE = re.compile(r"^([<>])\s*(\d+(?:\.\d+)?)\s*(m|ft)?$")
_THICK_SINGLE_RE = re.compile(r"^(\d+(?:\.\d+)?)\s*(m|ft)?$")


def parse_thickness_m(dd) -> float | None:
    """DD literal value → representative thickness (m); no magnitude information (nn/dis/unrecognized) → None.

    Parses segment by segment then averages; an unparseable segment is skipped (all segments invalid → None).
    All 52 disk literal forms are covered (tests assert each one).
    """
    if dd is None or not isinstance(dd, str) or not dd.strip():
        return None
    vals: list[float] = []
    for seg in dd.split(","):
        seg = seg.strip()
        if not seg:
            continue
        if (m := _THICK_RANGE_RE.match(seg)) is not None:
            op, a, b, unit = m.groups()
            f = FT_TO_M if unit == "ft" else 1.0
            vals.append((float(a) + float(b)) / 2.0 * f)
        elif (m := _THICK_BOUND_RE.match(seg)) is not None:
            op, a, unit = m.groups()
            f = FT_TO_M if unit == "ft" else 1.0
            vals.append(float(a) * f)
        elif (m := _THICK_SINGLE_RE.match(seg)) is not None:
            a, unit = m.groups()
            f = FT_TO_M if unit == "ft" else 1.0
            vals.append(float(a) * f)
    if not vals:
        return None
    return sum(vals) / len(vals)


def thickness_class(dd) -> int:
    """DD literal value → ordered thickness class code (0 = no magnitude information)."""
    t = parse_thickness_m(dd)
    if t is None:
        return 0
    for code, edge in enumerate(THICKNESS_BIN_EDGES, start=1):
        if t <= edge:
            return code
    return len(THICKNESS_BIN_EDGES) + 1


def gum_category_encoding() -> dict[str, str]:
    """GUM lithology encoding table (code → semantics)."""
    return {
        str(c): f"{xx} – {GUM_LITHOLOGY_LABELS[xx]}"
        for xx, c in sorted(GUM_LITHOLOGY_CODES.items(), key=lambda kv: kv[1])
    }


def gum_thickness_encoding() -> dict[str, str]:
    """GUM thickness-class encoding table (code → semantics, ordered)."""
    return {str(c): label for c, label in sorted(THICKNESS_CLASS_LABELS.items())}


def _iter_rings(geom):
    if geom.geom_type == "Polygon":
        yield geom.exterior
        yield from geom.interiors
    else:
        for p in geom.geoms:
            yield p.exterior
            yield from p.interiors


def _check_dateline_safe(gdf, rel_path: str) -> None:
    """Dateline-crossing ring check (planar-rasterization safety contract).

    A polygon whose bounding-box width ≤ 180° cannot contain an edge |Δlon|>180° (both edge endpoints lie within
    the bounding box) — only wide-bounding-box polygons are scanned ring by ring; an |Δlon|>180° edge is allowed only as a
    pole-closing edge (both endpoints |lat| ≥ 89.9, the source author's self-drawn pole cut), otherwise rejected.
    """
    bounds = gdf.geometry.bounds
    wide = np.nonzero(((bounds["maxx"] - bounds["minx"]) > 180.0).to_numpy())[0]
    for i in wide:
        for ring in _iter_rings(gdf.geometry.iloc[i]):
            xy = np.asarray(ring.coords)
            for k in np.nonzero(np.abs(np.diff(xy[:, 0])) > 180.0)[0]:
                if not (abs(xy[k][1]) >= 89.9 and abs(xy[k + 1][1]) >= 89.9):
                    raise ValueError(
                        f"{rel_path}: 环存在跨日期线边 {tuple(xy[k])}→{tuple(xy[k + 1])}，"
                        f"平面栅格化不安全"
                    )


def load_gum(src_dir: str | Path = GUM_SRC_DIR):
    """Read GUM_v1.0.shp → (GeoDataFrame, lithology codes, thickness class codes).

    Geometry contract (contractualized from disk evidence): 911551 rows, EPSG:4326, XX vocabulary ⊆ 41 classes
    with no null values, no empty geometries, wide-bounding-box polygons with no dateline-crossing edge (pole-closing edges excepted).
    """
    import geopandas as gpd

    path = Path(src_dir) / GUM_MAIN_REL
    if not path.is_file():
        raise FileNotFoundError(f"GUM 源文件不存在: {path}")
    gdf = gpd.read_file(path)
    if len(gdf) != GUM_EXPECTED_ROWS:
        raise ValueError(f"{GUM_MAIN_REL}: 行数 {len(gdf)} ≠ 期望 {GUM_EXPECTED_ROWS}")
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        raise ValueError(f"{GUM_MAIN_REL}: CRS {gdf.crs} ≠ EPSG:4326")
    if gdf["XX"].isna().any():
        raise ValueError(f"{GUM_MAIN_REL}: XX 存在空值")
    unknown = sorted(set(gdf["XX"]) - set(GUM_LITHOLOGY_CODES))
    if unknown:
        raise ValueError(f"{GUM_MAIN_REL}: XX 超出 41 类编码表: {unknown}")
    if gdf.geometry.isna().any() or gdf.geometry.is_empty.any():
        raise ValueError(f"{GUM_MAIN_REL}: 存在空几何")
    _check_dateline_safe(gdf, GUM_MAIN_REL)

    lith = gdf["XX"].map(GUM_LITHOLOGY_CODES).to_numpy(dtype=np.uint8)
    thick = gdf["DD"].map(thickness_class).to_numpy(dtype=np.uint8)
    return gdf, lith, thick


def load_gum_pyroclastics(src_dir: str | Path = GUM_SRC_DIR):
    """Read GUM_pyroclastics.shp → GeoDataFrame (sidecar only, does not participate in rasterization)."""
    import geopandas as gpd

    path = Path(src_dir) / GUM_PYRO_REL
    if not path.is_file():
        raise FileNotFoundError(f"GUM 碎屑流源文件不存在: {path}")
    gdf = gpd.read_file(path)
    if len(gdf) != GUM_PYRO_EXPECTED_ROWS:
        raise ValueError(f"{GUM_PYRO_REL}: 行数 {len(gdf)} ≠ 期望 {GUM_PYRO_EXPECTED_ROWS}")
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        raise ValueError(f"{GUM_PYRO_REL}: CRS {gdf.crs} ≠ EPSG:4326")
    unknown = sorted(set(gdf["XX"]) - GUM_PYRO_XX_VOCAB)
    if unknown:
        raise ValueError(f"{GUM_PYRO_REL}: XX 超出 {{Ic, Iy}}: {unknown}")
    return gdf


def rasterize_gum_classes(geoms, lith_codes: np.ndarray, thick_codes: np.ndarray,
                           tier: str = GUM_HOME_TIER):
    """GUM dual-code joint rasterization → (lith, thick, overlap_cells, uncovered_cells).

    Semantics the same as vector.rasterize_polygon_classes (windowed path): a cell center
    falling inside a polygon belongs to that class (all_touched=False); overlapping pixels take the smaller-area polygon
    (stable sort by descending area + last-burner-wins); polygons with thick_codes=0 (no thickness information)
    do not participate in burning the thickness layer (0 would overwrite valid codes, so it must be skipped; lithology codes all participate).
    A covered counter accumulates in the same loop → overlap/uncovered totals (uint8, max
    stacking measured on disk is 3).

    ⚠ Implementation constraint (empirical): rasterio 1.4.4's merge_alg=
    MergeAlg.add counting pass at this source (911k complex polygons × 30″ full grid) triggers
    glibc heap corruption ("malloc(): unaligned tcache chunk detected", reproduced with both uint16/
    uint32 dtype, max stack only 3, non-overflow; the replace semantics and windowed
    path are both safe — localized by control experiments B/C/D/D2), so this implementation drops the add pass and
    burns everything via per-polygon windowed replace (same as vector.py, positioned after a row-order flip).
    """
    import rasterio.features as rfeatures
    from rasterio.transform import from_origin

    geoms = list(geoms)
    if len(geoms) != len(lith_codes) or len(geoms) != len(thick_codes):
        raise ValueError("geoms 与码数组长度不一致")
    nlat, nlon = grid_shape(tier)
    res = tier_deg(tier)
    lith = np.zeros((nlat, nlon), dtype=np.uint8)
    thick = np.zeros((nlat, nlon), dtype=np.uint8)
    covered = np.zeros((nlat, nlon), dtype=np.uint8)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*geographic CRS.*")
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
            [(g, 1)], out_shape=(r1 - r0, c1 - c0), transform=tr, fill=0,
            dtype="uint8",
        ).astype(bool)
        burned = burned[::-1]
        win_l = lith[r0:r1, c0:c1]
        lith[r0:r1, c0:c1] = np.where(burned, lith_codes[i], win_l)
        if thick_codes[i] != NODATA:
            win_t = thick[r0:r1, c0:c1]
            thick[r0:r1, c0:c1] = np.where(burned, thick_codes[i], win_t)
        covered[r0:r1, c0:c1] += burned

    overlap_cells = int((covered > 1).sum())
    uncovered_cells = int((covered == 0).sum())
    return lith, thick, overlap_cells, uncovered_cells


_GUM_HOME_CACHE: dict[str, tuple] = {}
_GUM_MODE_CACHE: dict[tuple[str, str, str], np.ndarray] = {}
_GUM_SOURCE_CACHE: dict[str, tuple] = {}


def _gum_home_grids(src_dir: str | Path = GUM_SRC_DIR):
    """GUM primary-tier (30″) dual-code grid (in-process cache: the two layers' several tiers are rasterized only once)."""
    key = f"gum:{Path(src_dir).resolve()}"
    if key not in _GUM_HOME_CACHE:
        gdf, lith, thick = load_gum(src_dir)
        _GUM_SOURCE_CACHE[key] = (gdf, lith, thick)
        _GUM_HOME_CACHE[key] = rasterize_gum_classes(
            list(gdf.geometry), lith, thick, GUM_HOME_TIER
        )
    return _GUM_HOME_CACHE[key]


def _gum_tier_values(layer: str, tier: str, src_dir: str | Path = GUM_SRC_DIR) -> np.ndarray:
    """GUM class grid for this layer and tier (cached): 30″ output directly, coarse tiers = primary-tier mode aggregation."""
    idx = 0 if layer == GUM_LITHOLOGY_LAYER_ID else 1
    key = (layer, tier, str(Path(src_dir).resolve()))
    if key not in _GUM_MODE_CACHE:
        home = _gum_home_grids(src_dir)[idx]
        _GUM_MODE_CACHE[key] = (
            home if tier == GUM_HOME_TIER
            else aggregate_mode(home, aggregation_factor(tier, GUM_HOME_TIER), nodata=NODATA)
        )
    return _GUM_MODE_CACHE[key]


GUM_RASTERIZATION_RULE = (
    "胞心落入多边形即属该类（all_touched=False，开窗栅格化器双码"
    "扩展）；多边形重叠像元取面积较小多边形（局部细化优先；面积降序稳定"
    "排序 + 后烧者胜）；粗档 = 主档 30″ 众数聚合（kernels.aggregate_mode，"
    "nodata=0 不参与、平票取小、≥1 有效子胞即有值）。实现为逐多边形开窗"
    "replace 烧录——rasterio 1.4.4 的 MergeAlg.add 计数遍在本源规模实证"
    "触发 glibc 堆损坏（对照实验定位），弃用"
)

GUM_THICKNESS_RULE = (
    "源 DD 厚度信息 → 代表厚度（m）：区间取中点；单侧界值（>a/<a）取界值"
    "本身（保守）；ft × 0.3048；无单位字面值按 m（磁盘实证全部为黄土源"
    "El/Er，文献以 m 计）；逗号并列多段取各段代表值均值；nn/dis 无幅度"
    "信息 → 0。有序五档分箱（m）：≤2 / (2,10] / (10,50] / (50,200] / "
    ">200 → 码 1..5。有序类别仍按众数降采样，保序聚合留给下游（验收"
    "注记）；厚度信息仅 ~1.4% 多边形有值（898268/911551 为 nn），层覆盖率"
    "相应稀疏——侧车 DD 原值列保留全部信息"
)


def _build_gum_layer(entry: LayerEntry, tier: str, layer_id: str,
                     long_name: str, encoding: dict,
                     extra_attrs: dict) -> xr.DataArray:
    values = _gum_tier_values(layer_id, tier)
    lat, lon = tier_centers(tier)
    attrs = {
        "units": entry.unit,
        "long_name": long_name,
        "source": entry.source,
        "doi": GUM_DOI,
        "license_note": GUM_LICENSE_NOTE,
        "native_res": entry.native_res,
        "resampling": entry.resampling,
        "visibility": entry.visibility,
        "category_encoding": json.dumps(encoding, ensure_ascii=False, sort_keys=True),
        "rasterization_rule": GUM_RASTERIZATION_RULE,
        "source_file": GUM_MAIN_REL,
        **extra_attrs,
    }
    arr = da.from_array(values, chunks=tier_chunks(tier))
    return xr.DataArray(
        arr, dims=("lat", "lon"), coords={"lat": lat, "lon": lon},
        name=entry.id, attrs=attrs,
    )


def build_gum_lithology_class(entry: LayerEntry, tier: str) -> xr.DataArray:
    """Build this tier's sediment__gum_lithology_class (uint8 categorical layer, five tiers)."""
    return _build_gum_layer(
        entry, tier, GUM_LITHOLOGY_LAYER_ID,
        "未固结沉积物岩性分类（GUM，Börker et al. 2018，XX 41 类，像元内主导类别）",
        gum_category_encoding(),
        {
            "nodata_semantics": (
                "0 = 无类别（GUM 未覆盖区 + GUM 外海洋），类别编码自 1 起；"
                "水体（Wu/Wl/Wr）与冰（Du）为源固有类别，陆域语义与 "
                "derived__landsea_mask 组合由消费方完成；火山碎屑 Ic（固结，"
                "20424 条）不属未固结词汇不参与栅格化，见 gum_unconsolidated 侧车"
            ),
        },
    )


def build_gum_thickness_class(entry: LayerEntry, tier: str) -> xr.DataArray:
    """Build this tier's sediment__gum_thickness_class (uint8 ordered categorical layer, five tiers)."""
    return _build_gum_layer(
        entry, tier, GUM_THICKNESS_LAYER_ID,
        "未固结沉积物厚度分级（GUM DD 字段，有序五档，像元内主导类别）",
        gum_thickness_encoding(),
        {
            "classification_rule": GUM_THICKNESS_RULE,
            "nodata_semantics": (
                "0 = 无厚度信息（GUM 未覆盖 + DD=nn/dis），分级编码自 1 起"
                "（1 = ≤2 m … 5 = >200 m，有序）；陆域语义与 derived__landsea_mask"
                " 组合由消费方完成"
            ),
        },
    )


# =====================================================================
# =====================================================================

MOONEY_SRC_DIR = REPO_ROOT / "original data/lithosphere/crustal-age-mooney2023"
MOONEY_REL_PATH = "Shapefiles/ShapefilesGM/GubanovMooney_July2022.shp"
MOONEY_LAYER_ID = "lithosphere__crustal_age_class"
MOONEY_HOME_TIER = "3min"

MOONEY_DOI = "10.1016/j.earscirev.2023.104493"   # Mooney et al. 2023, ECM1, ESR 249, 104493
MOONEY_LICENSE_NOTE = (
    "发布依据作者明示 CC BY 4.0 可再分发（同封邮件含 ECM1）"
)

MOONEY_AGE_CODES: dict[str, int] = {
    "Archean": 1,
    "Paleoproterozoic": 2,
    "Mesoproterozoic": 3,
    "Neoproterozoic": 4,
    "Paleozoic": 5,
    "Cenozoic-Mesozoic": 6,
}
MOONEY_AGE_RANGES = {
    "Archean": "3.6–2.5 Ga", "Paleoproterozoic": "2.5–1.6 Ga",
    "Mesoproterozoic": "1.6–1.0 Ga", "Neoproterozoic": "1.0–0.54 Ga",
    "Paleozoic": "540–250 Ma", "Cenozoic-Mesozoic": "250–0 Ma",
}
MOONEY_EXPECTED_ROWS = 6


def mooney_category_encoding() -> dict[str, str]:
    """Mooney tectonic-age encoding table (code → semantics, ordered oldest to youngest)."""
    return {
        str(c): f"{age}（{MOONEY_AGE_RANGES[age]}）"
        for age, c in sorted(MOONEY_AGE_CODES.items(), key=lambda kv: kv[1])
    }


def load_mooney_provinces(src_dir: str | Path = MOONEY_SRC_DIR):
    """Read GubanovMooney_July2022.shp → (GeoDataFrame, age codes).

    Geometry contract (contractualized from disk evidence): 6 rows (one MultiPolygon per age),
    EPSG:4326, Age vocabulary ⊆ the encoding table, geometries valid, |Δlon|>180° edges allowed only as pole-closing
    edges (both endpoints |lat| ≥ 89.9 — the MES Antarctic ring measured -89.977/-89.979 pole
    cut, self-drawn by the source author; the 3′ tier's southernmost cell center -89.975° is above the cut, so planar rasterization
    is bit-identical to spherical semantics).
    """
    import geopandas as gpd

    path = Path(src_dir) / MOONEY_REL_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Mooney 源文件不存在: {path}")
    gdf = gpd.read_file(path)
    if len(gdf) != MOONEY_EXPECTED_ROWS:
        raise ValueError(f"{MOONEY_REL_PATH}: 行数 {len(gdf)} ≠ 期望 {MOONEY_EXPECTED_ROWS}")
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        raise ValueError(f"{MOONEY_REL_PATH}: CRS {gdf.crs} ≠ EPSG:4326")
    unknown = sorted(set(gdf["Age"]) - set(MOONEY_AGE_CODES))
    if unknown:
        raise ValueError(f"{MOONEY_REL_PATH}: Age 超出编码表: {unknown}")
    if gdf.geometry.isna().any() or gdf.geometry.is_empty.any():
        raise ValueError(f"{MOONEY_REL_PATH}: 存在空几何")
    if not gdf.geometry.is_valid.all():
        raise ValueError(f"{MOONEY_REL_PATH}: 存在无效几何")
    _check_dateline_safe(gdf, MOONEY_REL_PATH)
    codes = gdf["Age"].map(MOONEY_AGE_CODES).to_numpy(dtype=np.uint8)
    return gdf, codes


_MOONEY_CACHE: dict[str, tuple] = {}
_MOONEY_MODE_CACHE: dict[tuple[str, str], np.ndarray] = {}


def _mooney_home_classes(src_dir: str | Path = MOONEY_SRC_DIR) -> tuple:
    """Primary-tier (3′) class grid + overlap/uncovered (in-process cache; reused by the rasterizer)."""
    key = f"mooney:{Path(src_dir).resolve()}"
    if key not in _MOONEY_CACHE:
        gdf, codes = load_mooney_provinces(src_dir)
        _MOONEY_CACHE[key] = rasterize_polygon_classes(
            list(gdf.geometry), codes, MOONEY_HOME_TIER
        )
    return _MOONEY_CACHE[key]


def _mooney_tier_values(tier: str, src_dir: str | Path = MOONEY_SRC_DIR) -> np.ndarray:
    key = (tier, str(Path(src_dir).resolve()))
    if key not in _MOONEY_MODE_CACHE:
        home = _mooney_home_classes(src_dir)[0]
        _MOONEY_MODE_CACHE[key] = (
            home if tier == MOONEY_HOME_TIER
            else aggregate_mode(home, aggregation_factor(tier, MOONEY_HOME_TIER), nodata=NODATA)
        )
    return _MOONEY_MODE_CACHE[key]


MOONEY_RASTERIZATION_RULE = (
    "胞心落入多边形即属该类（all_touched=False）；"
    "多边形重叠像元取面积较小多边形（局部细化优先；源重叠 0.14 deg² = "
    "0.0006%，时代省界数字化狭条）；粗档 = 主档 3′ 众数聚合（kernels."
    "aggregate_mode，nodata=0 不参与、平票取小、≥1 有效子胞即有值）；"
    "MES 南极环极点 cut（lat≈-89.978）为源作者自绘，3′ 最南胞心 "
    "-89.975° 在 cut 之上，平面栅格化与球面语义一致"
)


def build_crustal_age_class(entry: LayerEntry, tier: str) -> xr.DataArray:
    """Build this tier's lithosphere__crustal_age_class (uint8 categorical layer, four tiers)."""
    values = _mooney_tier_values(tier)
    lat, lon = tier_centers(tier)
    attrs = {
        "units": entry.unit,
        "long_name": (
            "大陆地壳构造稳定化时代（Mooney et al. 2023 ECM1 的 Gubanov & "
            "Mooney 构造省，6 类自老至新，像元内主导类别）"
        ),
        "source": entry.source,
        "doi": MOONEY_DOI,
        "license_note": MOONEY_LICENSE_NOTE,
        "native_res": entry.native_res,
        "resampling": entry.resampling,
        "visibility": entry.visibility,
        "category_encoding": json.dumps(
            mooney_category_encoding(), ensure_ascii=False, sort_keys=True
        ),
        "rasterization_rule": MOONEY_RASTERIZATION_RULE,
        "source_file": MOONEY_REL_PATH,
        "nodata_semantics": (
            "0 = 无类别（大陆域外 + 时代省界数字化缝隙），编码自 1 起"
            "（1=Archean … 6=Cenozoic-Mesozoic，自老至新有序）；与 "
            "lithosphere__seafloor_age 覆盖域互补（陆/海），陆域语义与 "
            "derived__landsea_mask 组合由消费方完成"
        ),
    }
    arr = da.from_array(values, chunks=tier_chunks(tier))
    return xr.DataArray(
        arr, dims=("lat", "lon"), coords={"lat": lat, "lon": lon},
        name=entry.id, attrs=attrs,
    )


# =====================================================================
# =====================================================================

LANDCAT_FIDELITY_LAYERS = {
    GLIM_LAYER_ID: {"kind": "glim"},
    GUM_LITHOLOGY_LAYER_ID: {"kind": "gum", "field": "lithology"},
    GUM_THICKNESS_LAYER_ID: {"kind": "gum", "field": "thickness"},
    MOONEY_LAYER_ID: {"kind": "mooney"},
}

GLIM_LANDMARKS = [
    ("Greenland 冰盖", 72.0, -40.0, 16),        # ig
    ("撒哈拉（ss）", 23.0, 10.0, 3),            # ss
    ("德干溢流玄武岩（vb）", 19.0, 74.0, 2),    # vb
    ("亚马逊低地（ss）", -3.0, -60.0, 3),       # ss
    ("东南极（ig）", -80.0, 0.0, 16),           # ig
    ("阿拉伯（su）", 24.0, 45.0, 1),            # su
    ("太平洋（无类别）", 0.0, -150.0, 0),
]
MOONEY_LANDMARKS = [
    ("南极点行（MES 极点 cut 之上）", -89.975, 0.0, 3),   # Mesoproterozoic
    ("Yilgarn 克拉通", -28.0, 118.0, 1),                  # Archean
    ("坦桑尼亚克拉通", -6.0, 33.0, 1),
    ("印度地盾", 21.0, 78.0, 1),
    ("格陵兰", 72.0, -40.0, 2),                           # Paleoproterozoic
    ("西伯利亚", 62.0, 100.0, 2),
    ("中大西洋（无类别）", 0.0, -20.0, 0),
]

_LANDSEA_NOTE = (
    "陆域语义：类别层自带 validity 伴生掩膜（0 = 无类别），与 "
    "derived__landsea_mask 的组合分层由消费方完成（同源网格处置同款）"
)


def _class_checks(checks, values_by_tier, expect_by_tier, home_tier,
                  valid_codes: set, encoding_name: str) -> None:
    """Generic criteria for categorical layers: ① bit-identical ② mode consistency ③ value range ④ distribution (report-mode)."""
    for tier, v in values_by_tier.items():
        n_bad = int((v != expect_by_tier[tier]).sum())
        checks.append({
            "name": f"{tier}: 与源重算逐位一致（uint8 位级）",
            "status": "pass" if n_bad == 0 else "FAIL",
            "summary": f"不一致像元 {n_bad}（期望 0）",
        })
    for tier, v in values_by_tier.items():
        if tier == home_tier:
            continue
        expect = aggregate_mode(
            values_by_tier[home_tier],
            aggregation_factor(tier, home_tier), nodata=NODATA,
        )
        n_bad = int((v != expect).sum())
        checks.append({
            "name": f"{tier}: 类别众数一致性（= 主档 {home_tier} store 值众数聚合）",
            "status": "pass" if n_bad == 0 else "FAIL",
            "summary": f"不一致像元 {n_bad}（期望 0）",
        })
    for tier, v in values_by_tier.items():
        bad = int((~np.isin(v, np.array(sorted(valid_codes | {NODATA})))).sum())
        checks.append({
            "name": f"{tier}: 值域 ⊆ {{0}} ∪ {encoding_name}",
            "status": "pass" if bad == 0 else "FAIL",
            "summary": f"非法值像元 {bad}（期望 0）",
        })
        counts = np.bincount(v.ravel(), minlength=max(valid_codes) + 1)
        dist = ", ".join(
            f"code{c}={counts[c]}" for c in sorted(valid_codes) if counts[c] > 0
        )
        checks.append({
            "name": f"{tier}: 类别分布（报告制）",
            "status": "report",
            "summary": f"覆盖 {counts[1:].sum()}/{v.size}；{dist}",
        })


def _read_store_values(store_dir, layer_id: str, tiers: list[str]) -> dict:
    values: dict[str, np.ndarray] = {}
    with xr.open_datatree(store_dir, engine="zarr", chunks={}) as tree:
        for tier in tiers:
            node = tree[f"/{tier}"]
            if layer_id not in node.ds.data_vars:
                raise KeyError(
                    f"store {store_dir} 的 {tier} 档不含 {layer_id}（先 build 再 fidelity）"
                )
            values[tier] = node.ds[layer_id].values
    return values


def _landmark_checks(checks, values_home: np.ndarray, landmarks, home_tier) -> None:
    """Landmark final check (hard criterion): the store primary-tier value == the disk-verified cell-center expected value."""
    lat_c, lon_c = tier_centers(home_tier)
    for name, la, lo, expect in landmarks:
        r = int((la + 90.0) / tier_deg(home_tier))
        c = int((lo + 180.0) / tier_deg(home_tier))
        got = int(values_home[r, c])
        checks.append({
            "name": f"地标: {name} @({lat_c[r]:.4f},{lon_c[c]:.4f})",
            "status": "pass" if got == expect else "FAIL",
            "summary": f"store={got}，期望={expect}",
        })


def build_landcat_fidelity_report(store_dir, layer_id: str, tiers: list[str],
                                  src_dir=None) -> dict:
    """Fidelity report (categorical-layer criteria + this group's specific notes).

    Generic: ① store vs. source recompute bit-identical (uint8 bit level) ② coarse-tier class mode consistency
    (= mode aggregation of the primary-tier store values) ③ value range ⊆ {0} ∪ the encoding table ④ class distribution/coverage
    (report-mode). Specific: GLiM/Mooney landmark final check (disk-verified cell-center expected values);
    GUM overlap/uncovered totals; Mooney geodesic area × Age attribute cross-check (report-mode);
    land-semantics/complementarity notes (report-mode).
    """
    spec = LANDCAT_FIDELITY_LAYERS[layer_id]
    kind = spec["kind"]
    checks: list[dict] = []

    if kind == "glim":
        home = load_glim_grid(src_dir or GLIM_SRC_DIR)
        expect = {
            t: (home if t == GLIM_HOME_TIER else
                aggregate_mode(home, aggregation_factor(t, GLIM_HOME_TIER), nodata=NODATA))
            for t in tiers
        }
        values = _read_store_values(store_dir, layer_id, tiers)
        _class_checks(checks, values, expect, GLIM_HOME_TIER,
                      set(GLIM_CLASS_LABELS), "GLiM 源值码表")
        _landmark_checks(checks, values[GLIM_HOME_TIER], GLIM_LANDMARKS, GLIM_HOME_TIER)
        checks.append({
            "name": "GLiM/GUM 语义差异注记（报告制）",
            "status": "report",
            "summary": (
                "GLiM = 全岩性（基岩+表生，Hartmann & Moosdorf 2012），GUM = "
                "未固结沉积物细化（Börker et al. 2018）——同类产品互补保留"
                "（克制原则）；GLiM su（未固结沉积物）与 GUM 覆盖域"
                "同源（GUM 无图区以 GLiM su 补绘，论文自证）"
            ),
        })
        checks.append({"name": "陆域语义注记（报告制）", "status": "report",
                       "summary": _LANDSEA_NOTE})

    elif kind == "gum":
        grids = _gum_home_grids(src_dir or GUM_SRC_DIR)
        idx = 0 if spec["field"] == "lithology" else 1
        home, overlap_cells, uncovered_cells = grids[idx], grids[2], grids[3]
        expect = {
            t: (home if t == GUM_HOME_TIER else
                aggregate_mode(home, aggregation_factor(t, GUM_HOME_TIER), nodata=NODATA))
            for t in tiers
        }
        values = _read_store_values(store_dir, layer_id, tiers)
        if spec["field"] == "lithology":
            valid = set(GUM_LITHOLOGY_CODES.values())
            enc_name = "GUM 岩性编码表（1..41）"
        else:
            valid = set(THICKNESS_CLASS_LABELS)
            enc_name = "GUM 厚度分级表（1..5，有序）"
        _class_checks(checks, values, expect, GUM_HOME_TIER, valid, enc_name)
        nlat, nlon = grid_shape(GUM_HOME_TIER)
        checks.append({
            "name": f"{GUM_HOME_TIER}: 重叠/未覆盖总量（报告制）",
            "status": "report",
            "summary": (
                f"多边形重叠像元 {overlap_cells}（规则：面积较小者胜）；"
                f"未覆盖像元 {uncovered_cells}/{nlat * nlon}"
                f"（GUM 覆盖约半数陆域 + 全部海洋）"
            ),
        })
        key = f"gum:{Path(src_dir or GUM_SRC_DIR).resolve()}"
        if key in _GUM_SOURCE_CACHE:
            gdf, _lith, _thick = _GUM_SOURCE_CACHE[key]
        else:
            gdf, _lith, _thick = load_gum(src_dir or GUM_SRC_DIR)
        from pyproj import Geod
        geod = Geod(ellps="WGS84")
        areas = gdf.geometry.apply(
            lambda geom: abs(geod.geometry_area_perimeter(geom)[0])
        )
        total_m6 = float(areas.sum()) / 1e12
        excl_m6 = float(areas[~gdf["XX"].isin(("Du", "Wu", "Wl", "Wr"))].sum()) / 1e12
        checks.append({
            "name": "测地面积 × 论文自述覆盖交叉核对（报告制）",
            "status": "report",
            "summary": (
                f"Σ多边形 {total_m6:.1f}×10⁶ km²；扣除冰（Du）与水体"
                f"（Wu/Wl/Wr）后 {excl_m6:.1f}×10⁶ km²（论文自述 68×10⁶，"
                f"差 {(excl_m6 - 68.0) / 68.0:+.1%} = 多边形重叠/数字化狭条）；"
                f"像元计数份额高于面积份额为高纬集中效应"
            ),
        })
        if spec["field"] == "thickness":
            checks.append({
                "name": "厚度有序类别仍按众数注记（报告制）",
                "status": "report",
                "summary": (
                    "厚度分级为有序类别（1=≤2m … 5=>200m），降采样仍按众数"
                    "（平票取小 = 偏薄侧，确定性）；保序聚合（如面积加权"
                    "中位数）留给下游按需重算——侧车 DD 原值列保留全部信息"
                ),
            })
        checks.append({"name": "陆域语义注记（报告制）", "status": "report",
                       "summary": _LANDSEA_NOTE})

    else:  # mooney
        gdf, codes = load_mooney_provinces(src_dir or MOONEY_SRC_DIR)
        home, overlap_cells, uncovered_cells = _mooney_home_classes(src_dir or MOONEY_SRC_DIR)
        expect = {
            t: (home if t == MOONEY_HOME_TIER else
                aggregate_mode(home, aggregation_factor(t, MOONEY_HOME_TIER), nodata=NODATA))
            for t in tiers
        }
        values = _read_store_values(store_dir, layer_id, tiers)
        _class_checks(checks, values, expect, MOONEY_HOME_TIER,
                      set(MOONEY_AGE_CODES.values()),
                      "Mooney 构造时代码表（1..6，自老至新）")
        _landmark_checks(checks, values[MOONEY_HOME_TIER], MOONEY_LANDMARKS, MOONEY_HOME_TIER)
        nlat, nlon = grid_shape(MOONEY_HOME_TIER)
        checks.append({
            "name": f"{MOONEY_HOME_TIER}: 重叠/未覆盖总量（报告制）",
            "status": "report",
            "summary": (
                f"多边形重叠像元 {overlap_cells}（规则：面积较小者胜；源重叠 "
                f"0.14 deg² = 0.0006%，时代省界数字化狭条）；未覆盖像元 "
                f"{uncovered_cells}/{nlat * nlon}（大陆域外 + 缝隙）"
            ),
        })
        from pyproj import Geod
        from .pixel_area import pixel_area_rows
        geod = Geod(ellps="WGS84")
        row_area = pixel_area_rows(MOONEY_HOME_TIER)
        parts = []
        for i, row in gdf.iterrows():
            attr_m6 = float(row["Area"])
            geod_m6 = abs(geod.geometry_area_perimeter(row.geometry)[0]) / 1e12
            rast = float((row_area[:, None] * (values[MOONEY_HOME_TIER] == codes[i])).sum())
            parts.append(
                f"{row['Age']}: attr={attr_m6:.3f} geod={geod_m6:.3f} "
                f"raster@3min={rast / 1e6:.3f}（10⁶ km²）"
            )
        checks.append({
            "name": f"{MOONEY_HOME_TIER}: 逐时代面积 × Age 属性交叉核对（报告制）",
            "status": "report",
            "summary": "; ".join(parts),
        })
        checks.append({
            "name": "构造稳定化时代语义 + 洋龄互补注记（报告制）",
            "status": "report",
            "summary": (
                "6 类为构造稳定化时代（tectonothermal stabilization age，"
                "Gubanov & Mooney 构造省），非出露地层时代；与 "
                "lithosphere__seafloor_age（Seton 2020，海洋层）覆盖域互补"
                "（陆/海），联合可得全球地壳年龄场"
            ),
        })
        checks.append({"name": "陆域语义注记（报告制）", "status": "report",
                       "summary": _LANDSEA_NOTE})

    n_fail = sum(1 for c in checks if c["status"] == "FAIL")
    return {
        "layer": layer_id,
        "store": str(store_dir),
        "integral_tolerance": INTEGRAL_TOL,
        "result": "FAIL" if n_fail else "PASS",
        "n_checks": len(checks),
        "checks": checks,
    }
