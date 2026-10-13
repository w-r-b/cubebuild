"""Lithosphere raster group: GEMMA crust three layers + LITHO1.0 LAB + Seton present-day seafloor age.

Source contract (disk-verified; coordinates/value ranges/coverage all checked on the actual machine):

- GEMMA (ecm1-gemma/*.tif, Reguzzoni & Sampietro 2015 JAG, GOCE inversion):
  crust_bottom / crust_moho_err / crust_top all three tifs are 360×720 float32
  0.5°, bounds exactly (-180,-90,180,90) — pixel registration, cell edges aligned to -180/-90
  (lon_phase=0.25), CRS declared EPSG:4326, no nodata and all 259200 pixels
  finite (global full coverage, validity mask all 1). Row 0 = north (top=90) → flipped
  to south-first at read time. Value range km (sea-level datum, negative downward): moho −105.19..−4.80
  (deepest at (34.25, 79.25) Karakoram — the deep orogenic-belt Moho inherent to the GOCE inversion);
  moho_err 1.60..10.23; basement (the crust_top source file) −21.00..+5.86
  (positive = crystalline basement surface above sea level). Cross-verified: crust_bottom ≤
  crust_top holds everywhere (0 violations), crystalline crustal thickness = top − bottom is
  0.007..110.64 km (left to a downstream one-line subtraction, by convention).

- LITHO1.0 (LITHO1.0.r0.0-n4c.nc, Pasyanos et al. 2014): 180×360
  float32, centers −89.5..89.5 / −179.5..179.5 (the ±.5 set), bit-identical to the cube 1° tier
  centers (the half values are exact in float32) — direct mapping without resampling; the source latitude is south-first,
  no flip needed. Four literal defects in the source latitude coordinates (rows 75/77/102/104 write ±14.5/±12.5
  as ±14.6/±12.6, a GeoCSV upstream typo pattern) — data-row alignment is cross-verified via ETOPO to be
  unaffected (the 4 rows' water_bottom vs. store bedrock_elevation have optimal row
  shift 0, correlation 0.98–0.99, same distribution as defect-free control rows), geometry parsed on the nominal regular
  grid, the defect values contractualized (LITHO1_COORD_DEFECTS).
  asthenospheric_mantle_top_depth is all 64800 finite (0 NaN), value range
  8.87..320.42 km; depth datum sea level, positive downward (cross-attested by water_top=0 in the Pacific /
  Tibet sedimentary top −5.03 km basement elevation) — opposite in sign to the GEMMA group's negative-downward,
  so the consumer must read value_convention. This variable is bit-identical to lid_bottom_depth
  (64800/64800) — the same interface under two names. The nc's other 170-variable inventory is written to the build
  report via write_litho1_inventory_report (basis for the v1.1 expansion).

- Seton seafloor age (Seton_etal_2020_PresentDay_AgeGrid.nc): GMT gridline
  node registration 3601×1801 (nodes −180..180 / −90..90 step 0.1; coordinates carry ~1e-11
  GMT floating-point accumulation error, geometry derived exactly from the node counts); ±180 columns bit-identical (both finite
  rows max|Δ|=0.0, NaN distribution consistent) → the +180 duplicate column is dropped; z float32
  _FillValue=NaN, missing 3331489/6485401 = 51.37% (land + COB mask);
  value range 0.01..338.68 Ma. The north-pole row (lat=90) is all finite with the single value 54.66 Ma (north-pole
  point oceanic crust, grdfill-type fill), the south-pole row all NaN. Node → Voronoi cell interpretation
  (same treatment as WGM2012): each node represents a center ±0.05 cell, pole rows clamped to half weight,
  longitude wraps; 0.1° gridline → 6′ pixel tier is therefore a 50/50 mix of adjacent nodes
  (a geometric necessity of the registration-type conversion, handled exactly by the conservative kernel); the disk has only the present-day grid and no
  time series (Time-dependent raster sequences is a third-party reference,
  recorded in manifest time_semantics).

Resampling: continuous-variable area-weighted conservative mean (kernels.conservative_overlap_mean,
generic kernel) — GEMMA 0.5°→30′ identity mapping / →1° 2×2 weighting; Seton
0.1°→6′ adjacent nodes half each (2 cells half weight) / →30′ 6 cells (4 full + 2 half) /
→1° 11 cells (9 full + 2 half), missing values normalized by valid source-cell area weight (coastal partially
covered pixels undiluted). LITHO1.0 1° direct mapping (resampling: none).

Fidelity (isomorphic to the gravmag three-layer criteria): ① store vs. source recompute bit-identical (bit-level hard criterion);
② integral identity Σ v·W = Σ v·w (source statistics computed independently, so kernel weight errors are exposed here);
③ quantile/coverage report-mode. GEMMA basement additionally reports its empirical basis,
LITHO1.0 additionally reports interface identity (lab ≡ lid_bottom bit-identical) and the variable inventory report.
"""

import json
from fractions import Fraction
from pathlib import Path

import dask.array as da
import numpy as np
import xarray as xr

from .contract import LayerEntry
from .fidelity import INTEGRAL_TOL, fidelity_checks, hist_add
from .gravmag import RasterSource, gravmag_source_stats, hist_bins
from .grids import tier_centers, tier_chunks, tier_fraction
from .kernels import conservative_overlap_mean
from .pixel_area import EARTH_RADIUS_KM

REPO_ROOT = Path(__file__).resolve().parents[1]
GEMMA_SRC_DIR = REPO_ROOT / "original data/lithosphere/crust-models/ecm1-gemma"
LITHO1_SRC_DIR = REPO_ROOT / "original data/lithosphere/litho1"
LITHO1_FILE = "LITHO1.0.r0.0-n4c.nc"
SETON_SRC_DIR = (
    REPO_ROOT / "original data/lithosphere/seafloor-age-seton2020"
    / "Rasters/Seafloor_Age_Grid"
)
SETON_FILE = "Seton_etal_2020_PresentDay_AgeGrid.nc"

GEMMA_DOI = "10.1016/j.jag.2014.04.002"
LITHO1_DOI = "10.1002/2013JB010626"
SETON_DOI = "10.1029/2020GC009214"             # Seton et al. 2020, G³

GEMMA_HALF = Fraction(1, 2)
SETON_TENTH = Fraction(1, 10)

GEMMA_MOHO_ID = "lithosphere__gemma_moho"
GEMMA_MOHO_ERR_ID = "lithosphere__gemma_moho_err"
GEMMA_BASEMENT_ID = "lithosphere__gemma_basement_depth"
LITHO1_LAB_ID = "lithosphere__litho1_lab"
SEAFLOOR_AGE_ID = "lithosphere__seafloor_age"

GEMMA_FILES = {
    GEMMA_MOHO_ID: "crust_bottom.tif",
    GEMMA_MOHO_ERR_ID: "crust_moho_err.tif",
    GEMMA_BASEMENT_ID: "crust_top.tif",
}

LITHO1_LAB_VAR = "asthenospheric_mantle_top_depth"
LITHO1_LID_VAR = "lid_bottom_depth"

LITHO1_COORD_DEFECTS = {
    75: float(np.float32(-14.6)),
    77: float(np.float32(-12.6)),
    102: float(np.float32(12.6)),
    104: float(np.float32(14.6)),
}

GEMMA_EXPECTED = {
    "shape": (360, 720),
    "res": 0.5,
    "bounds": (-180.0, -90.0, 180.0, 90.0),
}
LITHO1_EXPECTED = {"shape": (180, 360)}
SETON_EXPECTED = {"shape": (1801, 3601)}

GEMMA_VALUE_RANGES = {
    GEMMA_MOHO_ID: (-105.194, -4.805),
    GEMMA_MOHO_ERR_ID: (1.597, 10.231),
    GEMMA_BASEMENT_ID: (-21.004, 5.862),
}

_BASEMENT_EVIDENCE = (
    "crust_top.tif 实证 = 结晶基底面：值域 −21.00..+5.86 km（正值 = 基底面"
    "高于海平面，安第斯/青藏基底隆起）排除「高程」解释（高程场不可能全域"
    "≤6 km），亦排除「沉积底界」解释（深海平原基底远深于 −5 km，与 GEMMA "
    "论文参数化一致：crust_top 为结晶地壳顶界，沉积另由 GST-1 层承担）；"
    "crust_bottom ≤ crust_top 全场成立（0 违例）；结晶地壳厚度（存储符号"
    "约定下）= gemma_basement_depth − gemma_moho（两层均负值向下，相减"
    "结果 ≥ 0，实测 0.007..110.64 km），下游一行减法可得（约定不在"
    "立方内展开）"
)

_GEMMA_SIGN_NOTE = (
    "km，海平面为 0，负值向下（GEMMA 输出原生符号约定，源值保真不做"
    "符号翻转；与 lithosphere__litho1_lab 的正值向下相反，消费方注意）"
)



def load_gemma_tif(filename: str, src_dir: str | Path = GEMMA_SRC_DIR) -> RasterSource:
    """Read a GEMMA crust tif → normalized source (row 0 = -90 side, column 0 = -180 side).

    Assertions (contractualized from disk evidence): 360×720, 0.5°, bounds exactly ±180/±90
    (pixel registration = cell edges aligned to -180/-90), CRS 4326 (declared or untagged),
    all pixels finite (disk-verified no nodata and no NaN; a source change fails explicitly).
    The north-first row order is flipped to south-first, and self-consistently checked by randomly sampling original rows/columns (a capturer of
    flipped-direction misalignment): raw(r,c) center = (90−(r+0.5)·res, −180+(c+0.5)·res)
    ↔ norm(nlat−1−r, c).
    """
    import rasterio

    path = Path(src_dir or GEMMA_SRC_DIR) / filename
    if not path.is_file():
        raise FileNotFoundError(f"GEMMA 源文件不存在: {path}")
    with rasterio.open(path) as src:
        if src.shape != GEMMA_EXPECTED["shape"]:
            raise ValueError(
                f"{filename}: 形状 {src.shape} ≠ 期望 {GEMMA_EXPECTED['shape']}"
            )
        res = GEMMA_EXPECTED["res"]
        if abs(src.res[0] - res) > 1e-9 or src.res[0] != src.res[1]:
            raise ValueError(f"{filename}: 分辨率 {src.res} ≠ 0.5°")
        b = src.bounds
        eb = GEMMA_EXPECTED["bounds"]
        if max(abs(b.left - eb[0]), abs(b.bottom - eb[1]),
               abs(b.right - eb[2]), abs(b.top - eb[3])) > 1e-6:
            raise ValueError(
                f"{filename}: 包络 {tuple(b)} ≠ 期望 {eb}（pixel 注册、边对齐 ±180/±90）"
            )
        if src.crs is not None and src.crs.to_epsg() != 4326:
            raise ValueError(f"{filename}: 意外 CRS {src.crs}（预期 EPSG:4326 或无标签）")
        a = src.read(1)

    if not np.isfinite(a).all():
        raise ValueError(f"{filename}: 存在非有限值（实证全覆盖无缺测，源已变更）")

    nlat, nlon = a.shape
    values = a[::-1].astype(np.float32)
    rng = np.random.default_rng(0)
    for _ in range(64):
        r, c = int(rng.integers(nlat)), int(rng.integers(nlon))
        if values[nlat - 1 - r, c] != a[r, c]:
            raise ValueError(f"{filename}: 翻转自洽核对失败 @ raw({r},{c})")

    lat_edges = -90.0 + np.arange(nlat + 1, dtype=np.float64) * res
    return RasterSource(
        values=values,
        valid=np.isfinite(values),
        lat_edges=lat_edges,
        lon_res=GEMMA_HALF,
        lon_phase=GEMMA_HALF / 2,
        attrs={
            "registration_rule": (
                "pixel 注册（0.5° 中心 ±0.25，bounds 恰 ±180/±90 = 源胞边"
                "对齐 -180/-90；保守核 lon_phase=0.25，30′ 档恒等映射）；"
                "源北起行序翻转为南起"
            ),
            "vertical_datum": "海平面（GEMMA 输出深度基准）",
            "source_file": filename,
        },
    )



def load_litho1_lab(src_dir: str | Path | None = LITHO1_SRC_DIR) -> RasterSource:
    """Read the LAB (asthenosphere-top depth) from the LITHO1.0 nc → normalized source (1° direct-mapping tier).

    Assertions (contractualized from disk evidence): 180×360; center coordinates bit-identical to the cube 1° tier
    centers (the float32 half values exact; latitude south-first, no flip) — with the known 4-row literal coordinate defect
    (LITHO1_COORD_DEFECTS, data-row alignment cross-verified via ETOPO, geometry parsed on the nominal
    grid); LAB all pixels finite (0 NaN measured); LAB and lid_bottom_depth
    bit-identical (interface identity: asthenosphere top ≡ lithospheric lid bottom, the same interface under two names).
    """
    import netCDF4

    path = Path(src_dir or LITHO1_SRC_DIR) / LITHO1_FILE
    if not path.is_file():
        raise FileNotFoundError(f"LITHO1.0 源文件不存在: {path}")
    ds = netCDF4.Dataset(path)
    try:
        if ds.variables[LITHO1_LAB_VAR].shape != LITHO1_EXPECTED["shape"]:
            raise ValueError(
                f"{LITHO1_FILE}: {LITHO1_LAB_VAR} 形状 "
                f"{ds.variables[LITHO1_LAB_VAR].shape} ≠ 期望 {LITHO1_EXPECTED['shape']}"
            )
        lat = np.asarray(ds.variables["latitude"][:], dtype=np.float64)
        lon = np.asarray(ds.variables["longitude"][:], dtype=np.float64)
        lab = np.ma.filled(ds.variables[LITHO1_LAB_VAR][:], np.nan).astype(np.float32)
        lid = np.ma.filled(ds.variables[LITHO1_LID_VAR][:], np.nan).astype(np.float32)
    finally:
        ds.close()

    exp_lat, exp_lon = tier_centers("1deg")
    bad_lat = np.where(lat != exp_lat)[0]
    unexpected = [
        int(i) for i in bad_lat
        if LITHO1_COORD_DEFECTS.get(int(i)) != lat[i]
    ]
    if unexpected or not np.array_equal(lon, exp_lon):
        raise ValueError(
            f"{LITHO1_FILE}: 中心坐标偏离立方 1° 档名义网格（未契约化的"
            f"纬度行 {unexpected}；经度一致 {np.array_equal(lon, exp_lon)}）"
        )
    if not np.isfinite(lab).all():
        raise ValueError(f"{LITHO1_FILE}: {LITHO1_LAB_VAR} 含非有限值（实证 0 NaN，源已变更）")
    if not np.array_equal(lab, lid):
        raise ValueError(
            f"{LITHO1_FILE}: {LITHO1_LAB_VAR} 与 {LITHO1_LID_VAR} 不再逐位相同"
            "（界面同一性契约被破坏，须复核 LAB 变量选择）"
        )

    return RasterSource(
        values=lab,
        valid=np.ones(lab.shape, dtype=bool),
        lat_edges=-90.0 + np.arange(181, dtype=np.float64),
        lon_res=Fraction(1),
        lon_phase=Fraction(1, 2),
        attrs={
            "registration_rule": (
                "1° pixel 注册（中心 ±.5 集，纬度南起免翻转），与立方 1° 档"
                "中心逐位一致——免重采样直映（resampling=none）；源纬度坐标"
                "4 行字面缺陷（±14.5/±12.5 写作 ±14.6/±12.6，GeoCSV 上游"
                "笔误）经 ETOPO 交叉实证数据行对齐不受影响（4 行最优行移位 0、"
                "相关 0.98–0.99），几何按名义规则网格解析，缺陷值契约化；"
                f"{LITHO1_LAB_VAR} 与 {LITHO1_LID_VAR} 逐位相同（软流圈顶界 ≡ "
                "岩石圈盖底界，同一界面双命名，读取期契约化）"
            ),
            "vertical_datum": "海平面（water_top=0 于太平洋 / Tibet 沉积顶 −5.03 km 双地标互证）",
            "source_file": LITHO1_FILE,
        },
    )



def load_seafloor_age(src_dir: str | Path | None = SETON_SRC_DIR) -> RasterSource:
    """Read the Seton present-day seafloor-age nc → normalized source (gridline node → Voronoi cell).

    Assertions (contractualized from disk evidence): 3601×1801 nodes (−180..180/−90..90 step
    0.1, coordinates carry ~1e-11 GMT floating-point accumulation error, tolerance 1e-8 anchoring the regular grid);
    ±180 columns bit-identical (tolerance at the float32 noise level 1e-3 Ma) → drop the +180 duplicate column;
    Voronoi cell edges = node ±0.05 (pole rows clamped to half weight).
    """
    import netCDF4

    path = Path(src_dir or SETON_SRC_DIR) / SETON_FILE
    if not path.is_file():
        raise FileNotFoundError(f"Seton 洋龄源文件不存在: {path}")
    ds = netCDF4.Dataset(path)
    try:
        x = np.asarray(ds.variables["x"][:], dtype=np.float64)
        y = np.asarray(ds.variables["y"][:], dtype=np.float64)
        z = np.ma.filled(ds.variables["z"][:], np.nan).astype(np.float32)
    finally:
        ds.close()

    ny, nx = SETON_EXPECTED["shape"]
    if z.shape != (ny, nx) or x.shape != (nx,) or y.shape != (ny,):
        raise ValueError(f"{SETON_FILE}: 形状 {z.shape}/{x.shape}/{y.shape} ≠ 期望 {(ny, nx)}")
    lon_res = Fraction(360, nx - 1)
    if lon_res != SETON_TENTH:
        raise ValueError(f"{SETON_FILE}: 经度步长 {lon_res} ≠ 期望 {SETON_TENTH}")
    if abs(float(Fraction(180, ny - 1)) - 0.1) > 1e-12:
        raise ValueError(f"{SETON_FILE}: 纬度步长 {(180 / (ny - 1))!r} ≠ 0.1°")
    for arr, start, n in ((x, -180.0, nx), (y, -90.0, ny)):
        expect = start + np.arange(n) * float(SETON_TENTH)
        if np.max(np.abs(arr - expect)) > 1e-8:
            raise ValueError(
                f"{SETON_FILE}: 节点坐标偏离规则网格（首末 {arr[0]}, {arr[-1]}）"
            )
    both = np.isfinite(z[:, 0]) & np.isfinite(z[:, -1])
    if int((np.isnan(z[:, 0]) != np.isnan(z[:, -1])).sum()) > 0:
        raise ValueError(f"{SETON_FILE}: ±180 列 NaN 分布不一致，周期重复假设被破坏")
    if both.any() and np.max(np.abs(
        z[both, 0].astype(np.float64) - z[both, -1].astype(np.float64)
    )) > 1e-3:
        raise ValueError(f"{SETON_FILE}: ±180 列不同值，周期重复假设被破坏")

    values = z[:, :-1]
    valid = np.isfinite(values)
    lat_edges = -90.0 + (np.arange(ny + 1, dtype=np.float64) - 0.5) * float(SETON_TENTH)
    lat_edges[0], lat_edges[-1] = -90.0, 90.0
    return RasterSource(
        values=values,
        valid=valid,
        lat_edges=lat_edges,
        lon_res=SETON_TENTH,
        attrs={
            "registration_rule": (
                "gridline 节点注册（节点 −180..180/−90..90 步 0.1）→ 节点代表"
                " Voronoi 胞（中心 ±0.05，经度环接日期线；极点行钳制半权；"
                "北极行单值 54.66 Ma、南极行全 NaN 为源固有）；±180 重复列弃除；"
                "0.1°→6′ 档为相邻节点 50/50 混合（gridline→pixel 注册转换的"
                "几何必然，保守核精确处理）"
            ),
            "vertical_datum": "不适用（年龄场）",
            "source_file": SETON_FILE,
        },
    )



def litho1_variable_inventory(src_dir: str | Path | None = LITHO1_SRC_DIR) -> dict:
    """Full LITHO1.0 nc variable inventory (basis for the v1.1 expansion; read at build time, deterministic, no timestamps).

    171 data variables indexed by "interface layer × property": water/ice/upper-middle-lower
    sediments/upper-middle-lower crust/lid/asthenospheric_mantle, 10 layer groups
    × top/bottom interfaces (asthenospheric_mantle has top only, the rest 2 each) × 9 properties
    (depth/density/vp/vs/qkappa/qmu/vp2/vs2/eta). This module ingests only the
    LAB variable; the other 170 are v1.1 candidates.
    """
    import netCDF4

    path = Path(src_dir or LITHO1_SRC_DIR) / LITHO1_FILE
    if not path.is_file():
        raise FileNotFoundError(f"LITHO1.0 源文件不存在: {path}")
    ds = netCDF4.Dataset(path)
    try:
        variables = [
            {
                "name": name,
                "long_name": str(v.getncattr("long_name")),
                "units": str(v.getncattr("units")),
            }
            for name, v in ds.variables.items()
            if name not in ("latitude", "longitude")
        ]
    finally:
        ds.close()

    index: dict[str, dict[str, dict[str, str]]] = {}
    for item in variables:
        parts = item["name"].rsplit("_", 2)
        if len(parts) != 3 or parts[1] not in ("top", "bottom"):
            raise ValueError(
                f"{LITHO1_FILE}: 变量名 {item['name']!r} 不符合 "
                "{{layer}}_{{top|bottom}}_{{property}} 命名契约"
            )
        layer, interface, prop = parts
        index.setdefault(layer, {}).setdefault(interface, {})[prop] = item["name"]

    ingested = {LITHO1_LAB_ID: LITHO1_LAB_VAR}
    return {
        "source_file": LITHO1_FILE,
        "doi": LITHO1_DOI,
        "n_variables": len(variables),
        "ingested": ingested,
        "n_v1_1_candidates": len(variables) - len(ingested),
        "variables": variables,
        "layer_interface_index": {
            layer: {
                interface: dict(sorted(props.items()))
                for interface, props in sorted(interfaces.items())
            }
            for layer, interfaces in sorted(index.items())
        },
        "v1_1_note": (
            "v1.0 仅收 LAB（asthenospheric_mantle_top_depth，与 lid_bottom_depth "
            "逐位相同）；密度/vp/vs/qkappa/qmu/vp2/vs2/eta 等 170 变量为 v1.1 "
            "展开候选，界面层 × 物性索引见 layer_interface_index"
        ),
    }


def write_litho1_inventory_report(reports_dir: str | Path,
                                  src_dir: str | Path | None = None) -> Path:
    """Writes the variable-inventory report to disk (the other-variable inventory goes into the build report; deterministic, no timestamps)."""
    inventory = litho1_variable_inventory(src_dir or LITHO1_SRC_DIR)
    path = Path(reports_dir) / "litho1-inventory.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return path



_SOURCE_CACHE: dict[str, RasterSource] = {}


def _cached_load(key: str, loader, src_dir=None) -> RasterSource:
    """In-process source cache: building one layer across several tiers reads the source file only once (loaders uniformly take src_dir,
    None = each loader's default path). The key includes the resolved src_dir (the gravmag/thermal
    convention) — mixing custom/default paths in one process will not fetch a stale source."""
    cache_key = key if src_dir is None else f"{key}:{Path(src_dir).resolve()}"
    if cache_key not in _SOURCE_CACHE:
        _SOURCE_CACHE[cache_key] = loader(src_dir)
    return _SOURCE_CACHE[cache_key]


def _gemma_source(layer_id: str, src_dir=None) -> RasterSource:
    filename = GEMMA_FILES[layer_id]
    return _cached_load(
        f"gemma:{layer_id}",
        lambda sd: load_gemma_tif(filename, sd),
        src_dir,
    )


def _build_gemma_layer(entry: LayerEntry, tier: str, layer_id: str,
                       long_name: str, extra_attrs: dict | None = None) -> xr.DataArray:
    src = _gemma_source(layer_id)
    values, _W = conservative_overlap_mean(
        src.values, src.valid, src.lat_edges, src.lon_res, tier_fraction(tier),
        lon_phase=src.lon_phase,
    )
    arr = da.from_array(values, chunks=tier_chunks(tier))
    lat, lon = tier_centers(tier)
    attrs = {
        "units": entry.unit,
        "long_name": long_name,
        "source": entry.source,
        "doi": GEMMA_DOI,
        "native_res": entry.native_res,
        "resampling": entry.resampling,
        "visibility": entry.visibility,
        "earth_radius_km": EARTH_RADIUS_KM,
        "value_convention": _GEMMA_SIGN_NOTE,
    }
    attrs.update(src.attrs)
    if extra_attrs:
        attrs.update(extra_attrs)
    return xr.DataArray(
        arr, dims=("lat", "lon"), coords={"lat": lat, "lon": lon},
        name=entry.id, attrs=attrs,
    )


def build_gemma_moho(entry: LayerEntry, tier: str) -> xr.DataArray:
    """Build this tier's lithosphere__gemma_moho (the cube's sole Moho layer; negative downward)."""
    return _build_gemma_layer(
        entry, tier, GEMMA_MOHO_ID,
        long_name="莫霍面深度（GEMMA，GOCE 反演；海平面基准，负值向下）",
        extra_attrs={
            "product_type": "model（GOCE 卫星重力反演地壳模型，非地震观测）",
        },
    )


def build_gemma_moho_err(entry: LayerEntry, tier: str) -> xr.DataArray:
    """Build this tier's lithosphere__gemma_moho_err (Moho depth uncertainty, for ML weighting)."""
    return _build_gemma_layer(
        entry, tier, GEMMA_MOHO_ERR_ID,
        long_name="莫霍面深度不确定度（GEMMA；供 ML 不确定性加权）",
        extra_attrs={
            "product_type": "model（GOCE 反演方差传播，非地震观测）",
            "usage_note": "与 lithosphere__gemma_moho 配对使用：ML 损失加权 1/σ² 类用途",
            "value_convention": "km，Moho 深度 1σ 不确定度（严格正值）",
        },
    )


def build_gemma_basement_depth(entry: LayerEntry, tier: str) -> xr.DataArray:
    """Build this tier's lithosphere__gemma_basement_depth (crystalline basement surface; negative downward)."""
    return _build_gemma_layer(
        entry, tier, GEMMA_BASEMENT_ID,
        long_name="结晶基底面深度（GEMMA crust_top；海平面基准，负值向下）",
        extra_attrs={
            "product_type": "model（GOCE 卫星重力反演地壳模型，非地震观测）",
            "basement_evidence": _BASEMENT_EVIDENCE,
        },
    )


def build_litho1_lab(entry: LayerEntry, tier: str) -> xr.DataArray:
    """Build the 1° tier's lithosphere__litho1_lab (direct mapping without resampling, resampling=none)."""
    if tier != "1deg":
        raise ValueError(f"{LITHO1_LAB_ID}: 仅 1° 档（契约 tiers=[1deg]），收到 {tier!r}")
    src = _cached_load("litho1", load_litho1_lab)
    arr = da.from_array(src.values, chunks=tier_chunks(tier))
    lat, lon = tier_centers(tier)
    attrs = {
        "units": entry.unit,
        "long_name": "岩石圈—软流圈界面深度（LITHO1.0，软流圈顶界；正值向下）",
        "source": entry.source,
        "doi": LITHO1_DOI,
        "native_res": entry.native_res,
        "resampling": entry.resampling,
        "visibility": entry.visibility,
        "earth_radius_km": EARTH_RADIUS_KM,
        "value_convention": (
            "km，海平面为 0，正值向下（LITHO1.0 原生符号约定；与 GEMMA "
            "地壳组的负值向下相反，消费方注意符号）"
        ),
        "license_note": "CC BY 4.0（作者明示可再分发）",
    }
    attrs.update(src.attrs)
    return xr.DataArray(
        arr, dims=("lat", "lon"), coords={"lat": lat, "lon": lon},
        name=entry.id, attrs=attrs,
    )


def build_seafloor_age(entry: LayerEntry, tier: str) -> xr.DataArray:
    """Build this tier's lithosphere__seafloor_age (present-day seafloor age; oceanic layer, land NaN)."""
    src = _cached_load("seton", load_seafloor_age)
    values, _W = conservative_overlap_mean(
        src.values, src.valid, src.lat_edges, src.lon_res, tier_fraction(tier),
        lon_phase=src.lon_phase,
    )
    arr = da.from_array(values, chunks=tier_chunks(tier))
    lat, lon = tier_centers(tier)
    attrs = {
        "units": entry.unit,
        "long_name": "洋壳年龄（Seton et al. 2020 现今年龄网格；海洋层，陆地缺测）",
        "source": entry.source,
        "doi": SETON_DOI,
        "native_res": entry.native_res,
        "resampling": entry.resampling,
        "visibility": entry.visibility,
        "earth_radius_km": EARTH_RADIUS_KM,
        "value_convention": "Ma（洋壳年龄，0 = 现今扩张脊）",
        "time_semantics": (
            "仅现今（Present Day）洋壳年龄网格；磁盘无时间序列"
            "（Time-dependent raster sequences/ 为第三方 MIT-P08 参考，"
            "该第三方参考不收）——manifest 注记无时间序列"
            ""
        ),
        "nodata_semantics": (
            "NaN = 非洋壳（陆地 + COB 掩膜边缘海）；与 derived__landsea_mask "
            "组合的海洋层语义由消费方完成"
        ),
    }
    attrs.update(src.attrs)
    return xr.DataArray(
        arr, dims=("lat", "lon"), coords={"lat": lat, "lon": lon},
        name=entry.id, attrs=attrs,
    )



LITHO_FIDELITY_LAYERS = {
    GEMMA_MOHO_ID: {
        "kind": "gemma",
        "hist_spec": (-110.0, 0.0, 0.5),
        "unit": "km",
    },
    GEMMA_MOHO_ERR_ID: {
        "kind": "gemma",
        "hist_spec": (1.0, 11.0, 0.05),
        "unit": "km",
    },
    GEMMA_BASEMENT_ID: {
        "kind": "gemma",
        "hist_spec": (-22.0, 7.0, 0.25),
        "unit": "km",
    },
    LITHO1_LAB_ID: {
        "kind": "litho1",
        "hist_spec": (8.0, 322.0, 2.0),
        "unit": "km",
    },
    SEAFLOOR_AGE_ID: {
        "kind": "seton",
        "hist_spec": (0.0, 340.0, 1.0),
        "unit": "Ma",
    },
}

_KIND_LOADERS = {
    "gemma": lambda layer_id, src_dir: _gemma_source(layer_id, src_dir),
    "litho1": lambda layer_id, src_dir: _cached_load(
        "litho1", load_litho1_lab, src_dir
    ),
    "seton": lambda layer_id, src_dir: _cached_load(
        "seton", load_seafloor_age, src_dir
    ),
}


def _expected_tier(src: RasterSource, kind: str, tier: str) -> tuple[np.ndarray, np.ndarray]:
    """Expected values and valid weights for this tier: gemma/seton via conservative-kernel recompute; litho1 direct-mapped
    (W = the target cell's analytic row area — the integral weight of a fully covered direct-mapped layer)."""
    if kind == "litho1":
        from .pixel_area import pixel_area_rows

        W = np.repeat(pixel_area_rows(tier)[:, None], src.values.shape[1], axis=1)
        return src.values.copy(), W
    return conservative_overlap_mean(
        src.values, src.valid, src.lat_edges, src.lon_res, tier_fraction(tier),
        lon_phase=src.lon_phase,
    )


def build_litho_fidelity_report(store_dir, layer_id: str, tiers: list[str],
                                src_dir=None) -> dict:
    """Fidelity report (isomorphic to the gravmag three-layer criteria): ① store vs. source recompute bit-identical (a bit-level
    hard criterion) + ② valid-mask consistency + ③ integral identity (an independent source-statistics path) + ④ quantile/
    coverage (report-mode); plus specific notes: GEMMA basement empirical basis, LITHO1.0
    interface identity and sign-convention notes.
    """
    spec = LITHO_FIDELITY_LAYERS[layer_id]
    kind = spec["kind"]
    src = _KIND_LOADERS[kind](layer_id, src_dir)
    hist_spec = spec["hist_spec"]
    unit = spec["unit"]
    src_stats = gravmag_source_stats(src, hist_spec)

    checks: list[dict] = []
    with xr.open_datatree(store_dir, engine="zarr", chunks={}) as tree:
        for tier in tiers:
            node = tree[f"/{tier}"]
            if layer_id not in node.ds.data_vars:
                raise KeyError(f"store {store_dir} 的 {tier} 档不含 {layer_id}（先 build 再 fidelity）")
            v = node.ds[layer_id].values
            expect, W = _expected_tier(src, kind, tier)
            ok = W > 0

            eq = (v == expect) | (np.isnan(v) & np.isnan(expect))
            n_bad = int((~eq).sum())
            checks.append({
                "name": f"{tier}: 与源{'直映' if kind == 'litho1' else '核'}重算逐位一致（float32 位级）",
                "status": "pass" if n_bad == 0 else "FAIL",
                "summary": f"不一致像元 {n_bad}（期望 0）",
            })

            mism = int((np.isfinite(v) != ok).sum())
            checks.append({
                "name": f"{tier}: 有效位与源掩膜复算一致（逐位）",
                "status": "pass" if mism == 0 else "FAIL",
                "summary": f"不一致 {mism} 像元（期望 0）",
            })

            tier_stats = {"sum_va": float(np.dot(v[ok].astype(np.float64), W[ok]))}
            lo, hi, bw = hist_spec
            tier_stats["hist"] = hist_add(
                np.zeros(hist_bins(hist_spec), dtype=np.uint64),
                v[np.isfinite(v)], lo=lo, hi=hi, bin_width=bw,
            )
            checks.extend(fidelity_checks(src_stats, tier_stats, tier, hist_spec=hist_spec, unit=unit))

            checks.append({
                "name": f"{tier}: 覆盖率（报告制）",
                "status": "report",
                "summary": f"档 {ok.mean():.4f} vs 源（像元计）{src.valid.mean():.4f}",
            })

    if layer_id == GEMMA_BASEMENT_ID:
        checks.append({
            "name": "basement 实证依据（报告制）",
            "status": "report",
            "summary": _BASEMENT_EVIDENCE,
        })
    if layer_id == LITHO1_LAB_ID:
        checks.append({
            "name": "LAB 界面同一性（读取期契约化，报告制）",
            "status": "report",
            "summary": (
                "asthenospheric_mantle_top_depth 与 lid_bottom_depth 逐位相同"
                "（64800/64800，磁盘实证；读取期断言，源漂移即显式失败）"
            ),
        })
        checks.append({
            "name": "符号约定注记（报告制）",
            "status": "report",
            "summary": (
                "LITHO1.0 正值向下 vs GEMMA 地壳组负值向下——两族符号相反，"
                "跨层联用（如 LAB − Moho 求岩石圈地幔厚度）须先统一符号"
            ),
        })
    if layer_id == SEAFLOOR_AGE_ID:
        checks.append({
            "name": "时间序列注记（报告制）",
            "status": "report",
            "summary": (
                "磁盘仅现今网格（PresentDay），无时间序列；Time-dependent "
                "raster sequences/ 为第三方 MIT-P08 参考"
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
