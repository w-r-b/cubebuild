"""Gravity/magnetics group: WGM2012 Bouguer gravity anomaly + EMAG2 v3 sea-level magnetic anomaly.

Source geometry (disk-verified; coordinate origin and grid definition both
checked on real hardware):

- WGM2012_Bouguer_ponc_2min.grd (BGI): COARDS netCDF, node_offset=0 gridline
  node registration; x 10801 nodes (-180..180 step 1/30, ±180 columns equal
  — periodic duplication, measured max difference 1e-4 mGal); y 5401 nodes
  (-90..90); z a complete global field (0 NaN; "ponc" nominal point set, the
  BGI distributed version is already a complete grid), value range [-529, 1005]
  mGal. License: BGI Terms are effectively NC → visibility=internal (not
  included in the public version).
  Node → cell interpretation (registration_rule): each node represents its
  Voronoi cell (center ±1/60), latitude clamped to ±90 (polar rows half weight),
  longitude wraps around the dateline. Rationale: a node is a point sample of
  the field, and a Voronoi cell is the minimal-bias discretization for
  conservative averaging; reordering the dropped edge column would introduce a
  half-cell (1/60°≈1.8km) systematic misregistration.

- EMAG2_V3_20170530_Sealevel.tif (NOAA NCEI): 10800×5399 float32, GeoTIFF with
  no CRS tag — registration explicitly assigns EPSG:4326 (the official definition
  is a geographic lat/lon grid; if the file carries a CRS it must be exactly
  4326, otherwise reject). transform measured: longitude 0..360 frame (column
  centers = 0, 1/30, …, 359.9667; origin at Greenwich, not -180), latitude cell
  edges ±89.98333 (polar cap 1/60° has no data). nodata = float32 minimum
  (-3.4e38), 47.9% missing. The two seams at 0|359.9667 and 180|179.9667 have
  differences distributed the same as the interior baseline (p50 3.2/4.3 nT vs
  interior 3.3) — an independent corroboration of the 0..360 frame reading.
  Wrapping normalization: new[j] = old[(j+nlon/2) % nlon] (longitude half wrap)
  + row flip to south-first (store row 0 = -90 side), cross-checked for
  self-consistency by randomly sampling original rows/columns.

Resampling: continuous variables use area-weighted conservative mean, the
general overlap kernel (kernels.conservative_overlap_mean) — the 2min→3min
fractional factor (1.5) is handled exactly, and missing data is normalized by
effective source-cell area weight. Both sources have four tiers (1°/30′/6′/3′);
WGM is fully covered (mask all 1), the EMAG mask carries the 47.9% missing
structure.

Fidelity: three criteria — ① the store tier array is bit-identical to the source
kernel recomputation (bit-level hard criterion, same philosophy as derived
layers); ② the integral identity Σ_t v_t·W_t = Σ_s v_s·w_s, with source stats
computed independently and directly (not through the kernel path, so a kernel
weight error is exposed here); ③ quantile/coverage reporting only. Row-flip /
longitude-rotation misregistrations are insensitive to ① and ② (self-consistent
misregistration), and are covered by the unit tests' analytic linear field
(direction exactly assertable) and the reader's sample self-consistency check
(raw rows/columns ↔ normalized array); an extremum-location check was found
empirically unusable (the source extremum is a single 2min cell spike, and after
coarse-tier conservative averaging the argmax drifts by tens of cells, which is
a physical necessity of smoothing).
"""

from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import dask.array as da
import numpy as np
import xarray as xr

from .contract import LayerEntry
from .fidelity import INTEGRAL_TOL, fidelity_checks, hist_add
from .grids import tier_centers, tier_chunks, tier_fraction
from .kernels import conservative_overlap_mean
from .pixel_area import EARTH_RADIUS_KM

REPO_ROOT = Path(__file__).resolve().parents[1]
WGM_SRC_DIR = REPO_ROOT / "original data/gravity/bouguer-wgm2012"
WGM_FILE = "WGM2012_Bouguer_ponc_2min.grd"
EMAG_SRC_DIR = REPO_ROOT / "original data/magnetics/emag2-v3"
EMAG_FILE = "EMAG2_V3_20170530_Sealevel.tif"

WGM_DOI = "10.1007/s00190-011-0533-4"
EMAG_DOI = "10.1002/2017GC007280"

TWO_MIN = Fraction(1, 30)

WGM_EXPECTED = {"shape": (5401, 10801), "spacing": TWO_MIN}
EMAG_EXPECTED = {
    "shape": (5399, 10800),
    "res": 1.0 / 30.0,
    "bounds": (-1.0 / 60.0, -90.0 + 1.0 / 60.0, 360.0 - 1.0 / 60.0, 90.0 - 1.0 / 60.0),
}


@dataclass
class RasterSource:
    """Normalized global raster source: row 0 = -90 side, column 0 = -180 side, NaN = missing.

    lon_phase: longitude phase of source-cell centers relative to −180 + k·lon_res
    (degrees, Fraction; intended for use by HFgrid14-like 0.25-offset pixel-
    registration grids). Default 0 = centers at integer multiples (the WGM node
    Voronoi / EMAG2 post-wrapping form).
    """

    values: np.ndarray            # (nlat, nlon) float32
    valid: np.ndarray             # (nlat, nlon) bool
    lat_edges: np.ndarray
    lon_res: Fraction
    lon_phase: Fraction = Fraction(0)
    attrs: dict = field(default_factory=dict)

    @property
    def lat_centers(self) -> np.ndarray:
        return 0.5 * (self.lat_edges[:-1] + self.lat_edges[1:])


def _source_cell_areas(src: RasterSource) -> np.ndarray:
    """Per-row source-cell areas (km²): R²·Δsin·Δλ_rad — the w_s of the integral identity (same convention as the kernel)."""
    dsin = np.sin(np.deg2rad(src.lat_edges[1:])) - np.sin(np.deg2rad(src.lat_edges[:-1]))
    return EARTH_RADIUS_KM**2 * dsin * float(src.lon_res) * np.pi / 180.0



def load_wgm2012(src_dir: str | Path = WGM_SRC_DIR) -> RasterSource:
    """Read the WGM2012 Bouguer anomaly grd → normalized source (node → Voronoi cell interpretation).

    Assertions (contractualized disk evidence): gridline registration, node
    coordinates ±180/±90 with a 1/30 step, ±180 columns periodically equal (tolerance
    1e-2 mGal = float32 noise magnitude).
    """
    import netCDF4

    path = Path(src_dir) / WGM_FILE
    if not path.is_file():
        raise FileNotFoundError(f"WGM2012 源文件不存在: {path}")
    ds = netCDF4.Dataset(path)
    try:
        if int(ds.getncattr("node_offset")) != 0:
            raise ValueError(f"{WGM_FILE}: node_offset≠0，非 gridline 节点注册")
        x = np.asarray(ds.variables["x"][:], dtype=np.float64)
        y = np.asarray(ds.variables["y"][:], dtype=np.float64)
        z = np.asarray(ds.variables["z"][:], dtype=np.float32)
    finally:
        ds.close()

    ny, nx = WGM_EXPECTED["shape"]
    if z.shape != (ny, nx) or x.shape != (nx,) or y.shape != (ny,):
        raise ValueError(f"{WGM_FILE}: 形状 {z.shape}/{x.shape}/{y.shape} ≠ 期望 {(ny, nx)}")
    spacing = Fraction(360, nx - 1)
    if spacing != WGM_EXPECTED["spacing"]:
        raise ValueError(
            f"{WGM_FILE}: 步长 {spacing} ≠ 期望 {WGM_EXPECTED['spacing']}"
        )
    for arr, start, n in ((x, -180.0, nx), (y, -90.0, ny)):
        expect = start + np.arange(n) * float(spacing)
        if np.max(np.abs(arr - expect)) > 1e-3:
            raise ValueError(f"{WGM_FILE}: 节点坐标偏离规则网格（首末 {arr[0]}, {arr[-1]}）")
    if np.nanmax(np.abs(z[:, 0].astype(np.float64) - z[:, -1].astype(np.float64))) > 1e-2:
        raise ValueError(f"{WGM_FILE}: ±180 列不同值，周期重复假设被破坏")

    values = z[:, :-1]
    valid = np.isfinite(values)
    lat_edges = -90.0 + (np.arange(ny + 1, dtype=np.float64) - 0.5) * float(spacing)
    lat_edges[0], lat_edges[-1] = -90.0, 90.0
    return RasterSource(
        values=values,
        valid=valid,
        lat_edges=lat_edges,
        lon_res=spacing,
        attrs={
            "registration_rule": (
                "gridline 节点注册（node_offset=0）→ 节点代表 Voronoi 胞"
                "（中心 ±半步长，经度环接；极点行纬度钳制半权）；±180 重复列弃除"
            ),
            "source_file": WGM_FILE,
        },
    )



def load_emag2_tif(filename: str, src_dir: str | Path = EMAG_SRC_DIR) -> RasterSource:
    """Read an EMAG2 v3 tif (sea-level or upward-continued version) → normalized source
    (0..360 half wrap + row flip to south-first). The two versions' grid geometries are
    bit-identical (disk-verified: UpCont and Sealevel same shape 5399×10800, same
    envelope, same nodata), sharing the geometry assertions.

    After normalization, self-consistency is cross-checked by randomly sampling original
    rows/columns (a catcher for flip/wrap direction misregistration):
    raw(r, c) center = (top−(r+0.5)·res, c·res) ↔ norm(nlat−1−r, (c+nlon/2) % nlon).
    """
    import rasterio

    path = Path(src_dir) / filename
    if not path.is_file():
        raise FileNotFoundError(f"EMAG2 源文件不存在: {path}")
    with rasterio.open(path) as src:
        if src.shape != EMAG_EXPECTED["shape"]:
            raise ValueError(f"{filename}: 形状 {src.shape} ≠ 期望 {EMAG_EXPECTED['shape']}")
        res = EMAG_EXPECTED["res"]
        if abs(src.res[0] - res) > 1e-9 or src.res[0] != src.res[1]:
            raise ValueError(f"{filename}: 分辨率 {src.res} ≠ 2 arc-min")
        b = src.bounds
        eb = EMAG_EXPECTED["bounds"]
        if max(abs(b.left - eb[0]), abs(b.bottom - eb[1]), abs(b.right - eb[2]), abs(b.top - eb[3])) > 1e-6:
            raise ValueError(f"{filename}: 包络 {tuple(b)} ≠ 期望 {eb}（0..360 框架 + 极冠缺口）")
        if src.crs is not None and src.crs.to_epsg() != 4326:
            raise ValueError(f"{filename}: 意外 CRS {src.crs}（预期无标签或 EPSG:4326）")
        nodata = src.nodata
        if nodata is None:
            raise ValueError(f"{filename}: 无 nodata 标签，与磁盘实证不符")
        a = src.read(1)

    nlat, nlon = a.shape
    half = nlon // 2
    lon_res = Fraction(360, nlon)
    if abs(float(lon_res) - EMAG_EXPECTED["res"]) > 1e-9 or nlon % 2:
        raise ValueError(f"{filename}: 列数 {nlon} 与 2 arc-min 经度环不符")
    valid_raw = a != nodata
    values = np.where(valid_raw, a, np.nan).astype(np.float32)
    values = np.roll(values, -half, axis=1)[::-1]
    valid = np.isfinite(values)

    rng = np.random.default_rng(0)
    for _ in range(64):
        r, c = int(rng.integers(nlat)), int(rng.integers(nlon))
        ii, jj = nlat - 1 - r, (c + half) % nlon
        if valid_raw[r, c] != valid[ii, jj] or (
            valid_raw[r, c] and values[ii, jj] != a[r, c]
        ):
            raise ValueError(f"{filename}: 归一数组自洽核对失败 @ raw({r},{c}) → norm({ii},{jj})")

    lat_edges = -90.0 + float(lon_res) / 2.0 + np.arange(nlat + 1, dtype=np.float64) * res
    return RasterSource(
        values=values,
        valid=valid,
        lat_edges=lat_edges,
        lon_res=lon_res,
        attrs={
            "crs_assignment": (
                "源栅格无 CRS 标签，按 EMAG2 官方定义显式指定 EPSG:4326"
                "（地理经纬度）；transform 磁盘实证经度 0..360 框架，"
                "半卷绕归一 -180..180"
            ),
            "registration_rule": (
                "pixel 注册（列中心 = 0..359.9667 步 1/30 即中心在整分点；"
                "纬度胞边 ±89.9833，极冠 1/60° 无数据）；行翻南起 + 经度半卷绕"
            ),
            "source_file": filename,
        },
    )


def load_emag2_sealevel(src_dir: str | Path = EMAG_SRC_DIR) -> RasterSource:
    """Read the EMAG2 v3 sea-level tif (the upward-continued version is covered as a crosscheck validation reference)."""
    return load_emag2_tif(EMAG_FILE, src_dir)



_SOURCE_CACHE: dict[str, RasterSource] = {}


def _cached_load(loader, src_dir) -> RasterSource:
    """In-process source cache: the four tiers of the same layer read the ~233MB source file only once."""
    key = f"{loader.__name__}:{Path(src_dir).resolve()}"
    if key not in _SOURCE_CACHE:
        _SOURCE_CACHE[key] = loader(src_dir)
    return _SOURCE_CACHE[key]


def _build_raster_layer(entry: LayerEntry, tier: str, loader, src_dir,
                        long_name: str, doi: str) -> xr.DataArray:
    src = _cached_load(loader, src_dir)
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
        "doi": doi,
        "native_res": entry.native_res,
        "resampling": entry.resampling,
        "visibility": entry.visibility,
        "earth_radius_km": EARTH_RADIUS_KM,
    }
    attrs.update(src.attrs)
    return xr.DataArray(
        arr, dims=("lat", "lon"), coords={"lat": lat, "lon": lon},
        name=entry.id, attrs=attrs,
    )


def build_wgm2012_bouguer(entry: LayerEntry, tier: str) -> xr.DataArray:
    """Build gravity__wgm2012_bouguer for the given tier (visibility=internal, BGI NC terms)."""
    return _build_raster_layer(
        entry, tier, load_wgm2012, WGM_SRC_DIR,
        long_name="布格重力异常（WGM2012，球面布格改正，DTU10 地形）",
        doi=WGM_DOI,
    )


def build_emag2_sealevel(entry: LayerEntry, tier: str) -> xr.DataArray:
    """Build magnetics__emag2_sealevel for the given tier (sea-level version; the upward-continued version becomes a validation reference)."""
    return _build_raster_layer(
        entry, tier, load_emag2_sealevel, EMAG_SRC_DIR,
        long_name="磁异常（EMAG2 v3 海平面版，sea-level residual field）",
        doi=EMAG_DOI,
    )



GRAVMAG_LAYERS = {
    "gravity__wgm2012_bouguer": {
        "loader": load_wgm2012,
        "src_dir": WGM_SRC_DIR,
        "hist_spec": (-1000.0, 1500.0, 1.0),
        "unit": "mGal",
    },
    "magnetics__emag2_sealevel": {
        "loader": load_emag2_sealevel,
        "src_dir": EMAG_SRC_DIR,
        "hist_spec": (-2500.0, 4000.0, 1.0),
        "unit": "nT",
    },
}


def hist_bins(hist_spec: tuple[float, float, float]) -> int:
    """Histogram geometry → bin count (hist_spec = (lo, hi, bin_width))."""
    lo, hi, bw = hist_spec
    return int(round((hi - lo) / bw))


def gravmag_source_stats(src: RasterSource, hist_spec: tuple[float, float, float]) -> dict:
    """Source statistics (independent of the kernel path): Σ v·w + histogram + count, w = source-cell row area km²."""
    w = _source_cell_areas(src)
    row_sums = np.nansum(src.values.astype(np.float64), axis=1)
    sum_va = float(np.dot(row_sums, w))
    hist = np.zeros(hist_bins(hist_spec), dtype=np.uint64)
    lo, hi, bw = hist_spec
    hist = hist_add(hist, src.values[src.valid], lo=lo, hi=hi, bin_width=bw)
    return {"sum_va": sum_va, "hist": hist, "count": int(src.valid.sum())}


def build_gravmag_fidelity_report(store_dir, layer_id: str, tiers: list[str], src_dir=None) -> dict:
    """Fidelity report: ① the store is bit-identical to the source kernel recomputation (hard criterion)
    + ② the integral (hard criterion, independent source-statistics path) + ③ quantile/coverage
    (reporting only).

    ① Same-code recomputation (the build and the kernel share a deterministic code path → bit
    identity is assertable), catching any misregistration in the write/storage chain; ② source
    statistics computed directly per source cell (not through the kernel), exposing kernel
    weight errors here; ③ reporting only.
    """
    spec = GRAVMAG_LAYERS[layer_id]
    src = spec["loader"](src_dir) if src_dir else spec["loader"]()
    hist_spec = spec["hist_spec"]
    unit = spec["unit"]
    src_stats = gravmag_source_stats(src, hist_spec)

    checks: list[dict] = []
    with xr.open_datatree(store_dir, engine="zarr", chunks={}) as tree:
        for tier in tiers:
            node = tree[f"/{tier}"]
            if layer_id not in node.ds.data_vars:
                raise KeyError(f"store {store_dir} 的 {tier} 档不含 {layer_id}（先 build 再 fidelity）")
            v = node.ds[layer_id].values                     # float32 (nlat_t, nlon_t)
            expect, W = conservative_overlap_mean(
                src.values, src.valid, src.lat_edges, src.lon_res, tier_fraction(tier),
                lon_phase=src.lon_phase,
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

    n_fail = sum(1 for c in checks if c["status"] == "FAIL")
    return {
        "layer": layer_id,
        "store": str(store_dir),
        "integral_tolerance": INTEGRAL_TOL,
        "result": "FAIL" if n_fail else "PASS",
        "n_checks": len(checks),
        "checks": checks,
    }
