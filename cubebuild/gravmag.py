"""重磁组：WGM2012 布格重力异常 + EMAG2 v3 海平面磁异常。

源几何（磁盘实证，坐标原点与网格定义均经实机核对）：

- WGM2012_Bouguer_ponc_2min.grd（BGI）：COARDS netCDF，node_offset=0
  gridline 节点注册；x 10801 节点（-180..180 步 1/30，±180 列同值
  ——周期重复，实测最大差 1e-4 mGal）；y 5401 节点（-90..90）；z 完整
  全球场（0 NaN；"ponc" 名义点集，BGI 分发版已为完整网格），值域
  [-529, 1005] mGal。许可：BGI Terms 实质 NC → visibility=internal
  （不纳入公开版）。
  节点→胞元解释（registration_rule）：每节点代表其 Voronoi 胞
  （中心 ±1/60），纬度钳制于 ±90（极点行半权），经度环接日期线。
  理由：节点是场的点采样，Voronoi 胞是保守平均的最小偏置离散化；
  弃边列重排会引入半胞（1/60°≈1.8km）系统错位。

- EMAG2_V3_20170530_Sealevel.tif（NOAA NCEI）：10800×5399 float32，
  GeoTIFF 无 CRS 标签——配准显式指定 EPSG:4326（官方定义即地理
  经纬度网格；若文件自带 CRS 则须恰为 4326，否则拒绝）。transform
  实证：经度 0..360 框架（列中心 = 0, 1/30, …, 359.9667；原点在
  格林尼治，非 -180），纬度胞边 ±89.98333（极冠 1/60° 无数据）。
  nodata = float32 最小值（-3.4e38），缺测 47.9%。0|359.9667 与
  180|179.9667 两接缝差分与内部基线同分布（p50 3.2/4.3 nT vs 内部
  3.3）——0..360 框架判读的独立实证。卷绕归一：new[j] =
  old[(j+nlon/2) % nlon]（经度半卷绕）+ 行翻转南起（store 行 0 =
  -90 侧），并以原始行列随机抽点自洽核对。

重采样：连续变量面积加权保守平均，通用重叠核（kernels.
conservative_overlap_mean）——2min→3min 分数倍（1.5）精确处理，
缺测按有效源胞面积权重归一。两源四档（1°/30′/6′/3′），WGM 全覆盖
（掩膜全 1），EMAG 掩膜携带 47.9% 缺测结构。

保真：三层判据——① store 档位数组与源核重算逐位一致（位级硬判据，
派生层同款哲学）；② 积分恒等式 Σ_t v_t·W_t = Σ_s v_s·w_s，
源统计独立直算（不走核路径，核权重错误在此暴露）；③ 分位数/覆盖率
报告制。行翻转/经度旋转类错位对 ①② 均不敏感（自洽错位），由单元
测试的解析线性场（方位精确可断言）与读取器抽点自洽核对（raw 行列
↔ 归一数组）覆盖；极值定位检查经实测验明不可用（源极值为单 2min
胞尖峰，粗档保守平均稀释后 argmax 漂移数十胞，属平滑的物理必然）。
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

WGM_DOI = "10.1007/s00190-011-0533-4"       # Balmino et al., J. Geodesy（refs PDF 自证）
EMAG_DOI = "10.1002/2017GC007280"           # Meyer et al. 2017, G³（refs PDF 自证）

TWO_MIN = Fraction(1, 30)                    # 两源同为 2 arc-min

# 读取期几何断言常量（测试可 monkeypatch 缩格）
WGM_EXPECTED = {"shape": (5401, 10801), "spacing": TWO_MIN}
EMAG_EXPECTED = {
    "shape": (5399, 10800),
    "res": 1.0 / 30.0,
    "bounds": (-1.0 / 60.0, -90.0 + 1.0 / 60.0, 360.0 - 1.0 / 60.0, 90.0 - 1.0 / 60.0),
}


@dataclass
class RasterSource:
    """规范化全球栅格源：行 0 = -90 侧、列 0 = -180 侧、NaN = 缺测。

    lon_phase：源胞中心相对 −180 + k·lon_res 的经度相位（度，Fraction；
    起由 HFgrid14 类 0.25 偏移像素注册网格使用）。默认 0 = 中心
    在整倍数处（WGM 节点 Voronoi / EMAG2 卷绕后形态）。
    """

    values: np.ndarray            # (nlat, nlon) float32
    valid: np.ndarray             # (nlat, nlon) bool
    lat_edges: np.ndarray          # (nlat+1,) float64，源胞纬度边
    lon_res: Fraction              # 源胞经度分辨率（度）
    lon_phase: Fraction = Fraction(0)   # 源经度相位（度）
    attrs: dict = field(default_factory=dict)   # 溯源（registration_rule 等）

    @property
    def lat_centers(self) -> np.ndarray:
        return 0.5 * (self.lat_edges[:-1] + self.lat_edges[1:])


def _source_cell_areas(src: RasterSource) -> np.ndarray:
    """源胞行面积（km²）：R²·Δsin·Δλ_rad——积分恒等式的 w_s（核同约定）。"""
    dsin = np.sin(np.deg2rad(src.lat_edges[1:])) - np.sin(np.deg2rad(src.lat_edges[:-1]))
    return EARTH_RADIUS_KM**2 * dsin * float(src.lon_res) * np.pi / 180.0


# ---------------- WGM2012 读取 ----------------

def load_wgm2012(src_dir: str | Path = WGM_SRC_DIR) -> RasterSource:
    """读 WGM2012 布格异常 grd → 规范化源（节点→Voronoi 胞解释）。

    断言（磁盘实证的契约化）：gridline 注册、节点坐标 ±180/±90 与
    1/30 步长、±180 列周期同值（容差 1e-2 mGal = float32 噪声量级）。
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
    # 分辨率自节点数导出精确有理数（360/(nx−1)），并与契约常量互证
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

    values = z[:, :-1]                                   # 弃 +180 重复列 → 10800
    valid = np.isfinite(values)
    # Voronoi 胞边：节点 ±spacing/2，极点行钳制半权（ny 行 → ny+1 边）
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


# ---------------- EMAG2 读取 ----------------

def load_emag2_tif(filename: str, src_dir: str | Path = EMAG_SRC_DIR) -> RasterSource:
    """读 EMAG2 v3 tif（海平面版或上延版）→ 规范化源（0..360 半卷绕 +
    行翻转南起）。两版网格几何逐位同格（磁盘实证：UpCont 与
    Sealevel 同形 5399×10800、同包络、同 nodata），共用几何断言。

    归一后以原始行列随机抽点自洽核对（翻转/卷绕方向错位的捕获器）：
    raw(r, c) 中心 = (top−(r+0.5)·res, c·res) ↔ norm(nlat−1−r, (c+nlon/2) % nlon)。
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
    # 分辨率自列数导出精确有理数（360/nlon），并与 transform 互证
    lon_res = Fraction(360, nlon)
    if abs(float(lon_res) - EMAG_EXPECTED["res"]) > 1e-9 or nlon % 2:
        raise ValueError(f"{filename}: 列数 {nlon} 与 2 arc-min 经度环不符")
    valid_raw = a != nodata
    values = np.where(valid_raw, a, np.nan).astype(np.float32)
    values = np.roll(values, -half, axis=1)[::-1]        # 经度半卷绕 + 南起翻转
    valid = np.isfinite(values)

    rng = np.random.default_rng(0)
    for _ in range(64):
        r, c = int(rng.integers(nlat)), int(rng.integers(nlon))
        ii, jj = nlat - 1 - r, (c + half) % nlon
        if valid_raw[r, c] != valid[ii, jj] or (
            valid_raw[r, c] and values[ii, jj] != a[r, c]
        ):
            raise ValueError(f"{filename}: 归一数组自洽核对失败 @ raw({r},{c}) → norm({ii},{jj})")

    # 纬度边自精确有理几何构造（transform 浮点误差 ~5e-9° 不入几何；
    # 与文件包络的一致性已由上方 bounds 断言锚定）
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
    """读 EMAG2 v3 海平面版 tif（上延版见 crosscheck 验证参考）。"""
    return load_emag2_tif(EMAG_FILE, src_dir)


# ---------------- 层构建 ----------------

_SOURCE_CACHE: dict[str, RasterSource] = {}


def _cached_load(loader, src_dir) -> RasterSource:
    """进程内源缓存：同层四档构建只读一次 ~233MB 源文件。"""
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
    """构建该档 gravity__wgm2012_bouguer（visibility=internal，BGI NC 条款）。"""
    return _build_raster_layer(
        entry, tier, load_wgm2012, WGM_SRC_DIR,
        long_name="布格重力异常（WGM2012，球面布格改正，DTU10 地形）",
        doi=WGM_DOI,
    )


def build_emag2_sealevel(entry: LayerEntry, tier: str) -> xr.DataArray:
    """构建该档 magnetics__emag2_sealevel（海平面版；上延版转验证参考）。"""
    return _build_raster_layer(
        entry, tier, load_emag2_sealevel, EMAG_SRC_DIR,
        long_name="磁异常（EMAG2 v3 海平面版，sea-level residual field）",
        doi=EMAG_DOI,
    )


# ---------------- 保真验证（本层自供源统计） ----------------

# 分位数直方图几何（值域自磁盘实证 ±裕量；等宽 1 单位 bin）
GRAVMAG_LAYERS = {
    "gravity__wgm2012_bouguer": {
        "loader": load_wgm2012,
        "src_dir": WGM_SRC_DIR,
        "hist_spec": (-1000.0, 1500.0, 1.0),   # 源值域 [-529, 1005] mGal
        "unit": "mGal",
    },
    "magnetics__emag2_sealevel": {
        "loader": load_emag2_sealevel,
        "src_dir": EMAG_SRC_DIR,
        "hist_spec": (-2500.0, 4000.0, 1.0),   # 源值域 [-1912, 3540] nT
        "unit": "nT",
    },
}


def hist_bins(hist_spec: tuple[float, float, float]) -> int:
    """直方图几何 → bin 数（hist_spec = (lo, hi, bin_width)）。"""
    lo, hi, bw = hist_spec
    return int(round((hi - lo) / bw))


def gravmag_source_stats(src: RasterSource, hist_spec: tuple[float, float, float]) -> dict:
    """源统计（独立于核路径）：Σ v·w + 直方图 + 计数，w = 源胞行面积 km²。"""
    w = _source_cell_areas(src)
    # 逐行求和再点积：invalid/NaN 经 nansum 归零（w·Σ_j v 等价于 Σ_s v_s·w_s）
    row_sums = np.nansum(src.values.astype(np.float64), axis=1)
    sum_va = float(np.dot(row_sums, w))
    hist = np.zeros(hist_bins(hist_spec), dtype=np.uint64)
    lo, hi, bw = hist_spec
    hist = hist_add(hist, src.values[src.valid], lo=lo, hi=hi, bin_width=bw)
    return {"sum_va": sum_va, "hist": hist, "count": int(src.valid.sum())}


def build_gravmag_fidelity_report(store_dir, layer_id: str, tiers: list[str], src_dir=None) -> dict:
    """保真报告：① store 与源核重算逐位一致（硬判据）+ ② 积分量（硬判据，
    独立源统计路径）+ ③ 分位数/覆盖率（报告制）。

    ① 同码重算（构建与核共享确定代码路径 → 位级一致可断言），捕获
    写入/存储链路的任何错位；② 源统计逐源胞直算（不经核），核权重
    错误在此暴露；③ 报告制。
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

            # ① 逐位一致（硬判据）：store 值 == 源核重算（位级，期望 0 不一致；
            #    NaN == NaN 视为一致——有效位一致性另由下方检查承担）
            eq = (v == expect) | (np.isnan(v) & np.isnan(expect))
            n_bad = int((~eq).sum())
            checks.append({
                "name": f"{tier}: 与源核重算逐位一致（float32 位级）",
                "status": "pass" if n_bad == 0 else "FAIL",
                "summary": f"不一致像元 {n_bad}（期望 0）",
            })

            # 有效位一致性（硬判据）：store 值 NaN ⟺ 源掩膜复算 W=0
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
