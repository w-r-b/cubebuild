"""岩石圈栅格组：GEMMA 地壳三层 + LITHO1.0 LAB + Seton 现今洋龄。

源契约（磁盘实证，坐标/值域/覆盖均实机核对）：

- GEMMA（ecm1-gemma/*.tif，Reguzzoni & Sampietro 2015 JAG，GOCE 反演）：
  crust_bottom / crust_moho_err / crust_top 三 tif 均 360×720 float32
  0.5°，bounds 恰 (-180,-90,180,90)——pixel 注册、胞边对齐 -180/-90
  （lon_phase=0.25），CRS 声明 EPSG:4326，无 nodata 且全 259200 像元
  有限（全球全覆盖，validity 掩膜全 1）。行 0 = 北（top=90）→ 读取期
  翻转南起。值域 km（海平面基准，负值向下）：moho −105.19..−4.80
  （最深处 (34.25, 79.25) 喀喇昆仑——GOCE 反演固有的造山带深 Moho）；
  moho_err 1.60..10.23；basement（crust_top 源文件）−21.00..+5.86
  （正值 = 结晶基底面高于海平面）。交叉实证：crust_bottom ≤
  crust_top 全场成立（0 违例），结晶地壳厚度 = top − bottom 为
  0.007..110.64 km（留给下游一行减法，约定）。

- LITHO1.0（LITHO1.0.r0.0-n4c.nc，Pasyanos et al. 2014）：180×360
  float32，中心 −89.5..89.5 / −179.5..179.5（±.5 集），与立方 1° 档
  中心逐位一致（半值在 float32 精确）——免重采样直映；纬度源南起，
  无需翻转。源纬度坐标 4 行字面缺陷（行 75/77/102/104 的 ±14.5/±12.5
  写作 ±14.6/±12.6，GeoCSV 上游笔误形态）——数据行对齐经 ETOPO 交叉
  实证不受影响（4 行 water_bottom 与 store bedrock_elevation 最优行
  移位 0、相关 0.98–0.99，与无缺陷对照行同分布），几何按名义规则
  网格解析，缺陷值契约化（LITHO1_COORD_DEFECTS）。
  asthenospheric_mantle_top_depth 全 64800 有限（0 NaN），值域
  8.87..320.42 km；深度基准海平面、正值向下（water_top=0 于太平洋/
  Tibet 沉积顶 −5.03 km 基底高程互证）——与 GEMMA 组负值向下符号
  相反，消费方须读 value_convention。该变量与 lid_bottom_depth 逐位
  相同（64800/64800）——同一界面双命名。nc 其余 170 变量清单经
  write_litho1_inventory_report 写入构建回报（v1.1 展开依据）。

- Seton 洋龄（Seton_etal_2020_PresentDay_AgeGrid.nc）：GMT gridline
  节点注册 3601×1801（节点 −180..180 / −90..90 步 0.1；坐标带 ~1e-11
  GMT 浮点累积误差，几何由节点数精确导出）；±180 列逐位相同（双有限
  行 max|Δ|=0.0、NaN 分布一致）→ 弃 +180 重复列；z float32
  _FillValue=NaN，缺测 3331489/6485401 = 51.37%（陆域 + COB 掩膜）；
  值域 0.01..338.68 Ma。北极行（lat=90）全有限且单值 54.66 Ma（北极
  点洋壳，grdfill 类填充），南极行全 NaN。节点→Voronoi 胞解释
  （同 WGM2012 处置）：每节点代表中心 ±0.05 胞、极行钳制半权、
  经度环接；0.1° gridline → 6′ pixel 档因此是相邻节点 50/50 混合
  （注册类型转换的几何必然，保守核精确处理）；磁盘仅现今网格、无
  时间序列（Time-dependent raster sequences 为第三方参考，
  manifest time_semantics 登记）。

重采样：连续变量面积加权保守平均（kernels.conservative_overlap_mean，
通用核）——GEMMA 0.5°→30′ 恒等映射 / →1° 2×2 加权；Seton
0.1°→6′ 相邻节点各半（2 胞半权）/ →30′ 6 胞（4 全 + 2 半权）/
→1° 11 胞（9 全 + 2 半权），缺测按有效源胞面积权重归一（海岸部分
覆盖像元无稀释）。LITHO1.0 1° 直映（resampling: none）。

保真（gravmag 三层判据同构）：① store 与源重算逐位一致（位级硬判据）；
② 积分恒等式 Σ v·W = Σ v·w（源统计独立直算，核权重错误在此暴露）；
③ 分位数/覆盖率报告制。GEMMA basement 另报实证依据，
LITHO1.0 另报界面同一性（lab ≡ lid_bottom 逐位）与变量清单回报。
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

GEMMA_DOI = "10.1016/j.jag.2014.04.002"        # Reguzzoni & Sampietro 2015（refs PDF 自证）
LITHO1_DOI = "10.1002/2013JB010626"            # nc reference_pid + refs PDF 双自证
SETON_DOI = "10.1029/2020GC009214"             # Seton et al. 2020, G³

GEMMA_HALF = Fraction(1, 2)                    # GEMMA 0.5°（精确有理数）
SETON_TENTH = Fraction(1, 10)                  # Seton 0.1°（精确有理数）

GEMMA_MOHO_ID = "lithosphere__gemma_moho"
GEMMA_MOHO_ERR_ID = "lithosphere__gemma_moho_err"
GEMMA_BASEMENT_ID = "lithosphere__gemma_basement_depth"
LITHO1_LAB_ID = "lithosphere__litho1_lab"
SEAFLOOR_AGE_ID = "lithosphere__seafloor_age"

# 层 id → GEMMA 源 tif 文件名（crust_top = 结晶基底面，见 _BASEMENT_EVIDENCE）
GEMMA_FILES = {
    GEMMA_MOHO_ID: "crust_bottom.tif",
    GEMMA_MOHO_ERR_ID: "crust_moho_err.tif",
    GEMMA_BASEMENT_ID: "crust_top.tif",
}

# LITHO1.0 LAB 源变量（与 lid_bottom_depth 逐位相同——界面同一性契约）
LITHO1_LAB_VAR = "asthenospheric_mantle_top_depth"
LITHO1_LID_VAR = "lid_bottom_depth"

# LITHO1.0 纬度坐标字面缺陷（磁盘实证）：行 75/77/102/104
# （名义 ±14.5/±12.5）坐标写作 ±14.6/±12.6（float32 精确值，GeoCSV
# 上游笔误形态）。数据行对齐经 ETOPO 交叉实证：4 行 water_bottom 与
# store 1° bedrock_elevation 相关 0.98–0.99 且最优行移位 = 0（与无缺陷
# 对照行同分布）——数据在名义规则网格上，仅坐标标签损坏。几何按名义
# 网格解析（同 Seton 节点数导出惯例），缺陷值契约化（漂移即显式失败）。
LITHO1_COORD_DEFECTS = {
    75: float(np.float32(-14.6)),
    77: float(np.float32(-12.6)),
    102: float(np.float32(12.6)),
    104: float(np.float32(14.6)),
}

# 读取期几何断言常量（磁盘实证的契约化；测试可 monkeypatch 缩格）
GEMMA_EXPECTED = {
    "shape": (360, 720),
    "res": 0.5,
    "bounds": (-180.0, -90.0, 180.0, 90.0),
}
LITHO1_EXPECTED = {"shape": (180, 360)}
SETON_EXPECTED = {"shape": (1801, 3601)}

# GEMMA 值域（km，磁盘实证；源漂移即保真/基线复核提示）
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


# ---------------- GEMMA 读取 ----------------

def load_gemma_tif(filename: str, src_dir: str | Path = GEMMA_SRC_DIR) -> RasterSource:
    """读 GEMMA 地壳 tif → 规范化源（行 0 = -90 侧、列 0 = -180 侧）。

    断言（磁盘实证的契约化）：360×720、0.5°、bounds 恰 ±180/±90
    （pixel 注册 = 胞边对齐 -180/-90）、CRS 4326（声明或无标签）、
    全像元有限（实证无 nodata 无 NaN，源变更即显式失败）。
    北起行序翻转为南起，并以原始行列随机抽点自洽核对（翻转方向
    错位的捕获器）：raw(r,c) 中心 = (90−(r+0.5)·res, −180+(c+0.5)·res)
    ↔ norm(nlat−1−r, c)。
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
    values = a[::-1].astype(np.float32)                  # 北起 → 南起
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
        lon_phase=GEMMA_HALF / 2,                        # 0.25 偏移 = 边对齐 -180
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


# ---------------- LITHO1.0 读取 ----------------

def load_litho1_lab(src_dir: str | Path | None = LITHO1_SRC_DIR) -> RasterSource:
    """读 LITHO1.0 nc 的 LAB（软流圈顶界深度）→ 规范化源（1° 直映档）。

    断言（磁盘实证的契约化）：180×360；中心坐标与立方 1° 档中心逐位
    一致（float32 半值精确；纬度南起免翻转）——已知坐标字面缺陷 4 行
    （LITHO1_COORD_DEFECTS，数据行对齐经 ETOPO 交叉实证，几何按名义
    网格解析）；LAB 全像元有限（实证 0 NaN）；LAB 与 lid_bottom_depth
    逐位相同（界面同一性：软流圈顶界 ≡ 岩石圈盖底界，同一界面双命名）。
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


# ---------------- Seton 洋龄读取 ----------------

def load_seafloor_age(src_dir: str | Path | None = SETON_SRC_DIR) -> RasterSource:
    """读 Seton 现今洋龄 nc → 规范化源（gridline 节点 → Voronoi 胞）。

    断言（磁盘实证的契约化）：3601×1801 节点（−180..180/−90..90 步
    0.1，坐标带 ~1e-11 GMT 浮点累积误差，容差 1e-8 锚定规则网格）；
    ±180 列逐位相同（容差 float32 噪声量级 1e-3 Ma）→ 弃 +180 重复列；
    Voronoi 胞边 = 节点 ±0.05（极行钳制半权）。
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
    # 分辨率自节点数导出精确有理数（360/(nx-1) = 1/10）
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
    # ±180 列周期重复（实证 max|Δ|=0.0；容差 float32 噪声量级）
    both = np.isfinite(z[:, 0]) & np.isfinite(z[:, -1])
    if int((np.isnan(z[:, 0]) != np.isnan(z[:, -1])).sum()) > 0:
        raise ValueError(f"{SETON_FILE}: ±180 列 NaN 分布不一致，周期重复假设被破坏")
    if both.any() and np.max(np.abs(
        z[both, 0].astype(np.float64) - z[both, -1].astype(np.float64)
    )) > 1e-3:
        raise ValueError(f"{SETON_FILE}: ±180 列不同值，周期重复假设被破坏")

    values = z[:, :-1]                                  # 弃 +180 重复列 → 3600
    valid = np.isfinite(values)
    # Voronoi 胞边：节点 ±0.05，极点行钳制半权（ny 行 → ny+1 边）
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


# ---------------- LITHO1.0 nc 变量清单（构建回报） ----------------

def litho1_variable_inventory(src_dir: str | Path | None = LITHO1_SRC_DIR) -> dict:
    """LITHO1.0 nc 全变量清单（v1.1 展开依据；构建期读取、确定性、无时间戳）。

    171 个数据变量按「界面层 × 物性」索引：water/ice/upper-middle-lower
    sediments/upper-middle-lower crust/lid/asthenospheric_mantle 10 层组
    × top/bottom 界面（asthenospheric_mantle 仅 top，其余各 2）× 9 物性
    （depth/density/vp/vs/qkappa/qmu/vp2/vs2/eta）。本模块 ingest 仅
    LAB 一变量，其余 170 项为 v1.1 候选。
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

    # 变量名解析 {layer}_{top|bottom}_{property}（171/171 全解析，硬断言）
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
    """变量清单回报落盘（其余变量清单写入构建回报；确定性无时间戳）。"""
    inventory = litho1_variable_inventory(src_dir or LITHO1_SRC_DIR)
    path = Path(reports_dir) / "litho1-inventory.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return path


# ---------------- 层构建 ----------------

_SOURCE_CACHE: dict[str, RasterSource] = {}


def _cached_load(key: str, loader, src_dir=None) -> RasterSource:
    """进程内源缓存：同层多档构建只读一次源文件（loader 统一收 src_dir，
    None = 各 loader 缺省路径）。键含解析后的 src_dir（gravmag/thermal
    惯例）——同进程混用自定义/缺省路径不会取到陈旧源。"""
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
    """构建该档 lithosphere__gemma_moho（立方唯一 Moho 层；负值向下）。"""
    return _build_gemma_layer(
        entry, tier, GEMMA_MOHO_ID,
        long_name="莫霍面深度（GEMMA，GOCE 反演；海平面基准，负值向下）",
        extra_attrs={
            "product_type": "model（GOCE 卫星重力反演地壳模型，非地震观测）",
        },
    )


def build_gemma_moho_err(entry: LayerEntry, tier: str) -> xr.DataArray:
    """构建该档 lithosphere__gemma_moho_err（Moho 深度不确定度，ML 加权用）。"""
    return _build_gemma_layer(
        entry, tier, GEMMA_MOHO_ERR_ID,
        long_name="莫霍面深度不确定度（GEMMA；供 ML 不确定性加权）",
        extra_attrs={
            "product_type": "model（GOCE 反演方差传播，非地震观测）",
            "usage_note": "与 lithosphere__gemma_moho 配对使用：ML 损失加权 1/σ² 类用途",
            # σ 层严格正值（1.6–10.23 km），覆盖通用符号注记（不挂"负值向下"）
            "value_convention": "km，Moho 深度 1σ 不确定度（严格正值）",
        },
    )


def build_gemma_basement_depth(entry: LayerEntry, tier: str) -> xr.DataArray:
    """构建该档 lithosphere__gemma_basement_depth（结晶基底面；负值向下）。"""
    return _build_gemma_layer(
        entry, tier, GEMMA_BASEMENT_ID,
        long_name="结晶基底面深度（GEMMA crust_top；海平面基准，负值向下）",
        extra_attrs={
            "product_type": "model（GOCE 卫星重力反演地壳模型，非地震观测）",
            "basement_evidence": _BASEMENT_EVIDENCE,
        },
    )


def build_litho1_lab(entry: LayerEntry, tier: str) -> xr.DataArray:
    """构建 1° 档 lithosphere__litho1_lab（免重采样直映，resampling=none）。"""
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
    """构建该档 lithosphere__seafloor_age（现今洋龄；海洋层，陆地 NaN）。"""
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


# ---------------- 保真验证（本层自供） ----------------

# 直方图几何（值域自磁盘实证 ±裕量；等宽 bin）
LITHO_FIDELITY_LAYERS = {
    GEMMA_MOHO_ID: {
        "kind": "gemma",
        "hist_spec": (-110.0, 0.0, 0.5),        # 源值域 [-105.19, -4.80] km
        "unit": "km",
    },
    GEMMA_MOHO_ERR_ID: {
        "kind": "gemma",
        "hist_spec": (1.0, 11.0, 0.05),         # 源值域 [1.60, 10.23] km
        "unit": "km",
    },
    GEMMA_BASEMENT_ID: {
        "kind": "gemma",
        "hist_spec": (-22.0, 7.0, 0.25),        # 源值域 [-21.00, 5.86] km
        "unit": "km",
    },
    LITHO1_LAB_ID: {
        "kind": "litho1",
        "hist_spec": (8.0, 322.0, 2.0),         # 源值域 [8.87, 320.42] km
        "unit": "km",
    },
    SEAFLOOR_AGE_ID: {
        "kind": "seton",
        "hist_spec": (0.0, 340.0, 1.0),         # 源值域 [0.01, 338.68] Ma
        "unit": "Ma",
    },
}

# 各 kind 的源重算（store 值的期望）：gemma/seton 走保守核，litho1 直映
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
    """该档期望值与有效权重：gemma/seton 保守核重算；litho1 直映
    （W = 目标胞解析行面积——全覆盖直映层的积分权重）。"""
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
    """保真报告（gravmag 三层判据同构）：① store 与源重算逐位一致（位级
    硬判据）+ ② 有效位一致 + ③ 积分恒等式（独立源统计路径）+ ④ 分位数/
    覆盖率（报告制）；另附专项：GEMMA basement 实证依据、LITHO1.0
    界面同一性与符号约定注记。
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

            # ① 逐位一致（硬判据；NaN == NaN 视为一致）
            eq = (v == expect) | (np.isnan(v) & np.isnan(expect))
            n_bad = int((~eq).sum())
            checks.append({
                "name": f"{tier}: 与源{'直映' if kind == 'litho1' else '核'}重算逐位一致（float32 位级）",
                "status": "pass" if n_bad == 0 else "FAIL",
                "summary": f"不一致像元 {n_bad}（期望 0）",
            })

            # 有效位一致性（硬判据）：store 值有限 ⟺ W > 0
            mism = int((np.isfinite(v) != ok).sum())
            checks.append({
                "name": f"{tier}: 有效位与源掩膜复算一致（逐位）",
                "status": "pass" if mism == 0 else "FAIL",
                "summary": f"不一致 {mism} 像元（期望 0）",
            })

            # ②③ 积分恒等式 + 分位数（源统计独立直算，不经核）
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

    # 专项注记
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
