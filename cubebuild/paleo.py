"""4D 古高程通道：Scotese & Wright 2018 PaleoDEM。

源契约（磁盘实证，109 片 × 2 产品全量扫描，坐标/值域/缺测/缝
均经实机核对）：

- paleodem-6min-nc（109 片）：1801×3601 gridline 节点注册，latitude 90→-90
  北起降序，longitude -180→180；变量 z float32；全场无 NaN（109 片 0 缺测
  ——古地理以数值编码：负值=古海洋深度、0=海平面、正值=古陆面高程，缺测
  语义不适用，有效性掩膜全 1）；值域 [-11000, 10500] m。±180 两列值不同
  （各片最大差可达 14620 m）——缝两侧独立采样（GLAD 同构：古地理
  转换网格的日期线缝），内部连续性对称（|179.9−180| 中位 40 m vs
  |-179.9−(-180)| 中位 120 m），不可择一弃置。
- paleodem-1deg-nc（109 片）：181×361 节点注册，lat -90→90 升序，lon
  -180→180；z float32 无 NaN，值域 [-9000, 10500] m。±180 两列**逐位相同**
  （109 片缝差恰 0）——官方 1° 产品经 GMT grdsample 生成，周期化时取
  -180 列采样（实测 1° ±180 列 == 6min -180 列，180/180 抽行全中）。
- time 轴 = 文件名 Ma 标签（正则 _(\\d+(?:\\.\\d+)?)Ma\\.nc$），两产品各 109
  片、（1° 对齐后）严格升序唯一、0–540 Ma；1° 两片（Map67.5/68，源文件名
  385.2/390.5 Ma）按官方整数命名惯例舍入为 385/390（官方
  命名双轨制——文件名整数粗标为系统惯例（6min 109 片中 85 片 ≠ 内部
  description 精确年代；6min Map67.5/68 文件名 385/390、内部即 385.2/390.5
  Ma），1° 两片精确命名才是惯例偏离者；官方自身舍入惯例 Map54 内部 286.8
  → 文件名 285）——两档标签集 109/109 逐值一致，跨档按年代联用可直接
  对齐。文件名字典序 ≠ Ma 序（Map43.5_205Ma 先于 Map43_200Ma），必须按
  解析值排序；跨档对齐按标签值而非 Map 图幅编号（Guadalupian 附近两档
  编号错位半格：6min Map53.5@280/Map54@285 ↔ 1° Map54@280/Map54.5@285，
  按编号对齐会错位）。

配准（registration_rule）：两产品均为 gridline 节点注册 →
Voronoi 胞解释（节点是场的点采样，Voronoi 胞是保守平均的最小偏置离散化）；
6min 档：±180 缝两列取均值（对称保守化）弃 +180 列 → (1801, 3600)，经
conservative_overlap_mean 汇入立方 6min 档（源胞 0.1° 与目标胞 0.1° 半胞
错位 → 目标胞 = 相邻节点面积加权平均，即原生分辨率档的配准解释而非档位
聚合，GLAD 1° 档同款）；1° 档：官方 1° 产品直读（免降采样），±180
逐位相同弃 +180 列 → (181, 360)，Voronoi 保守核汇入立方 1° 档（GLAD 几何
同款）。

写入结构：[time, lat, lon] 4D 数组与档位节点 2D 层形状不同 → 入档位子节点
/6min/paleodem 与 /1deg/paleodem（组节点机制；manifest datatree_node
登记相对组名，validate 按 /{tier}/{group} 解析）；time 逐片分块
（6min (1,900,900) / 1deg (1,180,360)）支持下游一次 .sel(time=) 取任意
地质年代切片。

线程安全：源为 HDF5（netCDF4）非线程安全 → 模块级读锁仅护文件 I/O，
numpy 核在锁外并行（dask 线程调度器下构建确定性不受影响：核为 einsum
单线程归约）。
"""

import re
import threading
from fractions import Fraction
from pathlib import Path

import dask.array as da
import numpy as np
import xarray as xr
from dask import delayed

from .contract import LayerEntry
from .etopo import SourceNotReady
from .fidelity import INTEGRAL_TOL
from .grids import grid_shape, tier_centers, tier_fraction
from .kernels import conservative_overlap_mean
from .pixel_area import EARTH_RADIUS_KM

# 源胞行面积因子（与 kernels._ROW_AREA_FACTOR 同约定：积分恒等式两侧
# 必须同一 R 与单位折算，否则守恒判据失真）
_ROW_AREA_FACTOR = EARTH_RADIUS_KM**2 * np.pi / 180.0

REPO_ROOT = Path(__file__).resolve().parents[1]
PALEO_SRC_DIR = REPO_ROOT / "original data/paleo/paleodem-scotese"
SIXMIN_DIR = PALEO_SRC_DIR / "paleodem-6min-nc"
ONEDEG_DIR = PALEO_SRC_DIR / "paleodem-1deg-nc"

# ---- 源契约常量（磁盘实证的契约化；源变更即读取期显式失败）----
SIXMIN_NLAT, SIXMIN_NLON = 1801, 3601          # 6min 产品节点网格
ONEDEG_NLAT, ONEDEG_NLON = 181, 361            # 1° 产品节点网格
N_SLICES = 109                                  # 每产品时间片数（0–540 Ma）
MA_RANGE = (0.0, 540.0)                         # time 轴标签范围（0–540Ma 全收）
SIXMIN_VALUE_RANGE = (-11000.0, 10500.0)        # 6min 产品值域（109 片实测）
ONEDEG_VALUE_RANGE = (-9000.0, 10500.0)         # 1° 产品值域（109 片实测）
SEAM_MAX_DIFF = 16000.0   # 6min ±180 缝差护栏（实证 ≤14620；须低于域内极差
                          # 21500 方为可达的功能性护栏，防缝语义漂移如同格重排）
SOURCE_URL = "https://www.earthbyte.org/paleodem-resource-scotese-and-wright-2018/"

# 组节点名（store 内 /{tier}/paleodem）与层路由（cli 路由 4D 层入子节点）
PALEO_GROUP = "paleodem"
PALEO_LAYER_ID = "paleo__paleodem_elevation"
PALEO_4D_GROUP_OF = {PALEO_LAYER_ID: PALEO_GROUP}
PALEO_LAYERS = frozenset({PALEO_LAYER_ID})     # 保真报告分发用（cli 成员测试）

_MA_RE = re.compile(r"_(\d+(?:\.\d+)?)Ma\.nc$")

# 1° 档标签对齐映射：官方命名双轨制——文件名整数粗标为
# 系统惯例（6min 109 片中 85 片文件名 ≠ 内部 description 精确年代；6min
# Map67.5/68 文件名 385/390、内部即 385.2/390.5 Ma），1° 两片精确命名是
# 惯例偏离者——按官方整数惯例舍入对齐（官方自身舍入惯例：Map54 内部
# 286.8 → 文件名 285）。按标签值映射而非 Map 图幅编号：Guadalupian 附近
# 两档编号错位半格（6min Map53.5@280/Map54@285 ↔ 1° Map54@280/Map54.5@285），
# 按编号对齐会错位。
_ONEDEG_LABEL_ALIGN = {385.2: 385.0, 390.5: 390.0}

# HDF5 读锁（见模块 docstring 线程安全注记）
_HDF5_LOCK = threading.Lock()

_REGISTRATION = (
    "两产品均 gridline 节点注册 → Voronoi 胞解释（同 WGM2012 / GLAD-M35 处置）："
    "纬度边 = 节点中点（极点行半胞钳制）、经度胞宽 = 源步长环接日期线，"
    "conservative_overlap_mean 汇入立方档位（目标胞 = 相邻节点面积加权平均）。"
    "6min 档：官方 6min 产品 ±180 缝两列独立采样（实证缝差 ≤14620 m，内部"
    "连续性对称）取均值弃 +180 列；源胞与目标胞同 0.1° 但半胞错位 → 目标 = "
    "相邻节点均值，为原生档的配准解释而非聚合。1° 档：官方 1° 产品直读"
    "（免降采样），±180 逐位相同（grdsample 周期化取 -180 列，实证 109 片"
    "缝差恰 0）弃 +180 列，GLAD 1° 几何同款"
)

_LICENSE_NOTE = (
    "CC BY 4.0（paleodem-1deg-nc/License.txt，PALEOMAP PaleoDEM 分发条款；"
    "可随公开版发布）"
)

_TIME_SEMANTICS = (
    "time 坐标 = 源文件名 Ma 标签（地质年代：距今百万年，升序，0–540 Ma，"
    "109 片）；1° 两片按官方整数命名惯例舍入（源文件名 385.2/390.5 Ma → "
    "385/390，官方双轨制：文件名整数粗标为系统惯例、1° 精确命名是惯例偏离者），"
    "两档标签集 109/109 逐值一致，跨档按年代联用可直接对齐"
)


def parse_ma_label(filename: str) -> float:
    """文件名 → Ma 标签（如 Map43.5_..._205Ma.nc → 205.0）。"""
    m = _MA_RE.search(filename)
    if m is None:
        raise ValueError(f"文件名无 Ma 标签: {filename}")
    return float(m.group(1))


def scan_slices(src_dir: str | Path, expected: int | None = None,
                tier: str | None = None) -> list[tuple[float, Path]]:
    """源目录 → [(Ma, 路径)] 按 Ma 升序。

    tier="1deg" 时标签经官方整数惯例对齐（385.2→385、390.5→390）；
    6min/None 返回文件名原始标签（6min 档零改动）。契约：恰 expected 片
    （缺省 = 模块常量 N_SLICES，调用期解析）、（对齐后）标签严格升序唯一。
    缺目录/片数不符 → SourceNotReady（构建期跳过该层）；片数相符但标签
    异常 → ValueError（源漂移硬失败）。
    """
    if expected is None:
        expected = N_SLICES
    src_dir = Path(src_dir)
    if not src_dir.is_dir():
        raise SourceNotReady([f"源目录不存在: {src_dir}"])
    found: list[tuple[float, Path]] = []
    for f in sorted(src_dir.glob("*.nc")):
        ma = parse_ma_label(f.name)
        if tier == "1deg":
            ma = _ONEDEG_LABEL_ALIGN.get(ma, ma)
        found.append((ma, f))
    if len(found) != expected:
        raise SourceNotReady(
            [f"{src_dir.name}: {len(found)} 片 ≠ 期望 {expected}（0–540 Ma 全收）"]
        )
    found.sort(key=lambda t: t[0])
    labels = [ma for ma, _ in found]
    if any(b <= a for a, b in zip(labels, labels[1:])):
        raise ValueError(f"{src_dir.name}: Ma 标签存在重复（源漂移，须复核）")
    if labels[0] < MA_RANGE[0] or labels[-1] > MA_RANGE[1]:
        raise ValueError(
            f"{src_dir.name}: Ma 标签范围 [{labels[0]:g}, {labels[-1]:g}] 超出"
            f"契约 {MA_RANGE[0]:g}–{MA_RANGE[1]:g} Ma（0–540Ma 全收）"
        )
    return found


def _grid_res(nlon_nodes: int) -> Fraction:
    """源网格步长（度，精确有理数）：360/(节点数-1)。"""
    return Fraction(360, nlon_nodes - 1)


def lat_nodes_asc(product: str) -> np.ndarray:
    """源纬度节点（升序，float64，自契约常量推导；读取器已断言文件坐标一致）。"""
    nlat = SIXMIN_NLAT if product == "6min" else ONEDEG_NLAT
    nlon = SIXMIN_NLON if product == "6min" else ONEDEG_NLON
    return -90.0 + float(_grid_res(nlon)) * np.arange(nlat, dtype=np.float64)


def voronoi_lat_edges(lat_asc: np.ndarray) -> np.ndarray:
    """节点（升序）→ Voronoi 胞纬度边：相邻节点中点 + 极点 ±90 钳制。"""
    mids = 0.5 * (lat_asc[1:] + lat_asc[:-1])
    return np.concatenate(([-90.0], mids, [90.0]))


def _read_nc(path: Path, lat_name: str, lon_name: str) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """HDF5 读锁内完整读取单片（netCDF4 非线程安全，文件 I/O 全程持锁；
    返回 (dims 尺寸表, z 数组, lat 坐标, lon 坐标)，dataset 已在锁内关闭）。"""
    path = Path(path)
    with _HDF5_LOCK:
        with xr.open_dataset(path) as ds:
            if "z" not in ds.data_vars:
                raise ValueError(f"{path.name}: 缺变量 z")
            for nm in (lat_name, lon_name):
                if nm not in ds.coords:
                    raise ValueError(f"{path.name}: 缺坐标 {nm}")
            return (
                dict(ds.sizes),
                ds["z"].values,
                ds[lat_name].values.astype(np.float64),
                ds[lon_name].values.astype(np.float64),
            )


def read_slice_6min(path: str | Path) -> np.ndarray:
    """读 6min 产品单片 → (1801, 3600) float32 南起、缝合并。

    契约：dims (1801, 3601)、纬度北起降序/经度升序均匀步长（浮点容差
    1e-6，源坐标有 ~1e-11 表示漂移）、z float32 全有限、±180 缝差 ≤ 护栏。
    归一：行翻转南起 + 缝两列（独立采样）取均值弃 +180 列。
    """
    sizes, a, lat, lon = _read_nc(Path(path), "latitude", "longitude")
    name = Path(path).name
    if sizes != {"latitude": SIXMIN_NLAT, "longitude": SIXMIN_NLON}:
        raise ValueError(
            f"{name}: dims {sizes} ≠ 期望 ({SIXMIN_NLAT}, {SIXMIN_NLON})"
        )
    if a.dtype != np.float32:
        raise ValueError(f"{name}: z dtype {a.dtype} ≠ float32（源漂移）")
    res = float(_grid_res(SIXMIN_NLON))
    exp_lat = 90.0 - res * np.arange(SIXMIN_NLAT, dtype=np.float64)
    exp_lon = -180.0 + res * np.arange(SIXMIN_NLON, dtype=np.float64)
    if not (np.allclose(lat, exp_lat, atol=1e-6) and np.allclose(lon, exp_lon, atol=1e-6)):
        raise ValueError(f"{name}: 纬/经度坐标 ≠ 北起降序节点注册（源漂移）")
    if not np.isfinite(a).all():
        raise ValueError(f"{name}: 含非有限值（实证 109 片 0 缺测，源已变更）")
    lo, hi = SIXMIN_VALUE_RANGE
    if a.min() < lo or a.max() > hi:
        raise ValueError(
            f"{name}: 值域 [{a.min():.0f}, {a.max():.0f}] 超出契约 "
            f"[{lo:g}, {hi:g}] m（源漂移，须复核）"
        )
    seam = float(np.abs(a[:, 0] - a[:, -1]).max())
    if seam > SEAM_MAX_DIFF:
        raise ValueError(
            f"{name}: ±180 缝差 {seam:.0f} m 超护栏 {SEAM_MAX_DIFF:g}"
            "（缝语义漂移，须复核 registration_rule）"
        )
    # 北起 → 南起
    a = a[::-1]
    # 缝合并：±180 两列（缝两侧独立采样）取均值，弃 +180 列
    merged = 0.5 * (a[:, 0] + a[:, -1])
    return np.concatenate([merged[:, None], a[:, 1:-1]], axis=1)


def read_slice_1deg(path: str | Path) -> np.ndarray:
    """读 1° 产品单片 → (181, 360) float32，弃逐位相同的 +180 列。

    契约：dims (181, 361)、纬度南起升序/经度升序整数节点、z float32 全有限、
    ±180 两列逐位相同（实证 109 片缝差恰 0，grdsample 周期化取 -180 列）。
    """
    sizes, a, lat, lon = _read_nc(Path(path), "lat", "lon")
    name = Path(path).name
    if sizes != {"lat": ONEDEG_NLAT, "lon": ONEDEG_NLON}:
        raise ValueError(
            f"{name}: dims {sizes} ≠ 期望 ({ONEDEG_NLAT}, {ONEDEG_NLON})"
        )
    if a.dtype != np.float32:
        raise ValueError(f"{name}: z dtype {a.dtype} ≠ float32（源漂移）")
    res = float(_grid_res(ONEDEG_NLON))
    exp_lat = -90.0 + res * np.arange(ONEDEG_NLAT, dtype=np.float64)
    exp_lon = -180.0 + res * np.arange(ONEDEG_NLON, dtype=np.float64)
    if not (np.allclose(lat, exp_lat, atol=1e-6) and np.allclose(lon, exp_lon, atol=1e-6)):
        raise ValueError(f"{name}: 纬/经度坐标 ≠ 南起升序节点注册（源漂移）")
    if not np.isfinite(a).all():
        raise ValueError(f"{name}: 含非有限值（实证 109 片 0 缺测，源已变更）")
    lo, hi = ONEDEG_VALUE_RANGE
    if a.min() < lo or a.max() > hi:
        raise ValueError(
            f"{name}: 值域 [{a.min():.0f}, {a.max():.0f}] 超出契约 "
            f"[{lo:g}, {hi:g}] m（源漂移，须复核）"
        )
    if not np.array_equal(a[:, 0], a[:, -1]):
        raise ValueError(
            f"{name}: ±180 两列不逐位相同（实证缝差恰 0，源已变更，"
            "须复核 registration_rule）"
        )
    return a[:, :-1].copy()


def _reader(tier: str):
    """档位 → 官方产品读取器（调用期解析，测试可缩格注入常量）。"""
    if tier == "6min":
        return read_slice_6min
    if tier == "1deg":
        return read_slice_1deg
    raise ValueError(f"非 PaleoDEM 档位: {tier}")


def _src_dir(tier: str) -> Path:
    """档位 → 官方产品源目录（调用期解析）。"""
    return SIXMIN_DIR if tier == "6min" else ONEDEG_DIR


def _geom(tier: str) -> tuple[int, int]:
    """档位 → 源节点网格 (nlat, nlon)（调用期解析）。"""
    return (SIXMIN_NLAT, SIXMIN_NLON) if tier == "6min" else (ONEDEG_NLAT, ONEDEG_NLON)


def paleo_tier_field(tier: str, path: str | Path) -> np.ndarray:
    """官方产品单片 → 立方档位保守场（纯函数，构建/保真/测试共用）。"""
    raw = _reader(tier)(path)                             # 锁外执行 numpy 核
    _nlat, nlon = _geom(tier)
    edges = voronoi_lat_edges(lat_nodes_asc(tier))
    values, _W = conservative_overlap_mean(
        raw, np.ones(raw.shape, dtype=bool), edges,
        _grid_res(nlon), tier_fraction(tier),
    )
    return values


def time_coord_attrs() -> dict:
    """time 坐标属性（构建端唯一来源，结构验证对照）。"""
    return {
        "units": "Ma",
        "long_name": "geologic age before present (source filename Ma label)",
        "direction": "increasing age (into the past)",
    }


def build_paleodem_elevation(entry: LayerEntry, tier: str) -> xr.DataArray:
    """构建 paleo__paleodem_elevation（[time, lat, lon]，1°/6min 两档）。"""
    if tier not in ("1deg", "6min"):
        raise ValueError(f"{entry.id}: 契约档位仅 [1deg, 6min]，收到 {tier}")
    slices = scan_slices(_src_dir(tier), tier=tier)
    nlat, nlon = grid_shape(tier)
    chunks = {"6min": (1, 900, 900), "1deg": (1, 180, 360)}[tier]
    fields = [
        da.from_delayed(
            delayed(paleo_tier_field, pure=True)(tier, path),
            shape=(nlat, nlon), dtype=np.float32,
        )
        for _ma, path in slices
    ]
    arr = da.stack(fields, axis=0).rechunk(chunks)
    lat, lon = tier_centers(tier)
    time = np.array([ma for ma, _ in slices], dtype=np.float64)
    attrs = {
        "units": entry.unit,
        "long_name": "古高程（Scotese & Wright 2018 PaleoDEM；负值=古海洋深度，海平面=0）",
        "source": entry.source,
        "source_url": SOURCE_URL,
        "native_res": entry.native_res,
        "resampling": entry.resampling,
        "visibility": entry.visibility,
        "product_type": "model（古地理重建 DEM，非观测）",
        "registration_rule": _REGISTRATION,
        "time_levels": len(slices),
        "time_semantics": _TIME_SEMANTICS,
        "datatree_node": PALEO_GROUP,          # 相对组名：store 内 /{tier}/paleodem
        "license_note": _LICENSE_NOTE,
        "source_file": (
            "paleodem-6min-nc/*.nc ×109（6min 档）/ paleodem-1deg-nc/*.nc ×109"
            "（1° 档，官方产品直读）"
        ),
    }
    return xr.DataArray(
        arr, dims=("time", "lat", "lon"),
        coords={
            "time": ("time", time, time_coord_attrs()),
            "lat": lat, "lon": lon,
        },
        name=entry.id, attrs=attrs,
    )


# ---------------- 保真验证（本层自供）----------------


def build_paleo_fidelity_report(store_dir, layer_id: str, tiers: list[str],
                                src_dir=None) -> dict:
    """保真报告：逐档位级重算一致 + 积分守恒 + time 坐标 + 跨档年代均值曲线。

    src_dir 参数接受但忽略（源路径由常量解析，两档各随其官方产品目录）。
    逐片单遍循环内同时完成位级比对、积分守恒与全球均值（跨档一致性信号），
    不二次读源。判据：位级 NaN 感知逐位一致；积分 Σv·W = Σv·w 相对偏差
    < 0.1%（硬判据）；time 坐标 = 文件名 Ma 标签（1° 档经官方
    整数惯例对齐）严格升序、两档标签集逐值相等（定向断言）。
    """
    if layer_id != PALEO_LAYER_ID:
        raise KeyError(f"非 PaleoDEM 层: {layer_id}")
    checks: list[dict] = []
    # 跨档一致性：{ma: 全球面积加权均值}（对齐后两档标签 109 全量共同）
    means: dict[str, dict[float, float]] = {}
    time_labels: dict[str, np.ndarray] = {}
    rounded_1deg: list[tuple[float, float]] = []   # 1° 被重标定文件（raw → 对齐）
    rounded_txt = ""                               # 重标定清单文案（"385.2→385、…"）

    for tier in tiers:
        if tier not in ("1deg", "6min"):
            continue
        reader = _reader(tier)
        slices = scan_slices(_src_dir(tier), tier=tier)
        if tier == "1deg":
            rounded_1deg = sorted(
                (parse_ma_label(p.name), ma) for ma, p in slices
                if parse_ma_label(p.name) != ma
            )
            rounded_txt = "、".join(f"{raw:g}→{a:g}" for raw, a in rounded_1deg)
        _nl, nlon_nodes = _geom(tier)
        time_labels[tier] = np.array([ma for ma, _ in slices])
        edges = voronoi_lat_edges(lat_nodes_asc(tier))
        src_area = _ROW_AREA_FACTOR * (
            np.sin(np.deg2rad(edges[1:])) - np.sin(np.deg2rad(edges[:-1]))
        ) * float(_grid_res(nlon_nodes))
        valid = np.ones((edges.size - 1, nlon_nodes - 1), dtype=bool)

        n_bad = 0
        worst_int = 0.0
        vmin, vmax = np.inf, -np.inf
        means[tier] = {}
        with xr.open_datatree(store_dir, engine="zarr", chunks={}) as tree:
            node = tree[f"/{tier}/{PALEO_GROUP}"]
            if layer_id not in node.ds.data_vars:
                raise KeyError(f"store {store_dir} 不含 {layer_id}（先 build 再 fidelity）")
            arr = node.ds[layer_id]
            t = arr.coords["time"].values.astype(np.float64)
            for k, (ma, path) in enumerate(slices):
                raw = reader(path)
                expect, W = conservative_overlap_mean(
                    raw, valid, edges, _grid_res(nlon_nodes), tier_fraction(tier),
                )
                v = arr.isel(time=k).values
                # NaN 感知位级比对（NaN != NaN；本层实证无缺测，防御式保留）
                eq = (v == expect) | (np.isnan(v) & np.isnan(expect))
                n_bad += int((~eq).sum())
                # 积分守恒（核恒等式独立路径：目标 Σv·W vs 源 Σv·w）
                s_t = float(np.dot(expect.astype(np.float64).ravel(), W.ravel()))
                s_s = float(np.dot(raw.astype(np.float64).sum(axis=1), src_area))
                worst_int = max(worst_int, abs(s_t - s_s) / abs(s_s))
                # 全球面积加权均值（跨档一致性信号，与积分同一遍）
                means[tier][ma] = s_t / float(W.sum())
                fv = expect[np.isfinite(expect)]
                vmin, vmax = min(vmin, float(fv.min())), max(vmax, float(fv.max()))

        checks.append({
            "name": f"{tier}: 与源核重算逐位一致（float32 位级，{len(slices)} 时间片全量）",
            "status": "pass" if n_bad == 0 else "FAIL",
            "summary": f"不一致元素 {n_bad}（期望 0）",
        })
        round_note = (
            f"；标签对齐：官方整数惯例舍入 {rounded_txt or '0 片'}"
            if tier == "1deg" else ""
        )
        checks.append({
            "name": (
                f"{tier}: time 坐标 = 文件名 Ma 标签（升序，{len(slices)} 片"
                + ("，1° 档经官方整数惯例对齐" if tier == "1deg" else "") + "）"
            ),
            "status": "pass" if np.array_equal(t, time_labels[tier]) else "FAIL",
            "summary": f"首末 {t[0]:g}/{t[-1]:g} Ma，期望首末 "
                       f"{time_labels[tier][0]:g}/{time_labels[tier][-1]:g}"
                       + round_note,
        })
        checks.append({
            "name": f"{tier}: 积分守恒（各时间片最大相对偏差 < 0.1%）",
            "status": "pass" if worst_int < INTEGRAL_TOL else "FAIL",
            "summary": f"{len(slices)} 片最大相对偏差 {worst_int:.2e}",
        })
        checks.append({
            "name": f"{tier}: 值域与覆盖（报告制）",
            "status": "report",
            "summary": (
                f"z ∈ [{vmin:.0f}, {vmax:.0f}] m，全覆盖（源 109 片 0 缺测 → "
                "有效性掩膜全 1；古地理以数值编码，负值=古海洋深度）"
            ),
        })

    # 定向断言：两档 time 标签集逐值相等（1° 被重标定清单随行展示；
    # 「恰两片 = 385/390」的非循环硬断言在 test_real_label_alignment，
    # 以硬编码期望对真实源实证）
    if "6min" in time_labels and "1deg" in time_labels:
        n6, n1 = len(time_labels["6min"]), len(time_labels["1deg"])
        aligned_ok = bool(np.array_equal(time_labels["6min"], time_labels["1deg"]))
        checks.append({
            "name": f"跨档 time 标签集对齐（两档 {n6}/{n1} 逐值相等）",
            "status": "pass" if aligned_ok else "FAIL",
            "summary": (
                f"两档标签集逐值{'相等' if aligned_ok else '不等'}；1° 被重标定 "
                f"{len(rounded_1deg)} 片：" + (rounded_txt or "无")
                + "（按官方整数命名惯例与标签值映射，非 Map 编号）"
            ),
        })

    # 跨档一致性（报告制）：共同 Ma 标签的全球面积加权均值差
    if "6min" in means and "1deg" in means:
        common = sorted(set(means["6min"]) & set(means["1deg"]))
        diffs = [abs(means["6min"][ma] - means["1deg"][ma]) for ma in common]
        full_common = len(common) == len(means["6min"]) == len(means["1deg"])
        checks.append({
            "name": f"跨档一致性：{len(common)} 共同年代的全球均值曲线（报告制）",
            "status": "report",
            "summary": (
                f"6min 档 vs 1° 档全球面积加权均值 |Δ| 均值 {np.mean(diffs):.1f} m / "
                f"最大 {np.max(diffs):.1f} m（两档各随其官方产品，独立重采样；"
                f"共同标签 {len(common)}/{len(means['6min'])}"
                + ("——标签对齐后全量共同，同年代切片可直接联用"
                   if full_common else "——未全量共同，见上方标签集对齐检查")
                + "）"
            ),
        })
    checks.append({
        "name": "time 轴语义登记",
        "status": "report",
        "summary": _TIME_SEMANTICS,
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
