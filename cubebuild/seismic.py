"""3D 地震模型通道：GLAD-M35 五变量 + SEMUCB-WM1 两变量。

源契约（磁盘实证，坐标/深度/值域均经实机核对）：

- GLAD-M35.r0.1-n4c.nc（EarthScope EMC，Cui et al. 2024 GJI）：289 深度层
  （10..2890 km，均匀步长 10 km）× 181 纬节点（-90..90 整数度）× 361 经
  节点（-180..180 整数度），gridline 节点注册；变量 vpv/vph/vsv/vsh/eta
  （另有 std_vp/std_vs，映射表未登记不 ingest）。全场无 NaN（含极区行，
  实证 5 变量 0 缺测）。±180 两列值**不同**（各变量最大差 0.008–0.120
  km/s）——GLAD 系全波形反演在立方球面网格上进行，±180 子午线是网格缝，
  两列是缝两侧的独立采样（与 WGM2012 的周期重复列性质不同，不可择一
  弃置）。DOI 10.1093/gji/ggae270。

- SEMUCB-WM1（French & Romanowicz 2014 GJI，DOI 10.1093/gji/ggu334）：
  IRIS EMC 的两个 netCDF 名含 viz-only——实测 vs 为 183×365 的 0.99°
  非均匀网格、xi 仅 47×93（~3.95°）抽稀网格，均为可视化精简版而非完整
  模型（viz 网格步长 0.99/3.95 且含调整间隔以凑 ±90/±180）。完整模型
  （球面样条连续场）在官方源码分发 UCB_a3d_dist（评估工具 a3d_dist.x，
  README 明示 "-d 1" 1° 网格为其设计用法）。本通道以官方工具在立方 1°
  档像元中心（edge-aligned 半度中心）逐点评估原生连续场——非重采样，
  是模型自身参数化在其设计分辨率下的采样。74 官方深度层取自 viz 产品
  深度坐标（30..2891 km = 模型发布采样，半径 = 6371 − 深度，CMB 3480）。
  - 地壳语义：a3d_dist 在平滑地壳层内输出 nan（作者声明该区值不可解释；
    moho.dat 实测范围 30..59.994 km → 深度 30 层全 NaN、40/50 层部分
    NaN、≥60 km 无 NaN）。viz 产品绕过了地壳检查（depth 30 有值）——
    本通道遵循官方工具语义（NaN + validity 掩膜携带），分歧登记
    registration_rule。
  - 点级交叉验证（磁盘实证）：depth=100 处 300 个随机 viz
    网格点 vs 值与工具输出全一致（max|Δ|=4.7e-4，即 %6.3f 文本输出精度）；
    xi 200 点 max|Δ|=4.4e-4——工具链复现官方 viz 产品，纬经度约定一致。
  - 分发包陷阱：UCB_a3d_dist 的 data/model.A3d 与 data/model.ref 为
    26/25 字节指针文本（symlink 在分发中退化为内容为文件名的普通文件），
    构建副本中须重建符号链接后方可编译运行（gen_1d 分发中同位置是正常
    symlink，佐证退化判读）。

写入结构：两模型深度坐标不同（289 vs 74 层），不能共存于同一 Dataset
节点 → 各入档位节点的子节点 /1deg/gladm35 与 /1deg/semucb；维序统一
[depth, lat, lon]，depth 逐层分块（chunks (1, 180, 360)）支持下游一次
.sel(depth=) 取任意深度切片。

GLAD 配准（registration_rule）：节点 → Voronoi 胞解释（WGM 惯例：
节点是场的点采样，Voronoi 胞是保守平均的最小偏置离散化）——纬度边
[-90, -89.5, ..., 89.5, 90]（极点行半胞）、经度 1° 胞环接日期线，
经保守核 conservative_overlap_mean 汇入立方 1° 档（目标胞 = 相邻节点
面积加权平均；映射表 resampling=none 指**不跨档重采样**——层仅入 1°
全集档，节点→胞的保守平均是配准解释而非档位聚合）。日期线缝：±180
两列取均值（缝两侧采样的对称保守化，实证差 ≤0.12 km/s 登记上界）。
"""

import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

import dask.array as da
import numpy as np
import xarray as xr

from .contract import LayerEntry
from .etopo import SourceNotReady
from .fidelity import INTEGRAL_TOL
from .grids import tier_centers
from .kernels import conservative_overlap_mean
from .pixel_area import EARTH_RADIUS_KM

# 源胞行面积因子（与 kernels._ROW_AREA_FACTOR 同约定：积分恒等式两侧
# 必须同一 R 与单位折算，否则守恒判据失真）
_ROW_AREA_FACTOR = EARTH_RADIUS_KM**2 * np.pi / 180.0

REPO_ROOT = Path(__file__).resolve().parents[1]
GLAD_SRC_DIR = REPO_ROOT / "original data/seismology/glad-m35"
GLAD_FILE = "GLAD-M35.r0.1-n4c.nc"
SEMUCB_SRC_DIR = REPO_ROOT / "original data/seismology/semucb-wm1"
SEMUCB_VIZ_VS = "semucb-2014-ucb-vs-viz-only.r0.1-n4c.nc"
SEMUCB_VIZ_XI = "semucb-2014-ucb-xi-viz-only.r0.1-n4c.nc"
SEMUCB_DIST_DIRNAME = "SEMUCB-WM1-Model-main/UCB_a3d_dist.SEMUCB-WM1.r20151019"
SEMUCB_DIST_DIR = SEMUCB_SRC_DIR / SEMUCB_DIST_DIRNAME
# 构建工作区（products/ 不入库）：编译工具与评估中间产物
BUILD_ROOT = REPO_ROOT / "products/.build"
A3D_BUILD_DIR = BUILD_ROOT / "ucb_a3d_dist"
A3D_MODEL_NAME = "Model-2.6S6X4-ZeroMean.A3d"
A3D_REF_NAME = "Model-2.6S6X4-ZeroMean_1D"

GLAD_DOI = "10.1093/gji/ggae270"               # Cui et al. 2024, GJI（nc 头 reference_pid 自证）
SEMUCB_DOI = "10.1093/gji/ggu334"              # French & Romanowicz 2014, GJI（nc 头自证）

# ---- 源契约常量（磁盘实证的契约化；源变更即读取期显式失败）----
GLAD_NDEPTH, GLAD_NLAT, GLAD_NLON = 289, 181, 361
GLAD_VARS = ("vsv", "vsh", "vpv", "vph", "eta")
GLAD_DEPTH = np.arange(10, 2891, 10, dtype=np.float64)          # 289 层，10 km 均匀
GLAD_SEAM_MAX_DIFF = 0.5        # ±180 列差上界（实证 0.008–0.120；宽松护栏防源替换）

R_EARTH_SEMUCB = 6371.0         # a3d_dist constants.h R_EARTH_KM
SEMUCB_NDEPTH = 74
# 74 官方深度层（viz 产品深度坐标的契约化；30..110 步 10 / 130..410 步 20 / 441..2891 步 50）
SEMUCB_DEPTHS = tuple(
    [30 + 10 * k for k in range(9)]
    + [130 + 20 * k for k in range(15)]
    + [441 + 50 * k for k in range(50)]
)
SEMUCB_CRUST_NAN_DEPTHS = (30.0, 40.0, 50.0)   # moho.dat∈[30,59.994] ⇒ 仅这三层可含 NaN
# CMB 层评估半径（磁盘实证 250 点：viz 2891 层 = r=3480.5 评估
# |Δ|≤9.3e-4 全中；r=3480.0 恰在 CMB，get_ucb_ref_ 取外核侧 vs=0 致
# (vsv,vsh) 重建 0/0=nan——重建缺陷非模型未定义（dvs/dxi 有限）。
# EMC viz 同约定取 CMB 上方 0.5 km，本通道跟随）
SEMUCB_CMB_RADIUS = 3480.5

# 3D 层 → 构建器变量名；组节点名（store 内 /1deg/<group>）
GLAD_GROUP = "gladm35"
SEMUCB_GROUP = "semucb"
GLAD_LAYER_VAR = {
    "seismology__gladm35_vsv": "vsv",
    "seismology__gladm35_vsh": "vsh",
    "seismology__gladm35_vpv": "vpv",
    "seismology__gladm35_vph": "vph",
    "seismology__gladm35_eta": "eta",
}
SEMUCB_LAYER_VAR = {
    "seismology__semucb_vs": "vs",
    "seismology__semucb_xi": "xi",
}
# 组成员登记（cli 路由 3D 层入子节点 + validate/coverage 节点解析用）
SEISMIC_3D_GROUPS: dict[str, tuple[str, ...]] = {
    GLAD_GROUP: tuple(GLAD_LAYER_VAR),
    SEMUCB_GROUP: tuple(SEMUCB_LAYER_VAR),
}
SEISMIC_3D_GROUP_OF = {lid: g for g, ids in SEISMIC_3D_GROUPS.items() for lid in ids}

_TIER = "1deg"
_ONE = Fraction(1)
# GLAD 节点 Voronoi 纬度边：[-90, -89.5, -88.5, ..., 89.5, 90]（182 = 181+1）
GLAD_LAT_EDGES = np.concatenate(
    ([-90.0], np.arange(-89.5, 90.0, 1.0), [90.0])
)

_OVERLAP_NOTE = (
    "GLAD-M35（上地幔，10–2890 km）与 SEMUCB-WM1（全地幔至 CMB，30–2891 km）"
    "上地幔深度段重叠为有意保留：两模型方法独立（波形反演 vs 谱元波形层析），"
    "重叠段差异即模型不确定性特征（克制原则互补豁免）"
)

_LICENSE_NOTE = (
    "两模型经 EarthScope/IRIS EMC 分发，netCDF 头无许可证字段；公开版发布前"
    "须核实 EMC 产品页授权条款（ Lucazeau 2019 SI 同款待办，GSRM 惯例可转 internal）"
)

_GLAD_REGISTRATION = (
    "gridline 节点注册（整数度节点 -90..90/-180..180）→ Voronoi 胞解释（"
    "WGM 惯例）：纬度边 [-90,-89.5,…,89.5,90] 极点行半胞、经度 1° 胞环接日期线，"
    "保守核汇入立方 1° 档（目标胞 = 相邻节点面积加权平均）；映射表 resampling=none "
    "指不跨档重采样（层仅入 1° 全集档），节点→胞保守平均是配准解释而非档位聚合。"
    "日期线缝：±180 两列（立方球面网格缝两侧独立采样，实证差 ≤0.12 km/s）取均值"
)

_SEMUCB_REGISTRATION = (
    "原生连续场逐点评估（非重采样）：官方工具 a3d_dist（UCB_a3d_dist 分发，gcc "
    "编译于 products/.build）在立方 1° 档像元中心逐点评估球面样条场，74 官方深度"
    "层（viz 产品深度坐标，半径 = 6371 − 深度；CMB 层 2891 km 在 r=3480.5 km "
    "评估——EMC viz 同约定，r=3480.0 恰在 CMB 处参考模型取外核侧 vs=0 致 "
    "(vsv,vsh) 重建 0/0，250 点实证 viz 2891 = r=3480.5 |Δ|≤9.3e-4 全中）。"
    "地壳语义遵循官方工具：平滑地壳层内输出 NaN（作者声明不可解释；"
    "moho.dat∈[30,59.994] km ⇒ 深度 30 层全 NaN、40/50 层部分 NaN、≥60 km "
    "全有效）；viz-only 产品绕过该检查（depth 30 有值），本通道不跟随——分歧即"
    "地壳不可解释语义的体现。viz-only nc 为可视化精简版（vs 0.99° 非均匀 / "
    "xi ~3.95° 抽稀），不作 ingest 源，仅作点级交叉验证参考（depth=100 处 "
    "300/300 点 vs 一致 max|Δ|=4.7e-4 = 文本输出精度）"
)


# ---------------- GLAD-M35：读取与保守配准 ----------------

_GLAD_CACHE: dict[str, dict] = {}


def load_glad_m35(src_dir: str | Path = GLAD_SRC_DIR) -> dict:
    """读 GLAD-M35 nc → 规范化源（节点 → Voronoi 胞，日期线缝合并）。

    契约：dims (289, 181, 361)、深度恰 10..2890 步 10、纬经度恰整数节点、
    五变量全场无 NaN（实证 0 缺测）、±180 列差 ≤ 护栏上界（缝语义漂移检测）。

    返回 {var: (289, 181, 360) float32（缝合并后，弃 +180 列）,
          "depth", "seam_max_diff"}；进程内缓存（五层只读一次源）。
    """
    key = f"glad:{Path(src_dir).resolve()}"
    if key in _GLAD_CACHE:
        return _GLAD_CACHE[key]
    path = Path(src_dir) / GLAD_FILE
    if not path.is_file():
        raise FileNotFoundError(f"GLAD-M35 源文件不存在: {path}")
    ds = xr.open_dataset(path)
    try:
        if dict(ds.sizes) != {
            "depth": len(GLAD_DEPTH), "latitude": GLAD_NLAT, "longitude": GLAD_NLON,
        }:
            raise ValueError(
                f"{GLAD_FILE}: dims {dict(ds.sizes)} ≠ 期望 "
                f"({len(GLAD_DEPTH)}, {GLAD_NLAT}, {GLAD_NLON})"
            )
        if not np.array_equal(ds.depth.values.astype(np.float64), GLAD_DEPTH):
            raise ValueError(
                f"{GLAD_FILE}: 深度坐标 ≠ {GLAD_DEPTH[0]:g}..{GLAD_DEPTH[-1]:g} km"
                f" 均匀 {GLAD_DEPTH[1] - GLAD_DEPTH[0]:g} km（{len(GLAD_DEPTH)} 层）"
            )
        lat = ds.latitude.values.astype(np.float64)
        lon = ds.longitude.values.astype(np.float64)
        if not (
            np.array_equal(lat, np.arange(-90.0, 91.0, 1.0))
            and np.array_equal(lon, np.arange(-180.0, 181.0, 1.0))
        ):
            raise ValueError(f"{GLAD_FILE}: 纬/经度坐标 ≠ 整数度节点注册")
        out: dict = {"depth": GLAD_DEPTH.copy(), "seam_max_diff": 0.0}
        for v in GLAD_VARS:
            if v not in ds.data_vars:
                raise ValueError(f"{GLAD_FILE}: 缺变量 {v}")
            a = ds[v].values
            if not np.isfinite(a).all():
                raise ValueError(f"{GLAD_FILE}: {v} 含非有限值（实证全场无 NaN，源已变更）")
            seam = float(np.abs(a[:, :, 0] - a[:, :, -1]).max())
            if seam > GLAD_SEAM_MAX_DIFF:
                raise ValueError(
                    f"{GLAD_FILE}: {v} ±180 列差 {seam:.3f} 超护栏 {GLAD_SEAM_MAX_DIFF}"
                    "（缝语义漂移，须复核 registration_rule）"
                )
            out["seam_max_diff"] = max(out["seam_max_diff"], seam)
            # 日期线缝合并：±180 两列（缝两侧独立采样）取均值，弃 +180 列
            merged = 0.5 * (a[:, :, 0] + a[:, :, -1])
            out[v] = np.concatenate([merged[:, :, None], a[:, :, 1:-1]], axis=2)
        _GLAD_CACHE[key] = out
        return out
    finally:
        ds.close()


def glad_conservative_to_tier(values: np.ndarray) -> np.ndarray:
    """GLAD 节点场 (289, 181, 360) → 立方 1° 档 (289, 180, 360)。

    逐深度层走通用重叠保守核（Voronoi 胞源几何；与重磁组同一实现，
    不另写核）。全场有效 → 输出无 NaN。
    """
    nd = values.shape[0]
    out = np.empty((nd, 180, 360), dtype=np.float32)
    valid = np.ones((GLAD_NLAT, 360), dtype=bool)
    for k in range(nd):
        v, _W = conservative_overlap_mean(
            values[k], valid, GLAD_LAT_EDGES, _ONE, _ONE
        )
        out[k] = v
    return out


_GLAD_TIER_CACHE: dict[str, np.ndarray] = {}


def _glad_tier_field(var: str) -> np.ndarray:
    """源 → 1° 档保守场（进程内缓存：五层共享同一次核计算）。"""
    key = f"glad-tier:{var}"
    if key not in _GLAD_TIER_CACHE:
        _GLAD_TIER_CACHE[key] = glad_conservative_to_tier(load_glad_m35()[var])
    return _GLAD_TIER_CACHE[key]


_GLAD_LONG_NAME = {
    "vsv": "垂直偏振 S 波速度（GLAD-M35）",
    "vsh": "水平偏振 S 波速度（GLAD-M35）",
    "vpv": "垂直偏振 P 波速度（GLAD-M35）",
    "vph": "水平偏振 P 波速度（GLAD-M35）",
    "eta": "无量纲各向异性参数 η（GLAD-M35）",
}


def build_glad_layer(entry: LayerEntry, tier: str) -> xr.DataArray:
    """构建 seismology__gladm35_*（[depth, lat, lon]，仅 1° 档）。"""
    if tier != _TIER:
        raise ValueError(f"{entry.id}: 契约档位仅 [{_TIER}]，收到 {tier}")
    var = GLAD_LAYER_VAR[entry.id]
    values = _glad_tier_field(var)
    arr = da.from_array(values, chunks=(1, 180, 360))
    lat, lon = tier_centers(_TIER)
    depth = load_glad_m35()["depth"]
    attrs = {
        "units": entry.unit,
        "long_name": _GLAD_LONG_NAME[var],
        "source": entry.source,
        "doi": GLAD_DOI,
        "native_res": entry.native_res,
        "resampling": entry.resampling,
        "visibility": entry.visibility,
        "product_type": "model（全波形反演层析模型，非观测）",
        "registration_rule": _GLAD_REGISTRATION,
        "depth_levels": int(values.shape[0]),
        "datatree_node": f"/{_TIER}/{GLAD_GROUP}",
        "overlap_note": _OVERLAP_NOTE,
        "license_note": _LICENSE_NOTE,
        "source_file": GLAD_FILE,
    }
    return xr.DataArray(
        arr, dims=("depth", "lat", "lon"),
        coords={"depth": ("depth", depth, depth_coord_attrs()), "lat": lat, "lon": lon},
        name=entry.id, attrs=attrs,
    )


# ---------------- SEMUCB-WM1：官方工具原生评估 ----------------

def semucb_depths(src_dir: str | Path = SEMUCB_SRC_DIR) -> np.ndarray:
    """74 官方深度层（读 viz vs nc 深度坐标并契约断言）。"""
    path = Path(src_dir) / SEMUCB_VIZ_VS
    if not path.is_file():
        raise FileNotFoundError(f"SEMUCB viz 源文件不存在: {path}")
    with xr.open_dataset(path) as ds:
        dep = ds.depth.values.astype(np.float64)
    if len(dep) != SEMUCB_NDEPTH or tuple(dep) != SEMUCB_DEPTHS:
        raise ValueError(
            f"{SEMUCB_VIZ_VS}: 深度坐标 {len(dep)} 层 ≠ 契约 74 层官方采样"
            f"（{SEMUCB_DEPTHS[0]}..{SEMUCB_DEPTHS[-1]}）——源变更须复核"
        )
    return dep


def ensure_a3d_tool() -> Path:
    """编译官方评估工具（products/.build/ucb_a3d_dist），返回工具目录。

    幂等：二进制存在即复用；首次复制分发树并修复 model.A3d/model.ref
    指针文本（symlink 退化产物）后 make。gcc 缺失或编译失败 →
    SourceNotReady（构建期跳过该组层）。
    """
    binary = A3D_BUILD_DIR / "a3d_dist.x"
    if binary.is_file():
        return A3D_BUILD_DIR
    if not (SEMUCB_DIST_DIR / "src/main.c").is_file():
        raise SourceNotReady([str(SEMUCB_DIST_DIR)])
    # 半成品目录（复制成功但编译失败等）清空重拷，保证幂等
    if A3D_BUILD_DIR.exists():
        shutil.rmtree(A3D_BUILD_DIR)
    shutil.copytree(SEMUCB_DIST_DIR, A3D_BUILD_DIR)
    data = A3D_BUILD_DIR / "data"
    for ptr, target in (("model.A3d", A3D_MODEL_NAME), ("model.ref", A3D_REF_NAME)):
        p = data / ptr
        if p.exists() and not p.is_symlink():
            content = p.read_text(encoding="utf-8").strip()
            if content == target:
                p.unlink()
            else:
                raise ValueError(
                    f"UCB_a3d_dist 分发包 {ptr} 为未知内容 {content!r}"
                    "（预期指针文本，须人工核查）"
                )
        if not p.exists():
            p.symlink_to(target)
    try:
        subprocess.run(
            ["make"], cwd=A3D_BUILD_DIR, check=True, capture_output=True, text=True,
        )
    except FileNotFoundError as e:
        raise SourceNotReady([f"make/gcc 不可用: {e}"]) from e
    except subprocess.CalledProcessError as e:
        raise SourceNotReady([f"a3d_dist 编译失败: {e.stderr[-500:]}"]) from e
    if not binary.is_file():
        raise SourceNotReady(["a3d_dist.x 编译后未生成"])
    return A3D_BUILD_DIR


def _cube_centers_1deg() -> tuple[np.ndarray, np.ndarray]:
    lat, lon = tier_centers(_TIER)
    return lat.astype(np.float64), lon.astype(np.float64)


def _eval_output_to_fields(
    df, depth: np.ndarray, radii: np.ndarray,
    lat: np.ndarray, lon: np.ndarray,
) -> dict:
    """a3d_dist 文本输出 → vs/xi 场（含行数/半径序/坐标回读自证）。

    vs = Voigt 平均 sqrt((2·vsv²+vsh²)/3)、xi = vsh²/vsv²（工具 main.c
    同公式自 3 位小数文本重建）；地壳内 nan 行原样传播。行序 = 半径块 ×
    纬外经内行主序。纯函数（df → dict），供评估与测试共用。
    """
    npts = lat.size * lon.size
    ndep = len(depth)
    if len(df) != ndep * npts:
        raise ValueError(f"a3d_dist 输出行数 {len(df)} ≠ {ndep}×{npts}")
    if not np.array_equal(df["radius"].to_numpy(), np.repeat(radii, npts)):
        raise ValueError("a3d_dist 输出半径序与请求不符")
    # 坐标回读自证（%8.3f 文本回显；3 位小数舍入比对——viz 网格 float32
    # 坐标如 -86.05000305 经 %.6f 喂入、%8.3f 回显，须舍入后比对）
    exp_lon = np.tile(lon, lat.size)
    exp_lat = np.repeat(lat, lon.size)
    for got, exp, name in (
        (df["lon"].to_numpy(), exp_lon, "lon"),
        (df["lat"].to_numpy(), exp_lat, "lat"),
    ):
        if not np.array_equal(np.round(got, 3), np.round(np.tile(exp, ndep), 3)):
            raise ValueError(f"a3d_dist 输出 {name} 坐标回读不符（配准自证失败）")

    vsv = df["vsv"].to_numpy()
    vsh = df["vsh"].to_numpy()
    with np.errstate(invalid="ignore"):
        vs = np.sqrt((2.0 * vsv**2 + vsh**2) / 3.0)
        xi = vsh**2 / vsv**2
    shape = (ndep, lat.size, lon.size)
    vs32 = vs.astype(np.float32).reshape(shape)
    return {
        "vs": vs32,
        "xi": xi.astype(np.float32).reshape(shape),
        "nan_counts": np.isnan(vs32).sum(axis=(1, 2)).astype(np.int64),
    }


def _run_a3d(radii: np.ndarray, lat: np.ndarray, lon: np.ndarray,
             pts_name: str, out_name: str) -> "object":
    """运行 a3d_dist（半径 × 点列表 → 七列 DataFrame）的唯一实现。

    点序：纬外经内行主序；评估与 viz 交叉验证共用（含 CMB 半径约定由
    调用方置入 radii）。输出列 radius/lon/lat/dvs_pct/dxi_pct/vsv/vsh。
    """
    import pandas as pd

    tool_dir = ensure_a3d_tool()
    pts = A3D_BUILD_DIR / pts_name
    out = A3D_BUILD_DIR / out_name
    with pts.open("w", encoding="ascii") as f:
        for la in lat:
            for lo in lon:
                f.write(f"{lo:.6f} {la:.6f}\n")
    cmd = [str(tool_dir / "a3d_dist.x")]
    for r in radii:
        cmd += ["-r", f"{r:.1f}"]
    cmd += ["-i", str(pts), "-o", str(out)]
    try:
        subprocess.run(cmd, cwd=tool_dir, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"a3d_dist 评估失败: {e.stderr[-500:]}") from e
    return pd.read_csv(
        out, sep=r"\s+", header=None,
        names=["radius", "lon", "lat", "dvs_pct", "dxi_pct", "vsv", "vsh"],
        dtype=np.float64,
    )


_SEMUCB_CACHE: dict[str, dict] = {}


def evaluate_semucb() -> dict:
    """a3d_dist 在立方 1° 档像元中心 × 74 官方深度层评估原生场。

    返回 {"vs": (74,180,360) float32, "xi": 同, "depth": (74,) float64,
    "nan_counts": (74,) 每层 NaN 数}；进程内缓存（build/保真共享一次评估）。
    点序：纬外经内行主序（输出行 ↔ 目标网格 ravel 顺序）。
    失败语义：viz nc 缺失 → FileNotFoundError（数据缺失硬失败，与 thermal/
    gravmag 同约定）；gcc/make 缺失或编译失败 → SourceNotReady（工具链
    问题，构建期优雅跳过该组层）。
    """
    if "semucb" in _SEMUCB_CACHE:
        return _SEMUCB_CACHE["semucb"]
    depth = semucb_depths()
    radii = R_EARTH_SEMUCB - depth
    radii[-1] = SEMUCB_CMB_RADIUS   # CMB 层：EMC viz 同约定（见常量注记）
    lat, lon = _cube_centers_1deg()
    df = _run_a3d(radii, lat, lon, "cube_1deg_pts.dat", "cube_1deg_eval.out")
    result = _eval_output_to_fields(df, depth, radii, lat, lon)
    result["depth"] = depth
    _SEMUCB_CACHE["semucb"] = result
    return result

_SEMUCB_LONG_NAME = {
    "vs": "Voigt 平均各向同性 S 波速度（SEMUCB-WM1）",
    "xi": "径向各向异性 ξ = (Vsh/Vsv)²（SEMUCB-WM1）",
}


def build_semucb_layer(entry: LayerEntry, tier: str) -> xr.DataArray:
    """构建 seismology__semucb_*（[depth, lat, lon]，仅 1° 档）。"""
    if tier != _TIER:
        raise ValueError(f"{entry.id}: 契约档位仅 [{_TIER}]，收到 {tier}")
    var = SEMUCB_LAYER_VAR[entry.id]
    ev = evaluate_semucb()
    arr = da.from_array(ev[var], chunks=(1, 180, 360))
    lat, lon = tier_centers(_TIER)
    attrs = {
        "units": entry.unit,
        "long_name": _SEMUCB_LONG_NAME[var],
        "source": entry.source,
        "doi": SEMUCB_DOI,
        "native_res": entry.native_res,
        "resampling": entry.resampling,
        "visibility": entry.visibility,
        "product_type": "model（谱元波形层析模型，非观测）",
        "registration_rule": _SEMUCB_REGISTRATION,
        "depth_levels": int(ev[var].shape[0]),
        "datatree_node": f"/{_TIER}/{SEMUCB_GROUP}",
        "overlap_note": _OVERLAP_NOTE,
        "license_note": _LICENSE_NOTE,
        "source_file": (
            f"{SEMUCB_DIST_DIRNAME}（a3d_dist 官方评估工具；viz-only nc 仅作"
            "交叉验证参考）"
        ),
    }
    return xr.DataArray(
        arr, dims=("depth", "lat", "lon"),
        coords={"depth": ("depth", ev["depth"], depth_coord_attrs()), "lat": lat, "lon": lon},
        name=entry.id, attrs=attrs,
    )


def depth_coord_attrs() -> dict:
    """depth 坐标属性（构建端唯一来源，结构验证对照）。"""
    return {
        "units": "km",
        "long_name": "depth below earth surface",
        "positive": "down",
    }


# ---------------- 保真验证（本层自供）----------------

# viz 交叉验证容差：a3d_dist 文本输出 %6.3f（vsv/vsh）量化误差传播
# （实证 vs 4.7e-4 / xi 4.4e-4，容差 = 实证 + 量化上界余量）
VS_TOL = 1.5e-3
XI_TOL = 3.0e-3

SEISMIC_LAYERS = {
    **{lid: {"kind": "glad", "var": v} for lid, v in GLAD_LAYER_VAR.items()},
    **{lid: {"kind": "semucb", "var": v} for lid, v in SEMUCB_LAYER_VAR.items()},
}


def _glad_fidelity(layer_id: str, var: str, tiers: list[str], store_dir) -> list[dict]:
    """GLAD：store 位级 == 核重算 + 积分守恒 + 深度层数/坐标。"""
    checks: list[dict] = []
    src = load_glad_m35()
    expect = glad_conservative_to_tier(src[var])
    with xr.open_datatree(store_dir, engine="zarr", chunks={}) as tree:
        node = tree[f"/{_TIER}/{GLAD_GROUP}"]
        if layer_id not in node.ds.data_vars:
            raise KeyError(f"store {store_dir} 不含 {layer_id}（先 build 再 fidelity）")
        arr = node.ds[layer_id]
        v = arr.values
        # NaN 感知位级比对（NaN != NaN；地壳内 NaN 行须视为一致）
        eq = (v == expect) | (np.isnan(v) & np.isnan(expect))
        n_bad = int((~eq).sum())
        checks.append({
            "name": f"{_TIER}: 与源核重算逐位一致（float32 位级，289 深度层全量）",
            "status": "pass" if n_bad == 0 else "FAIL",
            "summary": f"不一致元素 {n_bad}（期望 0）",
        })
        dep = arr.coords["depth"].values.astype(np.float64)
        checks.append({
            "name": f"{_TIER}: 深度坐标 = 源 289 层（10..2890 km 步 10）",
            "status": "pass" if np.array_equal(dep, GLAD_DEPTH) else "FAIL",
            "summary": f"深度层数 {len(dep)}，首末 {dep[0]:g}/{dep[-1]:g}",
        })
    # 积分守恒：逐深度层 Σ_t v·W 与源 Σ_s v·w（核恒等式，独立于位级比对；
    # w_s = 行面积因子×Δsin×lon_res，与核 _ROW_AREA_FACTOR 同 R 同折算）
    valid = np.ones((GLAD_NLAT, 360), dtype=bool)
    src_area = _ROW_AREA_FACTOR * (
        np.sin(np.deg2rad(GLAD_LAT_EDGES[1:])) - np.sin(np.deg2rad(GLAD_LAT_EDGES[:-1]))
    ) * float(_ONE)
    worst = 0.0
    for k in range(src[var].shape[0]):
        _v, W = conservative_overlap_mean(
            src[var][k], valid, GLAD_LAT_EDGES, _ONE, _ONE
        )
        s_t = float(np.dot(expect[k].astype(np.float64).ravel(), W.ravel()))
        # 源侧：先按经度求和再与行面积点积（w_s = 行面积×lon_res）
        s_s = float(np.dot(src[var][k].astype(np.float64).sum(axis=1), src_area))
        worst = max(worst, abs(s_t - s_s) / abs(s_s))
    checks.append({
        "name": f"{_TIER}: 积分守恒（各深度层最大相对偏差 < 0.1%）",
        "status": "pass" if worst < INTEGRAL_TOL else "FAIL",
        "summary": f"289 层最大相对偏差 {worst:.2e}",
    })
    finite = expect[np.isfinite(expect)]
    checks.append({
        "name": f"{_TIER}: 值域与覆盖（报告制）",
        "status": "report",
        "summary": (
            f"{var} ∈ [{finite.min():.4f}, {finite.max():.4f}]，"
            f"全覆盖（源无缺测 → validity 全 1）"
        ),
    })
    return checks


def _semucb_fidelity(layer_id: str, var: str, tiers: list[str], store_dir) -> list[dict]:
    """SEMUCB：store 位级 == 评估输出 + viz 点级交叉验证 + 地壳 NaN 结构。"""
    checks: list[dict] = []
    ev = evaluate_semucb()
    expect = ev[var]
    with xr.open_datatree(store_dir, engine="zarr", chunks={}) as tree:
        node = tree[f"/{_TIER}/{SEMUCB_GROUP}"]
        if layer_id not in node.ds.data_vars:
            raise KeyError(f"store {store_dir} 不含 {layer_id}（先 build 再 fidelity）")
        arr = node.ds[layer_id]
        v = arr.values
        # NaN 感知位级比对（NaN != NaN；地壳内 NaN 须视为一致）
        eq = (v == expect) | (np.isnan(v) & np.isnan(expect))
        n_bad = int((~eq).sum())
        checks.append({
            "name": f"{_TIER}: 与官方工具评估输出逐位一致（float32 位级）",
            "status": "pass" if n_bad == 0 else "FAIL",
            "summary": f"不一致元素 {n_bad}（期望 0）",
        })
        dep = arr.coords["depth"].values.astype(np.float64)
        checks.append({
            "name": f"{_TIER}: 深度坐标 = 官方 74 层采样（30..2891 km）",
            "status": "pass" if np.array_equal(dep, ev["depth"]) else "FAIL",
            "summary": f"深度层数 {len(dep)}（期望 {SEMUCB_NDEPTH}）",
        })
    # 地壳 NaN 结构（a3d_dist 语义的契约判据）
    nan_counts = ev["nan_counts"]
    n_cells = 180 * 360
    nan_depths = tuple(ev["depth"][nan_counts > 0].tolist())
    checks.append({
        "name": "地壳 NaN 深度层集合 = {30, 40, 50}（moho∈[30,59.994] km 语义）",
        "status": "pass" if nan_depths == SEMUCB_CRUST_NAN_DEPTHS else "FAIL",
        "summary": f"含 NaN 深度层 {nan_depths}（各层 NaN 数 {nan_counts[nan_counts > 0].tolist()}）",
    })
    checks.append({
        "name": "深度 30 层全 NaN（全球 Moho ≥ 30 km ⇒ 地壳内）",
        "status": "pass" if int(nan_counts[0]) == n_cells else "FAIL",
        "summary": f"depth=30 NaN {int(nan_counts[0])}/{n_cells}",
    })
    # viz-only 点级交叉验证（确定性抽点：viz 网格步 6 取 1，全 74 深度）
    checks.extend(_viz_cross_validation(var))
    finite = expect[np.isfinite(expect)]
    checks.append({
        "name": f"{_TIER}: 值域与覆盖（报告制）",
        "status": "report",
        "summary": (
            f"{var} ∈ [{finite.min():.4f}, {finite.max():.4f}]；"
            f"有效 {finite.size}/{expect.size}（NaN = 平滑地壳层，官方工具语义）"
        ),
    })
    return checks


def _viz_cross_validation(var: str) -> list[dict]:
    """对 viz-only 产品做点级交叉验证（官方产品一致性判据）。

    确定性抽样：viz 网格纬经各步 6 取 1；比对集 = 工具输出有限值点
    （地壳内 viz 有值而工具 NaN——viz 绕过地壳检查，非比对失败）。
    """
    fname = SEMUCB_VIZ_VS if var == "vs" else SEMUCB_VIZ_XI
    with xr.open_dataset(SEMUCB_SRC_DIR / fname) as ds:
        viz = ds[list(ds.data_vars)[0]].values
        la = ds.latitude.values.astype(np.float64)[::6]
        lo = ds.longitude.values.astype(np.float64)[::6]
        viz_dep = ds.depth.values.astype(np.float64)
    depth = semucb_depths()
    if not np.array_equal(viz_dep, depth):
        raise ValueError(f"{fname}: 深度坐标与契约不符")
    radii = R_EARTH_SEMUCB - depth
    radii[-1] = SEMUCB_CMB_RADIUS   # 与 store 评估同约定（CMB 层 r=3480.5）
    df = _run_a3d(radii, la, lo, "viz_xval_pts.dat", f"viz_xval_{var}.out")
    fields = _eval_output_to_fields(df, depth, radii, la, lo)
    tool = fields[var]
    viz_sub = viz[:, ::6, ::6]
    finite = np.isfinite(tool)
    n_cmp = int(finite.sum())
    diff = np.abs(tool[finite] - viz_sub[finite])
    tol = VS_TOL if var == "vs" else XI_TOL
    n_bad = int((diff > tol).sum())
    return [
        {
            "name": (
                f"viz-only 交叉验证：{n_cmp} 点×{SEMUCB_NDEPTH} 深度 |Δ| ≤ {tol:g}"
                "（官方产品一致性，容差 = 文本输出量化）"
            ),
            "status": "pass" if n_bad == 0 else "FAIL",
            "summary": f"超差点 {n_bad}/{n_cmp}，max|Δ|={float(diff.max()) if n_cmp else float('nan'):.2e}",
        },
        {
            "name": "viz-only 分歧注记（报告制）",
            "status": "report",
            "summary": (
                f"viz 产品绕过地壳检查（深度 30/40/50 地壳区有值，工具 NaN）——"
                f"本通道遵循官方工具语义；比对集排除地壳内点（工具有限值点）"
            ),
        },
    ]


def build_seismic_fidelity_report(store_dir, layer_id: str, tiers: list[str],
                                  src_dir=None) -> dict:
    """保真报告（各层自供报告器）。

    src_dir 参数接受但忽略（源路径由常量解析；GLAD/SEMUCB 单源单文件，
    与热流双源不同）。GLAD 走位级核重算 + 积分守恒；SEMUCB 走位级评估
    输出比对 + viz-only 点级交叉验证 + 地壳 NaN 结构判据。
    """
    spec = SEISMIC_LAYERS[layer_id]
    if spec["kind"] == "glad":
        checks = _glad_fidelity(layer_id, spec["var"], tiers, store_dir)
    else:
        checks = _semucb_fidelity(layer_id, spec["var"], tiers, store_dir)
    checks.append({
        "name": "上地幔重叠段有意保留注记",
        "status": "report",
        "summary": _OVERLAP_NOTE,
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
