"""GSRM 应变场通道：v2.2 0.1° 全球应变网格 → 四层三档。

stress_kinematics__gsrm_exx / eyy / exy / vorticity（均 internal，主档 6min，
1°/30′/6′ 三档保守核降采样 + 有效性掩膜伴生）。

源契约（磁盘实证，v2.2/GSRM_strain.txt = GSRM_strain.txt.Z 解压版
554872103 B、6305376 行）：

- 结构：25 行头（Version 2.2（2015-12 版）；GEM Foundation
  CC-BY-NC-SA 3.0）+ 6300000 完整网格数据行 + 5351 行尾部离格块。
- 网格：pixel 注册 0.1°——lat -87.45..87.45（1750 行升序）× lon
  -179.95..179.95（3600 列），行主序（lat 外循环、lon 内循环），完整
  覆盖无重复（逐行与索引数学比对，偏差 ≤ 6e-14 = 1 ulp 量级）。中心
  0.05 相位＝源胞边对齐 -180/±87.5（保守核 lon_phase = 1/20，6′ 档
  恒等映射）。极冠 |lat|>87.5 源网格天然不含 → 各档极区行 NaN、有效
  性掩膜 0（覆盖缺口如实保留，不做外推）。全数值无缺测（11 列严格
  数值 token 扫描 0 非数值；无 NaN/999 类哨兵）。
- 列：lat long exx eyy exy vorticity RL-NLC LL-NLC e1 e2 azi_e1。本组收
  前四列；e1/e2/azi_e1 不收（特征分解是确定性代数运算，下游可由三分量
  自算）；RL-NLC/LL-NLC 源头与 README 均未给出语义定义，
  v1 映射未收录。
- 单位：README 自证 "GSRM_strain.txt.Z … units are 1e-9/yr"（文件头无
  单位行）→ exx/eyy/exy = nano-strain/yr、vorticity = nano-rad/yr
  （unit_basis 登记 manifest）。值域（磁盘实证）：exx -12142.4..11638.6 /
  eyy -12146.2..14150.9 / exy -17129.9..10840.5 / vorticity
  -895.326..1210.912。
- 尾部离格块（物理行 6300026..6305376，5351 行）：3 零行 + 1070 索引行
  （field1 = 1.000..1070.000、lon = 0.000、exx = -87.5 常量）各随 4 零行
  （末索引行后 2 零行）。索引行 801..1070 带自洽应变张量值（e1/e2/azi_e1
  与 exx/eyy/exy 特征分解吻合，已验算）但坐标全部离格（整数 lat 与
  0.000 距最近网格中心 0.05°）。v2.1 同名文件无此块（6300025 行整）且
  官方 v2.2 目录无 README/changelog——判定为 v2.2 构建残留，剔除不入
  立方；剔除量读取期契约化（总行数 / 网格完整性 / 尾部全离格三项断言，
  源变更即显式失败）。

可见性：internal（文件头 GEM Foundation CC-BY-NC-SA 3.0（NC+SA
双条款），不符合开放许可要求，不纳入公开版，论文建议读者自行联用）。
"""

from dataclasses import dataclass
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
from .points import GSRM_DOI, GSRM_SRC_DIR

# v2.2 解压版（数据已落盘）
GSRM_V22_DIR = GSRM_SRC_DIR / "v2.2"
STRAIN_FILE = "GSRM_strain.txt"

# 四层规约（层 id → 源列 + 分位数直方图几何 (lo, hi, bin_width) 覆盖各层
# 源值域 + 单位文案）——构建路由与保真分派共用（STRAIN_FIELD_OF 由此派生）
STRAIN_LAYERS = {
    "stress_kinematics__gsrm_exx": {
        "field": "exx", "hist_spec": (-12500.0, 12500.0, 1.0), "unit": "nano-strain/yr",
    },
    "stress_kinematics__gsrm_eyy": {
        "field": "eyy", "hist_spec": (-13000.0, 15000.0, 1.0), "unit": "nano-strain/yr",
    },
    "stress_kinematics__gsrm_exy": {
        "field": "exy", "hist_spec": (-17500.0, 12500.0, 1.0), "unit": "nano-strain/yr",
    },
    "stress_kinematics__gsrm_vorticity": {
        "field": "vorticity", "hist_spec": (-1000.0, 1500.0, 1.0), "unit": "nano-rad/yr",
    },
}

STRAIN_FIELD_OF = {lid: spec["field"] for lid, spec in STRAIN_LAYERS.items()}

# 源契约常量（磁盘实证的契约化；源变更即读取期显式失败）
HEADER_LINES = 25                        # 24 '#' 头 + 1 列名行
GRID_NLAT, GRID_NLON = 1750, 3600        # lat -87.45..87.45 × lon -179.95..179.95
GRID_ROWS = GRID_NLAT * GRID_NLON        # 6300000 完整网格行
TRAILING_ROWS = 5351                     # 尾部离格块（v2.2 构建残留，剔除）

SRC_TENTH = Fraction(1, 10)              # 0.1° 源分辨率（精确有理数）
LON_PHASE = Fraction(1, 20)              # 中心 0.05 相位＝胞边对齐 -180
LAT_SOUTH_K = 25                         # 源南缘 -87.5 = -90 + 25×0.1（边索引基）

# 坐标比对容差：文件值 2–3 位小数、离格坐标距中心 ≥ 0.05°，1e-6 仅吸收
# 文本解析与索引数学间的 1 ulp 差（实测 ≤ 6e-14）
_COORD_TOL = 1e-6

UNIT_BASIS = (
    "README 自证 'GSRM_strain.txt.Z - strain rate results at global 0.1deg "
    "grid. units are 1e-9/yr'（文件头无单位行）——exx/eyy/exy = nano-strain/yr"
    "（应变无量纲，1e-9/yr 直读）；vorticity = nano-rad/yr【推断：README 整文件"
    "1e-9/yr 概称未点名 rad，依据 = 涡度角量纲 × 1e-9/yr + 值域 ±1211 与"
    "变形带旋转率 nrad/yr 量级自洽（若为 1e-9 deg/yr 则小 4–5 个量级）】"
)

# 四层值域（磁盘实证；保真报告源契约判据）
VALUE_RANGES = {
    "exx": (-12142.4, 11638.6), "eyy": (-12146.2, 14150.9),
    "exy": (-17129.9, 10840.5), "vorticity": (-895.326, 1210.912),
}

_LICENSE_NOTE = (
    "GSRM 组不纳入公开版：文件头 GEM Foundation CC-BY-NC-SA 3.0（NC+SA "
    "条款不符合开放许可要求），论文建议读者自行联用"
)

_REGISTRATION_RULE = (
    "pixel 注册（0.1° 中心 -87.45..87.45 / -179.95..179.95，0.05 相位）＝"
    "源胞边对齐 -180/±87.5（保守核 lon_phase = 0.05，6′ 档恒等映射）；"
    "行主序 lat 升序外循环；尾部 5351 行离格块（v2.2 构建残留：3 零行 + "
    "1070 索引行各随 4 零行，坐标距最近网格中心 0.05°）剔除不入立方，"
    "剔除量读取期契约化；极冠 |lat|>87.5 源网格天然不含，极区行 NaN 保留"
)

_LONG_NAMES = {
    "exx": "东西向正应变率（e 轴向东；张量对角分量）",
    "eyy": "南北向正应变率（n 轴向北；张量对角分量）",
    "exy": "剪切应变率（张量非对角分量；工程剪应变率 = 2·exy——尾部块特征分解验算自洽）",
    "vorticity": "垂向涡度率（绕垂直轴旋转率，正 = 逆时针；刚性板块区依参考系显示）",
}


@dataclass
class StrainSource:
    """规范化应变源：fields 四分量 (1750, 3600) float32，行 0 = -87.5 侧。

    raster(field) 供保守核与源统计以 RasterSource 视图复用（几何共享）。
    """

    fields: dict[str, np.ndarray]
    lat_edges: np.ndarray               # (1751,) -87.5..87.5
    lon_res: Fraction
    lon_phase: Fraction
    n_trailing: int                     # 剔除的尾部离格行数（契约报告用）

    def raster(self, field: str) -> RasterSource:
        v = self.fields[field]
        return RasterSource(
            values=v,
            valid=np.isfinite(v),
            lat_edges=self.lat_edges,
            lon_res=self.lon_res,
            lon_phase=self.lon_phase,
        )


# ---------------- 源读取 ----------------

def load_gsrm_strain(src_dir: str | Path = GSRM_V22_DIR) -> StrainSource:
    """读 v2.2/GSRM_strain.txt → StrainSource（尾部离格块剔除）。

    契约（三项断言，源变更即显式失败）：
    ① 总数据行 = 6305351（25 头跳过）；
    ② 前 6300000 行 = 完整规则网格：行 i 坐标 == (lat[i//3600], lon[i%3600])
       索引数学（容差 1e-6，仅吸收 1 ulp）——完整性与行主序一并保证；
    ③ 尾部 5351 行全部离格（lat/lon 距最近网格中心 > 1e-6）——若源更新
       把有效网格点挪进尾部，此断言拦截静默丢点。
    """
    import pandas as pd

    path = Path(src_dir) / STRAIN_FILE
    if not path.is_file():
        raise FileNotFoundError(f"GSRM v2.2 应变源不存在: {path}（.Z 解压版）")
    # usecols 只取 lat/long + 四目标列（0..5）；网格几何常量调用期解析
    # （定义期绑定会使测试缩格注入失效）
    nlat, nlon = GRID_NLAT, GRID_NLON
    grid_rows = nlat * nlon
    trailing_rows = TRAILING_ROWS
    df = pd.read_csv(
        path, sep=r"\s+", skiprows=HEADER_LINES, header=None,
        usecols=[0, 1, 2, 3, 4, 5], dtype="float64", engine="c",
    )
    if df.shape != (grid_rows + trailing_rows, 6):
        raise ValueError(
            f"{STRAIN_FILE}: 数据形状 {df.shape} ≠ 期望 "
            f"({grid_rows + trailing_rows}, 6)（源结构漂移——须复核契约）"
        )

    la = df.iloc[:grid_rows, 0].to_numpy()
    lo = df.iloc[:grid_rows, 1].to_numpy()
    # ② 完整网格：行主序索引数学比对（完整 + 有序 + 无重复三位一体）
    r_idx = np.repeat(np.arange(nlat), nlon)
    c_idx = np.tile(np.arange(nlon), nlat)
    if np.abs(la - (-87.45 + 0.1 * r_idx)).max() > _COORD_TOL:
        raise ValueError(f"{STRAIN_FILE}: 网格行纬度偏离 -87.45+0.1k 行主序")
    if np.abs(lo - (-179.95 + 0.1 * c_idx)).max() > _COORD_TOL:
        raise ValueError(f"{STRAIN_FILE}: 网格行经度偏离 -179.95+0.1m 列循环")

    # ③ 尾部全离格（searchsorted 最近中心距离；离格 ≥ 0.05°）
    t = df.iloc[grid_rows:].to_numpy()
    if not np.isfinite(t).all():
        raise ValueError(f"{STRAIN_FILE}: 尾部块含非有限值（实证全数值）")
    exp_lat_c = -87.45 + 0.1 * np.arange(nlat)
    exp_lon_c = -179.95 + 0.1 * np.arange(nlon)
    r = np.clip(np.searchsorted(exp_lat_c, t[:, 0]), 1, nlat - 1)
    d_lat = np.minimum(np.abs(t[:, 0] - exp_lat_c[r - 1]), np.abs(t[:, 0] - exp_lat_c[r]))
    c = np.clip(np.searchsorted(exp_lon_c, t[:, 1]), 1, nlon - 1)
    d_lon = np.minimum(np.abs(t[:, 1] - exp_lon_c[c - 1]), np.abs(t[:, 1] - exp_lon_c[c]))
    n_ongrid = int((np.minimum(d_lat, d_lon) <= _COORD_TOL).sum())
    if n_ongrid:
        raise ValueError(
            f"{STRAIN_FILE}: 尾部块含 {n_ongrid} 行网格坐标（实证全离格——"
            "源更新把有效点挪入尾部，须复核剔除规则）"
        )

    values = df.iloc[:grid_rows, 2:6].to_numpy()
    if not np.isfinite(values).all():
        raise ValueError(f"{STRAIN_FILE}: 网格值含非有限数（实证全数值无缺测）")
    fields = {
        name: values[:, j].reshape(nlat, nlon).astype(np.float32)
        for j, name in enumerate(("exx", "eyy", "exy", "vorticity"))
    }
    # 源胞纬度边必须与档位网格同式计算（-90 + 0.1·k，k = 25..1775）：
    # 0.1 非二进制精确，-87.5+0.1·k 与 -90+0.1·(25+k) 两条路径在个别 k
    # 差 1 ulp——保守核行窗按边二分，1 ulp 错位使邻行泄入 ~1e-16 相对权，
    # 6′ 恒等破位级（实测 349 胞偏移 ~1e-10）。同式即逐位一致；
    # 30′/1° 共享边（k≡0 mod 5/10）经 fl(0.1·5p)=0.5p 精确性同样对齐。
    lat_edges = -90.0 + 0.1 * np.arange(LAT_SOUTH_K, LAT_SOUTH_K + nlat + 1, dtype=np.float64)
    return StrainSource(
        fields=fields,
        lat_edges=lat_edges,
        lon_res=SRC_TENTH,
        lon_phase=LON_PHASE,
        n_trailing=trailing_rows,
    )


_SOURCE_CACHE: dict[str, StrainSource] = {}


def _strain_source(src_dir: str | Path = GSRM_V22_DIR) -> StrainSource:
    """进程内缓存（~4 s 解析，四层 × 三档只读一次）。"""
    key = f"strain:{Path(src_dir).resolve()}"
    if key not in _SOURCE_CACHE:
        _SOURCE_CACHE[key] = load_gsrm_strain(src_dir)
    return _SOURCE_CACHE[key]


# ---------------- 层构建 ----------------

def build_gsrm_strain_layer(entry: LayerEntry, tier: str) -> xr.DataArray:
    """构建该档 GSRM 应变四层之一（exx/eyy/exy/vorticity，internal）。"""
    field = STRAIN_FIELD_OF[entry.id]
    src = _strain_source()
    values, _W = conservative_overlap_mean(
        src.fields[field], np.isfinite(src.fields[field]), src.lat_edges,
        src.lon_res, tier_fraction(tier), lon_phase=src.lon_phase,
    )
    arr = da.from_array(values, chunks=tier_chunks(tier))
    lat, lon = tier_centers(tier)
    attrs = {
        "units": entry.unit,
        "long_name": f"GSRM v2.2 应变率场 {field}——{_LONG_NAMES[field]}",
        "source": entry.source,
        "doi": GSRM_DOI,
        "native_res": entry.native_res,
        "resampling": entry.resampling,
        "visibility": entry.visibility,
        "earth_radius_km": EARTH_RADIUS_KM,
        "product_type": (
            "model（Haines-Holt 方法自 GPS 速度场插值的应变率模型网格，"
            "非观测；v2.2（2015-12 版）无 changelog，与 v2.1 不可混）"
        ),
        "unit_basis": UNIT_BASIS,
        "registration_rule": _REGISTRATION_RULE,
        "source_file": f"v2.2/{STRAIN_FILE}（GSRM_strain.txt.Z 解压版）",
        "license_note": _LICENSE_NOTE,
    }
    return xr.DataArray(
        arr, dims=("lat", "lon"), coords={"lat": lat, "lon": lon},
        name=entry.id, attrs=attrs,
    )


# ---------------- 保真验证（本层自供） ----------------


def build_gsrm_strain_fidelity_report(store_dir, layer_id: str, tiers: list[str]) -> dict:
    """保真报告（组同构判据 + 源契约判据）：

    ① store 与源核重算逐位一致（float32 位级，NaN 感知）；
    ② 有效位与源掩膜复算一致（极冠缺口两通道同源）；
    ③ 积分量守恒（< 0.1% 硬判据）+ 分位数偏移（报告制）；
    ④ 覆盖率（报告制：源极冠缺口的档位几何预期）；
    ⑤ 源契约（硬判据）：完整网格 6300000 行 + 尾部剔除 5351 行 + 值域
       四数（读取器已断言，此处以契约常量复报——漂移即 FAIL）。
    """
    spec = STRAIN_LAYERS[layer_id]
    src = _strain_source()
    ras = src.raster(spec["field"])
    hist_spec = spec["hist_spec"]
    src_stats = gravmag_source_stats(ras, hist_spec)

    checks: list[dict] = []
    with xr.open_datatree(store_dir, engine="zarr", chunks={}) as tree:
        for tier in tiers:
            node = tree[f"/{tier}"]
            if layer_id not in node.ds.data_vars:
                raise KeyError(f"store {store_dir} 的 {tier} 档不含 {layer_id}（先 build 再 fidelity）")
            v = node.ds[layer_id].values
            expect, W = conservative_overlap_mean(
                ras.values, ras.valid, ras.lat_edges, ras.lon_res,
                tier_fraction(tier), lon_phase=ras.lon_phase,
            )
            ok = W > 0

            # ① 逐位一致（NaN == NaN 视为一致）
            eq = (v == expect) | (np.isnan(v) & np.isnan(expect))
            n_bad = int((~eq).sum())
            checks.append({
                "name": f"{tier}: 与源核重算逐位一致（float32 位级）",
                "status": "pass" if n_bad == 0 else "FAIL",
                "summary": f"不一致像元 {n_bad}（期望 0）",
            })

            # ② 有效位一致性
            mism = int((np.isfinite(v) != ok).sum())
            checks.append({
                "name": f"{tier}: 有效位与源掩膜复算一致（逐位）",
                "status": "pass" if mism == 0 else "FAIL",
                "summary": f"不一致 {mism} 像元（期望 0）",
            })

            # ③ 积分 + 分位数（fidelity_checks）
            tier_stats = {
                "sum_va": float(np.dot(v[ok].astype(np.float64), W[ok])),
            }
            lo, hi, bw = hist_spec
            tier_stats["hist"] = hist_add(
                np.zeros(hist_bins(hist_spec), dtype=np.uint64),
                v[np.isfinite(v)], lo=lo, hi=hi, bin_width=bw,
            )
            checks.extend(fidelity_checks(src_stats, tier_stats, tier, hist_spec=hist_spec, unit=spec["unit"]))

            # ④ 覆盖率（报告制）：极冠缺口 = 源纬度带 [-87.5, 87.5] 的档位几何
            checks.append({
                "name": f"{tier}: 覆盖率（报告制，极冠缺口）",
                "status": "report",
                "summary": (
                    f"档 {ok.mean():.4f} vs 源（像元计）{ras.valid.mean():.4f}"
                    "（源网格 lat ±87.5 天然不含极冠，|lat|>87.5 档位行 NaN"
                    " 保留，有效性掩膜 0）"
                ),
            })

    # ⑤ 源契约（硬判据）：读取器断言的契约常量复报（非同义反复——
    # 对加载结果断言，几何/剔除量漂移即 FAIL）
    shape = src.fields["exx"].shape
    grid_ok = shape == (GRID_NLAT, GRID_NLON) and GRID_NLAT * GRID_NLON == GRID_ROWS == 6300000
    checks.append({
        "name": "源契约：完整网格 == 1750×3600（6300000 行）",
        "status": "pass" if grid_ok else "FAIL",
        "summary": f"加载形状 {shape}（读取器已断言行主序完整网格）",
    })
    checks.append({
        "name": "源契约：尾部离格块剔除 == 5351 行",
        "status": "pass" if src.n_trailing == TRAILING_ROWS == 5351 else "FAIL",
        "summary": (
            f"实际剔除 {src.n_trailing}（v2.2 构建残留：3 零行 + 1070 索引行"
            "各随 4 零行；坐标全离格已断言）"
        ),
    })
    f = src.fields[spec["field"]]
    lo_e, hi_e = VALUE_RANGES[spec["field"]]
    got = (float(f.min()), float(f.max()))
    range_ok = abs(got[0] - lo_e) <= 0.01 and abs(got[1] - hi_e) <= 0.01
    checks.append({
        "name": f"源契约：{spec['field']} 值域 == 磁盘实证四数",
        "status": "pass" if range_ok else "FAIL",
        "summary": (
            f"实测 [{got[0]:g}, {got[1]:g}] vs 契约 [{lo_e:g}, {hi_e:g}]"
            "（容差 0.01 吸收 float32 舍入；源数值重算即 FAIL 提示复核）"
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
