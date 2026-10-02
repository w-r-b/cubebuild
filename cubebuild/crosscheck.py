"""物理交叉核对：四组独立产品的数值交叉——只报告不拦截。

四组（层映射表 validation_reference）：
1. 验证三元组：gemma_basement_depth + gst1_thickness ≈ bedrock_elevation
   （三层均取 store 1° 档；存储符号约定下恒等式 = 高程[m] ≈ 基底[km]×1000
   + 沉积厚[m]，GEMMA 负值向下）
2. GEMMA moho × CRUST1.0 moho（depthtomoho.xyz 1° 验证参考；CRUST1.0
   由此整体退出立方，仅本地核对）
3. GST1 × GlobSed v3 海洋段（GlobSed-v3.nc 5′ 验证参考，海洋段交叉）
4. EMAG2 海平面版 × 上延版谱一致性（UpCont；Meyer 2017 原文实证：
   "All continental grids … were upward continued to 4 km"，即上延版
   = 4 km 高度版——谱比拟合 Δz 与该名义值对照）

验证参考纪律（CONTEXT.md 术语）：参考数据仅本地核对，报告只引用统计量，
不发布其数据、不进立方；全部检查 status="report"，永不拦截。

源几何（磁盘实证）：
- CRUST1.0 depthtomoho.xyz：64800 行 = 180×360，首行 (lon, lat) =
  (-179.5, 89.5) 北起、经度内循环步 1°；值 km、海平面基准负值向下
  （地标实证：青藏 85.5E/32.5N = −65.85、安第斯 −50.06、太平洋
  深海平原 −11.13——与 GEMMA moho 同符号约定）；中心恰为立方 1° 档
  中心（±.5 集），翻转后免重采样直比。
- GlobSed-v3.nc：lon/lat 4321×2161 gridline 节点（-180..180/-90..90
  步 1/12°），纬度南起；z 沉积厚度 m，NaN 33%（陆地）；±180 列逐位
  相同（周期重复，弃 +180 列）；两极行行内常值（北极 3101.95 /
  南极 1873.60 m，极点单点）——节点→Voronoi 胞解释。
- EMAG2 UpCont：与海平面版同形 5399×10800、同包络（0..360 框架 +
  极冠缺口）、同 nodata——共用 gravmag.load_emag2_tif 归一化与几何断言。

内存纪律（WSL ≤40GB 硬约束）：EMAG2 两版全量 float32 ~233MB ×2 +
  512² 复谱窗逐窗计算，峰值 <2GB；其余组均为 1° 小网格。
"""

import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import xarray as xr

from .grids import tier_fraction
from .gravmag import EMAG_FILE, EMAG_SRC_DIR, load_emag2_tif
from .kernels import conservative_overlap_mean
from .pixel_area import EARTH_RADIUS_KM

REPO_ROOT = Path(__file__).resolve().parents[1]
CRUST1_SRC_DIR = REPO_ROOT / "original data/lithosphere/crust-models/crust1.0"
CRUST1_MOHO_FILE = "depthtomoho.xyz"
GLOBSED_SRC_DIR = REPO_ROOT / "original data/sediment/globsed"
GLOBSED_FILE = "GlobSed-v3.nc"
EMAG2_UPCONT_FILE = "EMAG2_V3_20170530_UpCont.tif"

# 读取期几何断言常量（磁盘实证的契约化；测试可 monkeypatch 缩格）
CRUST1_EXPECTED = {"n_rows": 64800, "first": (-179.5, 89.5), "value_range": (-80.0, -2.0)}
GLOBSED_EXPECTED = {"shape": (2161, 4321), "spacing": Fraction(1, 12)}

EMAG2_NOMINAL_UPCONT_KM = 4.0    # Meyer 2017 原文（refs PDF 自证）
EMAG2_WINDOW = 512                # 谱窗边长（px，~8.5°×8.5° @2′；磁盘实证
                                   # 2048/1024 窗双版全有效数为 0，512 窗 41 个）
EMAG2_N_WINDOWS = 12              # 汇入窗数（确定性扫描取前 N 个全有效窗）
EMAG2_KMIN_RAD_KM = 0.02          # 拟合下界（避开均值扣除/锥窗主导的超低波数）
EMAG2_KMAX_RAD_KM = 0.60          # 拟合上界（避开 Nyquist 邻域的数值噪声）

# 代表波长（km）处的谱比报告点
EMAG2_REPORT_WAVELENGTHS_KM = (100.0, 50.0, 25.0, 12.5)


# ---------------- 验证参考读取 ----------------

def load_crust1_moho(src_dir: str | Path = CRUST1_SRC_DIR) -> np.ndarray:
    """读 CRUST1.0 depthtomoho.xyz → (180, 360) km 负值向下，南起对齐立方 1° 中心。

    断言（磁盘实证的契约化）：64800 行；首行 (lon, lat) = (-179.5, 89.5)
    北起；行内纬度常值、行间步 -1°；行内经度 -179.5..179.5 步 1°；值域
    ⊂ [-80, -2] km（Moho 深度海平面基准负值向下；源漂移即显式失败）。
    """
    path = Path(src_dir) / CRUST1_MOHO_FILE
    if not path.is_file():
        raise FileNotFoundError(f"CRUST1.0 源文件不存在: {path}")
    rows = []
    with open(path, encoding="ascii") as f:
        for line in f:
            tok = line.split()
            if len(tok) != 3:
                raise ValueError(f"{CRUST1_MOHO_FILE}: 行 token 数 {len(tok)} ≠ 3")
            rows.append((float(tok[0]), float(tok[1]), float(tok[2])))
    if len(rows) != CRUST1_EXPECTED["n_rows"]:
        raise ValueError(
            f"{CRUST1_MOHO_FILE}: 行数 {len(rows)} ≠ 期望 {CRUST1_EXPECTED['n_rows']}"
        )

    arr = np.asarray(rows, dtype=np.float64)
    lon_g = arr[:, 0].reshape(180, 360)
    lat_g = arr[:, 1].reshape(180, 360)
    v = arr[:, 2].reshape(180, 360)

    exp_lon = -179.5 + np.arange(360)
    exp_lat = 89.5 - np.arange(180)
    if not (lon_g[0, 0], lat_g[0, 0]) == CRUST1_EXPECTED["first"]:
        raise ValueError(
            f"{CRUST1_MOHO_FILE}: 首行 ({lon_g[0, 0]}, {lat_g[0, 0]}) ≠ 期望 "
            f"{CRUST1_EXPECTED['first']}（北起/经度内循环假设被破坏）"
        )
    if not np.array_equal(lon_g, np.broadcast_to(exp_lon, lon_g.shape)):
        raise ValueError(f"{CRUST1_MOHO_FILE}: 经度坐标偏离规则网格（-179.5..179.5 步 1°）")
    if not np.array_equal(lat_g, np.broadcast_to(exp_lat[:, None], lat_g.shape)):
        raise ValueError(f"{CRUST1_MOHO_FILE}: 纬度坐标偏离规则网格（89.5..-89.5 步 -1°）")
    lo, hi = CRUST1_EXPECTED["value_range"]
    if not (v.min() >= lo and v.max() <= hi):
        raise ValueError(
            f"{CRUST1_MOHO_FILE}: 值域 [{v.min():.2f}, {v.max():.2f}] 超出 "
            f"[{lo}, {hi}] km（负值向下海平面基准假设被破坏）"
        )
    return v[::-1].astype(np.float32)            # 北起 → 南起（对齐 store 1°）


def load_globsed(src_dir: str | Path = GLOBSED_SRC_DIR):
    """读 GlobSed-v3.nc → RasterSource（gridline 节点 → Voronoi 胞，WGM2012 惯例）。

    断言（磁盘实证的契约化）：4321×2161 节点（步 1/12°）；纬度南起；
    ±180 列逐位相同（周期重复，弃 +180 列）；两极行行内常值（极点单点）；
    z 单位 m（海洋沉积厚度，陆地 NaN）。
    """
    import netCDF4

    from .gravmag import RasterSource

    path = Path(src_dir) / GLOBSED_FILE
    if not path.is_file():
        raise FileNotFoundError(f"GlobSed 源文件不存在: {path}")
    ds = netCDF4.Dataset(path)
    try:
        lon = np.asarray(ds.variables["lon"][:], dtype=np.float64)
        lat = np.asarray(ds.variables["lat"][:], dtype=np.float64)
        z = np.ma.filled(ds.variables["z"][:], np.nan).astype(np.float32)
    finally:
        ds.close()

    ny, nx = GLOBSED_EXPECTED["shape"]
    spacing = GLOBSED_EXPECTED["spacing"]
    if z.shape != (ny, nx) or lon.shape != (nx,) or lat.shape != (ny,):
        raise ValueError(f"{GLOBSED_FILE}: 形状 {z.shape}/{lon.shape}/{lat.shape} ≠ 期望 {(ny, nx)}")
    exp_lat = -90.0 + np.arange(ny) * float(spacing)
    exp_lon = -180.0 + np.arange(nx) * float(spacing)
    if np.max(np.abs(lat - exp_lat)) > 1e-9 or np.max(np.abs(lon - exp_lon)) > 1e-9:
        raise ValueError(f"{GLOBSED_FILE}: 节点坐标偏离规则网格（-90..90/-180..180 步 5′）")
    if not np.array_equal(z[:, 0], z[:, -1]):
        raise ValueError(f"{GLOBSED_FILE}: ±180 列不同值，周期重复假设被破坏")
    if np.unique(z[0]).size != 1 or np.unique(z[-1]).size != 1:
        raise ValueError(f"{GLOBSED_FILE}: 极点行行内非常值（极点单点假设被破坏）")

    values = z[:, :-1]                            # 弃 +180 重复列 → 4320
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
                "gridline 节点注册（节点 -180..180/-90..90 步 1/12°）→ 节点代表"
                " Voronoi 胞（中心 ±2.5′，经度环接；极点行钳制半权）；±180 周期"
                "重复列弃除；纬度南起免翻转"
            ),
            "vertical_datum": "不适用（厚度场，单位 m）",
            "source_file": GLOBSED_FILE,
        },
    )


# ---------------- 统计工具 ----------------

def _pair_stats(x: np.ndarray, y: np.ndarray, mask: np.ndarray | None = None) -> dict:
    """两场配对统计（NaN 感知）：diff = y − x，OLS y = slope·x + intercept。

    x = 立方层/观测，y = 参考/预测（调用方在 name/summary 里说明方向）。
    """
    if mask is None:
        mask = np.isfinite(x) & np.isfinite(y)
    else:
        mask = mask & np.isfinite(x) & np.isfinite(y)
    a = x[mask].astype(np.float64)
    b = y[mask].astype(np.float64)
    if a.size < 2:
        return {"n": int(a.size)}
    d = b - a
    sa, sb = a.std(), b.std()
    r = float(np.corrcoef(a, b)[0, 1]) if sa > 0 and sb > 0 else float("nan")
    slope = float(np.dot(a - a.mean(), b - b.mean()) / np.dot(a - a.mean(), a - a.mean()))
    return {
        "n": int(a.size),
        "pearson_r": r,
        "mean_diff": float(d.mean()),
        "median_diff": float(np.median(d)),
        "std_diff": float(d.std()),
        "mae": float(np.abs(d).mean()),
        "rmse": float(np.sqrt((d * d).mean())),
        "p05_diff": float(np.percentile(d, 5)),
        "p95_diff": float(np.percentile(d, 95)),
        "ols_slope": slope,
        "ols_intercept": float(b.mean() - slope * a.mean()),
    }


def _fmt_pair(s: dict, unit: str) -> str:
    if "pearson_r" not in s:
        return f"n = {s['n']}"
    return (
        f"n={s['n']}, r={s['pearson_r']:.4f}, bias(y−x)={s['mean_diff']:+.2f} "
        f"(median {s['median_diff']:+.2f}) {unit}, MAE={s['mae']:.2f}, "
        f"RMSE={s['rmse']:.2f} {unit}, OLS y={s['ols_slope']:.4f}x"
        f"{s['ols_intercept']:+.2f}"
    )


# ---------------- 组 1：验证三元组 ----------------

TRIPLE_POLAR_LAT = 60.0   # 极区分段界（|lat| ≥ 60°：南极/格陵兰冰盖带）


def crosscheck_triple(tree) -> list[dict]:
    """gemma_basement_depth + gst1_thickness ≈ bedrock_elevation（store 1° 三层）。

    存储符号约定下的恒等式：高程[m] ≈ 基底深度[km]×1000 + 沉积厚度[m]
    （GEMMA 负值向下：基底 −2 km + 沉积 1.5 km → 地表 −0.5 km）。
    残差 = 高程 −（基底×1000 + 沉积厚），全球/陆/海/极区/非极区五段统计
    （报告制）+ 已知偏差注记（极区冰下重复计数签名 + 频段错配下限，
    残差分解实证：全球 RMS ~1.0 km 中约 3/4 方差来自极区）。
    """
    node = tree["/1deg"]
    elev = node["topography__bedrock_elevation"].values
    base = node["lithosphere__gemma_basement_depth"].values
    sed = node["sediment__gst1_thickness"].values
    land = node["derived__landsea_mask"].values == 1
    lat = node["topography__bedrock_elevation"].coords["lat"].values

    pred = base.astype(np.float64) * 1000.0 + sed.astype(np.float64)
    obs = elev.astype(np.float64)
    polar = np.broadcast_to(
        (np.abs(lat) >= TRIPLE_POLAR_LAT)[:, None], land.shape
    )
    segs = {
        "全球": None,
        "陆（landsea=1）": land,
        "海（landsea=0）": ~land,
        f"极区（|lat|≥{TRIPLE_POLAR_LAT:g}°，冰盖带）": polar,
        f"非极区（|lat|<{TRIPLE_POLAR_LAT:g}°）": ~polar,
    }
    checks = [{
        "name": "三元组恒等式与符号约定（组说明）",
        "status": "report",
        "summary": (
            "bedrock_elevation[m] ≈ gemma_basement_depth[km]×1000 + "
            "gst1_thickness[m]；GEMMA 负值向下（−2 km 基底 + 1.5 km 沉积 → "
            "地表 −0.5 km）；残差 = 高程 − 预测（x=预测, y=高程）"
        ),
    }]
    for label, m in segs.items():
        s = _pair_stats(pred, obs, m)
        checks.append({
            "name": f"三元组 {label}（x=基底+沉积预测, y=高程）",
            "status": "report",
            "summary": _fmt_pair(s, "m") + (
                f"；残差 p05/p95 = {s['p05_diff']:+.1f}/{s['p95_diff']:+.1f} m"
                if "p05_diff" in s else ""
            ),
            "stats": s,
        })

    # 极区重复计数签名：resid 对 sed 的回归（残差 ≈ −sed ⟹ 斜率趋 −1，
    # 即 GEMMA 基底已隐含冰下沉积柱、恒等式重复计数）
    sig = {}
    sp, rp = sed[polar].astype(np.float64), (obs - pred)[polar]
    if sp.size > 10 and sp.std() > 0:
        sig["resid_on_sed_slope"] = float(
            np.dot(sp - sp.mean(), rp - rp.mean())
            / np.dot(sp - sp.mean(), sp - sp.mean())
        )
        sig["resid_on_sed_corr"] = float(np.corrcoef(sp, rp)[0, 1])
        sig["n_polar"] = int(sp.size)

    sig_str = (
        f"极区 resid~sed 回归斜率 {sig['resid_on_sed_slope']:+.2f}、"
        f"corr {sig['resid_on_sed_corr']:+.2f}（n={sig['n_polar']}）"
        if sig else "极区样本不足，签名未算"
    )
    checks.append({
        "name": "已知偏差注记：极区冰下重复计数 + 频段错配下限（报告制）",
        "status": "report",
        "summary": (
            "全球 RMS 的主导成分是极区（|lat|≥60°）冰盖带，非立方缺陷："
            "① 重复计数——GOCE 卫星重力反演在无地震约束区（南极/格陵兰）"
            "把 GEMMA 基底放在冰床面附近（基底里已隐含冰下沉积柱），恒等式"
            "再叠加 GST1 冰下沉积（南极最高 ~13 km）即重复计数，残差幅值与"
            f"沉积厚度同量级；实测签名 {sig_str}，斜率趋 −1 方向即此机制；"
            "② 频段错配——0.5° 重力反演模型携带不了 30″ 地形的短波能量，"
            "非极区残差（数百米量级）由此下限主导。掩膜冰面口径（v1 重建"
            "冰盖归陆）后，「海」段已不含冰下盆地像元。"
            "剔极区后见「非极区」行。"
        ),
        "polar_signature": sig,
    })
    return checks


# ---------------- 组 2：GEMMA moho × CRUST1.0 ----------------

def crosscheck_gemma_vs_crust1(tree, src_dir: str | Path | None = CRUST1_SRC_DIR) -> list[dict]:
    """GEMMA moho（GOCE 反演）× CRUST1.0 moho（地震学编图）1° 逐像元交叉。

    两者同为海平面基准负值向下（km）；CRUST1.0 中心恰为立方 1° 中心，
    翻转后免重采样直比。x=CRUST1.0（参考），y=GEMMA（立方层）。
    src_dir=None 走缺省路径（integration 的 .get() 约定）。
    """
    crust1 = load_crust1_moho(src_dir or CRUST1_SRC_DIR)
    node = tree["/1deg"]
    gemma = node["lithosphere__gemma_moho"].values
    land = node["derived__landsea_mask"].values == 1

    checks = [{
        "name": "组说明（GEMMA × CRUST1.0）",
        "status": "report",
        "summary": (
            "CRUST1.0 depthtomoho（1° 地震学编图，验证参考）对 GEMMA moho"
            "（0.5° GOCE 反演，立方层 1° 档）；两者同符号约定（km，海平面基准"
            "负值向下）；x=CRUST1.0, y=GEMMA，diff = GEMMA − CRUST1.0"
        ),
    }]
    for label, m in {"全球": None, "陆": land, "海": ~land}.items():
        s = _pair_stats(crust1, gemma, m)
        checks.append({
            "name": f"GEMMA moho × CRUST1.0 {label}（x=CRUST1.0, y=GEMMA）",
            "status": "report",
            "summary": _fmt_pair(s, "km"),
            "stats": s,
        })
    return checks


# ---------------- 组 3：GST1 × GlobSed 海洋段 ----------------

def crosscheck_gst1_vs_globsed(tree, src_dir: str | Path | None = GLOBSED_SRC_DIR) -> list[dict]:
    """GST1 沉积厚度（立方唯一沉积层）× GlobSed v3 海洋段（5′ 验证参考）。

    GlobSed 经保守核聚合到 1°（WGM2012 类 gridline→Voronoi 解释），海洋段
    = GlobSed 有效像元（陆地 NaN）。x=GlobSed 聚合（参考），y=GST1（立方层）。
    src_dir=None 走缺省路径（integration 的 .get() 约定）。
    """
    src = load_globsed(src_dir or GLOBSED_SRC_DIR)
    agg, W = conservative_overlap_mean(
        src.values, src.valid, src.lat_edges, src.lon_res,
        tier_fraction("1deg"), lon_phase=src.lon_phase,
    )
    marine = W > 0
    gst1 = tree["/1deg"]["sediment__gst1_thickness"].values

    s = _pair_stats(agg, gst1, marine)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = gst1[marine].astype(np.float64) / agg[marine].astype(np.float64)
    ratio = ratio[np.isfinite(ratio)]          # GlobSed 聚合 0 m 像元不可比
    checks = [{
        "name": "组说明（GST1 × GlobSed 海洋段）",
        "status": "report",
        "summary": (
            "GlobSed v3（5′ 海洋沉积厚度，验证参考，gridline→Voronoi 保守核"
            "聚合 1°）对 GST1（0.125°，立方唯一沉积层 1° 档）海洋段 = GlobSed"
            f"有效像元；x=GlobSed, y=GST1，diff = GST1 − GlobSed"
        ),
    }, {
        "name": "GST1 × GlobSed 海洋段（x=GlobSed 1° 聚合, y=GST1）",
        "status": "report",
        "summary": (
            f"海洋段像元 {int(marine.sum())}/{marine.size}；"
            + _fmt_pair(s, "m")
            + f"；厚度比 GST1/GlobSed median={np.median(ratio):.3f}、"
            f"p05/p95={np.percentile(ratio, 5):.2f}/{np.percentile(ratio, 95):.2f}"
        ),
        "stats": s,
    }]
    return checks


# ---------------- 组 4：EMAG2 海平面版 × 上延版谱一致性 ----------------

def _k_grid(n: int, res_deg: float, lat_c: float) -> np.ndarray:
    """rfft2 输出各元的物理波数模（rad/km），经度向按窗心纬度 cos 缩放。

    kx = 2π·fx / ((π/180)·R·cosφ)，ky = 2π·fy / ((π/180)·R)
    （fx/fy cycles/deg；全球经纬网格谱分析的窗心纬度局部平面近似，
    报告制交叉核对的既定精度口径）。
    """
    fx = np.fft.fftfreq(n, d=res_deg)
    fy = np.fft.fftfreq(n, d=res_deg)
    deg_km = np.pi / 180.0 * EARTH_RADIUS_KM
    kx = 2.0 * np.pi * fx / (deg_km * np.cos(np.deg2rad(lat_c)))
    ky = 2.0 * np.pi * fy / deg_km
    return np.sqrt(kx[None, :] ** 2 + ky[:, None] ** 2)


def _radial_power(win: np.ndarray, res_deg: float, lat_c: float,
                  n_bins: int = 48) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """窗场 → 径向平均功率谱：(k_bin 中心, 平均功率, 每 bin 谱元数)。

    均值扣除 + 可分 Hann 锥窗（周期图法）；k 分箱线性于 [0, k_max]。
    """
    n = win.shape[0]
    taper = np.hanning(n)[:, None] * np.hanning(n)[None, :]
    f = np.fft.rfft2((win - win.mean()) * taper)
    power = f.real ** 2 + f.imag ** 2
    k = _k_grid(n, res_deg, lat_c)[:, : f.shape[1]]
    edges = np.linspace(0.0, k.max(), n_bins + 1)
    idx = np.clip(np.digitize(k.ravel(), edges) - 1, 0, n_bins - 1)
    count = np.bincount(idx, minlength=n_bins)
    sump = np.bincount(idx, weights=power.ravel().astype(np.float64), minlength=n_bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_p = np.where(count > 0, sump / np.maximum(count, 1), np.nan)
    return centers, mean_p, count


def _continuation_fit(k: np.ndarray, ratio: np.ndarray, count: np.ndarray,
                      kmin: float = EMAG2_KMIN_RAD_KM,
                      kmax: float = EMAG2_KMAX_RAD_KM) -> dict:
    """谱比 log(P_up/P_sea) = −2Δz·k 最小二乘拟合 → {delta_z_km, r2, n_bins,
    frac_ratio_gt1}。

    frac_ratio_gt1：拟合带内谱比 >1 的 bin 占比——纯上延恒有 P_up < P_sea
    （ratio < 1），显著 >0 说明该窗两版差异非纯上延（数据内容差异），
    供调用方做窗一致性甄别。
    """
    sel = (
        np.isfinite(ratio) & (ratio > 0) & (count > 0)
        & (k >= kmin) & (k <= kmax)
    )
    if sel.sum() < 3:
        return {
            "delta_z_km": float("nan"), "r2": float("nan"),
            "n_bins": int(sel.sum()), "frac_ratio_gt1": float("nan"),
        }
    kk, lr = k[sel], np.log(ratio[sel])
    A = np.vstack([kk, np.ones_like(kk)]).T
    coef, *_ = np.linalg.lstsq(A, lr, rcond=None)
    slope = float(coef[0])
    pred = A @ coef
    ss_res = float(((lr - pred) ** 2).sum())
    ss_tot = float(((lr - lr.mean()) ** 2).sum())
    return {
        "delta_z_km": -slope / 2.0,
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "n_bins": int(sel.sum()),
        "frac_ratio_gt1": float((ratio[sel] > 1.0).mean()),
    }


def _emag2_windows(valid_sea: np.ndarray, valid_up: np.ndarray,
                   window: int = EMAG2_WINDOW, n_windows: int = EMAG2_N_WINDOWS,
                   stride: int | None = None):
    """确定性选窗（两版均无缺测），返回 n_windows 个 (i0, j0)。

    地理偏性防护（code-review 实证：朴素行主序「取前 N」会 12 窗全落
    南半球）：非重叠步进（stride=window）扫描全部候选 → 按纬度排序后
    等距抽取 N 个，保证纬度覆盖均布（缺测结构允许范围内）。
    """
    nlat, nlon = valid_sea.shape
    stride = stride or window                  # 非重叠（独立窗，不低估 std）
    both = valid_sea & valid_up
    cands = []
    for i0 in range(0, nlat - window + 1, stride):
        for j0 in range(0, nlon - window + 1, stride):
            if both[i0:i0 + window, j0:j0 + window].all():
                cands.append((i0, j0))
    if len(cands) <= n_windows:
        return cands
    # 纬度排序（升序）→ 等距抽 N：确定性 + 南北均布
    cands.sort(key=lambda t: (t[0], t[1]))
    idx = np.linspace(0, len(cands) - 1, n_windows).round().astype(int)
    return [cands[int(i)] for i in sorted(set(idx.tolist()))]


def crosscheck_emag2_spectral(src_dir: str | Path | None = EMAG_SRC_DIR,
                               window: int = EMAG2_WINDOW,
                               n_windows: int = EMAG2_N_WINDOWS) -> list[dict]:
    """EMAG2 海平面版 × 上延版谱一致性：谱比 → 拟合上延高度 Δz（报告制）。

    上延算子谱响应 P_up/P_sea = exp(−2Δz·k)；对每窗拟合 Δz，与 Meyer 2017
    原文实证的名义 4 km 对照；另报代表波长处实测谱比与 exp(−2Δz·k) 期望。
    src_dir=None 走缺省路径（integration 的 .get() 约定）。
    """
    src_dir = src_dir or EMAG_SRC_DIR
    sea = load_emag2_tif(EMAG_FILE, src_dir)
    up = load_emag2_tif(EMAG2_UPCONT_FILE, src_dir)
    res_deg = float(sea.lon_res)

    lat_edges = sea.lat_edges
    wins = _emag2_windows(sea.valid, up.valid, window=window, n_windows=n_windows)
    if not wins:
        return [{
            "name": "EMAG2 谱一致性（无可用全有效窗）",
            "status": "report",
            "summary": f"扫描未找到 {window}×{window} 双版全有效窗（缺测结构异常？）",
        }]

    fits = []
    ratio_curves = []
    for i0, j0 in wins:
        lat_c = float(0.5 * (lat_edges[i0] + lat_edges[i0 + window]))
        lon_c = -180.0 + (j0 + window / 2.0) * res_deg
        k, p_sea, cnt = _radial_power(
            sea.values[i0:i0 + window, j0:j0 + window], res_deg, lat_c
        )
        _, p_up, _ = _radial_power(
            up.values[i0:i0 + window, j0:j0 + window], res_deg, lat_c
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = p_up / p_sea
        fit = _continuation_fit(k, ratio, cnt)
        # 窗一致性甄别：纯上延恒有 ratio < 1（P_up < P_sea）且 Δz > 0、
        # 谱比对 k 的负指数关系成立（r²）。偏离窗（两版数据内容差异超
        # 纯上延，如实测夏威夷窗 ratio>1）不进均值。
        fit["consistent"] = bool(
            np.isfinite(fit["delta_z_km"]) and fit["delta_z_km"] > 0
            and fit["r2"] >= 0.9 and fit["frac_ratio_gt1"] <= 0.1
        )
        fit["window"] = {"i0": i0, "j0": j0, "lat": round(lat_c, 2),
                         "lon": round(lon_c, 2)}
        fits.append(fit)
        ratio_curves.append((k, ratio, cnt))

    good = [f for f in fits if f["consistent"]]
    bad = [f for f in fits if not f["consistent"]]
    dz = np.array([f["delta_z_km"] for f in good], dtype=np.float64)
    dz_mean = float(dz.mean()) if dz.size else float("nan")
    dz_std = float(dz.std()) if dz.size else float("nan")

    # 代表波长处谱比（一致窗的谱比平均后内插）
    rep = []
    dz_for_expected = dz_mean if np.isfinite(dz_mean) else EMAG2_NOMINAL_UPCONT_KM
    good_idx = [i for i, f in enumerate(fits) if f["consistent"]]
    for lam in EMAG2_REPORT_WAVELENGTHS_KM:
        k_target = 2.0 * np.pi / lam
        rs = []
        for i in good_idx:
            k, ratio, cnt = ratio_curves[i]
            sel = np.isfinite(ratio) & (cnt > 0)
            if sel.sum() < 3:
                continue
            rs.append(float(np.interp(k_target, k[sel], ratio[sel])))
        if rs:
            measured = float(np.mean(rs))
            expected = float(np.exp(-2.0 * dz_for_expected * k_target))
            rep.append({
                "wavelength_km": lam,
                "wavenumber_rad_km": k_target,
                "measured_ratio": measured,
                "expected_ratio_at_fit_dz": expected,
            })

    bad_str = (
        "；偏离窗（ratio>1 段显著 / Δz≤0 / r²<0.9，两版数据内容差异超纯上延，"
        "不进均值）："
        + ", ".join(
            f"({f['window']['lat']}, {f['window']['lon']}) Δz={f['delta_z_km']:.2f}"
            f" r²={f['r2']:.2f} ratio>1 占比 {f['frac_ratio_gt1']:.0%}"
            for f in bad
        )
    ) if bad else ""

    return [{
        "name": "组说明（EMAG2 海平面版 × 上延版谱一致性）",
        "status": "report",
        "summary": (
            "两版同格 2′ 网格（5399×10800）逐窗径向功率谱比 "
            "P_UpCont/P_Sealevel = exp(−2Δz·k)（上延算子谱响应）；"
            f"纬度均布确定性选 {len(fits)} 个 {window}×{window} 双版全有效窗"
        ),
    }, {
        "name": "谱比拟合上延高度 Δz（一致窗均值 + 逐窗）",
        "status": "report",
        "summary": (
            f"{len(good)}/{len(fits)} 窗与纯上延模型一致，拟合 Δz = "
            + ", ".join(f"{f['delta_z_km']:.2f}" for f in good)
            + f" km；mean±std = {dz_mean:.2f}±{dz_std:.2f} km"
            f"（名义 {EMAG2_NOMINAL_UPCONT_KM:g} km，Meyer 2017 原文实证）"
            + bad_str
        ),
        "fits": fits,
        "mean_delta_z_km": dz_mean,
        "std_delta_z_km": dz_std,
        "nominal_delta_z_km": EMAG2_NOMINAL_UPCONT_KM,
        "n_consistent_windows": len(good),
        "n_deviant_windows": len(bad),
    }, {
        "name": "代表波长谱比（一致窗实测 vs exp(−2Δz·k) 期望）",
        "status": "report",
        "summary": "; ".join(
            f"λ={r['wavelength_km']:g} km: 实测 {r['measured_ratio']:.4f} vs "
            f"期望 {r['expected_ratio_at_fit_dz']:.4f}"
            for r in rep
        ) or "无有效代表波长点",
        "wavelength_ratios": rep,
    }]


# ---------------- 报告组装 ----------------

CROSSCHECK_GROUPS = (
    ("验证三元组", "crosscheck_triple", "tree"),
    ("GEMMA moho × CRUST1.0", "crosscheck_gemma_vs_crust1", "tree+crust1"),
    ("GST1 × GlobSed 海洋段", "crosscheck_gst1_vs_globsed", "tree+globsed"),
    ("EMAG2 海平面版 × 上延版谱一致性", "crosscheck_emag2_spectral", "emag2"),
)


def assemble_crosscheck_groups(store_dir, src_dirs: dict | None = None,
                                catch_errors: bool = True) -> list[dict]:
    """四组交叉的统一装配（build_crosscheck_report 与 integration 共用，
    消除双实现漂移）。

    src_dirs：{"crust1": …, "globsed": …, "emag2": …}，缺项走各源缺省路径。
    catch_errors=True 时组异常落为 FAIL 检查项（无报告可出 = 交付缺失，
    「只报告不拦截」指数值结果，不含执行失败）；False 时异常上抛
    （独立调用方自行处置）。
    """
    src_dirs = src_dirs or {}
    groups: list[dict] = []
    with xr.open_datatree(store_dir, engine="zarr", chunks={}) as tree:
        fns = {
            "crosscheck_triple": lambda: crosscheck_triple(tree),
            "crosscheck_gemma_vs_crust1": lambda: crosscheck_gemma_vs_crust1(
                tree, src_dirs.get("crust1")),
            "crosscheck_gst1_vs_globsed": lambda: crosscheck_gst1_vs_globsed(
                tree, src_dirs.get("globsed")),
            "crosscheck_emag2_spectral": lambda: crosscheck_emag2_spectral(
                src_dirs.get("emag2")),
        }
        for name, fn_name, _kind in CROSSCHECK_GROUPS:
            try:
                groups.append({"group": name, "checks": fns[fn_name]()})
            except Exception as e:  # noqa: BLE001 —— 组失败须显式报告
                if not catch_errors:
                    raise
                groups.append({
                    "group": name,
                    "checks": [{
                        "name": f"{name}: 交叉执行失败（交付缺失）",
                        "status": "FAIL",
                        "summary": f"{type(e).__name__}: {e}",
                    }],
                })
    return groups


def build_crosscheck_report(store_dir, crust1_src_dir=None, globsed_src_dir=None,
                            emag2_src_dir=None) -> dict:
    """四组物理交叉 → 报告 dict（数值结果全部 status=report，只报告不拦截）。"""
    src_dirs = {
        k: v for k, v in (
            ("crust1", crust1_src_dir), ("globsed", globsed_src_dir),
            ("emag2", emag2_src_dir),
        ) if v is not None
    }
    groups = assemble_crosscheck_groups(store_dir, src_dirs)
    return {
        "store": str(store_dir),
        "policy": "只报告不拦截（验证参考仅本地核对，报告只引用统计量）；"
        "组执行失败（无报告可出）计 FAIL",
        "n_groups": len(groups),
        "groups": groups,
    }


def write_crosscheck_report(report: dict, reports_dir: str | Path) -> Path:
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / "physical-crosscheck.json"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# 物理交叉核对报告（四组，只报告不拦截）",
        "",
        f"- store：`{report['store']}`",
        f"- 纪律：{report['policy']}",
        "",
        "| 组 | 检查项 | 状态 | 数值 |",
        "|---|---|---|---|",
    ]
    for g in report["groups"]:
        for c in g["checks"]:
            detail = (c.get("summary", "") or "").replace("|", "\\|")
            lines.append(f"| {g['group']} | {c['name']} | {c['status']} | {detail} |")
    (reports_dir / "physical-crosscheck.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return json_path
