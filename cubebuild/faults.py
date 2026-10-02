"""断层双通道：GEM 活动断层 slip_type 类别层 + 球面大圆距离场。

源（磁盘实证，harmonized 主版本）：
- geopackage/gem_active_faults_harmonized.gpkg：13696 条 LineString（无
  MultiLineString），EPSG:4326，154994 顶点（平均 11.3 点/条）→ 有效段
  140955（零长段 343 剔除），总长 665632 km（球面，authalic R）；环边
  |Δlon|>180°（跨日期线）实证 0 条；slip_type 词汇表 23 类 + NaN 319 条
  （不参与类别栅格化，距离场仍计入）。
- 非 harmonized 版 gem_active_faults.gpkg（16195 行，CRS 未定义）为原始
  编汇版，不入立方。

距离场引擎（逐档从矢量源按球面大圆距离精确重算，与降采样核无关）：
- 每档像元值 = 该档像元中心到最近断层线要素的球面大圆距离（km，IUGG
  authalic R——与像元面积层同球）；不引用任何粗档数值，无降采样参与。
- 全网格对全部 ~1.4×10⁵ 段直接求 min 不可行（30″ 档 9.3×10⁸ 像元），
  采用层级环带候选剪枝（数学上保证不漏候选）：

  引理（候选环带）：胞 C 中心 c、半对角线 ρ（C 内任一点到 c 的球面距离
  上界），p ∈ C 的最近段 s_p 满足
      dist(c, s_p) ∈ [d(c) − 2ρ, d(c) + 2ρ]
  （两次三角不等式 + 距离函数 1-Lipschitz）。由此归纳：1° 档中心对全部
  段精确求 min 并收集环带 stored(1°)；每档子胞的候选集 = 父档 stored
  （归纳保证包含子胞像元的 argmin），对候选求 min 得该档精确值，再按
  自身环带过滤出 stored 传递给更细档。远场环带是半径 d(c)、宽 4ρ 的薄
  环——候选数不随档位细化膨胀，这是全网格精确计算可负担的关键。
- 值经引理保证无损；保真报告以独立实现的航空公式（半正矢 + 初始方位角）
  全段暴力抽样比对复核。
- 并行：按 1° 胞分解自包含任务（每胞自算 1° 暴力 min + 环带 + 逐档
  细化），fork 进程池；逐像元算术与任务序无关，构建确定性不受并行影响。

类别通道（复用栅格化机制）：线要素栅格化（GDAL 线默认走线算法，
all_touched=False，单像元宽迹线）→ 粗档 kernels.aggregate_mode 众数
聚合（nodata=0 不参与、平票取小、≥1 有效子胞即有值）；重叠像元取较短
断裂（局部细化优先，按「面积较小多边形」原则）；slip_type 缺失
断裂不烧录（0 = 无断层经过）。

许可：GEM GAF-DB 仓库 LICENSE.txt 为 CC BY-SA 4.0（SA 条款不符合
开放许可要求），两层与侧车不纳入公开版，论文建议读者直接使用 GEM GAF-DB。
"""

import json
import math
import multiprocessing
import os
import warnings
from fractions import Fraction
from pathlib import Path

import dask.array as da
import numpy as np
import xarray as xr

from .contract import LayerEntry
from .fidelity import INTEGRAL_TOL
from .grids import (
    TIER_ORDER,
    aggregation_factor,
    grid_shape,
    tier_centers,
    tier_chunks,
    tier_deg,
    tier_fraction,
)
from .kernels import aggregate_mode
from .pixel_area import EARTH_RADIUS_KM

REPO_ROOT = Path(__file__).resolve().parents[1]
GEM_SRC_DIR = REPO_ROOT / "original data/stress-kinematics/gem-active-faults"
GPKG_REL_PATH = "geopackage/gem_active_faults_harmonized.gpkg"

GEM_DOI = "10.1177/8755293020944182"   # Styron & Pagani 2020, Earthquake Spectra 36(1_suppl)
GEM_REPO_URL = "https://github.com/GEMScienceTools/gem-global-active-faults"

SLIP_CLASS_LAYER_ID = "stress_kinematics__gem_fault_slip_class"
DISTANCE_LAYER_ID = "stress_kinematics__gem_fault_distance"
HOME_TIER = "30sec"

EXPECTED_ROWS = 13696
EXPECTED_SLIP_NAN = 319
NODATA = 0

# slip_type 词汇表（磁盘实测 23 类，字典序固定编码 1..23；0 = 无类别）。
# NaN 319 条不占码位（不参与类别栅格化，距离场仍计入）。
SLIP_TYPE_CODES: dict[str, int] = {
    label: i for i, label in enumerate(sorted([
        "Anticline", "Blind Thrust", "Dextral", "Dextral-Normal",
        "Dextral-Oblique", "Dextral-Reverse", "Dextral_Transform", "Normal",
        "Normal-Dextral", "Normal-Sinistral", "Normal-Strike-Slip", "Reverse",
        "Reverse-Dextral", "Reverse-Sinistral", "Reverse-Strike-Slip",
        "Sinistral", "Sinistral-Normal", "Sinistral-Reverse",
        "Sinistral_Transform", "Spreading_Ridge", "Strike-Slip",
        "Subduction_Thrust", "Syncline",
    ]), start=1)
}

LICENSE_NOTE = (
    "GEM GAF-DB 许可 CC BY-SA 4.0（仓库 LICENSE.txt）——SA 条款不符合开放"
    "许可要求，不纳入公开版，论文建议读者直接使用 GEM GAF-DB"
    f"（{GEM_REPO_URL}）"
)


def category_encoding() -> dict[str, str]:
    """类别编码表（码 → 语义），attrs/manifest/sidecar 共用唯一来源。"""
    return {str(c): label for label, c in sorted(SLIP_TYPE_CODES.items(), key=lambda kv: kv[1])}


# ---------------- 源读取 ----------------

def load_gem_faults(src_dir: str | Path = GEM_SRC_DIR):
    """读 harmonized gpkg → (GeoDataFrame, slip_type 码数组)。

    几何契约（磁盘实证的契约化）：13696 行、EPSG:4326、全部 LineString、
    无空几何；slip_type 词汇 ⊆ 编码表（NaN 允许，319 条）；环边
    |Δlon|>180° 拒绝（跨日期线使平面栅格化静默错位；实证 0 条）。
    """
    import geopandas as gpd

    path = Path(src_dir) / GPKG_REL_PATH
    if not path.is_file():
        raise FileNotFoundError(f"GEM 活动断层源文件不存在: {path}")
    gdf = gpd.read_file(path)
    if len(gdf) != EXPECTED_ROWS:
        raise ValueError(f"{GPKG_REL_PATH}: 行数 {len(gdf)} ≠ 期望 {EXPECTED_ROWS}")
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        raise ValueError(f"{GPKG_REL_PATH}: CRS {gdf.crs} ≠ EPSG:4326")
    if gdf.geometry.isna().any() or gdf.geometry.is_empty.any():
        raise ValueError(f"{GPKG_REL_PATH}: 存在空几何")
    if not (gdf.geom_type == "LineString").all():
        raise ValueError(
            f"{GPKG_REL_PATH}: 存在非 LineString 几何 "
            f"{sorted(set(gdf.geom_type))}（MultiLineString 须先拆分）"
        )

    slips = gdf["slip_type"]
    unknown = sorted(set(slips.dropna()) - set(SLIP_TYPE_CODES))
    if unknown:
        raise ValueError(f"{GPKG_REL_PATH}: slip_type 超出编码表: {unknown}")
    n_nan = int(slips.isna().sum())
    if n_nan != EXPECTED_SLIP_NAN:
        raise ValueError(
            f"{GPKG_REL_PATH}: slip_type 缺失 {n_nan} 条 ≠ 期望 {EXPECTED_SLIP_NAN}"
            "（源变更须复评编码表与缺测语义）"
        )

    for geom in gdf.geometry:
        xy = np.asarray(geom.coords)
        jumps = np.abs(np.diff(xy[:, 0]))
        if (jumps > 180.0).any():
            k = int(np.argmax(jumps))
            raise ValueError(
                f"{GPKG_REL_PATH}: 断层存在跨日期线边 "
                f"{tuple(xy[k])}→{tuple(xy[k + 1])}，平面栅格化不安全"
            )
    codes = slips.map(lambda s: SLIP_TYPE_CODES.get(s, NODATA)).to_numpy(dtype=np.uint8)
    return gdf, codes


# ---------------- 球面几何：单位向量与段距离核 ----------------

def _unit_vec(lat_deg, lon_deg) -> np.ndarray:
    """(lat, lon) 度 → 3D 单位向量（..., 3）。日期线/极区无特例。"""
    phi = np.deg2rad(np.asarray(lat_deg, dtype=np.float64))
    lam = np.deg2rad(np.asarray(lon_deg, dtype=np.float64))
    cphi = np.cos(phi)
    return np.stack([cphi * np.cos(lam), cphi * np.sin(lam), np.sin(phi)], axis=-1)


def segment_table(geoms):
    """断层线要素 → 段表（引擎与保真共用唯一来源）。

    相邻顶点成段；零长段（连续重复顶点）剔除并计数。返回
    (A, B, NV, sinAB, cosAB, latlon)：
    - A/B (n,3) 段端点单位向量；NV = A×B（未归一化，模 = sin δ_AB）；
    - sinAB/cosAB (n,) 段角正弦/余弦；
    - latlon (4,n) 端点原坐标（度，独立航空公式核对路径用）。

    契约：段角 < 90°（cosAB > 0）——超长段意味着源形态异常，读取期拒绝。
    """
    va_list, vb_list = [], []
    n_dropped = 0
    for g in geoms:
        xy = np.asarray(g.coords)
        if len(xy) < 2:
            n_dropped += 1
            continue
        v = _unit_vec(xy[:, 1], xy[:, 0])
        a, b = v[:-1], v[1:]
        keep = np.linalg.norm(a - b, axis=1) > 1e-12
        n_dropped += int((~keep).sum())
        va_list.append(a[keep])
        vb_list.append(b[keep])
    A = np.concatenate(va_list, axis=0)
    B = np.concatenate(vb_list, axis=0)
    NV = np.cross(A, B)
    sinAB = np.linalg.norm(NV, axis=1)
    cosAB = np.einsum("ij,ij->i", A, B)
    if sinAB.min() <= 0.0 or cosAB.min() <= 0.0:
        k = int(np.argmin(cosAB))
        raise ValueError(f"段角 ≥ 90°（段 {k}，cos={cosAB[k]:.6f}）：源形态异常")
    latlon = np.stack([
        np.rad2deg(np.arcsin(np.clip(A[:, 2], -1.0, 1.0))),
        np.rad2deg(np.arctan2(A[:, 1], A[:, 0])),
        np.rad2deg(np.arcsin(np.clip(B[:, 2], -1.0, 1.0))),
        np.rad2deg(np.arctan2(B[:, 1], B[:, 0])),
    ])
    return (A, B, NV, sinAB, cosAB, latlon), n_dropped


def _point_seg_dists(p: np.ndarray, segs) -> np.ndarray:
    """点集 → 各段球面大圆距离 (m, n) km（引擎唯一距离核，纯函数）。

    3D 向量路径：sin_xt = p·(A×B)/|A×B|（叉积弦）；
    垂足在段内 ⟺ (p·A) ≥ cosAB·cos_xt 且 (p·B) ≥ cosAB·cos_xt
    （沿迹角 α ≤ δ_AB ⟺ cos α ≥ cos δ_AB 的余弦单调性变换）。
    垂足在段内 → R·|asin(sin_xt)|；否则 → R·arccos(max(p·A, p·B))
    （较近端点）。反侧远场（含对跖）由端点分支正确兜底。
    """
    A, B, NV, sinAB, cosAB = segs[0], segs[1], segs[2], segs[3], segs[4]
    sin_xt = (p @ NV.T) / sinAB
    np.clip(sin_xt, -1.0, 1.0, out=sin_xt)
    cos_xt = np.sqrt(1.0 - sin_xt * sin_xt)
    pa = p @ A.T
    pb = p @ B.T
    thr = cosAB * cos_xt
    interior = (pa >= thr) & (pb >= thr)
    pm = np.maximum(pa, pb)
    np.clip(pm, -1.0, 1.0, out=pm)
    d_end = EARTH_RADIUS_KM * np.arccos(pm)
    d_int = EARTH_RADIUS_KM * np.abs(np.arcsin(sin_xt))
    return np.where(interior, d_int, d_end)


def independent_min_distance_km(lat: float, lon: float, segs) -> float:
    """独立实现：经典航空公式（半正矢 + 初始方位角）全段暴力 min。

    与引擎（3D 向量叉积/点积路径）刻意异构，供保真交叉验证（验收：
    随机抽样像元与独立大圆距离计算比对）。逐段向量化、标量入参。
    """
    _, _, _, _, _, latlon = segs
    latA, lonA, latB, lonB = (np.deg2rad(a) for a in latlon)
    phi, lam = math.radians(lat), math.radians(lon)

    def _hav(dphi, dlam, p1, p2):
        return np.sin(dphi / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlam / 2.0) ** 2

    def _delta(p1, l1, p2, l2):
        return 2.0 * np.arcsin(np.sqrt(np.clip(_hav(p2 - p1, l2 - l1, p1, p2), 0.0, 1.0)))

    def _bearing(p1, l1, p2, l2):
        return np.arctan2(
            np.sin(l2 - l1) * np.cos(p2),
            np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(l2 - l1),
        )

    d12 = _delta(latA, lonA, latB, lonB)          # 段角
    d13 = _delta(latA, lonA, phi, lam)             # A→p
    t12 = _bearing(latA, lonA, latB, lonB)
    t13 = _bearing(latA, lonA, phi, lam)
    sin_xt = np.clip(np.sin(d13) * np.sin(t13 - t12), -1.0, 1.0)
    xt = np.arcsin(sin_xt)                         # 有符号叉积角
    d23 = _delta(latB, lonB, phi, lam)             # B→p
    # 垂足在段内 ⟺ A 侧与 B 侧沿迹折角均 ≤ 段角（双侧检验缺一不可：
    # 仅 A 侧会把越过 B 端的垂足误判为内点——开发期随机互证捕获）
    at = np.arccos(np.clip(np.cos(d13) / np.cos(xt), -1.0, 1.0))
    bt = np.arccos(np.clip(np.cos(d23) / np.cos(xt), -1.0, 1.0))
    interior = (at <= d12) & (bt <= d12)
    ang = np.where(interior, np.abs(xt), np.minimum(d13, d23))
    return float(EARTH_RADIUS_KM * ang.min())


# ---------------- 距离场引擎：层级环带候选剪枝 ----------------

# 细化链（引用 grids.TIER_ORDER 单一来源）：1° → 30′ → 6′ → 3′ → 30″
_TIER_CHAIN = TIER_ORDER

_KM_PER_DEG = 2.0 * math.pi * EARTH_RADIUS_KM / 360.0
# 环带半宽 W = 2ρ（ρ = 胞半对角线 km）+ 1 m 浮点安全余量。
# 引理保证：子档全体像元的 argmin 段落在父档中心距离环带 [d−W, d+W] 内。
_ANNULUS_KM = {
    t: 2.0 * tier_deg(t) * math.sqrt(0.5) * _KM_PER_DEG + 1e-3
    for t in _TIER_CHAIN
}

# fork 进程池共享段表（父进程 set 后 fork，COW 只读；完毕置 None）
_POOL_SEGMENTS = None


def _distance_cell(i: int, j: int, segs, chain) -> dict[str, np.ndarray]:
    """单 1° 胞全链任务：返回 {tier: (n, m) float64}（胞内各档像元精确值）。

    自包含（环带引理只引用本胞几何，无跨胞依赖）：
    1° 中心对全部段暴力 min → 环带 stored(1°)；逐档细化：子胞候选 =
    父胞 stored，求 min 得精确值，再按自身环带过滤出 stored 传给更细档；
    最细档只求 min 不过滤。parents/children 全程行主序 → 胞块落位确定。
    """
    A, B, NV, sinAB, cosAB = segs[0], segs[1], segs[2], segs[3], segs[4]
    res: dict[str, np.ndarray] = {}
    lat_lo, lon_lo = -90.0 + i, -180.0 + j

    c1 = _unit_vec(np.array([lat_lo + 0.5]), np.array([lon_lo + 0.5]))[0]
    d_all = _point_seg_dists(c1[None, :], segs)[0]
    d1 = float(d_all.min())
    res["1deg"] = np.array([[d1]])
    w1 = _ANNULUS_KM["1deg"]
    cand0 = np.nonzero((d_all >= d1 - w1) & (d_all <= d1 + w1))[0]
    if cand0.size == 0:
        raise RuntimeError(f"1° 胞 ({i},{j}) 环带为空——引理被破坏（实现缺陷）")

    # parents: (父胞行, 父胞列, 中心 lat, 中心 lon, 候选段索引)
    parents: list[tuple[int, int, float, float, np.ndarray]] = [
        (0, 0, lat_lo + 0.5, lon_lo + 0.5, cand0)
    ]
    for k in range(1, len(chain)):
        lvl, plvl = chain[k], chain[k - 1]
        f = int(tier_fraction(plvl) / tier_fraction(lvl))   # 父→子每边细化因子
        finest = k == len(chain) - 1
        w = _ANNULUS_KM[lvl]
        rp = tier_deg(plvl)
        sub_res = rp / f
        block = int(Fraction(1) / tier_fraction(lvl))   # 该档在 1° 胞内每边像元数
        out = np.empty((block, block), dtype=np.float64)
        children: list[tuple[int, int, float, float, np.ndarray]] = []
        for prow, pcol, plat, plon, cand in parents:
            sub = (A[cand], B[cand], NV[cand], sinAB[cand], cosAB[cand])
            offs = (np.arange(f) + 0.5) * sub_res
            lats = plat - rp / 2.0 + offs
            lons = plon - rp / 2.0 + offs
            la, lo = np.meshgrid(lats, lons, indexing="ij")
            flat_la, flat_lo = la.ravel(), lo.ravel()
            cc = _unit_vec(flat_la, flat_lo)            # (f², 3)
            dists = _point_seg_dists(cc, sub)            # (f², |cand|)
            d = dists.min(axis=1)
            out[prow * f:(prow + 1) * f, pcol * f:(pcol + 1) * f] = d.reshape(f, f)
            if not finest:
                keep = (dists >= d[:, None] - w) & (dists <= d[:, None] + w)
                flat_keep = np.nonzero(keep.ravel())[0]
                if flat_keep.size == 0:
                    raise RuntimeError(
                        f"胞 ({i},{j}) {lvl} 层环带为空——引理被破坏（实现缺陷）"
                    )
                n_cand = dists.shape[1]
                child_ids = flat_keep // n_cand
                cand_pos = flat_keep % n_cand
                bounds = np.searchsorted(child_ids, np.arange(f * f + 1))
                for t in range(f * f):
                    sel = cand_pos[bounds[t]:bounds[t + 1]]
                    if sel.size == 0:
                        raise RuntimeError(
                            f"胞 ({i},{j}) {lvl} 层子胞 {t} 环带为空——引理被破坏"
                        )
                    children.append((
                        prow * f + t // f, pcol * f + t % f,
                        float(flat_la[t]), float(flat_lo[t]),
                        cand[sel],
                    ))
        res[lvl] = out
        parents = children
    return res


def _pool_cell_task(cell):
    return _distance_cell(cell[0], cell[1], _POOL_SEGMENTS, _TIER_CHAIN)


def distance_grids(segs, chain=_TIER_CHAIN, workers: int = 1,
                   cache_dir=None) -> dict[str, np.ndarray]:
    """段表 → {tier: float32 全球网格}（chain 内各档精确距离值）。

    按 1° 胞分解自包含任务；workers > 1 时 fork 进程池并行（逐像元算术
    与任务序无关，确定性不受并行影响——复跑 checksum 一致性由验收双跑
    复核）。

    cache_dir（内存护栏）：给定时报各档网格写 .npy 后以只读
    mmap 返回——30″ 档 3.7GB 不驻留进程 RSS（文件页可回收），全量构建
    峰值预算（≤40GB）的关键减负件；None = 纯内存（测试短链）。
    """
    chain = tuple(chain)
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        out = {
            t: np.lib.format.open_memmap(
                cache_dir / f"{t}.npy", mode="w+", dtype=np.float32,
                shape=grid_shape(t),
            )
            for t in chain
        }
    else:
        out = {t: np.empty(grid_shape(t), dtype=np.float32) for t in chain}
    cells = [(i, j) for i in range(180) for j in range(360)]

    def fill(i: int, j: int, res: dict) -> None:
        for t in chain:
            f = int(Fraction(1) / tier_fraction(t))
            out[t][i * f:(i + 1) * f, j * f:(j + 1) * f] = res[t].astype(np.float32)

    try:
        if workers > 1:
            global _POOL_SEGMENTS
            _POOL_SEGMENTS = segs
            try:
                ctx = multiprocessing.get_context("fork")
                with ctx.Pool(workers) as pool:
                    for (i, j), res in zip(
                        cells, pool.imap(_pool_cell_task, cells, chunksize=16)
                    ):
                        fill(i, j, res)
            finally:
                _POOL_SEGMENTS = None
        else:
            for i, j in cells:
                fill(i, j, _distance_cell(i, j, segs, chain))
    except BaseException:
        for a in out.values():
            if isinstance(a, np.memmap):
                a._mmap.close()   # noqa: SLF001 —— 异常路径显式解除映射
        raise
    if cache_dir is not None:
        for t in chain:
            out[t].flush()
            out[t] = np.load(cache_dir / f"{t}.npy", mmap_mode="r")
    return out


# ---------------- 距离场：源级缓存与层构建 ----------------

# 距离金字塔磁盘缓存（内存护栏：30″ 档 3.7GB 走 .npy mmap，不驻留 RSS）
_PYRAMID_CACHE_DIR = REPO_ROOT / "products/.build/faults-distance"

_PYRAMID_CACHE: dict[str, dict[str, np.ndarray]] = {}


def _distance_pyramid(src_dir: str | Path = GEM_SRC_DIR) -> dict[str, np.ndarray]:
    """真实源全链金字塔（进程内缓存：五档只算一次，~1.4×10⁵ 段）。"""
    key = f"gem:{Path(src_dir).resolve()}"
    if key not in _PYRAMID_CACHE:
        gdf, _codes = load_gem_faults(src_dir)
        segs, _n_drop = segment_table(list(gdf.geometry))
        _PYRAMID_CACHE[key] = distance_grids(
            segs, _TIER_CHAIN, workers=max(1, os.cpu_count() or 1),
            cache_dir=_PYRAMID_CACHE_DIR,
        )
    return _PYRAMID_CACHE[key]


VALUE_CONVENTION = (
    "像元中心到最近断层线要素的球面大圆距离（km，IUGG authalic "
    "R=6371.0071810 km，与像元面积层同球）；逐档从矢量源精确重算（非跨档"
    "降采样）——候选剪枝用层级环带方案（父档中心距离环带 ±2×胞半对角线"
    "包含子档全体像元的 argmin，数学引理保证不漏候选），保真报告以独立"
    "航空公式抽样全段暴力比对复核"
)


def build_gem_fault_distance(entry: LayerEntry, tier: str) -> xr.DataArray:
    """构建该档 stress_kinematics__gem_fault_distance（float32，全域有值）。"""
    values = _distance_pyramid()[tier]
    lat, lon = tier_centers(tier)
    attrs = {
        "units": entry.unit,
        "long_name": "到最近活动断层（GEM GAF-DB）的球面大圆距离",
        "source": entry.source,
        "doi": GEM_DOI,
        "source_url": GEM_REPO_URL,
        "license_note": LICENSE_NOTE,
        "native_res": entry.native_res,
        "resampling": entry.resampling,
        "visibility": entry.visibility,
        "value_convention": VALUE_CONVENTION,
        "source_file": GPKG_REL_PATH,
        # 注意：不可命名 missing_value/_FillValue（zarr 后端按
        # CF 约定当 fill 编码，读回 0 被掩 NaN 且 dtype 升 float32）
        "nodata_semantics": "无缺测：距离场全域有值（免掩膜）",
    }
    arr = da.from_array(values, chunks=tier_chunks(tier))
    return xr.DataArray(
        arr, dims=("lat", "lon"), coords={"lat": lat, "lon": lon},
        name=entry.id, attrs=attrs,
    )


# ---------------- 类别通道：线栅格化 + 众数聚合 ----------------

def rasterize_fault_classes(geoms, codes: np.ndarray, tier: str) -> tuple[np.ndarray, int]:
    """断层线要素 → 该档类别网格（uint8，0 = 无断层经过）。

    语义：GDAL 线默认走线算法（all_touched=False）——线穿过的像元烧录，
    单像元宽迹线；同像元多断裂重叠取较短断裂（局部细化优先，
    「面积较小多边形」同款原则：长度降序稳定排序 + 后烧者胜）；
    codes=0（slip_type 缺失）不烧录、不参与重叠计数。

    返回 (classes (nlat, nlon) uint8, 重叠像元数)。逐断裂包围盒开窗
    烧录，全程确定性可复跑。
    """
    import rasterio.features as rfeatures
    from rasterio.transform import from_origin

    nlat, nlon = grid_shape(tier)
    res = tier_deg(tier)
    classes = np.zeros((nlat, nlon), dtype=np.uint8)
    covered = np.zeros((nlat, nlon), dtype=np.uint8)

    # 长度降序（稳定排序）：后烧者胜 → 较短断裂赢重叠像元。
    # 地理 CRS 下 shapely 的 length 单位为度——仅作相对排序，告警无谓。
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*geographic CRS.*")
        order = np.argsort([-g.length for g in geoms], kind="stable")

    for i in order:
        if codes[i] == NODATA:
            continue                       # slip_type 缺失：不参与类别栅格化
        g = geoms[i]
        minx, miny, maxx, maxy = g.bounds
        # 行号自南起（行 0 = −90 侧，档位网格约定）：
        # 行 r 跨 [−90+r·res, −90+(r+1)·res]；列 c 跨 [−180+c·res, ...]
        r0 = max(int(np.floor((miny + 90.0) / res)) - 1, 0)
        r1 = min(int(np.ceil((maxy + 90.0) / res)) + 1, nlat)    # exclusive
        c0 = max(int(np.floor((minx + 180.0) / res)) - 1, 0)
        c1 = min(int(np.ceil((maxx + 180.0) / res)) + 1, nlon)   # exclusive
        if r0 >= r1 or c0 >= c1:
            continue
        # 窗口北边 = 行 r1−1 的北缘；西边 = 列 c0 的西缘
        tr = from_origin(-180.0 + c0 * res, -90.0 + r1 * res, res, res)
        burned = rfeatures.rasterize(
            [(g, 1)], out_shape=(r1 - r0, c1 - c0), transform=tr, fill=0, dtype="uint8"
        ).astype(bool)
        # rasterio 窗口行序北起（local 0 = 行 r1−1），项目网格行 0 = −90
        # 侧——垂直翻转后落位（翻转缺陷由位级比对测试覆盖）
        burned = burned[::-1]
        win = classes[r0:r1, c0:c1]
        classes[r0:r1, c0:c1] = np.where(burned, codes[i], win)
        covered[r0:r1, c0:c1] += burned

    overlap_cells = int((covered > 1).sum())
    return classes, overlap_cells


_CLASS_CACHE: dict[str, tuple[np.ndarray, int]] = {}
_MODE_CACHE: dict[tuple[str, str], np.ndarray] = {}


def _home_tier_classes(src_dir: str | Path = GEM_SRC_DIR) -> tuple[np.ndarray, int]:
    """主档（30″）类别网格 + 重叠像元数（进程内缓存：五档只栅格化一次）。"""
    key = f"gem:{Path(src_dir).resolve()}"
    if key not in _CLASS_CACHE:
        gdf, codes = load_gem_faults(src_dir)
        _CLASS_CACHE[key] = rasterize_fault_classes(
            list(gdf.geometry), codes, HOME_TIER
        )
    return _CLASS_CACHE[key]


def _tier_classes(tier: str, src_dir: str | Path = GEM_SRC_DIR) -> np.ndarray:
    """该档类别网格：主档直出，粗档 = 主档众数聚合（缓存）。"""
    if tier == HOME_TIER:
        return _home_tier_classes(src_dir)[0]
    key = (tier, str(Path(src_dir).resolve()))
    if key not in _MODE_CACHE:
        home = _home_tier_classes(src_dir)[0]
        _MODE_CACHE[key] = aggregate_mode(
            home, aggregation_factor(tier, HOME_TIER), nodata=NODATA
        )
    return _MODE_CACHE[key]


RASTERIZATION_RULE = (
    "线要素栅格化 all_touched=False（GDAL 线默认走线算法：线穿过的像元"
    "烧录，单像元宽迹线）；同像元多断裂重叠取较短断裂（局部细化优先，"
    "面积较小多边形同款原则：长度降序稳定排序 + 后烧者胜）；"
    "slip_type 缺失断裂（319 条）不参与类别栅格化（0 = 无断层经过）；"
    "粗档 = 主档 30″ 众数聚合（kernels.aggregate_mode，nodata=0 不参与、"
    "平票取小、≥1 有效子胞即有值）"
)


def build_gem_fault_slip_class(entry: LayerEntry, tier: str) -> xr.DataArray:
    """构建该档 stress_kinematics__gem_fault_slip_class（uint8 类别层）。"""
    values = _tier_classes(tier)
    lat, lon = tier_centers(tier)
    attrs = {
        "units": entry.unit,
        "long_name": "活动断层滑动类型（GEM GAF-DB slip_type，像元内主导类别）",
        "source": entry.source,
        "doi": GEM_DOI,
        "source_url": GEM_REPO_URL,
        "license_note": LICENSE_NOTE,
        "native_res": entry.native_res,
        "resampling": entry.resampling,
        "visibility": entry.visibility,
        "category_encoding": json.dumps(
            category_encoding(), ensure_ascii=False, sort_keys=True
        ),
        "rasterization_rule": RASTERIZATION_RULE,
        "source_file": GPKG_REL_PATH,
        "nodata_semantics": "0 = 无断层经过（含 slip_type 缺失断裂不烧录），类别编码自 1 起",
    }
    arr = da.from_array(values, chunks=tier_chunks(tier))
    return xr.DataArray(
        arr, dims=("lat", "lon"), coords={"lat": lat, "lon": lon},
        name=entry.id, attrs=attrs,
    )


# ---------------- 保真验证（本层自供） ----------------

FAULT_FIDELITY_LAYERS = {
    SLIP_CLASS_LAYER_ID: {"kind": "class"},
    DISTANCE_LAYER_ID: {"kind": "distance"},
}

_SAMPLE_SEED = 20260912        # 抽样确定性（复跑同样本）
_N_SAMPLES = 120              # 每档抽样像元数（分层：均匀 + 近场）
_DIST_TOL_KM = 1e-3           # 基础容差 1 m（近场公式异构 ~1e-5 km 量级）
_DIST_REL_TOL = 3e-7          # 远场 float32 存储 ULP 主导：tol = max(1e-3, 3e-7·|d|)
                              # （float32 半 ULP ≈ 6e-8·|d|，20000 km 处 ≈ 1.2e-3 km）


def build_faults_fidelity_report(store_dir, layer_id: str, tiers: list[str],
                                  src_dir=None) -> dict:
    """保真报告。

    类别层判据（同构）：① store 与源重算逐位一致（uint8 位级）；
    ② 粗档类别众数一致性（= 主档 store 值众数聚合）；③ 值域 ⊆ {0} ∪
    编码表；④ 类别分布/重叠/未分类总量（报告制）。

    距离场判据（验收）：⑤ 每档分层随机抽样像元（均匀 + 近场 <50 km）
    与独立航空公式全段暴力比对，max|Δ| ≤ 1e-3 km（硬判据）；⑥ 全域有限
    且 ≥ 0（硬判据，免掩膜语义）；⑦ 距离分布统计（报告制）。
    """
    src_dir = src_dir or GEM_SRC_DIR
    kind = FAULT_FIDELITY_LAYERS[layer_id]["kind"]
    checks: list[dict] = []

    if kind == "class":
        home, overlap_cells = _home_tier_classes(src_dir)
        values: dict[str, np.ndarray] = {}
        with xr.open_datatree(store_dir, engine="zarr", chunks={}) as tree:
            for tier in tiers:
                node = tree[f"/{tier}"]
                if layer_id not in node.ds.data_vars:
                    raise KeyError(f"store {store_dir} 的 {tier} 档不含 {layer_id}（先 build 再 fidelity）")
                values[tier] = node.ds[layer_id].values

        # ① 逐位一致（硬判据）
        for tier in tiers:
            expect = _tier_classes(tier, src_dir)
            n_bad = int((values[tier] != expect).sum())
            checks.append({
                "name": f"{tier}: 与源栅格化重算逐位一致（uint8 位级）",
                "status": "pass" if n_bad == 0 else "FAIL",
                "summary": f"不一致像元 {n_bad}（期望 0）",
            })

        # ② 类别众数一致性（硬判据）：粗档 == 主档 store 值众数聚合
        for tier in tiers:
            if tier == HOME_TIER:
                continue
            expect = aggregate_mode(
                values[HOME_TIER], aggregation_factor(tier, HOME_TIER), nodata=NODATA
            )
            n_bad = int((values[tier] != expect).sum())
            checks.append({
                "name": f"{tier}: 类别众数一致性（= 主档 {HOME_TIER} store 值众数聚合）",
                "status": "pass" if n_bad == 0 else "FAIL",
                "summary": f"不一致像元 {n_bad}（期望 0）",
            })

        # ③ 值域（硬判据）+ ④ 类别分布（报告制）
        valid_codes = np.array(sorted(SLIP_TYPE_CODES.values()))
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
                sorted(SLIP_TYPE_CODES.items(), key=lambda kv: kv[1])
            )
            checks.append({
                "name": f"{tier}: 类别分布（报告制）",
                "status": "report",
                "summary": f"覆盖 {counts[1:].sum()}/{v.size}；{dist}",
            })

        _gdf, codes = load_gem_faults(src_dir)
        checks.append({
            "name": f"{HOME_TIER}: 重叠/未分类总量（报告制）",
            "status": "report",
            "summary": (
                f"多断裂重叠像元 {overlap_cells}（规则：较短断裂胜）；"
                f"slip_type 缺失断裂 {int((codes == 0).sum())} 条不参与栅格化"
                f"（距离场仍计入）"
            ),
        })

    else:  # distance
        gdf, _codes = load_gem_faults(src_dir)
        (segs, n_drop) = segment_table(list(gdf.geometry))
        rng = np.random.default_rng(_SAMPLE_SEED)
        with xr.open_datatree(store_dir, engine="zarr", chunks={}) as tree:
            for tier in tiers:
                node = tree[f"/{tier}"]
                if layer_id not in node.ds.data_vars:
                    raise KeyError(f"store {store_dir} 的 {tier} 档不含 {layer_id}（先 build 再 fidelity）")
                v = node.ds[layer_id].values
                lat, lon = tier_centers(tier)

                # ⑥ 全域有限且 ≥ 0（硬判据）
                n_bad = int((~np.isfinite(v)).sum() + (v < 0).sum())
                checks.append({
                    "name": f"{tier}: 全域有限且 ≥ 0（免掩膜语义）",
                    "status": "pass" if n_bad == 0 else "FAIL",
                    "summary": f"违法像元 {n_bad}（期望 0）",
                })

                # ⑤ 分层随机抽样 × 独立航空公式全段暴力比对（硬判据）
                n_flat = v.size
                n_near = _N_SAMPLES // 2
                near_idx = np.nonzero((v < 50.0).ravel())[0]
                picks = [int(rng.integers(0, n_flat)) for _ in range(_N_SAMPLES - n_near)]
                if near_idx.size >= n_near:
                    picks += [int(x) for x in near_idx[rng.integers(0, near_idx.size, n_near)]]
                else:
                    picks += [int(x) for x in near_idx] + [
                        int(rng.integers(0, n_flat)) for _ in range(n_near - near_idx.size)
                    ]
                max_abs = 0.0
                max_ratio = 0.0
                worst = None
                for flat in picks:
                    r, c = divmod(flat, v.shape[1])
                    indep = independent_min_distance_km(float(lat[r]), float(lon[c]), segs)
                    d = abs(indep - float(v[r, c]))
                    tol = max(_DIST_TOL_KM, _DIST_REL_TOL * abs(indep))
                    ratio = d / tol
                    if ratio > max_ratio:
                        max_ratio = ratio
                        worst = (float(lat[r]), float(lon[c]), indep, float(v[r, c]), d, tol)
                    max_abs = max(max_abs, d)
                ok = max_ratio <= 1.0
                checks.append({
                    "name": (
                        f"{tier}: 抽样 {len(picks)} 像元 vs 独立大圆（航空公式全段暴力）"
                        f"一致（tol = max(1e-3 km, 3e-7·d)）"
                    ),
                    "status": "pass" if ok else "FAIL",
                    "summary": (
                        f"max|Δ| = {max_abs:.3e} km，max |Δ|/tol = {max_ratio:.3f}"
                        + (f"；最差点 {worst[:2]}（独立 {worst[2]:.6f} vs store {worst[3]:.6f} km，"
                           f"Δ={worst[4]:.2e}/tol={worst[5]:.2e}）" if worst else "")
                    ),
                })

                # ⑦ 距离分布统计（报告制）
                checks.append({
                    "name": f"{tier}: 距离分布（报告制）",
                    "status": "report",
                    "summary": (
                        f"min {v.min():.3f} / p50 {np.percentile(v, 50):.2f} / "
                        f"mean {v.mean():.2f} / max {v.max():.2f} km；"
                        f"<1 km {float((v < 1).mean()):.4%}，<10 km {float((v < 10).mean()):.3%}，"
                        f"<100 km {float((v < 100).mean()):.3%}"
                    ),
                })

        checks.append({
            "name": "段表零长剔除（报告制）",
            "status": "report",
            "summary": f"连续重复顶点段 {n_drop} 条剔除（不参与距离场）",
        })

    n_fail = sum(1 for c in checks if c["status"] == "FAIL")
    return {
        "layer": layer_id,
        "store": str(store_dir),
        "integral_tolerance": INTEGRAL_TOL,   # 距离场无积分量；保留报告器统一字段
        "result": "FAIL" if n_fail else "PASS",
        "n_checks": len(checks),
        "checks": checks,
    }
