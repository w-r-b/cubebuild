"""重采样核库：按变量类型分派的纯函数核。

整数倍块聚合的面积加权保守平均（ETOPO 15s 走该路径）；
通用区间重叠保守核（conservative_overlap_mean）：支持分数倍档
（如 2min→3min 因子 1.5）、缺测权重归一、源网格极点半胞/极冠缺口——
单文件全球栅格源（WGM2012/EMAG2/GlobSed 等）统一走该路径。
 aggregate_sum（计数求和核）：点密度层主档 bincount 直出、
粗档块内求和（计数守恒构造性保证）。
 conservative_overlap_circular_axial（轴向 2θ 循环统计核）：
轴向角（0–180°，σ 与 σ+180 同义）经 φ=2θ 变换恢复整圆后做面积加权
循环统计，走向类变量（Slab2 strike）降采样走该路径。

约定：核为纯函数（源网格， 目标档位参数）→ 目标网格，不持有源特定几何。
"""

from fractions import Fraction
from math import lcm

import numpy as np

from .pixel_area import EARTH_RADIUS_KM


def aggregate_mode(data: np.ndarray, factor: int, nodata: int | None = None) -> np.ndarray:
    """众数聚合（类别变量核，纯函数）。

    粗像元 = factor×factor 块内计数的众数；平票取最小类值（确定性，
    scipy.stats.mode 同约定）。块内按类计数，类别升序遍历 + 严格大于
    比较保证平票时先到（最小）类胜出。

    nodata：类别编码中的缺测哨兵值（整型类别层 0 = 无类别，
    如矢量栅格化的未覆盖像元）。给定时不参与计数——粗像元 = 有效
    （≠nodata）子胞的众数，块内无任何有效子胞 → nodata。部分覆盖
    像元保留值的语义与保守核「W>0 即有值」一致。
    """
    m, n = data.shape
    if m % factor or n % factor:
        raise ValueError(f"shape {data.shape} 不是 factor {factor} 的整数倍")
    ot, on = m // factor, n // factor
    v = data.reshape(ot, factor, on, factor)
    if nodata is None:
        out = np.empty((ot, on), dtype=data.dtype)
    else:
        out = np.full((ot, on), nodata, dtype=data.dtype)
    best = np.zeros((ot, on), dtype=np.int64)
    for c in np.unique(data):          # 升序；严格 > → 平票保留最小类值
        if nodata is not None and c == nodata:
            continue
        cnt = (v == c).sum(axis=(1, 3))
        take = cnt > best
        out[take] = c
        np.maximum(best, cnt, out=best)
    return out


def aggregate_sum(data: np.ndarray, factor: int) -> np.ndarray:
    """计数求和聚合（点密度核，纯函数）。

    粗像元 = factor×factor 块内计数之和——计数守恒（Σ粗档 = Σ细档 =
    源记录数）由本核构造性保证。uint64 归约后回 uint32（单 3′ 胞计数
    远小于 2³²，回转纯防御）。
    """
    m, n = data.shape
    if m % factor or n % factor:
        raise ValueError(f"shape {data.shape} 不是 factor {factor} 的整数倍")
    ot, on = m // factor, n // factor
    v = data.reshape(ot, factor, on, factor)
    return v.sum(axis=(1, 3), dtype=np.uint64).astype(np.uint32)


def aggregate_weighted(data: np.ndarray, row_w: np.ndarray, factor: int) -> np.ndarray:
    """面积加权保守平均（连续变量核，纯函数，float64 计算）。

    data: (m, n)，m/n 均为 factor 整数倍；row_w: (m,) 各源行面积权重。
    粗像元值 = Σ_{i,j} w_i·v_ij / (factor·Σ_i w_i)（列等权、行按面积加权）。
    解析恒等式（行面积 telescoping，A_c = factor·Σ块内 w）：v_c·A_c = Σ_{fine} v·A
    → 积分量严格守恒（保真验证 <0.1% 判据可精确满足）。
    """
    m, n = data.shape
    if m % factor or n % factor:
        raise ValueError(f"shape {data.shape} 不是 factor {factor} 的整数倍")
    ot, on = m // factor, n // factor
    v = data.astype(np.float64, copy=False).reshape(ot, factor, on, factor)
    w = row_w.reshape(ot, factor)
    num = np.einsum("iajb,ia->ij", v, w)
    den = w.sum(axis=1)[:, None] * factor
    return num / den


# ---------------- 通用区间重叠保守核 ----------------

# 纬度行面积因子：A = R²·Δsin·Δλ_rad。列重叠以度计，此处折算 rad → W 单位 km²。
_ROW_AREA_FACTOR = EARTH_RADIUS_KM**2 * np.pi / 180.0


def _lcm_fraction(a: Fraction, b: Fraction) -> Fraction:
    """两个正有理数的最小公倍数（lcm(a/b, c/d) = lcm(ad, cb)/(bd)）。"""
    num = lcm(a.numerator * b.denominator, b.numerator * a.denominator)
    return Fraction(num, a.denominator * b.denominator)


def _column_pattern(lon_res: Fraction, tgt_res: Fraction,
                    lon_phase: Fraction = Fraction(0)) -> tuple[np.ndarray, int]:
    """经度周期块重叠模式：返回 (C, Q)。

    源胞中心在 lon_phase − 180 + k·lon_res（宽 lon_res，环接日期线）；目标
    胞边在 −180 + j·tgt_res。周期 L = lcm(lon_res, tgt_res)：每周期 P = L/tgt_res
    个目标列 ↔ Q = L/lon_res 个源列。C[u, v+1] = 目标列 u（周期内）与源列
    v ∈ [-1, Q+1] 的重叠长度（度）。分数倍档（如 1.5）由此精确处理。
    lon_phase：源网格经度相位（度，Fraction）——HFgrid14 类
    0.25 偏移中心（相位 = lon_res/2）即边缘对齐 -180 的像素注册网格；
    默认 0 = 中心在整倍数处（WGM 节点 Voronoi / EMAG2 卷绕后形态）。
    """
    if lon_res <= 0 or tgt_res <= 0:
        raise ValueError("分辨率必须为正")
    L = _lcm_fraction(lon_res, tgt_res)
    Q, P = int(L / lon_res), int(L / tgt_res)
    if Fraction(Q) * lon_res != L or Fraction(P) * tgt_res != L:
        raise ValueError(f"周期分解非整数: L={L}")
    C = np.zeros((P, Q + 3), dtype=np.float64)
    for u in range(P):
        t0, t1 = Fraction(u) * tgt_res, Fraction(u + 1) * tgt_res
        for v in range(-1, Q + 2):
            s0 = lon_phase + Fraction(v) * lon_res - lon_res / 2
            s1 = lon_phase + Fraction(v) * lon_res + lon_res / 2
            C[u, v + 1] = float(max(Fraction(0), min(t1, s1) - max(t0, s0)))
    if not np.allclose(C.sum(axis=1), float(tgt_res), atol=1e-12):
        raise ValueError("源经度胞未完整覆盖目标列（环接假设被破坏）")
    return C, Q


def _column_stage(X: np.ndarray, M: np.ndarray, C: np.ndarray, Q: int) -> tuple[np.ndarray, np.ndarray]:
    """列阶段：源行块 × 周期模式 → (加权值和, 加权有效和)，均为 (r, nlon_t)。

    X: (r, nlon_s) 源值（无效位已置 0）；M: (r, nlon_s) 0/1 有效位。
    环接：两侧各扩展 Q+1 列（源列索引 -1..-Q-1 ≡ 末尾回绕）。
    einsum 不优化（单线程 C 归约）——构建确定性可复跑的保证之一。
    """
    nlon = X.shape[1]
    P = C.shape[0]
    pad = Q + 1
    Xe = np.concatenate([X[:, -pad:], X, X[:, :pad]], axis=1)
    Me = np.concatenate([M[:, -pad:], M, M[:, :pad]], axis=1)
    A = np.empty((X.shape[0], nlon // Q * P), dtype=np.float64)
    B = np.empty_like(A)
    for p in range(nlon // Q):
        sl = slice(p * Q + Q, p * Q + 2 * Q + 3)
        A[:, p * P:(p + 1) * P] = np.einsum("rv,uv->ru", Xe[:, sl], C)
        B[:, p * P:(p + 1) * P] = np.einsum("rv,uv->ru", Me[:, sl], C)
    return A, B


def _row_windows(lat_edges: np.ndarray, tgt_res: Fraction) -> list[tuple[int, int, np.ndarray]]:
    """目标行的源行窗口与 sin 权重（conservative_overlap_mean 与
    conservative_overlap_circular_axial 共用；自前者无改动抽出——
    纯函数，输出逐位不变）。

    返回列表第 i 项 = (源行起, 源行止, 权重 (k,) km²)；目标行无任何源
    胞重叠时为 (-1, -2, None)。重叠区间的 sin 差 × R²·rad 折算。
    """
    nlat_t = int(Fraction(180) / tgt_res)
    last_row = len(lat_edges) - 2          # 源行末索引（edges = 行数 + 1）
    t_edges = -90.0 + np.arange(nlat_t + 1, dtype=np.float64) * float(tgt_res)
    win: list[tuple[int, int, np.ndarray]] = []
    for i in range(nlat_t):
        y0, y1 = t_edges[i], t_edges[i + 1]
        lo = max(int(np.searchsorted(lat_edges, y0, side="right")) - 1, 0)
        hi = min(int(np.searchsorted(lat_edges, y1, side="left")) - 1, last_row)
        rows, ws = [], []
        for s in range(lo, hi + 1):
            ov_lo, ov_hi = max(y0, lat_edges[s]), min(y1, lat_edges[s + 1])
            if ov_hi > ov_lo:
                rows.append(s)
                ws.append(
                    _ROW_AREA_FACTOR
                    * (np.sin(np.deg2rad(ov_hi)) - np.sin(np.deg2rad(ov_lo)))
                )
        win.append((rows[0], rows[-1], np.array(ws, dtype=np.float64)) if rows else (-1, -2, None))
    return win


def conservative_overlap_mean(
    data: np.ndarray,
    valid: np.ndarray,
    lat_edges: np.ndarray,
    lon_res: Fraction,
    tgt_res: Fraction,
    lon_phase: Fraction = Fraction(0),
) -> tuple[np.ndarray, np.ndarray]:
    """通用面积加权保守平均（连续变量核，纯函数，float64 计算）。

    适用源网格：全球规则经度环（源胞中心 lon_phase−180 + k·lon_res，
    nlon·lon_res=360；lon_phase 默认 0 = 中心在整倍数处，支持
    0.25 偏移类像素注册相位）+ 任意纬度边（可含极点半胞、极冠缺口、
    区域纬度带——Slab2 分区网格使用）；
    目标网格：全球档位网格（边对齐 -180/-90，中心偏移半档）。

    缺测语义：valid=False 的源像元不参与；目标值 = 有效源像元面积权重
    归一平均（部分覆盖像元无偏但代表性弱）；无有效源像元 → NaN。
    积分恒等式：Σ_t v_t·W_t = Σ_s v_s·w_s（s 限有效源胞，w_s = 源胞
    面积 km²；W_t = 目标胞内有效源面积权重 km²）——保真 <0.1% 判据
    可精确检验（W 由本核原样返回供保真复算）。

    返回 (values float32 (nlat_t, nlon_t), W float64 同 shape)。
    """
    data = np.asarray(data)
    valid = np.asarray(valid, dtype=bool)
    nlat_s, nlon_s = data.shape
    if valid.shape != data.shape or len(lat_edges) != nlat_s + 1:
        raise ValueError(f"形状不一致: data {data.shape}, valid {valid.shape}, edges {len(lat_edges)}")
    if Fraction(nlon_s) * lon_res != 360:
        raise ValueError(f"源经度胞未铺满整圆: {nlon_s}×{lon_res}")
    if np.any(np.diff(lat_edges) <= 0) or lat_edges[0] < -90.0 or lat_edges[-1] > 90.0:
        raise ValueError("纬度边须递增且在 [-90, 90] 内")

    nlat_t, nlon_t = int(Fraction(180) / tgt_res), int(Fraction(360) / tgt_res)
    C, Q = _column_pattern(lon_res, tgt_res, lon_phase)
    win = _row_windows(lat_edges, tgt_res)

    values = np.full((nlat_t, nlon_t), np.nan, dtype=np.float32)
    W = np.zeros((nlat_t, nlon_t), dtype=np.float64)
    vm = (valid & np.isfinite(data)).astype(np.float64)
    # 无效位（含 NaN）显式置 0：NaN×0=NaN 会污染 einsum 归约
    Xm = np.where(vm > 0, data.astype(np.float64, copy=False), 0.0)

    ROWS_PER_CHUNK = 256
    for i0 in range(0, nlat_t, ROWS_PER_CHUNK):
        i1 = min(i0 + ROWS_PER_CHUNK, nlat_t)
        rows = [r for r in range(i0, i1) if win[r][2] is not None]
        if not rows:
            continue
        s_lo = min(win[r][0] for r in rows)
        s_hi = max(win[r][1] for r in rows)
        A, B = _column_stage(Xm[s_lo:s_hi + 1], vm[s_lo:s_hi + 1], C, Q)
        for r in rows:
            s0, s1, w = win[r]
            num = np.einsum("k,kj->j", w, A[s0 - s_lo:s1 - s_lo + 1])
            den = np.einsum("k,kj->j", w, B[s0 - s_lo:s1 - s_lo + 1])
            W[r] = den
            ok = den > 0.0
            values[r] = np.where(ok, num / np.where(ok, den, 1.0), np.nan).astype(np.float32)
    return values, W


def conservative_overlap_circular_axial(
    data: np.ndarray,
    valid: np.ndarray,
    lat_edges: np.ndarray,
    lon_res: Fraction,
    tgt_res: Fraction,
    lon_phase: Fraction = Fraction(0),
) -> tuple[np.ndarray, np.ndarray]:
    """轴向 2θ 循环统计保守核（几何与 conservative_overlap_mean
    完全同构，仅值域换为轴向角循环统计）。

    轴向角 θ ∈ [0,180)：σ 与 σ+180° 同义（走向线无方向性），不能直接
    在 360° 圆上平均。φ = 2θ ∈ [0,360) 恢复整圆后做面积加权循环统计：

        C = Σ_s w_s·cos(2θ_s)，S = Σ_s w_s·sin(2θ_s)
        θ̄ = ½·atan2(S, C) mod 180

    w_s = 有效源胞面积权重 km²（与保守核同权）。输出角取 [0,180)；
    合成矢量对消（C=S=0，如 θ 与 θ+90 等权）时 atan2(0,0)=0 → θ̄=0
    （确定性约定，实测仅在正交双峰人工构造下出现）。无有效源像元 → NaN。

    轴向性质（由构造保证）：θ 输入换 θ+180 输出不变；跨 0/180 边界
    的均值正确（170/175/5/10 → ~0，朴素算术平均得 90 为其反面教材）。

    返回 (values float32 (nlat_t, nlon_t), W float64 同 shape)。
    """
    data = np.asarray(data)
    valid = np.asarray(valid, dtype=bool)
    nlat_s, nlon_s = data.shape
    if valid.shape != data.shape or len(lat_edges) != nlat_s + 1:
        raise ValueError(f"形状不一致: data {data.shape}, valid {valid.shape}, edges {len(lat_edges)}")
    if Fraction(nlon_s) * lon_res != 360:
        raise ValueError(f"源经度胞未铺满整圆: {nlon_s}×{lon_res}")
    if np.any(np.diff(lat_edges) <= 0) or lat_edges[0] < -90.0 or lat_edges[-1] > 90.0:
        raise ValueError("纬度边须递增且在 [-90, 90] 内")

    nlat_t, nlon_t = int(Fraction(180) / tgt_res), int(Fraction(360) / tgt_res)
    C, Q = _column_pattern(lon_res, tgt_res, lon_phase)
    win = _row_windows(lat_edges, tgt_res)

    values = np.full((nlat_t, nlon_t), np.nan, dtype=np.float32)
    W = np.zeros((nlat_t, nlon_t), dtype=np.float64)
    vm = (valid & np.isfinite(data)).astype(np.float64)
    # 2θ 通道（rad）：无效位显式置 0（cos0=1 但乘 M=0 归零，不污染归约）
    phi = np.deg2rad(2.0 * data.astype(np.float64, copy=False))
    cosm = np.where(vm > 0, np.cos(phi), 0.0)
    sinm = np.where(vm > 0, np.sin(phi), 0.0)

    ROWS_PER_CHUNK = 256
    for i0 in range(0, nlat_t, ROWS_PER_CHUNK):
        i1 = min(i0 + ROWS_PER_CHUNK, nlat_t)
        rows = [r for r in range(i0, i1) if win[r][2] is not None]
        if not rows:
            continue
        s_lo = min(win[r][0] for r in rows)
        s_hi = max(win[r][1] for r in rows)
        Ac, B = _column_stage(cosm[s_lo:s_hi + 1], vm[s_lo:s_hi + 1], C, Q)
        As, _ = _column_stage(sinm[s_lo:s_hi + 1], vm[s_lo:s_hi + 1], C, Q)
        for r in rows:
            s0, s1, w = win[r]
            num_c = np.einsum("k,kj->j", w, Ac[s0 - s_lo:s1 - s_lo + 1])
            num_s = np.einsum("k,kj->j", w, As[s0 - s_lo:s1 - s_lo + 1])
            den = np.einsum("k,kj->j", w, B[s0 - s_lo:s1 - s_lo + 1])
            W[r] = den
            ok = den > 0.0
            ang = 0.5 * np.degrees(np.arctan2(num_s, num_c)) % 180.0
            values[r] = np.where(ok, ang, np.nan).astype(np.float32)
    return values, W
