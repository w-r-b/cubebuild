"""Resampling-kernel library: pure-function kernels dispatched by variable type.

Integer-factor block aggregation with area-weighted conservative mean (ETOPO 15s takes this path);
generic interval-overlap conservative kernel (conservative_overlap_mean): supports fractional factors
(e.g. 2min→3min factor 1.5), missing-value weight normalization, source-grid pole half-cells/polar-cap gaps —
single-file global raster sources (WGM2012/EMAG2/GlobSed etc.) all take this path.
 aggregate_sum (count-summation kernel): point-density layers' primary tier via direct bincount,
 coarse tiers by in-block summation (count conservation guaranteed by construction).
 conservative_overlap_circular_axial (axial 2θ circular-statistics kernel):
axial angles (0–180°, σ and σ+180 synonymous) are transformed by φ=2θ to recover the full circle, then
area-weighted circular statistics are applied; strike-type variables (Slab2 strike) downsampling takes this path.

Convention: kernels are pure functions (source grid, target tier parameters) → target grid, holding no source-specific geometry.
"""

from fractions import Fraction
from math import lcm

import numpy as np

from .pixel_area import EARTH_RADIUS_KM


def aggregate_mode(data: np.ndarray, factor: int, nodata: int | None = None) -> np.ndarray:
    """Mode aggregation (categorical-variable kernel, pure function).

    Coarse pixel = the mode of counts within a factor×factor block; ties take the smallest class value (deterministic,
    same convention as scipy.stats.mode). Counting by class within the block, ascending class iteration + strict-greater
    comparison ensures the first-arriving (smallest) class wins the tie.

    nodata: the missing-value sentinel in the class encoding (0 = no class for integer categorical layers,
    e.g. uncovered pixels from vector rasterization). When given it does not participate in counting — a coarse pixel =
    the mode of valid (≠nodata) subcells, and no valid subcell in the block → nodata. The semantics of retaining a value
    for a partially covered pixel match the conservative kernel's "W>0 means a value".
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
    for c in np.unique(data):
        if nodata is not None and c == nodata:
            continue
        cnt = (v == c).sum(axis=(1, 3))
        take = cnt > best
        out[take] = c
        np.maximum(best, cnt, out=best)
    return out


def aggregate_sum(data: np.ndarray, factor: int) -> np.ndarray:
    """Count-summation aggregation (point-density kernel, pure function).

    Coarse pixel = the sum of counts within a factor×factor block — count conservation (Σ coarse tier = Σ fine tier =
    source record count) is guaranteed by construction. Reduces in uint64 then casts back to uint32 (a single 3′ cell count
    is far below 2³²; the cast-back is purely defensive).
    """
    m, n = data.shape
    if m % factor or n % factor:
        raise ValueError(f"shape {data.shape} 不是 factor {factor} 的整数倍")
    ot, on = m // factor, n // factor
    v = data.reshape(ot, factor, on, factor)
    return v.sum(axis=(1, 3), dtype=np.uint64).astype(np.uint32)


def aggregate_weighted(data: np.ndarray, row_w: np.ndarray, factor: int) -> np.ndarray:
    """Area-weighted conservative mean (continuous-variable kernel, pure function, computed in float64).

    data: (m, n), both m/n are integer multiples of factor; row_w: (m,) per-source-row area weights.
    Coarse pixel value = Σ_{i,j} w_i·v_ij / (factor·Σ_i w_i) (equal weight across columns, area-weighted across rows).
    Analytical identity (row-area telescoping, A_c = factor·Σ w within the block): v_c·A_c = Σ_{fine} v·A
    → the integral quantity is strictly conserved (the fidelity <0.1% criterion can be met exactly).
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



_ROW_AREA_FACTOR = EARTH_RADIUS_KM**2 * np.pi / 180.0


def _lcm_fraction(a: Fraction, b: Fraction) -> Fraction:
    """Least common multiple of two positive rationals (lcm(a/b, c/d) = lcm(ad, cb)/(bd))."""
    num = lcm(a.numerator * b.denominator, b.numerator * a.denominator)
    return Fraction(num, a.denominator * b.denominator)


def _column_pattern(lon_res: Fraction, tgt_res: Fraction,
                    lon_phase: Fraction = Fraction(0)) -> tuple[np.ndarray, int]:
    """Longitude periodic-block overlap pattern: returns (C, Q).

    Source-cell centers at lon_phase − 180 + k·lon_res (width lon_res, wrapping around the dateline); target
    cell edges at −180 + j·tgt_res. Period L = lcm(lon_res, tgt_res): per period P = L/tgt_res
    target columns ↔ Q = L/lon_res source columns. C[u, v+1] = the overlap length (in degrees) of target column u (within the period) with source column
    v ∈ [-1, Q+1]. Fractional factors (e.g. 1.5) are handled exactly by this.
    lon_phase: source-grid longitude phase (degrees, Fraction) — the HFgrid14-type
    0.25-offset center (phase = lon_res/2) is the pixel-registration grid with edges aligned to -180;
    the default 0 = centers at integer multiples (the WGM node-Voronoi / EMAG2 post-wrapping form).
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
    """Column stage: source-row block × periodic pattern → (weighted value sum, weighted valid sum), both (r, nlon_t).

    X: (r, nlon_s) source values (invalid positions already zeroed); M: (r, nlon_s) 0/1 valid mask.
    Wrapping: pad Q+1 columns on each side (source column indices -1..-Q-1 ≡ wrap around at the end).
    einsum is not optimized (single-threaded C reduction) — one of the guarantees of reproducible, deterministic builds.
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
    """Source-row windows and sin weights for each target row (shared by conservative_overlap_mean and
    conservative_overlap_circular_axial; extracted from the former unchanged —
    pure function, output bit-identical).

    List item i = (source row start, source row end, weights (k,) km²); a target row with no source
    cell overlap is (-1, -2, None). The sin difference over the overlap interval × R²·rad conversion.
    """
    nlat_t = int(Fraction(180) / tgt_res)
    last_row = len(lat_edges) - 2
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
    """Generic area-weighted conservative mean (continuous-variable kernel, pure function, computed in float64).

    Applicable source grids: global regular longitude ring (source-cell centers lon_phase−180 + k·lon_res,
    nlon·lon_res=360; lon_phase default 0 = centers at integer multiples, supports a
    0.25-offset-type pixel-registration phase) + arbitrary latitude edges (may include pole half-cells, polar-cap gaps,
    regional latitude bands — used by Slab2 zone grids);
    target grid: global tier grid (edges aligned to -180/-90, center offset by half a tier).

    Missing-value semantics: source pixels with valid=False do not participate; the target value = the area-weight
    normalized mean of valid source pixels (a partially covered pixel is unbiased but weakly representative); no valid source pixel → NaN.
    Integral identity: Σ_t v_t·W_t = Σ_s v_s·w_s (s restricted to valid source cells, w_s = source-cell
    area km²; W_t = valid source area weight within the target cell km²) — the fidelity <0.1% criterion
    can be checked exactly (W is returned verbatim by this kernel for fidelity recomputation).

    Returns (values float32 (nlat_t, nlon_t), W float64 of the same shape).
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
    """Axial 2θ circular-statistics conservative kernel (geometry fully isomorphic to conservative_overlap_mean,
    with only the value domain changed to axial-angle circular statistics).

    Axial angle θ ∈ [0,180): σ and σ+180° are synonymous (a strike line has no directionality), so they cannot
    be averaged directly on the 360° circle. After φ = 2θ ∈ [0,360) restores the full circle, area-weighted circular statistics are applied:

        C = Σ_s w_s·cos(2θ_s), S = Σ_s w_s·sin(2θ_s)
        θ̄ = ½·atan2(S, C) mod 180

    w_s = valid source-cell area weight km² (same weighting as the conservative kernel). The output angle is in [0,180);
    when the resultant vector cancels (C=S=0, e.g. θ and θ+90 with equal weight), atan2(0,0)=0 → θ̄=0
    (deterministic convention, empirically observed only under artificially constructed orthogonal bimodality). No valid source pixel → NaN.

    Axial properties (guaranteed by construction): replacing θ input with θ+180 leaves the output unchanged; the mean across the
    0/180 boundary is correct (170/175/5/10 → ~0; the naive arithmetic mean yielding 90 is its counterexample).

    Returns (values float32 (nlat_t, nlon_t), W float64 of the same shape).
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
