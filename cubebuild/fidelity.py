"""Fidelity-verification harness (integral hard criterion + quantile reporting + seam check).

Layer generality (stated honestly):
- layer-agnostic (reusable): fidelity_checks / seam_check / histogram quantiles /
  tier_stats — any layer that provides a "source-stats dict + tier arrays" can hook in;
- ETOPO-specific: source_stats / build_fidelity_report (288-tile grid, 15° seam geometry).
  Subsequent layers each supply their own source_stats and extend the CLI dispatch;
  they cannot be reused simply by swapping --layer.
- derived-layer-specific: build_derived_fidelity_report — for slope/relief the "source"
  is the elevation layer inside the store, and fidelity = per-tier agreement with the
  elevation layer (full grid); the land-sea mask, since the v1 rebuild, uses the
  ice-surface convention, its source = the global ice-surface array (surface-first
  tile selection), and fidelity = bit-identical match with that array's recomputation
  chain; plus physical value-range hard criteria.
  The consistency check is strictly stronger than the seam check (a derived layer
  bit-equal to f(elevation) ⇒ no seam introduced by derivation), so seam_check is
  not run separately.

Criteria:
- continuous-layer integral: per-tier deviation from source < 0.1% (hard threshold,
  FAIL blocks);
- quantile shift: reporting only (conservative averaging must physically compress
  extremes, no single-point hard tolerance);
- seam check: difference distribution at tile boundaries vs an interior baseline
  (reporting only; an anomalous ratio flags it).

Histogram quantiles: equal-width 1m bins covering [-11000, 9000] m, CDF linearly
interpolated within a bin, sub-meter precision — sufficient for meter-level
elevation quantile reporting.
"""

import json
from pathlib import Path

import dask.array as da
import numpy as np
import xarray as xr

from .derived import (
    LAND_FRACTION_BOUNDS,
    NATIVE_TIER,
    landsea_threshold,
    mode_downsample,
    relief_core,
    slope_core,
)
from .etopo import (
    LAT_BANDS,
    LON_COLS,
    NODATA,
    SRC_RES_DEG,
    TILE_SHAPE,
    ice_surface_array,
    label_bounds,
    read_tile_validated,
    row_areas_km2,
    scan_tiles,
)
from .grids import grid_shape, tier_centers, tier_deg

INTEGRAL_TOL = 1e-3
HIST_LO, HIST_HI = -11000.0, 9000.0
HIST_BIN = 1.0
N_BINS = int((HIST_HI - HIST_LO) / HIST_BIN)

DERIVED_FIDELITY_LAYERS = frozenset(
    {"derived__landsea_mask", "derived__slope", "derived__relief"}
)
ELEVATION_LAYER_ID = "topography__bedrock_elevation"


def hist_add(
    hist: np.ndarray,
    values,
    lo: float = HIST_LO,
    hi: float = HIST_HI,
    bin_width: float = HIST_BIN,
) -> np.ndarray:
    """Accumulate finite values into a fixed-bins histogram (uint64). values may be an ndarray/DataArray.

    The bins geometry is given by (lo, hi, bin_width) and must match the hist length
    (to prevent silent clipping when the caller's geometry mismatches the preallocation)
    — it is available for non-elevation layers (mGal/nT etc.) to reuse with their own
    value ranges; the default arguments are the ETOPO elevation geometry.
    """
    v = np.asarray(values).ravel()
    v = v[np.isfinite(v) & (v != NODATA)]
    if hist.shape[0] != int(round((hi - lo) / bin_width)):
        raise ValueError(
            f"直方图几何不一致：hist 长度 {hist.shape[0]} ≠ (hi−lo)/bin_width"
        )
    idx = np.floor((v - lo) / bin_width).astype(np.int64)
    np.clip(idx, 0, hist.shape[0] - 1, out=idx)
    return hist + np.bincount(idx, minlength=hist.shape[0]).astype(np.uint64)


def quantiles_from_hist(
    hist: np.ndarray,
    qs: tuple[float, ...],
    lo: float = HIST_LO,
    bin_width: float = HIST_BIN,
) -> dict[str, float]:
    """Histogram → quantiles: CDF linearly interpolated within a bin under a uniform distribution (sub-bin precision)."""
    total = int(hist.sum())
    if total == 0:
        return {f"q{q:g}": float("nan") for q in qs}
    nz = np.nonzero(hist)[0]
    cum = np.cumsum(hist[nz])
    out = {}
    for q in qs:
        t = q / 100.0 * total
        i = int(np.searchsorted(cum, t, side="left"))
        if i >= len(nz):
            val = lo + (nz[-1] + 1.0) * bin_width
        else:
            lower = float(cum[i - 1]) if i > 0 else 0.0
            frac = (t - lower) / float(hist[nz[i]])
            val = lo + (nz[i] + frac) * bin_width
        out[f"q{q:g}"] = float(val)
    return out


def _tile_stats(path: Path, lat_top: float) -> tuple[float, np.ndarray, int]:
    """Single tile → (Σ v·A, histogram, pixel count). A is the 15s analytic row area (km²).

    Shares read_tile_validated with the build path (nodata fails explicitly), ensuring
    the integral statistics use the same convention as the build artifacts.
    """
    data = read_tile_validated(path)
    w = row_areas_km2(lat_top, TILE_SHAPE[0], SRC_RES_DEG)
    sum_va = float(np.dot(data.sum(axis=1).astype(np.float64), w))
    hist = hist_add(np.zeros(N_BINS, dtype=np.uint64), data)
    return sum_va, hist, data.size


def source_stats(src_dir) -> dict:
    """Source (15s global mosaic) statistics: Σ v·A + histogram + count."""
    tiles = scan_tiles(src_dir)
    sum_va, hist, count = 0.0, np.zeros(N_BINS, dtype=np.uint64), 0
    for label in (b + c for b in LAT_BANDS for c in LON_COLS):
        path = tiles[label]
        top, _ = label_bounds(label)
        s, h, n = _tile_stats(path, top)
        sum_va += s
        hist += h
        count += n
    return {"sum_va": sum_va, "hist": hist, "count": count}


def tier_stats(arr: xr.DataArray, tier: str) -> dict:
    """Tier-array statistics (computed chunk-wise, row areas via the tier's analytic formula).

    Note: w is generated descending from 90° north, while store row 0 is the southernmost
    row — the correct row-by-row pairing relies on the spherical row area being palindromic
    about the equator (sine symmetry); if you switch to ellipsoidal/asymmetric weights, you
    must pair explicitly by row coordinate.
    """
    nlat, nlon = grid_shape(tier)
    if arr.shape != (nlat, nlon):
        raise ValueError(f"{tier}: 形状 {arr.shape} ≠ 全球 {(nlat, nlon)}")
    w = row_areas_km2(90.0, nlat, tier_deg(tier))
    sum_va = 0.0
    hist = np.zeros(N_BINS, dtype=np.uint64)
    count = 0
    for i0, i1, j0, j1 in _chunks(arr):
        block = arr[i0:i1, j0:j1].compute().astype(np.float32)
        sum_va += float(np.dot(block.sum(axis=1).astype(np.float64), w[i0:i1]))
        hist = hist_add(hist, block)
        count += block.size
    return {"sum_va": sum_va, "hist": hist, "count": count}


def _chunks(arr: xr.DataArray, rows: int = 512, cols: int = 512):
    nlat, nlon = arr.shape
    for i0 in range(0, nlat, rows):
        for j0 in range(0, nlon, cols):
            yield i0, min(i0 + rows, nlat), j0, min(j0 + cols, nlon)


def fidelity_checks(
    src: dict,
    tier_s: dict,
    tier: str,
    hist_spec: tuple[float, float, float] = (HIST_LO, HIST_HI, HIST_BIN),
    unit: str = "m",
) -> list[dict]:
    """Single-tier fidelity checks (integral hard criterion + quantile reporting).

    hist_spec = (lo, hi, bin_width): the quantile-histogram geometry (available for
    non-elevation layers to reuse, default ETOPO elevation); unit only enters the report text.
    """
    rel = abs(tier_s["sum_va"] - src["sum_va"]) / abs(src["sum_va"])
    checks = [
        {
            "name": f"{tier}: 积分量守恒（相对源偏差 < 0.1%）",
            "status": "pass" if rel < INTEGRAL_TOL else "FAIL",
            "rel_dev": rel,
            "source_sum_va": src["sum_va"],
            "tier_sum_va": tier_s["sum_va"],
        }
    ]
    qs = (1, 5, 25, 50, 75, 95, 99)
    lo, _, bw = hist_spec
    sq = quantiles_from_hist(src["hist"], qs, lo=lo, bin_width=bw)
    tq = quantiles_from_hist(tier_s["hist"], qs, lo=lo, bin_width=bw)
    checks.append(
        {
            "name": f"{tier}: 分位数偏移（报告制）",
            "status": "report",
            "unit": unit,
            "shift_m": {k: tq[k] - sq[k] for k in sq},
            "source_q": sq,
            "tier_q": tq,
        }
    )
    return checks


def seam_check(arr: xr.DataArray, tier: str, tile_deg: float = 15.0) -> dict:
    """Seam check: 15° tile-boundary differences vs an interior baseline (reporting only).

    The P99.9 of adjacent-pixel differences at boundary columns/rows is compared against
    the P99.9 of differences at the same spacing in the interior; a ratio ~1 means the
    boundary and interior gradients share the same distribution (no anomalous-value band).
    """
    nlat, nlon = arr.shape
    step = int(round(tile_deg / tier_deg(tier)))
    rng = np.random.default_rng(0)

    def p999(d):
        return float(np.percentile(d, 99.9)) if d.size else float("nan")

    def sample_diff(pairs: list[tuple[int, int]], axis: int) -> np.ndarray:
        vals = []
        for a, b in pairs:
            idx = rng.integers(0, nlat if axis == 0 else nlon, size=200)
            if axis == 0:
                vals.append(np.abs(arr[idx, a] - arr[idx, b]).compute())
            else:
                vals.append(np.abs(arr[a, idx] - arr[b, idx]).compute())
        return np.concatenate(vals)

    bcols = [c for c in range(step, nlon, step)]
    icols = [c for c in range(1, nlon) if c % step != 0]
    seam_h = p999(sample_diff([(c - 1, c) for c in bcols], axis=0))
    base_h = p999(sample_diff(
        [(c - 1, c) for c in rng.choice(icols, size=len(bcols))], axis=0))
    brows = [r for r in range(step, nlat, step)]
    irows = [r for r in range(1, nlat) if r % step != 0]
    seam_v = p999(sample_diff([(r - 1, r) for r in brows], axis=1))
    base_v = p999(sample_diff(
        [(r - 1, r) for r in rng.choice(irows, size=len(brows))], axis=1))

    return {
        "name": f"{tier}: 接缝检查（边界差分 P99.9 / 内部基线，报告制）",
        "status": "report",
        "lon_seam_p999": seam_h,
        "lon_base_p999": base_h,
        "lon_ratio": seam_h / base_h if base_h else float("inf"),
        "lat_seam_p999": seam_v,
        "lat_base_p999": base_v,
        "lat_ratio": seam_v / base_v if base_v else float("inf"),
    }


def build_fidelity_report(store_dir, layer_id: str, tiers: list[str], src_dir) -> dict:
    """Fidelity report: source statistics × per-tier statistics × seam check. Raises SourceNotReady when the source is incomplete."""
    from .etopo import require_full_coverage

    require_full_coverage(src_dir)
    src = source_stats(src_dir)
    checks: list[dict] = []
    with xr.open_datatree(store_dir, engine="zarr", chunks={}) as tree:
        for tier in tiers:
            node = tree[f"/{tier}"]
            if layer_id not in node.ds.data_vars:
                raise KeyError(
                    f"store {store_dir} 的 {tier} 档不含 {layer_id}（先 build 再 fidelity）"
                )
            arr = node.ds[layer_id]
            checks.extend(fidelity_checks(src, tier_stats(arr, tier), tier))
            checks.append(seam_check(arr, tier))
    n_fail = sum(1 for c in checks if c["status"] == "FAIL")
    return {
        "layer": layer_id,
        "store": str(store_dir),
        "integral_tolerance": INTEGRAL_TOL,
        "result": "FAIL" if n_fail else "PASS",
        "n_checks": len(checks),
        "checks": checks,
    }


def _summary_check(checks: list, name: str, ok: bool, summary: str) -> None:
    checks.append(
        {"name": name, "status": "pass" if ok else "FAIL", "summary": summary}
    )


def build_derived_fidelity_report(store_dir, layer_id: str, tiers: list[str]) -> dict:
    """Derived-layer fidelity report: per-tier consistency recomputation + value-range hard criteria.

    Slope/relief: source = the elevation layer inside the store (manifest derived_from), and
    the full-grid recomputation must be bit-identical (same deterministic operator, same
    float32 input → bit-identity is the hard criterion); land-sea mask (ice-surface
    convention): source = the global ice-surface array (tiles read directly, not a store
    layer), likewise required to be bit-identical on recomputation. Plus physical value
    ranges (slope∈[0°,90°], relief≥0, mask∈{0,1}, land-fraction interval). Raises KeyError
    when the store lacks the elevation layer or the derived layer.
    """
    checks: list[dict] = []
    with xr.open_datatree(store_dir, engine="zarr", chunks={}) as tree:
        for tier in tiers:
            node = tree[f"/{tier}"]
            for var in (layer_id, ELEVATION_LAYER_ID):
                if var not in node.ds.data_vars:
                    raise KeyError(
                        f"store {store_dir} 的 {tier} 档不含 {var}（先 build 再 fidelity）"
                    )
            arr = node.ds[layer_id].data
            elev = node.ds[ELEVATION_LAYER_ID].data
            loc = f"{tier}"

            if layer_id == "derived__landsea_mask":
                surf30 = ice_surface_array(NATIVE_TIER)
                if tier == NATIVE_TIER:
                    expect = landsea_threshold(surf30)
                else:
                    factor = int(round(tier_deg(tier) / tier_deg(NATIVE_TIER)))
                    expect = mode_downsample(
                        landsea_threshold(surf30), factor
                    )
                n_bad, frac, ok01 = da.compute(
                    (arr != expect).sum(),
                    arr.mean(dtype="float64"),
                    ((arr == 0) | (arr == 1)).all(),
                )
                _summary_check(
                    checks,
                    f"{loc}: 掩膜语义随冰面高程（0m 阈值/众数聚合重算，逐位一致）",
                    int(n_bad) == 0,
                    f"不一致像元 {int(n_bad)}（期望 0）",
                )
                _summary_check(
                    checks,
                    f"{loc}: 值域 {{0,1}}",
                    bool(ok01),
                    f"陆地占比 {frac:.4f}",
                )
                lo, hi = LAND_FRACTION_BOUNDS
                _summary_check(
                    checks,
                    f"{loc}: 陆地占比 ∈ [{lo:g}, {hi:g}]（粗错捕捉器）",
                    lo <= frac <= hi,
                    f"实测 {frac:.4f}（掩膜反转~0.7+ / 整体错位~0 或 ~1 会被拦截）",
                )
            elif layer_id == "derived__slope":
                lat, _ = tier_centers(tier)
                expect = slope_core(elev, lat, tier_deg(tier)).astype("float32")
                n_bad, rmin, rmax, rmean = da.compute(
                    (arr != expect).sum(),
                    arr.min(),
                    arr.max(),
                    arr.mean(dtype="float64"),
                )
                _summary_check(
                    checks,
                    f"{loc}: 与该档高程重算逐位一致（中央差分梯度）",
                    int(n_bad) == 0,
                    f"不一致像元 {int(n_bad)}（期望 0）",
                )
                _summary_check(
                    checks,
                    f"{loc}: 值域 [0°, 90°]",
                    bool(0.0 <= rmin and rmax <= 90.0),
                    f"min {rmin:.4f}° / max {rmax:.2f}° / mean {rmean:.3f}°",
                )
            elif layer_id == "derived__relief":
                expect = relief_core(elev).astype("float32")
                n_bad, rmin, rmax, rmean = da.compute(
                    (arr != expect).sum(),
                    arr.min(),
                    arr.max(),
                    arr.mean(dtype="float64"),
                )
                _summary_check(
                    checks,
                    f"{loc}: 与该档高程重算逐位一致（3×3 窗口极差）",
                    int(n_bad) == 0,
                    f"不一致像元 {int(n_bad)}（期望 0）",
                )
                _summary_check(
                    checks,
                    f"{loc}: 值域 ≥ 0",
                    bool(rmin >= 0.0),
                    f"min {rmin:.2f} m / max {rmax:.1f} m / mean {rmean:.2f} m",
                )
            else:
                raise KeyError(f"{layer_id} 不在派生层保真实现清单内")

        if layer_id == "derived__landsea_mask":
            checks.append(
                {
                    "name": "口径注记",
                    "status": "report",
                    "summary": (
                        "掩膜基于冰面高程（全域 ice-surface 0m 阈值，冰盖归陆，"
                        "常规地理口径），与高程层（裸地复合口径）"
                        "定义不同：存在 mask=陆而高程<0 的冰下盆地像元"
                        "（南极/格陵兰）——论文口径说明消费"
                    ),
                }
            )

    n_fail = sum(1 for c in checks if c["status"] == "FAIL")
    return {
        "layer": layer_id,
        "store": str(store_dir),
        "source_layer": (
            "etopo2022-bed ice-surface 全域复合（掩膜冰面口径）"
            if layer_id == "derived__landsea_mask"
            else ELEVATION_LAYER_ID
        ),
        "result": "FAIL" if n_fail else "PASS",
        "n_checks": len(checks),
        "checks": checks,
    }


def write_fidelity_report(report: dict, reports_dir: str | Path) -> Path:
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    layer = report["layer"]
    json_path = reports_dir / f"fidelity-{layer}.json"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if "integral_tolerance" in report:
        criterion = f"积分量判据 < {report['integral_tolerance']:g} 相对偏差"
    else:
        criterion = f"与源层 {report.get('source_layer', '?')} 逐位一致 + 值域硬判据"
    lines = [
        f"# 保真验证报告 — {layer}",
        "",
        f"- 结果：**{report['result']}**（{criterion}）",
        f"- store：`{report['store']}`",
        "",
        "| 检查项 | 状态 | 数值 |",
        "|---|---|---|",
    ]
    for c in report["checks"]:
        if "summary" in c:
            detail = c["summary"]
        elif "rel_dev" in c:
            detail = f"积分量相对偏差 {c['rel_dev']:.3e}"
        elif "shift_m" in c:
            unit = c.get("unit", "m")
            detail = "; ".join(f"{k}:{v:+.1f}{unit}" for k, v in c["shift_m"].items())
        else:
            detail = (
                f"经度边界比 {c['lon_ratio']:.2f}（{c['lon_seam_p999']:.1f}/"
                f"{c['lon_base_p999']:.1f} m），纬度边界比 {c['lat_ratio']:.2f}"
                f"（{c['lat_seam_p999']:.1f}/{c['lat_base_p999']:.1f} m）"
            )
        lines.append(f"| {c['name']} | {c['status']} | {detail} |")
    (reports_dir / f"fidelity-{layer}.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return json_path
