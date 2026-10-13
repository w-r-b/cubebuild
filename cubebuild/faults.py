"""Fault dual channel: GEM active-fault slip_type class layer + spherical great-circle distance field.

Source (disk-verified, harmonized main version):
- geopackage/gem_active_faults_harmonized.gpkg: 13696 LineStrings (no
  MultiLineString), EPSG:4326, 154994 vertices (avg 11.3 points/line) → 140955
  valid segments (343 zero-length segments dropped), total length 665632 km
  (spherical, authalic R); ring edges with |Δlon|>180° (crossing the dateline)
  measured 0; slip_type vocabulary 23 classes + NaN for 319 records (not
  rasterized as classes; still counted in the distance field).
- the non-harmonized gem_active_faults.gpkg (16195 rows, CRS undefined) is the
  original compilation and is not ingested into the cube.

Distance-field engine (recomputed exactly per tier from the vector source by
spherical great-circle distance, independent of the downsampling kernel):
- each tier's pixel value = spherical great-circle distance (km, IUGG authalic
  R — same sphere as the pixel-area layer) from that tier's pixel center to the
  nearest fault-line feature; it references no coarse-tier values and involves
  no downsampling.
- computing min over all ~1.4×10⁵ segments directly for the full grid is not
  feasible (30″ tier has 9.3×10⁸ pixels), so a hierarchical annulus candidate
  pruning is used (mathematically guaranteed not to miss candidates):

  Lemma (candidate annulus): for a cell C with center c and half-diagonal ρ
  (upper bound on the spherical distance from any point in C to c), the nearest
  segment s_p for any p ∈ C satisfies
      dist(c, s_p) ∈ [d(c) − 2ρ, d(c) + 2ρ]
  (two applications of the triangle inequality + the 1-Lipschitz property of
  the distance function). By induction: the 1° tier center computes an exact
  min over all segments and collects the annulus stored(1°); each tier's child
  cell candidate set = the parent tier's stored (the induction guarantees it
  contains the child pixel's argmin); taking the min over the candidates yields
  that tier's exact value, then filtering by its own annulus yields stored to
  pass to finer tiers. The far-field annulus is a thin ring of radius d(c) and
  width 4ρ — the candidate count does not blow up as tiers are refined, which is
  the key to making exact full-grid computation affordable.
- values are lossless by the lemma; the fidelity report cross-checks against an
  independently implemented aviation formula (haversine + initial bearing)
  brute-forced over all segments by sampling.
- parallelism: decomposed into self-contained tasks by 1° cell (each cell
  computes the 1° brute-force min + annulus + per-tier refinement), fork process
  pool; the per-pixel arithmetic is independent of task order, so build
  determinism is unaffected by parallelism.

Class channel (reuses the rasterization machinery): rasterize line features
(GDAL default line algorithm for lines, all_touched=False, single-pixel-wide
trace) → coarse tiers use kernels.aggregate_mode mode aggregation (nodata=0 not
participating, ties take the smaller, ≥1 valid subcell means a value); for
overlapping pixels the shorter fault wins (local refinement first, per the
"smaller-area polygon" principle); faults with missing slip_type are not burned
(0 = no fault passing through).

License: the GEM GAF-DB repository LICENSE.txt is CC BY-SA 4.0 (the SA clause
does not meet open-license requirements); the two layers and sidecar are not
included in the public version; the paper suggests readers use GEM GAF-DB directly.
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
    """Category encoding table (code → semantics); the single shared source for attrs/manifest/sidecar."""
    return {str(c): label for label, c in sorted(SLIP_TYPE_CODES.items(), key=lambda kv: kv[1])}



def load_gem_faults(src_dir: str | Path = GEM_SRC_DIR):
    """Read the harmonized gpkg → (GeoDataFrame, slip_type code array).

    Geometry contract (contractualized disk evidence): 13696 rows, EPSG:4326, all
    LineStrings, no empty geometries; slip_type vocabulary ⊆ the encoding table (NaN
    allowed, 319 records); ring edges with |Δlon|>180° rejected (crossing the dateline
    silently misregisters planar rasterization; measured 0 records).
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



def _unit_vec(lat_deg, lon_deg) -> np.ndarray:
    """(lat, lon) degrees → 3D unit vector (..., 3). No special cases at the dateline/poles."""
    phi = np.deg2rad(np.asarray(lat_deg, dtype=np.float64))
    lam = np.deg2rad(np.asarray(lon_deg, dtype=np.float64))
    cphi = np.cos(phi)
    return np.stack([cphi * np.cos(lam), cphi * np.sin(lam), np.sin(phi)], axis=-1)


def segment_table(geoms):
    """Fault line features → segment table (the single shared source for the engine and fidelity).

    Adjacent vertices form segments; zero-length segments (consecutive repeated
    vertices) are dropped and counted. Returns (A, B, NV, sinAB, cosAB, latlon):
    - A/B (n,3) segment-endpoint unit vectors; NV = A×B (unnormalized, magnitude = sin δ_AB);
    - sinAB/cosAB (n,) segment-angle sine/cosine;
    - latlon (4,n) endpoint original coordinates (degrees, for the independent aviation-formula cross-check path).

    Contract: segment angle < 90° (cosAB > 0) — an overlong segment implies an
    anomalous source geometry and is rejected at read time.
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
    """Point set → spherical great-circle distance (m, n) km to each segment (the engine's sole distance kernel, a pure function).

    3D-vector path: sin_xt = p·(A×B)/|A×B| (cross-product chord);
    the foot of the perpendicular lies within the segment ⟺ (p·A) ≥ cosAB·cos_xt and
    (p·B) ≥ cosAB·cos_xt (via the cosine-monotonicity transform of along-track angle
    α ≤ δ_AB ⟺ cos α ≥ cos δ_AB).
    Foot within the segment → R·|asin(sin_xt)|; otherwise → R·arccos(max(p·A, p·B))
    (the nearer endpoint). The anti-side far field (including antipodes) is correctly
    handled by the endpoint branch.
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
    """Independent implementation: classic aviation formula (haversine + initial bearing) brute-force min over all segments.

    Deliberately heterogeneous from the engine (3D-vector cross/dot-product path), for
    fidelity cross-validation (acceptance: sampled pixels compared against the independent
    great-circle distance computation). Vectorized per segment, scalar arguments.
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

    d12 = _delta(latA, lonA, latB, lonB)
    d13 = _delta(latA, lonA, phi, lam)             # A→p
    t12 = _bearing(latA, lonA, latB, lonB)
    t13 = _bearing(latA, lonA, phi, lam)
    sin_xt = np.clip(np.sin(d13) * np.sin(t13 - t12), -1.0, 1.0)
    xt = np.arcsin(sin_xt)
    d23 = _delta(latB, lonB, phi, lam)             # B→p
    at = np.arccos(np.clip(np.cos(d13) / np.cos(xt), -1.0, 1.0))
    bt = np.arccos(np.clip(np.cos(d23) / np.cos(xt), -1.0, 1.0))
    interior = (at <= d12) & (bt <= d12)
    ang = np.where(interior, np.abs(xt), np.minimum(d13, d23))
    return float(EARTH_RADIUS_KM * ang.min())



_TIER_CHAIN = TIER_ORDER

_KM_PER_DEG = 2.0 * math.pi * EARTH_RADIUS_KM / 360.0
_ANNULUS_KM = {
    t: 2.0 * tier_deg(t) * math.sqrt(0.5) * _KM_PER_DEG + 1e-3
    for t in _TIER_CHAIN
}

_POOL_SEGMENTS = None


def _distance_cell(i: int, j: int, segs, chain) -> dict[str, np.ndarray]:
    """Single 1° cell full-chain task: returns {tier: (n, m) float64} (exact values for each tier's pixels within the cell).

    Self-contained (the annulus lemma references only this cell's geometry, no cross-cell
    dependency): the 1° center computes a brute-force min over all segments → annulus
    stored(1°); per-tier refinement: child-cell candidates = the parent cell's stored,
    take the min for the exact value, then filter by its own annulus to obtain stored to
    pass to finer tiers; the finest tier only takes the min without filtering.
    parents/children stay row-major throughout → cell block placement is deterministic.
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

    parents: list[tuple[int, int, float, float, np.ndarray]] = [
        (0, 0, lat_lo + 0.5, lon_lo + 0.5, cand0)
    ]
    for k in range(1, len(chain)):
        lvl, plvl = chain[k], chain[k - 1]
        f = int(tier_fraction(plvl) / tier_fraction(lvl))
        finest = k == len(chain) - 1
        w = _ANNULUS_KM[lvl]
        rp = tier_deg(plvl)
        sub_res = rp / f
        block = int(Fraction(1) / tier_fraction(lvl))
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
    """Segment table → {tier: float32 global grid} (exact distance values for each tier in the chain).

    Decomposed into self-contained tasks by 1° cell; when workers > 1 a fork process pool
    runs in parallel (the per-pixel arithmetic is independent of task order, so determinism
    is unaffected by parallelism — rerun-checksum consistency is double-checked by the
    acceptance double-run).

    cache_dir (memory guardrail): when given, each tier's grid is written to .npy and then
    returned as a read-only mmap — the 30″ tier's 3.7GB does not reside in process RSS
    (file pages are reclaimable), the key relief for the full-build peak budget (≤40GB);
    None = pure in-memory (short chain for tests).
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
                a._mmap.close()
        raise
    if cache_dir is not None:
        for t in chain:
            out[t].flush()
            out[t] = np.load(cache_dir / f"{t}.npy", mmap_mode="r")
    return out



_PYRAMID_CACHE_DIR = REPO_ROOT / "products/.build/faults-distance"

_PYRAMID_CACHE: dict[str, dict[str, np.ndarray]] = {}


def _distance_pyramid(src_dir: str | Path = GEM_SRC_DIR) -> dict[str, np.ndarray]:
    """Real-source full-chain pyramid (in-process cache: the five tiers are computed once, ~1.4×10⁵ segments)."""
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
    """Build stress_kinematics__gem_fault_distance for the given tier (float32, values everywhere)."""
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
        "nodata_semantics": "无缺测：距离场全域有值（免掩膜）",
    }
    arr = da.from_array(values, chunks=tier_chunks(tier))
    return xr.DataArray(
        arr, dims=("lat", "lon"), coords={"lat": lat, "lon": lon},
        name=entry.id, attrs=attrs,
    )



def rasterize_fault_classes(geoms, codes: np.ndarray, tier: str) -> tuple[np.ndarray, int]:
    """Fault line features → the tier's class grid (uint8, 0 = no fault passing through).

    Semantics: GDAL default line algorithm for lines (all_touched=False) — pixels the line
    crosses are burned, a single-pixel-wide trace; for overlapping faults in the same pixel
    the shorter fault wins (local refinement first, the same "smaller-area polygon" principle:
    stable descending-length sort + last-burner-wins); codes=0 (missing slip_type) is not
    burned and not counted in overlap.

    Returns (classes (nlat, nlon) uint8, overlap pixel count). Burning is done per-fault in a
    bounding-box window, fully deterministic and rerunnable.
    """
    import rasterio.features as rfeatures
    from rasterio.transform import from_origin

    nlat, nlon = grid_shape(tier)
    res = tier_deg(tier)
    classes = np.zeros((nlat, nlon), dtype=np.uint8)
    covered = np.zeros((nlat, nlon), dtype=np.uint8)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*geographic CRS.*")
        order = np.argsort([-g.length for g in geoms], kind="stable")

    for i in order:
        if codes[i] == NODATA:
            continue
        g = geoms[i]
        minx, miny, maxx, maxy = g.bounds
        r0 = max(int(np.floor((miny + 90.0) / res)) - 1, 0)
        r1 = min(int(np.ceil((maxy + 90.0) / res)) + 1, nlat)    # exclusive
        c0 = max(int(np.floor((minx + 180.0) / res)) - 1, 0)
        c1 = min(int(np.ceil((maxx + 180.0) / res)) + 1, nlon)   # exclusive
        if r0 >= r1 or c0 >= c1:
            continue
        tr = from_origin(-180.0 + c0 * res, -90.0 + r1 * res, res, res)
        burned = rfeatures.rasterize(
            [(g, 1)], out_shape=(r1 - r0, c1 - c0), transform=tr, fill=0, dtype="uint8"
        ).astype(bool)
        burned = burned[::-1]
        win = classes[r0:r1, c0:c1]
        classes[r0:r1, c0:c1] = np.where(burned, codes[i], win)
        covered[r0:r1, c0:c1] += burned

    overlap_cells = int((covered > 1).sum())
    return classes, overlap_cells


_CLASS_CACHE: dict[str, tuple[np.ndarray, int]] = {}
_MODE_CACHE: dict[tuple[str, str], np.ndarray] = {}


def _home_tier_classes(src_dir: str | Path = GEM_SRC_DIR) -> tuple[np.ndarray, int]:
    """Home-tier (30″) class grid + overlap pixel count (in-process cache: the five tiers rasterize only once)."""
    key = f"gem:{Path(src_dir).resolve()}"
    if key not in _CLASS_CACHE:
        gdf, codes = load_gem_faults(src_dir)
        _CLASS_CACHE[key] = rasterize_fault_classes(
            list(gdf.geometry), codes, HOME_TIER
        )
    return _CLASS_CACHE[key]


def _tier_classes(tier: str, src_dir: str | Path = GEM_SRC_DIR) -> np.ndarray:
    """The tier's class grid: the home tier is emitted directly, coarse tiers = home-tier mode aggregation (cached)."""
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
    """Build stress_kinematics__gem_fault_slip_class for the given tier (uint8 class layer)."""
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



FAULT_FIDELITY_LAYERS = {
    SLIP_CLASS_LAYER_ID: {"kind": "class"},
    DISTANCE_LAYER_ID: {"kind": "distance"},
}

_SAMPLE_SEED = 20260912
_N_SAMPLES = 120
_DIST_TOL_KM = 1e-3
_DIST_REL_TOL = 3e-7


def build_faults_fidelity_report(store_dir, layer_id: str, tiers: list[str],
                                  src_dir=None) -> dict:
    """Fidelity report.

    Class-layer criteria (isomorphic): ① the store is bit-identical to the source recomputation
    (uint8 bit level); ② coarse-tier class-mode agreement (= mode aggregation of the home-tier
    store values); ③ value range ⊆ {0} ∪ the encoding table; ④ class distribution/overlap/
    unclassified totals (reporting only).

    Distance-field criteria (acceptance): ⑤ per-tier stratified random pixel sampling (uniform +
    near-field <50 km) brute-force compared against the independent aviation formula over all
    segments, max|Δ| ≤ 1e-3 km (hard criterion); ⑥ finite everywhere and ≥ 0 (hard criterion,
    the no-mask semantics); ⑦ distance-distribution statistics (reporting only).
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

        for tier in tiers:
            expect = _tier_classes(tier, src_dir)
            n_bad = int((values[tier] != expect).sum())
            checks.append({
                "name": f"{tier}: 与源栅格化重算逐位一致（uint8 位级）",
                "status": "pass" if n_bad == 0 else "FAIL",
                "summary": f"不一致像元 {n_bad}（期望 0）",
            })

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

                n_bad = int((~np.isfinite(v)).sum() + (v < 0).sum())
                checks.append({
                    "name": f"{tier}: 全域有限且 ≥ 0（免掩膜语义）",
                    "status": "pass" if n_bad == 0 else "FAIL",
                    "summary": f"违法像元 {n_bad}（期望 0）",
                })

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
        "integral_tolerance": INTEGRAL_TOL,
        "result": "FAIL" if n_fail else "PASS",
        "n_checks": len(checks),
        "checks": checks,
    }
