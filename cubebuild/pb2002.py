"""PB2002 plate three-channel set: plate_id categorical layer + zone_class tectonic-zone categorical layer
+ boundary_distance distance field + pb2002_plates sidecar.

Sources (disk-verified):
- PB2002_plates.dig.txt: 52 plate closed polygons (counterclockwise, last point = first point).
  Parsing traps (disk-verified): ① the ring coordinates span both sides of the dateline (PA crosses 4 times), so
  connecting them planarly would slash the whole map (a naive planar polygon once made an Alpine point fall in PA) — each ring must be
  longitude-unwrapped; ② AN rings the South Pole, NA rings the North Pole (net longitude winding ±360°), so a planar
  polygon must add pole-closing vertices, else the Jordan interior picks the wrong side (the polar cap is lost);
  ③ the MS (Molucca Sea) ring is a source-inherent zero-width spike shape (a back-and-forth self-intersection), and make_valid
  repairs it by removing only the zero-area remnant (area Δ=7e-15 deg², rasterization bit-identical).
  The orogenic belts (13 of them) do not occupy plate codes: PB2002 cuts closed plate polygons with arbitrary internal lines
  (readme "overlay relationship"), so the 52 polygons truly cover exhaustively (6′ cell-center
  rasterization measured: uncovered 0, inter-plate overlap 0, window-clipped union area = 64800.0 deg²
  exactly tiling the globe).
- PB2002_boundaries.dig.txt: 229 boundary segments, 6048 points; the 5-byte title = left plate
  [0:2] + polarity [2] + right plate [3:5], '/' = right subducts under left, '\\' = reverse,
  '-' = non-subducting (168/40/21 segments); the citation note starts at byte 27 (3 segments have no citation);
  6 segments cross the dateline (the spherical 3D vector engine is insensitive to the dateline).
- PB2002_poles.dat.txt: 52 Euler poles, fixed-width (A2,F9.3,F10.3,F9.4) + citation column
  [32:]; the first 3 rows carry an additional 3-line format legend from column 56 (not part of the citation). PB2002_poles.xls
  is a binary copy and is not consumed.
- PB2002_tectonic_zones.grd.txt: Kagan, Bird & Jackson (2010, Pure Appl.
  Geophys. 167, 721–741, doi:10.1007/s00024-010-0075-3) tectonic-zone map
  (five-zone scheme), Bird .GRD format (peterbird.name/guide/grd_format.htm):
  row 1 = lon_min,d_lon,lon_max = 0,0.1,360; row 2 = lat_min,d_lat,lat_max =
  -90,0.1,90; data starts at the top-left corner (max latitude, min longitude), rows north→south, columns west→east;
  1801×3601 nodes = point samples (paper §2.2 "assigns tectonic zone integers
  to grid points"), the last column = the first column (a dateline wraparound copy, bit-identical on disk;
  polar rows all 0). Code table (paper §2.1 + the PB2002_steps.dat 5819-step classification cross-table
  double-verified: continental boundaries CCB/CRB/CTF→1, ridges/transforms OSR/OTF→2/3, subduction/convergence
  SUB/OCB→4): 0 = plate interior, 1 = active continent (including the continental parts of all orogenic belts + continental plate
  boundaries), 2 = slow-spreading ridge (<40 mm/a, including transforms), 3 = fast-spreading ridge (≥40 mm/a, including
  transforms), 4 = trench (including incipient subduction, outer rise and overriding plate).

Channels (three tiers 1°/30′/6′, primary tier 6′):
- plate_id: primary-tier 6′ cell-center rasterization (unwrap + pole-closed
  polygons' ±360 shifted copies burned once as a MultiPolygon, descending area so the last burner wins) + coarse tiers
  via kernels.aggregate_mode mode aggregation. Exhaustive coverage needs no mask (plate distance
  is handled by boundary_distance; plate_id produces no distance field).
- zone_class: 0.1° node point samples → 6′ cell-center nearest-node sampling (a cell center is exactly equidistant from 4
  adjacent nodes, takes the mode of the 4 corners, ties take the smaller code — equivalent to a deterministic tie-breaking
  nearest-neighbor sampling; includes longitude wrap and ±90 edge rows); source code shifted +1 (1 = plate interior …
  5 = trench; 0 = missing sentinel, absent since the source grid is fully valued, so validity is identically 1); coarse tiers via mode
  aggregation (nodata=0 excluded, ties take the smaller value, ≥1 valid subcell means a value).
- boundary_distance: the 229 boundary segments reuse the distance engine (the hierarchical ring-band candidate pruning
  lemma guarantees no missed candidates, recomputed exactly from the vector source tier by tier, IUGG authalic R spherical
  great-circle km); fidelity re-checks it by brute-force comparison of all segments against the same independent haversine formula.

License: the PB2002 source (peterbird.name/oldFTP/PB2002/) has no explicit license,
listing only files and a citation format; not included in the public version.
"""

import json
import os
from pathlib import Path

import dask.array as da
import numpy as np
import xarray as xr

from .contract import LayerEntry
from .faults import distance_grids, independent_min_distance_km, segment_table
from .fidelity import INTEGRAL_TOL
from .grids import aggregation_factor, tier_centers, tier_chunks
from .kernels import aggregate_mode
from .vector import rasterize_polygon_classes

REPO_ROOT = Path(__file__).resolve().parents[1]
PB2002_SRC_DIR = REPO_ROOT / "original data/stress-kinematics/pb2002-plate-boundaries"
PLATES_NAME = "PB2002_plates.dig.txt"
BOUNDARIES_NAME = "PB2002_boundaries.dig.txt"
POLES_NAME = "PB2002_poles.dat.txt"
ZONES_NAME = "PB2002_tectonic_zones.grd.txt"

BIRD2003_DOI = "10.1029/2001GC000252"           # Bird 2003, G3 4(3), 1027
KAGAN2010_DOI = "10.1007/s00024-010-0075-3"     # Kagan et al. 2010, PAGEOPH 167, 721-741
PB2002_URL = "http://peterbird.name/oldFTP/PB2002/"
GRD_FORMAT_URL = "http://peterbird.name/guide/grd_format.htm"

PLATE_ID_LAYER_ID = "stress_kinematics__pb2002_plate_id"
ZONE_CLASS_LAYER_ID = "stress_kinematics__pb2002_zone_class"
BOUNDARY_DISTANCE_LAYER_ID = "stress_kinematics__pb2002_boundary_distance"
HOME_TIER = "6min"
DISTANCE_CHAIN = ("1deg", "30min", "6min")

EXPECTED_PLATES = 52
EXPECTED_WINDING = {"AN": -1, "NA": 1}
EXPECTED_BOUNDARY_SEGMENTS = 229
EXPECTED_BOUNDARY_POINTS = 6048
EXPECTED_POLES = 52
NODATA = 0

LICENSE_NOTE = (
    "PB2002 源（peterbird.name/oldFTP/PB2002/）无明示许可，仅列文件与引用"
    "格式；不纳入公开版。引用 Bird (2003) "
    f"doi:{BIRD2003_DOI}，区划图另引 Kagan et al. (2010) "
    f"doi:{KAGAN2010_DOI}"
)

ZONE_LABELS = {
    0: "plate-interior",
    1: "active-continent",
    2: "slow-spreading-ridge",
    3: "fast-spreading-ridge",
    4: "trench",
}

PLATE_CODES: dict[str, int] = {
    label: i for i, label in enumerate(sorted([
        "AF", "AM", "AN", "AP", "AR", "AS", "AT", "AU", "BH", "BR", "BS",
        "BU", "CA", "CL", "CO", "CR", "EA", "EU", "FT", "GP", "IN", "JF",
        "JZ", "KE", "MA", "MN", "MO", "MS", "NA", "NB", "ND", "NH", "NI",
        "NZ", "OK", "ON", "PA", "PM", "PS", "RI", "SA", "SB", "SC", "SL",
        "SO", "SS", "SU", "SW", "TI", "TO", "WL", "YA",
    ]), start=1)
}


def category_encoding() -> dict[str, str]:
    """Plate encoding table (code → semantics), the single source shared by attrs/manifest/sidecar."""
    return {str(c): label for label, c in sorted(PLATE_CODES.items(), key=lambda kv: kv[1])}


def zone_encoding() -> dict[str, str]:
    """Tectonic-zone encoding table (layer code → semantics; layer code = source code + 1)."""
    return {str(c + 1): label for c, label in sorted(ZONE_LABELS.items())}



def _parse_dig(path: Path) -> list[dict]:
    """Bird .dig file → [{'title', 'pts'}, ...] (segmented by *** end of line segment ***)."""
    segments: list[dict] = []
    cur: dict | None = None
    for ln in path.read_text(encoding="ascii").splitlines():
        s = ln.strip()
        if not s:
            continue
        if s.startswith("***"):
            if cur is not None:
                segments.append(cur)
            cur = None
            continue
        if cur is None:
            cur = {"title": s, "pts": []}
            continue
        lon, lat = (float(x) for x in s.split(","))
        cur["pts"].append((lon, lat))
    if cur is not None:
        segments.append(cur)
    return segments



def _ring_unwind(ring_pts: list[tuple[float, float]]):
    """Closed ring → (unwrapped ring, net longitude winding turns).

    When adjacent points' longitude difference |Δlon|>180, a ±360 correction makes the ring continuous (taking the short arc across the dateline);
    last point = first point (closed ring), winding = Σadj/360.
    """
    a = np.asarray(ring_pts, dtype=np.float64)
    lon, lat = a[:, 0].copy(), a[:, 1]
    d = np.diff(lon)
    adj = np.zeros_like(d)
    adj[d > 180.0] = -360.0
    adj[d < -180.0] = 360.0
    lon[1:] += np.cumsum(adj)
    winding = int(round(adj.sum() / 360.0))
    return list(zip(lon.tolist(), lat.tolist())), winding


def _polygonal_only(geom):
    """Take the areal components of a make_valid result (for a GeometryCollection, discard the zero-area line/point remnants)."""
    from shapely.ops import unary_union

    if geom.geom_type in ("Polygon", "MultiPolygon"):
        return geom
    parts = [g for g in geom.geoms if g.geom_type in ("Polygon", "MultiPolygon")]
    if not parts:
        raise ValueError("make_valid 后无面状组分——源几何异常")
    return unary_union(parts)


def plate_polygon(ring_pts: list[tuple[float, float]]):
    """Plate closed ring → planar shapely polygon (unwrap + pole closure + MS-type repair).

    Pole closure: net winding +1 (eastward around the pole) adds a north-pole vertex, -1 adds a south-pole vertex — a ring
    around a polar cap's planar projection excludes the cap, so it must be explicitly closed to the pole (the AN/NA disk-verified contract).
    Returns (Polygon/MultiPolygon, winding).
    """
    from shapely.geometry import Polygon
    from shapely.validation import make_valid

    ring, winding = _ring_unwind(ring_pts)
    if winding == 1:
        ring = ring + [(ring[-1][0], 90.0), (ring[0][0], 90.0)]
    elif winding == -1:
        ring = ring + [(ring[-1][0], -90.0), (ring[0][0], -90.0)]
    elif winding != 0:
        raise ValueError(f"板块环净缠绕 {winding}（>1 圈）：源形态异常")
    poly = Polygon(ring)
    if not poly.is_valid:
        poly = _polygonal_only(make_valid(poly))
    return poly, winding


def plate_window_geom(poly):
    """Correct polygon → standard geometry within the [-180,180] window (the union of three shifted copies ∩ the window).

    For the sidecar: after unwrap, the parts beyond ±180 are brought back into the window via ±360 shifted copies; only
    areal components are kept (line-like remnants at the window boundary are discarded). Disk-verified: the 52 plates' window-clipped
    union area = 64800.0 deg² (exactly tiling the global box).
    """
    from shapely.affinity import translate
    from shapely.geometry import box
    from shapely.ops import unary_union

    window = box(-180.0, -90.0, 180.0, 90.0)
    parts = []
    for shift in (0.0, 360.0, -360.0):
        clipped = translate(poly, shift).intersection(window)
        for g in (clipped.geoms if clipped.geom_type == "GeometryCollection"
                  else (clipped,)):
            if g.geom_type in ("Polygon", "MultiPolygon") and not g.is_empty:
                parts.append(g)
    if not parts:
        raise ValueError("板块多边形窗口裁剪后无面状组分——几何异常")
    return unary_union(parts)


def split_dateline_pts(pts: list[tuple[float, float]]) -> list[list[tuple[float, float]]]:
    """Line-segment points → the parts after dateline splitting (each part continuous and landing in [-180, 180)).

    After unwrap, each edge is checked for crossing a 180+360k meridian (boundary step ≤109 km, a single edge crosses at most
    one), with the split point at planar linear interpolation (sub-meter deviation from the source great-circle arc — the split point lies on the ±180
    meridian, both sides' coordinates coincide in longitude, and the latitude difference is the straight-line/great-circle interpolation difference); each part's end is
    translated as a whole into [-180, 180). The 6 dateline-crossing boundaries thereby obtain standard GIS geometry.
    """
    up = [pts[0]]
    for lon, lat in pts[1:]:
        x_prev = up[-1][0]
        d = lon - x_prev
        if d > 180.0:
            lon -= 360.0
        elif d < -180.0:
            lon += 360.0
        up.append((lon, lat))

    parts: list[list[tuple[float, float]]] = []
    cur = [up[0]]
    for (x0, y0), (x1, y1) in zip(up, up[1:]):
        lo, hi = min(x0, x1), max(x0, x1)
        for k in range(-3, 4):
            b = 180.0 + 360.0 * k
            if lo <= b < hi:
                t = (b - x0) / (x1 - x0)
                yc = y0 + t * (y1 - y0)
                cur.append((b, yc))
                parts.append(cur)
                cur = [(b, yc)]
        cur.append((x1, y1))
    parts.append(cur)

    out = []
    for p in parts:
        mean_x = sum(x for x, _ in p) / len(p)
        shift = -360.0 * round(mean_x / 360.0)
        out.append([(x + shift, y) for x, y in p])
    return out



def load_pb2002_plates(src_dir: str | Path = PB2002_SRC_DIR):
    """Read PB2002_plates.dig.txt → (polygon list, code array, plate-code list).

    Contract (contractualized from disk evidence): 52 segments, two-letter titles unique, rings closed (last point = first point),
    AN winding the South Pole (−1) / NA winding the North Pole (+1) net winding, all valid and non-empty after repair.
    """
    path = Path(src_dir) / PLATES_NAME
    if not path.is_file():
        raise FileNotFoundError(f"PB2002 板块源文件不存在: {path}")
    segments = _parse_dig(path)
    if len(segments) != EXPECTED_PLATES:
        raise ValueError(f"{PLATES_NAME}: 段数 {len(segments)} ≠ 期望 {EXPECTED_PLATES}")

    labels: list[str] = []
    polys = []
    windings: dict[str, int] = {}
    for seg in segments:
        title = seg["title"].split()[0]
        if len(title) != 2 or not title.isalpha() or not title.isupper():
            raise ValueError(f"{PLATES_NAME}: 板块标题非法: {seg['title']!r}")
        if seg["pts"][0] != seg["pts"][-1]:
            raise ValueError(f"{PLATES_NAME}: 板块 {title} 环未闭合（末点≠首点）")
        poly, winding = plate_polygon(seg["pts"])
        if poly.is_empty or not poly.is_valid:
            raise ValueError(f"{PLATES_NAME}: 板块 {title} 多边形无效")
        labels.append(title)
        polys.append(poly)
        windings[title] = winding
    if len(set(labels)) != EXPECTED_PLATES:
        raise ValueError(f"{PLATES_NAME}: 板块码重复: {sorted(labels)}")
    if set(labels) != set(PLATE_CODES):
        raise ValueError(
            f"{PLATES_NAME}: 板块码集合 ≠ 编码表: 多出 "
            f"{sorted(set(labels) - set(PLATE_CODES))}，缺失 "
            f"{sorted(set(PLATE_CODES) - set(labels))}"
        )
    for plate, want in EXPECTED_WINDING.items():
        if windings.get(plate) != want:
            raise ValueError(
                f"{PLATES_NAME}: 板块 {plate} 净缠绕 {windings.get(plate)} ≠ 期望 "
                f"{want}（极点闭合契约破坏，源变更须复评）"
            )
    codes = np.array([PLATE_CODES[k] for k in labels], dtype=np.uint8)
    return polys, codes, labels


def load_pb2002_boundaries(src_dir: str | Path = PB2002_SRC_DIR) -> list[dict]:
    """Read PB2002_boundaries.dig.txt → 229 segment attribute records (with coordinate point lists).

    Contract: 229 segments, 6048 points, 5-byte titles (left/right plates ∈ the encoding table, polarity byte
    ∈ {-, /, \\}), citation note from byte 27 (may be empty).
    """
    path = Path(src_dir) / BOUNDARIES_NAME
    if not path.is_file():
        raise FileNotFoundError(f"PB2002 边界源文件不存在: {path}")
    segments = _parse_dig(path)
    if len(segments) != EXPECTED_BOUNDARY_SEGMENTS:
        raise ValueError(
            f"{BOUNDARIES_NAME}: 段数 {len(segments)} ≠ 期望 {EXPECTED_BOUNDARY_SEGMENTS}"
        )
    n_pts = sum(len(s["pts"]) for s in segments)
    if n_pts != EXPECTED_BOUNDARY_POINTS:
        raise ValueError(
            f"{BOUNDARIES_NAME}: 总点数 {n_pts} ≠ 期望 {EXPECTED_BOUNDARY_POINTS}"
        )

    records = []
    for seg in segments:
        title = seg["title"]
        code = title[:5]
        left, pol, right = code[:2], code[2], code[3:5]
        if left not in PLATE_CODES or right not in PLATE_CODES:
            raise ValueError(f"{BOUNDARIES_NAME}: 边界标题板块码非法: {title!r}")
        if pol not in ("-", "/", "\\"):
            raise ValueError(f"{BOUNDARIES_NAME}: 边界极性字节非法: {title!r}")
        if len(seg["pts"]) < 2:
            raise ValueError(f"{BOUNDARIES_NAME}: 段 {code} 点数 < 2")
        citation = title[27:].rstrip() if len(title) > 27 else ""
        records.append({
            "code": code,
            "plate_left": left,
            "plate_right": right,
            "polarity": pol,
            "citation": citation,
            "pts": seg["pts"],
        })
    return records


def load_pb2002_poles(src_dir: str | Path = PB2002_SRC_DIR):
    """Read PB2002_poles.dat.txt → a pandas DataFrame (52 Euler poles).

    Fixed-width (A2,F9.3,F10.3,F9.4); citation column [32:], the first 3 rows carry a 3-line
    format legend from column 56 (not part of the citation, removed by row index). Contract: 52 rows, plate codes ⊆ the encoding table and unique.
    """
    import pandas as pd

    path = Path(src_dir) / POLES_NAME
    if not path.is_file():
        raise FileNotFoundError(f"PB2002 欧拉极源文件不存在: {path}")
    rows = []
    lines = path.read_text(encoding="ascii").splitlines()
    for i, line in enumerate(lines):
        if len(line) < 30:
            raise ValueError(f"{POLES_NAME}: 第 {i+1} 行短于 30 字符: {line!r}")
        plate = line[:2]
        if plate not in PLATE_CODES:
            raise ValueError(f"{POLES_NAME}: 第 {i+1} 行板块码非法: {plate!r}")
        citation = line[32:56].rstrip() if i < 3 else line[32:].rstrip()
        rows.append({
            "plate": plate,
            "lat_deg": float(line[2:11]),
            "lon_deg": float(line[11:21]),
            "rate_deg_per_ma": float(line[21:30]),
            "citation": citation,
        })
    if len(rows) != EXPECTED_POLES:
        raise ValueError(f"{POLES_NAME}: 行数 {len(rows)} ≠ 期望 {EXPECTED_POLES}")
    if len({r["plate"] for r in rows}) != EXPECTED_POLES:
        raise ValueError(f"{POLES_NAME}: 板块码重复")
    return pd.DataFrame(rows)


def load_tectonic_zones(src_dir: str | Path = PB2002_SRC_DIR) -> np.ndarray:
    """Read PB2002_tectonic_zones.grd.txt → (1801, 3600) uint8 source code 0..4.

    Bird .GRD format (GRD_FORMAT_URL): rows north→south (first row = max latitude +90), columns
    west→east (lon 0..359.9); node point samples inclusive of endpoints. Contract: the first two rows parse to
    (0, 0.1, 360)/(−90, 0.1, 90); total value count = 1801×3601; last column = first column
    (a dateline wraparound copy, bit-identical before dedup); value range ⊆ {0..4}.
    """
    path = Path(src_dir) / ZONES_NAME
    if not path.is_file():
        raise FileNotFoundError(f"PB2002 构造区划源文件不存在: {path}")
    lines = path.read_text(encoding="ascii").splitlines()
    if len(lines) < 3:
        raise ValueError(f"{ZONES_NAME}: 文件过短")
    lon_min, d_lon, lon_max = (float(x) for x in lines[0].split())
    lat_min, d_lat, lat_max = (float(x) for x in lines[1].split())
    if not (lon_min == 0.0 and d_lon == 0.1 and lon_max == 360.0
            and lat_min == -90.0 and d_lat == 0.1 and lat_max == 90.0):
        raise ValueError(
            f"{ZONES_NAME}: 网格头 {lines[0]!r} / {lines[1]!r} ≠ 约定 "
            "(0, 0.1, 360)/(−90, 0.1, 90)（源变更须复评解析）"
        )
    values: list[int] = []
    for ln in lines[2:]:
        values.extend(int(t) for t in ln.split())
    nrows, ncols = 1801, 3601
    if len(values) != nrows * ncols:
        raise ValueError(
            f"{ZONES_NAME}: 数值总数 {len(values)} ≠ {nrows}×{ncols}"
        )
    g = np.array(values, dtype=np.uint8).reshape(nrows, ncols)
    if not np.array_equal(g[:, 0], g[:, -1]):
        raise ValueError(f"{ZONES_NAME}: 末列 ≠ 首列（日期线卷绕契约破坏）")
    bad = sorted(int(v) for v in np.unique(g) if v not in ZONE_LABELS)
    if bad:
        raise ValueError(f"{ZONES_NAME}: 值域越界 {bad}（期望 ⊆ 0..4，源变更须复评）")
    return g[:, :-1]



_PLATE_HOME_CACHE: dict[str, tuple] = {}
_PLATE_MODE_CACHE: dict[tuple, np.ndarray] = {}
_ZONE_HOME_CACHE: dict[str, np.ndarray] = {}
_ZONE_MODE_CACHE: dict[tuple, np.ndarray] = {}


def _plate_render_geoms(polys):
    """Correct polygon → MultiPolygon for rasterization (the body + ±360 shifted copies).

    The shifted copies let the dateline-crossing parts (whose unwrapped lon exceeds ±180) land correctly within the [-180,180] window;
    the three copies do not overlap (after unwrap the ring span is ≤360°+ε), so a single burn suffices.
    """
    from shapely.affinity import translate
    from shapely.geometry import MultiPolygon

    return [
        MultiPolygon([p, translate(p, 360.0), translate(p, -360.0)])
        for p in polys
    ]


def _home_tier_classes(src_dir: str | Path = PB2002_SRC_DIR):
    """Primary-tier 6′ class grid + overlap/uncovered counts (in-process cache: the three tiers are rasterized only once)."""
    key = f"pb2002:{Path(src_dir).resolve()}"
    cache = _PLATE_HOME_CACHE.get(key)
    if cache is None:
        polys, codes, _labels = load_pb2002_plates(src_dir)
        cache = rasterize_polygon_classes(
            _plate_render_geoms(polys), codes, HOME_TIER
        )
        _PLATE_HOME_CACHE[key] = cache
    return cache


def _tier_classes(tier: str, src_dir: str | Path = PB2002_SRC_DIR) -> np.ndarray:
    if tier == HOME_TIER:
        return _home_tier_classes(src_dir)[0]
    key = (tier, str(Path(src_dir).resolve()))
    cache = _PLATE_MODE_CACHE.get(key)
    if cache is None:
        home = _home_tier_classes(src_dir)[0]
        cache = aggregate_mode(
            home, aggregation_factor(tier, HOME_TIER), nodata=NODATA
        )
        _PLATE_MODE_CACHE[key] = cache
    return cache


RASTERIZATION_RULE = (
    "板块闭合多边形 unwrap（跨日期线短边走短弧）+ 极帽闭合（AN 绕南极/NA "
    "绕北极补极点顶点）+ ±360 平移副本 MultiPolygon 单次烧录；胞心落入"
    "多边形即属该板块（all_touched=False）；MS 摩鹿加海"
    "零宽钉形环 make_valid 修复（面积 Δ=7e-15 deg²，栅格化逐位不变）；"
    "多板块重叠取面积较小者（面积降序后烧胜；实测重叠 0）；粗档 = 主档 "
    "6′ 众数聚合（kernels.aggregate_mode，nodata=0 不参与、平票取小、"
    "≥1 有效子胞即有值）"
)


def build_pb2002_plate_id(entry: LayerEntry, tier: str) -> xr.DataArray:
    """Build this tier's stress_kinematics__pb2002_plate_id (uint8 categorical layer, exhaustive coverage)."""
    values = _tier_classes(tier)
    lat, lon = tier_centers(tier)
    attrs = {
        "units": entry.unit,
        "long_name": "板块归属（Bird 2003 PB2002，52 板块，像元内主导板块）",
        "source": entry.source,
        "doi": BIRD2003_DOI,
        "source_url": PB2002_URL,
        "license_note": LICENSE_NOTE,
        "native_res": entry.native_res,
        "resampling": entry.resampling,
        "visibility": entry.visibility,
        "category_encoding": json.dumps(
            category_encoding(), ensure_ascii=False, sort_keys=True
        ),
        "rasterization_rule": RASTERIZATION_RULE,
        "source_file": PLATES_NAME,
        "nodata_semantics": (
            "无缺测：52 多边形穷尽覆盖（6′ 实测未覆盖 0 像元），免掩膜；"
            "类别编码自 1 起（0 保留为缺测哨兵，不出现）"
        ),
    }
    arr = da.from_array(values, chunks=tier_chunks(tier))
    return xr.DataArray(
        arr, dims=("lat", "lon"), coords={"lat": lat, "lon": lon},
        name=entry.id, attrs=attrs,
    )



def zone_home_grid(zones: np.ndarray) -> np.ndarray:
    """Source node grid (1801, 3600) (rows north→south, source code 0..4) → 6′ primary-tier layer code (1800, 3600).

    A 6′ cell center is equidistant from 4 adjacent source nodes (the source node 0.1° phase differs from the tier grid by half a tier),
    so it takes the mode of the 4 corners, ties take the smaller source code — equivalent to deterministic tie-breaking
    nearest-node sampling; longitude column wrap (±180 adjacent) and the ±90 edge rows are naturally covered by the node grid being endpoint-inclusive.
    Returns uint8 layer code 1..5 (source code +1; 0 = missing sentinel, absent since the source has no missing values).
    """
    if zones.shape != (1801, 3600):
        raise ValueError(f"源节点网格形状 {zones.shape} ≠ (1801, 3600)")
    north = zones[:-1][::-1]
    south = zones[1:][::-1]
    j = np.arange(3600)
    wc = (j + 1800) % 3600
    ec = (j + 1801) % 3600
    four = np.stack([north[:, wc], north[:, ec], south[:, wc], south[:, ec]])

    out = np.zeros((1800, 3600), dtype=np.uint8)
    best = np.zeros((1800, 3600), dtype=np.int64)
    for c in sorted(ZONE_LABELS):
        cnt = (four == c).sum(axis=0)
        take = cnt > best
        out[take] = c
        np.maximum(best, cnt, out=best)
    return (out + 1).astype(np.uint8)


def _zone_home(src_dir: str | Path = PB2002_SRC_DIR) -> np.ndarray:
    """Primary-tier 6′ tectonic-zone grid (in-process cache)."""
    key = f"pb2002:{Path(src_dir).resolve()}"
    cache = _ZONE_HOME_CACHE.get(key)
    if cache is None:
        cache = zone_home_grid(load_tectonic_zones(src_dir))
        _ZONE_HOME_CACHE[key] = cache
    return cache


def _tier_zones(tier: str, src_dir: str | Path = PB2002_SRC_DIR) -> np.ndarray:
    if tier == HOME_TIER:
        return _zone_home(src_dir)
    key = (tier, str(Path(src_dir).resolve()))
    cache = _ZONE_MODE_CACHE.get(key)
    if cache is None:
        home = _zone_home(src_dir)
        cache = aggregate_mode(
            home, aggregation_factor(tier, HOME_TIER), nodata=NODATA
        )
        _ZONE_MODE_CACHE[key] = cache
    return cache


ZONE_INGESTION_RULE = (
    "源 = Kagan et al. (2010) 五区构造区划图（Bird .GRD 节点点样本，"
    "1801×3601，行北→南、列西→东 0..360，末列为日期线卷绕副本已去重）；"
    "6′ 主档 = 胞心最近节点采样（胞心与 4 相邻节点等距，取 4 角众数、"
    "平票取小源码，含经度卷绕与 ±90 端行）；源码 +1 平移入层（1=板块"
    "内部 … 5=海沟，0 = 缺测哨兵——源全网格有值不出现，validity 恒 1）；"
    "粗档 = 主档众数聚合（kernels.aggregate_mode，nodata=0 不参与、"
    "平票取小、≥1 有效子胞即有值）"
)


def build_pb2002_zone_class(entry: LayerEntry, tier: str) -> xr.DataArray:
    """Build this tier's stress_kinematics__pb2002_zone_class (uint8 categorical layer)."""
    values = _tier_zones(tier)
    lat, lon = tier_centers(tier)
    attrs = {
        "units": entry.unit,
        "long_name": (
            "构造区分类（Kagan et al. 2010 五区方案：板块内部/活动大陆/"
            "慢速扩张脊/快速扩张脊/海沟，基于 Bird 2003 PB2002）"
        ),
        "source": entry.source,
        "doi": KAGAN2010_DOI,
        "source_url": PB2002_URL,
        "license_note": LICENSE_NOTE,
        "native_res": entry.native_res,
        "resampling": entry.resampling,
        "visibility": entry.visibility,
        "category_encoding": json.dumps(
            zone_encoding(), ensure_ascii=False, sort_keys=True
        ),
        "ingestion_rule": ZONE_INGESTION_RULE,
        "source_file": ZONES_NAME,
        "nodata_semantics": (
            "无缺测：源网格全域有值（0=板块内部为真实类，源码 +1 平移后 "
            "1=板块内部；缺测哨兵 0 不出现），validity 恒 1"
        ),
    }
    arr = da.from_array(values, chunks=tier_chunks(tier))
    return xr.DataArray(
        arr, dims=("lat", "lon"), coords={"lat": lat, "lon": lon},
        name=entry.id, attrs=attrs,
    )



_PYRAMID_CACHE_DIR = REPO_ROOT / "products/.build/pb2002-distance"

_PYRAMID_CACHE: dict[str, dict[str, np.ndarray]] = {}


def _distance_pyramid(src_dir: str | Path = PB2002_SRC_DIR) -> dict[str, np.ndarray]:
    """229 boundary segments → {tier: float32 global grid} (in-process cache: the three tiers are computed only once).

    Reuses the distance engine (faults.distance_grids): hierarchical ring-band candidate pruning, recomputed exactly
    from the vector source tier by tier; dateline-crossing segments have no effect on the spherical 3D vector engine.
    """
    key = f"pb2002:{Path(src_dir).resolve()}"
    if key not in _PYRAMID_CACHE:
        records = load_pb2002_boundaries(src_dir)
        geoms = _boundary_geoms(records)
        segs, _n_drop = segment_table(geoms)
        _PYRAMID_CACHE[key] = distance_grids(
            segs, DISTANCE_CHAIN,
            workers=max(1, os.cpu_count() or 1),
            cache_dir=_PYRAMID_CACHE_DIR,
        )
    return _PYRAMID_CACHE[key]


def _boundary_geoms(records: list[dict]):
    """Boundary records → a list of LineStrings (shared by the engine and fidelity)."""
    from shapely.geometry import LineString

    return [LineString(r["pts"]) for r in records]


VALUE_CONVENTION = (
    "像元中心到最近板块边界线段（PB2002 229 段）的球面大圆距离（km，"
    "IUGG authalic R=6371.0071810 km，与像元面积层同球）；逐档从矢量源"
    "精确重算（非跨档降采样）——候选剪枝用层级环带方案（父档中心距离环带 "
    "±2×胞半对角线包含子档全体像元的 argmin，数学引理保证不漏候选），"
    "保真报告以独立航空公式抽样全段暴力比对复核"
)


def build_pb2002_boundary_distance(entry: LayerEntry, tier: str) -> xr.DataArray:
    """Build this tier's stress_kinematics__pb2002_boundary_distance (float32, valued everywhere)."""
    values = _distance_pyramid()[tier]
    lat, lon = tier_centers(tier)
    attrs = {
        "units": entry.unit,
        "long_name": "到最近板块边界（PB2002）的球面大圆距离",
        "source": entry.source,
        "doi": BIRD2003_DOI,
        "source_url": PB2002_URL,
        "license_note": LICENSE_NOTE,
        "native_res": entry.native_res,
        "resampling": entry.resampling,
        "visibility": entry.visibility,
        "value_convention": VALUE_CONVENTION,
        "source_file": BOUNDARIES_NAME,
        "nodata_semantics": "无缺测：距离场全域有值（免掩膜）",
    }
    arr = da.from_array(values, chunks=tier_chunks(tier))
    return xr.DataArray(
        arr, dims=("lat", "lon"), coords={"lat": lat, "lon": lon},
        name=entry.id, attrs=attrs,
    )



PB2002_FIDELITY_LAYERS = {
    PLATE_ID_LAYER_ID: {"kind": "plate_class"},
    ZONE_CLASS_LAYER_ID: {"kind": "zone_class"},
    BOUNDARY_DISTANCE_LAYER_ID: {"kind": "distance"},
}

_SAMPLE_SEED = 20260913
_N_SAMPLES = 120
_DIST_TOL_KM = 1e-3
_DIST_REL_TOL = 3e-7


def build_pb2002_fidelity_report(store_dir, layer_id: str, tiers: list[str],
                                  src_dir=None) -> dict:
    """Fidelity report.

    plate_id criteria (isomorphic + exhaustive coverage): ① store vs. source recompute bit-identical;
    ② coarse tiers = mode aggregation of the primary-tier store values; ③ value range ⊆ the encoding table; ④ primary-tier
    uncovered pixels = 0 (hard criterion, exhaustive coverage) + class distribution/overlap totals (report-mode).

    zone_class criteria: ①②③④ isomorphic (code 1..5); ⑤ primary-tier class share vs. source-node
    class share drift (report-mode, the approximate class-quantity conservation of mode sampling).

    boundary_distance criteria (channel-isomorphic): ⑥ stratified random pixel sampling per tier (uniform +
    near-field <50 km) brute-forced against the independent haversine formula over all segments, tol = max(1e-3 km, 3e-7·|d|)
    (hard criterion); ⑦ globally finite and ≥ 0 (hard criterion, mask-free semantics); ⑧ distance distribution statistics
    (report-mode).
    """
    src_dir = src_dir or PB2002_SRC_DIR
    kind = PB2002_FIDELITY_LAYERS[layer_id]["kind"]
    checks: list[dict] = []

    if kind in ("plate_class", "zone_class"):
        if kind == "plate_class":
            home, overlap_cells, uncovered_cells = _home_tier_classes(src_dir)
            valid_codes = np.array(sorted(PLATE_CODES.values()))
        else:
            home = _zone_home(src_dir)
            valid_codes = np.arange(1, len(ZONE_LABELS) + 1)
            overlap_cells = uncovered_cells = None
        values: dict[str, np.ndarray] = {}
        with xr.open_datatree(store_dir, engine="zarr", chunks={}) as tree:
            for tier in tiers:
                node = tree[f"/{tier}"]
                if layer_id not in node.ds.data_vars:
                    raise KeyError(
                        f"store {store_dir} 的 {tier} 档不含 {layer_id}（先 build 再 fidelity）"
                    )
                values[tier] = node.ds[layer_id].values

        for tier in tiers:
            expect = (
                _tier_classes(tier, src_dir) if kind == "plate_class"
                else _tier_zones(tier, src_dir)
            )
            n_bad = int((values[tier] != expect).sum())
            checks.append({
                "name": f"{tier}: 与源栅格化/采样重算逐位一致（uint8 位级）",
                "status": "pass" if n_bad == 0 else "FAIL",
                "summary": f"不一致像元 {n_bad}（期望 0）",
            })

        for tier in tiers:
            if tier == HOME_TIER:
                continue
            expect = aggregate_mode(
                values[HOME_TIER], aggregation_factor(tier, HOME_TIER),
                nodata=NODATA,
            )
            n_bad = int((values[tier] != expect).sum())
            checks.append({
                "name": f"{tier}: 类别众数一致性（= 主档 {HOME_TIER} store 值众数聚合）",
                "status": "pass" if n_bad == 0 else "FAIL",
                "summary": f"不一致像元 {n_bad}（期望 0）",
            })

        for tier in tiers:
            v = values[tier]
            bad = int((~np.isin(v, np.append(valid_codes, NODATA))).sum())
            checks.append({
                "name": f"{tier}: 值域 ⊆ 编码表（1..{len(valid_codes)}）",
                "status": "pass" if bad == 0 else "FAIL",
                "summary": f"非法值像元 {bad}（期望 0）",
            })
            counts = np.bincount(v.ravel(), minlength=len(valid_codes) + 1)
            if kind == "plate_class":
                dist = ", ".join(
                    f"{label}={counts[c]}" for label, c in
                    sorted(PLATE_CODES.items(), key=lambda kv: kv[1])
                )
            else:
                dist = ", ".join(
                    f"{ZONE_LABELS[c - 1]}={counts[c]}"
                    for c in range(1, len(ZONE_LABELS) + 1)
                )
            checks.append({
                "name": f"{tier}: 类别分布（报告制）",
                "status": "report",
                "summary": f"覆盖 {counts[1:].sum()}/{v.size}；{dist}",
            })

        if kind == "plate_class":
            checks.append({
                "name": f"{HOME_TIER}: 穷尽覆盖（未覆盖像元 = 0）",
                "status": "pass" if uncovered_cells == 0 else "FAIL",
                "summary": f"未覆盖像元 {uncovered_cells}（期望 0；52 多边形平铺全球）",
            })
            checks.append({
                "name": f"{HOME_TIER}: 板块重叠总量（报告制）",
                "status": "report",
                "summary": (
                    f"多板块重叠像元 {overlap_cells}（规则：面积较小者胜；"
                    "源为共享数字化边界，预期 0）"
                ),
            })
        else:
            src_g = load_tectonic_zones(src_dir)
            src_counts = np.bincount(src_g.ravel(), minlength=5)
            src_share = src_counts / src_g.size
            home_counts = np.bincount(home.ravel(), minlength=6)
            home_share = home_counts[1:] / home.size
            drift = home_share - src_share
            worst = int(np.argmax(np.abs(drift)))
            checks.append({
                "name": f"{HOME_TIER}: 类别份额 vs 源节点份额（报告制）",
                "status": "report",
                "summary": (
                    "; ".join(
                        f"{ZONE_LABELS[c]}: 源 {src_share[c]:.4%} → 层 {home_share[c]:.4%}"
                        for c in range(5)
                    )
                    + f"；最大漂移 {ZONE_LABELS[worst]} {drift[worst]:+.4%}"
                    "（半档相位最近节点采样的类量近似守恒）"
                ),
            })

    else:  # distance
        records = load_pb2002_boundaries(src_dir)
        (segs, n_drop) = segment_table(_boundary_geoms(records))
        rng = np.random.default_rng(_SAMPLE_SEED)
        with xr.open_datatree(store_dir, engine="zarr", chunks={}) as tree:
            for tier in tiers:
                node = tree[f"/{tier}"]
                if layer_id not in node.ds.data_vars:
                    raise KeyError(
                        f"store {store_dir} 的 {tier} 档不含 {layer_id}（先 build 再 fidelity）"
                    )
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
                    indep = independent_min_distance_km(
                        float(lat[r]), float(lon[c]), segs
                    )
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
                        + (f"；最差点 {worst[:2]}（独立 {worst[2]:.6f} vs store "
                           f"{worst[3]:.6f} km，Δ={worst[4]:.2e}/tol={worst[5]:.2e}）"
                           if worst else "")
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
