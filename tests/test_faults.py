"""Tests: spherical great-circle distance kernel (cross-validated against an independent
aviation formula) + hierarchical annulus engine
(bitwise comparison against the brute-force all-segments path) + line rasterizer
(overlap rule / missing-data semantics) + sidecars.

Checks test external behavior only:
- distance kernel: midpoint 0 / endpoint extension / perpendicular cross product / antipodal
  endpoint branch, random geometries vs the independent aviation formula (haversine + azimuth,
  deliberately heterogeneous implementation) max|Δ| ≤ 1e-6 km;
- annulus engine: (1) random sparse segments at all tiers == brute-force min over all segments
  (direct verification that the lemma is lossless, tolerance 1e-6 km — the level of
  floating-point summation-order differences); (2) a 1° cell task agrees with the brute-force
  path across all 5 tiers (including 30″);
- rasterizer: a line burns through the pixels it crosses, overlap goes to the shorter fault
  (out-of-order input gives the same result), a missing
  slip_type (code 0) is not burned and not counted;
- reader: row count / vocabulary / dateline crossing / non-LineString contract;
- real source: baseline (13696/23-class vocabulary/segment table contract) + engine cell-task
  spot checks on the real source;
- e2e (synthetic source): dual channel into the store → structural validation PASS →
  manifest encoding table/
  validity link → fidelity report PASS (including independent distance-sampling comparison).
"""

import json
from pathlib import Path

import numpy as np
import pytest

from cubebuild.faults import (
    _ANNULUS_KM,
    _TIER_CHAIN,
    DISTANCE_LAYER_ID,
    EXPECTED_ROWS,
    EXPECTED_SLIP_NAN,
    GEM_SRC_DIR,
    SLIP_CLASS_LAYER_ID,
    SLIP_TYPE_CODES,
    _distance_cell,
    _point_seg_dists,
    _unit_vec,
    category_encoding,
    distance_grids,
    independent_min_distance_km,
    load_gem_faults,
    rasterize_fault_classes,
    segment_table,
)
from cubebuild.grids import grid_shape, tier_centers, tier_fraction
from cubebuild.kernels import aggregate_mode
from cubebuild.pixel_area import EARTH_RADIUS_KM

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_GPKG = (GEM_SRC_DIR / "geopackage/gem_active_faults_harmonized.gpkg").is_file()
requires_gem = pytest.mark.skipif(not REAL_GPKG, reason="GEM 活动断层源未集齐")

KM_PER_DEG = 2.0 * np.pi * EARTH_RADIUS_KM / 360.0


@pytest.fixture(autouse=True)
def _clear_fault_caches():
    """Module-level caches isolated across tests (synthetic/real source switching does not read stale data)."""
    import cubebuild.faults as f

    caches = (f._PYRAMID_CACHE, f._CLASS_CACHE, f._MODE_CACHE)
    for c in caches:
        c.clear()
    yield
    for c in caches:
        c.clear()


@pytest.fixture(autouse=True)
def _no_sidecars(monkeypatch):
    import cubebuild.cli

    monkeypatch.setattr(cubebuild.cli, "SIDECARS", {})



def test_slip_type_encoding_table():
    """Encoding table: 23 classes, codes 1..23 in lexicographic order, category_encoding round-trip consistent."""
    assert len(SLIP_TYPE_CODES) == 23
    assert sorted(SLIP_TYPE_CODES.values()) == list(range(1, 24))
    enc = category_encoding()
    assert enc["1"] == "Anticline" and enc["23"] == "Syncline"
    assert enc[str(SLIP_TYPE_CODES["Blind Thrust"])] == "Blind Thrust"
    for label, code in SLIP_TYPE_CODES.items():
        assert enc[str(code)] == label



def _fake_gdf(n, slip_types, geoms=None):
    import geopandas as gpd
    from shapely.geometry import LineString

    if geoms is None:
        geoms = [LineString([(i, 0.0), (i + 0.1, 0.1)]) for i in range(n)]
    return gpd.GeoDataFrame(
        {"slip_type": slip_types}, geometry=geoms, crs="EPSG:4326"
    )


def _fake_src(tmp_path):
    p = tmp_path / "geopackage/gem_active_faults_harmonized.gpkg"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("", encoding="utf-8")
    return tmp_path


def _patch_rows(monkeypatch, gdf):
    import geopandas as gpd

    monkeypatch.setattr(gpd, "read_file", lambda _p: gdf)


def test_load_gem_faults_rejects_wrong_rows(tmp_path, monkeypatch):
    _patch_rows(monkeypatch, _fake_gdf(2, ["Normal"] * 2))
    with pytest.raises(ValueError, match="行数"):
        load_gem_faults(_fake_src(tmp_path))


def test_load_gem_faults_rejects_unknown_slip_type(tmp_path, monkeypatch):
    n = EXPECTED_ROWS
    slips = ["Normal"] * (n - 1) + ["Mystery Fault"]
    _patch_rows(monkeypatch, _fake_gdf(n, slips))
    with pytest.raises(ValueError, match="超出编码表"):
        load_gem_faults(_fake_src(tmp_path))


def test_load_gem_faults_rejects_slip_nan_drift(tmp_path, monkeypatch):
    """The count of missing slip_type drifts (319 → 318) and is rejected at read time — a source change requires re-evaluation."""
    n = EXPECTED_ROWS
    slips = ["Normal"] * (n - 318) + [None] * 318
    _patch_rows(monkeypatch, _fake_gdf(n, slips))
    with pytest.raises(ValueError, match="缺失 318"):
        load_gem_faults(_fake_src(tmp_path))


def _slips_with_nan(n_classified):
    """n Normal + 319 None (satisfies the NaN contract, letting the subsequent geometry contract concern pass first)."""
    return ["Normal"] * n_classified + [None] * EXPECTED_SLIP_NAN


def test_load_gem_faults_rejects_dateline_crossing(tmp_path, monkeypatch):
    from shapely.geometry import LineString

    crossing = LineString([(179.0, 0.0), (-179.0, 0.5)])
    slips = _slips_with_nan(EXPECTED_ROWS - EXPECTED_SLIP_NAN - 1) + ["Normal"]
    gdf = _fake_gdf(
        EXPECTED_ROWS, slips,
        geoms=[crossing] + list(_fake_gdf(EXPECTED_ROWS - 1, slips[:-1]).geometry),
    )
    _patch_rows(monkeypatch, gdf)
    with pytest.raises(ValueError, match="跨日期线"):
        load_gem_faults(_fake_src(tmp_path))


def test_load_gem_faults_rejects_multilinestring(tmp_path, monkeypatch):
    from shapely.geometry import MultiLineString

    ml = MultiLineString([[(0, 0), (1, 1)], [(2, 2), (3, 3)]])
    slips = _slips_with_nan(EXPECTED_ROWS - EXPECTED_SLIP_NAN - 1) + ["Normal"]
    gdf = _fake_gdf(
        EXPECTED_ROWS, slips,
        geoms=[ml] + list(_fake_gdf(EXPECTED_ROWS - 1, slips[:-1]).geometry),
    )
    _patch_rows(monkeypatch, gdf)
    with pytest.raises(ValueError, match="LineString"):
        load_gem_faults(_fake_src(tmp_path))



def _equator_segment():
    """Segment table for the equator segment A=(0°,0°) → B=(0°,10°)."""
    from shapely.geometry import LineString

    return segment_table([LineString([(0.0, 0.0), (10.0, 0.0)])])[0]


def test_segment_distance_known_cases():
    segs = _equator_segment()
    km = KM_PER_DEG

    def d(lat, lon):
        p = _unit_vec(np.array([lat]), np.array([lon]))
        return float(_point_seg_dists(p, segs)[0, 0])

    assert d(0.0, 5.0) == pytest.approx(0.0, abs=1e-9)
    assert d(0.0, 12.0) == pytest.approx(2.0 * km, abs=1e-6)
    assert d(0.0, -3.0) == pytest.approx(3.0 * km, abs=1e-6)
    assert d(10.0, 5.0) == pytest.approx(10.0 * km, abs=1e-6)
    assert d(0.0, -175.0) == pytest.approx(175.0 * km, abs=1e-4)


def test_segment_distance_vs_independent_random():
    """Random segments + random points: engine (3D vector path) vs the independent aviation formula over all segments."""
    rng = np.random.default_rng(42)
    from shapely.geometry import LineString

    geoms = []
    for _ in range(30):
        lon1 = rng.uniform(-160, 160)
        lon2 = lon1 + rng.uniform(-20, 20)
        lat1 = rng.uniform(-75, 75)
        lat2 = lat1 + rng.uniform(-15, 15)
        geoms.append(LineString([(lon1, lat1), (lon2, lat2)]))
    segs, n_drop = segment_table(geoms)
    assert n_drop == 0
    for _ in range(60):
        lat, lon = rng.uniform(-89, 89), rng.uniform(-180, 180)
        p = _unit_vec(np.array([lat]), np.array([lon]))
        engine = float(_point_seg_dists(p, segs)[0].min())
        indep = independent_min_distance_km(float(lat), float(lon), segs)
        assert abs(engine - indep) <= 1e-6, (lat, lon, engine, indep)



def _random_segments(n, seed):
    """Random short-segment set (2–5 vertices, step ≤ 3°, coordinate domain constrained within
    ±170/±78 to guarantee drift stays in bounds) — covering near-field/crossing/far-field
    forms."""
    from shapely.geometry import LineString

    rng = np.random.default_rng(seed)
    geoms = []
    for _ in range(n):
        k = int(rng.integers(2, 6))
        lon = rng.uniform(-150, 150)
        lat = rng.uniform(-70, 70)
        pts = []
        for _ in range(k):
            pts.append((lon, lat))
            lon += rng.uniform(-3, 3)
            lat += rng.uniform(-2, 2)
        geoms.append(LineString(pts))
    return geoms


def _brute_force_grid(segs, tier):
    """Independent brute-force path: all pixel centers at the tier → min over all segments (no
    candidate pruning).

    Chunking guardrail: the intermediate array ≈ 10×(block, n_seg) float64 — block 500k × 40
    segments
    ≈ 1.6GB peak (within the WSL 40GB budget); block must scale with the segment count (see
    the real-source chunking in _min_dists_chunked).
    """
    lat, lon = tier_centers(tier)
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    p = _unit_vec(la.ravel(), lo.ravel())
    out = np.empty(p.shape[0], dtype=np.float64)
    for s in range(0, p.shape[0], 500_000):
        e = min(s + 500_000, p.shape[0])
        out[s:e] = _point_seg_dists(p[s:e], segs).min(axis=1)
    return out.reshape(la.shape)


def _min_dists_chunked(points: np.ndarray, segs, block: int = 32) -> np.ndarray:
    """Chunked brute-force min (memory guardrail: the intermediate array ≈ 10×(block, n_seg)
    float64).

    Real source 141k segments × block 32 ≈ 360MB peak — within the WSL 40GB budget. The
    unchunked
    (14400, 141298) matrix ≈ 160GB is an OOM bomb (proven by WSL crash,
    user hard constraint: long-task peak ≤40GB).
    """
    out = np.empty(points.shape[0], dtype=np.float64)
    for s in range(0, points.shape[0], block):
        e = min(s + block, points.shape[0])
        out[s:e] = _point_seg_dists(points[s:e], segs).min(axis=1)
    return out


def test_distance_grids_match_bruteforce_sparse():
    """Random sparse segments (40) over the 1°/30′ full grid == brute-force min over all segments (lemma-lossless verification)."""
    geoms = _random_segments(40, seed=7)
    segs, _ = segment_table(geoms)
    grids = distance_grids(segs, chain=("1deg", "30min"), workers=1)
    for tier in ("1deg", "30min"):
        got = grids[tier]
        expect = _brute_force_grid(segs, tier).astype(np.float32)
        np.testing.assert_allclose(
            got, expect, atol=1e-6, rtol=0,
            err_msg=f"{tier} 环带引擎 vs 暴力路径不一致（引理被破坏）",
        )


def test_distance_cell_matches_bruteforce_all_tiers():
    """A single 1° cell full-chain task (including 30″) vs the brute-force path — 3 cells sampled (near-field/far-field mixture)."""
    geoms = _random_segments(25, seed=11)
    segs, _ = segment_table(geoms)
    from shapely.geometry import LineString

    geoms2 = geoms + [LineString([(0.2, 5.2), (1.4, 5.8), (2.1, 6.3)])]
    segs2, _ = segment_table(geoms2)
    for (i, j) in [(95, 180), (10, 60), (150, 350)]:
        res = _distance_cell(i, j, segs2, _TIER_CHAIN)
        for tier in _TIER_CHAIN:
            f = int(round(1 / tier_fraction(tier)))
            block = res[tier]
            assert block.shape == (f, f)
            lat, lon = tier_centers(tier)
            la, lo = np.meshgrid(
                lat[i * f:(i + 1) * f], lon[j * f:(j + 1) * f], indexing="ij"
            )
            pc = _unit_vec(la.ravel(), lo.ravel())   # (f², 3) 2-D
            expect = _point_seg_dists(pc, segs2).min(axis=1)
            np.testing.assert_allclose(
                block.ravel(), expect, atol=1e-6, rtol=0,
                err_msg=f"胞 ({i},{j}) {tier} 环带引擎 vs 暴力不一致",
            )


def test_annulus_widths_monotone():
    """The annulus half-width decreases monotonically as tiers refine (engine geometry self-check)."""
    ws = [_ANNULUS_KM[t] for t in _TIER_CHAIN]
    assert all(a > b for a, b in zip(ws, ws[1:]))



def test_rasterize_fault_classes_burn_and_overlap():
    """3′ tier: a line crossing a pixel burns its trace; overlapping pixels go to the shorter
    fault (out-of-order gives the same result);
    a missing slip_type (code 0) is not burned and does not participate in overlap counting.

    The geometry avoids cell boundaries: horizontal line y=0.06 (row [0.05,0.10] center 0.075),
    x∈[0.01,0.29] (column centers 0.025..0.275); vertical line x=0.13 (column [0.10,0.15]
    center 0.125), y∈[−0.04,0.14] (row centers −0.025..0.125); the crossing point
    (0.13, 0.06) falls in the cell (center 0.075, 0.125) → that cell overlaps, and the shorter
    one
    (vertical line 0.18° < horizontal line 0.28°) wins.
    """
    from shapely.geometry import LineString

    long_line = LineString([(0.01, 0.06), (0.29, 0.06)])
    short_cross = LineString([(0.13, -0.04), (0.13, 0.14)])
    classes, overlap = rasterize_fault_classes(
        [long_line, short_cross], np.array([5, 9], dtype=np.uint8), "3min"
    )
    classes2, overlap2 = rasterize_fault_classes(
        [short_cross, long_line], np.array([9, 5], dtype=np.uint8), "3min"
    )
    np.testing.assert_array_equal(classes, classes2)
    assert overlap == overlap2 >= 1
    r_cross, c_cross = 1801, 3602
    assert classes[r_cross, c_cross] == 9
    assert classes[1801, 3600] == 5
    assert classes[1799, 3602] == 9
    classes3, overlap3 = rasterize_fault_classes(
        [long_line], np.array([0], dtype=np.uint8), "3min"
    )
    assert not classes3.any() and overlap3 == 0


def test_rasterize_fault_classes_mode_aggregation():
    """Coarse-tier mode aggregation: a 30″ synthetic primary tier → 3′ (factor 6) is semantically correct."""
    from shapely.geometry import LineString

    line = LineString([(0.0, 0.0), (0.1, 0.1)])
    classes, _ = rasterize_fault_classes(
        [line], np.array([4], dtype=np.uint8), "30sec"
    )
    coarse = aggregate_mode(classes, 6, nodata=0)
    assert coarse.shape == grid_shape("3min")
    assert set(np.unique(coarse)) <= {0, 4}
    assert (coarse != 0).sum() > 0
    blocks = classes.reshape(
        coarse.shape[0], 6, coarse.shape[1], 6
    ).any(axis=(1, 3))
    np.testing.assert_array_equal(coarse != 0, blocks)



@requires_gem
def test_real_gem_baselines():
    """Disk evidence made contractual: 13696 rows, 23-class vocabulary + 319 NaN, LineString,
    segment-table segment angle < 90°."""
    gdf, codes = load_gem_faults()
    assert len(gdf) == EXPECTED_ROWS
    assert (codes == 0).sum() == EXPECTED_SLIP_NAN
    segs, n_drop = segment_table(list(gdf.geometry))
    A, B, NV, sinAB, cosAB = segs[0], segs[1], segs[2], segs[3], segs[4]
    assert cosAB.min() > 0.0
    n_vertices = int(sum(len(g.coords) for g in gdf.geometry))
    assert len(A) + n_drop == n_vertices - EXPECTED_ROWS
    assert n_vertices == 154994


@requires_gem
def test_real_distance_cell_spotchecks():
    """Real-source single-cell task spot checks (memory-guardrail version): 2 cells × {1°, 6′,
    30″} tiers, each tier
    comparing ≤600 random pixels against the brute-force all-segments path in chunks (real source
    with 141k segments).

    Full-tier all-pixel bitwise consistency is already covered by the synthetic-source tests
    (few segments allow full comparison); this test
    only verifies the engine's correctness at the real-source candidate scale, with
    deterministic sampling (seed=13)."""
    gdf, _codes = load_gem_faults()
    segs, _ = segment_table(list(gdf.geometry))
    rng = np.random.default_rng(13)
    for (i, j) in [(60, 120), (5, 350)]:
        res = _distance_cell(i, j, segs, _TIER_CHAIN)
        for tier in ("1deg", "6min", "30sec"):
            f = int(round(1 / tier_fraction(tier)))
            block = res[tier]
            assert block.shape == (f, f)
            n = block.size
            idx = rng.choice(n, size=min(600, n), replace=False)
            lat, lon = tier_centers(tier)
            la, lo = np.meshgrid(
                lat[i * f:(i + 1) * f], lon[j * f:(j + 1) * f], indexing="ij"
            )
            pts = _unit_vec(la.ravel()[idx], lo.ravel()[idx])
            expect = _min_dists_chunked(pts, segs)
            np.testing.assert_allclose(
                block.ravel()[idx], expect, atol=1e-6, rtol=0,
                err_msg=f"实源胞 ({i},{j}) {tier}",
            )


@requires_gem
def test_real_gem_sidecar_roundtrip(tmp_path):
    """Sidecar GeoParquet round-trip: row count/attribute columns/encoding column/geometry/CRS/visibility."""
    import geopandas as gpd
    from cubebuild.sidecars import export_gem_active_faults

    rec = export_gem_active_faults(tmp_path)
    assert rec["id"] == "gem_active_faults"
    assert rec["visibility"] == "internal"
    assert rec["n_features"] == EXPECTED_ROWS
    assert rec["n_unclassified"] == EXPECTED_SLIP_NAN
    back = gpd.read_parquet(tmp_path / "gem_active_faults.parquet")
    assert len(back) == EXPECTED_ROWS
    assert back.crs.to_epsg() == 4326
    for col in ("slip_type", "average_dip", "average_rake", "net_slip_rate",
                "upper_seis_depth", "lower_seis_depth", "catalog_id", "name",
                "reference", "epistemic_quality", "accuracy", "notes"):
        assert col in back.columns
    assert (back["slip_type_code"] == back["slip_type"].map(
        SLIP_TYPE_CODES).fillna(0).astype("uint8")).all()



_E2E = """
version: test
status: test
tiers: [1deg, 30sec]
layers:
  - id: stress_kinematics__gem_fault_slip_class
    source: gem-active-faults（harmonized gpkg，13696 断裂）
    native_res: vector-line
    home_tier: 30sec
    tiers: [1deg, 30sec]
    dtype: uint8
    unit: category（slip_type：正断/逆断/走滑等）
    resampling: mode
    mask: validity
    visibility: internal
  - id: stress_kinematics__gem_fault_distance
    source: gem-active-faults
    native_res: vector-line
    home_tier: 30sec
    tiers: [1deg, 30sec]
    dtype: float32
    unit: km（球面大圆距离）
    resampling: none（逐档从矢量源精确重算）
    mask: 免（距离场全域有值）
    visibility: internal
"""


def _synthetic_source(monkeypatch, n=36, seed=3):
    """A synthetic GEM source (few segments, e2e completes in minutes) injected into load_gem_faults."""
    from shapely.geometry import LineString

    rng = np.random.default_rng(seed)
    slips = []
    geoms = []
    labels = sorted(SLIP_TYPE_CODES)
    for k in range(n):
        lon, lat = rng.uniform(-150, 150), rng.uniform(-60, 60)
        pts = [(lon, lat)]
        for _ in range(int(rng.integers(1, 4))):
            lon += rng.uniform(-2, 2)
            lat += rng.uniform(-2, 2)
            pts.append((lon, lat))
        geoms.append(LineString(pts))
        slips.append(labels[k % len(labels)])
    slips[0] = None
    gdf = _fake_gdf(n, slips, geoms=geoms)
    monkeypatch.setattr(
        "cubebuild.faults.load_gem_faults", lambda _src=None: (gdf, np.array(
            [SLIP_TYPE_CODES.get(s, 0) for s in slips], dtype=np.uint8))
    )
    monkeypatch.setattr(
        "cubebuild.sidecars.load_gem_faults", lambda _src=None: (gdf, np.array(
            [SLIP_TYPE_CODES.get(s, 0) for s in slips], dtype=np.uint8))
    )
    return gdf


def test_faults_e2e(tmp_path, monkeypatch):
    """Synthetic-source dual-channel full chain: build → structural validation PASS → manifest
    (encoding table/
    validity link/internal visibility) → fidelity report PASS (independent distance-sampling
    comparison)."""
    import cubebuild.cli
    from cubebuild.faults import build_faults_fidelity_report
    from cubebuild.manifest import read_manifest

    _synthetic_source(monkeypatch)

    mapping = tmp_path / "mini.yaml"
    mapping.write_text(_E2E, encoding="utf-8")
    out = tmp_path / "cube.zarr"
    rc = cubebuild.cli.main([
        "build", "--mapping", str(mapping), "--out", str(out),
        "--reports", str(tmp_path / "reports"),
        "--sidecars", str(tmp_path / "sidecars"),
    ])
    assert rc == 0

    manifest = read_manifest(out)
    entries = {m["id"]: m for m in manifest["layers"]}
    assert set(entries) == {SLIP_CLASS_LAYER_ID, DISTANCE_LAYER_ID}

    cls = entries[SLIP_CLASS_LAYER_ID]
    assert cls["dtype"] == "uint8" and cls["visibility"] == "internal"
    assert cls["validity_mask"] == f"{SLIP_CLASS_LAYER_ID}__validity"
    enc = json.loads(cls["category_encoding"])
    assert enc[str(SLIP_TYPE_CODES["Normal"])] == "Normal"
    assert "较短断裂" in cls["rasterization_rule"]

    dist = entries[DISTANCE_LAYER_ID]
    assert dist["dtype"] == "float32" and dist["visibility"] == "internal"
    assert dist["coverage"] == {"1deg": 1.0, "30sec": 1.0}

    import xarray as xr
    with xr.open_datatree(out, engine="zarr", chunks={}) as dt:
        v30 = dt["/30sec"][DISTANCE_LAYER_ID].values
        assert np.isfinite(v30).all() and (v30 >= 0).all()
        c30 = dt["/30sec"][SLIP_CLASS_LAYER_ID].values
        m30 = dt["/30sec"][f"{SLIP_CLASS_LAYER_ID}__validity"].values
        assert np.array_equal(m30, (c30 != 0).astype(np.uint8))

    for lid in (SLIP_CLASS_LAYER_ID, DISTANCE_LAYER_ID):
        report = build_faults_fidelity_report(out, lid, ["1deg", "30sec"])
        assert report["result"] == "PASS", report["checks"]
        fails = [c for c in report["checks"] if c["status"] == "FAIL"]
        assert not fails
