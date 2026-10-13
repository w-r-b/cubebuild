"""Tests: the sediment group's GST1 + Global Basins dual channel.

The criteria test external behavior only (same-style criteria):
- GST1 reader: column-order trap (header says Lon Lat meter / data is Lat Lon
  meter, columns taken by position) + first row (-90,-180) contract + grid
  geometry (constant latitude within a row / inner-loop stepping in longitude)
  + ±180 periodic duplicate column + constant polar row + rejection paths;
- GST1 conservative kernel: constant field bit-exact round-trip + linear field
  independent sin-weight analytic solution + Voronoi cell edges (polar row
  half-weight clamping);
- Real-source contract: GST1 full grid / value range / polar values / landmark
  node values; basins 768 polygons = 764 basins / 4 split polygons / 1 invalid
  geometry repair / 8 type vocabulary classes;
- Basin classes: 3′ home-tier rasterized landmarks (real data pre-placed) +
  coverage ~25.39% + overlapping pixels (3 pairs of digitized shared
  boundaries);
- Distance: regional distance semantics (0 inside the domain / >0 outside) +
  cross-validation against an independent great-circle formula;
- e2e (synthetic sources): three layers into the store -> structure validation
  PASS -> manifest (encoding table / validity link / internal visibility) ->
  fidelity PASS -> sidecar parquet.
"""

import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from cubebuild.sediment import (
    BASIN_DISTANCE_LAYER_ID,
    BASIN_CLASS_LAYER_ID,
    BASIN_TYPE_CODES,
    BASINS_HOME_TIER,
    BASINS_SRC_DIR,
    BASINS_SHP_NAME,
    EXPECTED_BASINS,
    EXPECTED_INVALID,
    EXPECTED_POLYGONS,
    EXPECTED_SPLIT_BASINS,
    GST1_EXPECTED,
    GST1_LAYER_ID,
    GST1_SRC_DIR,
    XYZ_NAME,
    basin_type_encoding,
    load_gst1,
)
from cubebuild.grids import grid_shape, tier_centers
from cubebuild.kernels import conservative_overlap_mean
from cubebuild.gravmag import RasterSource

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_GST1 = (GST1_SRC_DIR / XYZ_NAME).is_file()
requires_gst1 = pytest.mark.skipif(not REAL_GST1, reason="GST1 源未集齐")
REAL_BASINS = (BASINS_SRC_DIR / BASINS_SHP_NAME).is_file()
requires_basins = pytest.mark.skipif(not REAL_BASINS, reason="Global Basins 源未集齐")

_L3, _O3 = tier_centers(BASINS_HOME_TIER)


def _cell_at(tier: str, lat: float, lon: float) -> tuple[int, int]:
    """Landmark point -> nearest tier cell center (r, c) (immune to floating-point pitfalls: argmin nearest center)."""
    lats, lons = tier_centers(tier)
    return int(np.abs(lats - lat).argmin()), int(np.abs(lons - lon).argmin())


@pytest.fixture(autouse=True)
def _clear_sediment_caches(monkeypatch, tmp_path):
    """Module-level cache isolation across tests + distance-pyramid cache redirection (synthetic data is not written to the product directory)."""
    import cubebuild.sediment as m

    caches = (m._SOURCE_CACHE, m._GEOMS_CACHE, m._CLASS_HOME_CACHE,
              m._CLASS_MODE_CACHE, m._INSIDE_CACHE, m._PYRAMID_CACHE)
    for c in caches:
        c.clear()
    monkeypatch.setattr(m, "_PYRAMID_CACHE_DIR", tmp_path / "dist-cache")
    yield
    for c in caches:
        c.clear()


@pytest.fixture(autouse=True)
def _no_sidecars(monkeypatch):
    import cubebuild.cli

    monkeypatch.setattr(cubebuild.cli, "SIDECARS", {})



def test_basin_type_encoding_table():
    """Basin-type encoding table: 8 codes, dictionary order 1..8 (the encoding table round-trips through lookup)."""
    assert len(BASIN_TYPE_CODES) == 8
    assert sorted(BASIN_TYPE_CODES.values()) == list(range(1, 9))
    assert BASIN_TYPE_CODES["Backarc - Marginal Sea"] == 1
    assert BASIN_TYPE_CODES["Fold and Thrust Belt"] == 2
    assert BASIN_TYPE_CODES["Forearc"] == 3
    assert BASIN_TYPE_CODES["Foreland"] == 4
    assert BASIN_TYPE_CODES["Intracratonic"] == 5
    assert BASIN_TYPE_CODES["Passive Margin"] == 6
    assert BASIN_TYPE_CODES["Rift"] == 7
    assert BASIN_TYPE_CODES["Strike-Slip"] == 8
    assert basin_type_encoding() == {
        "1": "Backarc - Marginal Sea", "2": "Fold and Thrust Belt",
        "3": "Forearc", "4": "Foreland", "5": "Intracratonic",
        "6": "Passive Margin", "7": "Rift", "8": "Strike-Slip",
    }



_SYN_SPACING = Fraction(1, 1)


def _write_synthetic_gst1(path: Path, values_fn) -> None:
    """Write a synthetic XYZ: the header keeps the real trap form (Lon Lat meter) while the data columns are Lat Lon meter."""
    nlat, nlon = 181, 361
    lines = ["     Lon      Lat    meter"]
    for i in range(nlat):
        la = -90.0 + i * float(_SYN_SPACING)
        for j in range(nlon):
            lo = -180.0 + j * float(_SYN_SPACING)
            lines.append(f"{la:8.3f} {lo:9.3f} {values_fn(la, lo):9.2f}")
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


@pytest.fixture()
def _syn_gst1_env(tmp_path, monkeypatch):
    """Coarsened environment: GST1_EXPECTED -> a 1° node grid; returns the source directory."""
    monkeypatch.setattr(
        "cubebuild.sediment.GST1_EXPECTED",
        {"shape": (181, 361), "spacing": _SYN_SPACING},
    )
    src_dir = tmp_path / "gst1"
    src_dir.mkdir()
    return src_dir


def test_gst1_loader_column_order_trap(_syn_gst1_env):
    """Column-order trap: the header reads Lon Lat meter while the data is in fact Lat Lon meter -- columns are taken by position.

    Field v = 1000 + lat·10 + 50·cos(lon°)·cos(lat°) (equal across the ±180
    periodic seam; the polar row degenerates to a constant since cos(lat)=0,
    matching the single-point polar form of the real source): the orientation is
    determinable (linear in latitude + cosine in longitude); if columns were
    wrongly taken by header name (lat/lon swapped), the grid-geometry assertion
    (constant latitude within a row) would immediately reject it -- the reader
    contract is itself the trap catcher.
    """
    p = _syn_gst1_env / XYZ_NAME
    _write_synthetic_gst1(
        p,
        lambda la, lo: 1000.0 + la * 10.0
        + 50.0 * np.cos(np.deg2rad(lo)) * np.cos(np.deg2rad(la)),
    )
    src = load_gst1(_syn_gst1_env)
    assert src.values.shape == (181, 360)
    assert src.lon_res == _SYN_SPACING
    assert src.values[0, 0] == np.float32(100.0)
    assert src.values[90, 180] == np.float32(1050.0)
    assert src.valid.all()
    assert src.lat_edges[0] == -90.0 and src.lat_edges[1] == -89.5
    assert src.lat_edges[-1] == 90.0


def test_gst1_loader_rejects_header_change(_syn_gst1_env):
    p = _syn_gst1_env / XYZ_NAME
    p.write_text("Lat Lon meter\n-90.0 -180.0 1.0\n", encoding="ascii")
    with pytest.raises(ValueError, match="表头"):
        load_gst1(_syn_gst1_env)


def test_gst1_loader_rejects_swapped_columns(_syn_gst1_env):
    """Data columns swapped (col1=lon, col2=lat): rejected as soon as the first row (lat,lon) ≠ (-90,-180)."""
    p = _syn_gst1_env / XYZ_NAME
    lines = ["     Lon      Lat    meter"]
    for i in range(181):
        la = -90.0 + i
        for j in range(361):
            lo = -180.0 + j
            lines.append(f"{lo:9.3f} {la:8.3f} {100.0:9.2f}")
    p.write_text("\n".join(lines) + "\n", encoding="ascii")
    with pytest.raises(ValueError, match="首行"):
        load_gst1(_syn_gst1_env)


def test_gst1_loader_rejects_nonperiodic_seam(_syn_gst1_env):
    """The ±180 columns differ: rejected as soon as the periodic-duplicate assumption is broken (guarding against misaligned stitching)."""
    p = _syn_gst1_env / XYZ_NAME
    _write_synthetic_gst1(p, lambda la, lo: 1.0 if lo == 180.0 else la)
    with pytest.raises(ValueError, match="周期重复"):
        load_gst1(_syn_gst1_env)


def test_gst1_loader_rejects_nonconstant_pole_row(_syn_gst1_env):
    """The polar row is non-constant along the row: rejected as soon as the single-point-pole assumption is broken (the longitude dependence uses sin to avoid triggering the seam check first)."""
    p = _syn_gst1_env / XYZ_NAME
    _write_synthetic_gst1(
        p, lambda la, lo: np.sin(np.deg2rad(lo)) if la == -90.0 else 1.0
    )
    with pytest.raises(ValueError, match="极点行"):
        load_gst1(_syn_gst1_env)



def test_gst1_conservative_constant_and_linear(_syn_gst1_env):
    """Constant field bit-exact round-trip + linear field independent sin-weight analytic solution (a 1° target cell = weighted combination of two half cells)."""
    from cubebuild.grids import tier_fraction

    p = _syn_gst1_env / XYZ_NAME
    _write_synthetic_gst1(p, lambda la, lo: 1234.5)
    src = load_gst1(_syn_gst1_env)
    for tier in ("1deg", "30min"):
        v, W = conservative_overlap_mean(
            src.values, src.valid, src.lat_edges, src.lon_res,
            tier_fraction(tier), lon_phase=src.lon_phase,
        )
        assert np.isfinite(v).all() and (W > 0).all()
        assert np.allclose(v, 1234.5)

    _write_synthetic_gst1(p, lambda la, lo: la)
    src = load_gst1(_syn_gst1_env)
    v, _W = conservative_overlap_mean(
        src.values, src.valid, src.lat_edges, src.lon_res,
        tier_fraction("1deg"), lon_phase=src.lon_phase,
    )
    R2 = (6371.0071810) ** 2 * np.pi / 180.0
    for i in (0, 45, 90, 134, 179):
        y0, y1 = -90.0 + i, -90.0 + i + 1.0
        e_mid = y0 + 0.5
        w_a = R2 * (np.sin(np.deg2rad(e_mid)) - np.sin(np.deg2rad(y0)))
        w_b = R2 * (np.sin(np.deg2rad(y1)) - np.sin(np.deg2rad(e_mid)))
        expect = (w_a * y0 + w_b * y1) / (w_a + w_b)
        assert abs(v[i, 0] - np.float32(expect)) <= 1e-5, (i, v[i, 0], expect)



@requires_gst1
def test_real_gst1_contract_and_landmarks():
    """Real-source contract: full grid / value range / polar values / periodic column + landmark node values (pre-placed on disk)."""
    src = load_gst1()
    assert src.values.shape == (1441, 2880)
    assert src.valid.all()
    assert float(src.values.min()) == 0.0
    assert float(src.values.max()) == 20118.0
    assert src.values[0, 0] == np.float32(1356.8)
    assert src.values[-1, 0] == np.float32(1915.7)
    def node(la, lo):
        return src.values[int(round((la + 90.0) * 8)), int(round((lo + 180.0) * 8))]
    assert node(15.0, 88.0) == np.float32(8226.6)
    assert node(0.0, -160.0) == np.float32(246.5)
    assert node(32.0, 90.0) == np.float32(461.9)
    assert src.lat_edges[0] == -90.0 and src.lat_edges[1] == -89.9375
    assert src.lat_edges[-1] == 90.0
    assert src.lon_res == Fraction(1, 8)


@requires_gst1
def test_real_gst1_build_tiers():
    """Two-tier build: globally finite + validity identically 1 + landmark magnitudes + monotonicity of coarsening from 30′ to 1°."""
    from cubebuild.contract import LayerEntry
    from cubebuild.sediment import build_gst1_thickness

    entry = LayerEntry(
        id=GST1_LAYER_ID, dims=("lat", "lon"), source="gst1", native_res="0.125deg",
        native_res_deg=0.125, home_tier="30min", tiers=("1deg", "30min"),
        dtype="float32", unit="m", resampling="conservative_area_weighted",
        mask="validity", visibility="public",
    )
    v30 = build_gst1_thickness(entry, "30min").values
    v1 = build_gst1_thickness(entry, "1deg").values
    for v in (v30, v1):
        assert np.isfinite(v).all()
        assert float(v.min()) >= 0.0
    r, c = _cell_at("30min", 15.25, 88.25)
    assert v30[r, c] > 6000.0
    r, c = _cell_at("30min", 0.25, -159.75)
    assert v30[r, c] < 600.0
    assert float(v1.max()) <= float(v30.max())



@requires_basins
def test_real_basins_loader_contract():
    """768 polygons = 764 basins + 4 split polygons + 1 invalid-geometry repair + type vocabulary closure."""
    from cubebuild.sediment import load_global_basins

    gdf, codes, geoms = load_global_basins()
    assert len(gdf) == EXPECTED_POLYGONS == 768
    assert gdf["Basin UBI"].nunique() == EXPECTED_BASINS == 764
    split = [ubi for ubi, grp in gdf.groupby("Basin UBI") if len(grp) > 1]
    assert len(split) == EXPECTED_SPLIT_BASINS == 4
    assert set(gdf["Basin Name"]) >= {"Ross", "Anadyr", "Hope", "Khatyrka"}
    assert int((~gdf.geometry.is_valid).sum()) == EXPECTED_INVALID == 1
    assert gdf[~gdf.geometry.is_valid]["Basin Name"].iloc[0] == "Komandorskaya"
    assert all(g.is_valid and not g.is_empty for g in geoms)
    assert all(
        g.geom_type in ("Polygon", "MultiPolygon") for g in geoms
    )
    assert set(gdf["Basin Type"]) == set(BASIN_TYPE_CODES)
    assert (codes == gdf["Basin Type"].map(BASIN_TYPE_CODES).to_numpy()).all()
    for ubi in split:
        grp = gdf[gdf["Basin UBI"] == ubi]
        assert grp["Basin Type"].nunique() == 1


@requires_basins
def test_real_basins_class_landmarks_and_coverage():
    """3′ home-tier rasterized landmarks (real data pre-placed, interior points ≥0.9° from edges) + coverage ~25.39%."""
    from cubebuild.sediment import _home_tier_classes

    classes, overlap, uncovered = _home_tier_classes()
    assert classes.shape == grid_shape(BASINS_HOME_TIER)
    inv = {c: lab for lab, c in BASIN_TYPE_CODES.items()}

    coverage = 1.0 - uncovered / classes.size
    assert 0.2530 < coverage < 0.2545
    assert overlap <= 8

    for name, la, lo, want in [
        ("West Siberia", 60.0, 75.0, "Intracratonic"),
        ("Bengal", 18.0, 89.0, "Passive Margin"),
        ("Paris", 48.0, 2.0, "Intracratonic"),
        ("Amazonas", -3.0, -55.0, "Rift"),
        ("Middle Caspian", 42.0, 50.0, "Foreland"),
        ("Songliao", 45.0, 125.0, "Intracratonic"),
        ("Tarim", 39.0, 82.0, "Foreland"),
        ("Central Pacific", 0.0, -150.0, None),
    ]:
        r, c = _cell_at(BASINS_HOME_TIER, la, lo)
        got = inv.get(int(classes[r, c]))
        assert got == want, f"{name}: {got} ≠ {want}"


@requires_basins
def test_real_basins_distance_region_semantics():
    """Real distance pyramid: 0 inside the domain / finite positive outside / cross-validated by sampling an independent great-circle formula."""
    from cubebuild.faults import independent_min_distance_km, segment_table
    from cubebuild.sediment import _basin_rings, _basins_geoms, _distance_pyramid, _tier_inside

    pyr = _distance_pyramid()
    geoms, _codes = _basins_geoms()
    segs, _nd = segment_table(_basin_rings(geoms))
    for tier in ("1deg", "30min", "6min", "3min"):
        v = pyr[tier]
        inside = _tier_inside(tier)
        assert np.isfinite(v).all() and (v >= 0).all()
        assert (v[inside] == 0).all()
        lats, lons = tier_centers(tier)
        rng = np.random.default_rng(20260913)
        outside_idx = np.nonzero((~inside).ravel())[0]
        picks = outside_idx[rng.integers(0, outside_idx.size, 24)]
        for flat in picks:
            r, c = divmod(int(flat), v.shape[1])
            indep = independent_min_distance_km(
                float(lats[r]), float(lons[c]), segs
            )
            d = abs(indep - float(v[r, c]))
            assert d <= max(1e-3, 3e-7 * abs(indep)), (tier, r, c, indep, v[r, c])
    assert float(pyr["3min"][_cell_at("3min", 18.0, 89.0)]) == 0.0
    assert float(pyr["1deg"][_cell_at("1deg", 0.0, -150.0)]) > 1000.0



_E2E = """
version: test
status: test
tiers: [1deg, 30min, 6min, 3min]
layers:
  - id: sediment__gst1_thickness
    source: gst1（synthetic）
    native_res: 0.125deg
    home_tier: 30min
    tiers: [1deg, 30min]
    dtype: float32
    unit: m
    resampling: conservative_area_weighted
    mask: validity
  - id: sediment__global_basins_class
    source: global-basins（synthetic）
    native_res: vector-polygon
    home_tier: 3min
    tiers: [1deg, 30min, 6min, 3min]
    dtype: uint8
    unit: category（盆地类型）
    resampling: mode
    mask: validity
    visibility: internal
  - id: sediment__global_basins_distance
    source: global-basins（synthetic）
    native_res: vector-polygon
    home_tier: 3min
    tiers: [1deg, 30min, 6min, 3min]
    dtype: float32
    unit: km（球面大圆距离）
    resampling: none（逐档从矢量源精确重算）
    mask: 免
    visibility: internal
"""


def _synthetic_basins_gdf():
    """Synthetic basin source: 3 basins, 4 polygons (including a split-polygon
    pair crossing the date line) + 1 overlapping pair.

    - Large Bengal-type square (10..25, 0..15) Passive Margin (large area)
    - Small Rift square (15..18, 5..8) Rift (overlaps the large square -> the
      smaller area wins)
    - Date-line-crossing basin: two polygons (170..175, 40..45) /
      (-175..-170, 40..45) with the same UBI and the same type Rift
    """
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import Polygon

    rows = []
    def add(name, ubi, btype, minx, miny, maxx, maxy):
        rows.append({
            "Basin Name": name, "Basin UBI": ubi, "Basin Type": btype,
            "Split Poly": "No",
            "geometry": Polygon([(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)]),
        })
    add("Big", 1, "Passive Margin", 10.0, 0.0, 25.0, 15.0)
    add("SmallOverlap", 2, "Rift", 15.0, 5.0, 18.0, 8.0)
    add("CrossA", 3, "Rift", 170.0, 40.0, 175.0, 45.0)
    add("CrossB", 3, "Rift", -175.0, 40.0, -170.0, 45.0)
    rows[-2]["Split Poly"] = rows[-1]["Split Poly"] = "Yes"
    return gpd.GeoDataFrame(pd.DataFrame(rows), geometry="geometry", crs="EPSG:4326")


def _inject_synthetic(monkeypatch, tmp_path):
    import cubebuild.sediment as m
    import cubebuild.sidecars as sc

    monkeypatch.setattr(
        m, "GST1_EXPECTED", {"shape": (181, 361), "spacing": _SYN_SPACING},
    )
    src_dir = tmp_path / "gst1-syn"
    src_dir.mkdir()
    _write_synthetic_gst1(
        src_dir / XYZ_NAME,
        lambda la, lo: 500.0 + la + 0.5 * np.cos(np.deg2rad(lo)) * np.cos(np.deg2rad(la)),
    )

    orig_load_gst1 = m.load_gst1
    def fake_load_gst1(_src=None):
        return orig_load_gst1(src_dir)

    monkeypatch.setattr(m, "load_gst1", fake_load_gst1)

    def fake_load_basins(_src=None):
        gdf = _synthetic_basins_gdf()
        codes = gdf["Basin Type"].map(BASIN_TYPE_CODES).to_numpy(dtype=np.uint8)
        return gdf, codes, list(gdf.geometry)

    monkeypatch.setattr(m, "load_global_basins", fake_load_basins)
    monkeypatch.setattr(sc, "load_global_basins", fake_load_basins)


def test_sediment_e2e(tmp_path, monkeypatch):
    """Synthetic-source three-layer full pipeline: build -> structure validation PASS -> manifest -> fidelity PASS."""
    import cubebuild.cli
    from cubebuild.manifest import read_manifest
    from cubebuild.sediment import build_sediment_fidelity_report

    _inject_synthetic(monkeypatch, tmp_path)

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
    entries = {e["id"]: e for e in manifest["layers"]}
    assert set(entries) == {GST1_LAYER_ID, BASIN_CLASS_LAYER_ID, BASIN_DISTANCE_LAYER_ID}

    gst1 = entries[GST1_LAYER_ID]
    assert gst1["dtype"] == "float32"
    assert gst1["coverage"] == {"1deg": 1.0, "30min": 1.0}
    assert gst1["validity_mask"] == f"{GST1_LAYER_ID}__validity"
    assert gst1["visibility"] == "public"

    cls = entries[BASIN_CLASS_LAYER_ID]
    assert cls["dtype"] == "uint8" and cls["visibility"] == "internal"
    assert cls["validity_mask"] == f"{BASIN_CLASS_LAYER_ID}__validity"
    assert len(json.loads(cls["category_encoding"])) == 8

    dist = entries[BASIN_DISTANCE_LAYER_ID]
    assert dist["dtype"] == "float32" and dist["visibility"] == "internal"
    for tier in ("1deg", "30min", "6min", "3min"):
        assert dist["coverage"][tier] == 1.0

    import xarray as xr
    with xr.open_datatree(out, engine="zarr", chunks={}) as dt:
        c3 = dt["/3min"][BASIN_CLASS_LAYER_ID].values
        d3 = dt["/3min"][BASIN_DISTANCE_LAYER_ID].values
        cm3 = dt["/3min"][f"{BASIN_CLASS_LAYER_ID}__validity"].values
        v30 = dt["/30min"][GST1_LAYER_ID].values
        vm30 = dt["/30min"][f"{GST1_LAYER_ID}__validity"].values
        assert np.array_equal(cm3, (c3 != 0).astype(np.uint8))
        assert (vm30 == 1).all()
        r, c = _cell_at("3min", 6.5, 16.5)
        assert c3[r, c] == BASIN_TYPE_CODES["Rift"]
        assert d3[r, c] == 0.0
        assert d3[_cell_at("3min", -30.0, 60.0)] > 0.0
        assert (v30 >= 500.0 - 91.0).all() and (v30 <= 500.0 + 91.0).all()

    for lid in (GST1_LAYER_ID, BASIN_CLASS_LAYER_ID, BASIN_DISTANCE_LAYER_ID):
        report = build_sediment_fidelity_report(
            out, lid, [
                t for t in ("1deg", "30min", "6min", "3min")
                if t in entries[lid]["tiers"]
            ],
        )
        assert report["result"] == "PASS", report["checks"]
        fails = [c for c in report["checks"] if c["status"] == "FAIL"]
        assert not fails


def test_sediment_sidecar_synthetic(tmp_path, monkeypatch):
    """Sidecar parquet: a synthetic version of the 768-shape contract (4 polygons, 3 basins) + encoding column + internal."""
    import geopandas as gpd
    import cubebuild.sidecars as sc
    import cubebuild.sediment as m

    def fake_load_basins(_src=None):
        gdf = _synthetic_basins_gdf()
        codes = gdf["Basin Type"].map(BASIN_TYPE_CODES).to_numpy(dtype=np.uint8)
        return gdf, codes, list(gdf.geometry)

    monkeypatch.setattr(sc, "load_global_basins", fake_load_basins)
    monkeypatch.setattr(sc, "EXPECTED_POLYGONS", 4)
    monkeypatch.setattr(sc, "EXPECTED_BASINS", 3)

    out_dir = tmp_path / "sidecars"
    rec = sc.export_global_basins(out_dir)
    assert rec["id"] == "global_basins" and rec["visibility"] == "internal"
    assert rec["n_features"] == 4 and rec["n_basins"] == 3

    gdf = gpd.read_parquet(out_dir / "global_basins.parquet")
    assert len(gdf) == 4
    assert gdf.crs.to_epsg() == 4326
    assert "basin_type_code" in gdf.columns and "Basin UBI" in gdf.columns
    assert (gdf["basin_type_code"].to_numpy() == gdf["Basin Type"]
            .map(BASIN_TYPE_CODES).to_numpy()).all()
    cross = gdf[gdf["Basin UBI"] == 3]
    assert len(cross) == 2
    xs0 = cross.iloc[0].geometry.bounds
    xs1 = cross.iloc[1].geometry.bounds
    assert (xs0[0] > 0) != (xs1[0] > 0)
