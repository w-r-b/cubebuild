"""Tests: categorical mode kernel (nodata semantics) + polygon rasteriser + integer
missing-data two-channel consistency + sidecar exporters (GeoParquet / FileGDB
pass-through as-is).

The criteria test external behaviour only:
- mode kernel: nodata does not participate in the count, an all-nodata block → nodata, a
  tie picks the smaller class, nodata=None preserves the land-sea mask semantics (0 is a
  real class);
- rasteriser: cell-centre-containment semantics (a thin strip with no centre inside a polygon
  is not burned), on overlap the smaller area wins (deterministic rule), and planar
  rasterisation of the pole-cap cut representation (edge closed along lat=90) is correct;
- reader: prov_type vocabulary contract, rejection of a ring crossing the dateline (planar
  rasterisation is unsafe);
- mask mechanism: two-channel consistency of a uint8 categorical layer where 0=missing
  (build side + pipeline side);
- sidecars: hasterok GeoParquet round-trip (row count / encoding column / geometry), limw .gdb
  pass-through integrity as-is (synthetic tree; the real 1.2G pass-through is covered by the
  acceptance build);
- real-source integration (when present): rasterisation coverage / overlap baselines + e2e
  uint8 layer into the store.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from cubebuild.kernels import aggregate_mode
from cubebuild.masks import build_validity_mask, data_missing, validity_layer_name
from cubebuild.sidecars import export_hasterok_gprv, export_limw_polygons
from cubebuild.vector import (
    GPRV_SRC_DIR,
    HOME_TIER,
    PROV_TYPE_CODES,
    _aggregation_factor,
    category_encoding,
    load_gprv,
    rasterize_polygon_classes,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_GPRV = (REPO_ROOT / "original data/lithosphere/tectonic-provinces"
             / "plates&provinces/shp/global_gprv.shp").is_file()
requires_gprv = pytest.mark.skipif(not REAL_GPRV, reason="GPRV 源未集齐")


def _entry(mask="validity", dtype="uint8"):
    from cubebuild.contract import LayerEntry

    return LayerEntry(
        id="lithosphere__hasterok_province_class", dims=("lat", "lon"),
        source="test", native_res="vector-polygon", native_res_deg=None,
        home_tier="3min", tiers=("1deg", "30min", "6min", "3min"),
        dtype=dtype, unit="category（一级分类）", resampling="mode",
        mask=mask, visibility="public",
    )



def test_aggregate_mode_nodata_excluded():
    """nodata=0 does not participate in the count: the mode of the block [0,0,5,5] is 5, not 0."""
    data = np.array([[0, 0, 5, 5], [5, 5, 0, 0]], dtype=np.uint8)
    out = aggregate_mode(data, 2, nodata=0)
    assert out.tolist() == [[5, 5]]


def test_aggregate_mode_all_nodata_block():
    """All-nodata block → nodata (no valid sub-cell means missing)."""
    data = np.zeros((2, 4), dtype=np.uint8)
    out = aggregate_mode(data, 2, nodata=0)
    assert out.tolist() == [[0, 0]]


def test_aggregate_mode_tie_smallest_class():
    """Tie picks the smaller class: the block [3,7 / 3,7] counts 2:2 → 3 (deterministic, same convention as scipy)."""
    data = np.array([[3, 7, 3, 7], [3, 7, 3, 7]], dtype=np.uint8)
    assert aggregate_mode(data, 2, nodata=0).tolist() == [[3, 3]]


def test_aggregate_mode_without_nodata_keeps_landsea_semantics():
    """nodata=None preserves the semantics: 0 is a real class (land-sea mask 0=sea) —
    the block [0,1/0,1] ties and picks 0; with nodata=0 the same block's mode is 1."""
    data = np.array([[0, 1, 0, 1], [0, 1, 0, 1]], dtype=np.uint8)
    assert aggregate_mode(data, 2).tolist() == [[0, 0]]
    assert aggregate_mode(data, 2, nodata=0).tolist() == [[1, 1]]


def test_aggregation_factor_integers_only():
    """Home tier 3′ → 6′/30′/1° = 2/10/20; finer than the home tier (30″) fails explicitly."""
    assert _aggregation_factor("6min") == 2
    assert _aggregation_factor("30min") == 10
    assert _aggregation_factor("1deg") == 20
    with pytest.raises(ValueError, match="整数倍"):
        _aggregation_factor("30sec")



def test_rasterize_center_semantics():
    """Burns only where the cell centre falls inside the polygon: on the 1° grid box(0,0,1,1)
    burns exactly one cell at (0.5°,0.5°); the thin strip box(0.1,0.2,0.2,0.3) containing no
    cell centre burns no cell."""
    from shapely.geometry import box

    geoms = [box(0.0, 0.0, 1.0, 1.0), box(0.1, 0.2, 0.2, 0.3)]
    codes = np.array([7, 9], dtype=np.uint8)
    classes, overlap, uncovered = rasterize_polygon_classes(geoms, codes, "1deg")
    assert classes[90, 180] == 7
    classes[90, 180] = 0
    assert not classes.any()
    assert overlap == 0
    assert uncovered == classes.size - 1


def test_rasterize_overlap_smaller_wins():
    """On overlapping pixels the smaller-area polygon wins (deterministic rule): the big box
    has code 9, the inner small box has code 2 → the small-box region = 2 and the rest of the
    big box = 9."""
    from shapely.geometry import box

    big = box(0.0, 0.0, 10.0, 10.0)
    small = box(4.0, 4.0, 6.0, 6.0)
    classes, overlap, _ = rasterize_polygon_classes(
        [big, small], np.array([9, 2], dtype=np.uint8), "1deg"
    )
    classes2, _, _ = rasterize_polygon_classes(
        [small, big], np.array([2, 9], dtype=np.uint8), "1deg"
    )
    np.testing.assert_array_equal(classes, classes2)
    small_cells = classes[94:96, 184:186]
    assert (small_cells == 2).all()
    assert classes[90:100, 180:190].sum() - small_cells.sum() == 9 * (100 - 4)
    assert overlap == 4


def test_rasterize_pole_cap_representation():
    """Pole-cap cut representation (the ring closes along the lat=90 edge from −180 to 180):
    planar rasterisation should fill every row above 85° latitude (the Arctic-basin shape, the
    only >180° edge measured on disk)."""
    from shapely.geometry import Polygon

    cap = Polygon([(-180.0, 85.0), (-180.0, 90.0), (180.0, 90.0), (180.0, 85.0)])
    classes, _, _ = rasterize_polygon_classes([cap], np.array([4], dtype=np.uint8), "1deg")
    assert (classes[175:180, :] == 4).all()
    assert (classes[:175, :] == 0).all()



def test_prov_type_encoding_table():
    """Encoding table: 15 classes, codes 1..15 in lexicographic order, and category_encoding round-trips consistently."""
    assert len(PROV_TYPE_CODES) == 15
    assert sorted(PROV_TYPE_CODES.values()) == list(range(1, 16))
    enc = category_encoding()
    assert enc["1"] == "accretionary complex" and enc["15"] == "wide rift"
    for label, code in PROV_TYPE_CODES.items():
        assert enc[str(code)] == label


def _fake_gdf(rows, prov_types):
    import geopandas as gpd
    from shapely.geometry import box

    n = len(rows)
    return gpd.GeoDataFrame(
        {"prov_type": prov_types, "prov_name": [f"p{i}" for i in range(n)]},
        geometry=[box(*r) for r in rows],
        crs="EPSG:4326",
    )


def _fake_gdf_n(n, prov_type="basin"):
    """n rows of a homogeneous fake source (passes the 914-row contract, so subsequent checks can be triggered)."""
    return _fake_gdf(
        [(float(i), 0.0, float(i) + 0.5, 0.5) for i in range(n)],
        [prov_type] * n,
    )


def _fake_src(tmp_path):
    """Placeholder source file (passes the existence check; read_file is replaced by monkeypatch)."""
    p = tmp_path / "plates&provinces/shp/global_gprv.shp"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("", encoding="utf-8")
    return tmp_path


def test_load_gprv_rejects_unknown_prov_type(tmp_path, monkeypatch):
    import geopandas as gpd

    monkeypatch.setattr(
        gpd, "read_file", lambda _p: _fake_gdf_n(914, prov_type="mystery province")
    )
    with pytest.raises(ValueError, match="超出编码表"):
        load_gprv(_fake_src(tmp_path))


def test_load_gprv_rejects_dateline_crossing(tmp_path, monkeypatch):
    """Rejected at read time: a ring crossing the dateline (Δlon>180° and not a pole-closing
    edge) — planar rasterisation would silently misalign it."""
    import geopandas as gpd
    from shapely.geometry import Polygon

    crossing = Polygon([(179.0, 0.0), (179.0, 1.0), (-179.0, 1.0), (-179.0, 0.0)])
    gdf = gpd.GeoDataFrame(
        {"prov_type": ["basin"] * 913 + ["basin"]},
        geometry=[crossing] + list(_fake_gdf_n(913).geometry),
        crs="EPSG:4326",
    )
    monkeypatch.setattr(gpd, "read_file", lambda _p: gdf)
    with pytest.raises(ValueError, match="跨日期线"):
        load_gprv(_fake_src(tmp_path))


def test_load_gprv_rejects_wrong_rows(tmp_path, monkeypatch):
    import geopandas as gpd

    monkeypatch.setattr(
        gpd, "read_file",
        lambda _p: _fake_gdf([(0, 0, 1, 1), (2, 2, 3, 3)], ["basin", "craton"]),
    )
    with pytest.raises(ValueError, match="行数"):
        load_gprv(_fake_src(tmp_path))



def test_data_missing_by_dtype():
    import xarray as xr

    f = xr.DataArray(np.array([1.0, np.nan]))
    u = xr.DataArray(np.array([1, 0], dtype=np.uint8))
    assert data_missing(f).values.tolist() == [False, True]
    assert data_missing(u).values.tolist() == [False, True]


def test_build_validity_mask_uint8_layer():
    """uint8 categorical layer: mask 0 ⟺ data 0 (encoding starts at 1, 0 = no class)."""
    import dask.array as da
    import xarray as xr

    arr = xr.DataArray(
        da.from_array(np.array([[0, 3], [7, 0]], dtype=np.uint8), chunks=(2, 2))
    )
    mask = build_validity_mask(arr, _entry())
    assert mask.dtype == np.uint8
    assert mask.compute().values.tolist() == [[0, 1], [1, 0]]
    assert mask.name == validity_layer_name(_entry().id)
    assert mask.attrs["validity_of"] == "lithosphere__hasterok_province_class"



@requires_gprv
def test_export_hasterok_sidecar_roundtrip(tmp_path):
    """hasterok sidecar: GeoParquet round-trip (row count / attribute columns / encoding column / geometry / CRS)."""
    import geopandas as gpd

    rec = export_hasterok_gprv(tmp_path)
    assert rec["id"] == "hasterok_gprv"
    assert rec["format"] == "geoparquet"
    assert rec["visibility"] == "public"
    assert rec["n_features"] == 914
    back = gpd.read_parquet(tmp_path / "hasterok_gprv.parquet")
    assert len(back) == 914
    assert back.crs.to_epsg() == 4326
    for col in ("prov_name", "prov_type", "prov_group", "lastorogen",
                "crust_type", "area"):
        assert col in back.columns
    assert (back["prov_type_code"] == back["prov_type"].map(PROV_TYPE_CODES)).all()
    assert back.geometry.is_valid.all()


def test_export_limw_passthrough_synthetic(tmp_path):
    """limw sidecar: .gdb pass-through as-is (mechanism verified on a synthetic tree: the name is
    unchanged, files are byte-identical, integrity is checked, and manifest fields are
    recorded)."""
    src = tmp_path / "src"
    (src / "limw-gis-2015.gdb" / "sub").mkdir(parents=True)
    f1 = src / "limw-gis-2015.gdb" / "a.dat"
    f2 = src / "limw-gis-2015.gdb" / "sub" / "b.dat"
    f1.write_bytes(b"\x00\x01" * 1000)
    f2.write_bytes(b"\xff" * 500)

    out = tmp_path / "sidecars"
    rec = export_limw_polygons(out, src_dir=src)
    dst = out / "limw-gis-2015.gdb"
    assert dst.is_dir() and dst.name.endswith(".gdb")
    assert rec["id"] == "limw_polygons" and rec["format"] == "filegdb"
    assert rec["n_files"] == 2 and rec["size_bytes"] == 2500
    assert rec["visibility"] == "internal"
    assert "不栅格化" in rec["notes"] and "internal" in rec["notes"]
    assert (dst / "a.dat").read_bytes() == f1.read_bytes()
    assert (dst / "sub" / "b.dat").read_bytes() == f2.read_bytes()

    (dst / "stale.dat").write_bytes(b"stale")
    export_limw_polygons(out, src_dir=src)
    assert not (dst / "stale.dat").exists()


def test_export_limw_missing_source(tmp_path):
    with pytest.raises(FileNotFoundError):
        export_limw_polygons(tmp_path, src_dir=tmp_path / "nowhere")



_MINI = """
version: test
status: test
tiers: [1deg, 6min, 3min]
layers:
  - id: lithosphere__hasterok_province_class
    source: tectonic-provinces（global_gprv.shp，914 多边形，main 版）
    native_res: vector-polygon
    home_tier: 3min
    tiers: [1deg, 6min, 3min]
    dtype: uint8
    unit: category（一级分类）
    resampling: mode
    mask: validity
    notes: 穷尽覆盖型，豁免距离场
"""


@requires_gprv
def test_vector_layer_e2e(tmp_path, monkeypatch):
    """uint8 categorical layer full chain: build → structural validation PASS (including the
    integer two-channel consistency hard criterion) → manifest encoding table / coverage →
    fidelity report PASS (mode consistency)."""
    import cubebuild.cli
    from cubebuild.manifest import read_manifest
    from cubebuild.sidecars import export_hasterok_gprv
    from cubebuild.vector import build_hasterok_fidelity_report

    monkeypatch.setattr(
        cubebuild.cli, "SIDECARS", {"hasterok_gprv": export_hasterok_gprv}
    )

    mapping = tmp_path / "mini.yaml"
    mapping.write_text(_MINI, encoding="utf-8")
    out = tmp_path / "cube.zarr"
    rc = cubebuild.cli.main([
        "build", "--mapping", str(mapping), "--out", str(out),
        "--reports", str(tmp_path / "reports"),
        "--sidecars", str(tmp_path / "sidecars"),
    ])
    assert rc == 0

    manifest = read_manifest(out)
    entry = manifest["layers"][0]
    assert entry["id"] == "lithosphere__hasterok_province_class"
    assert entry["dtype"] == "uint8"
    assert entry["validity_mask"] == "lithosphere__hasterok_province_class__validity"
    enc = json.loads(entry["category_encoding"])
    assert enc[str(PROV_TYPE_CODES["craton"])] == "craton"
    assert entry["coverage"]["3min"] == pytest.approx(0.99999, abs=1e-5)
    assert entry["coverage"]["1deg"] == 1.0
    assert "rasterization_rule" in entry

    assert [s["id"] for s in manifest["sidecars"]] == ["hasterok_gprv"]

    report = build_hasterok_fidelity_report(out, "lithosphere__hasterok_province_class",
                                            ["1deg", "6min", "3min"])
    assert report["result"] == "PASS", report["checks"]
    n_hard = sum(1 for c in report["checks"] if c["status"] != "report")
    assert n_hard == 3 * 2 + 2


@requires_gprv
def test_real_gprv_rasterization_baselines():
    """Real-source baseline (contract derived from disk measurement): 914 rows, 3′ coverage
    0.99999, 319 overlapping cells (0.001%), 241 uncovered cells (digitising gaps at province
    boundaries)."""
    gdf, codes = load_gprv()
    assert len(gdf) == 914
    assert set(codes) <= set(range(1, 16))
    classes, overlap, uncovered = rasterize_polygon_classes(
        list(gdf.geometry), codes, HOME_TIER
    )
    assert classes.shape == (3600, 7200)
    assert (classes != 0).mean() == pytest.approx(0.9999907, abs=1e-6)
    assert overlap == 319
    assert uncovered == 241


@requires_gprv
def test_geo_spot_checks_against_source():
    """Geographic final check (standing regression criterion): at 10 landmark points (craton /
    mid-ocean ridge / back-arc / polar region / rift) the store value == the class of the source
    polygon at the nearest cell centre to that point (contains + smallest area, same overlap
    rule) — an independent geographic verification of orientation / mirror defects."""
    import geopandas as gpd
    import xarray as xr
    from shapely.geometry import Point

    store = REPO_ROOT / "products/cube-v1.0.zarr"
    if not (store / "zarr.json").exists():
        pytest.skip("验收构建 store 不存在（先跑真实构建）")

    gdf = gpd.read_file(
        REPO_ROOT / "original data/lithosphere/tectonic-provinces"
        / "plates&provinces/shp/global_gprv.shp"
    )
    dt = xr.open_datatree(store, engine="zarr", chunks={})
    arr = dt["/3min"].ds["lithosphere__hasterok_province_class"]
    pts = {
        "Tanzania Craton": (-6.0, 33.0), "Mid-Atlantic ridge": (0.0, -20.0),
        "Andes orogen": (-25.0, -68.0), "Izu-Bonin arc": (28.0, 141.0),
        "West Philippine Basin": (16.0, 130.0), "Siberian craton": (62.0, 100.0),
        "Zagros foredeep": (30.0, 50.0), "Red Sea rift": (20.0, 38.5),
        "Arctic ocean": (85.0, 0.0), "East African Rift": (-3.0, 36.0),
    }
    for name, (lat, lon) in pts.items():
        sel = arr.sel(lat=lat, lon=lon, method="nearest")
        clat, clon = float(sel.lat), float(sel.lon)
        got = int(sel.values)
        cand = gdf[gdf.contains(Point(clon, clat))]
        if len(cand):
            idx = cand.geometry.area.idxmin()
            expect = PROV_TYPE_CODES[cand.loc[idx, "prov_type"]]
        else:
            expect = 0
        assert got == expect, (
            f"{name}: store={got} vs 源@胞心({clat:.4f},{clon:.4f})={expect}"
        )
    dt.close()


@requires_gprv
def test_windowed_rasterize_equals_fullgrid_direct():
    """Independent-path bit-level comparison: per-polygon windowed burning (window geometry /
    transform computed independently) vs a single full-grid rasterio direct rasterisation (same
    overlap rule: descending area + later-wins). Agreement between the two ⇒ the windowed row /
    column / transform maths is correct (mirror / misalignment defects surface here — a mixed
    row-number north-south convention during development is caught by exactly this comparison)."""
    import rasterio.features as rfeatures
    from rasterio.transform import from_origin
    from cubebuild.grids import grid_shape, tier_deg

    gdf, codes = load_gprv()
    tier = "30min"
    nlat, nlon = grid_shape(tier)
    res = tier_deg(tier)
    order = np.argsort([-g.area for g in gdf.geometry], kind="stable")
    shapes = [(gdf.geometry[i], int(codes[i])) for i in order]
    direct = rfeatures.rasterize(
        shapes, out_shape=(nlat, nlon),
        transform=from_origin(-180.0, 90.0, res, res), fill=0, dtype="uint8",
    )[::-1]
    windowed, _, _ = rasterize_polygon_classes(list(gdf.geometry), codes, tier)
    np.testing.assert_array_equal(windowed, direct)
