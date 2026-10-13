"""Tests: count-sum kernel + point → cell binning (longitude wrapping / pole
edge) + reader contracts + sidecar exporters (GeoParquet round-trip / poles
merge) + count conservation + sidecar ↔ density-layer cross-check.

Criteria test external behaviour only (count-layer fidelity criterion = sum
conservation):
- sum kernel: in-block sum exact, count conserved, non-integer factor fails
  explicitly;
- binning: known points land in the exact cell, GSRM 0..360 and WSM ±180
  systems give the same cell for the same point, dateline wrapping
  (180 ≡ −180 → column 0), lat=90 pole edge goes to the northernmost cell,
  non-finite / out-of-bounds coordinates rejected, bitwise agreement between
  the two independent paths bincount and per-point loop accumulation;
- readers: shape contracts (WSM 100842×40 / GSRM 22511×9) reject drift,
  parse-failure rows rejected, poles plate code "NA" not swallowed to NaN by
  pandas;
- real-source baselines (on-disk evidenced contractualization): WSM 170
  empty-coordinate rows (all Xmi) locatable 100672; GSRM three counts
  22511/18440/16681; primary-tier density sum conserved;
- sidecars: WSM all rows kept + 170 empty geometries; GSRM 22511 rows + poles
  203 rows across 4 frames;
- e2e: two density layers into store (uint32, no mask, no 30″ tier) + manifest
  count_semantics + fidelity PASS + sidecar binning reproduces the density
  layer (cross-check item).
"""

from pathlib import Path

import numpy as np
import pytest

from cubebuild.kernels import aggregate_sum
from cubebuild.points import (
    GSRM_EXPECTED_ROWS,
    GSRM_LAYER_ID,
    GSRM_POLES_NAMES,
    GSRM_UNIQUE_COORDS,
    GSRM_UNIQUE_SITES,
    HOME_TIER,
    WSM_CSV_NAME,
    WSM_EXPECTED_LOCATABLE,
    WSM_EXPECTED_ROWS,
    WSM_LAYER_ID,
    _aggregation_factor,
    build_wsm_density,
    load_gsrm_gps,
    load_gsrm_poles,
    load_wsm_points,
    points_to_density,
    wsm_home_density,
    gsrm_home_density,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_WSM = (REPO_ROOT / "original data/stress-kinematics/wsm2025-stress"
            / WSM_CSV_NAME).is_file()
requires_wsm = pytest.mark.skipif(not REAL_WSM, reason="WSM 源未集齐")
REAL_GSRM = (REPO_ROOT / "original data/stress-kinematics/gsrm-strain"
             / "GPS_ITRF08.gmt").is_file()
requires_gsrm = pytest.mark.skipif(not REAL_GSRM, reason="GSRM 源未集齐")


def _entry(layer_id=WSM_LAYER_ID, visibility="public"):
    from cubebuild.contract import LayerEntry

    return LayerEntry(
        id=layer_id, dims=("lat", "lon"), source="test",
        native_res="point", native_res_deg=None, home_tier="3min",
        tiers=("1deg", "30min", "6min", "3min"), dtype="uint32",
        unit="count/cell", resampling="sum", mask="免",
        visibility=visibility,
    )



def test_aggregate_sum_exact_and_conservation():
    """In-block sum exact + count conservation (Σcoarse = Σfine) + dtype uint32."""
    data = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.uint32)
    out = aggregate_sum(data, 2)
    assert out.tolist() == [[14, 22]]
    assert out.dtype == np.uint32
    assert int(out.sum(dtype=np.uint64)) == int(data.sum(dtype=np.uint64))


def test_aggregate_sum_rejects_non_integer_factor():
    with pytest.raises(ValueError, match="整数倍"):
        aggregate_sum(np.zeros((3, 4), dtype=np.uint32), 2)


def test_aggregation_factor_integers_only():
    """Primary tier 3′ → 6′/30′/1° = 2/10/20; finer than the primary tier (30″)
    fails explicitly."""
    assert _aggregation_factor("6min") == 2
    assert _aggregation_factor("30min") == 10
    assert _aggregation_factor("1deg") == 20
    with pytest.raises(ValueError, match="整数倍"):
        _aggregation_factor("30sec")



def test_points_to_density_known_cells():
    """Known 1° grid cells: (0.3, 0.7) → row 90 column 180; (−89.9, −179.9) →
    (0, 0)."""
    grid = points_to_density([0.3, -89.9], [0.7, -179.9], "1deg")
    assert grid.dtype == np.uint32
    assert grid[90, 180] == 1
    assert grid[0, 0] == 1
    assert int(grid.sum(dtype=np.uint64)) == 2


def test_points_to_density_lon_conventions():
    """GSRM 0..360 and WSM ±180 systems give the same cell for the same point
    (wrapping 238.327 ≡ −121.673 → row 126 column 58); longitude 0 → column
    180."""
    a = points_to_density([36.914], [238.327], "1deg")
    b = points_to_density([36.914], [-121.673], "1deg")
    np.testing.assert_array_equal(a, b)
    assert a[126, 58] == 1
    assert points_to_density([0.0], [0.0], "1deg")[90, 180] == 1


def test_points_to_density_dateline_and_pole_edge():
    """Dateline wrapping: lon=180 ≡ −180 → column 0; lat=90 pole edge goes to
    the northernmost row."""
    grid = points_to_density([0.0, 90.0], [180.0, -180.0], "1deg")
    assert grid[90, 0] == 1
    assert grid[179, 0] == 1
    assert int(grid.sum(dtype=np.uint64)) == 2


def test_points_to_density_empty_input():
    grid = points_to_density([], [], "1deg")
    assert grid.shape == (180, 360)
    assert not grid.any()


def test_points_to_density_rejects_bad_inputs():
    with pytest.raises(ValueError, match="非有限"):
        points_to_density([np.nan], [0.0], "1deg")
    with pytest.raises(ValueError, match="非有限"):
        points_to_density([0.0], [np.inf], "1deg")
    with pytest.raises(ValueError, match="纬度出界"):
        points_to_density([90.5], [0.0], "1deg")
    with pytest.raises(ValueError, match="形状不一致"):
        points_to_density([0.0, 1.0], [0.0], "1deg")


def test_bincount_equals_loop_accumulation():
    """Independent-path bitwise comparison: bincount binning vs per-point loop
    accumulation — the row/column arithmetic, wrapping and clipping are correct
    only where the two paths agree."""
    rng = np.random.default_rng(42)
    lat = rng.uniform(-90, 90, 2000)
    lon = rng.uniform(-360, 360, 2000)
    grid = points_to_density(lat, lon, "6min")
    nlat, nlon = grid.shape
    res = 1.0 / 10.0
    direct = np.zeros((nlat, nlon), dtype=np.uint32)
    for la, lo in zip(lat, lon):
        lo_w = ((lo + 180.0) % 360.0) - 180.0
        r = min(int(np.floor((la + 90.0) / res)), nlat - 1)
        c = min(int(np.floor((lo_w + 180.0) / res)), nlon - 1)
        direct[r, c] += 1
    np.testing.assert_array_equal(grid, direct)
    assert int(grid.sum(dtype=np.uint64)) == 2000



def test_load_wsm_missing_source(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_wsm_points(tmp_path)


def test_load_wsm_rejects_wrong_shape(tmp_path):
    (tmp_path / WSM_CSV_NAME).write_text("LAT,LON\n1.0,2.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="形状"):
        load_wsm_points(tmp_path)


def test_load_gsrm_missing_source(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_gsrm_gps(tmp_path)


_GPS_LINE = "1.0 0.0 0 0 0 0 0 AB X\n"


def test_load_gsrm_rejects_wrong_rows(tmp_path):
    (tmp_path / "GPS_ITRF08.gmt").write_text(_GPS_LINE * 5, encoding="utf-8")
    with pytest.raises(ValueError, match="行数"):
        load_gsrm_gps(tmp_path)


def test_load_gsrm_rejects_parse_failure(tmp_path):
    """Rows with a missing field (8 tokens) are rejected by per-row validation —
    with keep_default_na=False pandas silently fills "", which isna cannot
    detect, so the token count is the only reliable contract."""
    p = tmp_path / "GPS_ITRF08.gmt"
    p.write_text(
        _GPS_LINE * (GSRM_EXPECTED_ROWS - 1) + "1.0 0.0 0 0 0 0 0 AB\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="非 9 字段"):
        load_gsrm_gps(tmp_path)


def test_load_gsrm_poles_na_plate_preserved(tmp_path):
    """Synthetic poles: plate code "NA" (North America) not swallowed to NaN by
    pandas' default na values."""
    for name in GSRM_POLES_NAMES:
        (tmp_path / name).write_text(
            "49.66 -78.08 0.285 NA\n0.0 10.0 0.5 EU\n", encoding="utf-8"
        )
    poles = load_gsrm_poles(tmp_path)
    assert len(poles) == 2 * len(GSRM_POLES_NAMES)
    assert sorted(poles["frame"].unique()) == ["APM", "IGS08", "NNR", "PA"]
    assert set(poles["plate"]) == {"NA", "EU"}
    assert not poles["plate"].isna().any()


def test_load_gsrm_poles_rejects_parse_failure(tmp_path):
    for name in GSRM_POLES_NAMES:
        (tmp_path / name).write_text(
            "49.66 -78.08 0.285 NA\n0.0 10.0\n", encoding="utf-8"
        )
    with pytest.raises(ValueError, match="非 4 字段"):
        load_gsrm_poles(tmp_path)



def test_builder_coarse_tier_from_home(monkeypatch):
    """Coarse tier = primary-tier sum aggregation (factor 20) + attrs template
    (count_semantics)."""
    import cubebuild.points as points

    home = np.zeros((3600, 7200), dtype=np.uint32)
    home[0, 0] = 5
    home[1, 0] = 7
    monkeypatch.setattr(points, "wsm_home_density", lambda *a, **k: home)
    arr = points.build_wsm_density(_entry(), "1deg")
    vals = arr.compute().values
    assert vals.dtype == np.uint32
    assert vals[0, 0] == 12
    assert int(vals.sum(dtype=np.uint64)) == 12
    assert arr.name == WSM_LAYER_ID
    assert "count_semantics" in arr.attrs
    assert arr.attrs["nodata_semantics"].startswith("0 = 无观测")
    assert "missing_value" not in arr.attrs and "_FillValue" not in arr.attrs



@requires_wsm
def test_wsm_source_baseline():
    """WSM baseline: 100842×40; 170 empty-coordinate rows all QUALITY=Xmi;
    locatable 100672; primary-tier density sum == locatable count (count
    conservation)."""
    df = load_wsm_points()
    assert df.shape == (WSM_EXPECTED_ROWS, 40)
    null = df[df["LAT"].isna() | df["LON"].isna()]
    assert len(null) == WSM_EXPECTED_ROWS - WSM_EXPECTED_LOCATABLE
    assert (null["QUALITY"] == "Xmi").all()
    home = wsm_home_density()
    assert home.shape == (3600, 7200)
    assert int(home.sum(dtype=np.uint64)) == WSM_EXPECTED_LOCATABLE


@requires_gsrm
def test_gsrm_source_baseline_three_numbers():
    """GSRM baseline three counts: 22511 rows / 18440 unique coordinates / 16681
    unique site codes; longitude 0..360 system; primary-tier density sum ==
    22511."""
    df = load_gsrm_gps()
    assert len(df) == GSRM_EXPECTED_ROWS
    assert df["lon"].between(0, 360).all()
    assert int(df.groupby(["lat", "lon"]).ngroups) == GSRM_UNIQUE_COORDS
    assert df["site"].nunique() == GSRM_UNIQUE_SITES
    home = gsrm_home_density()
    assert int(home.sum(dtype=np.uint64)) == GSRM_EXPECTED_ROWS



def test_export_sidecars_missing_source(tmp_path):
    from cubebuild.sidecars import export_gsrm_velocities, export_wsm_points

    with pytest.raises(FileNotFoundError):
        export_wsm_points(tmp_path, src_dir=tmp_path / "nowhere")
    with pytest.raises(FileNotFoundError):
        export_gsrm_velocities(tmp_path, src_dir=tmp_path / "nowhere")


@requires_wsm
def test_export_wsm_sidecar_roundtrip(tmp_path):
    """WSM sidecar: all 100842 rows + 40 attribute columns + 170 empty
    geometries + complete stress fields."""
    import geopandas as gpd

    from cubebuild.sidecars import export_wsm_points

    rec = export_wsm_points(tmp_path)
    assert rec["id"] == "wsm2025_points"
    assert rec["visibility"] == "public"
    assert rec["n_features"] == WSM_EXPECTED_ROWS
    assert rec["n_locatable"] == WSM_EXPECTED_LOCATABLE
    back = gpd.read_parquet(tmp_path / "wsm2025_points.parquet")
    assert len(back) == WSM_EXPECTED_ROWS
    assert back.crs.to_epsg() == 4326
    for col in ("AZI", "TYPE", "QUALITY", "REGIME", "S1AZ", "S1PL",
                "DEPTH", "SITE"):
        assert col in back.columns
    assert len(back.columns) == 41                       # 40 + geometry
    n_null_geom = int(back.geometry.isna().sum())
    assert n_null_geom == WSM_EXPECTED_ROWS - WSM_EXPECTED_LOCATABLE
    assert (back.loc[back.geometry.isna(), "QUALITY"] == "Xmi").all()


@requires_gsrm
def test_export_gsrm_sidecar_roundtrip(tmp_path):
    """GSRM sidecar: 22511 rows (raw lon 0..360 + lon_180 + geometry) + poles
    203 rows across 4 frames + NA plate code preserved; visibility=internal."""
    import geopandas as gpd

    from cubebuild.sidecars import export_gsrm_velocities

    rec = export_gsrm_velocities(tmp_path)
    assert rec["id"] == "gsrm_gps_velocities"
    assert rec["visibility"] == "internal"
    assert rec["n_features"] == GSRM_EXPECTED_ROWS
    assert rec["n_poles"] == 203

    back = gpd.read_parquet(
        tmp_path / "gsrm_gps_velocities" / "gps_itrf08_velocities.parquet"
    )
    assert len(back) == GSRM_EXPECTED_ROWS
    assert back.crs.to_epsg() == 4326
    assert list(back.columns[:9]) == [
        "lon", "lat", "ve", "vn", "se", "sn", "corr", "site", "study"
    ]
    assert back.iloc[0]["lon"] == pytest.approx(238.327, abs=1e-3)
    assert back.iloc[0]["lon_180"] == pytest.approx(-121.673, abs=1e-3)
    assert back.geometry.isna().sum() == 0

    import pandas as pd

    poles = pd.read_parquet(tmp_path / "gsrm_gps_velocities" / "poles.parquet")
    assert len(poles) == 203
    assert sorted(poles["frame"].unique()) == ["APM", "IGS08", "NNR", "PA"]
    assert (poles["plate"] == "NA").any() and not poles["plate"].isna().any()



_MINI = """
version: test
status: test
tiers: [1deg, 6min, 3min]
layers:
  - id: stress_kinematics__wsm_density
    source: wsm2025-stress（100842 条记录）
    native_res: point
    home_tier: 3min
    tiers: [1deg, 6min, 3min]
    dtype: uint32
    unit: count/cell
    resampling: sum
    mask: 免
    notes: 单层总密度；方向/机制/质量分级留 sidecar
  - id: stress_kinematics__gsrm_gps_density
    source: gsrm-strain（GPS_ITRF08.gmt，观测站点速度）
    native_res: point
    home_tier: 3min
    tiers: [1deg, 6min, 3min]
    dtype: uint32
    unit: count/cell
    resampling: sum
    mask: 免
    visibility: internal
    notes: ITRF08 单系计数，禁跨系
"""


@requires_wsm
@requires_gsrm
def test_point_density_e2e(tmp_path, monkeypatch):
    """uint32 density-layer full chain: build → structural validation PASS →
    manifest (dtype / count_semantics / visibility / sidecars) → fidelity PASS
    (bitwise / cross-tier / count conservation) → sidecar binning reproduces the
    density layer (cross-check item)."""
    import cubebuild.cli
    import geopandas as gpd
    import xarray as xr
    from cubebuild.manifest import read_manifest
    from cubebuild.points import build_point_density_fidelity_report
    from cubebuild.sidecars import export_gsrm_velocities, export_wsm_points

    monkeypatch.setattr(cubebuild.cli, "SIDECARS", {
        "wsm2025_points": export_wsm_points,
        "gsrm_gps_velocities": export_gsrm_velocities,
    })

    mapping = tmp_path / "mini.yaml"
    mapping.write_text(_MINI, encoding="utf-8")
    out = tmp_path / "cube.zarr"
    sidecars_dir = tmp_path / "sidecars"
    rc = cubebuild.cli.main([
        "build", "--mapping", str(mapping), "--out", str(out),
        "--reports", str(tmp_path / "reports"),
        "--sidecars", str(sidecars_dir),
    ])
    assert rc == 0

    manifest = read_manifest(out)
    entries = {e["id"]: e for e in manifest["layers"]}
    assert set(entries) == {WSM_LAYER_ID, GSRM_LAYER_ID}
    for eid, vis in ((WSM_LAYER_ID, "public"), (GSRM_LAYER_ID, "internal")):
        e = entries[eid]
        assert e["dtype"] == "uint32"
        assert e["visibility"] == vis
        assert e["resampling"] == "sum"
        assert "count_semantics" in e
        assert "validity_mask" not in e
        assert e["coverage"] == {"1deg": 1.0, "6min": 1.0, "3min": 1.0}
    assert "100672" in entries[WSM_LAYER_ID]["count_semantics"]
    assert "22511" in entries[GSRM_LAYER_ID]["count_semantics"]
    assert "18440" in entries[GSRM_LAYER_ID]["count_semantics"]
    assert "16681" in entries[GSRM_LAYER_ID]["count_semantics"]
    assert [s["id"] for s in manifest["sidecars"]] == [
        "wsm2025_points", "gsrm_gps_velocities"
    ]
    assert manifest["sidecars"][1]["visibility"] == "internal"

    for eid in (WSM_LAYER_ID, GSRM_LAYER_ID):
        report = build_point_density_fidelity_report(out, eid, ["1deg", "6min", "3min"])
        assert report["result"] == "PASS", report["checks"]
        n_hard = sum(1 for c in report["checks"] if c["status"] != "report")
        expect_hard = 3 * 3 + 2 + (1 if eid == GSRM_LAYER_ID else 0)
        assert n_hard == expect_hard

    with xr.open_datatree(out, engine="zarr", chunks={}) as dt:
        wsm_store = dt[f"/{HOME_TIER}"].ds[WSM_LAYER_ID].values
        gsrm_store = dt[f"/{HOME_TIER}"].ds[GSRM_LAYER_ID].values
    back_w = gpd.read_parquet(sidecars_dir / "wsm2025_points.parquet")
    ok = back_w.geometry.notna()
    np.testing.assert_array_equal(
        points_to_density(
            back_w.loc[ok, "geometry"].y, back_w.loc[ok, "geometry"].x, HOME_TIER
        ),
        wsm_store,
    )
    back_g = gpd.read_parquet(
        sidecars_dir / "gsrm_gps_velocities" / "gps_itrf08_velocities.parquet"
    )
    np.testing.assert_array_equal(
        points_to_density(
            back_g["geometry"].y, back_g["geometry"].x, HOME_TIER
        ),
        gsrm_store,
    )
