"""Tests: 0.1° pixel keys (left-closed / boundary-robust / dateline) + merge-dedup rule
(synthetic sources) + reader contract (shape / grid self-consistency / non-finite
rejection) + two-layer builders (density attrs / HFgrid14 phase identity and 1° weighting)
+ sidecar round-trip + real-source baselines (merged triple numbers / incorporation-evidence
figures) + e2e (two layers into the store, manifest provenance, fidelity PASS, sidecar
round-trip density reconstruction).
"""

from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

import cubebuild.thermal as th
from cubebuild.thermal import (
    GHFDB_EXPECTED_ROWS,
    GHFDB_SRC_DIR,
    GHFDB_XLSX_NAME,
    HEATFLOW_DENSITY_ID,
    HFGRID14_ID,
    HFGRID_SRC_DIR,
    MERGED_TOTAL,
    NGHF_EXACT_COORD_MATCHES,
    NGHF_EXPECTED_LOCATABLE,
    NGHF_EXPECTED_ROWS,
    NGHF_PIXEL_OVERLAP,
    NGHF_Q_COMPARABLE,
    NGHF_Q_IDENTICAL,
    NGHF_SRC_DIR,
    NGHF_SUPPLEMENT,
    pixel_keys,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_GHFDB = (GHFDB_SRC_DIR / GHFDB_XLSX_NAME).is_file()
REAL_NGHF = (NGHF_SRC_DIR / "NGHF.csv").is_file()
REAL_HFGRID = (HFGRID_SRC_DIR / "HFgrid14.csv").is_file()
requires_heatflow = pytest.mark.skipif(
    not (REAL_GHFDB and REAL_NGHF), reason="GHFDB/NGHF 源未集齐"
)
requires_hfgrid = pytest.mark.skipif(not REAL_HFGRID, reason="HFgrid14 源未集齐")


def _density_entry(layer_id=HEATFLOW_DENSITY_ID, visibility="public"):
    from cubebuild.contract import LayerEntry

    return LayerEntry(
        id=layer_id, dims=("lat", "lon"), source="test",
        native_res="point", native_res_deg=None, home_tier="3min",
        tiers=("1deg", "30min", "6min", "3min"), dtype="uint32",
        unit="count/cell", resampling="sum", mask="免",
        visibility=visibility,
    )


def _hfgrid_entry(visibility="public"):
    from cubebuild.contract import LayerEntry

    return LayerEntry(
        id=HFGRID14_ID, dims=("lat", "lon"), source="test",
        native_res="0.5deg", native_res_deg=0.5, home_tier="30min",
        tiers=("1deg", "30min"), dtype="float32", unit="mW/m2",
        resampling="conservative_area_weighted", mask="validity + landsea",
        visibility=visibility,
    )



def test_pixel_keys_basic_partition():
    """Left-closed partition: 0.05 and 0.15 get different keys; 0.09 and 0.1 get different keys (0.1 falls into the upper cell)."""
    a = pixel_keys([0.05], [0.05])
    b = pixel_keys([0.15], [0.15])
    assert a[0] != b[0]
    assert pixel_keys([0.09], [0.0])[0] != pixel_keys([0.10], [0.0])[0]


def test_pixel_keys_boundary_robustness():
    """A decimal boundary point (23.3) deterministically falls into the upper cell: a
    float-noise-equivalent point gets the same key while a true neighbouring value (23.29999)
    gets a different key — bare division is non-deterministic here (measured: 3808 points
    perturbed)."""
    base = pixel_keys([23.3], [10.0])
    noise = pixel_keys([23.300000000000004], [10.0])
    below = pixel_keys([23.29999], [10.0])
    assert base[0] == noise[0]
    assert base[0] != below[0]


def test_pixel_keys_dateline():
    """Longitude wrapping: 180 ≡ −180 get the same key; 179.95 and −179.95 belong to adjacent but different pixels."""
    assert pixel_keys([5.0], [180.0])[0] == pixel_keys([5.0], [-180.0])[0]
    assert pixel_keys([5.0], [179.95])[0] != pixel_keys([5.0], [-179.95])[0]


def test_pixel_keys_matches_exact_decimal_reference():
    """Independent path: for random 4-decimal coordinates (including boundary values) the
    keys equal the decimal exact-arithmetic (Fraction) floor — the left-closed convention is
    consistent across both paths."""
    rng = np.random.default_rng(7)
    lat = rng.uniform(-89, 89, 3000).round(4)
    lon = rng.uniform(-180, 180, 3000).round(4)
    keys = pixel_keys(lat, lon)
    for i in range(len(lat)):
        r = (Fraction(str(lat[i])) + 90) * 10
        c = (Fraction(str(lon[i])) + 180) * 10
        assert keys[i] == int(r) * 3600 + int(c), f"点 {i}: {lat[i]}, {lon[i]}"



def _synth_ghfdb():
    import pandas as pd

    return pd.DataFrame({
        "q": [50.0, 60.0, 70.0, 80.0, 90.0],
        "q_uncertainty": [np.nan] * 5,
        "name": ["g1", "g2", "g3", "g4", "g5"],
        "lat": [10.05, 10.15, 5.0, -45.3, 0.0],
        "lon": [20.05, 20.15, -179.95, 170.2, 0.0],
        "reference": ["R1"] * 5,
        "id": ["R24-000001", "R24-000002", "R24-000003", "R24-000004", "R24-000005"],
    })


def _synth_nghf():
    import pandas as pd

    return pd.DataFrame({
        "q": [51.0, 61.0, 62.0, 71.0, 81.0, np.nan],
        "q_uncertainty": [np.nan] * 6,
        "name": ["n1", "n2", "n3", "n4", "n5", "n6"],
        "lat": [10.08, 10.12, 10.1, 6.0, 12.0, 5.0],
        "lon": [20.08, 20.18, 20.1, np.nan, 21.0, 180.0],
        "reference": ["N1", "N2", "N3", "N4", "N5", "N6"],
        "source_row": np.arange(6),
    })


def test_merge_rule_synthetic(monkeypatch):
    """Rule verified case by case: same-pixel filtering (including the 0.1 boundary point and
    dateline wrapping), new-pixel retention, exclusion of non-locatable points, full retention
    of GHFDB, and source/source_row provenance."""
    monkeypatch.setattr(th, "load_ghfdb", lambda *a, **k: _synth_ghfdb())
    monkeypatch.setattr(th, "load_nghf", lambda *a, **k: _synth_nghf())
    monkeypatch.setattr(th, "NGHF_SUPPLEMENT", 1)
    monkeypatch.setattr(th, "MERGED_TOTAL", 6)
    df = th.merge_heatflow_points("x", "y")
    assert len(df) == 6
    assert (df["source"] == "GHFDB").sum() == 5
    kept = df[df["source"] == "NGHF"]
    assert list(kept["name"]) == ["n5"]
    names = set(df["name"])
    for dropped in ("n1", "n2", "n3", "n4", "n6"):
        assert dropped not in names
    g_part = df[df["source"] == "GHFDB"]
    assert list(g_part["source_row"]) == [0, 1, 2, 3, 4]
    assert list(df.loc[df["source"] == "NGHF", "source_row"]) == [4]


def test_merge_rejects_contract_drift(monkeypatch):
    """Source drift (supplement count deviating from the contract) fails explicitly."""
    monkeypatch.setattr(th, "load_ghfdb", lambda *a, **k: _synth_ghfdb())
    monkeypatch.setattr(th, "load_nghf", lambda *a, **k: _synth_nghf())
    with pytest.raises(ValueError, match="补充点数"):
        th.merge_heatflow_points("x-drift", "y-drift")



def _write_ghfdb_xlsx(path, n_data_rows, n_cols=67):
    import openpyxl

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "GHFDB R20024 v.2026.03"
    names = [f"col{i}" for i in range(n_cols)]
    names[0], names[1], names[2], names[3], names[4], names[19], names[66] = (
        "q", "q_uncertainty", "name", "lat_NS", "long_EW",
        "publication_reference", "ID",
    )
    for _ in range(5):
        ws.append(["x"] * n_cols)
    ws.append(names)
    for i in range(n_data_rows):
        row = [""] * n_cols
        row[0], row[1], row[2] = "50", "?", f"g{i}"
        row[3], row[4] = "10.05", "20.05"
        row[19] = "R"
        row[66] = f"R24-{i:06d}"
        ws.append(row)
    wb.save(path)


def test_load_ghfdb_missing_source(tmp_path):
    with pytest.raises(FileNotFoundError):
        th.load_ghfdb(tmp_path)


def test_load_ghfdb_rejects_wrong_shape(tmp_path):
    _write_ghfdb_xlsx(tmp_path / GHFDB_XLSX_NAME, 3)
    with pytest.raises(ValueError, match="形状"):
        th.load_ghfdb(tmp_path)


_NGHF_HEADER = (
    "country,code1,code2,code3,code4,code5,code6,Name of site,longitude,"
    "latitude,elevation (m),minimum depth (m),maximum depth (m),"
    "number of temperatures,temperature gradient (mK/m),"
    "number of conductivities,conductivity (W/mK),number of heat production,"
    "heat production (µW/m3),heat-flow (mW/m2),uncertainty hf (mW/m2),HFC,"
    "number of sites,reference,date_reference\n"
)


def test_load_nghf_missing_source(tmp_path):
    with pytest.raises(FileNotFoundError):
        th.load_nghf(tmp_path)


def test_load_nghf_rejects_wrong_shape(tmp_path):
    p = tmp_path / "NGHF.csv"
    p.write_text(_NGHF_HEADER + "A,E,L,O,N,A,A,s,1,2,\n", encoding="utf-8")
    with pytest.raises(ValueError, match="形状"):
        th.load_nghf(tmp_path)


def test_load_hfgrid14_missing_source(tmp_path):
    with pytest.raises(FileNotFoundError):
        th.load_hfgrid14(tmp_path)


def test_load_hfgrid14_rejects_wrong_shape(tmp_path):
    p = tmp_path / "HFgrid14.csv"
    p.write_text("longiyude;latitude;HF_pred;sHF_pred;Hf_obs\n"
                 "-179.75;89.75;100;10;\n", encoding="utf-8")
    with pytest.raises(ValueError, match="形状"):
        th.load_hfgrid14(tmp_path)


def _hfgrid_rows(n=259200):
    """Rows of the complete 0.5° global grid (north-first row order, same convention as the source)."""
    return [f"{-179.75 + (i % 720) * 0.5};{89.75 - (i // 720) * 0.5};1;;"
            for i in range(n)]


def test_load_hfgrid14_rejects_duplicate_cells(tmp_path):
    p = tmp_path / "HFgrid14.csv"
    rows = ["longiyude;latitude;HF_pred;sHF_pred;Hf_obs"] + _hfgrid_rows()
    rows[2] = rows[1]
    p.write_text("\n".join(rows) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="非完整网格"):
        th.load_hfgrid14(tmp_path)


def test_load_hfgrid14_rejects_offgrid_center(tmp_path):
    p = tmp_path / "HFgrid14.csv"
    rows = ["longiyude;latitude;HF_pred;sHF_pred;Hf_obs"] + _hfgrid_rows()
    rows[1] = "-179.75;89.749;100;10;"
    p.write_text("\n".join(rows) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="偏离"):
        th.load_hfgrid14(tmp_path)


def test_load_hfgrid14_rejects_nonfinite_pred(tmp_path):
    p = tmp_path / "HFgrid14.csv"
    rows = ["longiyude;latitude;HF_pred;sHF_pred;Hf_obs"] + _hfgrid_rows()
    rows[1] = "-179.75;89.75;abc;10;"
    p.write_text("\n".join(rows) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="非有限"):
        th.load_hfgrid14(tmp_path)



def _synth_hfgrid_source(values_2d):
    from cubebuild.gravmag import RasterSource

    nlat, nlon = values_2d.shape
    lat_edges = -90.0 + np.arange(nlat + 1, dtype=np.float64) * 0.5
    return RasterSource(
        values=values_2d.astype(np.float32),
        valid=np.ones(values_2d.shape, dtype=bool),
        lat_edges=lat_edges,
        lon_res=Fraction(1, 2),
        lon_phase=Fraction(1, 4),
    )


def test_hfgrid14_home_tier_identity(monkeypatch):
    """Identity at the 30′ tier (source-cell edges aligned to -180/-90 + lon_phase=0.25): bit-for-bit equality."""
    rng = np.random.default_rng(3)
    vals = rng.uniform(20, 150, (360, 720))
    monkeypatch.setattr(th, "_hfgrid14_source",
                        lambda *a, **k: _synth_hfgrid_source(vals))
    arr = th.build_hfgrid14_heatflow(_hfgrid_entry(), "30min").compute().values
    np.testing.assert_array_equal(arr, vals.astype(np.float32))


def test_hfgrid14_1deg_column_pairing(monkeypatch):
    """1° tier = area-weighted mean of 2×2 source blocks: with a column-constant field the adjacent column pair's mean is exact."""
    vals = np.tile(np.arange(720, dtype=np.float64), (360, 1))
    monkeypatch.setattr(th, "_hfgrid14_source",
                        lambda *a, **k: _synth_hfgrid_source(vals))
    arr = th.build_hfgrid14_heatflow(_hfgrid_entry(), "1deg").compute().values
    for j in (0, 100, 359):
        expect = (vals[0, 2 * j] + vals[0, 2 * j + 1]) / 2.0
        np.testing.assert_allclose(arr[:, j], expect, rtol=1e-6)


def test_hfgrid14_attrs_model_marker(monkeypatch):
    """Manifest provenance fields: product_type=model + license_note + doi."""
    monkeypatch.setattr(th, "_hfgrid14_source",
                        lambda *a, **k: _synth_hfgrid_source(np.zeros((360, 720))))
    attrs = th.build_hfgrid14_heatflow(_hfgrid_entry(), "30min").attrs
    assert attrs["product_type"].startswith("model")
    assert "license_note" in attrs
    assert attrs["doi"] == "10.1029/2019GC008389"
    assert "missing_value" not in attrs and "_FillValue" not in attrs



def test_density_builder_attrs(monkeypatch):
    home = np.zeros((3600, 7200), dtype=np.uint32)
    home[0, 0] = 9
    monkeypatch.setattr(th, "heatflow_home_density", lambda *a, **k: home)
    arr = th.build_heatflow_density(_density_entry(), "1deg")
    vals = arr.compute().values
    assert vals.dtype == np.uint32 and vals[0, 0] == 9
    assert int(vals.sum(dtype=np.uint64)) == 9
    for key in ("count_semantics", "composite_rule", "license_note",
                "source_file", "nodata_semantics"):
        assert key in arr.attrs
    assert "96368" in arr.attrs["count_semantics"]
    assert "missing_value" not in arr.attrs and "_FillValue" not in arr.attrs



@requires_heatflow
def test_real_merge_baseline():
    """Merged triple numbers: GHFDB 91182 all locatable + NGHF supplement 5186 = 96368."""
    g = th.load_ghfdb()
    assert len(g) == GHFDB_EXPECTED_ROWS
    assert g["lat"].notna().all() and g["lon"].notna().all()
    assert g["id"].is_unique
    n = th.load_nghf()
    assert len(n) == NGHF_EXPECTED_ROWS
    assert NGHF_EXPECTED_LOCATABLE == 69727
    df = th.merge_heatflow_points()
    assert len(df) == MERGED_TOTAL
    assert (df["source"] == "GHFDB").sum() == GHFDB_EXPECTED_ROWS
    assert (df["source"] == "NGHF").sum() == NGHF_SUPPLEMENT
    home = th.heatflow_home_density()
    assert home.shape == (3600, 7200)
    assert int(home.sum(dtype=np.uint64)) == MERGED_TOTAL


@requires_heatflow
def test_real_incorporation_evidence():
    """Incorporation-evidence figures: coordinate coincidence / q agreement / pixel overlap."""
    ev = th.verify_nghf_incorporation()
    assert ev["exact_coord_matches"] == NGHF_EXACT_COORD_MATCHES
    assert ev["q_comparable"] == NGHF_Q_COMPARABLE
    assert ev["q_identical"] == NGHF_Q_IDENTICAL
    assert ev["pixel_overlap"] == NGHF_PIXEL_OVERLAP
    assert ev["supplement"] == NGHF_SUPPLEMENT
    assert ev["nghf_locatable"] == NGHF_EXPECTED_LOCATABLE
    assert 0.7 < ev["exact_coord_matches"] / ev["nghf_locatable"] < 0.9


@requires_hfgrid
def test_real_hfgrid14_baseline():
    src = th.load_hfgrid14()
    assert src.values.shape == (360, 720)
    assert src.valid.all()
    assert float(np.nanmin(src.values)) == -6.0
    assert 2900 < float(np.nanmax(src.values)) < 3000
    arr = th.build_hfgrid14_heatflow(_hfgrid_entry(), "30min").compute().values
    assert np.array_equal(arr, src.values)



def test_export_sidecar_missing_source(tmp_path):
    from cubebuild.sidecars import export_heatflow_points_merged

    with pytest.raises(FileNotFoundError):
        export_heatflow_points_merged(tmp_path, ghfdb_src=tmp_path / "nowhere")


@requires_heatflow
def test_export_heatflow_sidecar_roundtrip(tmp_path):
    import geopandas as gpd

    from cubebuild.sidecars import export_heatflow_points_merged

    rec = export_heatflow_points_merged(tmp_path)
    assert rec["id"] == "heatflow_points_merged"
    assert rec["visibility"] == "internal"
    assert rec["n_features"] == MERGED_TOTAL
    assert rec["n_ghfdb"] == GHFDB_EXPECTED_ROWS
    assert rec["n_nghf"] == NGHF_SUPPLEMENT
    back = gpd.read_parquet(tmp_path / "heatflow_points_merged.parquet")
    assert len(back) == MERGED_TOTAL
    assert back.crs.to_epsg() == 4326
    assert back.geometry.isna().sum() == 0
    for col in ("source", "source_row", "name", "lat", "lon", "q",
                "q_uncertainty", "reference", "id"):
        assert col in back.columns
    assert set(back["source"].unique()) == {"GHFDB", "NGHF"}
    assert back.loc[back["source"] == "NGHF", "id"].isna().all()
    g = th.load_ghfdb()
    for i in (0, 50000, GHFDB_EXPECTED_ROWS - 1):
        row = back.iloc[i]
        assert row["id"] == g.iloc[i]["id"]
        assert row["lat"] == pytest.approx(g.iloc[i]["lat"], abs=1e-9)



_MINI = """
version: test
status: test
tiers: [1deg, 30min, 3min]
layers:
  - id: thermal__hfgrid14_heatflow
    source: heatflow（lucazeau2019/HFgrid14.csv）
    native_res: 0.5deg
    home_tier: 30min
    tiers: [1deg, 30min]
    dtype: float32
    unit: mW/m2
    resampling: conservative_area_weighted
    mask: validity + landsea
    notes: 模型产品
  - id: thermal__heatflow_density
    source: heatflow（GHFDB 2024 与 NGHF 合并去重后的单一点集）
    native_res: point
    home_tier: 3min
    tiers: [1deg, 3min]
    dtype: uint32
    unit: count/cell
    resampling: sum
    mask: 免
    notes: 合并去重
"""


@requires_heatflow
@requires_hfgrid
def test_thermal_e2e(tmp_path, monkeypatch):
    """Two-layer full chain: build → structural validation PASS → manifest (count_semantics /
    composite_rule / product_type=model marker / validity link) → fidelity PASS (density
    includes the incorporation-evidence hard criterion) → sidecar round-trip reconstructs the
    density layer (cross-check item)."""
    import cubebuild.cli
    import geopandas as gpd
    import xarray as xr
    from cubebuild.manifest import read_manifest
    from cubebuild.sidecars import export_heatflow_points_merged

    monkeypatch.setattr(cubebuild.cli, "SIDECARS",
                        {"heatflow_points_merged": export_heatflow_points_merged})

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
    assert set(entries) == {HEATFLOW_DENSITY_ID, HFGRID14_ID}
    d = entries[HEATFLOW_DENSITY_ID]
    assert d["dtype"] == "uint32" and d["resampling"] == "sum"
    assert "96368" in d["count_semantics"]
    assert "composite_rule" in d and "0.1°" in d["composite_rule"]
    assert "validity_mask" not in d
    h = entries[HFGRID14_ID]
    assert h["dtype"] == "float32"
    assert h["product_type"].startswith("model")
    assert h["validity_mask"] == "thermal__hfgrid14_heatflow__validity"
    assert h["coverage"] == {"1deg": 1.0, "30min": 1.0}
    assert [s["id"] for s in manifest["sidecars"]] == ["heatflow_points_merged"]

    for eid, tiers in ((HEATFLOW_DENSITY_ID, ["1deg", "3min"]),
                       (HFGRID14_ID, ["1deg", "30min"])):
        report = th.build_thermal_fidelity_report(out, eid, tiers)
        assert report["result"] == "PASS", report["checks"]

    with xr.open_datatree(out, engine="zarr", chunks={}) as dt:
        store_density = dt["/3min"].ds[HEATFLOW_DENSITY_ID].values
    back = gpd.read_parquet(sidecars_dir / "heatflow_points_merged.parquet")
    from cubebuild.points import points_to_density
    np.testing.assert_array_equal(
        points_to_density(back.geometry.y, back.geometry.x, "3min"),
        store_density,
    )
    with xr.open_datatree(out, engine="zarr", chunks={}) as dt:
        store_hf = dt["/30min"].ds[HFGRID14_ID].values
    assert np.array_equal(store_hf, th.load_hfgrid14().values)
