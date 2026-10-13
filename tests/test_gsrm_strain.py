"""Tests: reader contract (synthetic cropped-grid source: grid completeness/trailing
truncation/drift rejection) +
builders (6′ identity/polar-cap gap/1° column pairing/attrs provenance) + real-source
baseline (shape/trailing
truncation amount/four-layer 6′ bitwise identity/value-range contract) + e2e (four layers
into the store, manifest internal
provenance and unit_basis, fidelity PASS).
"""

from pathlib import Path

import numpy as np
import pytest

import cubebuild.gsrm as gs
from cubebuild.gsrm import (
    GRID_NLAT,
    GRID_NLON,
    STRAIN_FILE,
    STRAIN_LAYERS,
    TRAILING_ROWS,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_STRAIN = (gs.GSRM_V22_DIR / STRAIN_FILE).is_file()
requires_gsrm = pytest.mark.skipif(not REAL_STRAIN, reason="GSRM v2.2 解压版源未集齐")


def _strain_entry(layer_id="stress_kinematics__gsrm_exx",
                  unit="nano-strain/yr"):
    from cubebuild.contract import LayerEntry

    return LayerEntry(
        id=layer_id, dims=("lat", "lon"), source="test",
        native_res="0.1deg", native_res_deg=0.1, home_tier="6min",
        tiers=("1deg", "30min", "6min"), dtype="float32",
        unit=unit, resampling="conservative_area_weighted",
        mask="validity", visibility="internal",
    )



def _write_mini_strain(path, nlat, nlon, trailing, header_lines=2, bad_grid=False,
                       trailing_ongrid=False):
    """Synthetic cropped strain file: real geometry (-87.45/-179.95 base, 0.1 step, row-major order)."""
    lines = ["# synthetic header"] * header_lines
    for i in range(nlat * nlon):
        lat = -87.45 + 0.1 * (i // nlon)
        lon = -179.95 + 0.1 * (i % nlon)
        if bad_grid and i == 0:
            lat += 0.01
        lines.append(f"{lat:.3f} {lon:.3f} 1.0 2.0 3.0 4.0 0 0 0 0 0")
    for k in range(trailing):
        if trailing_ongrid and k == 0:
            lines.append(f"{-87.45 + 0.1 * (nlat - 1):.3f} {-179.95:.3f} 9 9 9 9 0 0 0 0 0")
        else:
            lines.append(f"{k + 1}.000 0.000 -87.5 0.0 0.0 0.000 0.0 0.0 0.0 -87.5 180.0")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_loader_mini_grid_and_trailing(tmp_path, monkeypatch):
    """Cropped synthetic source: grid rows loaded in row-major order + trailing block truncation count."""
    nlat, nlon, trailing = 6, 10, 5
    monkeypatch.setattr(gs, "GRID_NLAT", nlat)
    monkeypatch.setattr(gs, "GRID_NLON", nlon)
    monkeypatch.setattr(gs, "TRAILING_ROWS", trailing)
    monkeypatch.setattr(gs, "HEADER_LINES", 2)
    _write_mini_strain(tmp_path / STRAIN_FILE, nlat, nlon, trailing)
    src = gs.load_gsrm_strain(tmp_path)
    assert src.fields["exx"].shape == (nlat, nlon)
    assert src.n_trailing == trailing
    assert src.fields["exx"][0, 0] == np.float32(1.0)
    assert src.fields["vorticity"][nlat - 1, nlon - 1] == np.float32(4.0)
    assert src.lat_edges[0] == pytest.approx(-87.5)
    assert src.lat_edges[-1] == pytest.approx(-87.5 + 0.1 * nlat)
    assert src.lon_res == gs.SRC_TENTH


def test_loader_rejects_wrong_total_rows(tmp_path, monkeypatch):
    nlat, nlon = 4, 8
    monkeypatch.setattr(gs, "GRID_NLAT", nlat)
    monkeypatch.setattr(gs, "GRID_NLON", nlon)
    monkeypatch.setattr(gs, "TRAILING_ROWS", 3)
    monkeypatch.setattr(gs, "HEADER_LINES", 2)
    _write_mini_strain(tmp_path / STRAIN_FILE, nlat, nlon, trailing=7)
    with pytest.raises(ValueError, match="数据形状"):
        gs.load_gsrm_strain(tmp_path)


def test_loader_rejects_grid_coordinate_drift(tmp_path, monkeypatch):
    nlat, nlon = 4, 8
    monkeypatch.setattr(gs, "GRID_NLAT", nlat)
    monkeypatch.setattr(gs, "GRID_NLON", nlon)
    monkeypatch.setattr(gs, "TRAILING_ROWS", 3)
    monkeypatch.setattr(gs, "HEADER_LINES", 2)
    _write_mini_strain(tmp_path / STRAIN_FILE, nlat, nlon, 3, bad_grid=True)
    with pytest.raises(ValueError, match="偏离"):
        gs.load_gsrm_strain(tmp_path)


def test_loader_rejects_ongrid_trailing_row(tmp_path, monkeypatch):
    """A grid-coordinate row hidden in the trailing block → intercepted (guards against silent point loss)."""
    nlat, nlon = 4, 8
    monkeypatch.setattr(gs, "GRID_NLAT", nlat)
    monkeypatch.setattr(gs, "GRID_NLON", nlon)
    monkeypatch.setattr(gs, "TRAILING_ROWS", 3)
    monkeypatch.setattr(gs, "HEADER_LINES", 2)
    _write_mini_strain(tmp_path / STRAIN_FILE, nlat, nlon, 3, trailing_ongrid=True)
    with pytest.raises(ValueError, match="网格坐标"):
        gs.load_gsrm_strain(tmp_path)


def test_loader_missing_source(tmp_path):
    with pytest.raises(FileNotFoundError):
        gs.load_gsrm_strain(tmp_path)



def _synth_strain_source(nlat_rows=10, seed=5):
    """Synthetic StrainSource: real full-circle longitude geometry (3600 columns) + latitude
    sub-band.

    lat_edges uses the same formula as the real reader (-90 + 0.1·k, starting at k=25) — going
    through the
    -87.5+0.1·k path would have a 1 ulp misalignment with the tier edges (as the module comment
    states), making the bitwise
    identity assertion fragile.
    """
    rng = np.random.default_rng(seed)
    nlon = 3600
    fields = {
        name: rng.uniform(-100, 100, (nlat_rows, nlon)).astype(np.float32)
        for name in ("exx", "eyy", "exy", "vorticity")
    }
    return gs.StrainSource(
        fields=fields,
        lat_edges=-90.0 + 0.1 * np.arange(25, 25 + nlat_rows + 1, dtype=np.float64),
        lon_res=gs.SRC_TENTH,
        lon_phase=gs.LON_PHASE,
        n_trailing=TRAILING_ROWS,
    )


def test_build_6min_identity(monkeypatch):
    """6′ tier identity (0.1° source = 6′, cell edges aligned + phase 0.05): bitwise identical."""
    src = _synth_strain_source()
    monkeypatch.setattr(gs, "_strain_source", lambda *a, **k: src)
    arr = gs.build_gsrm_strain_layer(_strain_entry(), "6min").compute().values
    assert arr.shape == (1800, 3600)
    np.testing.assert_array_equal(arr[25:35], src.fields["exx"])
    assert np.isnan(arr[:25]).all() and np.isnan(arr[35:]).all()


def test_build_1deg_column_pairing(monkeypatch):
    """1° tier = 10-column paired area weighting: a column-identity field → the mean of 10 adjacent columns is exact."""
    nlon = 3600
    vals = np.tile(np.arange(nlon, dtype=np.float64), (10, 1))
    src = _synth_strain_source()
    src.fields["exx"] = vals.astype(np.float32)
    monkeypatch.setattr(gs, "_strain_source", lambda *a, **k: src)
    arr = gs.build_gsrm_strain_layer(_strain_entry(), "1deg").compute().values
    for j in (0, 17, 359):
        expect = vals[0, 10 * j:10 * j + 10].mean()
        np.testing.assert_allclose(arr[2, j], expect, rtol=1e-6)


def test_build_attrs_provenance(monkeypatch):
    """manifest provenance fields: internal visibility + unit_basis + license_note + doi."""
    src = _synth_strain_source()
    monkeypatch.setattr(gs, "_strain_source", lambda *a, **k: src)
    attrs = gs.build_gsrm_strain_layer(_strain_entry(), "6min").attrs
    assert attrs["visibility"] == "internal"
    assert "1e-9/yr" in attrs["unit_basis"] and "README" in attrs["unit_basis"]
    assert "CC-BY-NC-SA" in attrs["license_note"]
    assert attrs["doi"] == "10.1002/2014GC005407"
    assert attrs["product_type"].startswith("model")
    assert "v2.2" in attrs["product_type"]
    assert "5351" in attrs["registration_rule"]
    assert attrs["units"] == "nano-strain/yr"
    assert "missing_value" not in attrs and "_FillValue" not in attrs


def test_build_vorticity_entry_unit(monkeypatch):
    """vorticity layer: unit nano-rad/yr + the same builder routing."""
    src = _synth_strain_source()
    monkeypatch.setattr(gs, "_strain_source", lambda *a, **k: src)
    entry = _strain_entry("stress_kinematics__gsrm_vorticity", unit="nano-rad/yr")
    arr = gs.build_gsrm_strain_layer(entry, "6min")
    assert arr.attrs["units"] == "nano-rad/yr"
    np.testing.assert_array_equal(
        arr.compute().values[25:35], src.fields["vorticity"]
    )



@requires_gsrm
def test_real_strain_source_baseline():
    """Shape 1750×3600 / all finite / trailing truncation 5351 / value-range contract."""
    src = gs.load_gsrm_strain()
    assert src.fields["exx"].shape == (GRID_NLAT, GRID_NLON) == (1750, 3600)
    assert src.n_trailing == TRAILING_ROWS == 5351
    for name, v in src.fields.items():
        assert np.isfinite(v).all()
    ranges = {
        "exx": (-12142.4, 11638.6), "eyy": (-12146.2, 14150.9),
        "exy": (-17129.9, 10840.5), "vorticity": (-895.326, 1210.912),
    }
    for name, (lo, hi) in ranges.items():
        v = src.fields[name]
        assert abs(float(v.min()) - lo) <= 0.01, name
        assert abs(float(v.max()) - hi) <= 0.01, name
    assert src.lat_edges[0] == pytest.approx(-87.5)
    assert src.lat_edges[-1] == pytest.approx(87.5)


@requires_gsrm
def test_real_6min_identity_all_fields():
    """Bitwise identity of the four layers at the 6′ tier (real source, through the full builder chain)."""
    src = gs._strain_source()
    for layer_id, field in gs.STRAIN_FIELD_OF.items():
        unit = "nano-rad/yr" if field == "vorticity" else "nano-strain/yr"
        entry = _strain_entry(layer_id, unit=unit)
        arr = gs.build_gsrm_strain_layer(entry, "6min").compute().values
        np.testing.assert_array_equal(arr[25:1775], src.fields[field])
        assert np.isnan(arr[:25]).all() and np.isnan(arr[1775:]).all()



_MINI = """
version: test
status: test
tiers: [1deg, 30min, 6min]
layers:
  - id: stress_kinematics__gsrm_exx
    source: gsrm-strain（v2.2/GSRM_strain.txt.Z 解压）
    native_res: 0.1deg
    home_tier: 6min
    tiers: [1deg, 30min, 6min]
    dtype: float32
    unit: nano-strain/yr
    resampling: conservative_area_weighted
    mask: validity
    visibility: internal
    notes: test
  - id: stress_kinematics__gsrm_eyy
    source: gsrm-strain（v2.2/GSRM_strain.txt.Z 解压）
    native_res: 0.1deg
    home_tier: 6min
    tiers: [1deg, 30min, 6min]
    dtype: float32
    unit: nano-strain/yr
    resampling: conservative_area_weighted
    mask: validity
    visibility: internal
    notes: test
  - id: stress_kinematics__gsrm_exy
    source: gsrm-strain（v2.2/GSRM_strain.txt.Z 解压）
    native_res: 0.1deg
    home_tier: 6min
    tiers: [1deg, 30min, 6min]
    dtype: float32
    unit: nano-strain/yr
    resampling: conservative_area_weighted
    mask: validity
    visibility: internal
    notes: test
  - id: stress_kinematics__gsrm_vorticity
    source: gsrm-strain（v2.2/GSRM_strain.txt.Z 解压）
    native_res: 0.1deg
    home_tier: 6min
    tiers: [1deg, 30min, 6min]
    dtype: float32
    unit: nano-rad/yr
    resampling: conservative_area_weighted
    mask: validity
    visibility: internal
    notes: test
"""


@requires_gsrm
def test_gsrm_strain_e2e(tmp_path, monkeypatch):
    """Four-layer full chain: build → structural validation PASS → manifest
    (internal/unit_basis/
    validity link/coverage polar-cap gap) → fidelity PASS ×4 (including source-contract hard
    checks)."""
    import cubebuild.cli
    from cubebuild.manifest import read_manifest

    monkeypatch.setattr(cubebuild.cli, "SIDECARS", {})

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
    entries = {e["id"]: e for e in manifest["layers"]}
    assert set(entries) == set(STRAIN_LAYERS)
    for lid, e in entries.items():
        assert e["visibility"] == "internal"
        assert "README" in e["unit_basis"] and "1e-9/yr" in e["unit_basis"]
        assert e["validity_mask"] == f"{lid}__validity"
        assert e["coverage"]["6min"] == pytest.approx(1750 / 1800, abs=1e-9)
        assert e["coverage"]["1deg"] == pytest.approx(176 / 180, abs=1e-9)
        assert e["coverage"]["30min"] == pytest.approx(350 / 360, abs=1e-9)
    assert entries["stress_kinematics__gsrm_vorticity"]["unit"] == "nano-rad/yr"

    for lid in STRAIN_LAYERS:
        report = gs.build_gsrm_strain_fidelity_report(out, lid, ["1deg", "30min", "6min"])
        assert report["result"] == "PASS", report["checks"]
