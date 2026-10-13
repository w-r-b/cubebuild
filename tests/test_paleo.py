"""Tests: Ma label parsing / scan sorting + dual-product read contracts
(north-first descending flip / seam merge / ±180 bitwise-identical assertion) +
Voronoi edge construction + conservative-registration analytic criteria (nested
identity of the 0.2° synthetic grid + the 1° GLAD-style adjacent-node mean
formula) + 4D write-structure e2e (group node / time coordinate / Ma indexing /
mask companion / manifest provenance / structural validation PASS / fidelity
PASS / two-run determinism) + real-source baselines (109-slice label set / read
contract / kernel-mapping spot check).
Addendum: the two 1° slices' labels align by the official integer convention
(385.2→385, 390.5→390), and the two tiers' label sets are 109/109
value-for-value equal (synthetic alignment unit test + real-source acceptance
assertion).
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import cubebuild.paleo as pq
from cubebuild.cli import main
from cubebuild.etopo import SourceNotReady
from cubebuild.manifest import read_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_SIXMIN = (pq.SIXMIN_DIR / "Map01_PALEOMAP_6min_Holocene_0Ma.nc").is_file()
REAL_ONEDEG = (pq.ONEDEG_DIR / "Map01_PALEOMAP_1deg_Holocene_0Ma.nc").is_file()
requires_paleo = pytest.mark.skipif(
    not (REAL_SIXMIN and REAL_ONEDEG), reason="PaleoDEM 源未集齐"
)


def _entry(layer_id="paleo__paleodem_elevation"):
    from cubebuild.contract import LayerEntry

    return LayerEntry(
        id=layer_id, dims=("time", "lat", "lon"), source="paleodem-scotese",
        native_res="6min", native_res_deg=1.0 / 10.0, home_tier="6min",
        tiers=("1deg", "6min"), dtype="float32", unit="m",
        resampling="none", mask="validity（古地理随时间变化，逐时间片掩膜）",
    )



def test_parse_ma_label():
    assert pq.parse_ma_label("Map01_PALEOMAP_6min_Holocene_0Ma.nc") == 0.0
    assert pq.parse_ma_label("Map43.5_PALEOMAP_6min_Late_Triassic_205Ma.nc") == 205.0
    assert pq.parse_ma_label("Map67.5_PALEOMAP_1deg_Late_Devonian_385.2Ma.nc") == 385.2
    assert pq.parse_ma_label("Map24_PALEOMAP_6min_Early Cretaceous_105Ma.nc") == 105.0
    with pytest.raises(ValueError, match="Ma 标签"):
        pq.parse_ma_label("Map01_PALEOMAP_6min_Holocene.nc")


def test_scan_slices_sorts_by_ma_not_filename(tmp_path):
    """Filename lexicographic order ≠ Ma order (Map43.5_205Ma precedes
    Map43_200Ma) — sort by the parsed value."""
    for name in (
        "Map43.5_PALEOMAP_6min_Late_Triassic_205Ma.nc",
        "Map43_PALEOMAP_6min_Late_Triassic_200Ma.nc",
        "Map01_PALEOMAP_6min_Holocene_0Ma.nc",
    ):
        (tmp_path / name).write_bytes(b"")
    out = pq.scan_slices(tmp_path, expected=3)
    assert [ma for ma, _ in out] == [0.0, 200.0, 205.0]


def test_scan_slices_count_and_duplicate_contracts(tmp_path):
    (tmp_path / "a_0Ma.nc").write_bytes(b"")
    with pytest.raises(SourceNotReady):
        pq.scan_slices(tmp_path, expected=109)
    (tmp_path / "b_0Ma.nc").write_bytes(b"")
    with pytest.raises(ValueError, match="重复"):
        pq.scan_slices(tmp_path, expected=2)
    with pytest.raises(SourceNotReady):
        pq.scan_slices(tmp_path / "nonexistent")


def test_scan_slices_ma_range_contract(tmp_path):
    """'accept all 0–540 Ma': a label-range drift → explicit failure at build
    time (code-review fix)."""
    for name in ("a_0Ma.nc", "b_600Ma.nc"):
        (tmp_path / name).write_bytes(b"")
    with pytest.raises(ValueError, match="超出契约"):
        pq.scan_slices(tmp_path, expected=2)




def test_scan_slices_1deg_alignment(tmp_path):
    """tier="1deg": 385.2/390.5 rounded by the official integer convention
    (385/390), sorted by the aligned labels; paths follow the original files
    (the data arrays are untouched, time coordinates are the only change
    surface); 6min and bare scans are untouched."""
    for name in ("a_380Ma.nc", "b_385.2Ma.nc", "c_390.5Ma.nc", "d_395Ma.nc"):
        (tmp_path / name).write_bytes(b"")
    out = pq.scan_slices(tmp_path, expected=4, tier="1deg")
    assert [ma for ma, _ in out] == [380.0, 385.0, 390.0, 395.0]
    assert out[1][1].name == "b_385.2Ma.nc"
    assert out[2][1].name == "c_390.5Ma.nc"
    raw = [ma for ma, _ in pq.scan_slices(tmp_path, expected=4)]
    assert raw == [380.0, 385.2, 390.5, 395.0]
    assert [ma for ma, _ in pq.scan_slices(tmp_path, expected=4, tier="6min")] == raw


def test_scan_slices_alignment_duplicate_guard(tmp_path):
    """Alignment introduces a duplicate (source holds both 385 and 385.2) →
    hard failure on source drift (the duplicate check applies to the aligned
    labels, preventing alignment from silently merging two slices)."""
    for name in ("a_385Ma.nc", "b_385.2Ma.nc"):
        (tmp_path / name).write_bytes(b"")
    with pytest.raises(ValueError, match="重复"):
        pq.scan_slices(tmp_path, expected=2, tier="1deg")



SYN_NLAT, SYN_NLON = 901, 1801
SYN_RES = 0.2


def _syn_field_6min(nlat=SYN_NLAT, nlon=SYN_NLON, res=SYN_RES, base=100.0):
    """Separable analytic field v(lat, lon) = lat + 0.01·lon + base, the ±180
    columns offset by +1/-1 each."""
    lat = 90.0 - res * np.arange(nlat, dtype=np.float64)
    lon = -180.0 + res * np.arange(nlon, dtype=np.float64)
    v = (lat[:, None] + 0.01 * lon[None, :] + base).astype(np.float32)
    raw = v.copy()
    raw[:, 0] += 1.0
    raw[:, -1] -= 1.0
    return lat, lon, v, raw


def _make_synthetic_6min(dir_path: Path, ma: float):
    lat, lon, _v, raw = _syn_field_6min()
    ds = xr.Dataset(
        {"z": (("latitude", "longitude"), raw)},
        coords={"latitude": lat, "longitude": lon},
    )
    path = dir_path / f"Map_test_PALEOMAP_6min_{ma:g}Ma.nc"
    ds.to_netcdf(path)
    return path


def _make_synthetic_1deg(dir_path: Path, ma: float, base=100.0):
    """1° synthetic (real geometry 181×361): south-first ascending, ±180
    bitwise identical (official semantics)."""
    lat = -90.0 + np.arange(181, dtype=np.float64)
    lon = -180.0 + np.arange(361, dtype=np.float64)
    v = (lat[:, None] + 0.01 * lon[None, :] + base).astype(np.float32)
    v[:, -1] = v[:, 0]
    ds = xr.Dataset(
        {"z": (("lat", "lon"), v)},
        coords={"lat": lat, "lon": lon},
    )
    path = dir_path / f"Map_test_PALEOMAP_1deg_{ma:g}Ma.nc"
    ds.to_netcdf(path)
    return path


@pytest.fixture()
def syn6_geometry(monkeypatch):
    """Inject the synthetic 6min geometry contract (0.2° reduced grid) into
    module constants."""
    monkeypatch.setattr(pq, "SIXMIN_NLAT", SYN_NLAT)
    monkeypatch.setattr(pq, "SIXMIN_NLON", SYN_NLON)



def test_read_slice_6min_contract_and_seam_merge(tmp_path, syn6_geometry):
    path = _make_synthetic_6min(tmp_path, 0.0)
    a = pq.read_slice_6min(path)
    assert a.shape == (SYN_NLAT, SYN_NLON - 1)
    assert a.dtype == np.float32
    lat_nodes = -90.0 + SYN_RES * np.arange(SYN_NLAT)
    lon_nodes = -180.0 + SYN_RES * np.arange(SYN_NLON - 1)
    assert np.allclose(a[0, 0], -90.0 + 100.0, atol=1e-4)
    assert np.allclose(a[:, 0], lat_nodes + 100.0, atol=1e-4)
    assert np.allclose(a[:, 1], lat_nodes + 0.01 * lon_nodes[1] + 100.0, atol=1e-4)
    assert np.allclose(a[:, -1], lat_nodes + 0.01 * lon_nodes[-1] + 100.0, atol=1e-4)


def test_read_slice_6min_rejects_drift(tmp_path, syn6_geometry):
    """Contract drift fails explicitly: dimensions / latitude sign flip /
    non-finite values."""
    ds = xr.Dataset(
        {"z": (("latitude", "longitude"), np.zeros((11, 12), dtype=np.float32))},
        coords={"latitude": np.arange(10.0, -1.0, -1.0), "longitude": np.arange(-6.0, 6.0)},
    )
    ds.to_netcdf(tmp_path / "bad_dims_0Ma.nc")
    with pytest.raises(ValueError, match="dims"):
        pq.read_slice_6min(tmp_path / "bad_dims_0Ma.nc")
    lat, lon, _v, raw = _syn_field_6min()
    ds = xr.Dataset(
        {"z": (("latitude", "longitude"), raw[::-1])},
        coords={"latitude": lat[::-1], "longitude": lon},
    )
    ds.to_netcdf(tmp_path / "bad_lat_0Ma.nc")
    with pytest.raises(ValueError, match="纬/经度坐标"):
        pq.read_slice_6min(tmp_path / "bad_lat_0Ma.nc")
    lat, lon, _v, raw = _syn_field_6min()
    raw[5, 5] = np.nan
    ds = xr.Dataset(
        {"z": (("latitude", "longitude"), raw)},
        coords={"latitude": lat, "longitude": lon},
    )
    ds.to_netcdf(tmp_path / "bad_nan_0Ma.nc")
    with pytest.raises(ValueError, match="非有限值"):
        pq.read_slice_6min(tmp_path / "bad_nan_0Ma.nc")
    lat, lon, _v, raw = _syn_field_6min()
    raw[5, 5] = 9.9e4
    ds = xr.Dataset(
        {"z": (("latitude", "longitude"), raw)},
        coords={"latitude": lat, "longitude": lon},
    )
    ds.to_netcdf(tmp_path / "bad_range_0Ma.nc")
    with pytest.raises(ValueError, match="值域"):
        pq.read_slice_6min(tmp_path / "bad_range_0Ma.nc")


def test_read_slice_1deg_contract(tmp_path):
    path = _make_synthetic_1deg(tmp_path, 0.0)
    a = pq.read_slice_1deg(path)
    assert a.shape == (181, 360)
    assert a.dtype == np.float32
    lat = -90.0 + np.arange(181, dtype=np.float64)
    assert np.allclose(a[0, 0], -90.0 + 0.01 * -180.0 + 100.0)
    assert np.allclose(a[:, -1], lat + 0.01 * 179.0 + 100.0)
    lat, lon, _v, raw = _syn_field_6min(nlat=181, nlon=361, res=1.0)
    ds = xr.Dataset(
        {"z": (("lat", "lon"), raw)},
        coords={"lat": -90.0 + np.arange(181.0), "lon": -180.0 + np.arange(361.0)},
    )
    ds.to_netcdf(tmp_path / "bad_seam_0Ma.nc")
    with pytest.raises(ValueError, match="±180"):
        pq.read_slice_1deg(tmp_path / "bad_seam_0Ma.nc")



def test_voronoi_lat_edges_real_geometry():
    """Real geometry: 6min 1802 edges (midpoints clamped at ±90), 1° 182 edges
    (GLAD-style)."""
    e6 = pq.voronoi_lat_edges(pq.lat_nodes_asc("6min"))
    assert len(e6) == 1802
    assert e6[0] == -90.0 and e6[-1] == 90.0
    assert e6[1] == pytest.approx(-89.95) and e6[-2] == pytest.approx(89.95)
    e1 = pq.voronoi_lat_edges(pq.lat_nodes_asc("1deg"))
    assert len(e1) == 182
    assert e1[1] == pytest.approx(-89.5) and e1[-2] == pytest.approx(89.5)


def test_paleo_tier_field_6min_nested_identity(tmp_path, syn6_geometry):
    """0.2° synthetic source: the Voronoi edges coincide with the target-tier
    edges → each target cell is fully nested in a single source cell, target
    value = source node value (2 target cells per row/column ↔ 1 source cell),
    checked by the analytic identity."""
    path = _make_synthetic_6min(tmp_path, 0.0)
    merged = pq.read_slice_6min(path)                       # (901, 1800)
    out = pq.paleo_tier_field("6min", path)
    assert out.shape == (1800, 3600)
    assert np.isfinite(out).all()
    for i in (0, 1, 2, 3, 899, 900, 1798, 1799):
        for j in (0, 1, 2, 3598, 3599):
            expect = merged[(i + 1) // 2, ((j + 1) // 2) % merged.shape[1]]
            assert out[i, j] == pytest.approx(expect, abs=1e-4), (i, j)


def test_paleo_tier_field_1deg_analytic(tmp_path):
    """1° synthetic source (GLAD-style geometry): each target cell = adjacent
    nodes equally weighted in longitude and sin-weighted in latitude; polar rows
    have a half-cell full-weight overlap — the analytic formula is checked
    bitwise (including the j=359 dateline wrap)."""
    path = _make_synthetic_1deg(tmp_path, 0.0)
    a = pq.read_slice_1deg(path)
    out = pq.paleo_tier_field("1deg", path)
    assert out.shape == (180, 360)

    la = -90.0 + np.arange(181, dtype=np.float64)
    lo = -180.0 + np.arange(360, dtype=np.float64)

    def node(i, j):
        return la[i] + 0.01 * lo[j] + 100.0

    for i in (0, 1, 90, 179):
        e_mid = 0.5 * (la[i] + la[i + 1])
        w0 = math.sin(math.radians(e_mid)) - math.sin(math.radians(la[i]))
        w1 = math.sin(math.radians(la[i + 1])) - math.sin(math.radians(e_mid))
        for j in (0, 179, 359):
            j1 = (j + 1) % 360
            expect = (
                w0 * (node(i, j) + node(i, j1))
                + w1 * (node(i + 1, j) + node(i + 1, j1))
            ) / (2.0 * (w0 + w1))
            assert out[i, j] == pytest.approx(expect, abs=1e-4), (i, j)


def test_paleo_constant_field_invariant(tmp_path, syn6_geometry):
    """A constant field is unchanged by conservative averaging (including seam
    merge) — a constructive check of conservation."""
    lat = 90.0 - SYN_RES * np.arange(SYN_NLAT)
    lon = -180.0 + SYN_RES * np.arange(SYN_NLON)
    ds = xr.Dataset(
        {"z": (("latitude", "longitude"), np.full((SYN_NLAT, SYN_NLON), 4.5, np.float32))},
        coords={"latitude": lat, "longitude": lon},
    )
    ds.to_netcdf(tmp_path / "const_0Ma.nc")
    out = pq.paleo_tier_field("6min", tmp_path / "const_0Ma.nc")
    assert np.allclose(out, 4.5, atol=1e-6)



_E2E_CONTRACT = """
version: test
status: test
tiers: [1deg, 6min]
layers:
  - id: paleo__paleodem_elevation
    source: paleodem-scotese（paleodem-6min-nc/*.nc；paleodem-1deg-nc/*.nc）
    native_res: 6min
    dims: [time, lat, lon]
    home_tier: 6min
    tiers: [1deg, 6min]
    dtype: float32
    unit: m
    resampling: none（1deg 档直接用官方 1° nc，免降采样）
    mask: validity（古地理随时间变化，逐时间片掩膜）
"""


@pytest.fixture()
def e2e_mapping(tmp_path):
    p = tmp_path / "e2e-mapping.yaml"
    p.write_text(_E2E_CONTRACT, encoding="utf-8")
    return p


@pytest.fixture()
def synthetic_sources(tmp_path, monkeypatch):
    """Synthesize the dual-product sources (6min 0.2° reduced grid × 2 slices +
    1° real geometry × 2 slices)."""
    monkeypatch.setattr(pq, "SIXMIN_NLAT", SYN_NLAT)
    monkeypatch.setattr(pq, "SIXMIN_NLON", SYN_NLON)
    monkeypatch.setattr(pq, "N_SLICES", 2)
    d6, d1 = tmp_path / "paleodem-6min-nc", tmp_path / "paleodem-1deg-nc"
    d6.mkdir(); d1.mkdir()
    for ma in (0.0, 5.0):
        _make_synthetic_6min(d6, ma)
        _make_synthetic_1deg(d1, ma)
    monkeypatch.setattr(pq, "SIXMIN_DIR", d6)
    monkeypatch.setattr(pq, "ONEDEG_DIR", d1)


@pytest.fixture(autouse=True)
def _no_sidecars(monkeypatch):
    import cubebuild.cli

    monkeypatch.setattr(cubebuild.cli, "SIDECARS", {})


def test_paleo_4d_e2e(tmp_path, e2e_mapping, synthetic_sources):
    out = tmp_path / "cube.zarr"
    reports = tmp_path / "reports"
    rc = main(["build", "--mapping", str(e2e_mapping), "--out", str(out),
               "--reports", str(reports)])
    assert rc == 0

    dt = xr.open_datatree(out, engine="zarr")
    assert "paleodem" in dt["/1deg"].children
    assert "paleodem" in dt["/6min"].children
    n6 = dt["/6min/paleodem"].ds
    n1 = dt["/1deg/paleodem"].ds

    for node in (n6, n1):
        assert node["paleo__paleodem_elevation"].dims == ("time", "lat", "lon")
        assert node.sizes["time"] == 2
        t = node["paleo__paleodem_elevation"].coords["time"]
        assert t.attrs["units"] == "Ma"
        assert np.array_equal(t.values, [0.0, 5.0])
    slice5 = dt["/6min/paleodem"].sel(time=5.0)
    assert slice5.ds.sizes == {"lat": 1800, "lon": 3600}

    merged = pq.read_slice_6min(pq.SIXMIN_DIR / "Map_test_PALEOMAP_6min_0Ma.nc")
    v6 = n6["paleo__paleodem_elevation"].values[0]
    for i in (0, 900, 1799):
        for j in (0, 1799, 3599):
            assert v6[i, j] == pytest.approx(
                merged[(i + 1) // 2, ((j + 1) // 2) % merged.shape[1]], abs=1e-4
            )
    a1 = pq.read_slice_1deg(pq.ONEDEG_DIR / "Map_test_PALEOMAP_1deg_0Ma.nc")
    v1 = n1["paleo__paleodem_elevation"].values[0]
    la = -90.0 + np.arange(181.0); lo = -180.0 + np.arange(360.0)
    i, j = 90, 179
    e_mid = 0.5 * (la[i] + la[i + 1])
    w0 = math.sin(math.radians(e_mid)) - math.sin(math.radians(la[i]))
    w1 = math.sin(math.radians(la[i + 1])) - math.sin(math.radians(e_mid))
    expect = (w0 * (a1[i, j] + a1[i, j + 1]) + w1 * (a1[i + 1, j] + a1[i + 1, j + 1])) / (2 * (w0 + w1))
    assert v1[i, j] == pytest.approx(expect, abs=1e-4)

    m6 = n6["paleo__paleodem_elevation__validity"].values
    m1 = n1["paleo__paleodem_elevation__validity"].values
    assert m6.shape == (2, 1800, 3600) and bool((m6 == 1).all())
    assert m1.shape == (2, 180, 360) and bool((m1 == 1).all())

    manifest = read_manifest(out)
    entries = {m["id"]: m for m in manifest["layers"]}
    me = entries["paleo__paleodem_elevation"]
    assert me["dims"] == ["time", "lat", "lon"]
    assert me["time_levels"] == 2
    assert me["datatree_node"] == "paleodem"
    assert "385.2/390.5" in me["time_semantics"]
    assert "109/109" in me["time_semantics"]
    assert me["validity_mask"] == "paleo__paleodem_elevation__validity"
    assert me["coverage"] == {"1deg": 1.0, "6min": 1.0}
    assert me["license_note"].startswith("CC BY 4.0")

    report = json.loads((reports / "structure-validation.json").read_text())
    assert report["result"] == "PASS", [
        c for c in report["checks"] if c["status"] == "FAIL"
    ]

    rc = main(["fidelity", "--store", str(out), "--layer", "paleo__paleodem_elevation",
               "--mapping", str(e2e_mapping), "--reports", str(reports)])
    assert rc == 0
    fid = json.loads((reports / "fidelity-paleo__paleodem_elevation.json").read_text())
    assert fid["result"] == "PASS", [c for c in fid["checks"] if c["status"] == "FAIL"]


def test_paleo_4d_determinism(tmp_path, e2e_mapping, synthetic_sources):
    outs = []
    for name in ("run1.zarr", "run2.zarr"):
        rc = main(["build", "--mapping", str(e2e_mapping), "--out", str(tmp_path / name),
                   "--reports", str(tmp_path / "reports")])
        assert rc == 0
        outs.append(tmp_path / name)
    from cubebuild.checksum import checksums_equal, store_checksums

    assert checksums_equal(store_checksums(outs[0]), store_checksums(outs[1]))



@requires_paleo
def test_real_scan_labels_contract():
    """Raw scan convention (tier=None): the on-disk baseline facts of the
    filename label sets (for the two tiers' post-alignment agreement see
    test_real_label_alignment)."""
    s6 = pq.scan_slices(pq.SIXMIN_DIR)
    s1 = pq.scan_slices(pq.ONEDEG_DIR)
    l6 = [ma for ma, _ in s6]; l1 = [ma for ma, _ in s1]
    assert len(l6) == len(l1) == 109
    assert all(b > a for a, b in zip(l6, l6[1:]))
    assert l6[0] == 0.0 and l6[-1] == 540.0
    assert sorted(set(l6) - set(l1)) == [385.0, 390.0]
    assert sorted(set(l1) - set(l6)) == [385.2, 390.5]


@requires_paleo
def test_real_label_alignment():
    """Acceptance assertion: the two 1° slices are rounded by the official
    integer naming convention (385.2→385, 390.5→390), and the two tiers' time
    label sets are 109/109 value-for-value equal; 6min is untouched; alignment
    keys on the label value rather than the Map sheet number (exactly the two
    decimal-named files are relabelled, while the Guadalupian numbering-offset
    family is untouched)."""
    s6 = pq.scan_slices(pq.SIXMIN_DIR, tier="6min")
    s1 = pq.scan_slices(pq.ONEDEG_DIR, tier="1deg")
    l6 = [ma for ma, _ in s6]
    l1 = [ma for ma, _ in s1]
    assert len(l1) == len(l6) == 109
    assert l1 == l6
    deviants = sorted(
        (pq.parse_ma_label(p.name), ma) for ma, p in s1
        if pq.parse_ma_label(p.name) != ma
    )
    assert deviants == [(385.2, 385.0), (390.5, 390.0)]
    assert all(pq.parse_ma_label(p.name) == ma for ma, p in s6)
    raw1 = sorted(pq.parse_ma_label(p.name) for _, p in s1)
    assert [ma for ma in raw1 if ma not in (385.2, 390.5)] == [
        ma for ma in l6 if ma not in (385.0, 390.0)
    ]


@requires_paleo
def test_real_read_slice_6min_seam_merge():
    path = pq.SIXMIN_DIR / "Map01_PALEOMAP_6min_Holocene_0Ma.nc"
    a = pq.read_slice_6min(path)
    assert a.shape == (1801, 3600)
    assert np.isfinite(a).all()
    raw = xr.open_dataset(path)["z"].values
    assert np.allclose(a[:, 0], 0.5 * (raw[::-1][:, 0] + raw[::-1][:, -1]), atol=1e-3)
    assert np.array_equal(a[:, 1:], raw[::-1][:, 1:-1])


@requires_paleo
def test_real_read_slice_1deg_seam_identical():
    path = pq.ONEDEG_DIR / "Map01_PALEOMAP_1deg_Holocene_0Ma.nc"
    a = pq.read_slice_1deg(path)
    assert a.shape == (181, 360)
    assert np.isfinite(a).all()
    raw = xr.open_dataset(path)["z"].values
    assert np.array_equal(a, raw[:, :-1])


@requires_paleo
def test_real_6min_kernel_spot():
    """Analytic spot check of the real source's kernel mapping (sin-weighted
    mean of adjacent nodes at the 0.05 Voronoi boundary)."""
    path = pq.SIXMIN_DIR / "Map01_PALEOMAP_6min_Holocene_0Ma.nc"
    a = pq.read_slice_6min(path)
    out = pq.paleo_tier_field("6min", path)
    i, j = 900, 179
    e_mid = -90.0 + 0.1 * i + 0.05
    w0 = math.sin(math.radians(e_mid)) - math.sin(math.radians(-90.0 + 0.1 * i))
    w1 = math.sin(math.radians(-90.0 + 0.1 * (i + 1))) - math.sin(math.radians(e_mid))
    n00, n01 = a[i, j], a[i, j + 1]
    n10, n11 = a[i + 1, j], a[i + 1, j + 1]
    expect = (w0 * (n00 + n01) + w1 * (n10 + n11)) / (2 * (w0 + w1))
    assert out[i, j] == pytest.approx(expect, abs=1e-3)
