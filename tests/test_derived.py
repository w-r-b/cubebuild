"""Tests: three derived layers (land/sea mask / slope / relief).

Checks test external behavior only:
- mode kernel: hand-computed majority + tie broken to the smaller value + non-integer factor
  rejected;
- slope: exact values of an analytic linear field (formula correctness) + an independent numpy
  reference implementation (dask pipeline correctness)
  + longitude periodic boundary (dateline wrap) + latitude boundary one-sided difference;
- relief: brute-force window loop reference implementation (periodic longitude + clipped
  latitude) + hand-computed small case;
- mini end-to-end: 288 synthetic tiles (linear coordinate field) → CLI build →
  structural validation PASS → manifest derived-layer registration
  (derived_from/derivation/calibration note)
  → three-layer fidelity PASS → two-run determinism;
- mask ice-surface calibration: surface-preferred tile selection → new/old mask differing
  pixel set
  ≡ subglacial basin pixel set (composite elevation <0 and ice-surface >0) asserted bitwise
  (synthetic world).

Mini geometry: source tile 300×300 (1/20°), tiers [1deg, 30min, 6min]; landsea's
native tier is elastically monkeypatched to 6min (the real semantics 30sec depends on a 15s
source, which a mini source coarser than
30sec cannot form; the code path is identical, and the 30sec native semantics are borne by
the real-build acceptance).
"""

from pathlib import Path

import dask.array as da
import numpy as np
import pytest

from cubebuild.derived import (
    R_M,
    landsea_threshold,
    mode_downsample,
    relief_core,
    slope_core,
)
from cubebuild.etopo import LAT_BANDS, LON_COLS, label_bounds
from cubebuild.grids import tier_deg
from cubebuild.kernels import aggregate_mode

REPO_ROOT = Path(__file__).resolve().parents[1]



def test_aggregate_mode_majority_and_tie():
    data = np.array([[1, 1], [1, 0]], dtype=np.uint8)
    assert aggregate_mode(data, 2)[0, 0] == 1          # 3:1 → 1
    tie = np.array([[1, 0], [0, 0]], dtype=np.uint8)
    assert aggregate_mode(tie, 2)[0, 0] == 0


def test_aggregate_mode_multiclass_tie_smallest():
    data = np.array([[5, 2, 5, 5], [2, 5, 5, 2]], dtype=np.uint8)
    out = aggregate_mode(data, 2)
    assert out.shape == (1, 2)
    assert out[0, 0] == 2
    assert out[0, 1] == 5


def test_aggregate_mode_rejects_noninteger_factor():
    with pytest.raises(ValueError):
        aggregate_mode(np.zeros((3, 4), dtype=np.uint8), 2)


def test_mode_downsample_matches_numpy():
    rng = np.random.default_rng(3)
    z = rng.integers(0, 3, size=(6, 8)).astype(np.uint8)
    zz = da.from_array(z, chunks=(2, 3))
    with pytest.raises(ValueError):
        mode_downsample(zz, 4)
    zz = da.from_array(z, chunks=(2, 4))
    np.testing.assert_array_equal(
        mode_downsample(zz, 2).compute(), aggregate_mode(z, 2)
    )



def _small_grid(nlat=5, nlon=7, res=0.5):
    lat = -1.0 + (np.arange(nlat) + 0.5) * res
    lon = -1.75 + (np.arange(nlon) + 0.5) * res
    return lat, lon


def test_slope_flat_is_zero():
    lat, _ = _small_grid()
    z = da.full((5, 7), 123.0, chunks=(2, 3))
    out = slope_core(z, lat, 0.5).compute()
    assert out.shape == (5, 7)
    np.testing.assert_array_equal(out, 0.0)


def test_slope_analytic_linear_field():
    """Under a linear field (radian coordinates) both central and one-sided differences are
    analytically exact — formula correctness.

    z = p·φ + q·λ (radians) → gx = q/(R·cosφ), gy = p/R (pointwise analytic).
    When q=0 the whole grid (including longitude wrap columns and latitude boundary rows) is
    analytically exact; when q≠0 the linear field is discontinuous at the longitude wrap, so
    only interior columns participate in the analytic assertion (wrap columns are pinned by
    the next test).
    """
    lat, lon = _small_grid()
    phi = np.deg2rad(lat)[:, None]
    lam = np.deg2rad(lon)[None, :]
    res = 0.5

    p = 500.0
    z = da.from_array((p * phi + np.zeros_like(lam)).astype(np.float64), chunks=(2, 3))
    expect = np.degrees(np.arctan(np.abs(p) / R_M)) * np.ones((5, 7))
    np.testing.assert_allclose(slope_core(z, lat, res).compute(), expect, rtol=1e-12)

    q = 300.0
    z2 = da.from_array((p * phi + q * lam).astype(np.float64), chunks=(2, 3))
    gx = np.broadcast_to(q / (R_M * np.cos(phi)), (5, 7))
    gy = np.full((5, 7), p / R_M)
    expect2 = np.degrees(np.arctan(np.hypot(gx, gy)))
    out = slope_core(z2, lat, res).compute()
    np.testing.assert_allclose(out[:, 1:-1], expect2[:, 1:-1], rtol=1e-12)
    np.testing.assert_allclose(out[0, 1:-1], expect2[0, 1:-1], rtol=1e-12)
    np.testing.assert_allclose(out[-1, 1:-1], expect2[-1, 1:-1], rtol=1e-12)


def test_slope_periodic_wrap_columns():
    """Longitude wrap: column 0's left neighbor = column nlon−1, column nlon−1's right neighbor = column 0."""
    lat, _ = _small_grid()
    rng = np.random.default_rng(11)
    z = rng.uniform(-100, 4000, size=(5, 7))
    out = slope_core(da.from_array(z, chunks=(2, 3)), lat, 0.5).compute()

    dlon = R_M * np.deg2rad(0.5) * np.cos(np.deg2rad(lat))[:, None]
    gx_wrap0 = (z[:, 1] - z[:, -1]) / (2.0 * dlon[:, 0])
    gx_wrapN = (z[:, 0] - z[:, -2]) / (2.0 * dlon[:, 0])
    dy = R_M * np.deg2rad(0.5)
    gy = np.gradient(z, dy, axis=0)
    for j, gx_col in ((0, gx_wrap0), (6, gx_wrapN)):
        expect = np.degrees(np.arctan(np.hypot(gx_col, gy[:, j])))
        np.testing.assert_allclose(out[:, j], expect, rtol=1e-12)


def test_slope_matches_numpy_reference():
    """dask pipeline vs an independent numpy reference (np.roll periodic + np.gradient one-sided)."""
    lat, _ = _small_grid(nlat=9, nlon=11)
    rng = np.random.default_rng(5)
    z = rng.uniform(-5000, 5000, size=(9, 11))

    def ref(z, lat, res):
        gx = (np.roll(z, -1, 1) - np.roll(z, 1, 1)) / (
            2.0 * R_M * np.deg2rad(res) * np.cos(np.deg2rad(lat))[:, None]
        )
        gy = np.gradient(z, R_M * np.deg2rad(res), axis=0)
        return np.degrees(np.arctan(np.hypot(gx, gy)))

    out = slope_core(da.from_array(z, chunks=(3, 4)), lat, 0.5).compute()
    np.testing.assert_allclose(out, ref(z, lat, 0.5), rtol=1e-12)



def _relief_bruteforce(z):
    m, n = z.shape
    out = np.empty((m, n), dtype=np.float64)
    for i in range(m):
        rows = [r for r in (i - 1, i, i + 1) if 0 <= r < m]
        for j in range(n):
            cols = [(j - 1) % n, j, (j + 1) % n]
            w = z[np.ix_(rows, cols)]
            out[i, j] = w.max() - w.min()
    return out


def test_relief_matches_bruteforce():
    rng = np.random.default_rng(7)
    z = rng.uniform(-4000, 8000, size=(6, 9))
    out = relief_core(da.from_array(z, chunks=(2, 3))).compute()
    np.testing.assert_allclose(out, _relief_bruteforce(z), rtol=1e-12)


def test_relief_handcrafted():
    z = np.zeros((5, 7))
    z[1, 3] = 10.0
    out = relief_core(da.from_array(z, chunks=(5, 7))).compute()
    assert out[1, 3] == 10.0
    assert out[0, 3] == 10.0
    assert out[2, 3] == 10.0 and out[1, 2] == 10.0
    assert out[4, 3] == 0.0
    assert out[1, 6] == 0.0
    z2 = np.zeros((5, 7))
    z2[2, 0] = 10.0
    out2 = relief_core(da.from_array(z2, chunks=(5, 7))).compute()
    assert out2[2, 6] == 10.0
    assert out2[2, 5] == 0.0
    assert out2[2, 1] == 10.0
    assert out2[2, 2] == 0.0



def test_landsea_threshold_boundary():
    z = da.from_array(np.array([[-0.1, 0.0], [0.1, 5000.0]], dtype=np.float32), chunks=1)
    np.testing.assert_array_equal(
        landsea_threshold(z).compute(), np.array([[0, 0], [1, 1]], dtype=np.uint8)
    )



def test_builders_registered():
    from cubebuild.layers import LAYER_BUILDERS

    for lid in ("derived__landsea_mask", "derived__slope", "derived__relief"):
        assert lid in LAYER_BUILDERS



_MINI_MAPPING = """
version: test
status: test
tiers: [1deg, 30min, 6min]
layers:
  - id: topography__bedrock_elevation
    source: test-etopo
    native_res: 3min
    home_tier: 6min
    tiers: [1deg, 30min, 6min]
    dtype: float32
    unit: m
    resampling: conservative_area_weighted
    mask: none
  - id: derived__landsea_mask
    source: 派生自 test-etopo ice-surface（0m 等值线，冰面口径）
    native_res: 6min
    home_tier: 6min
    tiers: [1deg, 30min, 6min]
    dtype: uint8
    unit: flag（0=海，1=陆）
    resampling: mode
    mask: 自身即掩膜
    notes: 掩膜基于冰面高程（冰盖归陆），与高程层裸地复合口径不同——存在 mask=陆而高程<0 的冰下盆地像元（南极/格陵兰）
  - id: derived__slope
    source: 派生自 topography__bedrock_elevation
    native_res: 6min
    home_tier: 6min
    tiers: [1deg, 30min, 6min]
    dtype: float32
    unit: degree
    resampling: none（逐档从该档高程重算）
    mask: 随高程层
  - id: derived__relief
    source: 派生自 topography__bedrock_elevation
    native_res: 6min
    home_tier: 6min
    tiers: [1deg, 30min, 6min]
    dtype: float32
    unit: m
    resampling: none（逐档从该档高程重算）
    mask: 随高程层
"""

_TILE_SHAPE = (300, 300)
_SRC_RES = 15.0 / 300


@pytest.fixture()
def mini_env(tmp_path, monkeypatch):
    """288 synthetic tiles (linear coordinate field) + contract + module patches.

    NATIVE_TIER is elastically set to 6min: the mini source (1/20°) is coarser than 30sec, so
    it cannot form a 30sec
    native tier; the code path matches the real build, and the 30sec semantics are borne by the
    real-build acceptance.
    """
    import cubebuild.derived as derived_mod
    import cubebuild.etopo as etopo_mod
    import cubebuild.fidelity as fidelity_mod
    import rasterio

    monkeypatch.setattr(etopo_mod, "TILE_SHAPE", _TILE_SHAPE)
    monkeypatch.setattr(etopo_mod, "SRC_RES_DEG", _SRC_RES)

    paths = {}
    for bi, band in enumerate(LAT_BANDS):
        band_top = 90 - bi * 15
        for col in LON_COLS:
            label = band + col
            _, left = label_bounds(label)
            rr = np.arange(_TILE_SHAPE[0])[:, None]
            cc = np.arange(_TILE_SHAPE[1])[None, :]
            v = ((band_top - (rr + 0.5) * _SRC_RES) * 1000.0
                 + (left + (cc + 0.5) * _SRC_RES) - 20000.0)
            f = tmp_path / f"ETOPO_2022_v1_15s_{label}_bed.tif"
            with rasterio.open(
                f, "w", driver="GTiff", width=_TILE_SHAPE[1], height=_TILE_SHAPE[0],
                count=1, dtype="float32", nodata=-99999.0,
            ) as dst:
                dst.write(v.astype(np.float32), 1)
            paths[label] = f

    monkeypatch.setattr(etopo_mod, "require_full_coverage", lambda *a, **k: paths)
    monkeypatch.setattr(etopo_mod, "require_surface_coverage", lambda *a, **k: paths)
    monkeypatch.setattr(derived_mod, "NATIVE_TIER", "6min")
    monkeypatch.setattr(fidelity_mod, "NATIVE_TIER", "6min")

    mapping = tmp_path / "mini-mapping.yaml"
    mapping.write_text(_MINI_MAPPING, encoding="utf-8")
    return mapping


def _run_build(tmp_path, mapping, out_name):
    from cubebuild.cli import main

    out = tmp_path / out_name
    rc = main(["build", "--mapping", str(mapping), "--out", str(out),
               "--reports", str(tmp_path / "reports")])
    return rc, out


def test_mini_e2e_build_validate_manifest(tmp_path, mini_env):
    import json

    import xarray as xr

    from cubebuild.manifest import read_manifest

    rc, out = _run_build(tmp_path, mini_env, "cube.zarr")
    assert rc == 0

    dt = xr.open_datatree(out, engine="zarr")
    for tier in ("1deg", "30min", "6min"):
        node = dt[f"/{tier}"]
        assert str(node["derived__landsea_mask"].dtype) == "uint8"
        assert str(node["derived__slope"].dtype) == "float32"
        assert str(node["derived__relief"].dtype) == "float32"
    dt.close()

    manifest = read_manifest(out)
    entries = {e["id"]: e for e in manifest["layers"]}
    for lid in ("derived__slope", "derived__relief"):
        e = entries[lid]
        assert e["derived_from"] == "topography__bedrock_elevation"
        assert e["derivation"]
        assert e["coverage"] == {"1deg": 1.0, "30min": 1.0, "6min": 1.0}
    m = entries["derived__landsea_mask"]
    assert m["coverage"] == {"1deg": 1.0, "30min": 1.0, "6min": 1.0}
    assert "ice-surface" in m["derived_from"]
    assert "known_bias" not in m
    assert "冰面" in m["derivation"] and "冰下盆地" in m["derivation"]
    assert "冰面" in m["notes"] and "冰下盆地" in m["notes"]

    report = json.loads((tmp_path / "reports" / "structure-validation.json").read_text())
    assert report["result"] == "PASS"
    summary = json.loads((tmp_path / "reports" / "build-summary.json").read_text())
    assert set(summary["built"]) == {
        "topography__bedrock_elevation",
        "derived__landsea_mask",
        "derived__slope",
        "derived__relief",
    }


def test_mini_e2e_derived_fidelity(tmp_path, mini_env):
    """Three-layer fidelity CLI: bitwise consistency + value range + land fraction all PASS."""
    import json

    from cubebuild.cli import main

    rc, out = _run_build(tmp_path, mini_env, "cube.zarr")
    assert rc == 0
    for lid in ("derived__landsea_mask", "derived__slope", "derived__relief"):
        rc2 = main(["fidelity", "--store", str(out), "--layer", lid,
                    "--mapping", str(mini_env),
                    "--reports", str(tmp_path / "reports")])
        assert rc2 == 0, lid
        rep = json.loads(
            (tmp_path / "reports" / f"fidelity-{lid}.json").read_text()
        )
        assert rep["result"] == "PASS", rep["checks"]
        if lid == "derived__landsea_mask":
            assert "ice-surface" in rep["source_layer"]
        else:
            assert rep["source_layer"] == "topography__bedrock_elevation"
    rep = json.loads(
        (tmp_path / "reports" / "fidelity-derived__landsea_mask.json").read_text()
    )
    assert any("冰面" in c.get("summary", "") and "冰下盆地" in c.get("summary", "")
              for c in rep["checks"])


def test_mini_e2e_determinism(tmp_path, mini_env):
    from cubebuild.checksum import checksums_equal, store_checksums

    rc1, out1 = _run_build(tmp_path, mini_env, "run1.zarr")
    rc2, out2 = _run_build(tmp_path, mini_env, "run2.zarr")
    assert rc1 == rc2 == 0
    cs1, cs2 = store_checksums(out1), store_checksums(out2)
    assert cs1 and checksums_equal(cs1, cs2)



def test_landsea_ice_surface_basin_diff(tmp_path, monkeypatch):
    """The new/old mask differing pixel set ≡ subglacial basin pixel set (composite elevation <0
    and ice-surface >0).

    Mechanism-level evidence for the targeted verification assertion (real tile selection + real
    assembly + real threshold chain, with a synthetic world constructing decidable basin
    geometry):
    - 288-position bed tiles: linear field v = lat×1000 + lon − 20000 (composite elevation
      baseline);
    - ice-zone bands (N00/S15 two bands, 48 tiles, bed all <0 simulating subglacial basins)
      surface tiles
      = v + 25000 → surface ∈ (−10180, 20180), some >0 (ice surface above sea level);
    - the remaining 240-position surface tiles = v (non-ice zones bed ≡ surface).
    Expectation: diff(new, old) bitwise ≡ {composite<0} ∩ {ice-surface>0},
    and no "land→sea" flip (the directionality of surface ≥ bed).
    """
    import cubebuild.etopo as etopo_mod
    import rasterio

    monkeypatch.setattr(etopo_mod, "TILE_SHAPE", _TILE_SHAPE)
    monkeypatch.setattr(etopo_mod, "SRC_RES_DEG", _SRC_RES)

    src, ice = tmp_path / "src", tmp_path / "ice"
    src.mkdir()
    ice.mkdir()

    def write(directory, label, kind, v):
        f = directory / f"ETOPO_2022_v1_15s_{label}_{kind}.tif"
        with rasterio.open(
            f, "w", driver="GTiff", width=_TILE_SHAPE[1], height=_TILE_SHAPE[0],
            count=1, dtype="float32", nodata=-99999.0,
        ) as dst:
            dst.write(v.astype(np.float32), 1)

    ice_bands = {"N00", "S15"}
    for bi, band in enumerate(LAT_BANDS):
        band_top = 90 - bi * 15
        for col in LON_COLS:
            label = band + col
            _, left = label_bounds(label)
            rr = np.arange(_TILE_SHAPE[0])[:, None]
            cc = np.arange(_TILE_SHAPE[1])[None, :]
            v = ((band_top - (rr + 0.5) * _SRC_RES) * 1000.0
                 + (left + (cc + 0.5) * _SRC_RES) - 20000.0)
            write(tmp_path, label, "bed", v)
            if band in ice_bands:
                write(ice, label, "surface", v + 25000.0)
            else:
                write(src, label, "surface", v)

    composite = etopo_mod.elevation_array("6min", src_dir=tmp_path)
    surface = etopo_mod.ice_surface_array("6min", src_dir=src, ice_surface_dir=ice)
    old_mask = landsea_threshold(composite)
    new_mask = landsea_threshold(surface)

    diff = old_mask != new_mask
    basin = (composite < 0) & (surface > 0)
    land_to_sea = (old_mask == 1) & (new_mask == 0)
    n_diff, n_mismatch, n_l2s = da.compute(
        diff.sum(), (diff != basin).sum(), land_to_sea.sum()
    )
    assert int(n_mismatch) == 0
    assert int(n_l2s) == 0
    assert int(n_diff) > 0
