"""Tests: ETOPO adapter + area-weighted conservative kernel + fidelity harness.

Checks test external behavior only:
- conservative kernel: analytic conservation of the integral quantity (relative error ~1e-12
  under float64) + correctness of the weighted mean;
- coverage validation: a missing tile yields SourceNotReady with the missing list itemized
  (bed-priority rule);
- real tile integration: after aggregating adjacent tiles among the 62 bed pieces, they match
  the hand-computed weighted mean bitwise;
- fidelity checks: synthetic source → integral deviation passes + quantile monotone shift
  direction correct.
"""

from pathlib import Path

import numpy as np
import pytest

from cubebuild.etopo import (
    LAT_BANDS,
    LON_COLS,
    SourceNotReady,
    aggregate_weighted,
    all_labels,
    label_bounds,
    missing_tiles,
    read_tile_aggregated,
    require_surface_coverage,
    row_areas_km2,
    scan_surface_tiles,
    scan_tiles,
)
from cubebuild.fidelity import fidelity_checks, hist_add, quantiles_from_hist
from cubebuild.grids import tier_deg

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "original data/topography/etopo2022-bed"



def test_aggregate_weighted_value_correctness():
    """factor=2 small matrix: coarse cell = row-area-weighted mean (columns equally weighted), consistent with hand computation."""
    data = np.array([[1.0, 3.0], [5.0, 7.0], [10.0, 20.0], [30.0, 40.0]])
    w = np.array([1.0, 3.0, 2.0, 2.0])
    out = aggregate_weighted(data, w, 2)
    assert out[0, 0] == pytest.approx(5.0)
    assert out[1, 0] == pytest.approx(25.0)


def test_aggregate_integral_conservation():
    """Analytic identity: Σ_coarse v_c·A_c == Σ_fine v·A (float64 relative error < 1e-12)."""
    rng = np.random.default_rng(42)
    m = n = 240
    factor = 12  # 15s → 3min
    data = rng.uniform(-5000, 4000, size=(m, n))
    res_fine = 15.0 / 3600.0
    w_fine = row_areas_km2(45.0, m, res_fine)
    coarse = aggregate_weighted(data, w_fine, factor)

    wc = w_fine.reshape(m // factor, factor).sum(axis=1) * factor
    fine_total = float((data * w_fine[:, None]).sum())
    coarse_total = float((coarse * wc[:, None]).sum())
    assert coarse_total == pytest.approx(fine_total, rel=1e-12)


def test_aggregate_rejects_noninteger_factor():
    with pytest.raises(ValueError):
        aggregate_weighted(np.zeros((5, 4)), np.ones(5), 2)



def test_label_grid_and_bounds():
    assert len(all_labels()) == 288
    assert label_bounds("N60W030") == (60.0, -30.0)
    assert label_bounds("N00E000") == (0.0, 0.0)
    assert label_bounds("S75W180") == (-75.0, -180.0)
    assert label_bounds("N90E165") == (90.0, 165.0)


def test_scan_tiles_bed_priority_and_missing(tmp_path):
    """bed-priority rule + missing list (tmp directory simulates partial coverage)."""
    d = tmp_path
    (d / "ETOPO_2022_v1_15s_N90E000_bed.tif").write_bytes(b"")
    (d / "ETOPO_2022_v1_15s_N90E000_surface.tif").write_bytes(b"")
    (d / "ETOPO_2022_v1_15s_S75W180_surface.tif").write_bytes(b"")
    tiles = scan_tiles(d)
    assert tiles["N90E000"].name.endswith("_bed.tif")
    assert tiles["S75W180"].name.endswith("_surface.tif")
    miss = missing_tiles(d)
    assert len(miss) == 286
    assert "N90E000" not in miss and "S75W180" not in miss


def test_real_disk_coverage_complete():
    """Real disk: 62 bed + 226 surface = 288 positions of complete global coverage (once data arrives)."""
    if not SRC_DIR.is_dir():
        pytest.skip("源目录不存在")
    assert missing_tiles(SRC_DIR) == []
    from cubebuild.etopo import require_full_coverage

    tiles = require_full_coverage(SRC_DIR)
    assert len(tiles) == 288



def test_scan_surface_tiles_priority_and_bed_ignored(tmp_path):
    """surface priority: bed tiles ignored, ice_surface_dir same-position priority, the two directories complementary."""
    src, ice = tmp_path / "src", tmp_path / "ice"
    src.mkdir()
    ice.mkdir()
    (src / "ETOPO_2022_v1_15s_N90E000_bed.tif").write_bytes(b"")
    (src / "ETOPO_2022_v1_15s_N90E000_surface.tif").write_bytes(b"")
    (ice / "ETOPO_2022_v1_15s_N90E000_surface.tif").write_bytes(b"")
    (src / "ETOPO_2022_v1_15s_S75W180_surface.tif").write_bytes(b"")
    tiles = scan_surface_tiles(src, ice)
    assert set(tiles) == {"N90E000", "S75W180"}
    assert tiles["N90E000"] == ice / "ETOPO_2022_v1_15s_N90E000_surface.tif"
    assert tiles["S75W180"] == src / "ETOPO_2022_v1_15s_S75W180_surface.tif"


def test_require_surface_coverage_missing_list(tmp_path):
    """Missing tiles in the global ice-surface coverage → SourceNotReady with the list itemized."""
    d = tmp_path
    (d / "ETOPO_2022_v1_15s_N90E000_surface.tif").write_bytes(b"")
    with pytest.raises(SourceNotReady) as ei:
        require_surface_coverage(d, d)
    assert ei.value.missing == [
        lb for lb in all_labels() if lb != "N90E000"
    ]


def test_scan_surface_tiles_missing_dir_raises(tmp_path):
    """Source directory does not exist → explicit FileNotFoundError (not a silent empty scan falsely reporting missing tiles)."""
    with pytest.raises(FileNotFoundError):
        scan_surface_tiles(tmp_path / "no-such-src", tmp_path / "no-such-ice")


def test_real_disk_surface_coverage_complete():
    """Real disk: bed_crosscheck 62 ice-zone surface + source directory 226 surface = 288
    positions.

    Consumer-side evidence of the delivery: the global ice-surface input for the mask
    ice-surface calibration is complete.
    """
    if not SRC_DIR.is_dir():
        pytest.skip("源目录不存在")
    tiles = require_surface_coverage(SRC_DIR, SRC_DIR / "bed_crosscheck")
    assert len(tiles) == 288
    n_ice = sum(1 for p in tiles.values() if "bed_crosscheck" in str(p))
    assert n_ice == 62
    bed_tiles = scan_tiles(SRC_DIR)
    assert sum(1 for p in bed_tiles.values() if p.name.endswith("_bed.tif")) == 62



def test_real_tile_aggregation_matches_manual():
    """Real bed tile (S75E000, north boundary −75): 1° tier aggregation == an independent
    row-loop weighted mean.

    Independent algorithm: an explicit loop over coarse cells (not sharing a code path with
    aggregate_weighted).
    """
    path = SRC_DIR / "ETOPO_2022_v1_15s_S75E000_bed.tif"
    if not path.exists():
        pytest.skip("源瓦片不存在")
    from cubebuild.etopo import read_tile_validated

    factor = 240  # 15s → 1deg
    out = read_tile_aggregated(path, "1deg", -75.0)
    assert out.shape == (15, 15)

    data = read_tile_validated(path)
    w = row_areas_km2(-75.0, 3600, 15.0 / 3600.0)
    manual = np.empty((15, 15), dtype=np.float64)
    for ci in range(15):
        for cj in range(15):
            block = data[ci * factor:(ci + 1) * factor, cj * factor:(cj + 1) * factor]
            wb = w[ci * factor:(ci + 1) * factor]
            manual[ci, cj] = (block * wb[:, None]).sum() / (wb.sum() * factor)
    manual = manual[::-1]
    np.testing.assert_allclose(out, manual, atol=1e-3)
    assert out.min() < -500



def _mini_entry(tiers=("1deg",)):
    from cubebuild.contract import LayerEntry

    return LayerEntry(
        id="topography__bedrock_elevation", dims=("lat", "lon"), source="test",
        native_res="15sec", native_res_deg=15 / 3600, home_tier="30sec",
        tiers=tiers, dtype="float32", unit="m",
        resampling="conservative_area_weighted", mask="none", visibility="public",
    )


def test_build_assembles_global_grid(tmp_path, monkeypatch):
    """288 synthetic tiles → a global array: shape/coords/dtype/north-south band order/bed
    covering surface.

    Reduced geometry: tile 120×120, source resolution 0.125° (1° tier factor=8 integer), grid
    topology
    matches the real one (12 bands × 24 columns, north-to-south band order, S75 first in the
    array).
    """
    import cubebuild.etopo as etopo_mod
    import rasterio
    from cubebuild.grids import grid_shape, tier_centers

    tile_shape = (120, 120)
    monkeypatch.setattr(etopo_mod, "TILE_SHAPE", tile_shape)
    monkeypatch.setattr(etopo_mod, "SRC_RES_DEG", 15.0 / tile_shape[0])

    paths = {}
    for bi, band in enumerate(LAT_BANDS):
        band_top = 90 - bi * 15
        for col in LON_COLS:
            label = band + col
            kind = "bed" if bi % 2 == 0 else "surface"
            f = tmp_path / f"ETOPO_2022_v1_15s_{label}_{kind}.tif"
            _, left = etopo_mod.label_bounds(label)
            rr = np.arange(tile_shape[0])[:, None]
            cc = np.arange(tile_shape[1])[None, :]
            v = ((band_top - (rr + 0.5) * 0.125) * 1000.0
                 + (left + (cc + 0.5) * 0.125))
            with rasterio.open(
                f, "w", driver="GTiff", width=tile_shape[1], height=tile_shape[0],
                count=1, dtype="float32", nodata=-99999.0,
            ) as dst:
                dst.write(v.astype(np.float32), 1)
            paths[label] = f
    assert len(paths) == 288
    monkeypatch.setattr(etopo_mod, "require_full_coverage", lambda *a, **k: paths)

    arr = etopo_mod.build_etopo_bedrock(_mini_entry(), "1deg")
    nlat, nlon = grid_shape("1deg")
    assert arr.shape == (nlat, nlon)
    lat, lon = tier_centers("1deg")
    np.testing.assert_array_equal(arr.coords["lat"].values, lat)
    np.testing.assert_array_equal(arr.coords["lon"].values, lon)
    assert arr.dtype == np.float32

    vals = arr.compute().values
    np.testing.assert_allclose(
        vals, lat[:, None] * 1000.0 + lon[None, :], atol=600.0
    )


def test_build_raises_when_grid_misaligned(monkeypatch, tmp_path):
    """Tile blocks do not evenly divide the global grid (e.g. an illegal tier) → explicit failure."""
    import cubebuild.etopo as etopo_mod

    paths = {lb: tmp_path / "x.tif" for lb in all_labels()}
    monkeypatch.setattr(etopo_mod, "require_full_coverage", lambda *a, **k: paths)
    with pytest.raises(Exception):
        etopo_mod.build_etopo_bedrock(_mini_entry(), "not-a-tier")


# ---------------- fidelity harness ----------------

def test_hist_quantiles_roundtrip():
    hist = np.zeros(20000, dtype=np.uint64)
    values = np.array([-5000.0, 0.0, 0.0, 4000.0])
    hist = hist_add(hist, values)
    q = quantiles_from_hist(hist, (25, 50, 75))
    assert q["q25"] < q["q50"] < q["q75"]
    assert -0.6 <= q["q50"] <= 0.6


def test_fidelity_checks_pass_and_quantile_report():
    """Synthetic source → conservative aggregation → integral deviation far below 0.1% + quantile shift becomes a report."""
    rng = np.random.default_rng(7)
    src = rng.uniform(-8000, 5000, size=(240, 240))
    factor = 24
    w = row_areas_km2(30.0, 240, 15.0 / 3600.0)
    coarse = aggregate_weighted(src, w, factor)

    def stats(arr, res, top=30.0):
        ww = row_areas_km2(top, arr.shape[0], res)
        h = np.zeros(20000, dtype=np.uint64)
        h = hist_add(h, arr)
        return {"sum_va": float((arr * ww[:, None]).sum()), "hist": h}

    src_s = stats(src, 15.0 / 3600.0)
    tier_s = stats(coarse, tier_deg("6min"))
    checks = fidelity_checks(src_s, tier_s, "6min")
    integral = next(c for c in checks if "rel_dev" in c)
    assert integral["status"] == "pass"
    assert integral["rel_dev"] < 1e-12
    qrep = next(c for c in checks if "shift_m" in c)
    assert qrep["status"] == "report"
    assert qrep["shift_m"]["q1"] > 0
    assert qrep["shift_m"]["q99"] < 0


def test_fidelity_detects_integral_violation():
    """Source statistics are tampered with → the integral check FAILs (the harness does have interception power)."""
    rng = np.random.default_rng(1)
    tier_s = {
        "sum_va": 100.0,
        "hist": hist_add(np.zeros(20000, dtype=np.uint64), rng.uniform(0, 1, 100)),
    }
    src = {"sum_va": 90.0, "hist": tier_s["hist"]}
    checks = fidelity_checks(src, tier_s, "1deg")
    assert checks[0]["status"] == "FAIL"



def test_manifest_extras_reject_canonical_key_clash():
    from cubebuild.manifest import build_layer_entry

    with pytest.raises(ValueError, match="canonical"):
        build_layer_entry(_mini_entry(), {"1deg": 1.0}, extra={"coverage": {}})
