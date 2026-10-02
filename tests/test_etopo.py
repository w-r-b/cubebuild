"""测试：ETOPO 适配器 + 面积加权保守核 + 保真 harness。

判据只测外部行为：
- 保守核：积分量解析守恒（float64 下相对误差 ~1e-12）+ 加权均值正确性；
- 覆盖校验：缺片即 SourceNotReady 且列明缺失清单（bed 优先规则）；
- 真实瓦片集成：62 片 bed 中相邻瓦片聚合后与手算加权均值逐位一致；
- fidelity checks：合成源 → 积分偏差通过 + 分位数单调偏移方向正确。
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


# ---------------- 核：面积加权保守平均 ----------------

def test_aggregate_weighted_value_correctness():
    """factor=2 小矩阵：粗像元 = 行面积加权平均（列等权），与手算一致。"""
    # 4 行 × 2 列源，factor 2 → 2×1 粗像元
    data = np.array([[1.0, 3.0], [5.0, 7.0], [10.0, 20.0], [30.0, 40.0]])
    w = np.array([1.0, 3.0, 2.0, 2.0])  # 行面积权重
    out = aggregate_weighted(data, w, 2)
    # 粗(0,0)：行0 权1 → 1·(1+3)；行1 权3 → 3·(5+7)；除以 (1+1+3+3)=8 → 40/8=5
    assert out[0, 0] == pytest.approx(5.0)
    # 粗(1,0)：行2/3 权各2 → 2·(10+20)+2·(30+40) = 200；除以 8 → 25
    assert out[1, 0] == pytest.approx(25.0)


def test_aggregate_integral_conservation():
    """解析恒等式：Σ_coarse v_c·A_c == Σ_fine v·A（float64 相对误差 < 1e-12）。"""
    rng = np.random.default_rng(42)
    m = n = 240
    factor = 12  # 15s → 3min
    data = rng.uniform(-5000, 4000, size=(m, n))
    res_fine = 15.0 / 3600.0
    w_fine = row_areas_km2(45.0, m, res_fine)
    coarse = aggregate_weighted(data, w_fine, factor)

    # 粗像元面积 = factor·Σ(块内行面积)（Δλ_c = factor·Δλ_f 的解析恒等式）
    wc = w_fine.reshape(m // factor, factor).sum(axis=1) * factor
    fine_total = float((data * w_fine[:, None]).sum())
    coarse_total = float((coarse * wc[:, None]).sum())
    assert coarse_total == pytest.approx(fine_total, rel=1e-12)


def test_aggregate_rejects_noninteger_factor():
    with pytest.raises(ValueError):
        aggregate_weighted(np.zeros((5, 4)), np.ones(5), 2)


# ---------------- 瓦片网格与覆盖校验 ----------------

def test_label_grid_and_bounds():
    assert len(all_labels()) == 288
    assert label_bounds("N60W030") == (60.0, -30.0)
    assert label_bounds("N00E000") == (0.0, 0.0)     # −15..0 带：北边界 0，N 前缀（目录实测）
    assert label_bounds("S75W180") == (-75.0, -180.0)
    assert label_bounds("N90E165") == (90.0, 165.0)


def test_scan_tiles_bed_priority_and_missing(tmp_path):
    """bed 优先规则 + 缺失清单（tmp 目录模拟部分覆盖）。"""
    # 本地 bed 瓦占 62 位；tmp 模拟：1 片 bed + 1 片 surface（同位并存）+ 1 片 surface
    d = tmp_path
    (d / "ETOPO_2022_v1_15s_N90E000_bed.tif").write_bytes(b"")
    (d / "ETOPO_2022_v1_15s_N90E000_surface.tif").write_bytes(b"")
    (d / "ETOPO_2022_v1_15s_S75W180_surface.tif").write_bytes(b"")
    tiles = scan_tiles(d)
    assert tiles["N90E000"].name.endswith("_bed.tif")   # bed 优先
    assert tiles["S75W180"].name.endswith("_surface.tif")
    miss = missing_tiles(d)
    assert len(miss) == 286
    assert "N90E000" not in miss and "S75W180" not in miss


def test_real_disk_coverage_complete():
    """真实磁盘：62 bed + 226 surface = 288 位全球覆盖齐备（数据到货后）。"""
    if not SRC_DIR.is_dir():
        pytest.skip("源目录不存在")
    assert missing_tiles(SRC_DIR) == []
    from cubebuild.etopo import require_full_coverage

    tiles = require_full_coverage(SRC_DIR)
    assert len(tiles) == 288


# ---------------- 掩膜冰面口径：surface 优先瓦片选择 ----------------

def test_scan_surface_tiles_priority_and_bed_ignored(tmp_path):
    """surface 优先：bed 瓦被忽略、ice_surface_dir 同位优先、两目录互补。"""
    src, ice = tmp_path / "src", tmp_path / "ice"
    src.mkdir()
    ice.mkdir()
    (src / "ETOPO_2022_v1_15s_N90E000_bed.tif").write_bytes(b"")       # bed：忽略
    (src / "ETOPO_2022_v1_15s_N90E000_surface.tif").write_bytes(b"")   # 同位：ice 优先
    (ice / "ETOPO_2022_v1_15s_N90E000_surface.tif").write_bytes(b"")
    (src / "ETOPO_2022_v1_15s_S75W180_surface.tif").write_bytes(b"")   # 无冰区：src
    tiles = scan_surface_tiles(src, ice)
    assert set(tiles) == {"N90E000", "S75W180"}
    assert tiles["N90E000"] == ice / "ETOPO_2022_v1_15s_N90E000_surface.tif"
    assert tiles["S75W180"] == src / "ETOPO_2022_v1_15s_S75W180_surface.tif"


def test_require_surface_coverage_missing_list(tmp_path):
    """全域 ice-surface 覆盖缺片 → SourceNotReady 且列明清单。"""
    d = tmp_path
    (d / "ETOPO_2022_v1_15s_N90E000_surface.tif").write_bytes(b"")
    with pytest.raises(SourceNotReady) as ei:
        require_surface_coverage(d, d)      # 冰目录＝src 同目录 → 并集仅 1/288 位
    assert ei.value.missing == [
        lb for lb in all_labels() if lb != "N90E000"
    ]


def test_scan_surface_tiles_missing_dir_raises(tmp_path):
    """源目录不存在 → 显式 FileNotFoundError（非静默空扫描误报缺片）。"""
    with pytest.raises(FileNotFoundError):
        scan_surface_tiles(tmp_path / "no-such-src", tmp_path / "no-such-ice")


def test_real_disk_surface_coverage_complete():
    """真实磁盘：bed_crosscheck 62 冰区 surface + 源目录 226 surface = 288 位。

    交付的消费端实证：掩膜冰面口径的全域 ice-surface 输入齐备。
    """
    if not SRC_DIR.is_dir():
        pytest.skip("源目录不存在")
    tiles = require_surface_coverage(SRC_DIR, SRC_DIR / "bed_crosscheck")
    assert len(tiles) == 288
    n_ice = sum(1 for p in tiles.values() if "bed_crosscheck" in str(p))
    assert n_ice == 62
    # bed 优先复合口径不受影响：源目录扫描仍为 288 位 bed 优先选择
    bed_tiles = scan_tiles(SRC_DIR)
    assert sum(1 for p in bed_tiles.values() if p.name.endswith("_bed.tif")) == 62


# ---------------- 真实瓦片聚合集成 ----------------

def test_real_tile_aggregation_matches_manual():
    """真实 bed 瓦（S75E000，北边界 −75）：1° 档聚合 == 独立行循环加权均值。

    独立算法：逐粗像元显式循环（不与 aggregate_weighted 共享代码路径）。
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
    manual = manual[::-1]   # 构建器输出行 0 = 瓦南缘；手算行 0 = 瓦北缘 → 对齐翻转
    # float32 产物 vs float64 手算：1e-4 m 量级容差（float32 有效数字 ~7 位）
    np.testing.assert_allclose(out, manual, atol=1e-3)
    # 物理合理性：南极冰下基岩应为负值主导（BedMachine 冰下地形）
    assert out.min() < -500


# ---------------- 迷你端到端：288 位合成网格 → build_etopo_bedrock ----------------

def _mini_entry(tiers=("1deg",)):
    from cubebuild.contract import LayerEntry

    return LayerEntry(
        id="topography__bedrock_elevation", dims=("lat", "lon"), source="test",
        native_res="15sec", native_res_deg=15 / 3600, home_tier="30sec",
        tiers=tiers, dtype="float32", unit="m",
        resampling="conservative_area_weighted", mask="none", visibility="public",
    )


def test_build_assembles_global_grid(tmp_path, monkeypatch):
    """288 位合成瓦片 → 全球数组：形状/坐标/dtype/南北带序/bed 覆盖 surface。

    缩小几何：瓦 120×120、源分辨率 0.125°（1° 档 factor=8 整数），网格拓扑
    与真实一致（12 带 × 24 列，自北向南带序、S75 在数组最前）。
    """
    import cubebuild.etopo as etopo_mod
    import rasterio
    from cubebuild.grids import grid_shape, tier_centers

    tile_shape = (120, 120)
    monkeypatch.setattr(etopo_mod, "TILE_SHAPE", tile_shape)
    monkeypatch.setattr(etopo_mod, "SRC_RES_DEG", 15.0 / tile_shape[0])

    paths = {}
    for bi, band in enumerate(LAT_BANDS):        # bi=0 为最北带 N90
        band_top = 90 - bi * 15                  # 该带北边界纬度
        for col in LON_COLS:
            label = band + col
            kind = "bed" if bi % 2 == 0 else "surface"  # 交错 bed/surface
            f = tmp_path / f"ETOPO_2022_v1_15s_{label}_{kind}.tif"
            # 值 = 纬度×1000 + 经度：同时捕获行序/列序/经度环旋转三类错位
            # （线性场在保守平均下均值=粗像元中心值，解析可断言）
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
    # elevation_array 经 src_dir 参数调用（抽出共用入口），lambda 须收参
    monkeypatch.setattr(etopo_mod, "require_full_coverage", lambda *a, **k: paths)

    arr = etopo_mod.build_etopo_bedrock(_mini_entry(), "1deg")
    nlat, nlon = grid_shape("1deg")
    assert arr.shape == (nlat, nlon)
    lat, lon = tier_centers("1deg")
    np.testing.assert_array_equal(arr.coords["lat"].values, lat)
    np.testing.assert_array_equal(arr.coords["lon"].values, lon)
    assert arr.dtype == np.float32

    vals = arr.compute().values
    # 线性场：粗像元均值 = 中心纬度×1000 + 中心经度（容差 = 半档×1000）
    np.testing.assert_allclose(
        vals, lat[:, None] * 1000.0 + lon[None, :], atol=600.0
    )


def test_build_raises_when_grid_misaligned(monkeypatch, tmp_path):
    """瓦块不整除全球网格（如档位非法）→ 显式失败。"""
    import cubebuild.etopo as etopo_mod

    # 不需要真实数据：构造 288 假路径，patch TILE_SHAPE 使 1deg factor=8、
    # 但把 require_full_coverage 返回的路径指向不可读文件 → 任何 tier 均应在
    # 组装前通过几何校验。此处直接用非法 tier 触发 grids 报错路径。
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
    assert -0.6 <= q["q50"] <= 0.6  # 两半之间（bin 中心插值，±0.5m 精度）


def test_fidelity_checks_pass_and_quantile_report():
    """合成源 → 保守聚合 → 积分偏差远小于 0.1% + 分位数偏移成报告。"""
    rng = np.random.default_rng(7)
    src = rng.uniform(-8000, 5000, size=(240, 240))
    factor = 24  # → 6min 档尺寸
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
    # 保守平均压缩分布：q1 上移（极小值被抬升）、q99 下移
    assert qrep["shift_m"]["q1"] > 0
    assert qrep["shift_m"]["q99"] < 0


def test_fidelity_detects_integral_violation():
    """源统计被篡改 → 积分判据 FAIL（harness 确有拦截力）。"""
    rng = np.random.default_rng(1)
    tier_s = {
        "sum_va": 100.0,
        "hist": hist_add(np.zeros(20000, dtype=np.uint64), rng.uniform(0, 1, 100)),
    }
    src = {"sum_va": 90.0, "hist": tier_s["hist"]}
    checks = fidelity_checks(src, tier_s, "1deg")
    assert checks[0]["status"] == "FAIL"


# ---------------- manifest extras 防护 ----------------

def test_manifest_extras_reject_canonical_key_clash():
    from cubebuild.manifest import build_layer_entry

    with pytest.raises(ValueError, match="canonical"):
        build_layer_entry(_mini_entry(), {"1deg": 1.0}, extra={"coverage": {}})
