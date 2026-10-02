"""测试：球面大圆距离核（与独立航空公式互证）+ 层级环带引擎
（对全段暴力路径位级比对）+ 线栅格化器（重叠规则/缺测语义）+ 侧车。

判据只测外部行为：
- 距离核：中点 0 / 端点外延 / 垂直叉积 / 对跖端点分支，随机几何对
  独立航空公式（半正矢+方位角，刻意异构实现）max|Δ| ≤ 1e-6 km；
- 环带引擎：① 随机稀疏段全档位格网 == 全段暴力 min（引理无损的
  直接验证，容差 1e-6 km——浮点结合序差异级别）；② 1° 胞任务对
  全 5 档（含 30″）与暴力路径一致；
- 栅格化器：线穿过像元烧录、重叠较短断裂胜（乱序传入同结果）、
  slip_type 缺失（code 0）不烧录不计数；
- 读取器：行数/词汇表/跨日期线/非 LineString 契约；
- 真实源：基线（13696/23 类词汇/段表契约）+ 引擎胞任务实源抽检；
- e2e（合成源）：双通道入 store → 结构验证 PASS → manifest 编码表/
  validity 链接 → 保真报告 PASS（含距离抽样独立比对）。
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
    """模块级缓存跨测试隔离（合成源/真实源切换不脏读）。"""
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


# ---------------- 编码表 ----------------

def test_slip_type_encoding_table():
    """编码表：23 类、码 1..23 字典序、category_encoding 往返一致。"""
    assert len(SLIP_TYPE_CODES) == 23
    assert sorted(SLIP_TYPE_CODES.values()) == list(range(1, 24))
    enc = category_encoding()
    assert enc["1"] == "Anticline" and enc["23"] == "Syncline"
    assert enc[str(SLIP_TYPE_CODES["Blind Thrust"])] == "Blind Thrust"
    for label, code in SLIP_TYPE_CODES.items():
        assert enc[str(code)] == label


# ---------------- 读取器：契约 ----------------

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
    """slip_type 缺失数漂移（319 → 318）读取期即拒绝——源变更须复评。"""
    n = EXPECTED_ROWS
    slips = ["Normal"] * (n - 318) + [None] * 318
    _patch_rows(monkeypatch, _fake_gdf(n, slips))
    with pytest.raises(ValueError, match="缺失 318"):
        load_gem_faults(_fake_src(tmp_path))


def _slips_with_nan(n_classified):
    """n 条 Normal + 319 条 None（满足 NaN 契约，让后续几何契约先过关）。"""
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


# ---------------- 距离核：已知几何 + 独立公式互证 ----------------

def _equator_segment():
    """赤道段 A=(0°,0°) → B=(0°,10°) 的段表。"""
    from shapely.geometry import LineString

    return segment_table([LineString([(0.0, 0.0), (10.0, 0.0)])])[0]


def test_segment_distance_known_cases():
    segs = _equator_segment()
    km = KM_PER_DEG

    def d(lat, lon):
        p = _unit_vec(np.array([lat]), np.array([lon]))
        return float(_point_seg_dists(p, segs)[0, 0])

    assert d(0.0, 5.0) == pytest.approx(0.0, abs=1e-9)          # 段上中点
    assert d(0.0, 12.0) == pytest.approx(2.0 * km, abs=1e-6)     # B 端外延 → 端点距离
    assert d(0.0, -3.0) == pytest.approx(3.0 * km, abs=1e-6)     # A 端外延
    assert d(10.0, 5.0) == pytest.approx(10.0 * km, abs=1e-6)    # 垂直叉积
    # 对跖点（(0,5) 的对跖 (0,-175)）：最近点 = 段两端点，175° 大圆
    assert d(0.0, -175.0) == pytest.approx(175.0 * km, abs=1e-4)


def test_segment_distance_vs_independent_random():
    """随机段 + 随机点：引擎（3D 向量路径）vs 独立航空公式全段暴力。"""
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


# ---------------- 环带引擎：对全段暴力路径位级比对 ----------------

def _random_segments(n, seed):
    """随机短线段集（顶点数 2–5、步长 ≤ 3°，坐标域约束在 ±170/±78 内
    保证漂移不越界）——覆盖近场/交叉/远场形态。"""
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
    """独立暴力路径：该档全像元中心 → 全段 min（不经任何候选剪枝）。

    分块护栏：中间数组 ≈ 10×(block, n_seg) float64——block 500k × 40 段
    ≈ 1.6GB 峰值（WSL 40GB 预算内）；block 必须随段数缩放（见
    _min_dists_chunked 的实源分块）。
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
    """分块暴力 min（内存护栏：中间数组 ≈ 10×(block, n_seg) float64）。

    实源 141k 段 × block 32 ≈ 360MB 峰值——WSL 40GB 预算内。不分块的
    (14400, 141298) 矩阵 ≈ 160GB 是 OOM 炸弹（WSL 崩溃实证，
    用户硬约束：长任务峰值 ≤40GB）。
    """
    out = np.empty(points.shape[0], dtype=np.float64)
    for s in range(0, points.shape[0], block):
        e = min(s + block, points.shape[0])
        out[s:e] = _point_seg_dists(points[s:e], segs).min(axis=1)
    return out


def test_distance_grids_match_bruteforce_sparse():
    """随机稀疏段（40 条）1°/30′ 全网格 == 全段暴力 min（引理无损验证）。"""
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
    """单 1° 胞全链任务（含 30″）对暴力路径——抽 3 个胞（近场/远场混合）。"""
    geoms = _random_segments(25, seed=11)
    segs, _ = segment_table(geoms)
    # 撒一条断层确保 (i=95, j=180) 附近有近场（段集随机性不保证命中）
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
    """环带半宽随档位细化单调递减（引擎几何自检）。"""
    ws = [_ANNULUS_KM[t] for t in _TIER_CHAIN]
    assert all(a > b for a, b in zip(ws, ws[1:]))


# ---------------- 线栅格化器 ----------------

def test_rasterize_fault_classes_burn_and_overlap():
    """3′ 档：横穿像元的线烧录其迹线；重叠像元取较短断裂（乱序同结果）；
    slip_type 缺失（code 0）不烧录、不参与重叠计数。

    几何避开胞边界：水平线 y=0.06（行 [0.05,0.10] 中心 0.075）、
    x∈[0.01,0.29]（列中心 0.025..0.275）；垂直线 x=0.13（列 [0.10,0.15]
    中心 0.125）、y∈[−0.04,0.14]（行中心 −0.025..0.125）；交叉点
    (0.13, 0.06) 落在胞（中心 0.075, 0.125）内 → 该胞重叠，较短者
    （垂直线 0.18° < 水平线 0.28°）胜出。
    """
    from shapely.geometry import LineString

    long_line = LineString([(0.01, 0.06), (0.29, 0.06)])
    short_cross = LineString([(0.13, -0.04), (0.13, 0.14)])
    classes, overlap = rasterize_fault_classes(
        [long_line, short_cross], np.array([5, 9], dtype=np.uint8), "3min"
    )
    # 乱序传入 + 码随几何走 → 同结果（长度降序规则与传入顺序无关）
    classes2, overlap2 = rasterize_fault_classes(
        [short_cross, long_line], np.array([9, 5], dtype=np.uint8), "3min"
    )
    np.testing.assert_array_equal(classes, classes2)
    assert overlap == overlap2 >= 1
    # 3′ 网格索引：行 r = (lat+90)/0.05−0.5、列 c = (lon+180)/0.05−0.5
    r_cross, c_cross = 1801, 3602            # 胞心 (0.075, 0.125) —— 交叉胞
    assert classes[r_cross, c_cross] == 9    # 重叠胞较短断裂（垂直线）胜
    assert classes[1801, 3600] == 5           # 水平线独占胞（0.075, 0.025）
    assert classes[1799, 3602] == 9          # 垂直线独占胞（−0.025, 0.125）
    # code 0（slip_type 缺失）不烧录、不计数
    classes3, overlap3 = rasterize_fault_classes(
        [long_line], np.array([0], dtype=np.uint8), "3min"
    )
    assert not classes3.any() and overlap3 == 0


def test_rasterize_fault_classes_mode_aggregation():
    """粗档众数聚合：30″ 合成主档 → 3′（因子 6）语义正确。"""
    from shapely.geometry import LineString

    # 一条 30″ 档内斜线，跨越若干 3′ 块
    line = LineString([(0.0, 0.0), (0.1, 0.1)])
    classes, _ = rasterize_fault_classes(
        [line], np.array([4], dtype=np.uint8), "30sec"
    )
    coarse = aggregate_mode(classes, 6, nodata=0)
    assert coarse.shape == grid_shape("3min")
    assert set(np.unique(coarse)) <= {0, 4}
    assert (coarse != 0).sum() > 0
    # 众数一致性：粗档有值 ⟺ 对应 6×6 块内有烧录
    blocks = classes.reshape(
        coarse.shape[0], 6, coarse.shape[1], 6
    ).any(axis=(1, 3))
    np.testing.assert_array_equal(coarse != 0, blocks)


# ---------------- 真实源基线 ----------------

@requires_gem
def test_real_gem_baselines():
    """磁盘实证契约化：13696 行、23 类词汇 + 319 NaN、LineString、
    段表段角 < 90°。"""
    gdf, codes = load_gem_faults()
    assert len(gdf) == EXPECTED_ROWS
    assert (codes == 0).sum() == EXPECTED_SLIP_NAN
    segs, n_drop = segment_table(list(gdf.geometry))
    A, B, NV, sinAB, cosAB = segs[0], segs[1], segs[2], segs[3], segs[4]
    assert cosAB.min() > 0.0
    n_vertices = int(sum(len(g.coords) for g in gdf.geometry))
    assert len(A) + n_drop == n_vertices - EXPECTED_ROWS
    # 顶点 154994（实证）→ 段 141298 + 零长剔除
    assert n_vertices == 154994


@requires_gem
def test_real_distance_cell_spotchecks():
    """实源单胞任务抽检（内存护栏版）：2 胞 × {1°, 6′, 30″} 档、每档
    随机 ≤600 像元对全段暴力路径分块比对（141k 段实源）。

    全档全像元位级一致性已由合成源测试覆盖（段数小可全比对）；本测试
    只验引擎在实源候选规模下的正确性，抽样确定性（seed=13）。"""
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
    """侧车 GeoParquet 往返：行数/属性列/编码列/几何/CRS/可见性。"""
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


# ---------------- e2e（合成源）：双通道入 store ----------------

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
    """合成 GEM 源（少量线段，e2e 可在分钟级完成）注入 load_gem_faults。"""
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
    slips[0] = None          # 模拟 slip_type 缺失
    gdf = _fake_gdf(n, slips, geoms=geoms)
    monkeypatch.setattr(
        "cubebuild.faults.load_gem_faults", lambda _src=None: (gdf, np.array(
            [SLIP_TYPE_CODES.get(s, 0) for s in slips], dtype=np.uint8))
    )
    # 侧车/保真走同源注入
    monkeypatch.setattr(
        "cubebuild.sidecars.load_gem_faults", lambda _src=None: (gdf, np.array(
            [SLIP_TYPE_CODES.get(s, 0) for s in slips], dtype=np.uint8))
    )
    return gdf


def test_faults_e2e(tmp_path, monkeypatch):
    """合成源双通道全链路：build → 结构验证 PASS → manifest（编码表/
    validity 链接/internal 可见性）→ 保真报告 PASS（距离抽样独立比对）。"""
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
    assert dist["coverage"] == {"1deg": 1.0, "30sec": 1.0}   # 全域有值

    import xarray as xr
    with xr.open_datatree(out, engine="zarr", chunks={}) as dt:
        v30 = dt["/30sec"][DISTANCE_LAYER_ID].values
        assert np.isfinite(v30).all() and (v30 >= 0).all()
        c30 = dt["/30sec"][SLIP_CLASS_LAYER_ID].values
        m30 = dt["/30sec"][f"{SLIP_CLASS_LAYER_ID}__validity"].values
        assert np.array_equal(m30, (c30 != 0).astype(np.uint8))

    # 保真：类别（逐位/众数/值域）+ 距离（抽样独立比对/全域有限）
    for lid in (SLIP_CLASS_LAYER_ID, DISTANCE_LAYER_ID):
        report = build_faults_fidelity_report(out, lid, ["1deg", "30sec"])
        assert report["result"] == "PASS", report["checks"]
        fails = [c for c in report["checks"] if c["status"] == "FAIL"]
        assert not fails
