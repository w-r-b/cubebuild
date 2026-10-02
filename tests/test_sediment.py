"""测试：沉积组 GST1 + Global Basins 双通道。

判据只测外部行为（同款判据）：
- GST1 读取器：列序陷阱（表头 Lon Lat meter / 数据 Lat Lon meter，按
  位置取列）+ 首行 (-90,-180) 契约 + 网格几何（行内纬度常值/经度内循环
  步进）+ ±180 周期重复列 + 极点行常值 + 拒绝路径；
- GST1 保守核：常数场逐位还原 + 线性场独立 sin 权重解析解 + Voronoi
  胞边（极点行半权钳制）；
- 真实源契约：GST1 完整网格/值域/极点值/地标节点值；盆地 768 片 =
  764 盆地 / 4 拆分片 / 1 无效几何修复 / 类型词汇 8 类；
- 盆地类别：3′ 主档栅格化地标（真实数据预落位）+ 覆盖率 ~25.39% +
  重叠像元（3 对数字化共享边界）；
- 距离：区域距离语义（域内 0/域外 >0）+ 独立航空公式互证；
- e2e（合成源）：三层入 store → 结构验证 PASS → manifest（编码表/
  validity 链接/internal visibility）→ 保真 PASS → 侧车 parquet。
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

_L3, _O3 = tier_centers(BASINS_HOME_TIER)     # 3′ 胞心坐标（grids 单一来源）


def _cell_at(tier: str, lat: float, lon: float) -> tuple[int, int]:
    """地标点 → 最近档位胞心 (r, c)（浮点坑免疫：argmin 最近中心）。"""
    lats, lons = tier_centers(tier)
    return int(np.abs(lats - lat).argmin()), int(np.abs(lons - lon).argmin())


@pytest.fixture(autouse=True)
def _clear_sediment_caches(monkeypatch, tmp_path):
    """模块级缓存跨测试隔离 + 距离金字塔缓存重定向（合成数据不落产品目录）。"""
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


# ---------------- 编码表 ----------------

def test_basin_type_encoding_table():
    """盆地类型编码表：8 码、1..8 字典序（编码表可查闭环）。"""
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


# ---------------- GST1 读取器（合成源：列序陷阱 + 几何契约） ----------------

_SYN_SPACING = Fraction(1, 1)          # 合成缩格：1° 节点（181×361）


def _write_synthetic_gst1(path: Path, values_fn) -> None:
    """写合成 XYZ：表头保持真实陷阱形态（Lon Lat meter），数据列序 Lat Lon meter。"""
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
    """缩格环境：GST1_EXPECTED → 1° 节点网格；返回源目录。"""
    monkeypatch.setattr(
        "cubebuild.sediment.GST1_EXPECTED",
        {"shape": (181, 361), "spacing": _SYN_SPACING},
    )
    src_dir = tmp_path / "gst1"
    src_dir.mkdir()
    return src_dir


def test_gst1_loader_column_order_trap(_syn_gst1_env):
    """列序陷阱：表头写 Lon Lat meter、数据实为 Lat Lon meter——按位置取列。

    场 v = 1000 + lat·10 + 50·cos(lon°)·cos(lat°)（经向周期 ±180 相等；
    极点行 cos(lat)=0 退化常值，与真实源极点单点形态一致）：方位可判
    （纬向线性 + 经向余弦）；若误按表头列名取列（lat/lon 对调），网格
    几何断言（行内纬度常值）立即拒绝——读取器契约即陷阱捕获器。
    """
    p = _syn_gst1_env / XYZ_NAME
    _write_synthetic_gst1(
        p,
        lambda la, lo: 1000.0 + la * 10.0
        + 50.0 * np.cos(np.deg2rad(lo)) * np.cos(np.deg2rad(la)),
    )
    src = load_gst1(_syn_gst1_env)
    assert src.values.shape == (181, 360)            # 弃 +180 重复列
    assert src.lon_res == _SYN_SPACING
    # 节点 (0,0)=(-90,-180) → v = 1000-900（极点 cos(lat)=0）
    assert src.values[0, 0] == np.float32(100.0)
    # 节点 (90, 180)=(赤道, 格林尼治) → v = 1000 + 50
    assert src.values[90, 180] == np.float32(1050.0)
    assert src.valid.all()
    # Voronoi 胞边：极点行钳制半权（首边 = -90，第二边 = -89.5）
    assert src.lat_edges[0] == -90.0 and src.lat_edges[1] == -89.5
    assert src.lat_edges[-1] == 90.0


def test_gst1_loader_rejects_header_change(_syn_gst1_env):
    p = _syn_gst1_env / XYZ_NAME
    p.write_text("Lat Lon meter\n-90.0 -180.0 1.0\n", encoding="ascii")
    with pytest.raises(ValueError, match="表头"):
        load_gst1(_syn_gst1_env)


def test_gst1_loader_rejects_swapped_columns(_syn_gst1_env):
    """数据列对调（col1=lon, col2=lat）：首行 (lat,lon) ≠ (-90,-180) 即拒绝。"""
    p = _syn_gst1_env / XYZ_NAME
    lines = ["     Lon      Lat    meter"]
    for i in range(181):
        la = -90.0 + i
        for j in range(361):
            lo = -180.0 + j
            lines.append(f"{lo:9.3f} {la:8.3f} {100.0:9.2f}")   # col1=lon（对调形态）
    p.write_text("\n".join(lines) + "\n", encoding="ascii")
    with pytest.raises(ValueError, match="首行"):
        load_gst1(_syn_gst1_env)


def test_gst1_loader_rejects_nonperiodic_seam(_syn_gst1_env):
    """±180 列不同值：周期重复假设破坏即拒绝（防错位拼接）。"""
    p = _syn_gst1_env / XYZ_NAME
    _write_synthetic_gst1(p, lambda la, lo: 1.0 if lo == 180.0 else la)
    with pytest.raises(ValueError, match="周期重复"):
        load_gst1(_syn_gst1_env)


def test_gst1_loader_rejects_nonconstant_pole_row(_syn_gst1_env):
    """极点行行内非常值：极点单点假设破坏即拒绝（经向周期取 sin 防先触发缝检）。"""
    p = _syn_gst1_env / XYZ_NAME
    _write_synthetic_gst1(
        p, lambda la, lo: np.sin(np.deg2rad(lo)) if la == -90.0 else 1.0
    )
    with pytest.raises(ValueError, match="极点行"):
        load_gst1(_syn_gst1_env)


# ---------------- GST1 保守核（解析解互证） ----------------

def test_gst1_conservative_constant_and_linear(_syn_gst1_env):
    """常数场逐位还原 + 线性场独立 sin 权重解析解（1° 目标胞 = 两半胞加权）。"""
    from cubebuild.grids import tier_fraction

    # 常数场：目标值逐位 == 常数（float32 精确）
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

    # 线性场 v = lat：1° 目标行 i 覆盖源节点 −90+i 与 −90+i+1 的内半胞
    # （Voronoi 边 −90+i+0.5 对齐目标边）——期望值 = sin 差面积权重的
    # 独立解析解（不经核路径）
    _write_synthetic_gst1(p, lambda la, lo: la)
    src = load_gst1(_syn_gst1_env)
    v, _W = conservative_overlap_mean(
        src.values, src.valid, src.lat_edges, src.lon_res,
        tier_fraction("1deg"), lon_phase=src.lon_phase,
    )
    R2 = (6371.0071810) ** 2 * np.pi / 180.0          # 行面积因子（独立直算）
    for i in (0, 45, 90, 134, 179):                   # 极点行（半权）与常规行
        y0, y1 = -90.0 + i, -90.0 + i + 1.0
        e_mid = y0 + 0.5                               # Voronoi 边 = 目标边
        w_a = R2 * (np.sin(np.deg2rad(e_mid)) - np.sin(np.deg2rad(y0)))
        w_b = R2 * (np.sin(np.deg2rad(y1)) - np.sin(np.deg2rad(e_mid)))
        expect = (w_a * y0 + w_b * y1) / (w_a + w_b)
        assert abs(v[i, 0] - np.float32(expect)) <= 1e-5, (i, v[i, 0], expect)


# ---------------- 真实源契约 + 地标 ----------------

@requires_gst1
def test_real_gst1_contract_and_landmarks():
    """真实源契约：完整网格/值域/极点值/周期列 + 地标节点值（磁盘预落位）。"""
    src = load_gst1()
    assert src.values.shape == (1441, 2880)
    assert src.valid.all()                            # 全网格有值（0 为真实值）
    assert float(src.values.min()) == 0.0
    assert float(src.values.max()) == 20118.0
    # 极点行常值（磁盘实证：南极 1356.8 / 北极 1915.7 m）
    assert src.values[0, 0] == np.float32(1356.8)
    assert src.values[-1, 0] == np.float32(1915.7)
    # 地标节点值（磁盘预落位：孟加拉湾巨厚 / 中太平洋深海薄层 / 青藏高原薄）
    def node(la, lo):
        return src.values[int(round((la + 90.0) * 8)), int(round((lo + 180.0) * 8))]
    assert node(15.0, 88.0) == np.float32(8226.6)     # 孟加拉湾
    assert node(0.0, -160.0) == np.float32(246.5)     # 中太平洋
    assert node(32.0, 90.0) == np.float32(461.9)      # 青藏高原
    # Voronoi 胞边几何（0.125° 半步长 + 极点钳制）
    assert src.lat_edges[0] == -90.0 and src.lat_edges[1] == -89.9375
    assert src.lat_edges[-1] == 90.0
    assert src.lon_res == Fraction(1, 8)


@requires_gst1
def test_real_gst1_build_tiers():
    """两档构建：全域有限 + validity 恒 1 + 地标量级 + 30′ 与 1° 粗化单调性。"""
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
    # 孟加拉湾胞（30′ 胞心 15.25, 88.25）保守平均仍巨厚；中太平洋薄
    r, c = _cell_at("30min", 15.25, 88.25)
    assert v30[r, c] > 6000.0
    r, c = _cell_at("30min", 0.25, -159.75)
    assert v30[r, c] < 600.0
    # 粗档极值受保守平均压缩：max(1°) ≤ max(30′)
    assert float(v1.max()) <= float(v30.max())


# ---------------- Global Basins 读取器：真实源契约 ----------------

@requires_basins
def test_real_basins_loader_contract():
    """768 片 = 764 盆地 + 4 拆分片 + 1 无效几何修复 + 类型词汇闭环。"""
    from cubebuild.sediment import load_global_basins

    gdf, codes, geoms = load_global_basins()
    assert len(gdf) == EXPECTED_POLYGONS == 768
    assert gdf["Basin UBI"].nunique() == EXPECTED_BASINS == 764
    split = [ubi for ubi, grp in gdf.groupby("Basin UBI") if len(grp) > 1]
    assert len(split) == EXPECTED_SPLIT_BASINS == 4
    assert set(gdf["Basin Name"]) >= {"Ross", "Anadyr", "Hope", "Khatyrka"}
    # 无效几何恰 1 个（Komandorskaya），修复后全 valid 且保持面状
    assert int((~gdf.geometry.is_valid).sum()) == EXPECTED_INVALID == 1
    assert gdf[~gdf.geometry.is_valid]["Basin Name"].iloc[0] == "Komandorskaya"
    assert all(g.is_valid and not g.is_empty for g in geoms)
    assert all(
        g.geom_type in ("Polygon", "MultiPolygon") for g in geoms
    )
    # 类型词汇 ⊆ 编码表且码数组一致
    assert set(gdf["Basin Type"]) == set(BASIN_TYPE_CODES)
    assert (codes == gdf["Basin Type"].map(BASIN_TYPE_CODES).to_numpy()).all()
    # 拆分片类型一致（构造性前提，读取期契约）
    for ubi in split:
        grp = gdf[gdf["Basin UBI"] == ubi]
        assert grp["Basin Type"].nunique() == 1


@requires_basins
def test_real_basins_class_landmarks_and_coverage():
    """3′ 主档栅格化地标（真实数据预落位，内点距边 ≥0.9°）+ 覆盖率 ~25.39%。"""
    from cubebuild.sediment import _home_tier_classes

    classes, overlap, uncovered = _home_tier_classes()
    assert classes.shape == grid_shape(BASINS_HOME_TIER)
    inv = {c: lab for lab, c in BASIN_TYPE_CODES.items()}

    coverage = 1.0 - uncovered / classes.size
    assert 0.2530 < coverage < 0.2545          # 盆地面积份额 25.39%（测地近似）
    assert overlap <= 8                        # 3 对数字化共享边界 ≤ 数胞

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
    """真实距离金字塔：域内 0 / 域外有限正 / 独立航空公式抽样互证。"""
    from cubebuild.faults import independent_min_distance_km, segment_table
    from cubebuild.sediment import _basin_rings, _basins_geoms, _distance_pyramid, _tier_inside

    pyr = _distance_pyramid()
    geoms, _codes = _basins_geoms()
    segs, _nd = segment_table(_basin_rings(geoms))
    for tier in ("1deg", "30min", "6min", "3min"):
        v = pyr[tier]
        inside = _tier_inside(tier)
        assert np.isfinite(v).all() and (v >= 0).all()
        assert (v[inside] == 0).all()                  # 区域距离语义（域内 0）
        # 域外抽样 vs 独立航空公式全段暴力（通道同款容差）
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
    # 地标：孟加拉湾内点 = 0；中太平洋远点 > 0
    assert float(pyr["3min"][_cell_at("3min", 18.0, 89.0)]) == 0.0
    assert float(pyr["1deg"][_cell_at("1deg", 0.0, -150.0)]) > 1000.0


# ---------------- e2e（合成源）：三层入 store ----------------

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
    """合成盆地源：3 盆地 4 片（含跨日期线拆分片对）+ 1 对重叠。

    - Bengal 型大方块 (10..25, 0..15) Passive Margin（大面积）
    - Rift 小方块 (15..18, 5..8) Rift（与大方块重叠 → 面积较小者胜）
    - 跨日期线盆地：两片 (170..175, 40..45) / (-175..-170, 40..45) 同 UBI
      同类型 Rift
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

    # GST1：缩格 1° 节点合成源（线性场）
    monkeypatch.setattr(
        m, "GST1_EXPECTED", {"shape": (181, 361), "spacing": _SYN_SPACING},
    )
    src_dir = tmp_path / "gst1-syn"
    src_dir.mkdir()
    _write_synthetic_gst1(
        src_dir / XYZ_NAME,
        lambda la, lo: 500.0 + la + 0.5 * np.cos(np.deg2rad(lo)) * np.cos(np.deg2rad(la)),
    )

    orig_load_gst1 = m.load_gst1          # 先持原函数（替换前），防自递归
    def fake_load_gst1(_src=None):
        return orig_load_gst1(src_dir)

    monkeypatch.setattr(m, "load_gst1", fake_load_gst1)

    # 盆地：合成 gdf + 契约一致的码/几何
    def fake_load_basins(_src=None):
        gdf = _synthetic_basins_gdf()
        codes = gdf["Basin Type"].map(BASIN_TYPE_CODES).to_numpy(dtype=np.uint8)
        return gdf, codes, list(gdf.geometry)

    monkeypatch.setattr(m, "load_global_basins", fake_load_basins)
    monkeypatch.setattr(sc, "load_global_basins", fake_load_basins)


def test_sediment_e2e(tmp_path, monkeypatch):
    """合成源三层全链路：build → 结构验证 PASS → manifest → 保真 PASS。"""
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
    assert gst1["coverage"] == {"1deg": 1.0, "30min": 1.0}     # 源全覆盖
    assert gst1["validity_mask"] == f"{GST1_LAYER_ID}__validity"
    assert gst1["visibility"] == "public"

    cls = entries[BASIN_CLASS_LAYER_ID]
    assert cls["dtype"] == "uint8" and cls["visibility"] == "internal"
    assert cls["validity_mask"] == f"{BASIN_CLASS_LAYER_ID}__validity"
    assert len(json.loads(cls["category_encoding"])) == 8

    dist = entries[BASIN_DISTANCE_LAYER_ID]
    assert dist["dtype"] == "float32" and dist["visibility"] == "internal"
    for tier in ("1deg", "30min", "6min", "3min"):
        assert dist["coverage"][tier] == 1.0                    # 全域有值

    import xarray as xr
    with xr.open_datatree(out, engine="zarr", chunks={}) as dt:
        c3 = dt["/3min"][BASIN_CLASS_LAYER_ID].values
        d3 = dt["/3min"][BASIN_DISTANCE_LAYER_ID].values
        cm3 = dt["/3min"][f"{BASIN_CLASS_LAYER_ID}__validity"].values
        v30 = dt["/30min"][GST1_LAYER_ID].values
        vm30 = dt["/30min"][f"{GST1_LAYER_ID}__validity"].values
        # validity 两通道一致
        assert np.array_equal(cm3, (c3 != 0).astype(np.uint8))
        assert (vm30 == 1).all()                                # GST1 无缺测
        # 重叠像元取面积较小者：SmallOverlap(Rift=7) 胜 Big(PM=6)
        r, c = _cell_at("3min", 6.5, 16.5)                      # 重叠区中心
        assert c3[r, c] == BASIN_TYPE_CODES["Rift"]
        # 域内 0 / 域外正
        assert d3[r, c] == 0.0
        assert d3[_cell_at("3min", -30.0, 60.0)] > 0.0
        # GST1 合成场（500 + lat + 0.5·cos·cos）：30′ 胞值在源值域内
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
    """侧车 parquet：768 形态契约的合成版（4 片 3 盆地）+ 编码列 + internal。"""
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
    # 跨日期线拆分片同 UBI 两行保留源粒度
    cross = gdf[gdf["Basin UBI"] == 3]
    assert len(cross) == 2
    xs0 = cross.iloc[0].geometry.bounds
    xs1 = cross.iloc[1].geometry.bounds
    assert (xs0[0] > 0) != (xs1[0] > 0)          # 两片分落日期线两侧
