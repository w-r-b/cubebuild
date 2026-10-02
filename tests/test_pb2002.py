"""测试：PB2002 板块三通道。

判据只测外部行为（同款判据）：
- 几何件：unwrap（跨日期线短边走短弧）/ 极帽闭合（净缠绕 ±1 补极点；
  >1 圈拒绝）/ 日期线切分（part 落 [-180,180)、切点线性插值、东西行）；
- 区划网格解析：.GRD 头/数量/卷绕列/值域契约，行北→南定向；
- 6′ 节点采样：定向（北极邻域角点落位）+ 经度卷绕 + 4 角平票取小
  + 源码 +1 平移；
- 读取器：52 板块 / AN-NA 缠绕契约 / 229 段 6048 点 / 极性字节 /
  欧拉极定宽列与前 3 行图例剔除 / 造山带在板块内（穷尽覆盖）；
- 真实源地标：板块（Colorado→NA、南极→AN、跨日期线 PA 两侧…）+
  区划（MAR→慢脊、海沟→5、板块内部→1…）；
- e2e（合成源）：三通道入 store → 结构验证 PASS → manifest（编码表/
  validity 链接）→ 保真报告 PASS → 侧车三件 parquet。
"""

import json
from pathlib import Path

import numpy as np
import pytest

from cubebuild.pb2002 import (
    BOUNDARY_DISTANCE_LAYER_ID,
    EXPECTED_BOUNDARY_POINTS,
    EXPECTED_BOUNDARY_SEGMENTS,
    EXPECTED_PLATES,
    EXPECTED_POLES,
    HOME_TIER,
    PB2002_SRC_DIR,
    PLATE_CODES,
    PLATE_ID_LAYER_ID,
    ZONE_CLASS_LAYER_ID,
    ZONE_LABELS,
    _ring_unwind,
    load_pb2002_boundaries,
    load_pb2002_plates,
    load_pb2002_poles,
    load_tectonic_zones,
    plate_polygon,
    split_dateline_pts,
    zone_encoding,
    zone_home_grid,
)
from cubebuild.grids import grid_shape, tier_centers

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_SOURCE = (PB2002_SRC_DIR / "PB2002_plates.dig.txt").is_file()
requires_pb2002 = pytest.mark.skipif(not REAL_SOURCE, reason="PB2002 源未集齐")

_LATS, _LONS = tier_centers("6min")     # 6′ 胞心坐标（grids 单一来源）


def _nearest_cell(lat: float, lon: float) -> tuple[int, int]:
    """地标点 → 最近 6′ 胞心 (r, c)（浮点坑免疫：argmin 最近中心）。"""
    r = int(np.abs(_LATS - lat).argmin())
    c = int(np.abs(_LONS - lon).argmin())
    return r, c


@pytest.fixture(autouse=True)
def _clear_pb2002_caches(monkeypatch, tmp_path):
    """模块级缓存跨测试隔离 + 距离金字塔缓存重定向（合成数据不落产品目录）。"""
    import cubebuild.pb2002 as m

    caches = (m._PLATE_HOME_CACHE, m._PLATE_MODE_CACHE,
              m._ZONE_HOME_CACHE, m._ZONE_MODE_CACHE, m._PYRAMID_CACHE)
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

def test_plate_encoding_table():
    """板块编码表：52 码、1..52 字典序（编码表可查闭环）。"""
    assert len(PLATE_CODES) == 52
    assert sorted(PLATE_CODES.values()) == list(range(1, 53))
    assert PLATE_CODES["AF"] == 1 and PLATE_CODES["AN"] == 3
    assert PLATE_CODES["YA"] == 52


def test_zone_encoding_table():
    """区划编码表：源码 0..4 → 层码 1..5（+1 平移，0=缺测哨兵）。"""
    assert zone_encoding() == {
        "1": "plate-interior", "2": "active-continent",
        "3": "slow-spreading-ridge", "4": "fast-spreading-ridge", "5": "trench",
    }
    assert len(ZONE_LABELS) == 5


# ---------------- 几何件：unwrap / 极帽闭合 / 日期线切分 ----------------

def test_ring_unwind_dateline_short_way():
    """跨日期线环：短边走短弧，净缠绕 0。"""
    ring = [(179.0, 10.0), (-179.0, 10.0), (-179.0, 0.0), (179.0, 0.0), (179.0, 10.0)]
    unwrapped, winding = _ring_unwind(ring)
    lons = [x for x, _ in unwrapped]
    assert winding == 0
    assert max(abs(a - b) for a, b in zip(lons, lons[1:])) <= 2.0   # 全部短边（≤原最宽 2°）
    assert lons[1] == -179.0 + 360.0


def test_plate_polygon_pole_closure():
    """绕极环：净缠绕 ±1 补极点闭合后极帽在 Jordan 内部。"""
    from shapely.geometry import Point

    # 东向绕北极（0→350 经一圈，回跨日期线闭合）
    ring_n = [((lon if lon < 180 else lon - 360), 78.0 + 6.0 * np.cos(np.deg2rad(lon)))
              for lon in range(0, 360, 20)]
    ring_n.append(ring_n[0])
    poly_n, w_n = plate_polygon(ring_n)
    assert w_n == 1
    assert poly_n.contains(Point(100.0, 89.0))       # 极帽（无闭合时在环外）
    assert not poly_n.contains(Point(100.0, 70.0))
    assert poly_n.bounds[3] == 90.0

    # 西向绕南极（经度递减一圈）
    ring_s = [((lon if lon <= 180 else lon - 360), -80.0)
              for lon in range(360, -1, -20)]
    ring_s.append(ring_s[0])
    poly_s, w_s = plate_polygon(ring_s)
    assert w_s == -1
    assert poly_s.contains(Point(-100.0, -89.0))   # 极帽（unwrap 框 [-360,0] 内）
    assert poly_s.bounds[1] == -90.0


def test_plate_polygon_rejects_double_winding():
    """净缠绕 >1 圈：源形态异常，读取期拒绝。"""
    ring = [(((lon % 360) if (lon % 360) < 180 else (lon % 360) - 360),
             10.0 + 5.0 * np.sin(np.deg2rad(lon)))
            for lon in range(0, 720, 15)]
    with pytest.raises(ValueError, match="净缠绕"):
        plate_polygon(ring)


def test_plate_polygon_ms_spike_repair():
    """MS 型零宽钉形环：make_valid 修复，面积守恒（零面积钉清除）。"""
    # 主体方形 + 末端沿同一条边往复折返（零宽钉）
    ring = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0),
            (0.0, 5.0), (0.0, 10.0), (0.0, 0.0)]
    poly, winding = plate_polygon(ring)
    assert winding == 0
    assert poly.is_valid
    assert abs(poly.area - 100.0) < 1e-9


def test_split_dateline_pts_east_and_west():
    """跨日期线切分：东行/西行、切点插值、part 落 [-180,180)。"""
    parts = split_dateline_pts([(179.5, 10.0), (-179.5, 11.0)])
    assert len(parts) == 2
    assert parts[0][-1] == (180.0, 10.5)                # 切点线性插值
    assert parts[1][0][0] == -180.0                      # 反侧起点
    assert all(-180.0 <= x <= 180.0 for x, _ in parts[1])
    # 西行：-179.5 → 179.5（unwrap 为 −180.5）
    parts_w = split_dateline_pts([(-179.5, 10.0), (179.5, 11.0)])
    assert len(parts_w) == 2
    assert parts_w[0][-1] == (-180.0, 10.5)
    assert parts_w[1][0][0] == 180.0
    # 不跨线：单 part 且坐标不变
    parts_n = split_dateline_pts([(10.0, 0.0), (12.0, 1.0), (11.0, 2.0)])
    assert len(parts_n) == 1
    assert parts_n[0] == [(10.0, 0.0), (12.0, 1.0), (11.0, 2.0)]


# ---------------- 区划网格：解析与 6′ 节点采样 ----------------

def _write_grd(path: Path, grid: np.ndarray):
    """按 .GRD 格式写合成网格（含头两行 + 卷绕副本列）。"""
    lines = ["     0.000     0.100   360.000", "   -90.000     0.100    90.000"]
    for row in grid:
        for k in range(0, row.size, 40):
            lines.append(" ".join(str(int(v)) for v in row[k:k + 40]))
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _load_grd(path: Path):
    """load_tectonic_zones 以目录+固定文件名解析——移入命名目录。"""
    import shutil

    d = path.parent / ("dir_" + path.stem)
    d.mkdir(exist_ok=True)
    shutil.copy(path, d / "PB2002_tectonic_zones.grd.txt")
    return load_tectonic_zones(d)


def test_load_tectonic_zones_orientation_and_contracts(tmp_path):
    """.GRD 解析：行北→南定向 + 头/数量/卷绕列契约拒绝路径。"""
    g = np.zeros((1801, 3601), dtype=np.uint8)
    g[0, 0] = 4          # 北极 lon 0 节点（行北→南）
    g[1800, 0] = 3       # 南极 lon 0 节点
    g[900, 1800] = 2     # 赤道 lon 180 节点
    g[:, -1] = g[:, 0]   # 卷绕副本
    p = tmp_path / "ok.grd.txt"
    _write_grd(p, g)
    loaded = _load_grd(p)
    assert loaded.shape == (1801, 3600)
    assert loaded[0, 0] == 4 and loaded[1800, 0] == 3 and loaded[900, 1800] == 2

    bad = tmp_path / "bad_header.grd.txt"
    bad.write_text("0.000 0.200 360.000\n-90.000 0.100 90.000\n0\n", encoding="ascii")
    with pytest.raises(ValueError, match="网格头"):
        _load_grd(bad)

    bad2 = tmp_path / "bad_count.grd.txt"
    bad2.write_text(
        "     0.000     0.100   360.000\n   -90.000     0.100    90.000\n0 0\n",
        encoding="ascii",
    )
    with pytest.raises(ValueError, match="数值总数"):
        _load_grd(bad2)

    g_bad_wrap = np.zeros((1801, 3601), dtype=np.uint8)
    g_bad_wrap[:, -1] = 3              # 卷绕列 ≠ 首列
    p3 = tmp_path / "bad_wrap.grd.txt"
    _write_grd(p3, g_bad_wrap)
    with pytest.raises(ValueError, match="卷绕"):
        _load_grd(p3)

    g_bad_val = np.zeros((1801, 3601), dtype=np.uint8)
    g_bad_val[500, 500] = 5            # 值域越界（期望 ⊆ 0..4）
    g_bad_val[:, -1] = g_bad_val[:, 0]
    p4 = tmp_path / "bad_val.grd.txt"
    _write_grd(p4, g_bad_val)
    with pytest.raises(ValueError, match="值域"):
        _load_grd(p4)


def test_zone_home_grid_orientation_wrap_and_tie():
    """6′ 采样：北极邻域角点落位（行北→南 + 列西→东 + 半档相位）+ 平票取小。"""
    # 节点 (row 0..1, col 0..1) = (lat 90/89.9, lon 0/0.1)：是 6′ 胞
    # (1799, 1800)（胞心 89.95/0.05）的 4 角——同时验证行/列定向与卷绕
    g = np.zeros((1801, 3600), dtype=np.uint8)
    g[0:2, 0:2] = 4
    home = zone_home_grid(g)
    assert home[1799, 1800] == 5                      # 4 角全 4 → 层码 5
    # 胞 (1799, 1799)：4 角 = {col 3599 的 0, col 0 的 4}×2 → 平票取小源码 0
    # → 层码 1（经度卷绕后仍见 col 0 节点）
    assert home[1799, 1799] == 1
    assert home[0, 0] == 1                            # 远处 4 角全 0 → 1

    # 4 角各不相同 → 4 路平票取最小源码
    g2 = np.zeros((1801, 3600), dtype=np.uint8)
    g2[0:2, 0:2] = np.array([[0, 1], [2, 3]])
    assert zone_home_grid(g2)[1799, 1800] == 1        # min(0,1,2,3)=0 → +1

    # 码位 +1 平移全网格生效
    g3 = np.zeros((1801, 3600), dtype=np.uint8) + 2
    assert (zone_home_grid(g3) == 3).all()


# ---------------- 读取器：真实源契约 + 地标 ----------------

@requires_pb2002
def test_real_plates_contract_and_landmarks():
    """52 板块契约（闭合/缠绕/编码集合）+ 6′ 栅格化地标（含跨日期线与极帽）。"""
    from cubebuild.pb2002 import _home_tier_classes

    polys, codes, labels = load_pb2002_plates()
    assert len(polys) == EXPECTED_PLATES == len(set(labels))
    assert set(labels) == set(PLATE_CODES)
    assert (codes == np.array([PLATE_CODES[k] for k in labels])).all()
    assert all(p.is_valid and not p.is_empty for p in polys)

    classes, overlap, uncovered = _home_tier_classes()
    assert classes.shape == grid_shape(HOME_TIER)
    assert uncovered == 0          # 穷尽覆盖（验收项）
    assert overlap == 0            # 共享数字化边界：无板块间重叠

    inv = {v: k for k, v in PLATE_CODES.items()}

    def code_at(lat, lon):
        return inv[classes[_nearest_cell(lat, lon)]]

    for name, la, lo, want in [
        ("Colorado", 40.0, -105.0, "NA"),
        ("Sahara", 25.0, 10.0, "AF"),
        ("Antarctica deep interior", -85.0, 100.0, "AN"),
        ("South pole adjacent", -89.9, 100.0, "AN"),
        ("North pole adjacent", 89.9, 0.0, "NA"),
        ("Central Pacific", 0.0, -160.0, "PA"),
        ("Kamchatka", 56.0, 159.0, "OK"),
        ("Nazca interior", -20.0, -90.0, "NZ"),
        ("Iceland west of NVZ", 65.0, -22.0, "NA"),      # PB2002 EU/NA 过冰岛北部火山带
        ("North Japan", 37.0, 138.5, "OK"),               # PB2002 OK 含北日本
        ("Tonga trench west side", -20.0, -175.0, "TO"),
        ("Mariana", 16.0, 146.0, "MA"),
        ("Altiplano", -13.0, -73.0, "AP"),
        ("Tibet orogen (in EU)", 31.0, 88.0, "EU"),       # 造山带在板块内（任意内界）
        ("Scotia Sea", -57.5, -40.0, "SC"),
        ("Somali", 5.0, 45.0, "SO"),
        ("Caribbean", 15.0, -75.0, "CA"),
        ("Cocos", 8.0, -95.0, "CO"),
        ("Juan de Fuca", 46.0, -129.0, "JF"),
        ("Philippine Sea", 15.0, 130.0, "PS"),
        ("Amur", 45.0, 125.0, "AM"),
        ("Yangtze", 30.0, 110.0, "YA"),
        ("Sunda", 0.0, 110.0, "SU"),
        ("Arabian", 22.0, 45.0, "AR"),
        ("India", 20.0, 78.0, "IN"),
        ("Bering RU side", 66.0, -172.0, "NA"),
        ("PA west of dateline", 30.0, 179.0, "PA"),
        ("PA east of dateline", 30.0, -179.0, "PA"),    # 跨日期线两侧同板块
    ]:
        got = code_at(la, lo)
        assert got == want, f"{name}: {got} ≠ {want}"


@requires_pb2002
def test_real_boundaries_and_poles_contract():
    """229 段 / 6048 点 / 极性字节 168-40-21 / 欧拉极 52 + 集合互证。"""
    from collections import Counter

    records = load_pb2002_boundaries()
    assert len(records) == EXPECTED_BOUNDARY_SEGMENTS
    assert sum(len(r["pts"]) for r in records) == EXPECTED_BOUNDARY_POINTS
    assert Counter(r["polarity"] for r in records) == Counter({"-": 168, "/": 40, "\\": 21})
    assert all(r["plate_left"] in PLATE_CODES and r["plate_right"] in PLATE_CODES
               for r in records)

    poles = load_pb2002_poles()
    assert len(poles) == EXPECTED_POLES
    assert set(poles["plate"]) == set(PLATE_CODES)
    af = poles[poles["plate"] == "AF"].iloc[0]           # 定宽列 + 图例剔除自证
    assert (af["lat_deg"], af["lon_deg"], af["rate_deg_per_ma"]) == (59.160, -73.174, 0.9270)
    assert af["citation"] == "DeMets et al. [1994]"
    pa = poles[poles["plate"] == "PA"].iloc[0]           # 任意参考系零旋转原样保留
    assert pa["rate_deg_per_ma"] == 0.0
    assert "reference frame" in pa["citation"]


@requires_pb2002
def test_real_zones_semantic_landmarks():
    """区划码语义地标（Kagan 2010 五区；码表经 steps 交叉表实证）。"""
    from cubebuild.pb2002 import _zone_home

    home = _zone_home()
    inv_zone = {c + 1: lab for c, lab in ZONE_LABELS.items()}

    def zone_at(lat, lon):
        return inv_zone[home[_nearest_cell(lat, lon)]]

    for name, la, lo, want in [
        ("MAR equator (slow)", 0.0, -25.0, "slow-spreading-ridge"),
        ("Iceland (slow)", 64.5, -18.0, "slow-spreading-ridge"),
        ("EPR south (fast)", -20.0, -113.0, "fast-spreading-ridge"),
        ("Japan trench", 38.0, 143.0, "trench"),
        ("Tonga trench", -21.0, -173.5, "trench"),
        ("Chile trench", -23.0, -71.5, "trench"),
        ("Alps (active continent)", 46.5, 10.0, "active-continent"),
        ("Tibet", 31.0, 88.0, "active-continent"),
        ("San Andreas", 35.0, -120.0, "active-continent"),
        ("Baikal rift", 53.0, 107.0, "active-continent"),
        ("Hawaii (plate interior)", 20.0, -155.0, "plate-interior"),
        ("Sahara (plate interior)", 25.0, 10.0, "plate-interior"),
        ("Siberia (plate interior)", 62.0, 100.0, "plate-interior"),
    ]:
        got = zone_at(la, lo)
        assert got == want, f"{name}: {got} ≠ {want}"


@requires_pb2002
def test_real_zone_source_share_drift_small():
    """6′ 采样类量近似守恒：主档类别份额 vs 源节点份额漂移 < 1%。"""
    from cubebuild.pb2002 import _zone_home

    src = load_tectonic_zones()
    home = _zone_home()
    src_share = np.bincount(src.ravel(), minlength=5) / src.size
    home_share = np.bincount(home.ravel(), minlength=6)[1:] / home.size
    assert np.abs(home_share - src_share).max() < 0.01


# ---------------- e2e（合成源）：三通道入 store ----------------

_E2E = """
version: test
status: test
tiers: [1deg, 6min]
layers:
  - id: stress_kinematics__pb2002_plate_id
    source: pb2002-plate-boundaries（52 板块多边形）
    native_res: vector-polygon
    home_tier: 6min
    tiers: [1deg, 6min]
    dtype: uint8
    unit: category（52 板块）
    resampling: mode
    mask: 免（全球覆盖）
  - id: stress_kinematics__pb2002_zone_class
    source: pb2002-plate-boundaries（PB2002_tectonic_zones.grd.txt）
    native_res: 0.1deg
    home_tier: 6min
    tiers: [1deg, 6min]
    dtype: uint8
    unit: category（造山带/构造区分类）
    resampling: mode
    mask: validity
  - id: stress_kinematics__pb2002_boundary_distance
    source: pb2002-plate-boundaries（边界线段）
    native_res: vector-line
    home_tier: 6min
    tiers: [1deg, 6min]
    dtype: float32
    unit: km（球面大圆距离）
    resampling: none（逐档从矢量源精确重算）
    mask: 免
"""


def _synthetic_plates():
    """六板块精确平铺合成源（穷尽覆盖硬判据可过）：

    PP 南极帽（西向缠绕 −1 → 极帽闭合）、QQ 北极帽（+1）、
    AA 南中带跨日期线（unwrap 框 [65,235] → 窗口两侧落位）、
    DD 南中带补段、BB 北西带、CC 北东带。所有环内边 <180°
    （真实源同款数字化步进——>180° 内边会使 unwrap 翻转带段）。
    """
    # PP：绕南极帽（经度西向一圈，winding −1）
    ring_pp = [((lon if lon <= 180 else lon - 360), -60.0)
               for lon in range(360, -1, -20)]
    ring_pp.append(ring_pp[0])
    poly_pp, w_pp = plate_polygon(ring_pp)
    assert w_pp == -1
    # QQ：绕北极帽（东向一圈，winding +1）
    ring_qq = [((lon if lon < 180 else lon - 360), 60.0)
               for lon in range(0, 360, 20)]
    ring_qq.append(ring_qq[0])
    poly_qq, w_qq = plate_polygon(ring_qq)
    assert w_qq == 1
    # AA：南中带跨日期线（经度 65→−125 短弧跨 180°，unwrap 框 [65,235]）
    ring_aa = [(65.0, -60.0), (-125.0, -60.0), (-125.0, 0.0), (65.0, 0.0),
               (65.0, -60.0)]
    poly_aa, w_aa = plate_polygon(ring_aa)
    assert w_aa == 0 and poly_aa.bounds[0] == 65.0 and poly_aa.bounds[2] == 235.0
    # DD：南中带补段 [−125, 65]（中间顶点防 >180° 内边）
    ring_dd = [(-125.0, -60.0), (-30.0, -60.0), (65.0, -60.0), (65.0, 0.0),
               (-30.0, 0.0), (-125.0, 0.0), (-125.0, -60.0)]
    poly_dd, w_dd = plate_polygon(ring_dd)
    assert w_dd == 0
    # BB：北西带 [−180, −60]
    ring_bb = [(-180.0, 0.0), (-120.0, 0.0), (-60.0, 0.0), (-60.0, 60.0),
               (-120.0, 60.0), (-180.0, 60.0), (-180.0, 0.0)]
    poly_bb, w_bb = plate_polygon(ring_bb)
    assert w_bb == 0
    # CC：北东带 [−60, 180]
    ring_cc = [(-60.0, 0.0), (0.0, 0.0), (60.0, 0.0), (180.0, 0.0),
               (180.0, 60.0), (60.0, 60.0), (0.0, 60.0), (-60.0, 60.0),
               (-60.0, 0.0)]
    poly_cc, w_cc = plate_polygon(ring_cc)
    assert w_cc == 0
    labels = ["AA", "BB", "CC", "DD", "PP", "QQ"]
    return ([poly_aa, poly_bb, poly_cc, poly_dd, poly_pp, poly_qq],
            np.array([1, 2, 3, 4, 5, 6], dtype=np.uint8), labels)


def _synthetic_zones():
    """合成区划网格：经向条带 + 孤立块（覆盖卷绕/极区路径）。"""
    g = np.zeros((1801, 3600), dtype=np.uint8)
    g[:, 1000:2000] = 1            # lon −80..20 活动大陆条带
    g[700:1100, 2400:3000] = 4     # 俯冲带块（西太平洋）
    g[200:400, 3200:3400] = 3      # 快脊块（跨日期线东侧）
    g[1500:1700, 0:400] = 2        # 南大洋慢脊块（含 lon 0 卷绕西侧）
    return g


def _synthetic_boundaries():
    """合成边界段：常规 + 跨日期线（引擎 3D 路径对日期线不敏感）。"""
    return [
        {"code": "AA-BB", "plate_left": "AA", "plate_right": "BB",
         "polarity": "-", "citation": "synthetic",
         "pts": [(20.0, -35.0), (20.0, 55.0)]},
        {"code": "AA/CC", "plate_left": "AA", "plate_right": "CC",
         "polarity": "/", "citation": "synthetic",
         "pts": [(150.0, 10.0), (170.0, 20.0)]},
        {"code": "BB\\AA", "plate_left": "BB", "plate_right": "AA",
         "polarity": "\\", "citation": "synthetic",
         "pts": [(179.5, -30.0), (-179.5, -30.0)]},
    ]


def _inject_synthetic(monkeypatch):
    import cubebuild.pb2002 as m
    import cubebuild.sidecars as sc

    polys, codes, labels = _synthetic_plates()
    monkeypatch.setattr(m, "load_pb2002_plates", lambda _src=None: (polys, codes, labels))
    monkeypatch.setattr(m, "load_tectonic_zones", lambda _src=None: _synthetic_zones())
    monkeypatch.setattr(
        m, "load_pb2002_boundaries", lambda _src=None: _synthetic_boundaries()
    )
    # 侧车（单独测试）经 sidecars 命名空间注入
    monkeypatch.setattr(sc, "load_pb2002_plates", lambda _src=None: (polys, codes, labels))
    monkeypatch.setattr(
        sc, "load_pb2002_boundaries", lambda _src=None: _synthetic_boundaries()
    )


def test_pb2002_e2e(tmp_path, monkeypatch):
    """合成源三通道全链路：build → 结构验证 PASS → manifest → 保真 PASS。"""
    import cubebuild.cli
    from cubebuild.manifest import read_manifest
    from cubebuild.pb2002 import build_pb2002_fidelity_report

    _inject_synthetic(monkeypatch)

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
    assert set(entries) == {PLATE_ID_LAYER_ID, ZONE_CLASS_LAYER_ID,
                            BOUNDARY_DISTANCE_LAYER_ID}

    plate = entries[PLATE_ID_LAYER_ID]
    assert plate["dtype"] == "uint8" and plate["coverage"] == {"1deg": 1.0, "6min": 1.0}
    assert "validity_mask" not in plate                       # mask: 免
    assert len(json.loads(plate["category_encoding"])) == 52

    zone = entries[ZONE_CLASS_LAYER_ID]
    assert zone["validity_mask"] == f"{ZONE_CLASS_LAYER_ID}__validity"
    assert zone["coverage"] == {"1deg": 1.0, "6min": 1.0}      # 源全域有值

    dist = entries[BOUNDARY_DISTANCE_LAYER_ID]
    assert dist["dtype"] == "float32" and dist["coverage"] == {"1deg": 1.0, "6min": 1.0}

    import xarray as xr
    with xr.open_datatree(out, engine="zarr", chunks={}) as dt:
        p6 = dt["/6min"][PLATE_ID_LAYER_ID].values
        z6 = dt["/6min"][ZONE_CLASS_LAYER_ID].values
        zm6 = dt["/6min"][f"{ZONE_CLASS_LAYER_ID}__validity"].values
        d6 = dt["/6min"][BOUNDARY_DISTANCE_LAYER_ID].values
        assert np.array_equal(zm6, (z6 != 0).astype(np.uint8))
        assert (zm6 == 1).all()                                # 无缺测 → 恒 1
        assert np.isfinite(d6).all() and (d6 >= 0).all()
        # 合成地标：AA 跨日期线两侧 + DD 南中段 + BB 北西 + CC 北东 + 极帽 PP/QQ
        assert p6[_nearest_cell(-30.0, 170.0)] == 1
        assert p6[_nearest_cell(-30.0, -170.0)] == 1
        assert p6[_nearest_cell(-30.0, 0.0)] == 4
        assert p6[_nearest_cell(30.0, -120.0)] == 2
        assert p6[_nearest_cell(30.0, 100.0)] == 3
        assert p6[_nearest_cell(-75.0, 100.0)] == 5
        assert p6[_nearest_cell(75.0, -100.0)] == 6

    for lid in (PLATE_ID_LAYER_ID, ZONE_CLASS_LAYER_ID, BOUNDARY_DISTANCE_LAYER_ID):
        report = build_pb2002_fidelity_report(out, lid, ["1deg", "6min"])
        assert report["result"] == "PASS", report["checks"]
        fails = [c for c in report["checks"] if c["status"] == "FAIL"]
        assert not fails


def test_pb2002_sidecar_synthetic(tmp_path, monkeypatch):
    """侧车三件 parquet：跨日期线边界切 MultiLineString、板块窗口几何、极表。"""
    import geopandas as gpd
    import pandas as pd
    import cubebuild.sidecars as sc

    polys, codes, labels = _synthetic_plates()
    monkeypatch.setattr(sc, "load_pb2002_plates", lambda _src=None: (polys, codes, labels))
    monkeypatch.setattr(
        sc, "load_pb2002_boundaries", lambda _src=None: _synthetic_boundaries()
    )
    pole_rows = [
        {"plate": k, "lat_deg": 0.0, "lon_deg": 0.0,
         "rate_deg_per_ma": 0.0, "citation": "synthetic"}
        for k in labels
    ]
    monkeypatch.setattr(
        sc, "load_pb2002_poles", lambda _src=None: pd.DataFrame(pole_rows)
    )
    monkeypatch.setattr(sc, "EXPECTED_PLATES", len(labels))
    monkeypatch.setattr(sc, "EXPECTED_BOUNDARY_SEGMENTS", len(_synthetic_boundaries()))
    monkeypatch.setattr(sc, "EXPECTED_POLES", len(labels))

    out_dir = tmp_path / "sidecars"
    rec = sc.export_pb2002_plates(out_dir)
    assert rec["id"] == "pb2002_plates" and rec["n_files"] == 3
    # 板块侧车不纳入公开版
    assert rec["visibility"] == "internal"

    plates = gpd.read_parquet(out_dir / "pb2002_plates" / "plates.parquet")
    assert len(plates) == 6
    assert list(plates["plate"]) == ["AA", "BB", "CC", "DD", "PP", "QQ"]
    assert (plates["plate_code"].to_numpy() == np.array([1, 2, 3, 4, 5, 6])).all()
    assert plates.crs.to_epsg() == 4326
    for g in plates.geometry:
        for gg in (g.geoms if g.geom_type == "MultiPolygon" else (g,)):
            xs = np.asarray(gg.exterior.coords)[:, 0]
            assert xs.min() >= -180.0 and xs.max() <= 180.0
    aa = plates[plates["plate"] == "AA"].iloc[0]
    assert aa.geometry.geom_type == "MultiPolygon"            # 跨日期线 → 两侧组分

    bounds = gpd.read_parquet(out_dir / "pb2002_plates" / "boundaries.parquet")
    assert len(bounds) == 3
    split_row = bounds[bounds["code"] == "BB\\AA"].iloc[0]
    assert split_row.geometry.geom_type == "MultiLineString"  # 跨日期线切分
    for gg in split_row.geometry.geoms:
        xs = np.asarray(gg.coords)[:, 0]
        assert xs.min() >= -180.0 and xs.max() <= 180.0
    assert (bounds.iloc[0]["polarity"], bounds.iloc[1]["polarity"]) == ("-", "/")

    poles = pd.read_parquet(out_dir / "pb2002_plates" / "poles.parquet")
    assert len(poles) == 6
    assert set(poles["plate"]) == {"AA", "BB", "CC", "DD", "PP", "QQ"}
