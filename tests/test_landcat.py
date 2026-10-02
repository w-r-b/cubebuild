"""测试：陆域类别组——GLiM ASCII 直映 / GUM 双码栅格化 + 厚度分级
解析 / Mooney 构造时代 + 极点 cut 契约 + 侧车导出器 + e2e。

判据只测外部行为：
- 厚度解析：52 种磁盘 DD 字面形态全覆盖（每条 → 期望分级）、代表值
  规则（ft 折算/无单位按 m/单侧界值取界/多段均值）；
- GLiM：头部契约、行序翻转（ASCII 行 0 = 北）、nd/-9999 → 0、众数聚合、
  地标终检（磁盘实证的**胞心**期望值——非地标坐标处的值，两者实测不同）；
- GUM：41 类编码表、读取契约（行数/词汇/宽包围盒跨日期线拒绝、极点
  闭合边放行）、栅格化器与开窗参考实现位级互证、thick=0 跳过、
  乱序传入无关性；
- Mooney：6 时代自老至新有序编码、非极点跨日期线边拒绝、极点 cut
  放行、地标终检（含南极点行）、测地面积 × Age 属性核对；
- 侧车：gum 合并表（source 溯源 + 双编码列，合成源机制测试；真实
  911551+20958 传递由验收构建覆盖）、mooney 真实往返；
- e2e：四层全链路（GUM 合成源注入），结构验证 PASS + manifest 编码表
  + 保真报告 PASS。
"""

import json
from pathlib import Path

import numpy as np
import pytest

import cubebuild.landcat as lc
from cubebuild.kernels import aggregate_mode

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_GLIM = (REPO_ROOT / "original data/lithosphere/glim-lithology"
             / "glim-hartmann2012/glim_wgs84_0point5deg.txt.asc").is_file()
REAL_GUM = (REPO_ROOT / "original data/sediment/gum-unconsolidated"
            / "Boerker_et_al_GUM_v1.0/GUM_v1.0/GUM_v1.0.shp").is_file()
REAL_MOONEY = (REPO_ROOT / "original data/lithosphere/crustal-age-mooney2023"
               / "Shapefiles/ShapefilesGM/GubanovMooney_July2022.shp").is_file()
requires_glim = pytest.mark.skipif(not REAL_GLIM, reason="GLiM 源未集齐")
requires_gum = pytest.mark.skipif(not REAL_GUM, reason="GUM 源未集齐")
requires_mooney = pytest.mark.skipif(not REAL_MOONEY, reason="Mooney 源未集齐")


@pytest.fixture(autouse=True)
def _clear_caches():
    """进程内栅格化缓存逐测试清空（合成源注入不得污染其他测试）。"""
    for c in (lc._GLIM_CACHE, lc._GUM_HOME_CACHE, lc._GUM_MODE_CACHE,
              lc._GUM_SOURCE_CACHE, lc._MOONEY_CACHE, lc._MOONEY_MODE_CACHE):
        c.clear()
    yield
    for c in (lc._GLIM_CACHE, lc._GUM_HOME_CACHE, lc._GUM_MODE_CACHE,
              lc._GUM_SOURCE_CACHE, lc._MOONEY_CACHE, lc._MOONEY_MODE_CACHE):
        c.clear()


def _entry(layer_id, home_tier, tiers, unit="category"):
    from cubebuild.contract import LayerEntry

    return LayerEntry(
        id=layer_id, dims=("lat", "lon"), source="test",
        native_res="vector-polygon", native_res_deg=None,
        home_tier=home_tier, tiers=tuple(tiers), dtype="uint8", unit=unit,
        resampling="mode", mask="validity + landsea（陆域层）", visibility="public",
    )


# ---------------- 厚度分级解析 ----------------

# 52 种磁盘 DD 字面形态 → 期望分级（0 = 无幅度信息）。手工逐条推演
# （代表值规则见 GUM_THICKNESS_RULE），磁盘 value_counts 计数核对。
_DD_VOCABULARY = {
    "nn": 0, "dis": 0,
    "<100ft": 3, ">100ft": 3, "20-100ft": 3,
    ">2m": 1, "10-15m": 3, ">5m": 2, "90-110m": 4, "<2m": 1, "2-6": 2,
    ">5ft": 1, "25-35m": 3, "2.5-5m": 2, "3-5m": 2, ">10m": 2, "1-2.5m": 1,
    "150-160m": 4, "6-10": 2, "5-10m": 2, "<0.5m": 1, "3-6m": 2, "4-6m": 2,
    "6-10m": 2, "0.5-1.5m": 1, "0.5-1m": 1, "1.5-2.5m": 1, "2-4m,2.5-5m": 2,
    "10-20m": 3, "150m": 4, "50-60m": 4, "6-12m": 2, "1-1.5m": 1, "5-7.5m": 2,
    "20-30m": 3, "<0.25m": 1, "0.25-0.5m": 1, "<2": 1, "12-18m": 3, ">20m": 3,
    "100m": 4, "2-3m": 2, "200-210m": 5, "18-30m": 3, ">7.5m": 2, ">9m": 2,
    "1-2m,0.5-1.5m": 1, "0.5m,<1m": 1, "1-3m": 1, "1200m": 5, "1-6m": 2,
    "2-4m": 2,
}


def test_thickness_vocabulary_full_coverage():
    """52 种磁盘 DD 字面形态逐条 → 期望分级（全集，无未识别形态）。"""
    assert len(_DD_VOCABULARY) == 52
    for dd, expect in _DD_VOCABULARY.items():
        got = lc.thickness_class(dd)
        assert got == expect, f"DD={dd!r}: {got} ≠ {expect}"


def test_thickness_parse_rules():
    """代表值规则：ft 折算 0.3048、无单位按 m、单侧界值取界、区间中点、
    多段均值、nn/dis/None → 无幅度信息。"""
    assert lc.parse_thickness_m("10-15m") == pytest.approx(12.5)
    assert lc.parse_thickness_m("20-100ft") == pytest.approx(60 * 0.3048)
    assert lc.parse_thickness_m("2-6") == pytest.approx(4.0)       # 无单位按 m
    assert lc.parse_thickness_m(">5ft") == pytest.approx(5 * 0.3048)
    assert lc.parse_thickness_m("<100ft") == pytest.approx(100 * 0.3048)
    assert lc.parse_thickness_m(">10m") == pytest.approx(10.0)      # 单侧取界
    assert lc.parse_thickness_m("0.5m,<1m") == pytest.approx(0.75)  # 多段均值
    assert lc.parse_thickness_m("2-4m,2.5-5m") == pytest.approx(3.375)
    assert lc.parse_thickness_m("nn") is None
    assert lc.parse_thickness_m("dis") is None
    assert lc.parse_thickness_m(None) is None
    assert lc.parse_thickness_m("") is None
    assert lc.thickness_class("1200m") == 5 and lc.thickness_class("200-210m") == 5


def test_thickness_bins_ordered():
    """有序五档分箱边界：≤2 / (2,10] / (10,50] / (50,200] / >200。"""
    assert [lc.thickness_class(f"{t}m") for t in (0.25, 2, 2.5, 10, 50, 200, 201)] == \
        [1, 1, 2, 2, 3, 4, 5]
    enc = lc.gum_thickness_encoding()
    assert list(enc) == ["1", "2", "3", "4", "5"] and "≤ 2 m" in enc["1"]


# ---------------- GLiM ----------------

def test_glim_encoding_table():
    """GLiM 编码表：15 语义码位（源值码 1..16 除 nd=15）、往返一致。"""
    assert sorted(lc.GLIM_CLASS_LABELS) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16]
    enc = lc.glim_category_encoding()
    assert len(enc) == 15
    assert enc["1"] == "su — unconsolidated sediments"
    assert enc["16"] == "ig — ice and glaciers"
    assert "15" not in enc                       # nd 并入 0


def _write_glim_ascii(tmp_path, grid_north_up, header=None):
    """合成 GLiM ASCII（360×720）：grid 以行 0 = 北侧给出（源约定）。"""
    hdr = header or {
        "ncols": 720, "nrows": 360, "xllcorner": -180.0, "yllcorner": -90.0,
        "cellsize": 0.5, "nodata_value": -9999.0,
    }
    lines = [f"{k:<13}{v:g}" for k, v in hdr.items()]
    for row in grid_north_up:
        lines.append(" ".join(str(int(v)) for v in row))
    src = tmp_path / "glim-hartmann2012"
    src.mkdir(parents=True, exist_ok=True)
    (src / "glim_wgs84_0point5deg.txt.asc").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return tmp_path


def test_glim_loader_flip_and_nodata(tmp_path):
    """行序翻转（ASCII 行 0 = 北 → 输出行 0 = 南）、-9999 与 nd(15) → 0。"""
    grid = np.full((360, 720), -9999, dtype=int)
    grid[0, 0] = 16            # 北极角（源行 0）→ 输出行 359
    grid[359, 719] = 15        # 南极角 + nd → 0
    grid[100, 360] = 1         # 赤道本初子午线 → 输出行 259
    src = _write_glim_ascii(tmp_path, grid)
    out = lc.load_glim_grid(src)
    assert out.shape == (360, 720) and out.dtype == np.uint8
    assert out[359, 0] == 16                    # 北侧落到末行（翻转）
    assert out[0, 719] == 0                     # nd → 0
    assert out[259, 360] == 1
    assert (out == 0).sum() == 360 * 720 - 2    # -9999 与 nd 均归 0


def test_glim_loader_rejects_header_drift(tmp_path):
    """头部契约漂移（cellsize 或形状不符）读取期即拒绝。"""
    grid = np.zeros((360, 720), dtype=int)
    bad = {"ncols": 720, "nrows": 360, "xllcorner": -180.0, "yllcorner": -90.0,
           "cellsize": 0.25, "nodata_value": -9999.0}
    src = _write_glim_ascii(tmp_path, grid, header=bad)
    with pytest.raises(ValueError, match="头部契约漂移"):
        lc.load_glim_grid(src)
    bad2 = dict(bad, cellsize=0.5, nrows=180)
    src2 = _write_glim_ascii(tmp_path / "v2", np.zeros((180, 720), dtype=int), header=bad2)
    with pytest.raises(ValueError, match="数据体形状|头部契约"):
        lc.load_glim_grid(src2)


def test_glim_mode_aggregation_synthetic(tmp_path):
    """1° = 30′ 主档众数聚合（因子 2，nodata=0 不参与、平票取小）。"""
    grid = np.full((360, 720), -9999, dtype=int)
    grid[0:2, 0:4] = [3, 3, 7, 7]               # 北极块：3:3 平票 → 3
    grid[2:4, 0:4] = [-9999, 7, 7, 7]           # 部分缺测块：有效子胞众数 7
    src = _write_glim_ascii(tmp_path, grid)
    home = lc.load_glim_grid(src)
    one = aggregate_mode(home, 2, nodata=0)
    # 源行 0..1（北）→ 输出 30′ 行 358..359 → 1° 行 179；源行 2..3 → 1° 行 178
    assert one[179, 0] == 3                     # 平票取小
    assert one[178, 0] == 7                     # nodata 不参与，众数 7
    assert one[179, 1] == 7 and (one[179, 2:] == 0).all()
    assert (one[:178] == 0).all() and (one[180:] == 0).all()


@requires_glim
def test_glim_real_source_baselines():
    """真实源基线（磁盘实证的契约化）：陆域 87051 胞（87098 非 nodata
    − 47 nd）、值域 ⊆ {0,1..14,16}、地标（**档位胞心**期望值）。"""
    g = lc.load_glim_grid()
    assert int((g > 0).sum()) == 87051
    assert set(np.unique(g).tolist()) <= {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16}
    for name, la, lo, expect in lc.GLIM_LANDMARKS:
        r = int((la + 90.0) / 0.5)
        c = int((lo + 180.0) / 0.5)
        assert int(g[r, c]) == expect, (name, r, c, g[r, c], expect)


# ---------------- GUM ----------------

def test_gum_lithology_encoding_table():
    """41 类字典序编码 1..41、计数与磁盘 value_counts 一致、往返。"""
    codes = lc.GUM_LITHOLOGY_CODES
    assert len(codes) == 41
    assert sorted(codes.values()) == list(range(1, 42))
    assert codes["Ae"] == 1 and codes["Zu"] == 41
    enc = lc.gum_category_encoding()
    assert enc["1"].startswith("Ae – Alluvial")
    assert enc["41"].startswith("Zu – Anthropogenic")
    for xx, c in codes.items():
        assert enc[str(c)].startswith(f"{xx} – ")


def _fake_gum_gdf(rows, xxs, dds):
    import geopandas as gpd
    from shapely.geometry import box

    n = len(rows)
    return gpd.GeoDataFrame(
        {
            "OBJECTID": list(range(1, n + 1)), "Symbol": ["x"] * n,
            "Descriptio": ["d"] * n, "XX": xxs, "YY": ["nn"] * n,
            "ZZ": ["nn"] * n, "AA": ["nn"] * n, "DD": dds,
            "Shape_Leng": [1.0] * n, "Shape_Area": [1.0] * n,
        },
        geometry=[box(*r) for r in rows],
        crs="EPSG:4326",
    )


def _fake_gum_src(tmp_path):
    p = tmp_path / "Boerker_et_al_GUM_v1.0/GUM_v1.0"
    p.mkdir(parents=True, exist_ok=True)
    (p / "GUM_v1.0.shp").write_text("", encoding="utf-8")
    q = tmp_path / "Boerker_et_al_GUM_v1.0/GUM_pyroclastics"
    q.mkdir(parents=True, exist_ok=True)
    (q / "GUM_pyroclastics.shp").write_text("", encoding="utf-8")
    return tmp_path


def test_load_gum_rejects_wrong_rows(tmp_path, monkeypatch):
    import geopandas as gpd
    monkeypatch.setattr(gpd, "read_file", lambda _p: _fake_gum_gdf([(0, 0, 1, 1)], ["Au"], ["nn"]))
    with pytest.raises(ValueError, match="行数"):
        lc.load_gum(_fake_gum_src(tmp_path))


def test_load_gum_rejects_unknown_xx(tmp_path, monkeypatch):
    import geopandas as gpd
    rows = [(float(i), 0.0, float(i) + 0.5, 0.5) for i in range(lc.GUM_EXPECTED_ROWS)]
    monkeypatch.setattr(
        gpd, "read_file",
        lambda _p: _fake_gum_gdf(rows, ["Au"] * (lc.GUM_EXPECTED_ROWS - 1) + ["Xx"], ["nn"] * lc.GUM_EXPECTED_ROWS),
    )
    with pytest.raises(ValueError, match="超出 41 类"):
        lc.load_gum(_fake_gum_src(tmp_path))


def test_load_gum_rejects_dateline_edge(tmp_path, monkeypatch):
    """宽包围盒多边形的跨日期线边（非极点闭合）读取期拒绝。"""
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import Polygon
    crossing = Polygon([(170.0, 0.0), (170.0, 1.0), (-170.0, 1.0), (-170.0, 0.0)])
    rows = [(float(i), 10.0, float(i) + 0.5, 10.5) for i in range(lc.GUM_EXPECTED_ROWS - 1)]
    gdf = _fake_gum_gdf(rows, ["Au"] * (lc.GUM_EXPECTED_ROWS - 1),
                        ["nn"] * (lc.GUM_EXPECTED_ROWS - 1))
    extra = gpd.GeoDataFrame(
        {"XX": ["Au"], "DD": ["nn"], "Shape_Leng": [1.0], "Shape_Area": [1.0]},
        geometry=[crossing], crs="EPSG:4326",
    )
    full = pd.concat([gdf, extra], ignore_index=True)
    monkeypatch.setattr(gpd, "read_file", lambda _p: full)
    with pytest.raises(ValueError, match="跨日期线"):
        lc.load_gum(_fake_gum_src(tmp_path))


def test_load_gum_accepts_polar_cut(tmp_path, monkeypatch):
    """极点闭合边（两端 |lat| ≥ 89.9）放行（Mooney 惯例语义）。"""
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import Polygon
    # 绕南极 cut（Mooney MES 环形态）：仅极点闭合边跨日期线（两端
    # |lat| ≥ 89.9），其余边贴日期线两侧/低纬
    cap = Polygon([
        (-179.99, -88.0), (-179.99, -89.95), (150.0, -89.95), (150.0, -88.0),
        (120.0, -85.0), (60.0, -85.0), (0.0, -87.0), (-120.0, -87.0),
    ])
    rows = [(float(i), 0.0, float(i) + 0.5, 0.5) for i in range(lc.GUM_EXPECTED_ROWS - 1)]
    gdf = _fake_gum_gdf(rows, ["Au"] * (lc.GUM_EXPECTED_ROWS - 1),
                        ["nn"] * (lc.GUM_EXPECTED_ROWS - 1))
    extra = gpd.GeoDataFrame(
        {"XX": ["Au"], "DD": ["nn"], "Shape_Leng": [1.0], "Shape_Area": [1.0]},
        geometry=[cap], crs="EPSG:4326",
    )
    full = pd.concat([gdf, extra], ignore_index=True)
    monkeypatch.setattr(gpd, "read_file", lambda _p: full)
    g, lith, thick = lc.load_gum(_fake_gum_src(tmp_path))
    assert len(g) == lc.GUM_EXPECTED_ROWS


def test_rasterize_gum_classes_matches_windowed_reference():
    """位级互证：全网格单次调用实现 == 开窗参考实现（同重叠规则）。"""
    from shapely.geometry import box
    rng = np.random.default_rng(20260912)
    geoms, lith, thick = [], [], []
    for i in range(60):
        x, y = rng.uniform(-170, 170), rng.uniform(-80, 80)
        w, h = rng.uniform(0.5, 8.0), rng.uniform(0.5, 8.0)
        geoms.append(box(x, y, x + w, y + h))
        lith.append(int(rng.integers(1, 42)))
        thick.append(int(rng.integers(0, 6)))
    lith = np.array(lith, dtype=np.uint8)
    thick = np.array(thick, dtype=np.uint8)
    got_lt, got_th, ov, unc = lc.rasterize_gum_classes(geoms, lith, thick, "1deg")
    ref_lt, ref_ov, ref_unc = lc.rasterize_polygon_classes(geoms, lith, "1deg")
    np.testing.assert_array_equal(got_lt, ref_lt)
    assert ov == ref_ov and unc == ref_unc
    # 厚度层：thick=0 的多边形不烧录 → 与「过滤后几何」的开窗参考一致
    keep = thick != 0
    ref_th, _, _ = lc.rasterize_polygon_classes(
        [g for g, k in zip(geoms, keep) if k], thick[keep], "1deg"
    )
    np.testing.assert_array_equal(got_th, ref_th)


def test_rasterize_gum_classes_smaller_wins_and_order_free():
    """重叠取面积较小多边形；乱序传入（码随几何走）同结果。"""
    from shapely.geometry import box
    big, small = box(0, 0, 10, 10), box(4, 4, 6, 6)
    for geoms, lith, thick in (
        ([big, small], [9, 2], [0, 4]),
        ([small, big], [2, 9], [4, 0]),
    ):
        lt, th, ov, _ = lc.rasterize_gum_classes(
            geoms, np.array(lith, dtype=np.uint8), np.array(thick, dtype=np.uint8), "1deg"
        )
        assert (lt[94:96, 184:186] == 2).all()      # 小 box 胜出
        assert lt[90, 180] == 9
        assert ov == 4
        assert (th[94:96, 184:186] == 4).all() and th[90, 180] == 0


@requires_gum
def test_load_gum_real_baselines():
    """真实源基线：911551 行、41 类、厚码非零 12590（911551−898268−693）。"""
    gdf, lith, thick = lc.load_gum()
    assert len(gdf) == 911551
    assert set(lith.tolist()) <= set(range(1, 42))
    assert int((thick > 0).sum()) == 12590
    assert set(thick.tolist()) <= {0, 1, 2, 3, 4, 5}


# ---------------- Mooney ----------------

def test_mooney_encoding_table():
    """6 时代自老至新有序编码 1..6。"""
    assert lc.MOONEY_AGE_CODES == {
        "Archean": 1, "Paleoproterozoic": 2, "Mesoproterozoic": 3,
        "Neoproterozoic": 4, "Paleozoic": 5, "Cenozoic-Mesozoic": 6,
    }
    enc = lc.mooney_category_encoding()
    assert enc["1"].startswith("Archean") and enc["6"].startswith("Cenozoic-Mesozoic")


def _fake_mooney_gdf(geoms, ages):
    import geopandas as gpd
    return gpd.GeoDataFrame(
        {"Age": ages, "Area": [1.0] * len(ages), "layer": ages, "path": ["p"] * len(ages)},
        geometry=geoms, crs="EPSG:4326",
    )


def _fake_mooney_src(tmp_path):
    p = tmp_path / "Shapefiles/ShapefilesGM"
    p.mkdir(parents=True, exist_ok=True)
    (p / "GubanovMooney_July2022.shp").write_text("", encoding="utf-8")
    return tmp_path


_AGES6 = ["Archean", "Paleoproterozoic", "Mesoproterozoic",
          "Neoproterozoic", "Paleozoic", "Cenozoic-Mesozoic"]


def test_load_mooney_rejects_wrong_rows(tmp_path, monkeypatch):
    import geopandas as gpd
    from shapely.geometry import box
    monkeypatch.setattr(gpd, "read_file", lambda _p: _fake_mooney_gdf([box(0, 0, 1, 1)], ["Archean"]))
    with pytest.raises(ValueError, match="行数"):
        lc.load_mooney_provinces(_fake_mooney_src(tmp_path))


def test_load_mooney_rejects_unknown_age(tmp_path, monkeypatch):
    import geopandas as gpd
    from shapely.geometry import box
    geoms = [box(float(i), 0, float(i) + 1, 1) for i in range(6)]
    monkeypatch.setattr(gpd, "read_file", lambda _p: _fake_mooney_gdf(geoms, _AGES6[:5] + ["Jurassic"]))
    with pytest.raises(ValueError, match="超出编码表"):
        lc.load_mooney_provinces(_fake_mooney_src(tmp_path))


def test_load_mooney_rejects_dateline_edge(tmp_path, monkeypatch):
    """非极点闭合的跨日期线边（两端 |lat| < 89.9）读取期拒绝。"""
    import geopandas as gpd
    from shapely.geometry import Polygon, box
    crossing = Polygon([(179.0, 0.0), (179.0, 1.0), (-179.0, 1.0), (-179.0, 0.0)])
    geoms = [box(float(i), 10, float(i) + 1, 11) for i in range(5)] + [crossing]
    monkeypatch.setattr(gpd, "read_file", lambda _p: _fake_mooney_gdf(geoms, _AGES6))
    with pytest.raises(ValueError, match="跨日期线"):
        lc.load_mooney_provinces(_fake_mooney_src(tmp_path))


def test_load_mooney_accepts_polar_cut(tmp_path, monkeypatch):
    """极点闭合边（两端 |lat| ≥ 89.9，磁盘实证 -89.977/-89.979 形态）放行。"""
    import geopandas as gpd
    from shapely.geometry import Polygon, box
    # 模拟 MES 南极环：沿 -89.977 cut 的极帽（148.58E → -179.995W 闭合）
    cap = Polygon([
        (-179.99, -88.0), (-179.99, -89.977), (148.58, -89.977),
        (148.58, -88.0), (120.0, -85.0), (60.0, -85.0), (0.0, -87.0),
        (-120.0, -87.0),
    ])
    geoms = [box(float(i), 10, float(i) + 5, 15) for i in range(5)] + [cap]
    monkeypatch.setattr(gpd, "read_file", lambda _p: _fake_mooney_gdf(geoms, _AGES6))
    gdf, codes = lc.load_mooney_provinces(_fake_mooney_src(tmp_path))
    assert codes.tolist() == [1, 2, 3, 4, 5, 6]
    # 极帽栅格化：3′ 第 2 行（中心 -89.925，cap 底边 -89.977 之上、
    # 东缘 148.58 之内）应为 cap 时代码（本合成源 = MCE 码 6）；东缘外为 0
    classes, _, _ = lc.rasterize_polygon_classes(geoms, codes, "3min")
    inside = 6572      # 中心 lon < 148.58 的列数：(148.58+180)/0.05 = 6571.6
    assert (classes[1, :inside] == 6).all()
    assert (classes[1, inside:] == 0).all()


@requires_mooney
def test_mooney_real_baselines():
    """真实源基线：6 行、地标终检（含南极点行）、测地面积 × Age 属性。"""
    from pyproj import Geod
    gdf, codes = lc.load_mooney_provinces()
    assert len(gdf) == 6
    classes, overlap, uncovered = lc._mooney_home_classes()
    assert classes.shape == (3600, 7200)
    for name, la, lo, expect in lc.MOONEY_LANDMARKS:
        r = min(int((la + 90.0) * 20), 3599)
        c = min(int((lo + 180.0) * 20), 7199)
        assert int(classes[r, c]) == expect, (name, r, c, classes[r, c], expect)
    geod = Geod(ellps="WGS84")
    for i, row in gdf.iterrows():
        attr_km2 = float(row["Area"]) * 1e6        # Area 字段单位 = 10⁶ km²
        geod_km2 = abs(geod.geometry_area_perimeter(row.geometry)[0]) / 1e6
        assert abs(geod_km2 - attr_km2) / attr_km2 < 1e-3


# ---------------- 侧车导出器 ----------------

def test_export_gum_sidecar_synthetic(tmp_path, monkeypatch):
    """gum 侧车机制（合成源）：合并表 + source 溯源 + 双编码列 + Ic→0。"""
    import geopandas as gpd
    from cubebuild.sidecars import export_gum_unconsolidated

    rows = [(float(i), 0.0, float(i) + 0.5, 0.5) for i in range(lc.GUM_EXPECTED_ROWS)]
    main = _fake_gum_gdf(rows, ["Au"] * (lc.GUM_EXPECTED_ROWS - 1) + ["El"],
                         ["nn"] * (lc.GUM_EXPECTED_ROWS - 1) + ["10-15m"])
    pyro_n = lc.GUM_PYRO_EXPECTED_ROWS
    pyro = _fake_gum_gdf(
        [(float(i), 20.0, float(i) + 0.1, 20.1) for i in range(pyro_n)],
        ["Ic"] * (pyro_n - 1) + ["Iy"], ["nn"] * pyro_n,
    )
    monkeypatch.setattr(lc, "load_gum", lambda _s=None: (main, None, None))
    monkeypatch.setattr(lc, "load_gum_pyroclastics", lambda _s=None: pyro)
    import cubebuild.sidecars as sc
    monkeypatch.setattr(sc, "load_gum", lambda _s=None: (main, None, None))
    monkeypatch.setattr(sc, "load_gum_pyroclastics", lambda _s=None: pyro)

    rec = export_gum_unconsolidated(tmp_path)
    assert rec["id"] == "gum_unconsolidated" and rec["n_features"] == \
        lc.GUM_EXPECTED_ROWS + lc.GUM_PYRO_EXPECTED_ROWS
    assert rec["visibility"] == "public"
    back = gpd.read_parquet(tmp_path / "gum_unconsolidated.parquet")
    assert len(back) == rec["n_features"]
    assert back.crs.to_epsg() == 4326
    for col in ("XX", "YY", "ZZ", "AA", "DD", "Symbol", "Descriptio"):
        assert col in back.columns
    assert back["source"].value_counts().to_dict() == {
        "GUM_v1.0": lc.GUM_EXPECTED_ROWS, "GUM_pyroclastics": pyro_n}
    # 编码列：pyro 的 Ic → 0（不参与栅格化）；主文件 El 行厚码 3
    main_back = back[back["source"] == "GUM_v1.0"]
    pyro_back = back[back["source"] == "GUM_pyroclastics"]
    assert (pyro_back[pyro_back["XX"] == "Ic"]["lithology_code"] == 0).all()
    el = main_back[main_back["XX"] == "El"]
    assert int(el["lithology_code"].iloc[0]) == lc.GUM_LITHOLOGY_CODES["El"]
    assert int(el["thickness_class"].iloc[0]) == 3
    iy = pyro_back[pyro_back["XX"] == "Iy"]
    assert int(iy["lithology_code"].iloc[0]) == lc.GUM_LITHOLOGY_CODES["Iy"]


@requires_mooney
def test_export_mooney_sidecar_roundtrip(tmp_path):
    """mooney 侧车真实往返：6 行、全列 + age_code、CRS。"""
    import geopandas as gpd
    from cubebuild.sidecars import export_mooney2023_provinces

    rec = export_mooney2023_provinces(tmp_path)
    assert rec["id"] == "mooney2023_provinces" and rec["n_features"] == 6
    assert rec["visibility"] == "public"
    back = gpd.read_parquet(tmp_path / "mooney2023_provinces.parquet")
    assert len(back) == 6 and back.crs.to_epsg() == 4326
    for col in ("Age", "Area", "layer", "path"):
        assert col in back.columns
    assert (back["age_code"] == back["Age"].map(lc.MOONEY_AGE_CODES)).all()
    assert back.geometry.is_valid.all()


# ---------------- e2e：四层全链路 ----------------

_MINI = """
version: test
status: test
tiers: [1deg, 30min, 3min, 30sec]
layers:
  - id: lithosphere__glim_lithology_class
    source: glim-lithology（glim_wgs84_0point5deg.txt.asc）
    native_res: 0.5deg
    home_tier: 30min
    tiers: [1deg, 30min]
    dtype: uint8
    unit: category（GLiM 一级岩性，~16 类）
    resampling: mode
    mask: validity + landsea（陆域层）
    notes: 测试契约
  - id: sediment__gum_lithology_class
    source: gum-unconsolidated（911551 多边形）
    native_res: vector-polygon
    home_tier: 30sec
    tiers: [1deg, 30sec]
    dtype: uint8
    unit: category（未固结沉积物岩性分级）
    resampling: mode
    mask: validity + landsea（陆域层）
    notes: 测试契约
  - id: sediment__gum_thickness_class
    source: gum-unconsolidated
    native_res: vector-polygon
    home_tier: 30sec
    tiers: [1deg, 30sec]
    dtype: uint8
    unit: category（厚度分级，有序）
    resampling: mode
    mask: validity + landsea
    notes: 测试契约
  - id: lithosphere__crustal_age_class
    source: crustal-age-mooney2023（6 时代类别多边形）
    native_res: vector-polygon
    home_tier: 3min
    tiers: [1deg, 3min]
    dtype: uint8
    unit: category（ARC/PAP/MES/MCE/PAL/NEO 构造稳定化时代，非出露地层时代）
    resampling: mode
    mask: validity + landsea（大陆层）
    notes: 测试契约
"""


@requires_glim
@requires_mooney
def test_landcat_e2e(tmp_path, monkeypatch):
    """四层全链路：build → 结构验证 PASS（含整型两通道一致）→ manifest
    编码表/分类规则/覆盖率 → 保真报告 PASS。GUM 以合成源注入（真实
    911551 × 30″ 栅格化由验收构建覆盖）。"""
    import cubebuild.cli
    from cubebuild.manifest import read_manifest

    # GUM 合成源：0.1°×0.1° 覆盖块（30″ 主档每块烧 12×12 胞，真实路径）
    from shapely.geometry import box
    n = lc.GUM_EXPECTED_ROWS
    rows = [(float(i % 60) * 0.5, -60.0 + (i % 60) * 0.5,
             float(i % 60) * 0.5 + 0.1, -60.0 + (i % 60) * 0.5 + 0.1)
            for i in range(n)]
    xxs = ["Au"] * (n - 1) + ["Gt"]
    dds = ["nn"] * (n - 1) + ["25-35m"]
    fake = _fake_gum_gdf(rows, xxs, dds)
    monkeypatch.setattr(cubebuild.landcat, "load_gum", lambda _s=None: (
        fake, fake["XX"].map(lc.GUM_LITHOLOGY_CODES).to_numpy(dtype=np.uint8),
        fake["DD"].map(lc.thickness_class).to_numpy(dtype=np.uint8),
    ))
    monkeypatch.setattr(cubebuild.cli, "SIDECARS", {})

    mapping = tmp_path / "mini.yaml"
    mapping.write_text(_MINI, encoding="utf-8")
    out = tmp_path / "cube.zarr"
    rc = cubebuild.cli.main([
        "build", "--mapping", str(mapping), "--out", str(out),
        "--reports", str(tmp_path / "reports"),
    ])
    assert rc == 0

    manifest = read_manifest(out)
    entries = {e["id"]: e for e in manifest["layers"]}
    assert set(entries) == {
        "lithosphere__glim_lithology_class", "sediment__gum_lithology_class",
        "sediment__gum_thickness_class", "lithosphere__crustal_age_class",
    }
    glim_entry = entries["lithosphere__glim_lithology_class"]
    enc = json.loads(glim_entry["category_encoding"])
    assert len(enc) == 15 and enc["16"].startswith("ig")
    assert glim_entry["validity_mask"].endswith("__validity")
    assert 0.30 < glim_entry["coverage"]["30min"] < 0.40      # 陆域 ~33.6%
    thick_entry = entries["sediment__gum_thickness_class"]
    assert "classification_rule" in thick_entry and "有序" in thick_entry["classification_rule"]
    mooney_entry = entries["lithosphere__crustal_age_class"]
    m_enc = json.loads(mooney_entry["category_encoding"])
    assert m_enc["1"].startswith("Archean") and m_enc["6"].startswith("Cenozoic")

    # 保真：四层全 PASS（① 逐位 ② 众数一致 ③ 值域 + 地标/注记）
    for layer_id in entries:
        report = lc.build_landcat_fidelity_report(out, layer_id, [
            t for t in ("1deg", "30min", "3min", "30sec")
            if t in entries[layer_id]["tiers"]
        ])
        assert report["result"] == "PASS", (layer_id, report["checks"])
