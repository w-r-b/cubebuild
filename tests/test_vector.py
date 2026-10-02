"""测试：类别众数核（nodata 语义）+ 多边形栅格化器 + 整型缺测
两通道一致性 + 侧车导出器（GeoParquet / FileGDB 原样传递）。

判据只测外部行为：
- 众数核：nodata 不参与计数、全 nodata 块 → nodata、平票取小、
  nodata=None 保持陆海掩膜语义（0 是真实类）；
- 栅格化器：胞心落入语义（中心不在多边形内的细条不烧录）、重叠
  面积较小者胜出（确定性规则）、极帽 cut 表示（lat=90 闭合边）平面
  栅格化正确；
- 读取器：prov_type 词汇表契约、跨日期线环拒绝（平面栅格化不安全）；
- 掩膜机制：uint8 类别层 0=缺测 的两通道一致性（构建端 + 管线端）；
- 侧车：hasterok GeoParquet 往返（行数/编码列/几何）、limw .gdb
  原样传递完整性（合成树；真实 1.2G 传递由验收构建覆盖）；
- 真实源集成（存在时）：栅格化覆盖/重叠基线 + e2e uint8 层入 store。
"""

import json
from pathlib import Path

import numpy as np
import pytest

from cubebuild.kernels import aggregate_mode
from cubebuild.masks import build_validity_mask, data_missing, validity_layer_name
from cubebuild.sidecars import export_hasterok_gprv, export_limw_polygons
from cubebuild.vector import (
    GPRV_SRC_DIR,
    HOME_TIER,
    PROV_TYPE_CODES,
    _aggregation_factor,
    category_encoding,
    load_gprv,
    rasterize_polygon_classes,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_GPRV = (REPO_ROOT / "original data/lithosphere/tectonic-provinces"
             / "plates&provinces/shp/global_gprv.shp").is_file()
requires_gprv = pytest.mark.skipif(not REAL_GPRV, reason="GPRV 源未集齐")


def _entry(mask="validity", dtype="uint8"):
    from cubebuild.contract import LayerEntry

    return LayerEntry(
        id="lithosphere__hasterok_province_class", dims=("lat", "lon"),
        source="test", native_res="vector-polygon", native_res_deg=None,
        home_tier="3min", tiers=("1deg", "30min", "6min", "3min"),
        dtype=dtype, unit="category（一级分类）", resampling="mode",
        mask=mask, visibility="public",
    )


# ---------------- 类别众数核：nodata 语义 ----------------

def test_aggregate_mode_nodata_excluded():
    """nodata=0 不参与计数：块 [0,0,5,5] 的众数是 5 而非 0。"""
    data = np.array([[0, 0, 5, 5], [5, 5, 0, 0]], dtype=np.uint8)
    out = aggregate_mode(data, 2, nodata=0)
    assert out.tolist() == [[5, 5]]


def test_aggregate_mode_all_nodata_block():
    """全 nodata 块 → nodata（无有效子胞即缺测）。"""
    data = np.zeros((2, 4), dtype=np.uint8)
    out = aggregate_mode(data, 2, nodata=0)
    assert out.tolist() == [[0, 0]]


def test_aggregate_mode_tie_smallest_class():
    """平票取小：块 [3,7 / 3,7] 计数 2:2 → 3（确定性，scipy 同约定）。"""
    data = np.array([[3, 7, 3, 7], [3, 7, 3, 7]], dtype=np.uint8)
    assert aggregate_mode(data, 2, nodata=0).tolist() == [[3, 3]]


def test_aggregate_mode_without_nodata_keeps_landsea_semantics():
    """nodata=None 保持语义：0 是真实类（陆海掩膜 0=海）——
    块 [0,1/0,1] 平票取 0；nodata=0 时同一块众数为 1。"""
    data = np.array([[0, 1, 0, 1], [0, 1, 0, 1]], dtype=np.uint8)
    assert aggregate_mode(data, 2).tolist() == [[0, 0]]
    assert aggregate_mode(data, 2, nodata=0).tolist() == [[1, 1]]


def test_aggregation_factor_integers_only():
    """主档 3′ → 6′/30′/1° = 2/10/20；细于主档（30″）即显式失败。"""
    assert _aggregation_factor("6min") == 2
    assert _aggregation_factor("30min") == 10
    assert _aggregation_factor("1deg") == 20
    with pytest.raises(ValueError, match="整数倍"):
        _aggregation_factor("30sec")


# ---------------- 多边形栅格化器 ----------------

def test_rasterize_center_semantics():
    """胞心落入多边形才烧录：1° 网格上 box(0,0,1,1) 恰烧 (0.5°,0.5°)
    一胞；不含任何胞心的细条 box(0.1,0.2,0.2,0.3) 不烧任何胞。"""
    from shapely.geometry import box

    geoms = [box(0.0, 0.0, 1.0, 1.0), box(0.1, 0.2, 0.2, 0.3)]
    codes = np.array([7, 9], dtype=np.uint8)
    classes, overlap, uncovered = rasterize_polygon_classes(geoms, codes, "1deg")
    # 胞心 (lat=0.5, lon=0.5) → 行 90（-90+r+0.5）、列 180（-180+c+0.5）
    assert classes[90, 180] == 7
    classes[90, 180] = 0
    assert not classes.any()                     # 细条未烧任何胞
    assert overlap == 0
    assert uncovered == classes.size - 1


def test_rasterize_overlap_smaller_wins():
    """重叠像元取面积较小多边形（确定性规则）：大 box 码 9、内部小
    box 码 2 → 小 box 区域 = 2，其余大 box 区域 = 9。"""
    from shapely.geometry import box

    big = box(0.0, 0.0, 10.0, 10.0)
    small = box(4.0, 4.0, 6.0, 6.0)
    classes, overlap, _ = rasterize_polygon_classes(
        [big, small], np.array([9, 2], dtype=np.uint8), "1deg"
    )
    # 乱序传入也应同结果（面积降序规则与传入顺序无关）
    classes2, _, _ = rasterize_polygon_classes(
        [small, big], np.array([2, 9], dtype=np.uint8), "1deg"
    )
    np.testing.assert_array_equal(classes, classes2)
    # 小 box 胞心 (4.5..5.5)² → 4 胞为码 2
    small_cells = classes[94:96, 184:186]
    assert (small_cells == 2).all()
    # 大 box 其余胞 = 9；重叠计数 = 小 box 烧录胞数
    assert classes[90:100, 180:190].sum() - small_cells.sum() == 9 * (100 - 4)
    assert overlap == 4


def test_rasterize_pole_cap_representation():
    """极帽 cut 表示（环沿 lat=90 闭合边 −180→180）：平面栅格化应把
    纬度高于 85° 的全部行烧满（北极盆形态，磁盘实证的唯一 >180° 边）。"""
    from shapely.geometry import Polygon

    cap = Polygon([(-180.0, 85.0), (-180.0, 90.0), (180.0, 90.0), (180.0, 85.0)])
    classes, _, _ = rasterize_polygon_classes([cap], np.array([4], dtype=np.uint8), "1deg")
    # 胞心 lat 85.5..89.5（行 175..179）全列烧录；84.5（行 174）不烧
    assert (classes[175:180, :] == 4).all()
    assert (classes[:175, :] == 0).all()


# ---------------- 读取器：词汇表与几何契约 ----------------

def test_prov_type_encoding_table():
    """编码表：15 类、码 1..15 字典序、category_encoding 往返一致。"""
    assert len(PROV_TYPE_CODES) == 15
    assert sorted(PROV_TYPE_CODES.values()) == list(range(1, 16))
    enc = category_encoding()
    assert enc["1"] == "accretionary complex" and enc["15"] == "wide rift"
    for label, code in PROV_TYPE_CODES.items():
        assert enc[str(code)] == label


def _fake_gdf(rows, prov_types):
    import geopandas as gpd
    from shapely.geometry import box

    n = len(rows)
    return gpd.GeoDataFrame(
        {"prov_type": prov_types, "prov_name": [f"p{i}" for i in range(n)]},
        geometry=[box(*r) for r in rows],
        crs="EPSG:4326",
    )


def _fake_gdf_n(n, prov_type="basin"):
    """n 行同构假源（行数契约 914 通过，供后续检查触发）。"""
    return _fake_gdf(
        [(float(i), 0.0, float(i) + 0.5, 0.5) for i in range(n)],
        [prov_type] * n,
    )


def _fake_src(tmp_path):
    """占位源文件（存在性检查通过；read_file 由 monkeypatch 替换）。"""
    p = tmp_path / "plates&provinces/shp/global_gprv.shp"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("", encoding="utf-8")
    return tmp_path


def test_load_gprv_rejects_unknown_prov_type(tmp_path, monkeypatch):
    import geopandas as gpd

    monkeypatch.setattr(
        gpd, "read_file", lambda _p: _fake_gdf_n(914, prov_type="mystery province")
    )
    with pytest.raises(ValueError, match="超出编码表"):
        load_gprv(_fake_src(tmp_path))


def test_load_gprv_rejects_dateline_crossing(tmp_path, monkeypatch):
    """跨日期线环（Δlon>180° 且非极点闭合边）读取期即拒绝——
    平面栅格化会静默错位。"""
    import geopandas as gpd
    from shapely.geometry import Polygon

    crossing = Polygon([(179.0, 0.0), (179.0, 1.0), (-179.0, 1.0), (-179.0, 0.0)])
    gdf = gpd.GeoDataFrame(
        {"prov_type": ["basin"] * 913 + ["basin"]},
        geometry=[crossing] + list(_fake_gdf_n(913).geometry),
        crs="EPSG:4326",
    )
    monkeypatch.setattr(gpd, "read_file", lambda _p: gdf)
    with pytest.raises(ValueError, match="跨日期线"):
        load_gprv(_fake_src(tmp_path))


def test_load_gprv_rejects_wrong_rows(tmp_path, monkeypatch):
    import geopandas as gpd

    monkeypatch.setattr(
        gpd, "read_file",
        lambda _p: _fake_gdf([(0, 0, 1, 1), (2, 2, 3, 3)], ["basin", "craton"]),
    )
    with pytest.raises(ValueError, match="行数"):
        load_gprv(_fake_src(tmp_path))


# ---------------- 整型缺测两通道一致性 ----------------

def test_data_missing_by_dtype():
    import xarray as xr

    f = xr.DataArray(np.array([1.0, np.nan]))
    u = xr.DataArray(np.array([1, 0], dtype=np.uint8))
    assert data_missing(f).values.tolist() == [False, True]
    assert data_missing(u).values.tolist() == [False, True]


def test_build_validity_mask_uint8_layer():
    """uint8 类别层：掩膜 0 ⟺ 数据 0（编码自 1 起，0 = 无类别）。"""
    import dask.array as da
    import xarray as xr

    arr = xr.DataArray(
        da.from_array(np.array([[0, 3], [7, 0]], dtype=np.uint8), chunks=(2, 2))
    )
    mask = build_validity_mask(arr, _entry())
    assert mask.dtype == np.uint8
    assert mask.compute().values.tolist() == [[0, 1], [1, 0]]
    assert mask.name == validity_layer_name(_entry().id)
    assert mask.attrs["validity_of"] == "lithosphere__hasterok_province_class"


# ---------------- 侧车导出器 ----------------

@requires_gprv
def test_export_hasterok_sidecar_roundtrip(tmp_path):
    """hasterok 侧车：GeoParquet 往返（行数/属性列/编码列/几何/CRS）。"""
    import geopandas as gpd

    rec = export_hasterok_gprv(tmp_path)
    assert rec["id"] == "hasterok_gprv"
    assert rec["format"] == "geoparquet"
    assert rec["visibility"] == "public"
    assert rec["n_features"] == 914
    back = gpd.read_parquet(tmp_path / "hasterok_gprv.parquet")
    assert len(back) == 914
    assert back.crs.to_epsg() == 4326
    # 原始属性表全列保留 + 编码列与立方层一致
    for col in ("prov_name", "prov_type", "prov_group", "lastorogen",
                "crust_type", "area"):
        assert col in back.columns
    assert (back["prov_type_code"] == back["prov_type"].map(PROV_TYPE_CODES)).all()
    assert back.geometry.is_valid.all()


def test_export_limw_passthrough_synthetic(tmp_path):
    """limw 侧车：.gdb 原样传递（合成树验证机制：名字不变、逐文件
    字节级一致、完整性核对、manifest 记录字段）。"""
    src = tmp_path / "src"
    (src / "limw-gis-2015.gdb" / "sub").mkdir(parents=True)
    f1 = src / "limw-gis-2015.gdb" / "a.dat"
    f2 = src / "limw-gis-2015.gdb" / "sub" / "b.dat"
    f1.write_bytes(b"\x00\x01" * 1000)
    f2.write_bytes(b"\xff" * 500)

    out = tmp_path / "sidecars"
    rec = export_limw_polygons(out, src_dir=src)
    dst = out / "limw-gis-2015.gdb"
    assert dst.is_dir() and dst.name.endswith(".gdb")   # 扩展名不可改
    assert rec["id"] == "limw_polygons" and rec["format"] == "filegdb"
    assert rec["n_files"] == 2 and rec["size_bytes"] == 2500
    # LiMW 侧车不纳入公开版（GLiM 层维持
    # public——CC BY 3.0 等价口径）
    assert rec["visibility"] == "internal"
    assert "不栅格化" in rec["notes"] and "internal" in rec["notes"]
    assert (dst / "a.dat").read_bytes() == f1.read_bytes()
    assert (dst / "sub" / "b.dat").read_bytes() == f2.read_bytes()

    # 覆盖式复跑（可重入）：旧内容清空，不残留
    (dst / "stale.dat").write_bytes(b"stale")
    export_limw_polygons(out, src_dir=src)
    assert not (dst / "stale.dat").exists()


def test_export_limw_missing_source(tmp_path):
    with pytest.raises(FileNotFoundError):
        export_limw_polygons(tmp_path, src_dir=tmp_path / "nowhere")


# ---------------- 真实源集成：e2e uint8 类别层入 store ----------------

_MINI = """
version: test
status: test
tiers: [1deg, 6min, 3min]
layers:
  - id: lithosphere__hasterok_province_class
    source: tectonic-provinces（global_gprv.shp，914 多边形，main 版）
    native_res: vector-polygon
    home_tier: 3min
    tiers: [1deg, 6min, 3min]
    dtype: uint8
    unit: category（一级分类）
    resampling: mode
    mask: validity
    notes: 穷尽覆盖型，豁免距离场
"""


@requires_gprv
def test_vector_layer_e2e(tmp_path, monkeypatch):
    """uint8 类别层全链路：build → 结构验证 PASS（含整型两通道一致
    硬判据）→ manifest 编码表/覆盖率 → 保真报告 PASS（众数一致性）。"""
    import cubebuild.cli
    from cubebuild.manifest import read_manifest
    from cubebuild.sidecars import export_hasterok_gprv
    from cubebuild.vector import build_hasterok_fidelity_report

    # 只导出 hasterok 侧车（limw .gdb 1.2G 不入单测）
    monkeypatch.setattr(
        cubebuild.cli, "SIDECARS", {"hasterok_gprv": export_hasterok_gprv}
    )

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
    entry = manifest["layers"][0]
    assert entry["id"] == "lithosphere__hasterok_province_class"
    assert entry["dtype"] == "uint8"
    assert entry["validity_mask"] == "lithosphere__hasterok_province_class__validity"
    # 类别编码表随 manifest 可查（验收项）
    enc = json.loads(entry["category_encoding"])
    assert enc[str(PROV_TYPE_CODES["craton"])] == "craton"
    assert entry["coverage"]["3min"] == pytest.approx(0.99999, abs=1e-5)
    assert entry["coverage"]["1deg"] == 1.0
    assert "rasterization_rule" in entry

    # 侧车登记
    assert [s["id"] for s in manifest["sidecars"]] == ["hasterok_gprv"]

    # 保真：逐位一致 + 众数一致性 + 值域（硬判据全过）
    report = build_hasterok_fidelity_report(out, "lithosphere__hasterok_province_class",
                                            ["1deg", "6min", "3min"])
    assert report["result"] == "PASS", report["checks"]
    n_hard = sum(1 for c in report["checks"] if c["status"] != "report")
    assert n_hard == 3 * 2 + 2   # 3 档 × (逐位+值域) + 2 粗档众数一致性


@requires_gprv
def test_real_gprv_rasterization_baselines():
    """真实源基线（磁盘实证的契约化）：914 行、3′ 覆盖 0.99999、
    重叠 319 胞（0.001%）、未覆盖 241 胞（省界数字化缝隙）。"""
    gdf, codes = load_gprv()
    assert len(gdf) == 914
    assert set(codes) <= set(range(1, 16))
    classes, overlap, uncovered = rasterize_polygon_classes(
        list(gdf.geometry), codes, HOME_TIER
    )
    assert classes.shape == (3600, 7200)
    assert (classes != 0).mean() == pytest.approx(0.9999907, abs=1e-6)
    assert overlap == 319
    assert uncovered == 241


@requires_gprv
def test_geo_spot_checks_against_source():
    """地理终检（常驻回归判据）：10 地标点（克拉通/洋脊/弧后/
    极区/裂谷）处 store 值 == 源多边形在该点最近胞心处的类别
    （contains + 面积最小者，同重叠规则）——方位/镜像类缺陷的
    独立地理验证。"""
    import geopandas as gpd
    import xarray as xr
    from shapely.geometry import Point

    store = REPO_ROOT / "products/cube-v1.0.zarr"
    if not (store / "zarr.json").exists():
        pytest.skip("验收构建 store 不存在（先跑真实构建）")

    gdf = gpd.read_file(
        REPO_ROOT / "original data/lithosphere/tectonic-provinces"
        / "plates&provinces/shp/global_gprv.shp"
    )
    dt = xr.open_datatree(store, engine="zarr", chunks={})
    arr = dt["/3min"].ds["lithosphere__hasterok_province_class"]
    pts = {
        "Tanzania Craton": (-6.0, 33.0), "Mid-Atlantic ridge": (0.0, -20.0),
        "Andes orogen": (-25.0, -68.0), "Izu-Bonin arc": (28.0, 141.0),
        "West Philippine Basin": (16.0, 130.0), "Siberian craton": (62.0, 100.0),
        "Zagros foredeep": (30.0, 50.0), "Red Sea rift": (20.0, 38.5),
        "Arctic ocean": (85.0, 0.0), "East African Rift": (-3.0, 36.0),
    }
    for name, (lat, lon) in pts.items():
        sel = arr.sel(lat=lat, lon=lon, method="nearest")
        clat, clon = float(sel.lat), float(sel.lon)   # 最近胞心坐标
        got = int(sel.values)
        cand = gdf[gdf.contains(Point(clon, clat))]
        if len(cand):
            idx = cand.geometry.area.idxmin()          # 同重叠规则：面积最小者
            expect = PROV_TYPE_CODES[cand.loc[idx, "prov_type"]]
        else:
            expect = 0                                 # 胞心不在任何多边形内
        assert got == expect, (
            f"{name}: store={got} vs 源@胞心({clat:.4f},{clon:.4f})={expect}"
        )
    dt.close()


@requires_gprv
def test_windowed_rasterize_equals_fullgrid_direct():
    """独立路径位级比对：逐多边形开窗烧录（窗口几何/变换自算）vs
    全网格单次 rasterio 直核（同重叠规则：面积降序 + later-wins）。
    两者一致 ⇒ 窗口行/列/变换数学正确（镜像/错位类缺陷在此暴露——
    开发期行号南北约定混用即被本类比对擒获）。"""
    import rasterio.features as rfeatures
    from rasterio.transform import from_origin
    from cubebuild.grids import grid_shape, tier_deg

    gdf, codes = load_gprv()
    tier = "30min"                       # 直核全网格代价可承受的最细档
    nlat, nlon = grid_shape(tier)
    res = tier_deg(tier)
    # 面积降序 → later-wins = 面积较小者胜（与栅格化器同规则）
    order = np.argsort([-g.area for g in gdf.geometry], kind="stable")
    shapes = [(gdf.geometry[i], int(codes[i])) for i in order]
    direct = rfeatures.rasterize(
        shapes, out_shape=(nlat, nlon),
        transform=from_origin(-180.0, 90.0, res, res), fill=0, dtype="uint8",
    )[::-1]   # rasterio 全网格行序北起；项目档位网格行 0 = −90 侧
    windowed, _, _ = rasterize_polygon_classes(list(gdf.geometry), codes, tier)
    np.testing.assert_array_equal(windowed, direct)
