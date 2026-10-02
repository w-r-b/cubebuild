"""测试：三源读取器契约（形状/几何断言/翻转自洽/坐标缺陷契约化）+
合成源核验证（GEMMA 30′ 恒等与 1° 配对 / LITHO1.0 直映 / Seton Voronoi
半胞混合与缺测归一）+ attrs 溯源（符号约定/实证依据/时间序列注记）+
真源基线（值域/地标/覆盖/界面同一性）+ 变量清单回报 + e2e（五层入
store、manifest、保真 PASS）。
"""

from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

import cubebuild.litho as li
from cubebuild.contract import LayerEntry
from cubebuild.gravmag import RasterSource

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_GEMMA = (li.GEMMA_SRC_DIR / "crust_bottom.tif").is_file()
REAL_LITHO1 = (li.LITHO1_SRC_DIR / li.LITHO1_FILE).is_file()
REAL_SETON = (li.SETON_SRC_DIR / li.SETON_FILE).is_file()
requires_gemma = pytest.mark.skipif(not REAL_GEMMA, reason="GEMMA 源未集齐")
requires_litho1 = pytest.mark.skipif(not REAL_LITHO1, reason="LITHO1.0 源未集齐")
requires_seton = pytest.mark.skipif(not REAL_SETON, reason="Seton 洋龄源未集齐")


def _entry(layer_id, tiers, home, native_res, res_deg, unit,
           resampling="conservative_area_weighted", mask="validity"):
    return LayerEntry(
        id=layer_id, dims=("lat", "lon"), source="test",
        native_res=native_res, native_res_deg=res_deg, home_tier=home,
        tiers=tuple(tiers), dtype="float32", unit=unit,
        resampling=resampling, mask=mask, visibility="public",
    )


def _gemma_entry(layer_id=li.GEMMA_MOHO_ID):
    return _entry(layer_id, ["1deg", "30min"], "30min", "0.5deg", 0.5,
                  "km（海平面为 0，负值向下）")


def _litho1_entry():
    return _entry(li.LITHO1_LAB_ID, ["1deg"], "1deg", "1deg", 1.0, "km",
                  resampling="none")


def _seton_entry():
    return _entry(li.SEAFLOOR_AGE_ID, ["1deg", "30min", "6min"], "6min",
                  "0.1deg", 0.1, "Ma", mask="validity + landsea（海洋层）")


# ---------------- GEMMA 读取器（合成 tif） ----------------

def _write_tif(path, values, res=0.5, west=-180.0, north=90.0):
    import rasterio
    from rasterio.transform import from_origin

    nlat, nlon = values.shape
    with rasterio.open(
        path, "w", driver="GTiff", height=nlat, width=nlon, count=1,
        dtype="float32", transform=from_origin(west, north, res, res),
        crs="EPSG:4326",
    ) as dst:
        dst.write(values.astype(np.float32), 1)


def _patch_gemma_expected(monkeypatch, shape, res=0.5, west=-180.0, north=90.0):
    nlat, nlon = shape
    monkeypatch.setattr(li, "GEMMA_EXPECTED", {
        "shape": shape, "res": res,
        "bounds": (west, north - nlat * res, west + nlon * res, north),
    })


def test_load_gemma_missing_source(tmp_path):
    with pytest.raises(FileNotFoundError):
        li.load_gemma_tif("crust_bottom.tif", tmp_path)


def test_load_gemma_rejects_wrong_shape(tmp_path, monkeypatch):
    _patch_gemma_expected(monkeypatch, (180, 360))
    _write_tif(tmp_path / "t.tif", np.zeros((360, 720)))
    with pytest.raises(ValueError, match="形状"):
        li.load_gemma_tif("t.tif", tmp_path)


def test_load_gemma_rejects_wrong_bounds(tmp_path, monkeypatch):
    # 形状对但包络偏移（west=-179）→ 注册断言拦截
    _patch_gemma_expected(monkeypatch, (360, 720))
    _write_tif(tmp_path / "t.tif", np.zeros((360, 720)), west=-179.0)
    with pytest.raises(ValueError, match="包络"):
        li.load_gemma_tif("t.tif", tmp_path)


def test_load_gemma_rejects_nonfinite(tmp_path, monkeypatch):
    _patch_gemma_expected(monkeypatch, (360, 720))
    vals = np.zeros((360, 720))
    vals[10, 20] = np.nan
    _write_tif(tmp_path / "t.tif", vals)
    with pytest.raises(ValueError, match="非有限"):
        li.load_gemma_tif("t.tif", tmp_path)


def test_load_gemma_row_flip_and_geometry(tmp_path, monkeypatch):
    """北起 tif → 南起规范化：行序翻转（行号场自证）+ pixel 注册几何
    （lat_edges 对齐 -90、lon_phase=0.25 = 胞边对齐 -180）。"""
    _patch_gemma_expected(monkeypatch, (360, 720))
    raw = np.arange(360, dtype=np.float64)[:, None] * np.ones((1, 720))
    _write_tif(tmp_path / "t.tif", raw)
    src = li.load_gemma_tif("t.tif", tmp_path)
    assert src.values.shape == (360, 720)
    # 行 0（南起）应为 raw 末行（北起 359）
    np.testing.assert_array_equal(src.values[0], np.full(720, 359.0))
    np.testing.assert_array_equal(src.values[-1], np.zeros(720))
    np.testing.assert_array_equal(
        src.lat_edges, -90.0 + np.arange(361, dtype=np.float64) * 0.5
    )
    assert src.lon_res == Fraction(1, 2)
    assert src.lon_phase == Fraction(1, 4)
    assert src.valid.all()


# ---------------- GEMMA 构建器（合成源，核语义） ----------------

def _synth_gemma_source(values):
    nlat, nlon = values.shape
    return RasterSource(
        values=values.astype(np.float32),
        valid=np.ones(values.shape, dtype=bool),
        lat_edges=-90.0 + np.arange(nlat + 1, dtype=np.float64) * 0.5,
        lon_res=Fraction(1, 2),
        lon_phase=Fraction(1, 4),
    )


def _patch_gemma(monkeypatch, values, layer_id=li.GEMMA_MOHO_ID):
    monkeypatch.setattr(li, "_SOURCE_CACHE", {})
    monkeypatch.setattr(
        li, "_gemma_source",
        lambda lid, sd=None: _synth_gemma_source(values),
    )


def test_gemma_30min_identity(monkeypatch):
    """30′ 主档恒等（源胞边对齐 -180/-90 + lon_phase=0.25）：位级一致。"""
    rng = np.random.default_rng(5)
    vals = rng.uniform(-100, 0, (360, 720))
    _patch_gemma(monkeypatch, vals)
    arr = li.build_gemma_moho(_gemma_entry(), "30min").compute().values
    np.testing.assert_array_equal(arr, vals.astype(np.float32))


def test_gemma_1deg_column_pairing(monkeypatch):
    """1° 档 = 2×2 源块面积加权平均：经度恒等场 → 相邻列对均值精确。"""
    vals = np.tile(np.arange(720, dtype=np.float64), (360, 1))
    _patch_gemma(monkeypatch, vals)
    arr = li.build_gemma_moho(_gemma_entry(), "1deg").compute().values
    for j in (0, 100, 359):
        expect = (vals[0, 2 * j] + vals[0, 2 * j + 1]) / 2.0
        np.testing.assert_allclose(arr[:, j], expect, rtol=1e-6)


def test_gemma_basement_attrs(monkeypatch):
    """manifest 溯源：basement 实证依据+ 符号约定 + doi。"""
    _patch_gemma(monkeypatch, np.zeros((360, 720)),
                 layer_id=li.GEMMA_BASEMENT_ID)
    attrs = li.build_gemma_basement_depth(
        _gemma_entry(li.GEMMA_BASEMENT_ID), "30min"
    ).attrs
    assert "结晶基底面" in attrs["basement_evidence"]
    assert "crust_top" in attrs["basement_evidence"]
    assert "负值向下" in attrs["value_convention"]
    assert attrs["doi"] == "10.1016/j.jag.2014.04.002"
    assert attrs["product_type"].startswith("model")
    assert "missing_value" not in attrs and "_FillValue" not in attrs


# ---------------- LITHO1.0 构建器（直映） ----------------

def _synth_litho1_source(values):
    return RasterSource(
        values=values.astype(np.float32),
        valid=np.ones(values.shape, dtype=bool),
        lat_edges=-90.0 + np.arange(181, dtype=np.float64),
        lon_res=Fraction(1),
        lon_phase=Fraction(1, 2),
    )


def _patch_litho1(monkeypatch, values):
    monkeypatch.setattr(li, "_SOURCE_CACHE", {})
    monkeypatch.setattr(
        li, "load_litho1_lab", lambda sd=None: _synth_litho1_source(values)
    )


def test_litho1_1deg_identity(monkeypatch):
    """1° 档直映（免重采样）：位级一致 + 坐标 = 档位中心。"""
    rng = np.random.default_rng(6)
    vals = rng.uniform(10, 300, (180, 360))
    _patch_litho1(monkeypatch, vals)
    arr = li.build_litho1_lab(_litho1_entry(), "1deg")
    np.testing.assert_array_equal(arr.compute().values, vals.astype(np.float32))
    from cubebuild.grids import tier_centers
    lat, lon = tier_centers("1deg")
    np.testing.assert_array_equal(arr.coords["lat"].values, lat)
    np.testing.assert_array_equal(arr.coords["lon"].values, lon)


def test_litho1_rejects_non_1deg(monkeypatch):
    _patch_litho1(monkeypatch, np.zeros((180, 360)))
    with pytest.raises(ValueError, match="仅 1° 档"):
        li.build_litho1_lab(_litho1_entry(), "30min")


def test_litho1_attrs_sign_convention(monkeypatch):
    """符号约定注记：正值向下 + 与 GEMMA 相反提示 + CC BY 4.0
    （构建器自有语义；registration_rule 等源几何注记由真源基线覆盖）。"""
    _patch_litho1(monkeypatch, np.zeros((180, 360)))
    attrs = li.build_litho1_lab(_litho1_entry(), "1deg").attrs
    assert "正值向下" in attrs["value_convention"]
    assert "GEMMA" in attrs["value_convention"]
    assert "CC BY 4.0" in attrs["license_note"]


# ---------------- Seton 洋龄（合成源，Voronoi 语义） ----------------

def _synth_seton_source(values, lat_lo=-10.0, lat_hi=10.0):
    """0.1° gridline Voronoi 几何的合成源（区域纬度带，Slab2 惯例）。

    values: (nlat, 3600)——列 0 = 节点 -180（+180 重复列已弃）。
    """
    nlat = values.shape[0]
    lat_edges = lat_lo + (np.arange(nlat + 1, dtype=np.float64) - 0.5) * 0.1
    lat_edges[0], lat_edges[-1] = lat_lo, lat_hi
    return RasterSource(
        values=values.astype(np.float32),
        valid=np.isfinite(values),
        lat_edges=lat_edges,
        lon_res=Fraction(1, 10),
    )


def _patch_seton(monkeypatch, values):
    monkeypatch.setattr(li, "_SOURCE_CACHE", {})
    monkeypatch.setattr(
        li, "load_seafloor_age",
        lambda sd=None: _synth_seton_source(values),
    )


def test_seton_6min_half_cell_blend(monkeypatch):
    """6′ 档 = gridline→pixel 注册转换：目标像元 = 相邻节点各半混合
    （列恒等场 → 目标列 u = (v_u + v_{u+1})/2 精确）。"""
    nlat = 200                                   # 纬度带 -10..10
    vals = np.tile(np.arange(3600, dtype=np.float64), (nlat, 1))
    _patch_seton(monkeypatch, vals)
    arr = li.build_seafloor_age(_seton_entry(), "6min").compute().values
    # 源带覆盖 6′ 目标行 lat -9.95..9.95 → 行号 (−9.95+90)/0.1 .. (9.95+90)/0.1
    r0, r1 = int(round((-9.95 + 90) * 10)), int(round((9.95 + 90) * 10))
    # u=3599 涉日期线环接（半胞绕回源列 0），取样避开：0/500/3598
    for u in (0, 500, 3598):
        expect = (vals[0, u] + vals[0, u + 1]) / 2.0
        np.testing.assert_allclose(arr[r0:r1, u], expect, rtol=1e-6)
    # 带外行全 NaN（区域源语义）
    assert np.isnan(arr[:r0 - 1, :]).all() and np.isnan(arr[r1 + 1:, :]).all()


def test_seton_nan_normalization_no_dilution(monkeypatch):
    """缺测按有效源胞面积权重归一：单侧节点 NaN → 目标值 = 另一侧节点值
    （无稀释）；双侧 NaN → NaN。"""
    nlat = 200
    vals = np.full((nlat, 3600), 50.0)
    vals[:, 1::2] = np.nan                        # 奇数节点全缺测
    _patch_seton(monkeypatch, vals)
    arr = li.build_seafloor_age(_seton_entry(), "6min").compute().values
    r = int(round((0.05 + 90) * 10))              # 目标行 lat = 0.05（源带内）
    assert np.isfinite(arr[r, :]).all()
    np.testing.assert_allclose(arr[r, :], 50.0, rtol=1e-6)


def test_seton_1deg_block_mean(monkeypatch):
    """1° 档 = 10 节点跨度块加权平均（纬度恒等场：两端节点半权、
    内部 9 节点全权 → 解析均值精确核对）。"""
    nlat = 200
    vals = np.tile(np.arange(3600, dtype=np.float64), (nlat, 1))
    _patch_seton(monkeypatch, vals)
    arr = li.build_seafloor_age(_seton_entry(), "1deg").compute().values
    r = int(round(0.5 + 90))                     # 目标行 lat = 0.5（源带内）
    # j=359 涉日期线环接（末节点半权绕回源列 0），取样避开：0/50/358
    for j in (0, 50, 358):
        v = vals[0]
        k = 10 * j
        expect = (0.5 * v[k] + v[k + 1:k + 10].sum() + 0.5 * v[k + 10]) / 10.0
        np.testing.assert_allclose(arr[r, j], expect, rtol=1e-6)


def test_seton_attrs(monkeypatch):
    """溯源（构建器自有语义）：time_semantics（无时间序列）+ 海洋层语义
    （Voronoi 注册注记由真源基线覆盖）。"""
    _patch_seton(monkeypatch, np.full((200, 3600), 10.0))
    attrs = li.build_seafloor_age(_seton_entry(), "6min").attrs
    assert "仅现今" in attrs["time_semantics"]
    assert "landsea" in attrs["nodata_semantics"]
    assert attrs["doi"] == "10.1029/2020GC009214"


# ---------------- 真源基线（磁盘实证的契约化） ----------------

@requires_gemma
def test_real_gemma_baselines():
    """三 tif 几何/值域契约 + 地标（Karakoram 最深 Moho）+ 层序交叉实证。"""
    for layer_id, (lo, hi) in li.GEMMA_VALUE_RANGES.items():
        src = li.load_gemma_tif(li.GEMMA_FILES[layer_id])
        assert src.values.shape == (360, 720)
        assert src.valid.all()
        assert lo - 1e-3 <= float(src.values.min()) <= lo + 1e-3
        assert hi - 1e-3 <= float(src.values.max()) <= hi + 1e-3
    moho = li.load_gemma_tif("crust_bottom.tif")
    base = li.load_gemma_tif("crust_top.tif")
    # crust_bottom ≤ crust_top 全场（Moho 在基底之下）
    assert (moho.values <= base.values).all()
    thick = base.values - moho.values
    assert 0.0 <= float(thick.min()) and 100.0 < float(thick.max()) < 120.0
    # 最深 Moho = 喀喇昆仑 (34.25, 79.25)，30′ 胞心（南起行号 = 90−35.75)/0.5）
    i, j = np.unravel_index(np.argmin(moho.values), moho.values.shape)
    assert -90.0 + (i + 0.5) * 0.5 == pytest.approx(34.25)
    assert -180.0 + (j + 0.5) * 0.5 == pytest.approx(79.25)


@requires_litho1
def test_real_litho1_baseline():
    """几何/值域/界面同一性/坐标缺陷契约化/地标（Tibet LAB 125 km）。"""
    import netCDF4

    src = li.load_litho1_lab()
    assert src.values.shape == (180, 360)
    assert src.valid.all()
    assert 8.8 < float(src.values.min()) < 8.9
    assert 320.3 < float(src.values.max()) < 320.5
    ds = netCDF4.Dataset(li.LITHO1_SRC_DIR / li.LITHO1_FILE)
    try:
        lab = np.ma.filled(ds.variables[li.LITHO1_LAB_VAR][:], np.nan)
        lid = np.ma.filled(ds.variables[li.LITHO1_LID_VAR][:], np.nan)
        lat = np.asarray(ds.variables["latitude"][:], dtype=np.float64)
    finally:
        ds.close()
    assert np.array_equal(lab, lid)               # 界面同一性（64800/64800）
    # 坐标缺陷 4 行契约化：±14.5/±12.5 写作 ±14.6/±12.6
    for idx, defect in li.LITHO1_COORD_DEFECTS.items():
        assert lat[idx] == defect
    # 地标（中心 → 行列号：row = lat+89.5、col = lon+179.5）：
    # Tibet (30.5, 88.5) LAB = 125.0；太平洋 (0.5, −150.5) = 89.895
    assert float(src.values[120, 268]) == pytest.approx(125.0)
    assert float(src.values[90, 29]) == pytest.approx(89.895, abs=1e-2)


@requires_seton
def test_real_seton_baseline():
    """几何（弃重复列）/覆盖 48.63%/±180 周期/极行/地标（MAR 新生洋壳）。"""
    src = li.load_seafloor_age()
    assert src.values.shape == (1801, 3600)
    assert src.valid.mean() == pytest.approx(0.486275, abs=1e-4)
    assert float(np.nanmin(src.values)) == pytest.approx(0.01, abs=1e-3)
    assert 338.5 < float(np.nanmax(src.values)) < 339.0
    # 北极行单值、南极行全 NaN（源固有）
    assert len(np.unique(src.values[-1])) == 1
    assert np.isnan(src.values[0]).all()
    # 地标：MAR (−30, −13) 新生洋壳 ~4 Ma；NW 太平洋 (15, 150) 老洋壳
    def node(la, lo):
        return float(src.values[int(round((la + 90) * 10)), int(round((lo + 180) * 10))])
    assert 0 < node(-30, -13) < 10
    assert 140 < node(15, 150) < 170
    # 陆地缺测（Tibet/Sahara）
    assert np.isnan(node(30, 88)) and np.isnan(node(23, 10))
    # 溯源：Voronoi 注册注记（真源路径）
    assert "Voronoi" in src.attrs["registration_rule"]


@requires_litho1
def test_real_litho1_inventory():
    """变量清单：171 项、10 层组、ingest 仅 LAB、物性索引完备。"""
    inv = li.litho1_variable_inventory()
    assert inv["n_variables"] == 171
    assert inv["n_v1_1_candidates"] == 170
    assert inv["ingested"] == {li.LITHO1_LAB_ID: li.LITHO1_LAB_VAR}
    idx = inv["layer_interface_index"]
    assert len(idx) == 10
    assert set(idx["asthenospheric_mantle"]) == {"top"}
    assert set(idx["lid"]) == {"top", "bottom"}
    assert set(idx["water"]["top"]) == {
        "depth", "density", "vp", "vs", "qkappa", "qmu", "vp2", "vs2", "eta"
    }


def test_inventory_missing_source(tmp_path):
    with pytest.raises(FileNotFoundError):
        li.litho1_variable_inventory(tmp_path)


# ---------------- e2e：五层全链路 ----------------

_MINI = """
version: test
status: test
tiers: [1deg, 30min, 6min]
layers:
  - id: lithosphere__gemma_moho
    source: crust-models（ecm1-gemma/crust_bottom.tif，GEMMA 输出）
    native_res: 0.5deg
    home_tier: 30min
    tiers: [1deg, 30min]
    dtype: float32
    unit: km（海平面为 0，负值向下）
    resampling: conservative_area_weighted
    mask: validity
    notes: 测试
  - id: lithosphere__gemma_moho_err
    source: crust-models（ecm1-gemma/crust_moho_err.tif）
    native_res: 0.5deg
    home_tier: 30min
    tiers: [1deg, 30min]
    dtype: float32
    unit: km
    resampling: conservative_area_weighted
    mask: validity
    notes: 测试
  - id: lithosphere__gemma_basement_depth
    source: crust-models（ecm1-gemma/crust_top.tif，GEMMA 结晶基底面）
    native_res: 0.5deg
    home_tier: 30min
    tiers: [1deg, 30min]
    dtype: float32
    unit: km（海平面为 0，负值向下）
    resampling: conservative_area_weighted
    mask: validity
    notes: 测试
  - id: lithosphere__litho1_lab
    source: litho1（LITHO1.0 nc，软流圈顶界深度）
    native_res: 1deg
    home_tier: 1deg
    tiers: [1deg]
    dtype: float32
    unit: km
    resampling: none
    mask: validity
    notes: 测试
  - id: lithosphere__seafloor_age
    source: seafloor-age-seton2020（Seafloor_Age_Grid/Seton_etal_2020_PresentDay_AgeGrid.nc）
    native_res: 0.1deg
    home_tier: 6min
    tiers: [1deg, 30min, 6min]
    dtype: float32
    unit: Ma
    resampling: conservative_area_weighted
    mask: validity + landsea（海洋层）
    notes: 测试
"""


@pytest.mark.skipif(
    not (REAL_GEMMA and REAL_LITHO1 and REAL_SETON), reason="岩石圈三源未集齐"
)
def test_litho_e2e(tmp_path, monkeypatch):
    """五层全链路：build → 结构验证 PASS → manifest（validity 链接/覆盖率/
    溯源字段）→ 保真五层 PASS → LITHO1 变量清单回报落盘。"""
    import cubebuild.cli
    from cubebuild.manifest import read_manifest

    monkeypatch.setattr(cubebuild.cli, "SIDECARS", {})   # 侧车不参与该路径

    mapping = tmp_path / "mini.yaml"
    mapping.write_text(_MINI, encoding="utf-8")
    out = tmp_path / "cube.zarr"
    reports = tmp_path / "reports"
    rc = cubebuild.cli.main([
        "build", "--mapping", str(mapping), "--out", str(out),
        "--reports", str(reports),
        "--sidecars", str(tmp_path / "sidecars"),
    ])
    assert rc == 0

    manifest = read_manifest(out)
    entries = {e["id"]: e for e in manifest["layers"]}
    assert set(entries) == {
        li.GEMMA_MOHO_ID, li.GEMMA_MOHO_ERR_ID, li.GEMMA_BASEMENT_ID,
        li.LITHO1_LAB_ID, li.SEAFLOOR_AGE_ID,
    }
    for lid in entries:
        assert entries[lid]["validity_mask"] == f"{lid}__validity"
        assert entries[lid]["resampling"] == (
            "none" if lid == li.LITHO1_LAB_ID else "conservative_area_weighted"
        )
    # 全覆盖层（GEMMA 三层 + LITHO1）覆盖率 1.0；洋龄 < 1（海洋层）
    for lid in (li.GEMMA_MOHO_ID, li.GEMMA_MOHO_ERR_ID,
                li.GEMMA_BASEMENT_ID, li.LITHO1_LAB_ID):
        assert all(v == 1.0 for v in entries[lid]["coverage"].values())
    assert all(0.45 < v < 0.6 for v in entries[li.SEAFLOOR_AGE_ID]["coverage"].values())
    # 溯源字段流转（attrs → manifest）
    assert "结晶基底面" in entries[li.GEMMA_BASEMENT_ID]["basement_evidence"]
    assert "正值向下" in entries[li.LITHO1_LAB_ID]["value_convention"]
    assert "仅现今" in entries[li.SEAFLOOR_AGE_ID]["time_semantics"]
    assert entries[li.LITHO1_LAB_ID]["doi"] == "10.1002/2013JB010626"

    # LITHO1 变量清单回报
    inv_path = reports / "litho1-inventory.json"
    assert inv_path.is_file()
    import json
    inv = json.loads(inv_path.read_text(encoding="utf-8"))
    assert inv["n_variables"] == 171

    # 保真五层 PASS
    for lid, tiers in (
        (li.GEMMA_MOHO_ID, ["1deg", "30min"]),
        (li.GEMMA_MOHO_ERR_ID, ["1deg", "30min"]),
        (li.GEMMA_BASEMENT_ID, ["1deg", "30min"]),
        (li.LITHO1_LAB_ID, ["1deg"]),
        (li.SEAFLOOR_AGE_ID, ["1deg", "30min", "6min"]),
    ):
        report = li.build_litho_fidelity_report(out, lid, tiers)
        assert report["result"] == "PASS", report["checks"]
