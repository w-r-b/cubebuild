"""测试：0.1° 像元键（左闭/边界鲁棒/日期线）+ 合并去重规则（合成
源）+ 读取器契约（形状/网格自洽/非有限拒绝）+ 双层构建器（密度 attrs /
HFgrid14 相位恒等与 1° 加权）+ 侧车往返 + 真实源基线（合并三数/整编核实
证据数字）+ e2e（双层入 store、manifest 溯源、保真 PASS、侧车落格还原）。
"""

from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

import cubebuild.thermal as th
from cubebuild.thermal import (
    GHFDB_EXPECTED_ROWS,
    GHFDB_SRC_DIR,
    GHFDB_XLSX_NAME,
    HEATFLOW_DENSITY_ID,
    HFGRID14_ID,
    HFGRID_SRC_DIR,
    MERGED_TOTAL,
    NGHF_EXACT_COORD_MATCHES,
    NGHF_EXPECTED_LOCATABLE,
    NGHF_EXPECTED_ROWS,
    NGHF_PIXEL_OVERLAP,
    NGHF_Q_COMPARABLE,
    NGHF_Q_IDENTICAL,
    NGHF_SRC_DIR,
    NGHF_SUPPLEMENT,
    pixel_keys,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_GHFDB = (GHFDB_SRC_DIR / GHFDB_XLSX_NAME).is_file()
REAL_NGHF = (NGHF_SRC_DIR / "NGHF.csv").is_file()
REAL_HFGRID = (HFGRID_SRC_DIR / "HFgrid14.csv").is_file()
requires_heatflow = pytest.mark.skipif(
    not (REAL_GHFDB and REAL_NGHF), reason="GHFDB/NGHF 源未集齐"
)
requires_hfgrid = pytest.mark.skipif(not REAL_HFGRID, reason="HFgrid14 源未集齐")


def _density_entry(layer_id=HEATFLOW_DENSITY_ID, visibility="public"):
    from cubebuild.contract import LayerEntry

    return LayerEntry(
        id=layer_id, dims=("lat", "lon"), source="test",
        native_res="point", native_res_deg=None, home_tier="3min",
        tiers=("1deg", "30min", "6min", "3min"), dtype="uint32",
        unit="count/cell", resampling="sum", mask="免",
        visibility=visibility,
    )


def _hfgrid_entry(visibility="public"):
    from cubebuild.contract import LayerEntry

    return LayerEntry(
        id=HFGRID14_ID, dims=("lat", "lon"), source="test",
        native_res="0.5deg", native_res_deg=0.5, home_tier="30min",
        tiers=("1deg", "30min"), dtype="float32", unit="mW/m2",
        resampling="conservative_area_weighted", mask="validity + landsea",
        visibility=visibility,
    )


# ---------------- 0.1° 像元键 ----------------

def test_pixel_keys_basic_partition():
    """左闭分区：0.05 与 0.15 异键；0.09 与 0.1 异键（0.1 归上胞）。"""
    a = pixel_keys([0.05], [0.05])
    b = pixel_keys([0.15], [0.15])
    assert a[0] != b[0]
    assert pixel_keys([0.09], [0.0])[0] != pixel_keys([0.10], [0.0])[0]


def test_pixel_keys_boundary_robustness():
    """十进制边界点（23.3）确定性归上胞：浮点噪声等价点同键、
    真邻域值（23.29999）异键——裸除法在此非确定（实证 3808 点受扰）。"""
    base = pixel_keys([23.3], [10.0])
    noise = pixel_keys([23.300000000000004], [10.0])   # float 噪声等价
    below = pixel_keys([23.29999], [10.0])
    assert base[0] == noise[0]
    assert base[0] != below[0]


def test_pixel_keys_dateline():
    """经度卷绕：180 ≡ −180 同键；179.95 与 −179.95 属相邻不同像元。"""
    assert pixel_keys([5.0], [180.0])[0] == pixel_keys([5.0], [-180.0])[0]
    assert pixel_keys([5.0], [179.95])[0] != pixel_keys([5.0], [-179.95])[0]


def test_pixel_keys_matches_exact_decimal_reference():
    """独立路径：随机 4 位小数坐标（含边界值），键 == 十进制精确算术
    （Fraction）floor——左闭约定两路径一致。"""
    rng = np.random.default_rng(7)
    lat = rng.uniform(-89, 89, 3000).round(4)
    lon = rng.uniform(-180, 180, 3000).round(4)
    keys = pixel_keys(lat, lon)
    for i in range(len(lat)):
        r = (Fraction(str(lat[i])) + 90) * 10
        c = (Fraction(str(lon[i])) + 180) * 10
        assert keys[i] == int(r) * 3600 + int(c), f"点 {i}: {lat[i]}, {lon[i]}"


# ---------------- 合并去重（合成源，monkeypatch 读取器） ----------------

def _synth_ghfdb():
    import pandas as pd

    return pd.DataFrame({
        "q": [50.0, 60.0, 70.0, 80.0, 90.0],
        "q_uncertainty": [np.nan] * 5,
        "name": ["g1", "g2", "g3", "g4", "g5"],
        "lat": [10.05, 10.15, 5.0, -45.3, 0.0],
        "lon": [20.05, 20.15, -179.95, 170.2, 0.0],
        "reference": ["R1"] * 5,
        "id": ["R24-000001", "R24-000002", "R24-000003", "R24-000004", "R24-000005"],
    })


def _synth_nghf():
    import pandas as pd

    return pd.DataFrame({
        "q": [51.0, 61.0, 62.0, 71.0, 81.0, np.nan],
        "q_uncertainty": [np.nan] * 6,
        "name": ["n1", "n2", "n3", "n4", "n5", "n6"],
        # n1 同像元(10.05 侧)；n2 同像元(10.15 侧)；n3 边界 10.1 → [10.1,10.2)
        # 与 g2(10.15) 同像元；n4 经度空不可定位；n5 新像元；n6 日期线卷绕同 g3
        "lat": [10.08, 10.12, 10.1, 6.0, 12.0, 5.0],
        "lon": [20.08, 20.18, 20.1, np.nan, 21.0, 180.0],
        "reference": ["N1", "N2", "N3", "N4", "N5", "N6"],
        "source_row": np.arange(6),
    })


def test_merge_rule_synthetic(monkeypatch):
    """规则逐案验证：同像元滤除（含 0.1 边界点与日期线卷绕）、新像元保留、
    不可定位排除、GHFDB 全保留、source/source_row 溯源。"""
    monkeypatch.setattr(th, "load_ghfdb", lambda *a, **k: _synth_ghfdb())
    monkeypatch.setattr(th, "load_nghf", lambda *a, **k: _synth_nghf())
    monkeypatch.setattr(th, "NGHF_SUPPLEMENT", 1)
    monkeypatch.setattr(th, "MERGED_TOTAL", 6)
    df = th.merge_heatflow_points("x", "y")
    assert len(df) == 6
    assert (df["source"] == "GHFDB").sum() == 5
    kept = df[df["source"] == "NGHF"]
    # n5（12.0, 21.0 新像元）唯一保留——n6（5.0, 180.0 ≡ −180 → 与 g3
    # (−179.95) 同像元 [−180,−179.9)）被日期线卷绕滤除
    assert list(kept["name"]) == ["n5"]
    # 逐案：同像元三案 + 卷绕案 + 不可定位案全滤除
    names = set(df["name"])
    for dropped in ("n1", "n2", "n3", "n4", "n6"):
        assert dropped not in names
    # 溯源列：GHFDB source_row 0..4；NGHF 保留原行号（n5 = 源第 4 行）
    g_part = df[df["source"] == "GHFDB"]
    assert list(g_part["source_row"]) == [0, 1, 2, 3, 4]
    assert list(df.loc[df["source"] == "NGHF", "source_row"]) == [4]


def test_merge_rejects_contract_drift(monkeypatch):
    """源漂移（补充点数偏离契约）显式失败。"""
    monkeypatch.setattr(th, "load_ghfdb", lambda *a, **k: _synth_ghfdb())
    monkeypatch.setattr(th, "load_nghf", lambda *a, **k: _synth_nghf())
    # 不 patch 契约常量 → 合成源 1 补充 ≠ 期望 5186 → 显式失败
    # （用另一 src 路径避开进程内合并缓存）
    with pytest.raises(ValueError, match="补充点数"):
        th.merge_heatflow_points("x-drift", "y-drift")


# ---------------- 读取器契约 ----------------

def _write_ghfdb_xlsx(path, n_data_rows, n_cols=67):
    import openpyxl

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "GHFDB R20024 v.2026.03"
    names = [f"col{i}" for i in range(n_cols)]
    names[0], names[1], names[2], names[3], names[4], names[19], names[66] = (
        "q", "q_uncertainty", "name", "lat_NS", "long_EW",
        "publication_reference", "ID",
    )
    for _ in range(5):                      # M/R/O、B,S、U score、单位、长名
        ws.append(["x"] * n_cols)
    ws.append(names)
    for i in range(n_data_rows):
        row = [""] * n_cols
        row[0], row[1], row[2] = "50", "?", f"g{i}"
        row[3], row[4] = "10.05", "20.05"
        row[19] = "R"
        row[66] = f"R24-{i:06d}"
        ws.append(row)
    wb.save(path)


def test_load_ghfdb_missing_source(tmp_path):
    with pytest.raises(FileNotFoundError):
        th.load_ghfdb(tmp_path)


def test_load_ghfdb_rejects_wrong_shape(tmp_path):
    _write_ghfdb_xlsx(tmp_path / GHFDB_XLSX_NAME, 3)
    with pytest.raises(ValueError, match="形状"):
        th.load_ghfdb(tmp_path)


_NGHF_HEADER = (
    "country,code1,code2,code3,code4,code5,code6,Name of site,longitude,"
    "latitude,elevation (m),minimum depth (m),maximum depth (m),"
    "number of temperatures,temperature gradient (mK/m),"
    "number of conductivities,conductivity (W/mK),number of heat production,"
    "heat production (µW/m3),heat-flow (mW/m2),uncertainty hf (mW/m2),HFC,"
    "number of sites,reference,date_reference\n"
)


def test_load_nghf_missing_source(tmp_path):
    with pytest.raises(FileNotFoundError):
        th.load_nghf(tmp_path)


def test_load_nghf_rejects_wrong_shape(tmp_path):
    p = tmp_path / "NGHF.csv"
    p.write_text(_NGHF_HEADER + "A,E,L,O,N,A,A,s,1,2,\n", encoding="utf-8")
    with pytest.raises(ValueError, match="形状"):
        th.load_nghf(tmp_path)


def test_load_hfgrid14_missing_source(tmp_path):
    with pytest.raises(FileNotFoundError):
        th.load_hfgrid14(tmp_path)


def test_load_hfgrid14_rejects_wrong_shape(tmp_path):
    p = tmp_path / "HFgrid14.csv"
    p.write_text("longiyude;latitude;HF_pred;sHF_pred;Hf_obs\n"
                 "-179.75;89.75;100;10;\n", encoding="utf-8")
    with pytest.raises(ValueError, match="形状"):
        th.load_hfgrid14(tmp_path)


def _hfgrid_rows(n=259200):
    """完整 0.5° 全球网格行（北起行序，同源约定）。"""
    return [f"{-179.75 + (i % 720) * 0.5};{89.75 - (i // 720) * 0.5};1;;"
            for i in range(n)]


def test_load_hfgrid14_rejects_duplicate_cells(tmp_path):
    p = tmp_path / "HFgrid14.csv"
    rows = ["longiyude;latitude;HF_pred;sHF_pred;Hf_obs"] + _hfgrid_rows()
    rows[2] = rows[1]                          # 首胞重复 → 非完整网格
    p.write_text("\n".join(rows) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="非完整网格"):
        th.load_hfgrid14(tmp_path)


def test_load_hfgrid14_rejects_offgrid_center(tmp_path):
    p = tmp_path / "HFgrid14.csv"
    rows = ["longiyude;latitude;HF_pred;sHF_pred;Hf_obs"] + _hfgrid_rows()
    rows[1] = "-179.75;89.749;100;10;"         # 非规则网格中心
    p.write_text("\n".join(rows) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="偏离"):
        th.load_hfgrid14(tmp_path)


def test_load_hfgrid14_rejects_nonfinite_pred(tmp_path):
    p = tmp_path / "HFgrid14.csv"
    rows = ["longiyude;latitude;HF_pred;sHF_pred;Hf_obs"] + _hfgrid_rows()
    rows[1] = "-179.75;89.75;abc;10;"          # 非数值 HF_pred
    p.write_text("\n".join(rows) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="非有限"):
        th.load_hfgrid14(tmp_path)


# ---------------- HFgrid14 构建器（相位核） ----------------

def _synth_hfgrid_source(values_2d):
    from cubebuild.gravmag import RasterSource

    nlat, nlon = values_2d.shape
    lat_edges = -90.0 + np.arange(nlat + 1, dtype=np.float64) * 0.5
    return RasterSource(
        values=values_2d.astype(np.float32),
        valid=np.ones(values_2d.shape, dtype=bool),
        lat_edges=lat_edges,
        lon_res=Fraction(1, 2),
        lon_phase=Fraction(1, 4),
    )


def test_hfgrid14_home_tier_identity(monkeypatch):
    """30′ 档恒等（源胞边对齐 -180/-90 + lon_phase=0.25）：位级一致。"""
    rng = np.random.default_rng(3)
    vals = rng.uniform(20, 150, (360, 720))
    monkeypatch.setattr(th, "_hfgrid14_source",
                        lambda *a, **k: _synth_hfgrid_source(vals))
    arr = th.build_hfgrid14_heatflow(_hfgrid_entry(), "30min").compute().values
    np.testing.assert_array_equal(arr, vals.astype(np.float32))


def test_hfgrid14_1deg_column_pairing(monkeypatch):
    """1° 档 = 2×2 源块面积加权平均：列恒等场 → 相邻列对均值精确。"""
    vals = np.tile(np.arange(720, dtype=np.float64), (360, 1))
    monkeypatch.setattr(th, "_hfgrid14_source",
                        lambda *a, **k: _synth_hfgrid_source(vals))
    arr = th.build_hfgrid14_heatflow(_hfgrid_entry(), "1deg").compute().values
    for j in (0, 100, 359):
        expect = (vals[0, 2 * j] + vals[0, 2 * j + 1]) / 2.0
        np.testing.assert_allclose(arr[:, j], expect, rtol=1e-6)


def test_hfgrid14_attrs_model_marker(monkeypatch):
    """manifest 溯源字段：product_type=model + license_note + doi。"""
    monkeypatch.setattr(th, "_hfgrid14_source",
                        lambda *a, **k: _synth_hfgrid_source(np.zeros((360, 720))))
    attrs = th.build_hfgrid14_heatflow(_hfgrid_entry(), "30min").attrs
    assert attrs["product_type"].startswith("model")
    assert "license_note" in attrs
    assert attrs["doi"] == "10.1029/2019GC008389"
    assert "missing_value" not in attrs and "_FillValue" not in attrs


# ---------------- 密度层构建器 ----------------

def test_density_builder_attrs(monkeypatch):
    home = np.zeros((3600, 7200), dtype=np.uint32)
    home[0, 0] = 9
    monkeypatch.setattr(th, "heatflow_home_density", lambda *a, **k: home)
    arr = th.build_heatflow_density(_density_entry(), "1deg")
    vals = arr.compute().values
    assert vals.dtype == np.uint32 and vals[0, 0] == 9
    assert int(vals.sum(dtype=np.uint64)) == 9
    for key in ("count_semantics", "composite_rule", "license_note",
                "source_file", "nodata_semantics"):
        assert key in arr.attrs
    assert "96368" in arr.attrs["count_semantics"]
    assert "missing_value" not in arr.attrs and "_FillValue" not in arr.attrs


# ---------------- 真实源基线（磁盘实证的契约化） ----------------

@requires_heatflow
def test_real_merge_baseline():
    """合并三数：GHFDB 91182 全可定位 + NGHF 补充 5186 = 96368。"""
    g = th.load_ghfdb()
    assert len(g) == GHFDB_EXPECTED_ROWS
    assert g["lat"].notna().all() and g["lon"].notna().all()
    assert g["id"].is_unique
    n = th.load_nghf()
    assert len(n) == NGHF_EXPECTED_ROWS
    assert NGHF_EXPECTED_LOCATABLE == 69727
    df = th.merge_heatflow_points()
    assert len(df) == MERGED_TOTAL
    assert (df["source"] == "GHFDB").sum() == GHFDB_EXPECTED_ROWS
    assert (df["source"] == "NGHF").sum() == NGHF_SUPPLEMENT
    home = th.heatflow_home_density()
    assert home.shape == (3600, 7200)
    assert int(home.sum(dtype=np.uint64)) == MERGED_TOTAL


@requires_heatflow
def test_real_incorporation_evidence():
    """整编核实证据数字：坐标重合/q 一致/像元重叠。"""
    ev = th.verify_nghf_incorporation()
    assert ev["exact_coord_matches"] == NGHF_EXACT_COORD_MATCHES
    assert ev["q_comparable"] == NGHF_Q_COMPARABLE
    assert ev["q_identical"] == NGHF_Q_IDENTICAL
    assert ev["pixel_overlap"] == NGHF_PIXEL_OVERLAP
    assert ev["supplement"] == NGHF_SUPPLEMENT
    assert ev["nghf_locatable"] == NGHF_EXPECTED_LOCATABLE
    # 主体整编但未完全 → 合并而非整体转验证参考
    assert 0.7 < ev["exact_coord_matches"] / ev["nghf_locatable"] < 0.9


@requires_hfgrid
def test_real_hfgrid14_baseline():
    src = th.load_hfgrid14()
    assert src.values.shape == (360, 720)
    assert src.valid.all()                       # 预测场全球全覆盖
    assert float(np.nanmin(src.values)) == -6.0
    assert 2900 < float(np.nanmax(src.values)) < 3000
    # 30′ 档位级恒等（相位正确性真实源实证）
    arr = th.build_hfgrid14_heatflow(_hfgrid_entry(), "30min").compute().values
    assert np.array_equal(arr, src.values)


# ---------------- 侧车导出 ----------------

def test_export_sidecar_missing_source(tmp_path):
    from cubebuild.sidecars import export_heatflow_points_merged

    with pytest.raises(FileNotFoundError):
        export_heatflow_points_merged(tmp_path, ghfdb_src=tmp_path / "nowhere")


@requires_heatflow
def test_export_heatflow_sidecar_roundtrip(tmp_path):
    import geopandas as gpd

    from cubebuild.sidecars import export_heatflow_points_merged

    rec = export_heatflow_points_merged(tmp_path)
    assert rec["id"] == "heatflow_points_merged"
    # 热流部分退出：合并点侧车（含 NGHF 记录
    # 本体）转 internal；密度层（聚合分析产物）维持公开
    assert rec["visibility"] == "internal"
    assert rec["n_features"] == MERGED_TOTAL
    assert rec["n_ghfdb"] == GHFDB_EXPECTED_ROWS
    assert rec["n_nghf"] == NGHF_SUPPLEMENT
    back = gpd.read_parquet(tmp_path / "heatflow_points_merged.parquet")
    assert len(back) == MERGED_TOTAL
    assert back.crs.to_epsg() == 4326
    assert back.geometry.isna().sum() == 0        # 合并集全部可定位
    for col in ("source", "source_row", "name", "lat", "lon", "q",
                "q_uncertainty", "reference", "id"):
        assert col in back.columns
    assert set(back["source"].unique()) == {"GHFDB", "NGHF"}
    assert back.loc[back["source"] == "NGHF", "id"].isna().all()
    # GHFDB 行 id 与源记录一一对应（抽 3 行回查）
    g = th.load_ghfdb()
    for i in (0, 50000, GHFDB_EXPECTED_ROWS - 1):
        row = back.iloc[i]
        assert row["id"] == g.iloc[i]["id"]
        assert row["lat"] == pytest.approx(g.iloc[i]["lat"], abs=1e-9)


# ---------------- e2e：双层全链路 ----------------

_MINI = """
version: test
status: test
tiers: [1deg, 30min, 3min]
layers:
  - id: thermal__hfgrid14_heatflow
    source: heatflow（lucazeau2019/HFgrid14.csv）
    native_res: 0.5deg
    home_tier: 30min
    tiers: [1deg, 30min]
    dtype: float32
    unit: mW/m2
    resampling: conservative_area_weighted
    mask: validity + landsea
    notes: 模型产品
  - id: thermal__heatflow_density
    source: heatflow（GHFDB 2024 与 NGHF 合并去重后的单一点集）
    native_res: point
    home_tier: 3min
    tiers: [1deg, 3min]
    dtype: uint32
    unit: count/cell
    resampling: sum
    mask: 免
    notes: 合并去重
"""


@requires_heatflow
@requires_hfgrid
def test_thermal_e2e(tmp_path, monkeypatch):
    """双层全链路：build → 结构验证 PASS → manifest（count_semantics/
    composite_rule/product_type=模型标注/validity 链接）→ 保真 PASS
    （密度含整编核实硬判据）→ 侧车落格还原密度层（互查项）。"""
    import cubebuild.cli
    import geopandas as gpd
    import xarray as xr
    from cubebuild.manifest import read_manifest
    from cubebuild.sidecars import export_heatflow_points_merged

    monkeypatch.setattr(cubebuild.cli, "SIDECARS",
                        {"heatflow_points_merged": export_heatflow_points_merged})

    mapping = tmp_path / "mini.yaml"
    mapping.write_text(_MINI, encoding="utf-8")
    out = tmp_path / "cube.zarr"
    sidecars_dir = tmp_path / "sidecars"
    rc = cubebuild.cli.main([
        "build", "--mapping", str(mapping), "--out", str(out),
        "--reports", str(tmp_path / "reports"),
        "--sidecars", str(sidecars_dir),
    ])
    assert rc == 0

    manifest = read_manifest(out)
    entries = {e["id"]: e for e in manifest["layers"]}
    assert set(entries) == {HEATFLOW_DENSITY_ID, HFGRID14_ID}
    d = entries[HEATFLOW_DENSITY_ID]
    assert d["dtype"] == "uint32" and d["resampling"] == "sum"
    assert "96368" in d["count_semantics"]
    assert "composite_rule" in d and "0.1°" in d["composite_rule"]
    assert "validity_mask" not in d
    h = entries[HFGRID14_ID]
    assert h["dtype"] == "float32"
    assert h["product_type"].startswith("model")       # 模型产品标注（验收项）
    assert h["validity_mask"] == "thermal__hfgrid14_heatflow__validity"
    assert h["coverage"] == {"1deg": 1.0, "30min": 1.0}
    assert [s["id"] for s in manifest["sidecars"]] == ["heatflow_points_merged"]

    # 保真：密度（逐位/计数守恒/整编核实）+ HFgrid14（逐位/积分/分位数）
    for eid, tiers in ((HEATFLOW_DENSITY_ID, ["1deg", "3min"]),
                       (HFGRID14_ID, ["1deg", "30min"])):
        report = th.build_thermal_fidelity_report(out, eid, tiers)
        assert report["result"] == "PASS", report["checks"]

    # 互查：侧车点按坐标落像元 == 密度层 3′ 档逐位
    with xr.open_datatree(out, engine="zarr", chunks={}) as dt:
        store_density = dt["/3min"].ds[HEATFLOW_DENSITY_ID].values
    back = gpd.read_parquet(sidecars_dir / "heatflow_points_merged.parquet")
    from cubebuild.points import points_to_density
    np.testing.assert_array_equal(
        points_to_density(back.geometry.y, back.geometry.x, "3min"),
        store_density,
    )
    # HFgrid14 30′ 档与源位级一致（e2e 再证一次经 store 往返）
    with xr.open_datatree(out, engine="zarr", chunks={}) as dt:
        store_hf = dt["/30min"].ds[HFGRID14_ID].values
    assert np.array_equal(store_hf, th.load_hfgrid14().values)
