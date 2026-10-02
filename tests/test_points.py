"""测试：计数求和核 + 点 → 像元落格器（经度卷绕/极缘）+ 读取器
契约 + 侧车导出器（GeoParquet 往返 / poles 合并）+ 计数守恒 + 侧车 ↔
密度层互查。

判据只测外部行为（计数层保真判据 = 总和守恒）：
- sum 核：块内求和精确、计数守恒、非整数倍显式失败；
- 落格器：已知点精确落胞、GSRM 0..360 与 WSM ±180 两制同点同胞、
  日期线卷绕（180 ≡ −180 → 列 0）、lat=90 极缘归最北胞、非有限/
  出界坐标拒绝、bincount 与逐点循环加法两独立路径位级一致；
- 读取器：形状契约（WSM 100842×40 / GSRM 22511×9）拒漂移、解析
  失败行拒绝、poles 板块码 "NA" 不被 pandas 吞成 NaN；
- 真实源基线（磁盘实证的契约化）：WSM 170 空坐标行（全 Xmi）可定位
  100672；GSRM 三数 22511/18440/16681；主档密度总和守恒；
- 侧车：WSM 全行保留 + 170 空几何；GSRM 22511 行 + poles 203 行
  4 框架；
- e2e：两密度层入 store（uint32 免掩膜、无 30″ 档）+ manifest
  count_semantics + fidelity PASS + 侧车落格还原密度层（互查项）。
"""

from pathlib import Path

import numpy as np
import pytest

from cubebuild.kernels import aggregate_sum
from cubebuild.points import (
    GSRM_EXPECTED_ROWS,
    GSRM_LAYER_ID,
    GSRM_POLES_NAMES,
    GSRM_UNIQUE_COORDS,
    GSRM_UNIQUE_SITES,
    HOME_TIER,
    WSM_CSV_NAME,
    WSM_EXPECTED_LOCATABLE,
    WSM_EXPECTED_ROWS,
    WSM_LAYER_ID,
    _aggregation_factor,
    build_wsm_density,
    load_gsrm_gps,
    load_gsrm_poles,
    load_wsm_points,
    points_to_density,
    wsm_home_density,
    gsrm_home_density,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_WSM = (REPO_ROOT / "original data/stress-kinematics/wsm2025-stress"
            / WSM_CSV_NAME).is_file()
requires_wsm = pytest.mark.skipif(not REAL_WSM, reason="WSM 源未集齐")
REAL_GSRM = (REPO_ROOT / "original data/stress-kinematics/gsrm-strain"
             / "GPS_ITRF08.gmt").is_file()
requires_gsrm = pytest.mark.skipif(not REAL_GSRM, reason="GSRM 源未集齐")


def _entry(layer_id=WSM_LAYER_ID, visibility="public"):
    from cubebuild.contract import LayerEntry

    return LayerEntry(
        id=layer_id, dims=("lat", "lon"), source="test",
        native_res="point", native_res_deg=None, home_tier="3min",
        tiers=("1deg", "30min", "6min", "3min"), dtype="uint32",
        unit="count/cell", resampling="sum", mask="免",
        visibility=visibility,
    )


# ---------------- 计数求和核 ----------------

def test_aggregate_sum_exact_and_conservation():
    """块内求和精确 + 计数守恒（Σ粗 = Σ细）+ dtype uint32。"""
    data = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.uint32)
    out = aggregate_sum(data, 2)
    assert out.tolist() == [[14, 22]]
    assert out.dtype == np.uint32
    assert int(out.sum(dtype=np.uint64)) == int(data.sum(dtype=np.uint64))


def test_aggregate_sum_rejects_non_integer_factor():
    with pytest.raises(ValueError, match="整数倍"):
        aggregate_sum(np.zeros((3, 4), dtype=np.uint32), 2)


def test_aggregation_factor_integers_only():
    """主档 3′ → 6′/30′/1° = 2/10/20；细于主档（30″）即显式失败。"""
    assert _aggregation_factor("6min") == 2
    assert _aggregation_factor("30min") == 10
    assert _aggregation_factor("1deg") == 20
    with pytest.raises(ValueError, match="整数倍"):
        _aggregation_factor("30sec")


# ---------------- 点 → 像元落格器 ----------------

def test_points_to_density_known_cells():
    """1° 网格已知落胞：(0.3, 0.7) → 行 90 列 180；(−89.9, −179.9) → (0, 0)。"""
    grid = points_to_density([0.3, -89.9], [0.7, -179.9], "1deg")
    assert grid.dtype == np.uint32
    assert grid[90, 180] == 1
    assert grid[0, 0] == 1
    assert int(grid.sum(dtype=np.uint64)) == 2


def test_points_to_density_lon_conventions():
    """GSRM 0..360 制与 WSM ±180 制同点同胞（卷绕 238.327 ≡ −121.673
    → 行 126 列 58）；经度 0 → 列 180。"""
    a = points_to_density([36.914], [238.327], "1deg")
    b = points_to_density([36.914], [-121.673], "1deg")
    np.testing.assert_array_equal(a, b)
    assert a[126, 58] == 1
    assert points_to_density([0.0], [0.0], "1deg")[90, 180] == 1


def test_points_to_density_dateline_and_pole_edge():
    """日期线卷绕：lon=180 ≡ −180 → 列 0；lat=90 极缘归最北行。"""
    grid = points_to_density([0.0, 90.0], [180.0, -180.0], "1deg")
    assert grid[90, 0] == 1
    assert grid[179, 0] == 1
    assert int(grid.sum(dtype=np.uint64)) == 2


def test_points_to_density_empty_input():
    grid = points_to_density([], [], "1deg")
    assert grid.shape == (180, 360)
    assert not grid.any()


def test_points_to_density_rejects_bad_inputs():
    with pytest.raises(ValueError, match="非有限"):
        points_to_density([np.nan], [0.0], "1deg")
    with pytest.raises(ValueError, match="非有限"):
        points_to_density([0.0], [np.inf], "1deg")
    with pytest.raises(ValueError, match="纬度出界"):
        points_to_density([90.5], [0.0], "1deg")
    with pytest.raises(ValueError, match="形状不一致"):
        points_to_density([0.0, 1.0], [0.0], "1deg")


def test_bincount_equals_loop_accumulation():
    """独立路径位级比对：bincount 落格 vs 逐点循环加法——行/列数学、
    卷绕与裁剪的正确性在两路径一致时才成立。"""
    rng = np.random.default_rng(42)
    lat = rng.uniform(-90, 90, 2000)
    lon = rng.uniform(-360, 360, 2000)      # 混合两种经度制
    grid = points_to_density(lat, lon, "6min")
    nlat, nlon = grid.shape
    res = 1.0 / 10.0
    direct = np.zeros((nlat, nlon), dtype=np.uint32)
    for la, lo in zip(lat, lon):
        lo_w = ((lo + 180.0) % 360.0) - 180.0
        r = min(int(np.floor((la + 90.0) / res)), nlat - 1)
        c = min(int(np.floor((lo_w + 180.0) / res)), nlon - 1)
        direct[r, c] += 1
    np.testing.assert_array_equal(grid, direct)
    assert int(grid.sum(dtype=np.uint64)) == 2000   # 构造性守恒


# ---------------- 读取器契约 ----------------

def test_load_wsm_missing_source(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_wsm_points(tmp_path)


def test_load_wsm_rejects_wrong_shape(tmp_path):
    (tmp_path / WSM_CSV_NAME).write_text("LAT,LON\n1.0,2.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="形状"):
        load_wsm_points(tmp_path)


def test_load_gsrm_missing_source(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_gsrm_gps(tmp_path)


_GPS_LINE = "1.0 0.0 0 0 0 0 0 AB X\n"


def test_load_gsrm_rejects_wrong_rows(tmp_path):
    (tmp_path / "GPS_ITRF08.gmt").write_text(_GPS_LINE * 5, encoding="utf-8")
    with pytest.raises(ValueError, match="行数"):
        load_gsrm_gps(tmp_path)


def test_load_gsrm_rejects_parse_failure(tmp_path):
    """缺字段行（8 token）逐行校验拒绝——keep_default_na=False 下 pandas
    会静默补 ""，isna 检测不到，故 token 数是唯一可靠契约。"""
    p = tmp_path / "GPS_ITRF08.gmt"
    p.write_text(
        _GPS_LINE * (GSRM_EXPECTED_ROWS - 1) + "1.0 0.0 0 0 0 0 0 AB\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="非 9 字段"):
        load_gsrm_gps(tmp_path)


def test_load_gsrm_poles_na_plate_preserved(tmp_path):
    """合成 poles：板块码 "NA"（北美）不被 pandas 默认 na 值吞成 NaN。"""
    for name in GSRM_POLES_NAMES:
        (tmp_path / name).write_text(
            "49.66 -78.08 0.285 NA\n0.0 10.0 0.5 EU\n", encoding="utf-8"
        )
    poles = load_gsrm_poles(tmp_path)
    assert len(poles) == 2 * len(GSRM_POLES_NAMES)
    assert sorted(poles["frame"].unique()) == ["APM", "IGS08", "NNR", "PA"]
    assert set(poles["plate"]) == {"NA", "EU"}
    assert not poles["plate"].isna().any()


def test_load_gsrm_poles_rejects_parse_failure(tmp_path):
    for name in GSRM_POLES_NAMES:
        (tmp_path / name).write_text(
            "49.66 -78.08 0.285 NA\n0.0 10.0\n", encoding="utf-8"  # 第 2 行缺字段
        )
    with pytest.raises(ValueError, match="非 4 字段"):
        load_gsrm_poles(tmp_path)


# ---------------- 构建器（合成主档，核与装配链路） ----------------

def test_builder_coarse_tier_from_home(monkeypatch):
    """粗档 = 主档 sum 聚合（factor 20）+ attrs 模板（count_semantics）。"""
    import cubebuild.points as points

    home = np.zeros((3600, 7200), dtype=np.uint32)
    home[0, 0] = 5
    home[1, 0] = 7
    monkeypatch.setattr(points, "wsm_home_density", lambda *a, **k: home)
    arr = points.build_wsm_density(_entry(), "1deg")
    vals = arr.compute().values
    assert vals.dtype == np.uint32
    assert vals[0, 0] == 12                    # 3′ 行 0/1 → 1° 行 0
    assert int(vals.sum(dtype=np.uint64)) == 12
    assert arr.name == WSM_LAYER_ID
    assert "count_semantics" in arr.attrs
    assert arr.attrs["nodata_semantics"].startswith("0 = 无观测")
    # 整型计数层 attrs 禁 missing_value/_FillValue
    assert "missing_value" not in arr.attrs and "_FillValue" not in arr.attrs


# ---------------- 真实源基线（磁盘实证的契约化） ----------------

@requires_wsm
def test_wsm_source_baseline():
    """WSM 基线：100842×40；170 空坐标行全 QUALITY=Xmi；可定位 100672；
    主档密度总和 == 可定位数（计数守恒）。"""
    df = load_wsm_points()
    assert df.shape == (WSM_EXPECTED_ROWS, 40)
    null = df[df["LAT"].isna() | df["LON"].isna()]
    assert len(null) == WSM_EXPECTED_ROWS - WSM_EXPECTED_LOCATABLE
    assert (null["QUALITY"] == "Xmi").all()
    home = wsm_home_density()
    assert home.shape == (3600, 7200)
    assert int(home.sum(dtype=np.uint64)) == WSM_EXPECTED_LOCATABLE


@requires_gsrm
def test_gsrm_source_baseline_three_numbers():
    """GSRM 基线三数：22511 行 / 18440 唯一坐标 /
    16681 唯一站码；经度 0..360 制；主档密度总和 == 22511。"""
    df = load_gsrm_gps()
    assert len(df) == GSRM_EXPECTED_ROWS
    assert df["lon"].between(0, 360).all()
    assert int(df.groupby(["lat", "lon"]).ngroups) == GSRM_UNIQUE_COORDS
    assert df["site"].nunique() == GSRM_UNIQUE_SITES
    home = gsrm_home_density()
    assert int(home.sum(dtype=np.uint64)) == GSRM_EXPECTED_ROWS


# ---------------- 侧车导出器 ----------------

def test_export_sidecars_missing_source(tmp_path):
    from cubebuild.sidecars import export_gsrm_velocities, export_wsm_points

    with pytest.raises(FileNotFoundError):
        export_wsm_points(tmp_path, src_dir=tmp_path / "nowhere")
    with pytest.raises(FileNotFoundError):
        export_gsrm_velocities(tmp_path, src_dir=tmp_path / "nowhere")


@requires_wsm
def test_export_wsm_sidecar_roundtrip(tmp_path):
    """WSM 侧车：全 100842 行 + 40 属性列 + 170 空几何 + 应力字段齐全。"""
    import geopandas as gpd

    from cubebuild.sidecars import export_wsm_points

    rec = export_wsm_points(tmp_path)
    assert rec["id"] == "wsm2025_points"
    assert rec["visibility"] == "public"
    assert rec["n_features"] == WSM_EXPECTED_ROWS
    assert rec["n_locatable"] == WSM_EXPECTED_LOCATABLE
    back = gpd.read_parquet(tmp_path / "wsm2025_points.parquet")
    assert len(back) == WSM_EXPECTED_ROWS
    assert back.crs.to_epsg() == 4326
    # 40 属性列全保留（应力方向/机制/质量分级验收项）+ geometry
    for col in ("AZI", "TYPE", "QUALITY", "REGIME", "S1AZ", "S1PL",
                "DEPTH", "SITE"):
        assert col in back.columns
    assert len(back.columns) == 41                       # 40 + geometry
    n_null_geom = int(back.geometry.isna().sum())
    assert n_null_geom == WSM_EXPECTED_ROWS - WSM_EXPECTED_LOCATABLE
    # 空几何行即空坐标行（QUALITY=Xmi）
    assert (back.loc[back.geometry.isna(), "QUALITY"] == "Xmi").all()


@requires_gsrm
def test_export_gsrm_sidecar_roundtrip(tmp_path):
    """GSRM 侧车：22511 行（lon 原值 0..360 + lon_180 + 几何）+ poles
    203 行 4 框架 + NA 板块码保留；visibility=internal。"""
    import geopandas as gpd

    from cubebuild.sidecars import export_gsrm_velocities

    rec = export_gsrm_velocities(tmp_path)
    assert rec["id"] == "gsrm_gps_velocities"
    assert rec["visibility"] == "internal"
    assert rec["n_features"] == GSRM_EXPECTED_ROWS
    assert rec["n_poles"] == 203

    back = gpd.read_parquet(
        tmp_path / "gsrm_gps_velocities" / "gps_itrf08_velocities.parquet"
    )
    assert len(back) == GSRM_EXPECTED_ROWS
    assert back.crs.to_epsg() == 4326
    # psvelo 9 列原值 + lon_180 卷绕列 + geometry（首行实测 238.327 → −121.673）
    assert list(back.columns[:9]) == [
        "lon", "lat", "ve", "vn", "se", "sn", "corr", "site", "study"
    ]
    assert back.iloc[0]["lon"] == pytest.approx(238.327, abs=1e-3)
    assert back.iloc[0]["lon_180"] == pytest.approx(-121.673, abs=1e-3)
    assert back.geometry.isna().sum() == 0

    import pandas as pd

    poles = pd.read_parquet(tmp_path / "gsrm_gps_velocities" / "poles.parquet")
    assert len(poles) == 203
    assert sorted(poles["frame"].unique()) == ["APM", "IGS08", "NNR", "PA"]
    assert (poles["plate"] == "NA").any() and not poles["plate"].isna().any()


# ---------------- e2e：两密度层 + 两侧车全链路 ----------------

_MINI = """
version: test
status: test
tiers: [1deg, 6min, 3min]
layers:
  - id: stress_kinematics__wsm_density
    source: wsm2025-stress（100842 条记录）
    native_res: point
    home_tier: 3min
    tiers: [1deg, 6min, 3min]
    dtype: uint32
    unit: count/cell
    resampling: sum
    mask: 免
    notes: 单层总密度；方向/机制/质量分级留 sidecar
  - id: stress_kinematics__gsrm_gps_density
    source: gsrm-strain（GPS_ITRF08.gmt，观测站点速度）
    native_res: point
    home_tier: 3min
    tiers: [1deg, 6min, 3min]
    dtype: uint32
    unit: count/cell
    resampling: sum
    mask: 免
    visibility: internal
    notes: ITRF08 单系计数，禁跨系
"""


@requires_wsm
@requires_gsrm
def test_point_density_e2e(tmp_path, monkeypatch):
    """uint32 密度层全链路：build → 结构验证 PASS → manifest
    （dtype/count_semantics/visibility/侧车）→ 保真 PASS（逐位/跨档/
    计数守恒）→ 侧车落格还原密度层（互查项）。"""
    import cubebuild.cli
    import geopandas as gpd
    import xarray as xr
    from cubebuild.manifest import read_manifest
    from cubebuild.points import build_point_density_fidelity_report
    from cubebuild.sidecars import export_gsrm_velocities, export_wsm_points

    monkeypatch.setattr(cubebuild.cli, "SIDECARS", {
        "wsm2025_points": export_wsm_points,
        "gsrm_gps_velocities": export_gsrm_velocities,
    })

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
    assert set(entries) == {WSM_LAYER_ID, GSRM_LAYER_ID}
    for eid, vis in ((WSM_LAYER_ID, "public"), (GSRM_LAYER_ID, "internal")):
        e = entries[eid]
        assert e["dtype"] == "uint32"
        assert e["visibility"] == vis
        assert e["resampling"] == "sum"
        assert "count_semantics" in e          # 三数/计数基准注记（验收项）
        assert "validity_mask" not in e        # 免掩膜
        assert e["coverage"] == {"1deg": 1.0, "6min": 1.0, "3min": 1.0}
    assert "100672" in entries[WSM_LAYER_ID]["count_semantics"]
    assert "22511" in entries[GSRM_LAYER_ID]["count_semantics"]
    assert "18440" in entries[GSRM_LAYER_ID]["count_semantics"]
    assert "16681" in entries[GSRM_LAYER_ID]["count_semantics"]
    assert [s["id"] for s in manifest["sidecars"]] == [
        "wsm2025_points", "gsrm_gps_velocities"
    ]
    assert manifest["sidecars"][1]["visibility"] == "internal"

    # 保真：逐位一致 + 跨档 sum 一致 + 计数守恒 + dtype/值域（+ GSRM 三数）
    for eid in (WSM_LAYER_ID, GSRM_LAYER_ID):
        report = build_point_density_fidelity_report(out, eid, ["1deg", "6min", "3min"])
        assert report["result"] == "PASS", report["checks"]
        n_hard = sum(1 for c in report["checks"] if c["status"] != "report")
        expect_hard = 3 * 3 + 2 + (1 if eid == GSRM_LAYER_ID else 0)
        assert n_hard == expect_hard

    # 互查：侧车点按坐标落像元 == 密度层 3′ 档逐位
    with xr.open_datatree(out, engine="zarr", chunks={}) as dt:
        wsm_store = dt[f"/{HOME_TIER}"].ds[WSM_LAYER_ID].values
        gsrm_store = dt[f"/{HOME_TIER}"].ds[GSRM_LAYER_ID].values
    back_w = gpd.read_parquet(sidecars_dir / "wsm2025_points.parquet")
    ok = back_w.geometry.notna()
    np.testing.assert_array_equal(
        points_to_density(
            back_w.loc[ok, "geometry"].y, back_w.loc[ok, "geometry"].x, HOME_TIER
        ),
        wsm_store,
    )
    back_g = gpd.read_parquet(
        sidecars_dir / "gsrm_gps_velocities" / "gps_itrf08_velocities.parquet"
    )
    np.testing.assert_array_equal(
        points_to_density(
            back_g["geometry"].y, back_g["geometry"].x, HOME_TIER
        ),
        gsrm_store,
    )
