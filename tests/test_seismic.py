"""测试：GLAD 读取契约与保守配准（解析场/缝合并/极区行）+ SEMUCB
解析器（vs/xi 重建/NaN 传播/自证判据）+ 3D 写入结构（组节点/depth 坐标/
掩膜伴生/.sel(depth=) 切片/与 2D 层共存）+ e2e（合成源全链路 + manifest
溯源 + 结构验证 PASS）+ 真实源基线（缝差/NaN 深度集合/CMB 层/viz 交叉验证）。
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import cubebuild.seismic as sq
from cubebuild.cli import main
from cubebuild.manifest import read_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_GLAD = (sq.GLAD_SRC_DIR / sq.GLAD_FILE).is_file()
REAL_SEMUCB_VIZ = (sq.SEMUCB_SRC_DIR / sq.SEMUCB_VIZ_VS).is_file()
requires_glad = pytest.mark.skipif(not REAL_GLAD, reason="GLAD-M35 源未集齐")
requires_semucb = pytest.mark.skipif(
    not REAL_SEMUCB_VIZ, reason="SEMUCB viz 源未集齐"
)


def _glad_entry(layer_id="seismology__gladm35_vsv"):
    from cubebuild.contract import LayerEntry

    return LayerEntry(
        id=layer_id, dims=("depth", "lat", "lon"), source="glad-m35",
        native_res="1deg", native_res_deg=1.0, home_tier="1deg",
        tiers=("1deg",), dtype="float32", unit="km/s",
        resampling="none", mask="validity",
    )


def _semucb_entry(layer_id="seismology__semucb_vs"):
    from cubebuild.contract import LayerEntry

    return LayerEntry(
        id=layer_id, dims=("depth", "lat", "lon"), source="semucb-wm1",
        native_res="1deg", native_res_deg=1.0, home_tier="1deg",
        tiers=("1deg",), dtype="float32", unit="km/s",
        resampling="none", mask="validity",
    )


def _make_synthetic_glad_nc(tmp_dir: Path, ndepth: int = 2):
    """合成 GLAD nc（契约文件名）：可分解析场 v(lat, lon) = lat + 0.01·lon
    + 100，±180 列各偏 +1/-1（缝语义：两列不同）。"""
    path = tmp_dir / sq.GLAD_FILE
    depth = np.arange(10.0, 10.0 + 10 * ndepth, 10.0)
    lat = np.arange(-90.0, 91.0, 1.0, dtype=np.float32)
    lon = np.arange(-180.0, 181.0, 1.0, dtype=np.float32)
    la = lat.astype(np.float64)[None, :, None]
    lo = lon.astype(np.float64)[None, None, :]
    v = (la + 0.01 * lo + 100.0).astype(np.float32) * np.ones((ndepth, 1, 1))
    # 五变量同场 + 缝：-180 列 +1、+180 列 -1（与真实缝语义同构）
    data_vars = {}
    for name in sq.GLAD_VARS:
        a = v.copy()
        a[:, :, 0] += 1.0
        a[:, :, -1] -= 1.0
        data_vars[name] = (("depth", "latitude", "longitude"), a)
    ds = xr.Dataset(
        data_vars,
        coords={"depth": depth, "latitude": lat, "longitude": lon},
    )
    ds.to_netcdf(path)
    return path


# ---------------- GLAD 读取契约 + 缝合并 ----------------

def test_load_glad_contract_and_seam_merge(tmp_path, monkeypatch):
    monkeypatch.setattr(sq, "GLAD_DEPTH", np.arange(10.0, 30.0, 10.0))  # 缩格 2 层
    monkeypatch.setattr(sq, "GLAD_SEAM_MAX_DIFF", 10.0)  # 合成场 ±180 基差 1.6
    nc = _make_synthetic_glad_nc(tmp_path, ndepth=2)
    src = sq.load_glad_m35(tmp_path)
    # 缝差 = |基场 ±180 差(−3.6) + 偏置(+2)| = 1.6
    assert src["seam_max_diff"] == pytest.approx(1.6, abs=1e-5)
    for v in sq.GLAD_VARS:
        a = src[v]
        assert a.shape == (2, 181, 360)                 # +180 列弃置
        # 缝合并：col0 = 两缝列均值 = 解析场值（+1 与 -1 抵消）
        assert np.allclose(a[:, :, 0], (100.0 + np.arange(-90.0, 91.0))[None, :])
        # 内部列不受影响：col1 = 解析场在 lon -179
        assert np.allclose(a[:, :, 1], (100.0 + np.arange(-90.0, 91.0) - 1.79)[None, :])


def test_load_glad_rejects_shape_drift(tmp_path):
    """源契约：维度漂移即显式失败（289×181×361 契约化）。"""
    ds = xr.Dataset(
        {"vsv": (("depth", "latitude", "longitude"),
                 np.zeros((5, 11, 12), dtype=np.float32))},
        coords={
            "depth": np.arange(5.0),
            "latitude": np.arange(-5.0, 6.0),
            "longitude": np.arange(-6.0, 6.0),
        },
    )
    ds.to_netcdf(tmp_path / sq.GLAD_FILE)
    with pytest.raises(ValueError, match="dims"):
        sq.load_glad_m35(tmp_path)


# ---------------- GLAD 保守配准（解析判据） ----------------

def test_glad_conservative_analytic(tmp_path, monkeypatch):
    """可分线性场：目标胞 = 界节点均值（经等权、纬 sin 加权）；
    极区行 = 极点节点行（半胞满权重叠）——解析恒等式逐位核对。"""
    monkeypatch.setattr(sq, "GLAD_DEPTH", np.array([10.0]))  # 缩格 1 层
    monkeypatch.setattr(sq, "GLAD_SEAM_MAX_DIFF", 10.0)  # 合成场 ±180 基差 1.6
    _make_synthetic_glad_nc(tmp_path, ndepth=1)
    src = sq.load_glad_m35(tmp_path)
    out = sq.glad_conservative_to_tier(src["vsv"])
    assert out.shape == (1, 180, 360)
    assert np.isfinite(out).all()

    e = sq.GLAD_LAT_EDGES                       # e[k], e[k+1] = 节点 k 的 Voronoi 胞边
    la_nodes = np.arange(-90.0, 91.0)
    lo_nodes = np.arange(-180.0, 180.0)          # 缝合并后 360 列
    base = 100.0

    def node(i, j):
        # j≡0 (mod 360) 为缝合并列（含 j=359 的右邻回绕）：偏置抵消、
        # ±180 线性项对消 → lat + 100
        if j % 360 == 0:
            return base + la_nodes[i]
        return base + la_nodes[i] + 0.01 * lo_nodes[j]

    for i in (0, 1, 90, 179):                    # 极区/赤道/对称抽查
        # 目标行 i 跨 [la_i, la_{i+1}]，被节点 i/i+1 胞界 e[i+1] 分割
        w0 = math.sin(math.radians(e[i + 1])) - math.sin(math.radians(la_nodes[i]))
        w1 = math.sin(math.radians(la_nodes[i + 1])) - math.sin(math.radians(e[i + 1]))
        for j in (0, 179, 359):
            n00 = node(i, j)
            n01 = node(i, j + 1)
            n10 = node(i + 1, j)
            n11 = node(i + 1, j + 1)
            # 缝列 j=0 的节点值 = 两缝列均值（±1 抵消 → 解析场值）
            expect = (w0 * (n00 + n01) + w1 * (n10 + n11)) / (2 * (w0 + w1))
            assert out[0, i, j] == pytest.approx(expect, abs=1e-4), (i, j)

    # 极区行 0（目标胞 [-90,-89]）仅与节点 -90/-89 交叠：
    # 权重 = 半胞 [-90,-89.5] 与 [-89.5,-89] 的 sin 差（数值同上 w0/w1）
    w0 = math.sin(math.radians(e[1])) - math.sin(math.radians(e[0]))
    w1 = math.sin(math.radians(-89.0)) - math.sin(math.radians(e[1]))
    r0 = 0.5 * (node(0, 0) + node(0, 1))
    r1 = 0.5 * (node(1, 0) + node(1, 1))
    expect_polar = (w0 * r0 + w1 * r1) / (w0 + w1)
    assert out[0, 0, 0] == pytest.approx(expect_polar, abs=1e-5)


def test_glad_constant_field_invariant(tmp_path, monkeypatch):
    """常数场保守平均不变（含缝合并后）——守恒的构造性检验。"""
    monkeypatch.setattr(sq, "GLAD_DEPTH", np.array([10.0]))  # 缩格 1 层
    depth = np.array([10.0])
    lat = np.arange(-90.0, 91.0, 1.0, dtype=np.float32)
    lon = np.arange(-180.0, 181.0, 1.0, dtype=np.float32)
    ds = xr.Dataset(
        {v: (("depth", "latitude", "longitude"),
             np.full((1, 181, 361), 4.5, dtype=np.float32))
         for v in sq.GLAD_VARS},
        coords={"depth": depth, "latitude": lat, "longitude": lon},
    )
    ds.to_netcdf(tmp_path / sq.GLAD_FILE)
    src = sq.load_glad_m35(tmp_path)
    out = sq.glad_conservative_to_tier(src["vsv"])
    assert np.allclose(out, 4.5, atol=1e-6)


# ---------------- SEMUCB 解析器 ----------------

def _fake_eval_df(lat, lon, radii, vsv_val=4.5, vsh_val=4.7, nan_mask=None):
    import pandas as pd

    rows = []
    for r in radii:
        for la in lat:
            for lo in lon:
                rows.append((r, lo, la, 1.0, 1.0, vsv_val, vsh_val))
    df = pd.DataFrame(rows, columns=["radius", "lon", "lat", "dvs_pct", "dxi_pct", "vsv", "vsh"])
    if nan_mask is not None:
        df.loc[nan_mask, ["dvs_pct", "dxi_pct", "vsv", "vsh"]] = np.nan
    return df


def test_eval_output_vs_xi_reconstruction():
    lat = np.array([-89.5, 0.5, 89.5])
    lon = np.array([-179.5, 0.5, 179.5])
    depth = np.array([60.0, 2891.0])
    radii = np.array([6311.0, 3480.5])
    df = _fake_eval_df(lat, lon, radii)
    out = sq._eval_output_to_fields(df, depth, radii, lat, lon)
    assert out["vs"].shape == (2, 3, 3)
    assert out["xi"].shape == (2, 3, 3)
    # Voigt 平均与 ξ = (Vsh/Vsv)² 解析核对
    assert np.allclose(out["vs"], math.sqrt((2 * 4.5**2 + 4.7**2) / 3), rtol=1e-6)
    assert np.allclose(out["xi"], (4.7 / 4.5) ** 2, rtol=1e-6)
    assert int(out["nan_counts"].sum()) == 0


def test_eval_output_nan_propagation_and_counts():
    """地壳 nan 行：vs/xi 双双 NaN 并计入 nan_counts（行序自证覆盖）。"""
    lat = np.array([-89.5, 0.5, 89.5])
    lon = np.array([-179.5, 0.5, 179.5])
    depth = np.array([30.0, 100.0])
    radii = np.array([6341.0, 6271.0])
    df = _fake_eval_df(lat, lon, radii)
    # depth=30 块（前 9 行）全 NaN
    df.loc[:8, ["dvs_pct", "dxi_pct", "vsv", "vsh"]] = np.nan
    out = sq._eval_output_to_fields(df, depth, radii, lat, lon)
    assert bool(np.isnan(out["vs"][0]).all())
    assert bool(np.isnan(out["xi"][0]).all())
    assert np.isfinite(out["vs"][1]).all()
    assert out["nan_counts"].tolist() == [9, 0]


def test_eval_output_rejects_radius_order_drift():
    lat = np.array([0.5]); lon = np.array([0.5])
    depth = np.array([10.0]); radii = np.array([6361.0])
    df = _fake_eval_df(lat, lon, [6362.0])          # 半径漂移
    with pytest.raises(ValueError, match="半径序"):
        sq._eval_output_to_fields(df, depth, radii, lat, lon)


# ---------------- 3D 写入结构（e2e 合成源） ----------------

_E2E_CONTRACT = """
version: test
status: test
tiers: [1deg]
layers:
  - id: derived__pixel_area
    source: 纬度解析公式
    native_res: exact
    home_tier: 逐档
    tiers: [1deg]
    dtype: float32
    unit: km2
    resampling: exact_formula（逐档精确计算，不重采样）
    mask: 免
  - id: seismology__gladm35_vsv
    source: glad-m35
    native_res: 1deg
    dims: [depth, lat, lon]
    home_tier: 1deg
    tiers: [1deg]
    dtype: float32
    unit: km/s
    resampling: none（原生 1° 即全集档）
    mask: validity
  - id: seismology__gladm35_eta
    source: glad-m35
    native_res: 1deg
    dims: [depth, lat, lon]
    home_tier: 1deg
    tiers: [1deg]
    dtype: float32
    unit: dimensionless
    resampling: none
    mask: validity
  - id: seismology__semucb_vs
    source: semucb-wm1
    native_res: 1deg
    dims: [depth, lat, lon]
    home_tier: 1deg
    tiers: [1deg]
    dtype: float32
    unit: km/s
    resampling: none
    mask: validity
  - id: seismology__semucb_xi
    source: semucb-wm1
    native_res: 1deg
    dims: [depth, lat, lon]
    home_tier: 1deg
    tiers: [1deg]
    dtype: float32
    unit: dimensionless（径向各向异性 ξ）
    resampling: none
    mask: validity
"""


@pytest.fixture()
def e2e_mapping(tmp_path):
    p = tmp_path / "e2e-mapping.yaml"
    p.write_text(_E2E_CONTRACT, encoding="utf-8")
    return p


@pytest.fixture()
def synthetic_sources(tmp_path, monkeypatch):
    """合成 GLAD 源（1 深度层解析场）+ 合成 SEMUCB 评估（3 深度层，
    depth 30 全 NaN / 40 部分 NaN / 2891 有效——真实 NaN 结构同构）。"""
    monkeypatch.setattr(sq, "GLAD_DEPTH", np.array([10.0]))  # 缩格 1 层
    monkeypatch.setattr(sq, "GLAD_SEAM_MAX_DIFF", 10.0)  # 合成场 ±180 基差 1.6
    _make_synthetic_glad_nc(tmp_path, ndepth=1)

    depth = np.array([30.0, 40.0, 2891.0])
    vs = np.full((3, 180, 360), 4.5, dtype=np.float32)
    vs[0] = np.nan                                    # depth 30 全 NaN
    vs[1, :20] = np.nan                               # depth 40 部分 NaN
    xi = np.where(np.isnan(vs), np.nan, 1.05).astype(np.float32)
    fake_eval = {
        "vs": vs, "xi": xi, "depth": depth,
        "nan_counts": np.isnan(vs).sum(axis=(1, 2)).astype(np.int64),
    }
    real_load = sq.load_glad_m35
    monkeypatch.setattr(
        sq, "load_glad_m35", lambda src_dir=None: real_load(tmp_path)
    )
    monkeypatch.setattr(sq, "evaluate_semucb", lambda: fake_eval)


@pytest.fixture(autouse=True)
def _clear_seismic_caches():
    """模块级缓存跨测试隔离（合成源/真实源切换不脏读）。"""
    caches = (sq._GLAD_CACHE, sq._GLAD_TIER_CACHE, sq._SEMUCB_CACHE)
    for c in caches:
        c.clear()
    yield
    for c in caches:
        c.clear()


@pytest.fixture(autouse=True)
def _no_sidecars(monkeypatch):
    import cubebuild.cli

    monkeypatch.setattr(cubebuild.cli, "SIDECARS", {})


def test_seismic_3d_e2e(tmp_path, e2e_mapping, synthetic_sources):
    out = tmp_path / "cube.zarr"
    reports = tmp_path / "reports"
    rc = main(["build", "--mapping", str(e2e_mapping), "--out", str(out),
               "--reports", str(reports)])
    assert rc == 0

    dt = xr.open_datatree(out, engine="zarr")
    # 3D 组子节点与 2D 档位节点共存
    assert "gladm35" in dt["/1deg"].children
    assert "semucb" in dt["/1deg"].children
    glad = dt["/1deg/gladm35"].ds
    semu = dt["/1deg/semucb"].ds

    # 验收：Datatree 读入后可按 depth 选择读取（一次 .sel）
    slice1 = dt["/1deg/gladm35"].sel(depth=10.0)
    assert slice1.ds.sizes == {"lat": 180, "lon": 360}
    slice2 = dt["/1deg/semucb"].sel(depth=2891.0)
    assert slice2.ds.sizes == {"lat": 180, "lon": 360}

    # 深度层数与源一致（1 合成层 / 3 合成层）；维序 [depth, lat, lon]
    assert glad["seismology__gladm35_vsv"].dims == ("depth", "lat", "lon")
    assert glad.sizes["depth"] == 1
    assert semu.sizes["depth"] == 3
    # depth 坐标登记（单位 km、正向下）
    dep = semu["seismology__semucb_vs"].coords["depth"]
    assert dep.attrs["units"] == "km" and dep.attrs["positive"] == "down"

    # 有效性掩膜按层伴生；SEMUCB 掩膜携带地壳 NaN 结构
    for lid in ("seismology__gladm35_vsv", "seismology__gladm35_eta"):
        assert f"{lid}__validity" in glad.data_vars
        assert bool((glad[f"{lid}__validity"] == 1).all())
    vm = semu["seismology__semucb_vs__validity"].values
    assert bool((vm[0] == 0).all())                    # depth 30 全缺测
    assert vm[1].sum() == 180 * 360 - 20 * 360         # depth 40 前 20 行缺测
    assert bool((vm[2] == 1).all())                    # depth 2891 全有效
    # 掩膜与数据两通道一致
    vv = semu["seismology__semucb_vs"].values
    assert bool(np.array_equal(vm, np.isfinite(vv).astype(np.uint8)))

    # manifest：3D 层条目 + depth_levels/datatree_node/overlap_note 溯源
    manifest = read_manifest(out)
    entries = {m["id"]: m for m in manifest["layers"]}
    assert entries["seismology__gladm35_vsv"]["dims"] == ["depth", "lat", "lon"]
    assert entries["seismology__gladm35_vsv"]["depth_levels"] == 1
    assert entries["seismology__semucb_vs"]["depth_levels"] == 3
    assert entries["seismology__semucb_vs"]["datatree_node"] == "/1deg/semucb"
    assert "有意保留" in entries["seismology__semucb_vs"]["overlap_note"]
    assert entries["seismology__semucb_vs"]["validity_mask"] == (
        "seismology__semucb_vs__validity"
    )
    # 覆盖率登记（SEMUCB = 1 − NaN 占比；NaN = 64800 + 20·360）
    cov = entries["seismology__semucb_vs"]["coverage"]["1deg"]
    assert cov == pytest.approx(1.0 - (64800 + 20 * 360) / (3 * 64800))

    # 结构验证 PASS（含 depth 维检查/组内配准/掩膜两通道）
    report = json.loads((reports / "structure-validation.json").read_text())
    assert report["result"] == "PASS", [
        c for c in report["checks"] if c["status"] == "FAIL"
    ]


def test_seismic_3d_determinism(tmp_path, e2e_mapping, synthetic_sources):
    outs = []
    for name in ("run1.zarr", "run2.zarr"):
        rc = main(["build", "--mapping", str(e2e_mapping), "--out", str(tmp_path / name),
                   "--reports", str(tmp_path / "reports")])
        assert rc == 0
        outs.append(tmp_path / name)
    from cubebuild.checksum import checksums_equal, store_checksums

    assert checksums_equal(store_checksums(outs[0]), store_checksums(outs[1]))


# ---------------- 真实源基线 ----------------

@requires_glad
def test_real_glad_baseline():
    src = sq.load_glad_m35()
    assert src["vsv"].shape == (289, 181, 360)
    assert np.array_equal(src["depth"], sq.GLAD_DEPTH)
    assert src["seam_max_diff"] == pytest.approx(0.1198, abs=1e-3)   # 实证值
    for v in sq.GLAD_VARS:
        assert np.isfinite(src[v]).all()


@requires_semucb
def test_real_semucb_depths_contract():
    dep = sq.semucb_depths()
    assert len(dep) == 74
    assert tuple(dep) == sq.SEMUCB_DEPTHS
    assert dep[0] == 30.0 and dep[-1] == 2891.0


@requires_glad
def test_real_glad_conservative_spot():
    """真实源核映射的解析抽检（赤道 4 节点均值）。"""
    src = sq.load_glad_m35()
    out = sq.glad_conservative_to_tier(src["vsv"][:1])
    a = src["vsv"][0]
    e = sq.GLAD_LAT_EDGES
    w0 = math.sin(math.radians(e[91])) - math.sin(math.radians(e[90]))
    w1 = math.sin(math.radians(e[92])) - math.sin(math.radians(e[91]))
    n00, n01 = a[90, 179], a[90, 180]
    n10, n11 = a[91, 179], a[91, 180]
    expect = (w0 * (n00 + n01) + w1 * (n10 + n11)) / (2 * (w0 + w1))
    assert out[0, 90, 179] == pytest.approx(expect, abs=1e-5)
