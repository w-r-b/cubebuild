"""测试：Ma 标签解析/扫描排序 + 双产品读取契约（北起降序翻转/缝
合并/±180 逐位同断言）+ Voronoi 边构造 + 保守配准解析判据（0.2° 合成网格
的嵌套恒等 + 1° GLAD 同款相邻节点均值公式）+ 4D 写入结构 e2e（组节点/
time 坐标/按 Ma 索引/掩膜伴生/manifest 溯源/结构验证 PASS/保真 PASS/
双跑确定性）+ 真实源基线（109 片标签集/读取契约/核映射抽检）。
增补：1° 两片标签按官方整数惯例对齐（385.2→385、390.5→390），
两档标签集 109/109 逐值相等（合成对齐单测 + 真实源验收断言）。
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import cubebuild.paleo as pq
from cubebuild.cli import main
from cubebuild.etopo import SourceNotReady
from cubebuild.manifest import read_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_SIXMIN = (pq.SIXMIN_DIR / "Map01_PALEOMAP_6min_Holocene_0Ma.nc").is_file()
REAL_ONEDEG = (pq.ONEDEG_DIR / "Map01_PALEOMAP_1deg_Holocene_0Ma.nc").is_file()
requires_paleo = pytest.mark.skipif(
    not (REAL_SIXMIN and REAL_ONEDEG), reason="PaleoDEM 源未集齐"
)


def _entry(layer_id="paleo__paleodem_elevation"):
    from cubebuild.contract import LayerEntry

    return LayerEntry(
        id=layer_id, dims=("time", "lat", "lon"), source="paleodem-scotese",
        native_res="6min", native_res_deg=1.0 / 10.0, home_tier="6min",
        tiers=("1deg", "6min"), dtype="float32", unit="m",
        resampling="none", mask="validity（古地理随时间变化，逐时间片掩膜）",
    )


# ---------------- Ma 标签与扫描 ----------------

def test_parse_ma_label():
    assert pq.parse_ma_label("Map01_PALEOMAP_6min_Holocene_0Ma.nc") == 0.0
    assert pq.parse_ma_label("Map43.5_PALEOMAP_6min_Late_Triassic_205Ma.nc") == 205.0
    assert pq.parse_ma_label("Map67.5_PALEOMAP_1deg_Late_Devonian_385.2Ma.nc") == 385.2
    assert pq.parse_ma_label("Map24_PALEOMAP_6min_Early Cretaceous_105Ma.nc") == 105.0
    with pytest.raises(ValueError, match="Ma 标签"):
        pq.parse_ma_label("Map01_PALEOMAP_6min_Holocene.nc")


def test_scan_slices_sorts_by_ma_not_filename(tmp_path):
    """文件名字典序 ≠ Ma 序（Map43.5_205Ma 先于 Map43_200Ma）——按解析值排序。"""
    for name in (
        "Map43.5_PALEOMAP_6min_Late_Triassic_205Ma.nc",
        "Map43_PALEOMAP_6min_Late_Triassic_200Ma.nc",
        "Map01_PALEOMAP_6min_Holocene_0Ma.nc",
    ):
        (tmp_path / name).write_bytes(b"")
    out = pq.scan_slices(tmp_path, expected=3)
    assert [ma for ma, _ in out] == [0.0, 200.0, 205.0]


def test_scan_slices_count_and_duplicate_contracts(tmp_path):
    (tmp_path / "a_0Ma.nc").write_bytes(b"")
    with pytest.raises(SourceNotReady):
        pq.scan_slices(tmp_path, expected=109)          # 片数不符 → 跳过语义
    (tmp_path / "b_0Ma.nc").write_bytes(b"")            # 标签重复 → 源漂移硬失败
    with pytest.raises(ValueError, match="重复"):
        pq.scan_slices(tmp_path, expected=2)
    with pytest.raises(SourceNotReady):
        pq.scan_slices(tmp_path / "nonexistent")         # 目录缺失 → 跳过语义


def test_scan_slices_ma_range_contract(tmp_path):
    """「0–540Ma 全收」：标签范围漂移 → 构建期显式失败（code-review 修正）。"""
    for name in ("a_0Ma.nc", "b_600Ma.nc"):
        (tmp_path / name).write_bytes(b"")
    with pytest.raises(ValueError, match="超出契约"):
        pq.scan_slices(tmp_path, expected=2)


# ---------------- 1° 标签对齐 ----------------


def test_scan_slices_1deg_alignment(tmp_path):
    """tier="1deg"：385.2/390.5 按官方整数惯例舍入（385/390），排序按对齐后
    标签；路径随原文件（数据数组零改动，time 坐标是唯一改动面）；6min/裸
    扫描零改动。"""
    for name in ("a_380Ma.nc", "b_385.2Ma.nc", "c_390.5Ma.nc", "d_395Ma.nc"):
        (tmp_path / name).write_bytes(b"")
    out = pq.scan_slices(tmp_path, expected=4, tier="1deg")
    assert [ma for ma, _ in out] == [380.0, 385.0, 390.0, 395.0]
    assert out[1][1].name == "b_385.2Ma.nc"
    assert out[2][1].name == "c_390.5Ma.nc"
    # 6min 档与裸扫描：文件名原始标签
    raw = [ma for ma, _ in pq.scan_slices(tmp_path, expected=4)]
    assert raw == [380.0, 385.2, 390.5, 395.0]
    assert [ma for ma, _ in pq.scan_slices(tmp_path, expected=4, tier="6min")] == raw


def test_scan_slices_alignment_duplicate_guard(tmp_path):
    """对齐引入重复（源并存 385 与 385.2）→ 源漂移硬失败（查重对对齐后
    标签生效，防对齐静默合并两片）。"""
    for name in ("a_385Ma.nc", "b_385.2Ma.nc"):
        (tmp_path / name).write_bytes(b"")
    with pytest.raises(ValueError, match="重复"):
        pq.scan_slices(tmp_path, expected=2, tier="1deg")


# ---------------- 合成源（几何契约化，可缩格）----------------

# 6min 合成几何：0.2° 步长（901×1801）——Voronoi 边落在 0.1 的整数倍上，
# 目标 6min 档胞完全嵌套于单一源胞 → 恒等映射（解析可验）；缝语义与真实同构。
SYN_NLAT, SYN_NLON = 901, 1801
SYN_RES = 0.2


def _syn_field_6min(nlat=SYN_NLAT, nlon=SYN_NLON, res=SYN_RES, base=100.0):
    """可分解析场 v(lat, lon) = lat + 0.01·lon + base，±180 列各偏 +1/-1。"""
    lat = 90.0 - res * np.arange(nlat, dtype=np.float64)      # 北起降序（真实同构）
    lon = -180.0 + res * np.arange(nlon, dtype=np.float64)
    v = (lat[:, None] + 0.01 * lon[None, :] + base).astype(np.float32)
    raw = v.copy()
    raw[:, 0] += 1.0
    raw[:, -1] -= 1.0
    return lat, lon, v, raw


def _make_synthetic_6min(dir_path: Path, ma: float):
    lat, lon, _v, raw = _syn_field_6min()
    ds = xr.Dataset(
        {"z": (("latitude", "longitude"), raw)},
        coords={"latitude": lat, "longitude": lon},
    )
    path = dir_path / f"Map_test_PALEOMAP_6min_{ma:g}Ma.nc"
    ds.to_netcdf(path)
    return path


def _make_synthetic_1deg(dir_path: Path, ma: float, base=100.0):
    """1° 合成（真实几何 181×361）：南起升序，±180 逐位相同（官方语义）。"""
    lat = -90.0 + np.arange(181, dtype=np.float64)
    lon = -180.0 + np.arange(361, dtype=np.float64)
    v = (lat[:, None] + 0.01 * lon[None, :] + base).astype(np.float32)
    v[:, -1] = v[:, 0]                                     # +180 ≡ -180 采样
    ds = xr.Dataset(
        {"z": (("lat", "lon"), v)},
        coords={"lat": lat, "lon": lon},
    )
    path = dir_path / f"Map_test_PALEOMAP_1deg_{ma:g}Ma.nc"
    ds.to_netcdf(path)
    return path


@pytest.fixture()
def syn6_geometry(monkeypatch):
    """合成 6min 几何契约（0.2° 缩格）注入模块常量。"""
    monkeypatch.setattr(pq, "SIXMIN_NLAT", SYN_NLAT)
    monkeypatch.setattr(pq, "SIXMIN_NLON", SYN_NLON)


# ---------------- 读取契约 + 缝合并 ----------------

def test_read_slice_6min_contract_and_seam_merge(tmp_path, syn6_geometry):
    path = _make_synthetic_6min(tmp_path, 0.0)
    a = pq.read_slice_6min(path)
    assert a.shape == (SYN_NLAT, SYN_NLON - 1)
    assert a.dtype == np.float32
    lat_nodes = -90.0 + SYN_RES * np.arange(SYN_NLAT)      # 南起节点
    lon_nodes = -180.0 + SYN_RES * np.arange(SYN_NLON - 1)
    # 行翻转：行 0 = 节点 -90；缝合并：列 0 = 两缝列均值 = 解析场值
    # （±180 线性项对消 + 缝偏置 ±1 对消 → lat + base，GLAD 同款）
    assert np.allclose(a[0, 0], -90.0 + 100.0, atol=1e-4)
    assert np.allclose(a[:, 0], lat_nodes + 100.0, atol=1e-4)
    assert np.allclose(a[:, 1], lat_nodes + 0.01 * lon_nodes[1] + 100.0, atol=1e-4)
    # 内部列不受缝影响（末列 = 节点 179.8）
    assert np.allclose(a[:, -1], lat_nodes + 0.01 * lon_nodes[-1] + 100.0, atol=1e-4)


def test_read_slice_6min_rejects_drift(tmp_path, syn6_geometry):
    """契约漂移显式失败：维度/纬向反号/非有限值。"""
    # 维度漂移
    ds = xr.Dataset(
        {"z": (("latitude", "longitude"), np.zeros((11, 12), dtype=np.float32))},
        coords={"latitude": np.arange(10.0, -1.0, -1.0), "longitude": np.arange(-6.0, 6.0)},
    )
    ds.to_netcdf(tmp_path / "bad_dims_0Ma.nc")
    with pytest.raises(ValueError, match="dims"):
        pq.read_slice_6min(tmp_path / "bad_dims_0Ma.nc")
    # 纬向反号（南起升序 = 与 6min 产品北起契约相反）
    lat, lon, _v, raw = _syn_field_6min()
    ds = xr.Dataset(
        {"z": (("latitude", "longitude"), raw[::-1])},
        coords={"latitude": lat[::-1], "longitude": lon},
    )
    ds.to_netcdf(tmp_path / "bad_lat_0Ma.nc")
    with pytest.raises(ValueError, match="纬/经度坐标"):
        pq.read_slice_6min(tmp_path / "bad_lat_0Ma.nc")
    # 非有限值（实证 109 片 0 缺测）
    lat, lon, _v, raw = _syn_field_6min()
    raw[5, 5] = np.nan
    ds = xr.Dataset(
        {"z": (("latitude", "longitude"), raw)},
        coords={"latitude": lat, "longitude": lon},
    )
    ds.to_netcdf(tmp_path / "bad_nan_0Ma.nc")
    with pytest.raises(ValueError, match="非有限值"):
        pq.read_slice_6min(tmp_path / "bad_nan_0Ma.nc")
    # 值域漂移（实证 [-11000, 10500] m，code-review 修正后契约化）
    lat, lon, _v, raw = _syn_field_6min()
    raw[5, 5] = 9.9e4
    ds = xr.Dataset(
        {"z": (("latitude", "longitude"), raw)},
        coords={"latitude": lat, "longitude": lon},
    )
    ds.to_netcdf(tmp_path / "bad_range_0Ma.nc")
    with pytest.raises(ValueError, match="值域"):
        pq.read_slice_6min(tmp_path / "bad_range_0Ma.nc")


def test_read_slice_1deg_contract(tmp_path):
    path = _make_synthetic_1deg(tmp_path, 0.0)
    a = pq.read_slice_1deg(path)
    assert a.shape == (181, 360)
    assert a.dtype == np.float32
    # 南起升序保持 + 弃 +180 后列 0 = -180 采样、末列 = 179 采样
    lat = -90.0 + np.arange(181, dtype=np.float64)
    assert np.allclose(a[0, 0], -90.0 + 0.01 * -180.0 + 100.0)
    assert np.allclose(a[:, -1], lat + 0.01 * 179.0 + 100.0)
    # ±180 不逐位相同 → 显式失败（实证缝差恰 0，源漂移）
    lat, lon, _v, raw = _syn_field_6min(nlat=181, nlon=361, res=1.0)
    ds = xr.Dataset(
        {"z": (("lat", "lon"), raw)},
        coords={"lat": -90.0 + np.arange(181.0), "lon": -180.0 + np.arange(361.0)},
    )
    ds.to_netcdf(tmp_path / "bad_seam_0Ma.nc")
    with pytest.raises(ValueError, match="±180"):
        pq.read_slice_1deg(tmp_path / "bad_seam_0Ma.nc")


# ---------------- Voronoi 边 + 保守配准解析判据 ----------------

def test_voronoi_lat_edges_real_geometry():
    """真实几何：6min 1802 边（中点 ±90 钳制）、1° 182 边（GLAD 同款）。"""
    e6 = pq.voronoi_lat_edges(pq.lat_nodes_asc("6min"))
    assert len(e6) == 1802
    assert e6[0] == -90.0 and e6[-1] == 90.0
    assert e6[1] == pytest.approx(-89.95) and e6[-2] == pytest.approx(89.95)
    e1 = pq.voronoi_lat_edges(pq.lat_nodes_asc("1deg"))
    assert len(e1) == 182
    assert e1[1] == pytest.approx(-89.5) and e1[-2] == pytest.approx(89.5)


def test_paleo_tier_field_6min_nested_identity(tmp_path, syn6_geometry):
    """0.2° 合成源：Voronoi 边与目标档边重合 → 目标胞完全嵌套单一源胞，
    目标值 = 源节点值（行/列各 2 目标胞 ↔ 1 源胞），解析恒等式核对。"""
    path = _make_synthetic_6min(tmp_path, 0.0)
    merged = pq.read_slice_6min(path)                       # (901, 1800)
    out = pq.paleo_tier_field("6min", path)
    assert out.shape == (1800, 3600)
    assert np.isfinite(out).all()
    # 目标 (i, j) = 源节点 ((i+1)//2, (j+1)//2)（极区/赤道/边界抽查；
    # j=3599 跨日期线回绕到源列 0）
    for i in (0, 1, 2, 3, 899, 900, 1798, 1799):
        for j in (0, 1, 2, 3598, 3599):
            expect = merged[(i + 1) // 2, ((j + 1) // 2) % merged.shape[1]]
            assert out[i, j] == pytest.approx(expect, abs=1e-4), (i, j)


def test_paleo_tier_field_1deg_analytic(tmp_path):
    """1° 合成源（GLAD 同款几何）：目标胞 = 相邻节点经向等权、纬向 sin 加权
    平均；极区行半胞满权重叠——解析公式逐位核对（含 j=359 日期线回绕）。"""
    path = _make_synthetic_1deg(tmp_path, 0.0)
    a = pq.read_slice_1deg(path)                           # (181, 360) 节点
    out = pq.paleo_tier_field("1deg", path)
    assert out.shape == (180, 360)

    la = -90.0 + np.arange(181, dtype=np.float64)
    lo = -180.0 + np.arange(360, dtype=np.float64)

    def node(i, j):
        return la[i] + 0.01 * lo[j] + 100.0

    for i in (0, 1, 90, 179):
        e_mid = 0.5 * (la[i] + la[i + 1])                  # 节点 i/i+1 的 Voronoi 界
        w0 = math.sin(math.radians(e_mid)) - math.sin(math.radians(la[i]))
        w1 = math.sin(math.radians(la[i + 1])) - math.sin(math.radians(e_mid))
        for j in (0, 179, 359):
            j1 = (j + 1) % 360                             # 日期线回绕
            expect = (
                w0 * (node(i, j) + node(i, j1))
                + w1 * (node(i + 1, j) + node(i + 1, j1))
            ) / (2.0 * (w0 + w1))
            assert out[i, j] == pytest.approx(expect, abs=1e-4), (i, j)


def test_paleo_constant_field_invariant(tmp_path, syn6_geometry):
    """常数场保守平均不变（含缝合并）——守恒的构造性检验。"""
    lat = 90.0 - SYN_RES * np.arange(SYN_NLAT)
    lon = -180.0 + SYN_RES * np.arange(SYN_NLON)
    ds = xr.Dataset(
        {"z": (("latitude", "longitude"), np.full((SYN_NLAT, SYN_NLON), 4.5, np.float32))},
        coords={"latitude": lat, "longitude": lon},
    )
    ds.to_netcdf(tmp_path / "const_0Ma.nc")
    out = pq.paleo_tier_field("6min", tmp_path / "const_0Ma.nc")
    assert np.allclose(out, 4.5, atol=1e-6)


# ---------------- 4D 写入结构 e2e（合成源全链路）----------------

_E2E_CONTRACT = """
version: test
status: test
tiers: [1deg, 6min]
layers:
  - id: paleo__paleodem_elevation
    source: paleodem-scotese（paleodem-6min-nc/*.nc；paleodem-1deg-nc/*.nc）
    native_res: 6min
    dims: [time, lat, lon]
    home_tier: 6min
    tiers: [1deg, 6min]
    dtype: float32
    unit: m
    resampling: none（1deg 档直接用官方 1° nc，免降采样）
    mask: validity（古地理随时间变化，逐时间片掩膜）
"""


@pytest.fixture()
def e2e_mapping(tmp_path):
    p = tmp_path / "e2e-mapping.yaml"
    p.write_text(_E2E_CONTRACT, encoding="utf-8")
    return p


@pytest.fixture()
def synthetic_sources(tmp_path, monkeypatch):
    """合成双产品源（6min 0.2° 缩格 × 2 片 + 1° 真实几何 × 2 片）。"""
    monkeypatch.setattr(pq, "SIXMIN_NLAT", SYN_NLAT)
    monkeypatch.setattr(pq, "SIXMIN_NLON", SYN_NLON)
    monkeypatch.setattr(pq, "N_SLICES", 2)
    d6, d1 = tmp_path / "paleodem-6min-nc", tmp_path / "paleodem-1deg-nc"
    d6.mkdir(); d1.mkdir()
    for ma in (0.0, 5.0):
        _make_synthetic_6min(d6, ma)
        _make_synthetic_1deg(d1, ma)
    monkeypatch.setattr(pq, "SIXMIN_DIR", d6)
    monkeypatch.setattr(pq, "ONEDEG_DIR", d1)


@pytest.fixture(autouse=True)
def _no_sidecars(monkeypatch):
    import cubebuild.cli

    monkeypatch.setattr(cubebuild.cli, "SIDECARS", {})


def test_paleo_4d_e2e(tmp_path, e2e_mapping, synthetic_sources):
    out = tmp_path / "cube.zarr"
    reports = tmp_path / "reports"
    rc = main(["build", "--mapping", str(e2e_mapping), "--out", str(out),
               "--reports", str(reports)])
    assert rc == 0

    dt = xr.open_datatree(out, engine="zarr")
    # 4D 组子节点与（空）档位节点共存
    assert "paleodem" in dt["/1deg"].children
    assert "paleodem" in dt["/6min"].children
    n6 = dt["/6min/paleodem"].ds
    n1 = dt["/1deg/paleodem"].ds

    # 验收：Datatree 读入后可按 Ma 索引任意时间片（一次 .sel）
    for node in (n6, n1):
        assert node["paleo__paleodem_elevation"].dims == ("time", "lat", "lon")
        assert node.sizes["time"] == 2
        t = node["paleo__paleodem_elevation"].coords["time"]
        assert t.attrs["units"] == "Ma"
        assert np.array_equal(t.values, [0.0, 5.0])
    slice5 = dt["/6min/paleodem"].sel(time=5.0)
    assert slice5.ds.sizes == {"lat": 1800, "lon": 3600}

    # 数值 = 解析判据（6min 嵌套恒等 / 1° 相邻节点均值，slice 0）
    merged = pq.read_slice_6min(pq.SIXMIN_DIR / "Map_test_PALEOMAP_6min_0Ma.nc")
    v6 = n6["paleo__paleodem_elevation"].values[0]
    for i in (0, 900, 1799):
        for j in (0, 1799, 3599):
            assert v6[i, j] == pytest.approx(
                merged[(i + 1) // 2, ((j + 1) // 2) % merged.shape[1]], abs=1e-4
            )
    a1 = pq.read_slice_1deg(pq.ONEDEG_DIR / "Map_test_PALEOMAP_1deg_0Ma.nc")
    v1 = n1["paleo__paleodem_elevation"].values[0]
    la = -90.0 + np.arange(181.0); lo = -180.0 + np.arange(360.0)
    i, j = 90, 179
    e_mid = 0.5 * (la[i] + la[i + 1])
    w0 = math.sin(math.radians(e_mid)) - math.sin(math.radians(la[i]))
    w1 = math.sin(math.radians(la[i + 1])) - math.sin(math.radians(e_mid))
    expect = (w0 * (a1[i, j] + a1[i, j + 1]) + w1 * (a1[i + 1, j] + a1[i + 1, j + 1])) / (2 * (w0 + w1))
    assert v1[i, j] == pytest.approx(expect, abs=1e-4)

    # 有效性掩膜按片伴生（[time, lat, lon]，合成源全覆盖 → 全 1）
    m6 = n6["paleo__paleodem_elevation__validity"].values
    m1 = n1["paleo__paleodem_elevation__validity"].values
    assert m6.shape == (2, 1800, 3600) and bool((m6 == 1).all())
    assert m1.shape == (2, 180, 360) and bool((m1 == 1).all())

    # manifest：4D 层条目 + time_levels/time_semantics/datatree_node 溯源
    manifest = read_manifest(out)
    entries = {m["id"]: m for m in manifest["layers"]}
    me = entries["paleo__paleodem_elevation"]
    assert me["dims"] == ["time", "lat", "lon"]
    assert me["time_levels"] == 2
    assert me["datatree_node"] == "paleodem"           # 相对组名（两档各自解析）
    assert "385.2/390.5" in me["time_semantics"]       # 舍入注记
    assert "109/109" in me["time_semantics"]
    assert me["validity_mask"] == "paleo__paleodem_elevation__validity"
    assert me["coverage"] == {"1deg": 1.0, "6min": 1.0}
    assert me["license_note"].startswith("CC BY 4.0")

    # 结构验证 PASS（含 time 维检查/组内配准/掩膜两通道一致）
    report = json.loads((reports / "structure-validation.json").read_text())
    assert report["result"] == "PASS", [
        c for c in report["checks"] if c["status"] == "FAIL"
    ]

    # 保真验证 PASS（位级重算一致/积分守恒/time 坐标/跨档均值曲线）
    rc = main(["fidelity", "--store", str(out), "--layer", "paleo__paleodem_elevation",
               "--mapping", str(e2e_mapping), "--reports", str(reports)])
    assert rc == 0
    fid = json.loads((reports / "fidelity-paleo__paleodem_elevation.json").read_text())
    assert fid["result"] == "PASS", [c for c in fid["checks"] if c["status"] == "FAIL"]


def test_paleo_4d_determinism(tmp_path, e2e_mapping, synthetic_sources):
    outs = []
    for name in ("run1.zarr", "run2.zarr"):
        rc = main(["build", "--mapping", str(e2e_mapping), "--out", str(tmp_path / name),
                   "--reports", str(tmp_path / "reports")])
        assert rc == 0
        outs.append(tmp_path / name)
    from cubebuild.checksum import checksums_equal, store_checksums

    assert checksums_equal(store_checksums(outs[0]), store_checksums(outs[1]))


# ---------------- 真实源基线 ----------------

@requires_paleo
def test_real_scan_labels_contract():
    """原始扫描口径（tier=None）：文件名标签集的磁盘基线事实（对齐
    后的两档一致见 test_real_label_alignment）。"""
    s6 = pq.scan_slices(pq.SIXMIN_DIR)
    s1 = pq.scan_slices(pq.ONEDEG_DIR)
    l6 = [ma for ma, _ in s6]; l1 = [ma for ma, _ in s1]
    assert len(l6) == len(l1) == 109
    assert all(b > a for a, b in zip(l6, l6[1:]))       # 严格升序
    assert l6[0] == 0.0 and l6[-1] == 540.0
    # 两产品文件名标签集仅在 385/385.2、390/390.5 两处不同（磁盘实证）
    assert sorted(set(l6) - set(l1)) == [385.0, 390.0]
    assert sorted(set(l1) - set(l6)) == [385.2, 390.5]


@requires_paleo
def test_real_label_alignment():
    """验收断言：1° 两片按官方整数命名惯例舍入（385.2→385、390.5→390），
    两档 time 标签集 109/109 逐值相等；6min 零改动；对齐按标签值而非 Map
    图幅编号（被重标定的恰为两片小数命名文件，Guadalupian 编号错位族
    零改动）。"""
    s6 = pq.scan_slices(pq.SIXMIN_DIR, tier="6min")
    s1 = pq.scan_slices(pq.ONEDEG_DIR, tier="1deg")
    l6 = [ma for ma, _ in s6]
    l1 = [ma for ma, _ in s1]
    # 两档标签集 109/109 逐值相等（升序逐位）
    assert len(l1) == len(l6) == 109
    assert l1 == l6
    # 1° 两片 = 385/390：恰两片被重标定，源文件名 385.2/390.5（按标签值映射）
    deviants = sorted(
        (pq.parse_ma_label(p.name), ma) for ma, p in s1
        if pq.parse_ma_label(p.name) != ma
    )
    assert deviants == [(385.2, 385.0), (390.5, 390.0)]
    # 6min 档零改动（109 片标签 = 文件名原始值）
    assert all(pq.parse_ma_label(p.name) == ma for ma, p in s6)
    # 其余 107 片两档一致（既有事实保持）：除两片舍入外，1° 原始标签即 6min 标签
    raw1 = sorted(pq.parse_ma_label(p.name) for _, p in s1)
    assert [ma for ma in raw1 if ma not in (385.2, 390.5)] == [
        ma for ma in l6 if ma not in (385.0, 390.0)
    ]


@requires_paleo
def test_real_read_slice_6min_seam_merge():
    path = pq.SIXMIN_DIR / "Map01_PALEOMAP_6min_Holocene_0Ma.nc"
    a = pq.read_slice_6min(path)
    assert a.shape == (1801, 3600)
    assert np.isfinite(a).all()
    # 缝合并：归一列 0 = 原始 ±180 两列均值；行翻转：行 0 = 原始末行（节点 -90）
    raw = xr.open_dataset(path)["z"].values
    assert np.allclose(a[:, 0], 0.5 * (raw[::-1][:, 0] + raw[::-1][:, -1]), atol=1e-3)
    # 内部列直通（除行序翻转外逐位一致）
    assert np.array_equal(a[:, 1:], raw[::-1][:, 1:-1])


@requires_paleo
def test_real_read_slice_1deg_seam_identical():
    path = pq.ONEDEG_DIR / "Map01_PALEOMAP_1deg_Holocene_0Ma.nc"
    a = pq.read_slice_1deg(path)
    assert a.shape == (181, 360)
    assert np.isfinite(a).all()
    raw = xr.open_dataset(path)["z"].values
    assert np.array_equal(a, raw[:, :-1])               # 仅弃逐位相同的 +180 列


@requires_paleo
def test_real_6min_kernel_spot():
    """真实源核映射的解析抽检（0.05 Voronoi 界的相邻节点 sin 加权均值）。"""
    path = pq.SIXMIN_DIR / "Map01_PALEOMAP_6min_Holocene_0Ma.nc"
    a = pq.read_slice_6min(path)
    out = pq.paleo_tier_field("6min", path)
    i, j = 900, 179                                       # 目标胞 [0,0.1]×[-162.1,-162]
    e_mid = -90.0 + 0.1 * i + 0.05                        # 节点 i/i+1 的 Voronoi 界
    w0 = math.sin(math.radians(e_mid)) - math.sin(math.radians(-90.0 + 0.1 * i))
    w1 = math.sin(math.radians(-90.0 + 0.1 * (i + 1))) - math.sin(math.radians(e_mid))
    n00, n01 = a[i, j], a[i, j + 1]
    n10, n11 = a[i + 1, j], a[i + 1, j + 1]
    expect = (w0 * (n00 + n01) + w1 * (n10 + n11)) / (2 * (w0 + w1))
    assert out[i, j] == pytest.approx(expect, abs=1e-3)
