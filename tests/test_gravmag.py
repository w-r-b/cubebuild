"""测试：通用重叠保守核 + 有效性掩膜机制 + 重磁组适配器/保真。

判据只测外部行为：
- 保守核：列周期模式手算核对、线性场方位精确（值=源胞重心处场值）、
  积分恒等式（缺测归一下仍严格守恒）、无覆盖 → NaN、W = 全覆盖时
  目标胞解析面积；
- 掩膜机制：needs_validity_mask 判定表、伴生层取值/属性、管线集成
  （store 伴生层 + manifest 链接 + 结构验证两通道一致硬判据）；
- 读取器：合成 WGM 式 netCDF / EMAG 式 tif（几何断言 + 翻转/卷绕方向）；
  真实源集成（存在时）：几何契约 + 1° 档端到端 + 保真报告 PASS。
"""

from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from cubebuild.contract import needs_validity_mask
from cubebuild.gravmag import EMAG_FILE as EMAG_FILE_NAME, WGM_FILE as WGM_FILE_NAME
from cubebuild.kernels import _column_pattern, conservative_overlap_mean
from cubebuild.masks import build_validity_mask, validity_layer_name

REPO_ROOT = Path(__file__).resolve().parents[1]

TWO_MIN = Fraction(1, 30)


def _entry(mask="validity", layer_id="gravity__wgm2012_bouguer", visibility="internal"):
    from cubebuild.contract import LayerEntry

    return LayerEntry(
        id=layer_id, dims=("lat", "lon"), source="test", native_res="2min",
        native_res_deg=1.0 / 30.0, home_tier="3min",
        tiers=("1deg", "30min", "6min", "3min"), dtype="float32", unit="mGal",
        resampling="conservative_area_weighted", mask=mask, visibility=visibility,
    )


# ---------------- 保守核：列周期模式 ----------------

def test_column_pattern_fractional_factor():
    """2min→3min 因子 1.5：周期 0.1° = 2 目标列 ↔ 3 源列，重叠手算核对。"""
    C, Q = _column_pattern(TWO_MIN, Fraction(1, 20))
    assert Q == 3 and C.shape[0] == 2
    # 目标列 u0 [0,3/60]：源v0（[-1/60,1/60]）重叠 1/60；源v1（[1/60,3/60]）全 2/60
    assert np.allclose(C[0], [0, 1 / 60, 2 / 60, 0, 0, 0])
    # 目标列 u1 [3/60,6/60]：源v2（[3/60,5/60]）全；源v3（[5/60,7/60]）半
    assert np.allclose(C[1], [0, 0, 0, 2 / 60, 1 / 60, 0])
    assert np.allclose(C.sum(axis=1), 0.05)      # 每目标列被源胞完整覆盖


def test_column_pattern_integer_factor():
    """2min→6min 整数倍（3）：目标胞 [0,6/60] 与中心对齐源胞的
    重叠 = 半/全/全/半（源胞边在 ±1/60 奇点，两端半权）。"""
    C, Q = _column_pattern(TWO_MIN, Fraction(1, 10))
    assert Q == 3 and C.shape[0] == 1
    assert np.allclose(C, [[0, 1 / 60, 2 / 60, 2 / 60, 1 / 60, 0]])
    assert np.allclose(C.sum(axis=1), 0.1)


# ---------------- 保守核：端到端语义 ----------------

def _wgm_like_geometry(rs_deg=0.5):
    """WGM 式全球源几何：节点含两极，Voronoi 边 ±rs/2 钳制。"""
    nlat = int(round(180 / rs_deg)) + 1
    nlon = int(round(360 / rs_deg))
    lat_edges = -90.0 + (np.arange(nlat + 1) - 0.5) * rs_deg
    lat_edges[0], lat_edges[-1] = -90.0, 90.0
    return nlat, nlon, lat_edges


def test_linear_field_and_cell_area():
    """线性场（值=lat*1000+lon）：保守均值 = 源胞重心处场值（方位精确）；
    W = 全覆盖时目标胞解析面积 R²ΔsinΔλ。"""
    rs = Fraction(1, 2)
    nlat, nlon, lat_edges = _wgm_like_geometry(0.5)
    latc = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    lonc = -180.0 + np.arange(nlon) * 0.5
    data = latc[:, None] * 1000.0 + lonc[None, :]

    vals, W = conservative_overlap_mean(data, np.ones_like(data, bool), lat_edges, rs, Fraction(1))
    lat_t = -90.0 + np.arange(180) + 0.5
    lon_t = -180.0 + np.arange(360) + 0.5
    expect = lat_t[:, None] * 1000.0 + lon_t[None, :]
    # 离散采样偏置 ≤ 半源胞（0.25°）× 梯度（1000+1）/deg
    assert np.abs(vals - expect).max() < 300.0
    # 内部行（离极远，sin 权重近均匀）应远更精确
    assert np.abs(vals[60:120] - expect[60:120]).max() < 260.0

    from cubebuild.pixel_area import EARTH_RADIUS_KM

    A_cell = EARTH_RADIUS_KM**2 * (np.pi / 180.0) * (
        np.sin(np.deg2rad(lat_t + 0.5)) - np.sin(np.deg2rad(lat_t - 0.5))
    )
    # 全覆盖全有效 → W 逐胞 == 目标胞解析面积 R²ΔsinΔλ（行向环和 = 360 胞）
    assert np.allclose(W, A_cell[:, None], rtol=1e-9)


def test_polar_cap_gives_nan_and_partial_weight():
    """EMAG 式极冠：源边 ±80 → |中心|≥80.5 的目标行全 NaN 且 W=0；
    覆盖行取常数值（部分覆盖行权重归一无偏）。"""
    rs = Fraction(1, 2)
    lat_edges = -80.0 + np.arange(321) * 0.5
    data = np.full((320, 720), 7.0)
    vals, W = conservative_overlap_mean(data, np.ones(data.shape, bool), lat_edges, rs, Fraction(1))
    lat_t = -90.0 + np.arange(180) + 0.5
    assert (np.isnan(vals).any(axis=1) == (np.abs(lat_t) >= 80.5)).all()
    assert (W[np.abs(lat_t) >= 80.5] == 0).all()
    assert np.nanmin(np.abs(vals - 7.0)) == 0.0


def test_integral_identity_with_holes():
    """积分恒等式：Σ v_t·W_t == Σ_{valid} v_s·w_s——含缺测洞与随机场。"""
    rs = Fraction(1, 2)
    lat_edges = -80.0 + np.arange(321) * 0.5
    from cubebuild.pixel_area import EARTH_RADIUS_KM

    w_s = EARTH_RADIUS_KM**2 * (np.pi / 180.0) * (
        np.sin(np.deg2rad(lat_edges[1:])) - np.sin(np.deg2rad(lat_edges[:-1]))
    ) * float(rs)

    rng = np.random.default_rng(3)
    data = rng.uniform(0, 100, (320, 720))
    valid = rng.random((320, 720)) > 0.3
    vals, W = conservative_overlap_mean(data, valid, lat_edges, rs, Fraction(1))
    ok = W > 0
    assert np.array_equal(np.isfinite(vals), ok)          # NaN ⟺ W=0
    src_total = (np.where(valid, data, 0.0) * w_s[:, None]).sum()
    tier_total = (vals.astype(np.float64) * W)[ok].sum()
    # float32 落档舍入 → ~1e-7 量级（保真判据 1e-3 富余 4 个量级）
    assert abs(tier_total - src_total) / abs(src_total) < 1e-6


def test_missing_row_renormalized_not_lost():
    """源整行缺测 → 目标行取邻行值（权重归一），非 NaN：缺测不扩散。"""
    rs = Fraction(1, 2)
    lat_edges = -80.0 + np.arange(321) * 0.5
    data = np.full((320, 720), 7.0)
    valid = np.ones((320, 720), dtype=bool)
    valid[180:181, :] = False
    vals, W = conservative_overlap_mean(data, valid, lat_edges, rs, Fraction(1))
    # 源行 180 覆盖 lat [10,10.5]；目标行 100（[10,11]）仍有行 181 供给
    assert np.isfinite(vals[100]).all()
    assert np.allclose(vals[100], 7.0)


def test_lon_wrap_at_dateline():
    """经度环接：目标列 0（[-180,-180+rt]）由跨 ±180 源胞供给——
    单调场下不因环接出 NaN/跳变。"""
    rs = Fraction(1, 2)
    nlat, nlon, lat_edges = _wgm_like_geometry(0.5)
    lonc = -180.0 + np.arange(nlon) * 0.5
    data = np.broadcast_to(lonc[None, :] * 1.0, (nlat, nlon)).copy()
    vals, W = conservative_overlap_mean(data, np.ones((nlat, nlon), bool), lat_edges, rs, Fraction(1))
    assert (W > 0).all()
    # 目标列 0 期望值 = 覆盖源胞（跨缝 v0 与 v1）的加权重心场值
    assert -180.0 <= vals[:, 0].mean() <= -179.0


# ---------------- 掩膜机制 ----------------

@pytest.mark.parametrize(
    "mask_field,expected",
    [
        ("validity", True),
        ("validity + landsea（陆域层）", True),
        ("validity（古地理随时间变化，逐时间片掩膜）", True),
        ("NaN（板片区外）+ validity", True),      # Slab2 形态
        ("NaN + validity", True),
        ("免", False),
        ("全球全覆盖，免有效性掩膜", False),
        ("随高程层", False),
        ("自身即掩膜", False),
    ],
)
def test_needs_validity_mask_table(mask_field, expected):
    assert needs_validity_mask(mask_field) is expected


def test_build_validity_mask_values_and_attrs():
    import xarray as xr

    entry = _entry()
    data = xr.DataArray(
        np.array([[1.0, np.nan], [np.nan, 2.0]], dtype=np.float32),
        dims=("lat", "lon"),
        coords={"lat": [-1.0, 0.0], "lon": [0.0, 1.0]},
        name=entry.id,
    )
    m = build_validity_mask(data, entry)
    assert m.name == validity_layer_name(entry.id) == f"{entry.id}__validity"
    assert m.dtype == np.uint8
    np.testing.assert_array_equal(m.values, [[1, 0], [0, 1]])
    assert m.attrs["validity_of"] == entry.id
    assert m.dims == ("lat", "lon")


def test_build_validity_mask_lazy_dask():
    import dask.array
    import xarray as xr

    entry = _entry()
    arr = dask.array.from_array(np.array([[np.nan, 1.0]]), chunks=(1, 1))
    data = xr.DataArray(arr, dims=("lat", "lon"))
    m = build_validity_mask(data, entry)
    assert isinstance(m.data, dask.array.Array)   # 惰性透传
    np.testing.assert_array_equal(m.compute().values, [[0, 1]])


# ---------------- 读取器（合成文件） ----------------

def _write_synthetic_wgm(tmp_path, monkeypatch, res_deg=0.5):
    """合成 WGM 式 grd：节点注册、±180 同值、线性场。返回期望源值。"""
    import netCDF4

    step = res_deg
    nx, ny = int(round(360 / step)) + 1, int(round(180 / step)) + 1
    x = -180.0 + np.arange(nx) * step
    y = -90.0 + np.arange(ny) * step
    xx, yy = np.meshgrid(x, y)
    z = (yy * 1000.0 + xx).astype(np.float32)
    z[:, -1] = z[:, 0]                                # ±180 周期同值
    p = tmp_path / WGM_FILE_NAME
    ds = netCDF4.Dataset(p, "w", format="NETCDF3_CLASSIC")
    ds.createDimension("x", nx)
    ds.createDimension("y", ny)
    ds.node_offset = 0
    vx = ds.createVariable("x", "f4", ("x",))
    vy = ds.createVariable("y", "f4", ("y",))
    vz = ds.createVariable("z", "f4", ("y", "x"), fill_value=np.nan)
    vx[:] = x.astype(np.float32)
    vy[:] = y.astype(np.float32)
    vz[:] = z
    ds.close()

    import cubebuild.gravmag as gm

    monkeypatch.setattr(
        gm, "WGM_EXPECTED", {"shape": (ny, nx), "spacing": Fraction(1, int(round(1 / step)))}
    )
    return p


def test_load_wgm2012_synthetic(tmp_path, monkeypatch):
    import cubebuild.gravmag as gm

    p = _write_synthetic_wgm(tmp_path, monkeypatch, res_deg=0.5)
    src = gm.load_wgm2012(tmp_path)
    nlat, nlon = src.values.shape
    assert (nlat, nlon) == (361, 720)                  # 弃 +180 列
    assert float(src.lon_res) == 0.5
    # 极点半权：首末行胞宽 0.25，内部 0.5
    widths = np.diff(src.lat_edges)
    assert widths[0] == pytest.approx(0.25) and widths[-1] == pytest.approx(0.25)
    assert np.allclose(widths[1:-1], 0.5)
    assert widths.sum() == pytest.approx(180.0)
    # 值 = 线性场在节点处（南起行 0 = lat -90；列 0 = lon -180）
    assert src.values[0, 0] == pytest.approx(-90180.0, abs=1.0)
    assert src.values[-1, -1] == pytest.approx(90 * 1000 + 179.5, abs=1.0)


def test_load_wgm2012_rejects_bad_registration(tmp_path, monkeypatch):
    """node_offset=1（pixel 注册）→ 显式拒绝（几何契约）。"""
    import netCDF4

    import cubebuild.gravmag as gm

    nx, ny = 721, 361
    p = tmp_path / WGM_FILE_NAME
    ds = netCDF4.Dataset(p, "w", format="NETCDF3_CLASSIC")
    ds.createDimension("x", nx)
    ds.createDimension("y", ny)
    ds.node_offset = 1
    ds.createVariable("x", "f4", ("x",))[:] = np.linspace(-180, 180, nx).astype(np.float32)
    ds.createVariable("y", "f4", ("y",))[:] = np.linspace(-90, 90, ny).astype(np.float32)
    ds.createVariable("z", "f4", ("y", "x"), fill_value=np.nan)[:] = np.zeros((ny, nx), np.float32)
    ds.close()
    monkeypatch.setattr(gm, "WGM_EXPECTED", {"shape": (ny, nx), "spacing": Fraction(1, 2)})
    with pytest.raises(ValueError, match="node_offset"):
        gm.load_wgm2012(tmp_path)


def _write_synthetic_emag(tmp_path, res_deg=0.5, nodata_val=-3.4e38):
    """合成 EMAG 式 tif：0..360 框架、纬度缺极冠、nodata 洞。"""
    import rasterio
    from rasterio.transform import Affine

    nlat, nlon = int(round(180 / res_deg)) - 1, int(round(360 / res_deg))
    top = 90.0 - res_deg / 2.0
    transform = Affine(res_deg, 0.0, -res_deg / 2.0, 0.0, -res_deg, top)
    # 值 = lat*1000 + lon（0..360 框架原始；lon≥180 折为负经度值）
    # EMAG 列中心在整步点 c·res（transform 左缘 -res/2 偏半胞）
    a = np.empty((nlat, nlon), dtype=np.float32)
    for r in range(nlat):
        lat = top - (r + 0.5) * res_deg
        for c in range(nlon):
            lon = c * res_deg
            a[r, c] = lat * 1000.0 + (lon - 360.0 if lon >= 180.0 else lon)
    rng = np.random.default_rng(1)
    a[rng.random(a.shape) < 0.2] = nodata_val            # 20% 缺测洞
    p = tmp_path / EMAG_FILE_NAME
    with rasterio.open(
        p, "w", driver="GTiff", width=nlon, height=nlat, count=1,
        dtype="float32", nodata=nodata_val, transform=transform,
    ) as dst:
        dst.write(a, 1)
    return a, top, res_deg, nlat, nlon


def test_load_emag2_synthetic(tmp_path, monkeypatch):
    """归一化方向：行翻南起 + 经度半卷绕 + nodata→NaN；raw↔norm 抽点等值。"""
    import cubebuild.gravmag as gm

    a, top, res, nlat, nlon = _write_synthetic_emag(tmp_path, res_deg=0.5)
    monkeypatch.setattr(
        gm, "EMAG_EXPECTED",
        {
            "shape": (nlat, nlon),
            "res": res,
            "bounds": (-res / 2.0, -top, 360.0 - res / 2.0, top),
        },
    )
    src = gm.load_emag2_sealevel(tmp_path)
    assert src.values.shape == (nlat, nlon)
    assert np.array_equal(np.isfinite(src.values), src.valid)
    # 行方向：norm 行 0 = 最南（raw 行 nlat-1）；列方向：norm 列 0 中心 = -180（raw 列 nlon/2）
    # 用值场锚定：raw(r,c) 中心 lat = top-(r+0.5)res、lon = c·res
    # → norm(nlat-1-r, (c+nlon//2)%nlon) 应逐点同值（读器内已自洽核对 64 点，此处独立复核）
    half = nlon // 2
    rng = np.random.default_rng(7)
    for _ in range(50):
        r, c = int(rng.integers(nlat)), int(rng.integers(nlon))
        ii, jj = nlat - 1 - r, (c + half) % nlon
        if np.isfinite(src.values[ii, jj]):
            assert src.values[ii, jj] == a[r, c]
        else:
            assert a[r, c] == -3.4e38
    # 纬度边：底 = top - nlat·res = -top（对称极冠）
    assert src.lat_edges[0] == pytest.approx(-top)
    assert src.lat_edges[-1] == pytest.approx(top)


def test_load_emag2_rejects_wrong_frame(tmp_path, monkeypatch):
    """-180..180 框架的 tif（left=-180-res/2）→ 断言拒绝（0..360 契约）。"""
    import rasterio
    from rasterio.transform import Affine

    import cubebuild.gravmag as gm

    res = 0.5
    nlat, nlon = 359, 720
    p = tmp_path / EMAG_FILE_NAME
    with rasterio.open(
        p, "w", driver="GTiff", width=nlon, height=nlat, count=1,
        dtype="float32", nodata=-3.4e38,
        transform=Affine(res, 0.0, -180.0 - res / 2.0, 0.0, -res, 90.0 - res / 2.0),
    ) as dst:
        dst.write(np.zeros((nlat, nlon), np.float32), 1)
    monkeypatch.setattr(
        gm, "EMAG_EXPECTED",
        {"shape": (nlat, nlon), "res": res,
         "bounds": (-res / 2.0, -90.0 + res / 2.0, 360.0 - res / 2.0, 90.0 - res / 2.0)},
    )
    with pytest.raises(ValueError, match="包络"):
        gm.load_emag2_sealevel(tmp_path)


# ---------------- 构建器 + 管线集成（合成源缩格） ----------------

def test_build_wgm2012_bouguer_synthetic(tmp_path, monkeypatch):
    """合成 WGM → 1° 档：形状/坐标/dtype/attrs/线性场值（方位精确）。"""
    import cubebuild.gravmag as gm

    _write_synthetic_wgm(tmp_path, monkeypatch, res_deg=0.5)
    monkeypatch.setattr(gm, "WGM_SRC_DIR", tmp_path)
    entry = _entry()
    da = gm.build_wgm2012_bouguer(entry, "1deg")
    assert da.shape == (180, 360)
    assert da.dtype == np.float32
    assert da.attrs["visibility"] == "internal"
    assert "registration_rule" in da.attrs and da.attrs["doi"]
    vals = da.compute().values
    lat_t = -90.0 + np.arange(180) + 0.5
    lon_t = -180.0 + np.arange(360) + 0.5
    expect = lat_t[:, None] * 1000.0 + lon_t[None, :]
    assert np.abs(vals - expect).max() < 300.0          # 半源胞偏置界内


_MINI_CONTRACT = """
version: test
status: test
tiers: [1deg]
layers:
  - id: gravity__wgm2012_bouguer
    source: bouguer-wgm2012
    native_res: 30min
    home_tier: 1deg
    tiers: [1deg]
    dtype: float32
    unit: mGal
    resampling: conservative_area_weighted
    mask: validity
    visibility: internal
"""


def test_pipeline_mask_mechanism_e2e(tmp_path, monkeypatch):
    """管线集成：掩膜伴生层入 store、manifest validity_mask 链接、
    结构验证含两通道一致硬判据（验收机制点）。"""
    import cubebuild.gravmag as gm
    from cubebuild.cli import main
    from cubebuild.manifest import read_manifest

    _write_synthetic_wgm(tmp_path, monkeypatch, res_deg=0.5)
    monkeypatch.setattr(gm, "WGM_SRC_DIR", tmp_path)

    mapping = tmp_path / "mini.yaml"
    mapping.write_text(_MINI_CONTRACT, encoding="utf-8")
    out = tmp_path / "cube.zarr"
    reports = tmp_path / "reports"
    rc = main(["build", "--mapping", str(mapping), "--out", str(out), "--reports", str(reports)])
    assert rc == 0

    import json

    import xarray as xr

    with xr.open_datatree(out, engine="zarr") as dt:
        node = dt["/1deg"]
        lid = "gravity__wgm2012_bouguer"
        mname = validity_layer_name(lid)
        assert lid in node.ds.data_vars and mname in node.ds.data_vars
        m = node.ds[mname]
        assert m.dtype == np.uint8
        v = node.ds[lid]
        # 两通道一致（合成全有效源 → 掩膜全 1）
        assert (m.values == 1).all()
        assert np.isfinite(v.values).all()

    manifest = read_manifest(out)
    entry = next(l for l in manifest["layers"] if l["id"] == "gravity__wgm2012_bouguer")
    assert entry["validity_mask"] == "gravity__wgm2012_bouguer__validity"
    assert entry["visibility"] == "internal"
    assert entry["coverage"] == {"1deg": 1.0}

    report = json.loads((reports / "structure-validation.json").read_text())
    assert report["result"] == "PASS"
    names = [c["name"] for c in report["checks"]]
    assert any("两通道一致" in n for n in names)
    assert any("掩膜 uint8" in n for n in names)


def test_mask_consistency_check_catches_divergence(tmp_path):
    """结构验证确有拦截力：掩膜与数据层脱钩 → 两通道一致 FAIL。"""
    import xarray as xr

    from cubebuild.contract import load_contract
    from cubebuild.manifest import build_manifest, write_manifest
    from cubebuild.validate import validate_store

    mapping = tmp_path / "mini.yaml"
    mapping.write_text(_MINI_CONTRACT, encoding="utf-8")
    contract = load_contract(mapping)
    lid = "gravity__wgm2012_bouguer"
    mname = validity_layer_name(lid)

    tree = xr.DataTree.from_dict(
        {
            "/": xr.Dataset(attrs={"crs": "EPSG:4326"}),
            "/1deg": xr.Dataset(
                {
                    lid: xr.DataArray(
                        np.array([[1.0, np.nan], [3.0, 4.0]], dtype=np.float32),
                        dims=("lat", "lon"),
                        coords={"lat": [-0.5, 0.5], "lon": [-0.5, 0.5]},
                        attrs={"units": "mGal"},
                    ),
                    mname: xr.DataArray(
                        np.ones((2, 2), dtype=np.uint8),
                        dims=("lat", "lon"),
                        coords={"lat": [-0.5, 0.5], "lon": [-0.5, 0.5]},
                    ),
                }
            ),
        }
    )
    out = tmp_path / "bad.zarr"
    tree.to_zarr(out, mode="w", zarr_format=3)
    # manifest（结构验证入口需要）
    entry = contract.layers[0]
    manifest = build_manifest(contract, [entry], {(lid, "1deg"): 0.75})
    write_manifest(manifest, out)

    report = validate_store(out, contract)
    failed = [c for c in report["checks"] if c["status"] == "FAIL"]
    assert any("两通道一致" in c["name"] for c in failed)


# ---------------- 真实源集成（存在时执行） ----------------

@pytest.mark.skipif(not (REPO_ROOT / "original data/gravity/bouguer-wgm2012").is_dir(),
                    reason="WGM 源目录不存在")
def test_real_wgm_geometry_and_tier():
    import cubebuild.gravmag as gm

    src = gm.load_wgm2012()
    assert src.values.shape == (5401, 10800)
    assert src.valid.all()
    assert float(src.lon_res) == pytest.approx(1 / 30)
    assert np.diff(src.lat_edges).sum() == pytest.approx(180.0)
    arr = gm.build_wgm2012_bouguer(_entry(), "1deg")
    vals = arr.compute().values
    assert vals.shape == (180, 360)
    assert np.isfinite(vals).all()
    # 布格异常物理界（源值域 [-529, 1005] mGal 的保守均值界内）
    assert -600 < vals.min() < 0 and 0 < vals.max() < 1100


@pytest.mark.skipif(not (REPO_ROOT / "original data/magnetics/emag2-v3").is_dir(),
                    reason="EMAG 源目录不存在")
def test_real_emag_geometry_and_coverage():
    import cubebuild.gravmag as gm

    src = gm.load_emag2_sealevel()
    assert src.values.shape == (5399, 10800)
    assert src.lat_edges[0] == pytest.approx(-90.0 + 1 / 60.0, abs=1e-9)
    assert src.lat_edges[-1] == pytest.approx(90.0 - 1 / 60.0, abs=1e-9)
    assert 0.4 < src.valid.mean() < 0.6
    arr = gm.build_emag2_sealevel(
        _entry(mask="validity", layer_id="magnetics__emag2_sealevel", visibility="public"), "1deg"
    )
    vals = arr.compute().values
    cov = np.isfinite(vals).mean()
    assert 0.45 < cov < 0.65                            # 缺测结构在粗档部分补足
