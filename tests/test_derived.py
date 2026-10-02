"""测试：派生层三件（陆海掩膜/坡度/起伏度）。

判据只测外部行为：
- 众数核：手算多数 + 平票取小值 + 非整数倍拒绝；
- 坡度：解析线性场精确值（公式正确性）+ numpy 独立参考实现（dask 管线正确性）
  + 经度周期边界（日期线环接）+ 纬度边界单侧差分；
- 起伏度：暴力窗口循环参考实现（周期经度 + 裁剪纬度）+ 手算小例；
- 迷你端到端：288 位合成瓦片（坐标线性场）→ CLI build →
  结构验证 PASS → manifest 派生层登记（derived_from/derivation/口径注记）
  → 三层保真 PASS → 双跑确定性；
- 掩膜冰面口径：surface 优先瓦片选择 → 新旧掩膜差异像元集
  ≡ 冰下盆地像元集（复合高程 <0 且 ice-surface >0）逐位断言（合成世界）。

迷你几何：源瓦 300×300（1/20°），档位 [1deg, 30min, 6min]；landsea 的
原生档经 monkeypatch 弹性为 6min（真实语义 30sec 依赖 15s 源，迷你源粗于
30sec 无法构成；代码路径完全一致，30sec 原生语义由真实构建验收承担）。
"""

from pathlib import Path

import dask.array as da
import numpy as np
import pytest

from cubebuild.derived import (
    R_M,
    landsea_threshold,
    mode_downsample,
    relief_core,
    slope_core,
)
from cubebuild.etopo import LAT_BANDS, LON_COLS, label_bounds
from cubebuild.grids import tier_deg
from cubebuild.kernels import aggregate_mode

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------- 众数核 ----------------

def test_aggregate_mode_majority_and_tie():
    data = np.array([[1, 1], [1, 0]], dtype=np.uint8)
    assert aggregate_mode(data, 2)[0, 0] == 1          # 3:1 → 1
    tie = np.array([[1, 0], [0, 0]], dtype=np.uint8)
    assert aggregate_mode(tie, 2)[0, 0] == 0           # 2:2 平票 → 小值


def test_aggregate_mode_multiclass_tie_smallest():
    # 块(0,0)=[5,2|2,5] 平2 → 2；块(0,1)=[5,5|5,2] 3:1 → 5
    data = np.array([[5, 2, 5, 5], [2, 5, 5, 2]], dtype=np.uint8)
    out = aggregate_mode(data, 2)
    assert out.shape == (1, 2)
    assert out[0, 0] == 2                               # 平票 → 小值
    assert out[0, 1] == 5                               # 多数 → 5


def test_aggregate_mode_rejects_noninteger_factor():
    with pytest.raises(ValueError):
        aggregate_mode(np.zeros((3, 4), dtype=np.uint8), 2)


def test_mode_downsample_matches_numpy():
    rng = np.random.default_rng(3)
    z = rng.integers(0, 3, size=(6, 8)).astype(np.uint8)
    zz = da.from_array(z, chunks=(2, 3))                # 块不整除 factor 的反例防护
    with pytest.raises(ValueError):
        mode_downsample(zz, 4)
    zz = da.from_array(z, chunks=(2, 4))                # 各块 2/4 均整除 factor 2
    np.testing.assert_array_equal(
        mode_downsample(zz, 2).compute(), aggregate_mode(z, 2)
    )


# ---------------- 坡度 ----------------

def _small_grid(nlat=5, nlon=7, res=0.5):
    lat = -1.0 + (np.arange(nlat) + 0.5) * res
    lon = -1.75 + (np.arange(nlon) + 0.5) * res
    return lat, lon


def test_slope_flat_is_zero():
    lat, _ = _small_grid()
    z = da.full((5, 7), 123.0, chunks=(2, 3))
    out = slope_core(z, lat, 0.5).compute()
    assert out.shape == (5, 7)
    np.testing.assert_array_equal(out, 0.0)


def test_slope_analytic_linear_field():
    """线性场（弧度坐标）下中央/单侧差分均解析精确——公式正确性。

    z = p·φ + q·λ（弧度）→ gx = q/(R·cosφ)，gy = p/R（逐点解析）。
    q=0 时全网格（含经度环接列、纬度边界行）解析精确；q≠0 时线性场在
    经度环接处不连续，仅内部列参与解析断言（环接列由下一条测试钉死）。
    """
    lat, lon = _small_grid()
    phi = np.deg2rad(lat)[:, None]
    lam = np.deg2rad(lon)[None, :]
    res = 0.5

    # q=0：仅随纬度线性 → 全网格解析精确（含首/末行单侧差分）
    p = 500.0
    z = da.from_array((p * phi + np.zeros_like(lam)).astype(np.float64), chunks=(2, 3))
    expect = np.degrees(np.arctan(np.abs(p) / R_M)) * np.ones((5, 7))
    np.testing.assert_allclose(slope_core(z, lat, res).compute(), expect, rtol=1e-12)

    # p, q ≠ 0：内部列解析精确
    q = 300.0
    z2 = da.from_array((p * phi + q * lam).astype(np.float64), chunks=(2, 3))
    gx = np.broadcast_to(q / (R_M * np.cos(phi)), (5, 7))
    gy = np.full((5, 7), p / R_M)
    expect2 = np.degrees(np.arctan(np.hypot(gx, gy)))
    out = slope_core(z2, lat, res).compute()
    np.testing.assert_allclose(out[:, 1:-1], expect2[:, 1:-1], rtol=1e-12)
    np.testing.assert_allclose(out[0, 1:-1], expect2[0, 1:-1], rtol=1e-12)   # 边界行单侧
    np.testing.assert_allclose(out[-1, 1:-1], expect2[-1, 1:-1], rtol=1e-12)


def test_slope_periodic_wrap_columns():
    """经度环接：列 0 的左邻 = 列 nlon−1，列 nlon−1 的右邻 = 列 0。"""
    lat, _ = _small_grid()
    rng = np.random.default_rng(11)
    z = rng.uniform(-100, 4000, size=(5, 7))
    out = slope_core(da.from_array(z, chunks=(2, 3)), lat, 0.5).compute()

    dlon = R_M * np.deg2rad(0.5) * np.cos(np.deg2rad(lat))[:, None]
    gx_wrap0 = (z[:, 1] - z[:, -1]) / (2.0 * dlon[:, 0])       # j=0：右邻 1、左邻 6
    gx_wrapN = (z[:, 0] - z[:, -2]) / (2.0 * dlon[:, 0])       # j=6：右邻 0（环）、左邻 5
    dy = R_M * np.deg2rad(0.5)
    gy = np.gradient(z, dy, axis=0)
    for j, gx_col in ((0, gx_wrap0), (6, gx_wrapN)):
        expect = np.degrees(np.arctan(np.hypot(gx_col, gy[:, j])))
        np.testing.assert_allclose(out[:, j], expect, rtol=1e-12)


def test_slope_matches_numpy_reference():
    """dask 管线 vs 独立 numpy 参考（np.roll 周期 + np.gradient 单侧）。"""
    lat, _ = _small_grid(nlat=9, nlon=11)
    rng = np.random.default_rng(5)
    z = rng.uniform(-5000, 5000, size=(9, 11))

    def ref(z, lat, res):
        gx = (np.roll(z, -1, 1) - np.roll(z, 1, 1)) / (
            2.0 * R_M * np.deg2rad(res) * np.cos(np.deg2rad(lat))[:, None]
        )
        gy = np.gradient(z, R_M * np.deg2rad(res), axis=0)
        return np.degrees(np.arctan(np.hypot(gx, gy)))

    out = slope_core(da.from_array(z, chunks=(3, 4)), lat, 0.5).compute()
    np.testing.assert_allclose(out, ref(z, lat, 0.5), rtol=1e-12)


# ---------------- 起伏度 ----------------

def _relief_bruteforce(z):
    m, n = z.shape
    out = np.empty((m, n), dtype=np.float64)
    for i in range(m):
        rows = [r for r in (i - 1, i, i + 1) if 0 <= r < m]     # 纬度边界裁剪
        for j in range(n):
            cols = [(j - 1) % n, j, (j + 1) % n]                # 经度周期
            w = z[np.ix_(rows, cols)]
            out[i, j] = w.max() - w.min()
    return out


def test_relief_matches_bruteforce():
    rng = np.random.default_rng(7)
    z = rng.uniform(-4000, 8000, size=(6, 9))
    out = relief_core(da.from_array(z, chunks=(2, 3))).compute()
    np.testing.assert_allclose(out, _relief_bruteforce(z), rtol=1e-12)


def test_relief_handcrafted():
    # 5×7，峰 (1,3)=10，其余 0
    z = np.zeros((5, 7))
    z[1, 3] = 10.0
    out = relief_core(da.from_array(z, chunks=(5, 7))).compute()
    assert out[1, 3] == 10.0                             # 峰自身
    assert out[0, 3] == 10.0                             # 纬度边界行：裁剪窗 2×3 含峰
    assert out[2, 3] == 10.0 and out[1, 2] == 10.0       # 内部邻域
    assert out[4, 3] == 0.0                              # 远离峰（窗行 3,4）
    assert out[1, 6] == 0.0                              # 峰在 j=3 不触环接列
    # 峰挪到 j=0：环接使 j=6（峰左邻 via 周期）的窗含峰
    z2 = np.zeros((5, 7))
    z2[2, 0] = 10.0
    out2 = relief_core(da.from_array(z2, chunks=(5, 7))).compute()
    assert out2[2, 6] == 10.0                            # j=6 窗 = 列 5,6,0（环接含峰）
    assert out2[2, 5] == 0.0                             # j=5 窗 = 列 4,5,6（不含峰）
    assert out2[2, 1] == 10.0
    assert out2[2, 2] == 0.0


# ---------------- 陆海阈值 ----------------

def test_landsea_threshold_boundary():
    z = da.from_array(np.array([[-0.1, 0.0], [0.1, 5000.0]], dtype=np.float32), chunks=1)
    np.testing.assert_array_equal(
        landsea_threshold(z).compute(), np.array([[0, 0], [1, 1]], dtype=np.uint8)
    )


# ---------------- 构建器登记 ----------------

def test_builders_registered():
    from cubebuild.layers import LAYER_BUILDERS

    for lid in ("derived__landsea_mask", "derived__slope", "derived__relief"):
        assert lid in LAYER_BUILDERS


# ---------------- 迷你端到端（288 位合成瓦片 → CLI 全链路）----------------

_MINI_MAPPING = """
version: test
status: test
tiers: [1deg, 30min, 6min]
layers:
  - id: topography__bedrock_elevation
    source: test-etopo
    native_res: 3min
    home_tier: 6min
    tiers: [1deg, 30min, 6min]
    dtype: float32
    unit: m
    resampling: conservative_area_weighted
    mask: none
  - id: derived__landsea_mask
    source: 派生自 test-etopo ice-surface（0m 等值线，冰面口径）
    native_res: 6min
    home_tier: 6min
    tiers: [1deg, 30min, 6min]
    dtype: uint8
    unit: flag（0=海，1=陆）
    resampling: mode
    mask: 自身即掩膜
    notes: 掩膜基于冰面高程（冰盖归陆），与高程层裸地复合口径不同——存在 mask=陆而高程<0 的冰下盆地像元（南极/格陵兰）
  - id: derived__slope
    source: 派生自 topography__bedrock_elevation
    native_res: 6min
    home_tier: 6min
    tiers: [1deg, 30min, 6min]
    dtype: float32
    unit: degree
    resampling: none（逐档从该档高程重算）
    mask: 随高程层
  - id: derived__relief
    source: 派生自 topography__bedrock_elevation
    native_res: 6min
    home_tier: 6min
    tiers: [1deg, 30min, 6min]
    dtype: float32
    unit: m
    resampling: none（逐档从该档高程重算）
    mask: 随高程层
"""

_TILE_SHAPE = (300, 300)              # 15° / 300 = 1/20° 源分辨率
_SRC_RES = 15.0 / 300


@pytest.fixture()
def mini_env(tmp_path, monkeypatch):
    """合成 288 瓦（坐标线性场）+ 契约 + 模块补丁。

    NATIVE_TIER 弹性为 6min：迷你源（1/20°）粗于 30sec，无法构成 30sec
    原生档；代码路径与真实构建一致，30sec 语义由真实构建验收。
    """
    import cubebuild.derived as derived_mod
    import cubebuild.etopo as etopo_mod
    import cubebuild.fidelity as fidelity_mod
    import rasterio

    monkeypatch.setattr(etopo_mod, "TILE_SHAPE", _TILE_SHAPE)
    monkeypatch.setattr(etopo_mod, "SRC_RES_DEG", _SRC_RES)

    paths = {}
    for bi, band in enumerate(LAT_BANDS):
        band_top = 90 - bi * 15
        for col in LON_COLS:
            label = band + col
            _, left = label_bounds(label)
            rr = np.arange(_TILE_SHAPE[0])[:, None]
            cc = np.arange(_TILE_SHAPE[1])[None, :]
            # 值 = 纬度×1000 + 经度 − 20000（陆地 ≈ lat>20°，占比 ~0.39 在
            # 粗错捕捉器界内；线性场捕获行/列/环旋转错位）
            v = ((band_top - (rr + 0.5) * _SRC_RES) * 1000.0
                 + (left + (cc + 0.5) * _SRC_RES) - 20000.0)
            f = tmp_path / f"ETOPO_2022_v1_15s_{label}_bed.tif"
            with rasterio.open(
                f, "w", driver="GTiff", width=_TILE_SHAPE[1], height=_TILE_SHAPE[0],
                count=1, dtype="float32", nodata=-99999.0,
            ) as dst:
                dst.write(v.astype(np.float32), 1)
            paths[label] = f

    monkeypatch.setattr(etopo_mod, "require_full_coverage", lambda *a, **k: paths)
    # 掩膜冰面口径：ice-surface 选择同样指向合成瓦（迷你世界
    # ice-surface ≡ bed 值；选择/组装真实路径由专项机制测试覆盖）
    monkeypatch.setattr(etopo_mod, "require_surface_coverage", lambda *a, **k: paths)
    monkeypatch.setattr(derived_mod, "NATIVE_TIER", "6min")
    monkeypatch.setattr(fidelity_mod, "NATIVE_TIER", "6min")

    mapping = tmp_path / "mini-mapping.yaml"
    mapping.write_text(_MINI_MAPPING, encoding="utf-8")
    return mapping


def _run_build(tmp_path, mapping, out_name):
    from cubebuild.cli import main

    out = tmp_path / out_name
    rc = main(["build", "--mapping", str(mapping), "--out", str(out),
               "--reports", str(tmp_path / "reports")])
    return rc, out


def test_mini_e2e_build_validate_manifest(tmp_path, mini_env):
    import json

    import xarray as xr

    from cubebuild.manifest import read_manifest

    rc, out = _run_build(tmp_path, mini_env, "cube.zarr")
    assert rc == 0

    # 结构：三层全档入 store，dtype/维序/配准由结构验证把关
    dt = xr.open_datatree(out, engine="zarr")
    for tier in ("1deg", "30min", "6min"):
        node = dt[f"/{tier}"]
        assert str(node["derived__landsea_mask"].dtype) == "uint8"
        assert str(node["derived__slope"].dtype) == "float32"
        assert str(node["derived__relief"].dtype) == "float32"
    dt.close()

    # manifest：派生层登记（来源 + 派生方法 + 口径语义——掩膜冰面口径）
    manifest = read_manifest(out)
    entries = {e["id"]: e for e in manifest["layers"]}
    for lid in ("derived__slope", "derived__relief"):
        e = entries[lid]
        assert e["derived_from"] == "topography__bedrock_elevation"
        assert e["derivation"]
        assert e["coverage"] == {"1deg": 1.0, "30min": 1.0, "6min": 1.0}
    m = entries["derived__landsea_mask"]
    assert m["coverage"] == {"1deg": 1.0, "30min": 1.0, "6min": 1.0}
    assert "ice-surface" in m["derived_from"]              # 来源＝冰面复合，非高程层
    assert "known_bias" not in m                            # 冰区偏差已修
    assert "冰面" in m["derivation"] and "冰下盆地" in m["derivation"]
    assert "冰面" in m["notes"] and "冰下盆地" in m["notes"]  # manifest 注记

    report = json.loads((tmp_path / "reports" / "structure-validation.json").read_text())
    assert report["result"] == "PASS"
    summary = json.loads((tmp_path / "reports" / "build-summary.json").read_text())
    assert set(summary["built"]) == {
        "topography__bedrock_elevation",
        "derived__landsea_mask",
        "derived__slope",
        "derived__relief",
    }


def test_mini_e2e_derived_fidelity(tmp_path, mini_env):
    """三层保真 CLI：逐位一致性 + 值域 + 陆地占比全 PASS。"""
    import json

    from cubebuild.cli import main

    rc, out = _run_build(tmp_path, mini_env, "cube.zarr")
    assert rc == 0
    for lid in ("derived__landsea_mask", "derived__slope", "derived__relief"):
        rc2 = main(["fidelity", "--store", str(out), "--layer", lid,
                    "--mapping", str(mini_env),
                    "--reports", str(tmp_path / "reports")])
        assert rc2 == 0, lid
        rep = json.loads(
            (tmp_path / "reports" / f"fidelity-{lid}.json").read_text()
        )
        assert rep["result"] == "PASS", rep["checks"]
        if lid == "derived__landsea_mask":
            # 掩膜源＝全域 ice-surface 复合（冰面口径），非 store 高程层
            assert "ice-surface" in rep["source_layer"]
        else:
            assert rep["source_layer"] == "topography__bedrock_elevation"
    # landsea 报告含口径注记（语义注记供论文侧消费）
    rep = json.loads(
        (tmp_path / "reports" / "fidelity-derived__landsea_mask.json").read_text()
    )
    assert any("冰面" in c.get("summary", "") and "冰下盆地" in c.get("summary", "")
              for c in rep["checks"])


def test_mini_e2e_determinism(tmp_path, mini_env):
    from cubebuild.checksum import checksums_equal, store_checksums

    rc1, out1 = _run_build(tmp_path, mini_env, "run1.zarr")
    rc2, out2 = _run_build(tmp_path, mini_env, "run2.zarr")
    assert rc1 == rc2 == 0
    cs1, cs2 = store_checksums(out1), store_checksums(out2)
    assert cs1 and checksums_equal(cs1, cs2)


# ---------------- 掩膜冰面口径：机制级断言 ----------------

def test_landsea_ice_surface_basin_diff(tmp_path, monkeypatch):
    """新旧掩膜差异像元集 ≡ 冰下盆地像元集（复合高程 <0 且 ice-surface >0）。

    定向验证断言的机制级实证（真实瓦片选择 + 真实组装 + 真实阈值链，
    合成世界构造可判定的盆地几何）：
    - 288 位 bed 瓦：线性场 v = lat×1000 + lon − 20000（复合高程基线）；
    - 冰区带（N00/S15 两带 48 瓦，bed 全 <0 模拟冰下盆地）surface 瓦
      = v + 25000 → surface ∈ (−10180, 20180)，部分 >0（冰面出水）；
    - 其余 240 位 surface 瓦 = v（无冰区 bed ≡ surface）。
    期望：diff(new, old) 逐位 ≡ {composite<0} ∩ {ice-surface>0}，
    且无「陆→海」翻转（surface ≥ bed 的方向性）。
    """
    import cubebuild.etopo as etopo_mod
    import rasterio

    monkeypatch.setattr(etopo_mod, "TILE_SHAPE", _TILE_SHAPE)
    monkeypatch.setattr(etopo_mod, "SRC_RES_DEG", _SRC_RES)

    src, ice = tmp_path / "src", tmp_path / "ice"
    src.mkdir()
    ice.mkdir()

    def write(directory, label, kind, v):
        f = directory / f"ETOPO_2022_v1_15s_{label}_{kind}.tif"
        with rasterio.open(
            f, "w", driver="GTiff", width=_TILE_SHAPE[1], height=_TILE_SHAPE[0],
            count=1, dtype="float32", nodata=-99999.0,
        ) as dst:
            dst.write(v.astype(np.float32), 1)

    ice_bands = {"N00", "S15"}                  # 合成冰区（两带 48 位）
    for bi, band in enumerate(LAT_BANDS):
        band_top = 90 - bi * 15
        for col in LON_COLS:
            label = band + col
            _, left = label_bounds(label)
            rr = np.arange(_TILE_SHAPE[0])[:, None]
            cc = np.arange(_TILE_SHAPE[1])[None, :]
            v = ((band_top - (rr + 0.5) * _SRC_RES) * 1000.0
                 + (left + (cc + 0.5) * _SRC_RES) - 20000.0)
            write(tmp_path, label, "bed", v)                 # 复合口径基线
            if band in ice_bands:
                write(ice, label, "surface", v + 25000.0)    # 冰区 surface（冰目录）
            else:
                write(src, label, "surface", v)               # 无冰区 ≡ bed

    # 真实选择 + 真实组装（无 monkeypatch 掩蔽）：N00/S15 的 surface 来自
    # ice 目录、其余来自 src；bed 优先复合（elevation_array）不受影响
    composite = etopo_mod.elevation_array("6min", src_dir=tmp_path)
    surface = etopo_mod.ice_surface_array("6min", src_dir=src, ice_surface_dir=ice)
    old_mask = landsea_threshold(composite)
    new_mask = landsea_threshold(surface)

    diff = old_mask != new_mask
    basin = (composite < 0) & (surface > 0)
    land_to_sea = (old_mask == 1) & (new_mask == 0)          # 方向性：不应存在
    n_diff, n_mismatch, n_l2s = da.compute(
        diff.sum(), (diff != basin).sum(), land_to_sea.sum()
    )
    assert int(n_mismatch) == 0          # 断言：差异像元集 ≡ 冰下盆地像元集
    assert int(n_l2s) == 0              # 冰面口径只增陆不减陆
    assert int(n_diff) > 0              # 机制确有差异（合成盆地非空）
