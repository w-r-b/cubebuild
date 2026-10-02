"""集成验证测试：全量结构验证拦截力 + 保真汇总 + 回报四项 +
物理交叉（合成源 + 真实源集成）。

真实源集成测试（skipif 数据/制品存在才跑）：物理交叉四组对真实 store 与
验证参考出数值报告——只验证「报告可出且全部 report 制」，不判数值优劣
（只报告不拦截）。
"""

import json
import os
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import cubebuild.crosscheck as cc
import cubebuild.integration as ci
from cubebuild.contract import load_contract
from cubebuild.grids import tier_centers
from cubebuild.manifest import build_manifest, write_manifest
from cubebuild.validate import validate_store

REPO_ROOT = Path(__file__).resolve().parents[1]

# ---------------- 合成契约/工具 ----------------

_WGM_LAYER = """
  - id: gravity__wgm2012_bouguer
    source: test
    native_res: 2min
    home_tier: 1deg
    tiers: [1deg]
    dtype: float32
    unit: mGal
    resampling: none
    mask: validity
"""

_PIXEL_AREA_LAYER = """
  - id: derived__pixel_area
    source: 纬度解析公式
    native_res: exact
    home_tier: 逐档
    tiers: [1deg]
    dtype: float32
    unit: km2
    resampling: exact_formula
    mask: 免
"""


def _mapping(tmp_path, layers: str, name="mini.yaml") -> Path:
    p = tmp_path / name
    p.write_text(
        f"version: test\nstatus: test\ntiers: [1deg]\nlayers:\n{layers}",
        encoding="utf-8",
    )
    return p


def _write_store(tmp_path, contract, entries, tiers=("1deg",)) -> Path:
    """合成 store：各层 1° 网格全数组 + manifest（覆盖率 1.0）。

    pixel_area 层填解析公式真值（结构验证有数值=公式逐档核对）。
    """
    from cubebuild.masks import validity_layer_name
    from cubebuild.pixel_area import pixel_area_rows

    lat, lon = tier_centers("1deg")
    vars_ = {}
    for e in entries:
        if e.id == "derived__pixel_area":
            arr = np.repeat(
                pixel_area_rows("1deg")[:, None].astype(np.float32), 360, axis=1
            )
        else:
            arr = np.zeros((180, 360), dtype=np.float32)
        vars_[e.id] = xr.DataArray(arr, dims=("lat", "lon"),
                                   coords={"lat": lat, "lon": lon},
                                   attrs={"units": e.unit})
        if "validity" in e.mask:
            vars_[validity_layer_name(e.id)] = xr.DataArray(
                np.ones((180, 360), dtype=np.uint8), dims=("lat", "lon"),
                coords={"lat": lat, "lon": lon},
            )
    tree = xr.DataTree.from_dict({
        "/": xr.Dataset(attrs={"crs": "EPSG:4326"}),
        "/1deg": xr.Dataset(vars_),
    })
    out = tmp_path / "cube.zarr"
    tree.to_zarr(out, mode="w", zarr_format=3)
    coverage = {(e.id, t): 1.0 for e in entries for t in e.tiers if t in tiers}
    manifest = build_manifest(contract, entries, coverage)
    write_manifest(manifest, out)
    return out


# ---------------- 全量结构验证拦截力 ----------------

def test_validate_missing_layer_fails(tmp_path):
    """契约层未入 manifest（全量验证：缺层即 FAIL，不静默跳过）。"""
    contract = load_contract(_mapping(tmp_path, _WGM_LAYER + _PIXEL_AREA_LAYER))
    out = _write_store(tmp_path, contract, [contract.layers[0]])   # 只建 wgm

    report = validate_store(out, contract)
    assert report["result"] == "FAIL"
    names = {c["name"]: c for c in report["checks"]}
    assert names["derived__pixel_area: manifest 登记"]["status"] == "FAIL"
    assert names["1° 全集档完整性（全部层含 3D/4D 节点与 internal 齐汇）"]["status"] == "FAIL"


def test_validate_manifest_extra_layer_fails(tmp_path):
    """manifest 多出契约外层 → 集合互检 FAIL。"""
    contract_big = load_contract(_mapping(tmp_path, _WGM_LAYER + _PIXEL_AREA_LAYER, "big.yaml"))
    out = _write_store(tmp_path, contract_big, list(contract_big.layers))
    contract_small = load_contract(_mapping(tmp_path, _WGM_LAYER, "small.yaml"))

    report = validate_store(out, contract_small)
    names = {c["name"]: c for c in report["checks"]}
    assert names["manifest 与契约层集合一致"]["status"] == "FAIL"
    assert "derived__pixel_area" in names["manifest 与契约层集合一致"]["detail"]


def test_validate_full_set_tier_missing_array_fails(tmp_path):
    """层在 manifest 但 store 1° 档缺数组 → 全集档完整性 FAIL。"""
    two_tier = """
  - id: gravity__wgm2012_bouguer
    source: test
    native_res: 2min
    home_tier: 30min
    tiers: [1deg, 30min]
    dtype: float32
    unit: mGal
    resampling: none
    mask: validity
"""
    contract = load_contract(_mapping(tmp_path, two_tier))
    # store 只放 30min 档（1° 缺失）；manifest 按契约两层档登记
    from cubebuild.masks import validity_layer_name

    lat, lon = tier_centers("30min")
    e = contract.layers[0]
    tree = xr.DataTree.from_dict({
        "/": xr.Dataset(attrs={"crs": "EPSG:4326"}),
        "/30min": xr.Dataset({
            e.id: xr.DataArray(np.zeros((360, 720), dtype=np.float32),
                               dims=("lat", "lon"),
                               coords={"lat": lat, "lon": lon},
                               attrs={"units": "mGal"}),
            validity_layer_name(e.id): xr.DataArray(
                np.ones((360, 720), dtype=np.uint8), dims=("lat", "lon"),
                coords={"lat": lat, "lon": lon}),
        }),
    })
    out = tmp_path / "cube.zarr"
    tree.to_zarr(out, mode="w", zarr_format=3)
    write_manifest(build_manifest(contract, [e], {(e.id, t): 1.0 for t in e.tiers}), out)

    report = validate_store(out, contract)
    assert report["result"] == "FAIL"
    names = {c["name"]: c for c in report["checks"]}
    assert names["1° 全集档完整性（全部层含 3D/4D 节点与 internal 齐汇）"]["status"] == "FAIL"
    assert any(
        n.startswith("gravity__wgm2012_bouguer@1deg") and c["status"] == "FAIL"
        for n, c in names.items()
    )


def test_validate_full_contract_passes(tmp_path):
    """全契约层齐备 → PASS（全量语义的正向用例）。"""
    contract = load_contract(_mapping(tmp_path, _WGM_LAYER + _PIXEL_AREA_LAYER))
    out = _write_store(tmp_path, contract, list(contract.layers))
    report = validate_store(out, contract)
    assert report["result"] == "PASS", [
        c for c in report["checks"] if c["status"] == "FAIL"
    ]


# ---------------- 统计工具 ----------------

def test_pair_stats_linear():
    x = np.linspace(-100, 100, 201)
    y = 2.0 * x + 1.0
    s = cc._pair_stats(x, y)          # diff = y − x = x + 1
    assert s["n"] == 201
    assert s["pearson_r"] == pytest.approx(1.0)
    assert s["ols_slope"] == pytest.approx(2.0)
    assert s["ols_intercept"] == pytest.approx(1.0)
    assert s["mean_diff"] == pytest.approx(1.0)           # mean(x)+1, mean(x)=0
    assert s["rmse"] == pytest.approx(np.sqrt(1.0 + np.mean(x * x)))


def test_pair_stats_mask_and_nan():
    x = np.array([1.0, 2.0, np.nan, 4.0])
    y = np.array([1.0, 2.0, 3.0, np.nan])
    s = cc._pair_stats(x, y)
    assert s["n"] == 2          # NaN 对剔除


# ---------------- 验证参考读取器（合成源） ----------------

def test_load_crust1_moho(tmp_path):
    """CRUST1.0 解析：北起行序翻转南起、经度内循环、值域契约。"""
    lines = []
    for i in range(180):
        la = 89.5 - i
        v = -(10.0 + (90.0 - la) / 3.0)      # -10.17..-69.83，值域内
        for j in range(360):
            lo = -179.5 + j
            lines.append(f"{lo:.1f}  {la:.1f}  {v:.4f}")
    src = tmp_path / "crust1.0"
    src.mkdir()
    (src / "depthtomoho.xyz").write_text("\n".join(lines) + "\n", encoding="ascii")

    arr = cc.load_crust1_moho(src)
    assert arr.shape == (180, 360)
    # 南起翻转：行 0 = lat -89.5（最深处），行 -1 = lat 89.5（最浅处）
    assert float(arr[0, 0]) == pytest.approx(-(10.0 + 179.5 / 3.0), abs=1e-3)
    assert float(arr[-1, 0]) == pytest.approx(-(10.0 + 0.5 / 3.0), abs=1e-3)
    # 地标位：85.5E/32.5N → 行 = 32.5-(-89.5) = 122，列 = 85.5+179.5 = 265
    assert float(arr[122, 265]) == pytest.approx(-(10.0 + (90.0 - 32.5) / 3.0), abs=1e-3)


def test_load_crust1_moho_rejects_drift(tmp_path):
    src = tmp_path / "crust1.0"
    src.mkdir()
    lines = []
    for i in range(180):                 # 坐标规则但值域越界（正值向上漂移）
        la = 89.5 - i
        for j in range(360):
            lines.append(f"{-179.5 + j:.1f}  {la:.1f}  5.0000")
    (src / "depthtomoho.xyz").write_text("\n".join(lines) + "\n", encoding="ascii")
    with pytest.raises(ValueError, match="值域"):
        cc.load_crust1_moho(src)


def test_load_globsed(tmp_path, monkeypatch):
    """GlobSed 解析：南起免翻、±180 弃列、极点行常值、Voronoi 胞边。"""
    monkeypatch.setattr(cc, "GLOBSED_EXPECTED", {"shape": (5, 7), "spacing": Fraction(1, 12)})
    nlat, nlon = 5, 7
    z = np.full((nlat, nlon), 500.0, dtype=np.float32)
    z[2, 3] = np.nan                      # 「陆地」缺测
    # 源纬度南起：行 0 = lat -90（南极行），行 -1 = 北极行；两极行都须常值
    z[0, :] = 3101.95                     # 南极行常值
    z[-1, :] = 1873.60                    # 北极行常值
    ds = xr.Dataset(
        {"z": (("lat", "lon"), z)},
        coords={
            "lat": np.arange(nlat) / 12.0 - 90.0,
            "lon": np.arange(nlon) / 12.0 - 180.0,
        },
    )
    ds.to_netcdf(tmp_path / "GlobSed-v3.nc", engine="netcdf4")

    src = cc.load_globsed(tmp_path)
    assert src.values.shape == (5, 6)          # 弃 +180 重复列
    assert src.valid.sum() == 5 * 6 - 1
    assert src.lat_edges[0] == -90.0 and src.lat_edges[-1] == 90.0
    assert float(src.lon_res) == pytest.approx(1 / 12)


# ---------------- EMAG2 谱一致性（合成） ----------------

def test_continuation_fit_recovers_delta_z():
    """已知 Δz 的合成上延对 → 谱比拟合恢复 Δz（拟合器的数值正确性）。"""
    rng = np.random.default_rng(0)
    n, res, lat_c = 512, 1.0 / 30.0, 30.0
    f = rng.standard_normal((n, n)).astype(np.float32)
    k = cc._k_grid(n, res, lat_c)[:, : n // 2 + 1]
    dz0 = 3.0
    up = np.fft.irfft2(np.fft.rfft2(f) * np.exp(-dz0 * k), s=f.shape)

    k_bins, p_sea, cnt = cc._radial_power(f, res, lat_c)
    _, p_up, _ = cc._radial_power(up.astype(np.float32), res, lat_c)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = p_up / p_sea
    fit = cc._continuation_fit(k_bins, ratio, cnt)
    assert fit["delta_z_km"] == pytest.approx(dz0, abs=0.3)
    assert fit["r2"] > 0.99


def test_continuation_fit_flags_deviant():
    """非纯上延对（放大版，ratio>1 全带）→ frac_ratio_gt1 诊断满格。"""
    rng = np.random.default_rng(1)
    f = rng.standard_normal((256, 256)).astype(np.float32)
    k, p_sea, cnt = cc._radial_power(f, 1.0 / 30.0, 30.0)
    _, p_up, _ = cc._radial_power((2.0 * f).astype(np.float32), 1.0 / 30.0, 30.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = p_up / p_sea          # 功率比恒 4（幅度 2×）
    fit = cc._continuation_fit(k, ratio, cnt)
    assert fit["frac_ratio_gt1"] > 0.9


def test_emag2_windows_deterministic():
    """确定性选窗：候选 ≤ N 全取；候选 > N 按纬度均布抽取（防地理偏聚）。"""
    valid = np.zeros((1024, 1024), dtype=bool)
    valid[256:512, 256:512] = True
    valid[256:512, 512:768] = True
    wins = cc._emag2_windows(valid, valid, window=256, n_windows=2)
    assert wins == [(256, 256), (256, 512)]       # 候选 = 2 ≤ N → 全取
    assert cc._emag2_windows(valid, valid, window=256, n_windows=5) == wins

    # 纬度均布：4 个候选（南→北），取 2 → 首末（最南 + 最北），非「前 2」
    v2 = np.zeros((1024, 1024), dtype=bool)
    for i0 in (0, 256, 512, 768):
        v2[i0:i0 + 256, 0:256] = True
    wins2 = cc._emag2_windows(v2, v2, window=256, n_windows=2)
    assert wins2 == [(0, 0), (768, 0)]


# ---------------- 三元组（合成 store 树） ----------------

def test_crosscheck_triple_synthetic():
    """恒等式精确成立的合成三层 → 残差 0 / r=1 / 斜率 1（只报告不拦截）。

    行 0/3 = 极区（|lat|≥60°），并构造重复计数机制：极区 base ≈ elev
    （基底已隐含沉积柱）→ 残差 = −sed，签名斜率 ≈ −1。
    """
    lat = np.array([-75.0, -25.0, 25.0, 75.0])
    lon = np.array([-150.0, -90.0, -30.0, 30.0, 90.0, 150.0])
    landsea = np.array([[1, 1, 1, 0, 0, 0]] * 4, dtype=np.uint8)
    sed = np.array([[100.0, 500.0, 2000.0, 3000.0, 500.0, 0.0]] * 4, dtype=np.float32)
    base = np.array([[-0.5, -1.0, -2.0, -4.0, -3.0, -1.5]] * 4, dtype=np.float32)
    elev = (base.astype(np.float64) * 1000.0 + sed).astype(np.float32)
    # 极区两行改成重复计数形态：base 直接贴 elev（隐含沉积），resid = −sed
    base[0] = (elev[0] / 1000.0).astype(np.float32)
    base[3] = (elev[3] / 1000.0).astype(np.float32)
    tree = xr.DataTree.from_dict({
        "/1deg": xr.Dataset(
            {
                "topography__bedrock_elevation": (("lat", "lon"), elev),
                "lithosphere__gemma_basement_depth": (("lat", "lon"), base),
                "sediment__gst1_thickness": (("lat", "lon"), sed),
                "derived__landsea_mask": (("lat", "lon"), landsea),
            },
            coords={"lat": lat, "lon": lon},
        ),
    })
    checks = cc.crosscheck_triple(tree)
    assert all(c["status"] == "report" for c in checks)
    by_name = {c["name"]: c for c in checks}
    # 五段统计 + 组说明 + 注记
    assert any("全球" in k for k in by_name)
    assert any("极区" in k for k in by_name)
    assert any("非极区" in k for k in by_name)
    # 非极区恒等式精确成立（构造行 1/2 未改）
    np_stats = [c for c in checks if "非极区" in c["name"]][0]["stats"]
    assert np_stats["rmse"] == pytest.approx(0.0, abs=1e-3)
    assert np_stats["ols_slope"] == pytest.approx(1.0)
    # 极区重复计数签名：resid = −sed → 斜率 ≈ −1、corr ≈ −1
    note = [c for c in checks if "已知偏差注记" in c["name"]][0]
    sig = note["polar_signature"]
    assert sig["resid_on_sed_slope"] == pytest.approx(-1.0, abs=1e-6)
    assert sig["resid_on_sed_corr"] == pytest.approx(-1.0, abs=1e-6)
    assert "重复计数" in note["summary"]
    assert "频段错配" in note["summary"]


# ---------------- 保真汇总 ----------------

def _write_json(path: Path, obj) -> Path:
    path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    return path


def test_fidelity_summary(tmp_path):
    layers = """
  - id: alpha
    source: test
    native_res: 1deg
    home_tier: 1deg
    tiers: [1deg]
    dtype: float32
    unit: m
    resampling: none
    mask: validity
  - id: beta
    source: test
    native_res: 1deg
    home_tier: 1deg
    tiers: [1deg]
    dtype: float32
    unit: m
    resampling: none
    mask: validity
""" + _PIXEL_AREA_LAYER
    contract = load_contract(_mapping(tmp_path, layers))
    reports = tmp_path / "reports"
    reports.mkdir()
    _write_json(reports / "fidelity-alpha.json", {
        "layer": "alpha", "result": "PASS", "n_checks": 3,
        "checks": [
            {"name": "1deg: 积分量守恒（相对源偏差 < 0.1%）", "status": "pass"},
            {"name": "1deg: 分位数偏移（报告制）", "status": "report"},
            {"name": "3min: 抽样 120 像元 vs 独立航空公式一致", "status": "pass"},
        ],
    })
    _write_json(reports / "fidelity-beta.json", {
        "layer": "beta", "result": "FAIL", "n_checks": 2,
        "checks": [
            {"name": "1deg: 与源核重算逐位一致（float32 位级）", "status": "FAIL",
             "summary": "不一致像元 3（期望 0）"},
            {"name": "1deg: 类别众数一致性", "status": "pass"},
        ],
    })
    store = tmp_path / "store.zarr"
    store.mkdir()
    # 时效基准 = cube_manifest.json（构建终点标记）；zarr.json 是建库起点
    zj = store / "cube_manifest.json"
    zj.write_text("{}", encoding="utf-8")
    # alpha 报告早于最终构建 → stale；beta 晚于 → fresh
    old = zj.stat().st_mtime - 100
    os.utime(reports / "fidelity-alpha.json", (old, old))

    summary = ci.fidelity_summary(reports, contract, store)
    assert summary["result"] == "FAIL"                       # beta 超容差
    assert summary["n_reports"] == 2
    assert summary["per_layer"]["alpha"]["families"] == ["积分量", "分位数", "距离抽样"]
    fails = summary["per_layer"]["beta"]["fails"]
    assert len(fails) == 1 and "不一致像元 3" in fails[0]["summary"]
    cov = {c["name"]: c for c in summary["checks"]}
    assert cov["保真报告覆盖（契约层 ↔ fidelity 报告）"]["status"] == "pass"  # pixel_area 例外
    assert "alpha" in cov["保真报告时效性（vs 最终构建时间；报告制）"]["summary"]


def test_fidelity_summary_missing_report_fails(tmp_path):
    contract = load_contract(_mapping(tmp_path, _WGM_LAYER))
    reports = tmp_path / "reports"
    reports.mkdir()
    store = tmp_path / "store.zarr"
    store.mkdir()
    (store / "zarr.json").write_text("{}", encoding="utf-8")
    summary = ci.fidelity_summary(reports, contract, store)
    assert summary["result"] == "FAIL"
    cov = {c["name"]: c for c in summary["checks"]}
    assert cov["保真报告覆盖（契约层 ↔ fidelity 报告）"]["status"] == "FAIL"


def test_fidelity_summary_excludes_itself(tmp_path):
    """fidelity-summary.json 自身不得被 glob 计入逐层清单（自我包含防护）。"""
    contract = load_contract(_mapping(tmp_path, _WGM_LAYER))
    reports = tmp_path / "reports"
    reports.mkdir()
    _write_json(reports / "fidelity-gravity__wgm2012_bouguer.json", {
        "layer": "gravity__wgm2012_bouguer", "result": "PASS",
        "n_checks": 1, "checks": [{"name": "1deg: 逐位一致", "status": "pass"}],
    })
    _write_json(reports / "fidelity-summary.json", {"result": "PASS"})
    store = tmp_path / "store.zarr"
    store.mkdir()
    (store / "zarr.json").write_text("{}", encoding="utf-8")
    summary = ci.fidelity_summary(reports, contract, store)
    assert summary["n_reports"] == 1
    assert "summary" not in summary["per_layer"]


def test_crosscheck_none_src_dir_uses_default(tmp_path, monkeypatch):
    """integration 以 .get() 传 None → 交叉函数须落缺省路径（不 TypeError）。"""
    # CRUST1.0 合成源 + monkeypatch 缺省目录
    src = tmp_path / "crust1"
    src.mkdir()
    lines = []
    for i in range(180):
        la = 89.5 - i
        v = -(10.0 + (90.0 - la) / 3.0)
        for j in range(360):
            lines.append(f"{-179.5 + j:.1f}  {la:.1f}  {v:.4f}")
    (src / "depthtomoho.xyz").write_text("\n".join(lines) + "\n", encoding="ascii")
    monkeypatch.setattr(cc, "CRUST1_SRC_DIR", src)

    gemma = np.full((180, 360), -30.0, dtype=np.float32)
    tree = xr.DataTree.from_dict({
        "/1deg": xr.Dataset({
            "lithosphere__gemma_moho": (("lat", "lon"), gemma),
            "derived__landsea_mask": (
                ("lat", "lon"), np.ones((180, 360), dtype=np.uint8)),
        }),
    })
    checks = cc.crosscheck_gemma_vs_crust1(tree, None)   # None → 缺省路径
    assert len(checks) == 4
    assert all(c["status"] == "report" for c in checks)


# ---------------- 构建回报四项 ----------------

def _obligations_env(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir(exist_ok=True)
    _write_json(reports / "fidelity-thermal__heatflow_density.json", {
        "layer": "thermal__heatflow_density", "result": "PASS",
        "checks": [
            {"name": "整编核实: 坐标圆整 4 位小数后重合 == 53394", "status": "pass"},
            {"name": "整编核实结论（报告制）", "status": "report",
             "summary": "GHFDB 整编了 NGHF 主体但未完全（合成测试数据）"},
        ],
    })
    _write_json(reports / "slab2-merge.json", {
        "zones": 27, "covered_3min_pixels": 923250,
        "overlap_3min_pixels": 15241, "overlap_share_of_coverage": 0.0165,
    })
    _write_json(reports / "litho1-inventory.json", {
        "n_variables": 171,
        "ingested": {"lithosphere__litho1_lab": "asthenospheric_mantle_top_depth"},
        "n_v1_1_candidates": 170, "v1_1_note": "v1.1 候选（合成测试数据）",
    })
    contract = load_contract(_mapping(tmp_path, _PIXEL_AREA_LAYER))
    store = _write_store(tmp_path, contract, list(contract.layers))
    return reports, store


def test_build_obligations(tmp_path):
    reports, store = _obligations_env(tmp_path)
    ob = ci.build_obligations(reports, store)
    assert ob["result"] == "PASS"
    assert len(ob["checks"]) == 4
    assert all(c["status"] == "pass" for c in ob["checks"])
    assert "ghfdb_nghf_incorporation" in ob["items"]
    assert ob["items"]["slab2_overlap"]["overlap_3min_pixels"] == 15241
    assert ob["items"]["litho1_inventory"]["n_variables"] == 171
    assert ob["items"]["coverage_registry"]["n_layers"] == 1


def test_build_obligations_missing_artifact_fails(tmp_path):
    reports, store = _obligations_env(tmp_path)
    (reports / "slab2-merge.json").unlink()
    ob = ci.build_obligations(reports, store)
    assert ob["result"] == "FAIL"
    by_name = {c["name"]: c for c in ob["checks"]}
    assert by_name["回报② Slab2 重叠像元总量"]["status"] == "FAIL"
    assert by_name["回报① GHFDB/NGHF 整编核实结论"]["status"] == "pass"


# ---------------- 集成总报告（接线 + 判据组合） ----------------

def _stub_crosschecks(monkeypatch, fail_group=None):
    """stub 打在 crosscheck 模块（统一装配 assemble_crosscheck_groups 在
    该模块内调用这些函数名）。"""
    def _ok(group):
        def fn(*a, **k):
            return [{"name": f"{group}: 合成交叉", "status": "report", "summary": "ok"}]
        return fn
    for group, fname in (
        ("三元组", "crosscheck_triple"),
        ("GEMMA", "crosscheck_gemma_vs_crust1"),
        ("GST1", "crosscheck_gst1_vs_globsed"),
        ("EMAG2", "crosscheck_emag2_spectral"),
    ):
        if group == fail_group:
            def boom(*a, _g=group, **k):
                raise FileNotFoundError(f"{_g} 源缺失（合成）")
            monkeypatch.setattr(cc, fname, boom)
        else:
            monkeypatch.setattr(cc, fname, _ok(group))


def test_build_integration_report_pass(tmp_path, monkeypatch):
    reports, store = _obligations_env(tmp_path)
    contract = load_contract(_mapping(tmp_path, _PIXEL_AREA_LAYER))
    _stub_crosschecks(monkeypatch)
    report = ci.build_integration_report(store, contract, reports)
    assert report["result"] == "PASS"
    assert report["structure"]["result"] == "PASS"
    assert report["fidelity"]["result"] == "PASS"     # pixel_area 例外
    assert report["obligations"]["result"] == "PASS"
    assert report["crosscheck"]["n_groups"] == 4
    # 分报告落盘
    for name in ("integration-report.json", "structure-validation.json",
                 "fidelity-summary.json", "build-obligations.json",
                 "physical-crosscheck.json"):
        assert (reports / name).is_file(), name


def test_build_integration_report_judgement_combination(tmp_path, monkeypatch):
    """判据组合：回报缺失 / 交叉组执行失败任一 → 总 FAIL。"""
    reports, store = _obligations_env(tmp_path)
    contract = load_contract(_mapping(tmp_path, _PIXEL_AREA_LAYER))

    _stub_crosschecks(monkeypatch)
    (reports / "litho1-inventory.json").unlink()
    report = ci.build_integration_report(store, contract, reports)
    assert report["result"] == "FAIL"
    assert report["obligations"]["result"] == "FAIL"

    (reports / "litho1-inventory.json").write_text(
        json.dumps({"n_variables": 171,
                    "ingested": {"lithosphere__litho1_lab": "x"},
                    "n_v1_1_candidates": 170}), encoding="utf-8")
    _stub_crosschecks(monkeypatch, fail_group="EMAG2")
    report = ci.build_integration_report(store, contract, reports)
    assert report["result"] == "FAIL"                 # 交叉组执行失败计 FAIL


def test_integrate_cli(tmp_path, monkeypatch):
    from cubebuild.cli import main

    reports, store = _obligations_env(tmp_path)
    mapping = _mapping(tmp_path, _PIXEL_AREA_LAYER)
    _stub_crosschecks(monkeypatch)
    rc = main(["integrate", "--store", str(store), "--mapping", str(mapping),
               "--reports", str(reports)])
    assert rc == 0
    rep = json.loads((reports / "integration-report.json").read_text())
    assert rep["result"] == "PASS"


# ---------------- 真实源集成（存在时执行） ----------------

@pytest.mark.skipif(
    not (REPO_ROOT / "products/cube-v1.0.zarr").is_dir()
    or not (REPO_ROOT / "original data/lithosphere/crust-models/crust1.0").is_dir()
    or not (REPO_ROOT / "original data/sediment/globsed").is_dir()
    or not (REPO_ROOT / "original data/magnetics/emag2-v3").is_dir(),
    reason="真实 store 或验证参考源不存在",
)
def test_crosscheck_real_store():
    """四组物理交叉对真实 store 出数值报告（只验证可出且全 report 制）。"""
    report = cc.build_crosscheck_report(REPO_ROOT / "products/cube-v1.0.zarr")
    assert report["n_groups"] == 4
    for g in report["groups"]:
        assert g["checks"], g["group"]
        assert all(c["status"] == "report" for c in g["checks"])
    # 数值合理性抽样：三元组全球 n = 64800；EMAG2 拟合有限
    triple = report["groups"][0]["checks"]
    stats = [c["stats"] for c in triple if "stats" in c][0]
    assert stats["n"] == 64800
    emag = report["groups"][3]["checks"]
    fit = [c for c in emag if "mean_delta_z_km" in c][0]
    assert np.isfinite(fit["mean_delta_z_km"]) and fit["mean_delta_z_km"] > 0
