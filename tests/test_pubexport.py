"""公开导出器测试：许可证状态分支规则 + internal 剥离 + 公开 store 数值保真
+ manifest 缺口注记 + 侧车复制 + dry-run 不落盘。

合成 store 含 2D 公开层（NaN 值保真）、internal 层（带掩膜）、3D 层组
（组节点剥离路径）与 pixel_area 解析值（结构验证数值核对项）。
"""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from cubebuild.contract import DEFAULT_CONTRACT_PATH, load_contract
from cubebuild.grids import tier_centers
from cubebuild.manifest import build_manifest, read_manifest, write_manifest
from cubebuild.masks import validity_layer_name
from cubebuild.pubexport import (
    LAYER_DATASET,
    _array_dir_stats,
    build_public_manifest,
    build_strip_plan,
    copy_public_sidecars,
    export_public,
    load_t1_state,
    packaging_recommendation,
    tree_stats,
    write_public_store,
)
from cubebuild.validate import validate_store

# ---------------- 合成契约 / store ----------------

_LAYERS = """
  - id: derived__pixel_area
    source: 纬度解析公式
    native_res: exact
    home_tier: 逐档
    tiers: [1deg]
    dtype: float32
    unit: km2
    resampling: exact_formula
    mask: 免
  - id: sediment__gst1_thickness
    source: gst1
    native_res: 0.125deg
    home_tier: 1deg
    tiers: [1deg]
    dtype: float32
    unit: m
    resampling: conservative_area_weighted
    mask: validity
  - id: gravity__wgm2012_bouguer
    source: bouguer-wgm2012
    native_res: 2min
    home_tier: 1deg
    tiers: [1deg]
    dtype: float32
    unit: mGal
    resampling: none
    mask: validity
    visibility: internal
  - id: stress_kinematics__pb2002_plate_id
    source: pb2002-plate-boundaries
    native_res: vector-polygon
    home_tier: 1deg
    tiers: [1deg]
    dtype: uint8
    unit: category
    resampling: mode
    mask: 免
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
"""

_MINI_SIDECARS = [
    {"id": "gem_active_faults", "format": "geoparquet",
     "path": "sidecars/gem.parquet", "visibility": "internal"},
    {"id": "pb2002_plates", "format": "geoparquet",
     "path": "sidecars/pb2002", "visibility": "public"},
]


def _mapping(tmp_path, layers: str = _LAYERS) -> Path:
    p = tmp_path / "mini.yaml"
    p.write_text(
        f"version: test\nstatus: test\ntiers: [1deg]\nlayers:\n{layers}",
        encoding="utf-8",
    )
    return p


def _write_full_store(tmp_path, contract) -> Path:
    """合成完整 store：五层数组（含 3D 组节点与 NaN）+ manifest + 侧车文件。"""
    from cubebuild.pixel_area import pixel_area_rows

    lat, lon = tier_centers("1deg")
    entries = contract.layers
    tier_vars = {
        e.id: xr.DataArray(
            (
                np.repeat(pixel_area_rows("1deg")[:, None].astype(np.float32),
                          360, axis=1)
                if e.id == "derived__pixel_area" else
                np.where(
                    (np.arange(180)[:, None] + np.arange(360)[None, :]) % 7 == 0,
                    np.nan, (np.arange(180)[:, None] * 360 + np.arange(360)[None, :]),
                ).astype(np.float32) if e.dtype == "float32" else
                ((np.arange(180)[:, None] * 360 + np.arange(360)[None, :]) % 52 + 1
                 ).astype(np.uint8)
            ),
            dims=("lat", "lon"), coords={"lat": lat, "lon": lon},
            attrs={"units": e.unit},
        )
        for e in entries if "depth" not in e.dims
    }
    for e in entries:
        if "validity" in e.mask and "depth" not in e.dims:
            tier_vars[validity_layer_name(e.id)] = xr.DataArray(
                ((np.arange(180)[:, None] + np.arange(360)[None, :]) % 7 != 0
                 ).astype(np.uint8),
                dims=("lat", "lon"), coords={"lat": lat, "lon": lon},
            )
    depth = xr.DataArray(
        np.array([10.0, 20.0], dtype=np.float32), dims="depth",
        attrs={"units": "km", "positive": "down"},
    )
    sem = [e for e in entries if e.id == "seismology__semucb_vs"][0]
    sem_arr = xr.DataArray(
        np.random.default_rng(18).random((2, 180, 360)).astype(np.float32),
        dims=("depth", "lat", "lon"),
        coords={"depth": depth, "lat": lat, "lon": lon},
        attrs={"units": sem.unit},
    )
    sem_mask = xr.DataArray(
        np.ones((2, 180, 360), dtype=np.uint8),
        dims=("depth", "lat", "lon"),
        coords={"depth": depth, "lat": lat, "lon": lon},
    )
    tree = xr.DataTree.from_dict({
        "/": xr.Dataset(attrs={"crs": "EPSG:4326"}),
        "/1deg": xr.Dataset(dict(tier_vars)),
        "/1deg/semucb": xr.Dataset({
            "seismology__semucb_vs": sem_arr,
            "seismology__semucb_vs__validity": sem_mask,
        }),
    })
    out = tmp_path / "cube.zarr"
    tree.to_zarr(out, mode="w", zarr_format=3)

    coverage = {(e.id, t): 0.5 for e in entries for t in e.tiers}
    manifest = build_manifest(
        contract, list(entries), coverage,
        extras={"seismology__semucb_vs": {
            "datatree_node": "/1deg/semucb", "depth_levels": 2,
        }},
        sidecars=_MINI_SIDECARS,
    )
    write_manifest(manifest, out)

    sc_dir = tmp_path / "sidecars"
    (sc_dir / "pb2002").mkdir(parents=True)
    (sc_dir / "pb2002" / "plates.parquet").write_bytes(b"pb2002" * 100)
    (sc_dir / "gem.parquet").write_bytes(b"gem" * 100)
    return out


_T1_ALL_PENDING = """
silent_deadline: "待定"
datasets:
  litho1: {catalog_id: 20, status: confirmed, note: 邮件确认}
  pb2002-plate-boundaries: {catalog_id: 10, status: pending, note: 已发}
  gst1: {catalog_id: 13, status: pending, note: 已发}
  semucb-wm1: {catalog_id: 5, status: pending, note: 已发}
"""


def _t1_state(tmp_path, text: str) -> Path:
    p = tmp_path / "t1.yaml"
    p.write_text(text, encoding="utf-8")
    return p


def _fixture(tmp_path):
    contract = load_contract(_mapping(tmp_path))
    store = _write_full_store(tmp_path, contract)
    state = load_t1_state(_t1_state(tmp_path, _T1_ALL_PENDING))
    return contract, store, state


# ---------------- 剥离计划 ----------------

def test_strip_plan_internal(tmp_path):
    """internal 层+掩膜+侧车剥离；缺口注记含数据集/原因/读者指引。"""
    contract, store, state = _fixture(tmp_path)
    plan = build_strip_plan(contract, _MINI_SIDECARS, state)
    assert plan["strip_layers"] == ["gravity__wgm2012_bouguer"]
    assert plan["strip_masks"] == ["gravity__wgm2012_bouguer__validity"]
    assert plan["strip_sidecars"] == ["gem_active_faults"]
    assert plan["n_layers_public"] == 4
    gaps = {g["dataset"]: g for g in plan["gaps"]}
    assert "bouguer-wgm2012" in gaps
    assert "BGI" in gaps["bouguer-wgm2012"]["reason"]
    assert "论文建议读者" in gaps["bouguer-wgm2012"]["guidance"]
    # 仅侧车 internal 的数据集（无 internal 层）同样入缺口注记（禁止无名缺席）
    assert gaps["gem-active-faults"]["layers"] == []
    assert gaps["gem-active-faults"]["sidecars"] == ["gem_active_faults"]


def test_strip_plan_t1_branches(tmp_path):
    """分支一：confirmed → 维持；分支二：rejected / silent_expired → 退出
    + catalog 动作清单。"""
    contract, store, _ = _fixture(tmp_path)
    state = load_t1_state(_t1_state(tmp_path, """
silent_deadline: "待定"
datasets:
  pb2002-plate-boundaries: {catalog_id: 10, status: confirmed, note: 确认开放}
  gst1: {catalog_id: 13, status: rejected, note: 拒复}
  semucb-wm1: {catalog_id: 5, status: silent_expired, note: 静默}
"""))
    plan = build_strip_plan(contract, _MINI_SIDECARS, state)
    # confirmed 维持
    assert "stress_kinematics__pb2002_plate_id" not in plan["strip_layers"]
    assert "pb2002_plates" in plan["kept_sidecar_ids"]
    # rejected / silent_expired 退出（层+掩膜）
    for lid in ("sediment__gst1_thickness", "seismology__semucb_vs"):
        assert lid in plan["strip_layers"]
        assert f"{lid}__validity" in plan["strip_masks"]
    # catalog 动作清单（人工执行，不自动落盘）
    actions = {a["dataset_dir"]: a for a in plan["catalog_actions"]}
    assert set(actions) == {"gst1", "semucb-wm1"}
    assert actions["gst1"]["catalog_id"] == 13
    assert "out-of-scope" in actions["gst1"]["action"]
    gaps = {g["dataset"]: g for g in plan["gaps"]}
    assert gaps["gst1"]["kind"] == "t1_rejected"
    assert gaps["semucb-wm1"]["kind"] == "t1_silent_expired"


def test_strip_plan_exit_layers_override(tmp_path):
    """exit_layers/exit_sidecars 部分退出：未退出项保留并记 needs_hitl。"""
    contract, store, _ = _fixture(tmp_path)
    state = load_t1_state(_t1_state(tmp_path, """
silent_deadline: "待定"
datasets:
  gst1: {catalog_id: 13, status: rejected, note: 拒复,
         exit_layers: [], exit_sidecars: []}
"""))
    plan = build_strip_plan(contract, _MINI_SIDECARS, state)
    assert "sediment__gst1_thickness" not in plan["strip_layers"]
    assert plan["partial_exits"] == [{
        "dataset": "gst1", "status": "rejected",
        "kept_layers": ["sediment__gst1_thickness"], "kept_sidecars": [],
        "note": plan["partial_exits"][0]["note"],
    }]


def test_strip_plan_mapping_coverage(tmp_path):
    """契约层缺数据集映射 → 显式失败（防新层静默绕过门禁）。"""
    p = _mapping(tmp_path, layers="""
  - id: unknown__layer
    source: x
    native_res: 1deg
    home_tier: 1deg
    tiers: [1deg]
    dtype: float32
    unit: m
    resampling: none
    mask: 免
""")
    contract = load_contract(p)
    with pytest.raises(ValueError, match="LAYER_DATASET"):
        build_strip_plan(contract, [], load_t1_state(_t1_state(
            tmp_path, "datasets: {}")))


def test_load_t1_state_validates(tmp_path):
    """非法状态值 / 缺 catalog_id / 未知数据集 / 空文件 → 显式失败。"""
    with pytest.raises(ValueError, match="非法 status"):
        load_t1_state(_t1_state(tmp_path, "datasets:\n  gst1: {catalog_id: 13, status: bogus}"))
    with pytest.raises(ValueError, match="catalog_id"):
        load_t1_state(_t1_state(tmp_path, "datasets:\n  gst1: {status: pending}"))
    with pytest.raises(ValueError, match="无层/侧车归属"):
        load_t1_state(_t1_state(
            tmp_path, "datasets:\n  no-such-dataset: {catalog_id: 1, status: pending}"))
    empty = tmp_path / "empty.yaml"
    empty.write_text("", encoding="utf-8")     # 空文件（safe_load → None）不崩
    state = load_t1_state(empty)
    assert state["datasets"] == {}


def test_export_public_path_overlap_guard(tmp_path):
    """输出路径与源重叠（out == store / out 在 store 内）→ 前置拦截，
    覆盖式写入不会删源（不可恢复事故防线）。"""
    contract, store, state = _fixture(tmp_path)
    with pytest.raises(ValueError, match="重叠"):
        export_public(
            store=store, contract=contract, sidecars_dir=tmp_path / "sidecars",
            out=store, out_sidecars=tmp_path / "sc-out",
            t1_state_path=_t1_state(tmp_path, _T1_ALL_PENDING),
            reports_dir=tmp_path / "reports", dry_run=True,
        )
    with pytest.raises(ValueError, match="重叠"):
        export_public(
            store=store, contract=contract, sidecars_dir=tmp_path / "sidecars",
            out=store / "nested", out_sidecars=tmp_path / "sc-out",
            t1_state_path=_t1_state(tmp_path, _T1_ALL_PENDING),
            reports_dir=tmp_path / "reports", dry_run=True,
        )
    assert store.exists()             # 源未被触碰


# ---------------- 公开 store / manifest / 侧车 ----------------

def _nan_aware_equal(a, b) -> bool:
    return bool(np.array_equal(a, b, equal_nan=True))


def test_write_public_store_fidelity(tmp_path):
    """公开 store：internal 层+掩膜消失、3D 组剥空即删、公开层数值 NaN 感知
    逐位一致、根 attrs 标注公开版。"""
    contract, store, state = _fixture(tmp_path)
    plan = build_strip_plan(contract, _MINI_SIDECARS, state)
    strip = set(plan["strip_layers"]) | set(plan["strip_masks"])
    out = write_public_store(store, strip, tmp_path / "pub.zarr", contract)
    with xr.open_datatree(store, engine="zarr", chunks={}) as full, \
            xr.open_datatree(out, engine="zarr", chunks={}) as pub:
        assert "gravity__wgm2012_bouguer" not in pub["/1deg"].ds.data_vars
        assert "gravity__wgm2012_bouguer__validity" not in pub["/1deg"].ds.data_vars
        assert "semucb" in pub["/1deg"].children          # 公开 3D 组保留
        for name in ("sediment__gst1_thickness", "derived__pixel_area",
                     "stress_kinematics__pb2002_plate_id"):
            assert _nan_aware_equal(
                pub["/1deg"][name].values, full["/1deg"][name].values
            ), f"{name} 数值漂移"
        assert _nan_aware_equal(
            pub["/1deg/semucb"]["seismology__semucb_vs"].values,
            full["/1deg/semucb"]["seismology__semucb_vs"].values,
        )
        assert pub.attrs["edition"] == "public"
        assert "public_export" in pub.attrs
    # 剥空组删除（semucb 全剥离时）
    plan2 = build_strip_plan(contract, _MINI_SIDECARS, load_t1_state(_t1_state(
        tmp_path, "datasets:\n  semucb-wm1: {catalog_id: 5, status: rejected, note: x}")))
    strip2 = set(plan2["strip_layers"]) | set(plan2["strip_masks"])
    out2 = write_public_store(store, strip2, tmp_path / "pub2.zarr", contract)
    with xr.open_datatree(out2, engine="zarr", chunks={}) as pub2:
        assert "semucb" not in pub2["/1deg"].children


def test_public_manifest_gaps(tmp_path):
    """公开 manifest：层/侧车同步剥离 + public_export 缺口节 + 待确认数据集注记。"""
    contract, store, state = _fixture(tmp_path)
    manifest = read_manifest(store)
    plan = build_strip_plan(contract, manifest["sidecars"], state)
    pub = build_public_manifest(manifest, plan)
    ids = {l["id"] for l in pub["layers"]}
    assert "gravity__wgm2012_bouguer" not in ids
    assert {s["id"] for s in pub["sidecars"]} == {"pb2002_plates"}
    pe = pub["public_export"]
    assert {g["dataset"] for g in pe["gaps"]} == {
        "bouguer-wgm2012", "gem-active-faults",
    }
    assert pe["t1_pending"]            # gst1/semucb/pb2002 pending 注记
    assert "待定" in pe["t1_note"]


def test_copy_public_sidecars(tmp_path):
    """公开侧车复制：字节级完整、internal 侧车不出现在公开目录。"""
    contract, store, state = _fixture(tmp_path)
    kept = [s for s in _MINI_SIDECARS if s["id"] == "pb2002_plates"]
    out_dir = tmp_path / "sidecars-public"
    copied = copy_public_sidecars(tmp_path / "sidecars", kept, out_dir)
    assert [c["id"] for c in copied] == ["pb2002_plates"]
    assert (out_dir / "pb2002" / "plates.parquet").read_bytes() == b"pb2002" * 100
    assert not (out_dir / "gem.parquet").exists()
    assert tree_stats(out_dir / "pb2002") == {"n_files": 1, "n_bytes": 600}


# ---------------- 编排（含独立结构验证与 dry-run） ----------------

def test_export_public_validation_pass(tmp_path):
    """公开版候选独立通过结构验证（剥离后契约子集 × 公开 store × 公开 manifest）。"""
    contract, store, state = _fixture(tmp_path)
    report = export_public(
        store=store, contract=contract, sidecars_dir=tmp_path / "sidecars",
        out=tmp_path / "pub.zarr", out_sidecars=tmp_path / "sidecars-public",
        t1_state_path=_t1_state(tmp_path, _T1_ALL_PENDING),
        reports_dir=tmp_path / "reports",
    )
    assert report["result"] == "PASS", report["validation"]
    assert report["validation"]["total_checks"] > 0
    # 公开 store 的 manifest 即剥离版（验证三方一致的 manifest 来源）
    pub_manifest = read_manifest(tmp_path / "pub.zarr")
    assert len(pub_manifest["layers"]) == 4
    # 报告落盘
    rep = json.loads((tmp_path / "reports" / "public-export-report.json")
                     .read_text(encoding="utf-8"))
    assert rep["plan"]["n_layers_public"] == 4
    # 体积核实字段在位
    for part in ("full", "public"):
        for prod in ("store", "sidecars"):
            assert rep["sizes"][part][prod]["n_bytes"] > 0


def test_export_public_dry_run_writes_nothing(tmp_path):
    """dry-run：只出计划/决策/估算，不写任何产物。"""
    contract, store, state = _fixture(tmp_path)
    out = tmp_path / "pub.zarr"
    out_sc = tmp_path / "sidecars-public"
    report = export_public(
        store=store, contract=contract, sidecars_dir=tmp_path / "sidecars",
        out=out, out_sidecars=out_sc,
        t1_state_path=_t1_state(tmp_path, _T1_ALL_PENDING),
        reports_dir=tmp_path / "reports", dry_run=True,
    )
    assert report["result"] == "DRY-RUN"
    assert not out.exists() and not out_sc.exists()
    assert not (tmp_path / "reports").exists()
    # 估算 = 完整 store 字节 − 剥离数组磁盘量
    assert report["sizes"]["public"]["store"]["estimated"] is True
    assert (report["sizes"]["public"]["store"]["n_bytes"]
            < report["sizes"]["full"]["store"]["n_bytes"])


def test_packaging_recommendation(tmp_path):
    """打包建议：超文件数门槛 → 必须整包；超字节门槛 → 分卷建议。"""
    sizes = {
        "full": {"store": {"n_files": 10, "n_bytes": 10},
                 "sidecars": {"n_files": 1, "n_bytes": 1}},
        "public": {"store": {"n_files": 200, "n_bytes": 10},
                   "sidecars": {"n_files": 1, "n_bytes": 1}},
    }
    rec = packaging_recommendation(sizes)
    assert any("远超" in r for r in rec["rationale"])
    sizes["public"]["store"]["n_bytes"] = 60_000_000_000
    rec = packaging_recommendation(sizes)
    assert any("分卷" in r or "配额" in r for r in rec["rationale"])
    assert rec["zenodo_limits"]["max_files_per_record"] == 100


def test_array_dir_stats_cross_tier_accumulation(tmp_path):
    """同名数组跨档位累加（回归：字典覆盖使多档层只剩一档的量）+ chunk
    子树全量计入（回归：直读直接子文件只见 zarr.json）。"""
    store = tmp_path / "s.zarr"
    for tier, size in (("1deg", 100), ("30sec", 400)):
        d = store / tier / "some__layer" / "c" / "0"
        d.mkdir(parents=True)
        (d / "chunk").write_bytes(b"x" * size)
        (store / tier / "some__layer" / "zarr.json").write_text("{}")
    arr = _array_dir_stats(store)
    assert arr["some__layer"]["n_bytes"] == 500 + 2 * 2   # 两档 chunk + 两份 zarr.json
    assert arr["some__layer"]["n_files"] == 4


# ---------------- 契约真实映射完备性（防漂移） ----------------

def test_real_contract_mapping_complete():
    """真实契约（layer-mapping-v0.yaml）全部层 + manifest 侧车有数据集映射：
    公开门禁无静默绕行。"""
    mapping = DEFAULT_CONTRACT_PATH
    contract = load_contract(mapping)
    unmapped = [e.id for e in contract.layers if e.id not in LAYER_DATASET]
    assert not unmapped
    # internal 数据集全集 = 既有 4（WGM2012/GSRM/
    # GEM 断层/Global Basins）+ 六项待确认再分发许可的数据集
    # （GLAD-M35/SEMUCB/PB2002/GST-1/heatflow 部分退出/glim 仅 LiMW 侧车）
    internal_ds = {LAYER_DATASET[e.id] for e in contract.layers
                   if e.visibility == "internal"}
    assert internal_ds <= {
        "bouguer-wgm2012", "gsrm-strain", "gem-active-faults", "global-basins",
        "glad-m35", "semucb-wm1", "pb2002-plate-boundaries", "gst1",
        "heatflow", "glim-lithology",
    }
    # 门禁停用：t1_state_path=None 空置分支可用
    state = load_t1_state(None)
    assert state["datasets"] == {}
