"""Public-exporter tests: licence-status branching rules + internal stripping +
public-store numeric fidelity + manifest gap notes + sidecar copying + dry-run
writes nothing.

The synthetic store contains a 2D public layer (NaN-value fidelity), an internal
layer (with mask), a 3D layer group (group-node stripping path) and pixel_area
analytic values (numeric check item for structural validation).
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
    """Synthesize a full store: five-layer arrays (including a 3D group node and
    NaN) + manifest + sidecar files."""
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



def test_strip_plan_internal(tmp_path):
    """internal layer + mask + sidecar stripped; gap notes carry the dataset /
    reason / reader guidance."""
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
    assert gaps["gem-active-faults"]["layers"] == []
    assert gaps["gem-active-faults"]["sidecars"] == ["gem_active_faults"]


def test_strip_plan_t1_branches(tmp_path):
    """Branch one: confirmed → keep; branch two: rejected / silent_expired →
    exit + catalog action list."""
    contract, store, _ = _fixture(tmp_path)
    state = load_t1_state(_t1_state(tmp_path, """
silent_deadline: "待定"
datasets:
  pb2002-plate-boundaries: {catalog_id: 10, status: confirmed, note: 确认开放}
  gst1: {catalog_id: 13, status: rejected, note: 拒复}
  semucb-wm1: {catalog_id: 5, status: silent_expired, note: 静默}
"""))
    plan = build_strip_plan(contract, _MINI_SIDECARS, state)
    assert "stress_kinematics__pb2002_plate_id" not in plan["strip_layers"]
    assert "pb2002_plates" in plan["kept_sidecar_ids"]
    for lid in ("sediment__gst1_thickness", "seismology__semucb_vs"):
        assert lid in plan["strip_layers"]
        assert f"{lid}__validity" in plan["strip_masks"]
    actions = {a["dataset_dir"]: a for a in plan["catalog_actions"]}
    assert set(actions) == {"gst1", "semucb-wm1"}
    assert actions["gst1"]["catalog_id"] == 13
    assert "out-of-scope" in actions["gst1"]["action"]
    gaps = {g["dataset"]: g for g in plan["gaps"]}
    assert gaps["gst1"]["kind"] == "t1_rejected"
    assert gaps["semucb-wm1"]["kind"] == "t1_silent_expired"


def test_strip_plan_exit_layers_override(tmp_path):
    """exit_layers/exit_sidecars partial exit: non-exited items are kept and
    marked needs_hitl."""
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
    """A contract layer missing a dataset mapping → explicit failure (prevents a
    new layer from silently bypassing the gate)."""
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
    """Invalid status value / missing catalog_id / unknown dataset / empty file
    → explicit failure."""
    with pytest.raises(ValueError, match="非法 status"):
        load_t1_state(_t1_state(tmp_path, "datasets:\n  gst1: {catalog_id: 13, status: bogus}"))
    with pytest.raises(ValueError, match="catalog_id"):
        load_t1_state(_t1_state(tmp_path, "datasets:\n  gst1: {status: pending}"))
    with pytest.raises(ValueError, match="无层/侧车归属"):
        load_t1_state(_t1_state(
            tmp_path, "datasets:\n  no-such-dataset: {catalog_id: 1, status: pending}"))
    empty = tmp_path / "empty.yaml"
    empty.write_text("", encoding="utf-8")
    state = load_t1_state(empty)
    assert state["datasets"] == {}


def test_export_public_path_overlap_guard(tmp_path):
    """Output path overlaps the source (out == store / out inside store) → caught
    up front, so overwrite-style writing never deletes the source (guard against
    an unrecoverable incident)."""
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
    assert store.exists()



def _nan_aware_equal(a, b) -> bool:
    return bool(np.array_equal(a, b, equal_nan=True))


def test_write_public_store_fidelity(tmp_path):
    """Public store: internal layers + masks disappear, an emptied 3D group is
    deleted, public-layer values are NaN-aware bitwise identical, and root attrs
    mark the public edition."""
    contract, store, state = _fixture(tmp_path)
    plan = build_strip_plan(contract, _MINI_SIDECARS, state)
    strip = set(plan["strip_layers"]) | set(plan["strip_masks"])
    out = write_public_store(store, strip, tmp_path / "pub.zarr", contract)
    with xr.open_datatree(store, engine="zarr", chunks={}) as full, \
            xr.open_datatree(out, engine="zarr", chunks={}) as pub:
        assert "gravity__wgm2012_bouguer" not in pub["/1deg"].ds.data_vars
        assert "gravity__wgm2012_bouguer__validity" not in pub["/1deg"].ds.data_vars
        assert "semucb" in pub["/1deg"].children
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
    plan2 = build_strip_plan(contract, _MINI_SIDECARS, load_t1_state(_t1_state(
        tmp_path, "datasets:\n  semucb-wm1: {catalog_id: 5, status: rejected, note: x}")))
    strip2 = set(plan2["strip_layers"]) | set(plan2["strip_masks"])
    out2 = write_public_store(store, strip2, tmp_path / "pub2.zarr", contract)
    with xr.open_datatree(out2, engine="zarr", chunks={}) as pub2:
        assert "semucb" not in pub2["/1deg"].children


def test_public_manifest_gaps(tmp_path):
    """Public manifest: layers / sidecars stripped in sync + public_export gap
    section + notes for datasets pending confirmation."""
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
    assert pe["t1_pending"]
    assert "待定" in pe["t1_note"]


def test_copy_public_sidecars(tmp_path):
    """Public sidecar copying: byte-level complete, internal sidecars absent from
    the public directory."""
    contract, store, state = _fixture(tmp_path)
    kept = [s for s in _MINI_SIDECARS if s["id"] == "pb2002_plates"]
    out_dir = tmp_path / "sidecars-public"
    copied = copy_public_sidecars(tmp_path / "sidecars", kept, out_dir)
    assert [c["id"] for c in copied] == ["pb2002_plates"]
    assert (out_dir / "pb2002" / "plates.parquet").read_bytes() == b"pb2002" * 100
    assert not (out_dir / "gem.parquet").exists()
    assert tree_stats(out_dir / "pb2002") == {"n_files": 1, "n_bytes": 600}



def test_export_public_validation_pass(tmp_path):
    """The public candidate independently passes structural validation
    (post-strip contract subset × public store × public manifest)."""
    contract, store, state = _fixture(tmp_path)
    report = export_public(
        store=store, contract=contract, sidecars_dir=tmp_path / "sidecars",
        out=tmp_path / "pub.zarr", out_sidecars=tmp_path / "sidecars-public",
        t1_state_path=_t1_state(tmp_path, _T1_ALL_PENDING),
        reports_dir=tmp_path / "reports",
    )
    assert report["result"] == "PASS", report["validation"]
    assert report["validation"]["total_checks"] > 0
    pub_manifest = read_manifest(tmp_path / "pub.zarr")
    assert len(pub_manifest["layers"]) == 4
    rep = json.loads((tmp_path / "reports" / "public-export-report.json")
                     .read_text(encoding="utf-8"))
    assert rep["plan"]["n_layers_public"] == 4
    for part in ("full", "public"):
        for prod in ("store", "sidecars"):
            assert rep["sizes"][part][prod]["n_bytes"] > 0


def test_export_public_dry_run_writes_nothing(tmp_path):
    """dry-run: emits only the plan / decisions / estimates, writes no
    artifact."""
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
    assert report["sizes"]["public"]["store"]["estimated"] is True
    assert (report["sizes"]["public"]["store"]["n_bytes"]
            < report["sizes"]["full"]["store"]["n_bytes"])


def test_packaging_recommendation(tmp_path):
    """Packaging advice: exceeding the file-count threshold → single package
    required; exceeding the byte threshold → split-volume advice."""
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
    """Same-named arrays accumulated across tiers (regression: a dict overwrite
    left a multi-tier layer with only one tier's volume) + the chunk subtree
    fully counted (regression: reading direct children shows only
    zarr.json)."""
    store = tmp_path / "s.zarr"
    for tier, size in (("1deg", 100), ("30sec", 400)):
        d = store / tier / "some__layer" / "c" / "0"
        d.mkdir(parents=True)
        (d / "chunk").write_bytes(b"x" * size)
        (store / tier / "some__layer" / "zarr.json").write_text("{}")
    arr = _array_dir_stats(store)
    assert arr["some__layer"]["n_bytes"] == 500 + 2 * 2
    assert arr["some__layer"]["n_files"] == 4



def test_real_contract_mapping_complete():
    """Real contract (layer-mapping-v0.yaml): every layer + manifest sidecar has
    a dataset mapping, so the public gate has no silent bypass."""
    mapping = DEFAULT_CONTRACT_PATH
    contract = load_contract(mapping)
    unmapped = [e.id for e in contract.layers if e.id not in LAYER_DATASET]
    assert not unmapped
    internal_ds = {LAYER_DATASET[e.id] for e in contract.layers
                   if e.visibility == "internal"}
    assert internal_ds <= {
        "bouguer-wgm2012", "gsrm-strain", "gem-active-faults", "global-basins",
        "glad-m35", "semucb-wm1", "pb2002-plate-boundaries", "gst1",
        "heatflow", "glim-lithology",
    }
    state = load_t1_state(None)
    assert state["datasets"] == {}
