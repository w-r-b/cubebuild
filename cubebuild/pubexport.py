"""Public-edition exporter: strip by contract visibility + gap notes +
packaging verification.

Mechanism:

- Public candidate = full store − visibility=internal layers (including the
  ``__validity`` companion masks) − internal sidecars. Artifacts: public store
  + public sidecar directory + stripped cube_manifest (gap notes: which
  dataset is absent for which license status) + machine-readable export
  report.
- Status branches (gate input = license confirmation status YAML, maintained
  manually):
    confirmed                → layer stays public;
    pending                  → layer stays a public candidate (confirmation wait
                               period up to silent_deadline);
    rejected / silent_expired → layer exits the public edition + catalog marked
                       out-of-scope (catalog sync is a manual step; only a
                       catalog_actions action list is produced, not written to
                       disk automatically inside the exporter);
    exit_layers / exit_sidecars overrides → partial exit (e.g. the merged
    heat-flow set exits only HFgrid14, while the GHFDB confirmed body is
    judged separately)—layers/sidecars not in the overrides are kept and
    flagged needs_hitl.
- The final application of the status file is a manual step: statuses do not
  flip automatically by date (silent_expired must be set manually). The
  mechanism is currently dormant (no status file is passed by default; the
  strip follows only the contract visibility), kept rather than removed; later,
  if new datasets awaiting confirmation appear, passing a status file enables
  it.
- Size verification (full vs public) and packaging-form recommendations (tar
  splitting vs Zarr v3 sharding) are written into the export report; actual
  data upload and publication are out of this tool's scope.

Public store writes go through the same write_store implementation (overwrite
rewrite, deterministic); values are moved lazily via dask (read full store →
strip → write public store) without changing values.
"""

import json
import os
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import xarray as xr
import yaml
from xarray import DataTree

from .contract import Contract, needs_validity_mask
from .manifest import read_manifest, write_manifest
from .masks import validity_layer_name
from .validate import validate_store, write_report
from .writer import write_store

REPO_ROOT = Path(__file__).resolve().parents[1]

T1_STATUSES = {"confirmed", "pending", "rejected", "silent_expired"}
T1_EXIT_STATUSES = {"rejected", "silent_expired"}

ZENODO_MAX_FILES = 100
ZENODO_MAX_BYTES = 50_000_000_000
ZENODO_QUOTA_NOTE = "可一次性申请提升至 200GB（文件数上限 100 不变）"

LAYER_DATASET = {
    "topography__bedrock_elevation": "etopo2022-bed",
    "derived__landsea_mask": "etopo2022-bed",
    "derived__slope": "etopo2022-bed",
    "derived__relief": "etopo2022-bed",
    "derived__pixel_area": None,
    "gravity__wgm2012_bouguer": "bouguer-wgm2012",
    "magnetics__emag2_sealevel": "emag2-v3",
    "seismology__gladm35_vsv": "glad-m35",
    "seismology__gladm35_vsh": "glad-m35",
    "seismology__gladm35_vpv": "glad-m35",
    "seismology__gladm35_vph": "glad-m35",
    "seismology__gladm35_eta": "glad-m35",
    "seismology__semucb_vs": "semucb-wm1",
    "seismology__semucb_xi": "semucb-wm1",
    "thermal__hfgrid14_heatflow": "heatflow",
    "thermal__heatflow_density": "heatflow",
    "stress_kinematics__wsm_density": "wsm2025-stress",
    "stress_kinematics__gsrm_exx": "gsrm-strain",
    "stress_kinematics__gsrm_eyy": "gsrm-strain",
    "stress_kinematics__gsrm_exy": "gsrm-strain",
    "stress_kinematics__gsrm_vorticity": "gsrm-strain",
    "stress_kinematics__gsrm_gps_density": "gsrm-strain",
    "stress_kinematics__gem_fault_slip_class": "gem-active-faults",
    "stress_kinematics__gem_fault_distance": "gem-active-faults",
    "stress_kinematics__pb2002_plate_id": "pb2002-plate-boundaries",
    "stress_kinematics__pb2002_zone_class": "pb2002-plate-boundaries",
    "stress_kinematics__pb2002_boundary_distance": "pb2002-plate-boundaries",
    "stress_kinematics__slab2_depth": "slab2",
    "stress_kinematics__slab2_dip": "slab2",
    "stress_kinematics__slab2_strike": "slab2",
    "stress_kinematics__slab2_thickness": "slab2",
    "stress_kinematics__slab2_uncertainty": "slab2",
    "stress_kinematics__slab2_overlap_mask": "slab2",
    "sediment__gst1_thickness": "gst1",
    "sediment__gum_lithology_class": "gum-unconsolidated",
    "sediment__gum_thickness_class": "gum-unconsolidated",
    "sediment__global_basins_class": "global-basins",
    "sediment__global_basins_distance": "global-basins",
    "lithosphere__gemma_moho": "crust-models",
    "lithosphere__gemma_moho_err": "crust-models",
    "lithosphere__gemma_basement_depth": "crust-models",
    "lithosphere__litho1_lab": "litho1",
    "lithosphere__crustal_age_class": "crustal-age-mooney2023",
    "lithosphere__seafloor_age": "seafloor-age-seton2020",
    "lithosphere__hasterok_province_class": "tectonic-provinces",
    "lithosphere__glim_lithology_class": "glim-lithology",
    "paleo__paleodem_elevation": "paleodem-scotese",
}

SIDECAR_DATASET = {
    "hasterok_gprv": "tectonic-provinces",
    "limw_polygons": "glim-lithology",
    "wsm2025_points": "wsm2025-stress",
    "gsrm_gps_velocities": "gsrm-strain",
    "heatflow_points_merged": "heatflow",
    "gem_active_faults": "gem-active-faults",
    "pb2002_plates": "pb2002-plate-boundaries",
    "gum_unconsolidated": "gum-unconsolidated",
    "mooney2023_provinces": "crustal-age-mooney2023",
    "global_basins": "global-basins",
}

INTERNAL_GAPS = {
    "bouguer-wgm2012": {
        "reason": (
            "BGI Terms of Use 实质 NC（非商业限定 + 禁入商业数据库），"
            "不纳入公开版"
        ),
        "guidance": "论文建议读者自行向 BGI（bgi.obs-mip.fr）联用 WGM2012",
    },
    "gsrm-strain": {
        "reason": (
            "文件头 GEM Foundation CC-BY-NC-SA 3.0（NC+SA 不符合开放许可要求），"
            "不纳入公开版（应变组 5 层 + GPS 密度层 + 侧车缺席）"
        ),
        "guidance": "论文建议读者自行联用 GSRM（geodesy.unr.edu/GSRM/）",
    },
    "gem-active-faults": {
        "reason": (
            "GEM GAF-DB 仓库 LICENSE.txt 为 CC BY-SA 4.0（SA 条款不符合开放"
            "许可要求），不纳入公开版"
        ),
        "guidance": (
            "论文建议读者直接使用 GEM GAF-DB"
            "（github.com/GEMScienceTools/gem-global-active-faults）"
        ),
    },
    "global-basins": {
        "reason": (
            "Evenick (2021) CC BY-NC-ND 4.0（ND 限衍生再分发），不纳入公开版"
        ),
        "guidance": "论文建议读者自行联用 Evenick (2021) 补充材料",
    },
    "glad-m35": {
        "reason": (
            "模型 nc 经 EMC/OneDrive/GitHub 独立分发且无许可声明，论文"
            "（GJI 2024 OA）许可不传导至独立分发的数据文件；不纳入公开版"
        ),
        "guidance": (
            "论文建议读者经 IRIS EMC"
            "（ds.iris.edu/ds/products/emc-glad-m35/）获取 GLAD-M35"
        ),
    },
    "semucb-wm1": {
        "reason": (
            "源无明示许可（viz-only nc + 源码分发；2014 论文为订阅时代，"
            "HAL 自存档 CC BY 仅覆盖文稿不传导数据）；不纳入公开版（vs/xi 两层 internal）"
        ),
        "guidance": (
            "论文建议读者经 IRIS EMC"
            "（ds.iris.edu/ds/products/emc-semucb-wm1/）获取 SEMUCB-WM1"
        ),
    },
    "pb2002-plate-boundaries": {
        "reason": (
            "源无明示许可，不纳入公开版（3 层 + pb2002_plates 侧车 internal）"
        ),
        "guidance": "论文建议读者经 peterbird.name/oldFTP/PB2002/ 获取 PB2002",
    },
    "gst1": {
        "reason": (
            "源无明示许可，不纳入公开版（沉积厚度层 internal）"
        ),
        "guidance": "论文建议读者经 birdgeo.com/gst-1.htm 获取 GST-1（Bird & Mooney 2026）",
    },
    "heatflow": {
        "reason": (
            "热流部分退出：HFgrid14 预测网格层退出"
            "（Lucazeau 2019 SI 无明示再分发许可，逐点值即再分发）；合并点"
            "侧车退出（GHFDB 主体 CC BY 4.0，但混合制品含 NGHF 记录本体、"
            "再分发许可未明示）；thermal__heatflow_density 密度层保留公开"
            "——聚合统计不可恢复源记录，属分析产物非再分发"
        ),
        "guidance": (
            "论文建议读者联用 GHFDB（doi:10.5880/fidgeo.2024.014）与 "
            "Lucazeau (2019)（doi:10.1029/2019GC008389）原源"
        ),
    },
    "glim-lithology": {
        "reason": (
            "GLiM 0.5° 栅格层维持公开（PANGAEA CC BY 3.0，符合开放许可"
            "口径）；LiMW gdb 侧车退出（源许可含商业限制）"
        ),
        "guidance": (
            "论文建议读者经 Hartmann & Moosdorf (2012) 页面"
            "（geo.uni-hamburg.de）联系作者获取 LiMW"
        ),
    },
}



def load_t1_state(path: str | Path | None) -> dict:
    """Load the license confirmation status YAML (manually maintained gate
    input).

    Validation: status values legal, catalog_id present, dataset has
    layer/sidecar ownership (typo protection).

    The gate is currently dormant (its mission is carried by the contract
    visibility): ``path=None`` is the default form—the dormant branch (no
    datasets; the strip follows only the contract visibility). The mechanism is
    kept rather than removed; later, if new datasets awaiting confirmation
    appear, a status file can be passed in again to enable it.
    """
    if path is None:
        return {"path": None, "silent_deadline": "", "datasets": {}}
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"许可证状态文件不存在: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    datasets = raw.get("datasets") or {}
    for ds, st in datasets.items():
        status = st.get("status")
        if status not in T1_STATUSES:
            raise ValueError(
                f"许可证状态 {ds}: 非法 status {status!r}（合法值 {sorted(T1_STATUSES)}）"
            )
        if "catalog_id" not in st:
            raise ValueError(f"许可证状态 {ds}: 缺 catalog_id（台账行号，"
                             "台账同步动作清单需要）")
        if (ds not in set(LAYER_DATASET.values())
                and ds not in set(SIDECAR_DATASET.values())):
            raise ValueError(
                f"许可证状态 {ds}: 无层/侧车归属（未知数据集名或拼写错误）"
            )
    return {
        "path": str(path),
        "silent_deadline": str(raw.get("silent_deadline", "")),
        "datasets": datasets,
    }



def build_strip_plan(contract: Contract, sidecar_records: list[dict],
                     t1_state: dict) -> dict:
    """Strip plan: internal + status exits → keep/drop of layers/masks/sidecars
    + gap notes + catalog action list.

    Coverage assertion (bidirectional): contract layers and manifest sidecars
    must all have a dataset mapping, preventing new layers from silently
    bypassing the visibility gate.
    """
    unmapped = [e.id for e in contract.layers if e.id not in LAYER_DATASET]
    if unmapped:
        raise ValueError(f"层缺数据集映射（LAYER_DATASET）: {unmapped}")
    unmapped_sc = [s["id"] for s in sidecar_records if s["id"] not in SIDECAR_DATASET]
    if unmapped_sc:
        raise ValueError(f"侧车缺数据集映射（SIDECAR_DATASET）: {unmapped_sc}")

    dataset_layers: dict[str, list] = {}
    for e in contract.layers:
        ds = LAYER_DATASET[e.id]
        if ds is not None:
            dataset_layers.setdefault(ds, []).append(e)
    dataset_sidecars: dict[str, list[dict]] = {}
    for s in sidecar_records:
        dataset_sidecars.setdefault(SIDECAR_DATASET[s["id"]], []).append(s)

    internal_entries = [e for e in contract.layers if e.visibility == "internal"]
    internal_sidecar_ids = {
        s["id"] for s in sidecar_records if s.get("visibility") == "internal"
    }

    t1_exit_layer_ids: set[str] = set()
    t1_exit_sidecar_ids: set[str] = set()
    decisions, catalog_actions, partial_exits, pending = [], [], [], []
    deadline = t1_state.get("silent_deadline", "")
    t1_active = bool(t1_state.get("datasets"))
    for ds, st in t1_state.get("datasets", {}).items():
        status = st["status"]
        note = st.get("note", "")
        dec = {"dataset": ds, "catalog_id": st["catalog_id"],
               "status": status, "note": note}
        if status == "confirmed":
            dec["decision"] = "维持公开（作者确认开放条款）"
            decisions.append(dec)
            continue
        if status == "pending":
            dec["decision"] = f"维持公开候选（确认等待期至 {deadline}，最终应用前）"
            decisions.append(dec)
            pending.append({
                "dataset": ds, "catalog_id": st["catalog_id"], "note": note,
                "silent_deadline": deadline,
            })
            continue
        known_layers = {e.id for e in dataset_layers.get(ds, [])}
        known_sidecars = {s["id"] for s in dataset_sidecars.get(ds, [])}
        exit_layers = st.get("exit_layers")
        if exit_layers is None:
            exit_layers = sorted(known_layers)
        exit_sidecars = st.get("exit_sidecars")
        if exit_sidecars is None:
            exit_sidecars = sorted(known_sidecars)
        bad = [l for l in exit_layers if l not in known_layers]
        if bad:
            raise ValueError(f"许可证状态 {ds}: exit_layers 含非该数据集的层 {bad}")
        bad_sc = [s for s in exit_sidecars if s not in known_sidecars]
        if bad_sc:
            raise ValueError(f"许可证状态 {ds}: exit_sidecars 含非该数据集的侧车 {bad_sc}")
        t1_exit_layer_ids |= set(exit_layers)
        t1_exit_sidecar_ids |= set(exit_sidecars)
        dec["decision"] = "退出公开版（层+侧车剥离，台账标 out-of-scope）"
        dec["exit_layers"] = exit_layers
        dec["exit_sidecars"] = exit_sidecars
        decisions.append(dec)
        reason = (
            f"作者拒复（{note}）" if status == "rejected"
            else f"确认等待期满（截止 {deadline} 无回复；{note}）"
        )
        catalog_actions.append({
            "dataset_dir": ds,
            "catalog_id": st["catalog_id"],
            "action": "status: ok → out-of-scope（v1.0 公开版退出）",
            "reason": reason,
            "howto": (
                "台账定点修改（diff 敏感文档纪律：行内文本级替换 + "
                "字段级比对）；人工执行，不在导出器内自动落盘"
            ),
        })
        kept_l = sorted(known_layers - set(exit_layers))
        kept_s = sorted(known_sidecars - set(exit_sidecars))
        if kept_l or kept_s:
            partial_exits.append({
                "dataset": ds, "status": status,
                "kept_layers": kept_l, "kept_sidecars": kept_s,
                "note": (
                    "部分退出（exit_layers/exit_sidecars 覆盖项）：保留项的"
                    "去留须人工判定（如合并数据集的主体来源已确认开放）"
                ),
            })

    strip_entry_ids = {e.id for e in internal_entries} | t1_exit_layer_ids
    strip_entries = [e for e in contract.layers if e.id in strip_entry_ids]
    strip_names = set()
    for e in strip_entries:
        strip_names.add(e.id)
        if needs_validity_mask(e.mask):
            strip_names.add(validity_layer_name(e.id))
    strip_sidecar_ids = internal_sidecar_ids | t1_exit_sidecar_ids
    kept_entries = [e for e in contract.layers if e.id not in strip_entry_ids]
    kept_sidecar_records = [
        s for s in sidecar_records if s["id"] not in strip_sidecar_ids
    ]

    gaps = []
    internal_ds = sorted(
        {LAYER_DATASET[e.id] for e in internal_entries}
        | {SIDECAR_DATASET[s["id"]] for s in sidecar_records
           if s["id"] in internal_sidecar_ids}
    )
    for ds in internal_ds:
        if ds not in INTERNAL_GAPS:
            raise ValueError(
                f"internal 数据集 {ds} 缺缺席原因登记（INTERNAL_GAPS）——"
                "禁止无名缺席"
            )
        gaps.append({
            "dataset": ds,
            "kind": "internal",
            "reason": INTERNAL_GAPS[ds]["reason"],
            "guidance": INTERNAL_GAPS[ds]["guidance"],
            "layers": sorted(e.id for e in internal_entries
                             if LAYER_DATASET[e.id] == ds),
            "sidecars": sorted(s["id"] for s in sidecar_records
                               if SIDECAR_DATASET[s["id"]] == ds
                               and s["id"] in internal_sidecar_ids),
        })
    for ds, st in sorted(t1_state.get("datasets", {}).items()):
        if st["status"] not in T1_EXIT_STATUSES:
            continue
        reason = (
            f"作者拒复（{st.get('note', '')}）" if st["status"] == "rejected"
            else f"确认等待期满（截止 {deadline} 无回复；{st.get('note', '')}）"
        )
        gaps.append({
            "dataset": ds,
            "kind": f"t1_{st['status']}",
            "reason": reason + "——层退出公开版，台账标 out-of-scope",
            "guidance": "论文建议读者自行联用原源（见 catalog 条目 url_or_doi）",
            "layers": sorted(
                e.id for e in dataset_layers.get(ds, [])
                if e.id in t1_exit_layer_ids
            ),
            "sidecars": sorted(
                s["id"] for s in dataset_sidecars.get(ds, [])
                if s["id"] in t1_exit_sidecar_ids
            ),
        })

    return {
        "n_layers_full": len(contract.layers),
        "n_layers_public": len(kept_entries),
        "kept_layer_ids": [e.id for e in kept_entries],
        "strip_layers": [e.id for e in strip_entries],
        "strip_masks": sorted(n for n in strip_names if n.endswith("__validity")),
        "strip_sidecars": sorted(strip_sidecar_ids),
        "kept_sidecar_ids": [s["id"] for s in kept_sidecar_records],
        "gaps": gaps,
        "t1_decisions": decisions,
        "pending": pending,
        "partial_exits": partial_exits,
        "catalog_actions": catalog_actions,
        "silent_deadline": deadline,
        "t1_active": t1_active,
    }



def tree_stats(path: str | Path) -> dict:
    """File/directory-tree measurement: (file count, total bytes)—shared by
    sidecar copy-integrity checks and size verification."""
    path = Path(path)
    if path.is_file():
        return {"n_files": 1, "n_bytes": path.stat().st_size}
    files = [p for p in path.rglob("*") if p.is_file()]
    return {"n_files": len(files), "n_bytes": sum(p.stat().st_size for p in files)}


def _array_dir_stats(store_dir: str | Path) -> dict[str, dict]:
    """Subtree totals (bytes, file count) within the store, keyed by array name
    (directory name).

    Zarr v3 directory layout: under {tier}/[group/]{array}/ are zarr.json + the
    chunk subtree (the c/ prefix hierarchy of the chunk key encoding)—must be
    accumulated over the whole subtree; reading direct children only counts
    zarr.json (measured discrepancy of 5 orders of magnitude). Layer ids, mask
    names, lat/lon, and group names are globally unique, so merging by directory
    name yields the per-array on-disk size (used by dry-run to estimate public
    size without writing to disk; group/tier names are also keys but never
    queried—the strip list contains only array names).

    Same-named arrays exist across tiers (a layer enters multiple tiers), so
    they must be summed rather than overwritten (verified pitfall: a dict
    comprehension overwrote all but the last-iterated tier of a 5-tier layer,
    yielding an estimate off by 3 orders of magnitude).
    """
    subtree: dict[str, dict] = {}
    for root, dirs, files in os.walk(store_dir, topdown=False):
        n_bytes = sum(
            os.path.getsize(os.path.join(root, f)) for f in files
        )
        n_files = len(files)
        for dd in dirs:
            s = subtree[os.path.join(root, dd)]
            n_bytes += s["n_bytes"]
            n_files += s["n_files"]
        subtree[root] = {"n_bytes": n_bytes, "n_files": n_files}
    by_name: dict[str, dict] = {}
    for k, v in subtree.items():
        name = Path(k).name
        if name in by_name:
            by_name[name]["n_bytes"] += v["n_bytes"]
            by_name[name]["n_files"] += v["n_files"]
        else:
            by_name[name] = dict(v)
    return by_name



def write_public_store(full_store: str | Path, strip_names: set[str],
                       out_dir: str | Path, contract: Contract) -> Path:
    """Full store → public store: strip strip_names arrays (layers + masks);
    tier subnodes (3D/4D layer groups) are removed as whole nodes once stripped
    empty; tier nodes are kept (the five-tier structure stays complete).

    Values are moved lazily via dask (the same write_store implementation)
    without changing values; root attrs mark the public edition and the gap-note
    entry point.
    """
    out_dir = Path(out_dir)
    with xr.open_datatree(full_store, engine="zarr", chunks={}) as tree:
        root_attrs = dict(tree.attrs)
        root_attrs["title"] = "地质数据立方 v1.0 公开版候选（public candidate）"
        root_attrs["edition"] = "public"
        root_attrs["public_export"] = (
            "visibility=internal 层/侧车与状态退出项已剥离；缺席数据集与"
            "许可原因见 cube_manifest.json 的 public_export 节"
        )
        nodes = {"/": xr.Dataset(attrs=root_attrs)}
        for tier in contract.tiers:
            node = tree[f"/{tier}"]
            ds = node.ds.drop_vars(
                [v for v in node.ds.data_vars if v in strip_names]
            )
            nodes[f"/{tier}"] = ds
            for gname, child in node.children.items():
                gds = child.ds.drop_vars(
                    [v for v in child.ds.data_vars if v in strip_names]
                )
                if len(gds.data_vars) > 0:
                    nodes[f"/{tier}/{gname}"] = gds
        pub_tree = DataTree.from_dict(nodes)
        return write_store(pub_tree, out_dir)


def build_public_manifest(manifest: dict, plan: dict) -> dict:
    """Full manifest → public manifest: layers/sidecars stripped in sync +
    public_export gap section."""
    kept_layers = set(plan["kept_layer_ids"])
    kept_sidecars = set(plan["kept_sidecar_ids"])
    pub = dict(manifest)
    pub["layers"] = [l for l in manifest["layers"] if l["id"] in kept_layers]
    pub["sidecars"] = [
        s for s in manifest["sidecars"] if s["id"] in kept_sidecars
    ]
    pub["public_export"] = {
        "edition": "public",
        "note": (
            "公开版候选：剥离 visibility=internal 条目与状态退出项；"
            "以下数据集因何许可证状态缺席（论文读者指引见各条 guidance）"
        ),
        "gaps": plan["gaps"],
        "t1_pending": plan["pending"],
        "t1_note": (
            "许可证状态门禁已停用（使命由契约 visibility 承担），分支空置"
            if not plan["t1_active"] else
            "t1_pending 数据集在确认等待期内维持公开候选；最终去留是 "
            f"{plan['silent_deadline']} 确认等待期满后的人工步骤"
            "（状态文件翻转后重跑导出）"
        ),
    }
    return pub


def copy_public_sidecars(sidecars_dir: str | Path, kept_records: list[dict],
                         out_dir: str | Path) -> list[dict]:
    """Public sidecar copy (by the file name of the path recorded in the
    manifest; integrity checked after copying).

    The output directory is rebuilt as a whole (cleared first, the same
    discipline as write_store): on an overwrite rerun, stale internal sidecars
    would remain in the public directory if only new ones were added and old
    ones were not cleared (the old export left a residual 1.26GB LiMW gdb,
    inflating the measured size).
    """
    sidecars_dir = Path(sidecars_dir)
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for rec in kept_records:
        name = rec["path"].split("/", 1)[1]      # "sidecars/<name>" → <name>
        src = sidecars_dir / name
        if not src.exists():
            raise FileNotFoundError(f"侧车源缺失: {src}")
        dst = out_dir / name
        stats = tree_stats(src)
        if dst.exists():
            if dst.is_dir():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
        if tree_stats(dst) != stats:
            raise RuntimeError(f"侧车复制不完整: {name}")
        copied.append({
            "id": rec["id"], "name": name,
            "n_files": stats["n_files"], "n_bytes": stats["n_bytes"],
        })
    return copied



def packaging_recommendation(sizes: dict) -> dict:
    """Packaging-form recommendation (tar splitting vs Zarr v3 sharding)—based
    on measured sizes and the Zenodo per-record thresholds (100 files / 50GB)."""

    def _total(part: str) -> dict:
        s, c = sizes[part]["store"], sizes[part]["sidecars"]
        return {
            "n_files": s["n_files"] + c["n_files"],
            "n_bytes": s["n_bytes"] + c["n_bytes"],
        }

    full, public = _total("full"), _total("public")
    over_files = public["n_files"] > ZENODO_MAX_FILES
    over_bytes = public["n_bytes"] > ZENODO_MAX_BYTES

    if over_bytes:
        split = (
            f"公开版 {public['n_bytes'] / 1e9:.1f}GB 超 Zenodo 50GB/记录门槛："
            "tar 分卷（如 split -b 49G，逐卷 sha256）或一次性申请 200GB 配额"
            f"（{ZENODO_QUOTA_NOTE}）"
        )
    else:
        split = (
            f"公开版 {public['n_bytes'] / 1e9:.1f}GB 在 Zenodo 50GB/记录门槛内："
            "单 tar 整包即可，无需分卷"
        )
    if over_files:
        file_note = (
            f"公开版 store+侧车共 {public['n_files']} 个文件，远超 Zenodo "
            f"每记录 {ZENODO_MAX_FILES} 文件上限——逐文件上传不可行，必须整包"
        )
    elif public["n_files"] > 20:
        file_note = (
            f"公开版 store+侧车共 {public['n_files']} 个文件，超过 Zenodo "
            "建议打包阈值（20+）——整包上传是唯一可行形态"
        )
    else:
        file_note = (
            f"公开版共 {public['n_files']} 个文件，无需为文件数打包"
        )
    return {
        "zenodo_limits": {
            "max_files_per_record": ZENODO_MAX_FILES,
            "max_bytes_per_record": ZENODO_MAX_BYTES,
            "note": ZENODO_QUOTA_NOTE,
        },
        "measured": {
            "full": full,
            "public": public,
            "public_estimated": sizes["public"].get("store", {}).get(
                "estimated", False
            ),
        },
        "recommendation": (
            "tar 整包（不压缩）发布：cube-v1.0-public.zarr + sidecars/ 打包为"
            "单一 tar，随包附 sha256；数据数组已是 zstd 压缩，tar 层再压缩"
            "收益趋零且拖慢解包"
        ),
        "rationale": [
            file_note,
            split,
            (
                "Zarr v3 sharding 不改变结论：Zenodo 不是随机访问对象存储，"
                "分片只减少文件数、不产生直接 zarr 访问收益；仅当后续提供"
                "HTTP range 直读镜像（自建/云对象存储）时才值得以 sharding"
                "重打包（v1.1 候选增强，需重建 store）"
            ),
        ],
    }



def _assert_no_overlap(store, sidecars_dir, out, out_sidecars) -> None:
    """Output paths and source paths must not contain each other—write_store and
    the sidecar copy are both overwrite-style (rmtree the target first), so an
    out falling inside the store/sidecar directory (or vice versa, or
    coinciding) would delete the source first, irrecoverably; a single wrong CLI
    argument triggers this, so it must be intercepted up front."""
    store_r = Path(store).resolve()
    sc_r = Path(sidecars_dir).resolve()
    out_r = Path(out).resolve()
    osc_r = Path(out_sidecars).resolve()
    for name, src, dst in (
        ("store", store_r, out_r), ("store", store_r, osc_r),
        ("sidecars", sc_r, out_r), ("sidecars", sc_r, osc_r),
    ):
        if dst == src or dst in src.parents or src in dst.parents:
            raise ValueError(
                f"输出路径 {dst} 与源 {name} {src} 重叠"
                "（覆盖式写入会先删除源，已拦截）"
            )
    if out_r == osc_r or out_r in osc_r.parents or osc_r in out_r.parents:
        raise ValueError(f"两个输出路径互相重叠: {out_r} vs {osc_r}")


def _sidecars_subset_stats(sidecars_dir, kept_ids, manifest) -> dict:
    """Size measurement of the public sidecar subset (for dry-run; the files are
    already on disk, measured directly)."""
    sidecars_dir = Path(sidecars_dir)
    n_files = n_bytes = 0
    by_id = {s["id"]: s for s in manifest["sidecars"]}
    for sid in kept_ids:
        name = by_id[sid]["path"].split("/", 1)[1]
        st = tree_stats(sidecars_dir / name)
        n_files += st["n_files"]
        n_bytes += st["n_bytes"]
    return {"n_files": n_files, "n_bytes": n_bytes}


def export_public(store: str | Path, contract: Contract,
                  sidecars_dir: str | Path, out: str | Path,
                  out_sidecars: str | Path, *,
                  t1_state_path: str | Path | None = None,
                  reports_dir: str | Path,
                  dry_run: bool = False) -> dict:
    """Public-edition export orchestration: strip plan → (dry-run: estimate +
    decision list only) → public store + manifest + sidecar copy + independent
    structural validation + size verification + report.

    With the license status gate retired, t1_state_path defaults to None = the
    dormant branch, and the strip follows only the contract visibility; passing
    a status file later re-enables that branch.
    """
    _assert_no_overlap(store, sidecars_dir, out, out_sidecars)
    store = Path(store)
    manifest = read_manifest(store)
    t1_state = load_t1_state(t1_state_path)
    plan = build_strip_plan(contract, manifest["sidecars"], t1_state)
    strip_names = set(plan["strip_layers"]) | set(plan["strip_masks"])

    t1_state_desc = (
        str(t1_state_path) if t1_state_path is not None
        else "（许可证状态门禁已停用，分支空置——剥离仅按契约 visibility）"
    )

    full_sizes = {
        "store": tree_stats(store),
        "sidecars": tree_stats(sidecars_dir),
    }

    if dry_run:
        arr = _array_dir_stats(store)
        stripped = [arr.get(n, {"n_bytes": 0, "n_files": 0}) for n in strip_names]
        public_sizes = {
            "store": {
                "n_files": full_sizes["store"]["n_files"]
                           - sum(s["n_files"] for s in stripped),
                "n_bytes": full_sizes["store"]["n_bytes"]
                           - sum(s["n_bytes"] for s in stripped),
                "estimated": True,
                "note": "估算 = 完整 store − 剥离数组磁盘量（重压缩可有小差异）",
            },
            "sidecars": _sidecars_subset_stats(
                sidecars_dir, plan["kept_sidecar_ids"], manifest
            ),
        }
        report = {
            "result": "DRY-RUN",
            "dry_run": True,
            "store": str(store),
            "public_store": str(out),
            "public_sidecars": str(out_sidecars),
            "t1_state": t1_state_desc,
            "plan": plan,
            "sizes": {"full": full_sizes, "public": public_sizes},
            "packaging": packaging_recommendation({
                "full": full_sizes, "public": public_sizes,
            }),
            "validation": None,
        }
        return report

    out = write_public_store(store, strip_names, out, contract)
    pub_manifest = build_public_manifest(manifest, plan)
    write_manifest(pub_manifest, out)

    kept_records = [
        s for s in manifest["sidecars"] if s["id"] in set(plan["kept_sidecar_ids"])
    ]
    copied = copy_public_sidecars(sidecars_dir, kept_records, out_sidecars)

    kept_entries = [e for e in contract.layers
                    if e.id in set(plan["kept_layer_ids"])]
    public_contract = replace(contract, layers=tuple(kept_entries))
    validation = validate_store(out, public_contract)
    write_report(validation, reports_dir, stem="structure-validation-public")

    public_sizes = {
        "store": tree_stats(out),
        "sidecars": tree_stats(out_sidecars),
    }
    report = {
        "result": validation["result"],
        "dry_run": False,
        "store": str(store),
        "public_store": str(out),
        "public_sidecars": str(out_sidecars),
        "t1_state": t1_state_desc,
        "plan": plan,
        "copied_sidecars": copied,
        "sizes": {"full": full_sizes, "public": public_sizes},
        "packaging": packaging_recommendation({
            "full": full_sizes, "public": public_sizes,
        }),
        "validation": {
            "result": validation["result"],
            "passed": validation["passed"],
            "total_checks": validation["total_checks"],
            "failed": validation["failed"],
            "report": "structure-validation-public.json",
        },
    }
    write_export_report(report, reports_dir)
    return report


def write_export_report(report: dict, reports_dir: str | Path) -> Path:
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / "public-export-report.json"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    p = report["plan"]
    sizes = report["sizes"]
    gb = lambda b: f"{b / 1e9:.2f}GB"  # noqa: E731
    lines = [
        "# 公开版导出报告（公开导出器 + 打包核实）",
        "",
        f"- 结果：**{report['result']}**"
        + ("（dry-run，未写盘）" if report["dry_run"] else ""),
        f"- 完整 store：`{report['store']}`",
        f"- 公开 store：`{report['public_store']}`（许可证状态输入：`{report['t1_state']}`）",
        "",
        "## 剥离清单（机器可读全文见 JSON plan 节）",
        "",
        f"- 层：{p['n_layers_full']} → **{p['n_layers_public']}**"
        f"（剥离 {len(p['strip_layers'])} 层 + {len(p['strip_masks'])} 掩膜）",
        f"- 侧车：保留 {', '.join(p['kept_sidecar_ids']) or '无'}",
        f"- 剥离层：{', '.join(p['strip_layers'])}",
        f"- 剥离侧车：{', '.join(p['strip_sidecars']) or '无'}",
        "",
        "## 缺口注记（哪个数据集因何许可证状态缺席）",
        "",
        "| 数据集 | 类别 | 原因 | 读者指引 |",
        "|---|---|---|---|",
    ]
    for g in p["gaps"]:
        lines.append(
            f"| {g['dataset']} | {g['kind']} | {g['reason']} | {g['guidance']} |"
        )
    lines += ["", "## 许可证状态分支决策", ""]
    for d in p["t1_decisions"]:
        lines.append(f"- {d['dataset']}（台账 id={d['catalog_id']}，"
                     f"{d['status']}）→ {d['decision']}")
    if p["pending"]:
        lines += [
            "",
            f"待确认（确认等待期至 {p['silent_deadline']}，最终应用是人工步骤）："
            + "、".join(
                f"{x['dataset']}(id={x['catalog_id']})" for x in p["pending"]
            ),
        ]
    if p["partial_exits"]:
        lines += ["", "## 部分退出（needs_hitl）", ""]
        for x in p["partial_exits"]:
            lines.append(
                f"- {x['dataset']}（{x['status']}）：保留层 "
                f"{x['kept_layers']} / 侧车 {x['kept_sidecars']}——{x['note']}"
            )
    if p["catalog_actions"]:
        lines += ["", "## 台账同步动作清单（人工执行，不自动落盘）", ""]
        for a in p["catalog_actions"]:
            lines.append(
                f"- id={a['catalog_id']} {a['dataset_dir']}：{a['action']}"
                f"——{a['reason']}；{a['howto']}"
            )
    lines += [
        "",
        "## 体积核实（完整版 vs 公开版）",
        "",
        "| 制品 | 文件数 | 字节 |",
        "|---|---|---|",
        f"| 完整 store | {sizes['full']['store']['n_files']} "
        f"| {sizes['full']['store']['n_bytes']:,} |",
        f"| 完整侧车 | {sizes['full']['sidecars']['n_files']} "
        f"| {sizes['full']['sidecars']['n_bytes']:,} |",
    ]
    ps = sizes["public"]["store"]
    lines.append(f"| 公开 store | {ps['n_files']} | {ps['n_bytes']:,} |")
    psc = sizes["public"]["sidecars"]
    lines.append(f"| 公开侧车 | {psc['n_files']} | {psc['n_bytes']:,} |")
    lines += [
        "",
        f"- 完整版合计 ≈ {gb(sizes['full']['store']['n_bytes'] + sizes['full']['sidecars']['n_bytes'])}"
        f"；公开版合计 ≈ {gb(ps['n_bytes'] + psc['n_bytes'])}",
        "",
        "## 打包形态建议",
        "",
        f"- **{report['packaging']['recommendation']}**",
        "",
    ]
    lines += [f"- {r}" for r in report["packaging"]["rationale"]]
    if report.get("validation"):
        v = report["validation"]
        lines += [
            "",
            "## 独立结构验证（公开版候选）",
            "",
            f"- 结果：**{v['result']}**（{v['passed']}/{v['total_checks']} 项通过"
            f" → {v['report']}）",
        ]
    lines += [
        "",
        "实际数据上传发布不在本工具范围（后续独立工作）。",
    ]
    (reports_dir / "public-export-report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return json_path
