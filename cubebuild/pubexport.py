"""公开版导出器：按契约 visibility 剥离 + 缺口注记 + 打包核实。

机制：

- 公开版候选 = 完整 store − visibility=internal 层（含 ``__validity``
  伴生掩膜）− internal 侧车。产物：公开 store + 公开侧车目录 + 剥离后的
  cube_manifest（缺口注记：哪个数据集因何许可证状态缺席）+ 机器可读导出
  报告。
- 状态分支（门禁输入 = 许可证确认状态 YAML，人工维护）：
    confirmed                → 层维持公开；
    pending                  → 层维持公开候选（确认等待期至 silent_deadline）；
    rejected / silent_expired → 层退出公开版 + 台账标 out-of-scope
                       （台账同步为人工步骤，只产出 catalog_actions 动作
                       清单，不在导出器内自动落盘）；
    exit_layers / exit_sidecars 覆盖项 → 部分退出（如热流合并集只退
    HFgrid14、GHFDB 确认主体另行判定）——不在覆盖项内的层/侧车保留并
    记 needs_hitl 标记。
- 状态文件的最终应用是人工步骤：状态不按日期自动翻转（silent_expired
  须人工置位）。当前该机制空置（缺省不传状态文件，剥离只按契约
  visibility 走），保留不删，后续若出现新的待确认数据集传入状态文件
  即可启用。
- 体积核实（完整版 vs 公开版）与打包形态建议（tar 分卷 vs Zarr v3
  sharding）写入导出报告；实际数据上传发布不在本工具范围。

公开 store 写入走 write_store 同一实现（覆盖式重写，确定性）；
数值经 dask 惰性搬运（读完整 store → 剥离 → 写公开 store），不改值。
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
# 许可证确认状态门禁当前空置：缺省状态文件已移入内部存档；
# CLI --t1-state 缺省 None = 空置分支（剥离只按契约 visibility 走）。

T1_STATUSES = {"confirmed", "pending", "rejected", "silent_expired"}
T1_EXIT_STATUSES = {"rejected", "silent_expired"}

# Zenodo 每记录门槛（官方文档查证；打包建议的事实基准）
ZENODO_MAX_FILES = 100
ZENODO_MAX_BYTES = 50_000_000_000
ZENODO_QUOTA_NOTE = "可一次性申请提升至 200GB（文件数上限 100 不变）"

# 层 id → 台账 dataset_dir（许可判定的数据集粒度）。
# None = 无数据集归属（解析公式层）。覆盖完备性由 build_strip_plan 断言
# （契约漂移即显式失败，防止新层静默绕过 visibility 门禁）。
LAYER_DATASET = {
    "topography__bedrock_elevation": "etopo2022-bed",
    "derived__landsea_mask": "etopo2022-bed",       # 派生自高程层（同源数据集）
    "derived__slope": "etopo2022-bed",
    "derived__relief": "etopo2022-bed",
    "derived__pixel_area": None,                     # 纬度解析公式，无源数据集
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

# 侧车 id → catalog dataset_dir（公开导出按同一数据集粒度剥离）
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

# internal 数据集的缺席原因与读者指引（公开 manifest 缺口注记来源：
# 「哪个数据集因何许可证状态缺席」）。新增 internal 数据集必须在此登记，
# 否则 build_strip_plan 显式失败（禁止无名缺席）。
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
    # ---- 以下六项为源再分发许可未明示的数据集（契约 visibility 已置 internal）
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


# ---------------- 许可证确认状态加载 ----------------

def load_t1_state(path: str | Path | None) -> dict:
    """加载许可证确认状态 YAML（人工维护的门禁输入）。

    校验：状态值合法、catalog_id 在位、数据集有层/侧车归属（防拼写错误）。

    该门禁当前空置（使命由契约 visibility 承担）：``path=None`` 为缺省形态
    ——空置分支（无数据集，剥离只按契约 visibility 走）。机制保留不删，
    后续若出现新的待确认数据集可重新传入状态文件启用。
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


# ---------------- 剥离计划 ----------------

def build_strip_plan(contract: Contract, sidecar_records: list[dict],
                     t1_state: dict) -> dict:
    """剥离计划：internal + 状态退出项 → 层/掩膜/侧车去留 + 缺口注记 +
    台账动作清单。

    覆盖断言（双向）：契约层与 manifest 侧车必须全部有数据集映射，
    防止新层静默绕过 visibility 门禁。
    """
    # 0. 映射覆盖断言
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

    # 1. internal 剥离（契约 visibility 为唯一依据）
    internal_entries = [e for e in contract.layers if e.visibility == "internal"]
    internal_sidecar_ids = {
        s["id"] for s in sidecar_records if s.get("visibility") == "internal"
    }

    # 2. 许可证状态分支
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
        # rejected / silent_expired → 退出 v1.0 公开版
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

    # 3. 汇总剥离集合（internal ∪ 状态退出；掩膜随数据层）
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

    # 4. 缺口注记（manifest public_export 节来源）
    #    internal 数据集全集 = internal 层 ∪ internal 侧车的数据集——
    #    仅侧车 internal 的数据集同样必须登记缺席原因（禁止无名缺席）
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


# ---------------- 体积度量 ----------------

def tree_stats(path: str | Path) -> dict:
    """文件/目录树度量：(文件数, 字节总量)——侧车复制完整性与体积核实共用。"""
    path = Path(path)
    if path.is_file():
        return {"n_files": 1, "n_bytes": path.stat().st_size}
    files = [p for p in path.rglob("*") if p.is_file()]
    return {"n_files": len(files), "n_bytes": sum(p.stat().st_size for p in files)}


def _array_dir_stats(store_dir: str | Path) -> dict[str, dict]:
    """store 内按数组名（目录名）汇总的子树（字节数, 文件数）。

    Zarr v3 目录布局：{tier}/[group/]{array}/ 下是 zarr.json + chunk 子树
    （chunk key encoding 的 c/ 前缀层级）——须按整子树累计，直读直接子文件
    只会数到 zarr.json（实测差 5 个量级）。层 id、掩膜名、lat/lon、组名
    全局唯一，按目录名归并即得单数组磁盘量（dry-run 估算公开版体积用，
    不写盘；组/档位名同为键但不会被查询——剥离名单只含数组名）。

    同名数组跨档位存在（层入多档），须累加而非覆盖（实测踩坑：字典
    推导式覆盖使 5 档层只剩最后遍历的一档，估算差 3 个量级）。
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


# ---------------- 公开 store / manifest / 侧车 ----------------

def write_public_store(full_store: str | Path, strip_names: set[str],
                       out_dir: str | Path, contract: Contract) -> Path:
    """完整 store → 公开 store：剥离 strip_names 数组（层+掩膜），档位子节点
    （3D/4D 层组）剥空即整节点移除；档位节点保留（五档结构完整）。

    数值经 dask 惰性搬运（同一 write_store 实现），不改值；根 attrs 标注
    公开版与缺口注记入口。
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
    """完整 manifest → 公开 manifest：层/侧车同步剥离 + public_export 缺口节。"""
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
    """公开侧车复制（按 manifest 记录 path 的文件名；复制后完整性核对）。

    输出目录整体重建（先清空，与 write_store 同一纪律）：覆盖式重跑时
    陈旧 internal 侧车若只新增不清旧会留存公开目录（旧导出残留
    1.26GB LiMW gdb，体积实测随之虚高）。
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


# ---------------- 打包形态建议 ----------------

def packaging_recommendation(sizes: dict) -> dict:
    """打包形态建议（tar 分卷 vs Zarr v3 sharding）——基于实测体积与
    Zenodo 每记录门槛（100 文件 / 50GB）。"""

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


# ---------------- 导出编排 ----------------

def _assert_no_overlap(store, sidecars_dir, out, out_sidecars) -> None:
    """输出路径与源路径互不包含——write_store/侧车复制均为覆盖式（先
    rmtree 目标），out 落在 store/侧车目录内（或反之、或重合）会先删除
    源，不可恢复；一个错误 CLI 参数即触发，必须前置拦截。"""
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
    """公开侧车子集的体积度量（dry-run 用；文件已在盘，直接量）。"""
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
    """公开版导出编排：剥离计划 →（dry-run：估算+决策清单即止）→
    公开 store + manifest + 侧车复制 + 独立结构验证 + 体积核实 + 报告。

    许可证状态门禁停用后 t1_state_path 缺省 None = 空置分支，
    剥离只按契约 visibility 走；后续传入状态文件可重新启用该分支。
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
        # 估算公开 store 体积（完整 store − 剥离数组磁盘量；不动盘）
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

    # 1. 公开 store + manifest
    out = write_public_store(store, strip_names, out, contract)
    pub_manifest = build_public_manifest(manifest, plan)
    write_manifest(pub_manifest, out)

    # 2. 公开侧车复制（完整性核对在复制器内）
    kept_records = [
        s for s in manifest["sidecars"] if s["id"] in set(plan["kept_sidecar_ids"])
    ]
    copied = copy_public_sidecars(sidecars_dir, kept_records, out_sidecars)

    # 3. 独立结构验证（剥离后的契约子集 × 公开 store × 公开 manifest）
    kept_entries = [e for e in contract.layers
                    if e.id in set(plan["kept_layer_ids"])]
    public_contract = replace(contract, layers=tuple(kept_entries))
    validation = validate_store(out, public_contract)
    write_report(validation, reports_dir, stem="structure-validation-public")

    # 4. 体积核实（实测）
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
