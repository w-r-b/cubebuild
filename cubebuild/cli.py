"""构建入口。

用法（geo_dl_clean_wsl 环境）：
  ~/miniforge3/envs/geo_dl_clean_wsl/bin/python -m cubebuild build \\
      --mapping cubebuild/layer-mapping-v0.yaml \\
      --out products/cube-v1.0.zarr --reports products/reports
  python -m cubebuild validate --store products/cube-v1.0.zarr --mapping <同上>
  python -m cubebuild integrate --store products/cube-v1.0.zarr --mapping <同上>
  python -m cubebuild export-public --store products/cube-v1.0.zarr [--dry-run]
  python -m cubebuild checksum --store products/cube-v1.0.zarr
"""

import argparse
import json
import sys
from pathlib import Path

import xarray as xr

from . import __version__
from .checksum import dumps, store_checksums
from .contract import (
    ContractViolation,
    DEFAULT_CONTRACT_PATH,
    load_contract,
    needs_validity_mask,
    validate_contract,
)
from .etopo import DEFAULT_SRC_DIR as ETOPO_SRC_DIR
from .etopo import SourceNotReady
from .faults import FAULT_FIDELITY_LAYERS, build_faults_fidelity_report
from .fidelity import (
    DERIVED_FIDELITY_LAYERS,
    ELEVATION_LAYER_ID,
    build_derived_fidelity_report,
    build_fidelity_report,
    write_fidelity_report,
)
from .gravmag import GRAVMAG_LAYERS, build_gravmag_fidelity_report
from .gsrm import STRAIN_LAYERS, build_gsrm_strain_fidelity_report
from .integration import build_integration_report
from .layers import LAYER_BUILDERS
from .landcat import LANDCAT_FIDELITY_LAYERS, build_landcat_fidelity_report
from .litho import (
    LITHO1_LAB_ID,
    LITHO_FIDELITY_LAYERS,
    build_litho_fidelity_report,
    write_litho1_inventory_report,
)
from .manifest import PROVENANCE_ATTR_KEYS, build_manifest, write_manifest
from .masks import build_validity_mask, data_missing, validity_layer_name
from .paleo import PALEO_4D_GROUP_OF, PALEO_LAYERS, build_paleo_fidelity_report
from .pb2002 import PB2002_FIDELITY_LAYERS, build_pb2002_fidelity_report
from .points import POINT_DENSITY_LAYERS, build_point_density_fidelity_report
from .pubexport import export_public
from .seismic import SEISMIC_3D_GROUP_OF, SEISMIC_LAYERS, build_seismic_fidelity_report
from .sediment import SEDIMENT_FIDELITY_LAYERS, build_sediment_fidelity_report
from .sidecars import SIDECARS
from .slab2 import SLAB2_LAYERS, build_slab2_fidelity_report, merge_slab2, write_slab2_merge_report
from .thermal import THERMAL_LAYERS, build_thermal_fidelity_report
from .validate import validate_store, write_report
from .vector import VECTOR_FIDELITY_LAYERS, build_hasterok_fidelity_report
from .writer import assemble_datatree, write_store

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAPPING = DEFAULT_CONTRACT_PATH
DEFAULT_OUT = REPO_ROOT / "products/cube-v1.0.zarr"
DEFAULT_REPORTS = REPO_ROOT / "products/reports"
DEFAULT_SIDECARS = REPO_ROOT / "products/sidecars"

# 非档位本节点层的组路由（3D 地震 / 4D 古高程）：
# 层 id → 档位子节点名（store 内 /{tier}/{group}）
LAYER_GROUP_OF = {**SEISMIC_3D_GROUP_OF, **PALEO_4D_GROUP_OF}


def cmd_build(args) -> int:
    # 1. 契约加载 + 铁律前置校验（违规即失败并指明违规层）
    contract = load_contract(args.mapping)
    try:
        validate_contract(contract)
    except ContractViolation as e:
        print(f"[铁律] 前置校验失败：\n{e}", file=sys.stderr)
        return 2
    n_pub = sum(1 for e in contract.layers if e.visibility == "public")
    n_int = len(contract.layers) - n_pub
    print(
        f"[契约] {contract.path.name}：{len(contract.layers)} 层"
        f"（{n_pub} public / {n_int} internal）"
    )
    print("[铁律] 前置校验通过：无层进入细于原生分辨率的档；1° 全集档齐备")

    # 2. 逐层构建（已登记构建器的层；源未集齐的层跳过并记 pending）
    tier_datasets: dict[str, dict] = {t: {} for t in contract.tiers}
    # 3D 层组：深度坐标互不相同的模型层入档位子节点
    group_datasets: dict[tuple[str, str], dict] = {}
    built_entries, pending = [], []
    for entry in contract.layers:
        builder = LAYER_BUILDERS.get(entry.id)
        if builder is None:
            pending.append(entry.id)
            continue
        group = LAYER_GROUP_OF.get(entry.id)
        try:
            for tier in entry.tiers:
                arr = builder(entry, tier)
                if group is not None:
                    bucket = group_datasets.setdefault((tier, group), {})
                else:
                    bucket = tier_datasets[tier]
                bucket[entry.id] = arr
                # 有效性掩膜伴生层：契约 mask 含 validity 即生成
                if needs_validity_mask(entry.mask):
                    bucket[validity_layer_name(entry.id)] = (
                        build_validity_mask(arr, entry)
                    )
        except SourceNotReady as e:
            pending.append(entry.id)
            print(f"[跳过] {entry.id}：源未集齐 — {e}")
            for tier in entry.tiers:
                tier_datasets[tier].pop(entry.id, None)
                tier_datasets[tier].pop(validity_layer_name(entry.id), None)
                if group is not None:
                    # 逐层清理（不动兄弟层：组内其余成员可能已成功入桶）
                    bucket = group_datasets.get((tier, group))
                    if bucket is not None:
                        bucket.pop(entry.id, None)
                        bucket.pop(validity_layer_name(entry.id), None)
            continue
        built_entries.append(entry)
        print(f"[构建] {entry.id}：{len(entry.tiers)} 档（{'/'.join(entry.tiers)}）")
    print(
        f"[跳过] {len(pending)} 层未入本版 store（或待后续补入），"
        "当前 store 仅含已实现层"
    )

    # 3. 写 store（3D 层组为档位子节点：/1deg/gladm35、/1deg/semucb）
    datasets = {t: xr.Dataset(dict(vars_)) for t, vars_ in tier_datasets.items()}
    groups = {
        key: xr.Dataset(dict(vars_)) for key, vars_ in group_datasets.items()
    }
    tree = assemble_datatree(contract, datasets, groups)
    store_dir = write_store(tree, args.out)
    print(f"[写入] {store_dir}")

    # 3.5 矢量侧车导出（逐侧车调用，源缺失跳过记 pending）
    sidecar_records: list[dict] = []
    sidecar_pending: list[str] = []
    for sid, exporter in SIDECARS.items():
        try:
            sidecar_records.append(exporter(args.sidecars))
        except FileNotFoundError as e:
            sidecar_pending.append(sid)
            print(f"[跳过] 侧车 {sid}：源未集齐 — {e}")
    for rec in sidecar_records:
        print(f"[侧车] {rec['id']} → {rec['path']}（{rec['format']}）")

    # 3.6 Slab2 合并回报（重叠像元总量写入构建回报；合并已缓存）
    if any(e.id in SLAB2_LAYERS for e in built_entries):
        merge_path = write_slab2_merge_report(args.reports)
        st = merge_slab2()["stats"]
        print(
            f"[Slab2] 27 分区合并：覆盖 3′ 像元 {st['covered_3min_pixels']}，"
            f"重叠 {st['overlap_3min_pixels']}（{st['overlap_share_of_coverage']:.2%}）"
            f"→ {merge_path.name}"
        )

    # 3.7 LITHO1.0 变量清单回报（nc 其余变量清单写入构建回报，
    #     v1.1 展开依据；确定性无时间戳）
    if any(e.id == LITHO1_LAB_ID for e in built_entries):
        inv_path = write_litho1_inventory_report(args.reports)
        inv = json.loads(inv_path.read_text(encoding="utf-8"))
        print(
            f"[LITHO1] nc 变量清单：{inv['n_variables']} 项（ingest 1 = LAB，"
            f"v1.1 候选 {inv['n_v1_1_candidates']}）→ {inv_path.name}"
        )

    # 4. 覆盖率（单遍读取）+ cube_manifest
    coverage: dict[tuple[str, str], float] = {}
    layer_extras: dict[str, dict] = {}   # 构建器补充溯源字段（doi/composite_rule 等）
    with xr.open_datatree(store_dir, engine="zarr", chunks={}) as dt:
        for entry in built_entries:
            group = LAYER_GROUP_OF.get(entry.id)
            for tier in entry.tiers:
                node = dt[f"/{tier}/{group}"] if group is not None else dt[f"/{tier}"]
                arr = node[entry.id]
                # 缺测判定按层 dtype 与契约：浮点 NaN；整型类别层仅在
                # 契约要求 validity 时 0=缺测（landsea 0=海为真实类）
                zero_missing = needs_validity_mask(entry.mask)
                missing = data_missing(arr, zero_is_missing=zero_missing)
                cov = 1.0 - (missing.sum(dtype="float64") / arr.size).compute().item()
                coverage[(entry.id, tier)] = cov
                if entry.id not in layer_extras:
                    layer_extras[entry.id] = {
                        k: arr.attrs[k]
                        for k in PROVENANCE_ATTR_KEYS
                        if k in arr.attrs
                    }
    manifest = build_manifest(
        contract, built_entries, coverage, extras=layer_extras, sidecars=sidecar_records
    )
    manifest_path = write_manifest(manifest, store_dir)
    cov_str = " ".join(
        f"{e.id}={'/'.join(f'{coverage[(e.id, t)]:.3f}' for t in e.tiers)}"
        for e in built_entries
    )
    print(
        f"[注册表] {manifest_path.name}：{len(manifest['layers'])} 条层条目 + "
        f"{len(sidecar_records)} 侧车；覆盖率 {cov_str}"
    )

    # 5. 结构验证 + 报告
    report = validate_store(store_dir, contract)
    report_path = write_report(report, args.reports)
    print(
        f"[验证] 结构验证 {report['result']}"
        f"（{report['passed']}/{report['total_checks']} 项通过）→ {report_path}"
    )

    # 6. 构建摘要
    summary = {
        "built": [e.id for e in built_entries],
        "pending": pending,
        "sidecars": [r["id"] for r in sidecar_records],
        "sidecars_pending": sidecar_pending,
        "store": str(store_dir),
        "manifest": str(manifest_path),
        "validation": {"result": report["result"], "path": str(report_path)},
    }
    summary_path = Path(args.reports) / "build-summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0 if report["result"] == "PASS" else 1


def cmd_validate(args) -> int:
    contract = load_contract(args.mapping)
    try:
        validate_contract(contract)
    except ContractViolation as e:
        print(f"[铁律] 契约校验失败：\n{e}", file=sys.stderr)
        return 2
    report = validate_store(args.store, contract)
    report_path = write_report(report, args.reports)
    print(
        f"[验证] 结构验证 {report['result']}"
        f"（{report['passed']}/{report['total_checks']} 项通过）→ {report_path}"
    )
    return 0 if report["result"] == "PASS" else 1


def cmd_fidelity(args) -> int:
    """保真验证：ETOPO 走源统计 harness，派生层走高程层一致性，重磁组走
    本层自供源统计 + 逐位重算/积分判据（各层自供报告器）。"""
    contract = load_contract(args.mapping)
    entry = next((e for e in contract.layers if e.id == args.layer), None)
    if entry is None:
        print(f"[保真] 契约中无层 {args.layer}", file=sys.stderr)
        return 2
    # 源目录缺省按层解析（--src-dir 显式给出时优先）
    src_dir = args.src_dir
    try:
        if entry.id in DERIVED_FIDELITY_LAYERS:
            report = build_derived_fidelity_report(args.store, entry.id, list(entry.tiers))
        elif entry.id == ELEVATION_LAYER_ID:
            report = build_fidelity_report(
                args.store, args.layer, list(entry.tiers),
                src_dir or ETOPO_SRC_DIR,
            )
        elif entry.id in GRAVMAG_LAYERS:
            report = build_gravmag_fidelity_report(
                args.store, entry.id, list(entry.tiers),
                src_dir or GRAVMAG_LAYERS[entry.id]["src_dir"],
            )
        elif entry.id in VECTOR_FIDELITY_LAYERS:
            report = build_hasterok_fidelity_report(
                args.store, entry.id, list(entry.tiers),
                src_dir or VECTOR_FIDELITY_LAYERS[entry.id]["src_dir"],
            )
        elif entry.id in FAULT_FIDELITY_LAYERS:
            # 断层双通道：单源单文件，源路径由常量解析（src_dir 忽略）
            report = build_faults_fidelity_report(
                args.store, args.layer, list(entry.tiers), src_dir
            )
        elif entry.id in PB2002_FIDELITY_LAYERS:
            # PB2002 三通道：单源单文件，源路径由常量解析（src_dir 忽略）
            report = build_pb2002_fidelity_report(
                args.store, args.layer, list(entry.tiers), src_dir
            )
        elif entry.id in POINT_DENSITY_LAYERS:
            report = build_point_density_fidelity_report(
                args.store, entry.id, list(entry.tiers),
                src_dir or POINT_DENSITY_LAYERS[entry.id]["src_dir"],
            )
        elif entry.id in STRAIN_LAYERS:
            # GSRM 应变四层：单源单文件，源路径由常量解析（src_dir 忽略）
            report = build_gsrm_strain_fidelity_report(
                args.store, entry.id, list(entry.tiers)
            )
        elif entry.id in THERMAL_LAYERS:
            # 热流两层：密度层（双源）src_dir 缺省交报告器解析，
            # HFgrid14 = 其 SI-S01 目录
            default_src = (
                None if THERMAL_LAYERS[entry.id]["kind"] == "density"
                else THERMAL_LAYERS[entry.id]["src_dir"]
            )
            report = build_thermal_fidelity_report(
                args.store, entry.id, list(entry.tiers), src_dir or default_src
            )
        elif entry.id in SEISMIC_LAYERS:
            # 3D 地震层：单源单文件，源路径由常量解析（src_dir 忽略）
            report = build_seismic_fidelity_report(
                args.store, entry.id, list(entry.tiers)
            )
        elif entry.id in PALEO_LAYERS:
            # 4D 古高程层：两档各随其官方产品目录（src_dir 忽略）
            report = build_paleo_fidelity_report(
                args.store, entry.id, list(entry.tiers)
            )
        elif entry.id in SLAB2_LAYERS:
            # Slab2 六层：合并基准档派生，源路径由常量解析（src_dir 忽略）
            report = build_slab2_fidelity_report(
                args.store, entry.id, list(entry.tiers)
            )
        elif entry.id in LANDCAT_FIDELITY_LAYERS:
            # 陆域类别四层：单源单文件，源路径由常量解析（src_dir 忽略）
            report = build_landcat_fidelity_report(
                args.store, args.layer, list(entry.tiers)
            )
        elif entry.id in LITHO_FIDELITY_LAYERS:
            # 岩石圈栅格五层：三源常量解析（src_dir 忽略）
            report = build_litho_fidelity_report(
                args.store, args.layer, list(entry.tiers)
            )
        elif entry.id in SEDIMENT_FIDELITY_LAYERS:
            # 沉积组三层：GST1 + Global Basins，源路径由常量解析
            report = build_sediment_fidelity_report(
                args.store, args.layer, list(entry.tiers)
            )
        else:
            print(f"[保真] {args.layer}：暂无保真实现（待后续接入）", file=sys.stderr)
            return 2
    except SourceNotReady as e:
        print(f"[保真] {args.layer}：源未集齐，无法比对 — {e}", file=sys.stderr)
        return 2
    except KeyError as e:
        print(f"[保真] {args.layer}：{e}", file=sys.stderr)
        return 2
    except FileNotFoundError as e:
        print(f"[保真] {args.layer}：源文件缺失 — {e}", file=sys.stderr)
        return 2
    report_path = write_fidelity_report(report, args.reports)
    n_fail = sum(1 for c in report["checks"] if c["status"] == "FAIL")
    print(
        f"[保真] {args.layer}：{report['result']}"
        f"（{len(report['checks'])} 项检查，{n_fail} 项 FAIL）→ {report_path}"
    )
    return 0 if report["result"] == "PASS" else 1


def cmd_checksum(args) -> int:
    cs = store_checksums(args.store)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(dumps(cs), encoding="utf-8")
        print(f"[checksum] {len(cs)} 个数组 → {args.json}")
    else:
        print(dumps(cs))
    return 0


def cmd_integrate(args) -> int:
    """集成验证总验收单：全量结构验证 + 保真汇总 + 构建回报四项
    + 物理交叉四组（只报告不拦截）。"""
    contract = load_contract(args.mapping)
    try:
        validate_contract(contract)
    except ContractViolation as e:
        print(f"[铁律] 契约校验失败：\n{e}", file=sys.stderr)
        return 2
    report = build_integration_report(args.store, contract, args.reports)
    report_path = Path(args.reports) / "integration-report.json"
    print(
        f"[集成] 总验收单 {report['result']}："
        f"结构 {report['structure']['result']}"
        f"（{report['structure']['passed']}/{report['structure']['total_checks']}），"
        f"保真 {report['fidelity']['result']}"
        f"（{report['fidelity']['n_reports']} 报告，"
        f"{report['fidelity']['n_layers_with_fails']} 层超容差，"
        f"{report['fidelity']['n_stale_reports']} 早于最终构建[依赖确定性]），"
        f"回报 {report['obligations']['result']}，"
        f"物理交叉 {report['crosscheck']['n_groups']} 组（报告制）"
        f"→ {report_path}"
    )
    return 0 if report["result"] == "PASS" else 1


def cmd_export_public(args) -> int:
    """公开版导出：按 visibility 与许可证状态剥离 internal 与退出项，
    产出公开版候选（store + 侧车 + manifest 缺口注记 + 独立结构验证 +
    体积核实与打包建议）。许可证状态门禁已停用（剥离以契约 visibility
    为准），--t1-state 缺省空置分支。"""
    contract = load_contract(args.mapping)
    try:
        validate_contract(contract)
    except ContractViolation as e:
        print(f"[铁律] 契约校验失败：\n{e}", file=sys.stderr)
        return 2
    report = export_public(
        store=args.store, contract=contract, sidecars_dir=args.sidecars,
        out=args.out, out_sidecars=args.out_sidecars,
        t1_state_path=args.t1_state, reports_dir=args.reports,
        dry_run=args.dry_run,
    )
    p = report["plan"]
    mode = "dry-run" if report["dry_run"] else "导出"
    print(
        f"[公开{mode}] 层 {p['n_layers_full']} → {p['n_layers_public']}"
        f"（剥离 {len(p['strip_layers'])} 层 + {len(p['strip_masks'])} 掩膜，"
        f"侧车剥离 {len(p['strip_sidecars'])}）"
    )
    for d in p["t1_decisions"]:
        print(f"[许可状态] {d['dataset']}（{d['status']}）→ {d['decision']}")
    for g in p["gaps"]:
        print(f"[缺口] {g['dataset']}（{g['kind']}）：{g['reason']}")
    for a in p["catalog_actions"]:
        print(
            f"[catalog] id={a['catalog_id']} {a['dataset_dir']}: {a['action']}"
            "（人工执行）"
        )
    if report["dry_run"]:
        ps = report["sizes"]["public"]["store"]
        print(
            f"[体积] 公开 store 估算 {ps['n_bytes'] / 1e9:.2f}GB"
            f"（完整 {report['sizes']['full']['store']['n_bytes'] / 1e9:.2f}GB，"
            "未写盘）"
        )
        return 0
    v = report["validation"]
    s = report["sizes"]
    print(
        f"[写入] {report['public_store']} + {report['public_sidecars']}"
        f"（{len(report['copied_sidecars'])} 侧车，完整性核对通过）"
    )
    print(
        f"[体积] 完整 store {s['full']['store']['n_bytes'] / 1e9:.2f}GB / "
        f"公开 store {s['public']['store']['n_bytes'] / 1e9:.2f}GB；"
        f"完整侧车 {s['full']['sidecars']['n_bytes'] / 1e9:.2f}GB / "
        f"公开侧车 {s['public']['sidecars']['n_bytes'] / 1e9:.2f}GB"
    )
    print(f"[打包] {report['packaging']['recommendation']}")
    print(
        f"[验证] 公开版候选结构验证 {v['result']}"
        f"（{v['passed']}/{v['total_checks']} 项通过）"
        f"→ {v['report']}"
    )
    print("[报告] public-export-report.{json,md}")
    return 0 if report["result"] == "PASS" else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="cubebuild", description="数据立方 v1.0 构建管线")
    p.add_argument("--version", action="version", version=f"cubebuild {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    pb = sub.add_parser("build", help="构建立方（契约 → store → 侧车 → manifest → 验证报告）")
    pb.add_argument("--mapping", default=str(DEFAULT_MAPPING), help="层映射表 YAML")
    pb.add_argument("--out", default=str(DEFAULT_OUT), help="Zarr store 输出目录")
    pb.add_argument("--reports", default=str(DEFAULT_REPORTS), help="报告输出目录")
    pb.add_argument("--sidecars", default=str(DEFAULT_SIDECARS), help="矢量侧车输出目录")
    pb.set_defaults(func=cmd_build)

    pv = sub.add_parser("validate", help="对已有 store 重跑结构验证")
    pv.add_argument("--store", required=True, help="Zarr store 目录")
    pv.add_argument("--mapping", default=str(DEFAULT_MAPPING), help="层映射表 YAML")
    pv.add_argument("--reports", default=str(DEFAULT_REPORTS), help="报告输出目录")
    pv.set_defaults(func=cmd_validate)

    pf = sub.add_parser("fidelity", help="保真验证：积分量/分位数/接缝（保真 harness）")
    pf.add_argument("--store", required=True, help="Zarr store 目录")
    pf.add_argument("--layer", required=True, help="层 id（如 topography__bedrock_elevation）")
    pf.add_argument("--src-dir", default=None, help="该层源数据目录（缺省按层解析）")
    pf.add_argument("--mapping", default=str(DEFAULT_MAPPING), help="层映射表 YAML")
    pf.add_argument("--reports", default=str(DEFAULT_REPORTS), help="报告输出目录")
    pf.set_defaults(func=cmd_fidelity)

    pc = sub.add_parser("checksum", help="输出 store 各数组 sha256（确定性复跑比对）")
    pc.add_argument("--store", required=True, help="Zarr store 目录")
    pc.add_argument("--json", default=None, help="可选：写入 JSON 文件")
    pc.set_defaults(func=cmd_checksum)

    pi = sub.add_parser(
        "integrate",
        help="集成验证总验收单：全量结构 + 保真汇总 + 回报四项 + 物理交叉",
    )
    pi.add_argument("--store", required=True, help="Zarr store 目录")
    pi.add_argument("--mapping", default=str(DEFAULT_MAPPING), help="层映射表 YAML")
    pi.add_argument("--reports", default=str(DEFAULT_REPORTS), help="报告输出目录")
    pi.set_defaults(func=cmd_integrate)

    pe = sub.add_parser(
        "export-public",
        help="公开版导出：剥离 internal 与状态退出项 + 缺口注记 + 打包核实",
    )
    pe.add_argument("--store", required=True, help="完整 Zarr store 目录")
    pe.add_argument("--mapping", default=str(DEFAULT_MAPPING), help="层映射表 YAML")
    pe.add_argument("--sidecars", default=str(DEFAULT_SIDECARS), help="完整侧车目录")
    pe.add_argument("--out", default=str(DEFAULT_OUT.parent / "cube-v1.0-public.zarr"),
                    help="公开版候选 store 输出目录")
    pe.add_argument("--out-sidecars", default=str(DEFAULT_SIDECARS.parent / "sidecars-public"),
                    help="公开侧车输出目录")
    pe.add_argument("--t1-state", default=None,
                    help="许可证状态 YAML（门禁停用后缺省空置："
                         "剥离仅按契约 visibility；v1.1 重启用时传入）")
    pe.add_argument("--reports", default=str(DEFAULT_REPORTS), help="报告输出目录")
    pe.add_argument("--dry-run", action="store_true",
                    help="只出剥离计划/决策/估算体积，不写任何产物")
    pe.set_defaults(func=cmd_export_public)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
