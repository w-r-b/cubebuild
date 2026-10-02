"""集成验证总报告：立方 v1.0 质量的总验收单。

四件套：
1. 结构层：validate_store 全量（47 层 × YAML 契约逐条 + 1° 全集档完整性
   + 铁律复检 + 同档配准）——validate.py 承担，本模块汇总其结果；
2. 保真层：既有逐层保真报告（fidelity-*.json，各层自供）的
   汇总——逐层列出超容差（FAIL）项；不重跑（距离场类报告单次耗时小时级，
   汇总以报告为口径，附时效性核对）；
3. 物理交叉四组：crosscheck.py 承担（只报告不拦截）；
4. 构建回报四项：GHFDB/NGHF 整编核实结论、Slab2 重叠像元总量、LITHO1 nc
   变量清单、各层覆盖率登记——自既有回报制品提取汇总。

总结果判据：结构验证 PASS ∧ 保真汇总无 FAIL ∧ 回报四项齐备；物理交叉
不参与判据（：只报告不拦截，但组执行失败=无报告可出，属交付缺失，
计 FAIL）。
"""

import json
from pathlib import Path

from .contract import Contract
from .crosscheck import (
    assemble_crosscheck_groups,
    write_crosscheck_report,
)
from .manifest import read_manifest
from .validate import validate_store, write_report

# 保真报告覆盖例外：该层的保真不由 fidelity-* 报告承担
FIDELITY_COVERAGE_EXCEPTIONS = {
    "derived__pixel_area": "由结构验证承担（数值=解析公式逐档核对 + 全球面积守恒）",
}

# 检查项家族归类（积分量/分位数/众数一致性/计数守恒/距离抽样……汇总）
_FAMILY_RULES = (
    ("积分量", ("积分量守恒", "积分恒等式")),
    ("分位数", ("分位数",)),
    ("众数一致性", ("众数",)),
    ("计数守恒", ("计数守恒", "sum 一致性")),
    ("距离抽样", ("抽样",)),
    ("逐位一致", ("逐位一致", "位级")),
    ("有效位", ("有效位",)),
    ("值域", ("值域",)),
    ("两通道一致", ("两通道一致", "掩膜")),
    ("覆盖率", ("覆盖率", "覆盖", "分布")),
)


def _add_check(checks: list, name: str, status: str, summary: str) -> None:
    checks.append({"name": name, "status": status, "summary": summary})


def _families(check_names: list[str]) -> list[str]:
    fams = []
    for fam, keys in _FAMILY_RULES:
        if any(any(k in n for k in keys) for n in check_names):
            fams.append(fam)
    return fams


# ---------------- 保真汇总 ----------------

def fidelity_summary(reports_dir: str | Path, contract: Contract,
                     store_dir: str | Path) -> dict:
    """逐层保真报告汇总：结果/家族/超容差（FAIL）项逐层列出 + 覆盖 + 时效性。

    时效性（报告制）：早于最终构建的保真报告标记 stale——其结论依赖
    构建确定性（两轮 checksum 实证）成立。构建终点基准 = store 根
    cube_manifest.json mtime（构建管线最后写入，晚于全部数组；zarr.json
    是建库起点，code-review 实证会低估 stale 面）。
    """
    reports_dir = Path(reports_dir)
    per_layer: dict[str, dict] = {}
    for path in sorted(reports_dir.glob("fidelity-*.json")):
        if path.name == "fidelity-summary.json":
            continue          # 本汇总自身不入逐层清单（自我包含防护）
        try:
            rep = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            per_layer[path.stem.removeprefix("fidelity-")] = {
                "result": "UNREADABLE", "error": str(e), "fails": []
            }
            continue
        fails = [
            {"name": c["name"], "summary": c.get("summary", c.get("detail", ""))}
            for c in rep.get("checks", []) if c.get("status") == "FAIL"
        ]
        per_layer[rep.get("layer", path.stem.removeprefix("fidelity-"))] = {
            "result": rep.get("result", "?"),
            "n_checks": rep.get("n_checks", len(rep.get("checks", []))),
            "n_fail": len(fails),
            "families": _families([c["name"] for c in rep.get("checks", [])]),
            "fails": fails,
            "report_mtime": path.stat().st_mtime,
        }

    checks: list[dict] = []
    contract_ids = [e.id for e in contract.layers]
    missing = [
        lid for lid in contract_ids
        if lid not in per_layer and lid not in FIDELITY_COVERAGE_EXCEPTIONS
    ]
    # 双向完备（与 validate.py manifest↔契约互检同哲学）：孤儿报告
    # （报告层不在契约内，如历史遗留）不 FAIL 但显式报告
    orphans = sorted(set(per_layer) - set(contract_ids))
    _add_check(
        checks,
        "保真报告覆盖（契约层 ↔ fidelity 报告）",
        "pass" if not missing else "FAIL",
        (
            f"{len(per_layer)} 份报告 + {len(FIDELITY_COVERAGE_EXCEPTIONS)} 例外"
            f"（{'；'.join(f'{k}: {v}' for k, v in FIDELITY_COVERAGE_EXCEPTIONS.items())}）"
            f"；缺报告层 {missing or '无'}；契约外孤儿报告 {orphans or '无'}"
        ),
    )

    # 时效性：报告早于最终构建 → stale（依赖构建确定性，报告制）
    manifest_meta = Path(store_dir) / "cube_manifest.json"
    store_mtime = manifest_meta.stat().st_mtime if manifest_meta.is_file() else None
    stale = []
    if store_mtime is not None:
        stale = sorted(
            lid for lid, s in per_layer.items()
            if s.get("report_mtime", 0) < store_mtime
        )
    _add_check(
        checks,
        "保真报告时效性（vs 最终构建时间；报告制）",
        "report",
        (
            f"{len(stale)}/{len(per_layer)} 份报告早于最终构建（依赖构建确定性"
            "——两轮 checksum 实证；stale 层：" + ", ".join(stale) + "）"
            if stale else
            f"全部 {len(per_layer)} 份报告不早于最终构建"
        ) if store_mtime is not None else "store cube_manifest.json 不可读，跳过时效核对",
    )

    n_fail_layers = [lid for lid, s in per_layer.items() if s.get("n_fail")]
    unreadable = [lid for lid, s in per_layer.items() if s.get("result") == "UNREADABLE"]
    result = "FAIL" if missing or n_fail_layers or unreadable else "PASS"
    return {
        "result": result,
        "n_contract_layers": len(contract_ids),
        "n_reports": len(per_layer),
        "n_exceptions": len(FIDELITY_COVERAGE_EXCEPTIONS),
        "n_layers_with_fails": len(n_fail_layers),
        "n_stale_reports": len(stale),
        "per_layer": {
            lid: {k: v for k, v in s.items() if k != "report_mtime"}
            for lid, s in sorted(per_layer.items())
        },
        "checks": checks,
    }


def write_fidelity_summary(summary: dict, reports_dir: str | Path) -> Path:
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / "fidelity-summary.json"
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# 保真验证汇总（全量层积分量/分位数/众数/计数/距离抽样汇总）",
        "",
        f"- 结果：**{summary['result']}**"
        f"（{summary['n_reports']} 份逐层报告 + {summary['n_exceptions']} 例外，"
        f"{summary['n_layers_with_fails']} 层含超容差项）",
        "",
        "逐层明细（积分量/分位数等报告制统计量）见各行「明细报告」列的",
        "fidelity-*.json；本汇总聚合硬判据结果与超容差项。",
        "",
        "| 层 | 结果 | 检查数 | FAIL | 检查家族 | 明细报告 |",
        "|---|---|---|---|---|---|",
    ]
    for lid, s in summary["per_layer"].items():
        lines.append(
            f"| {lid} | {s.get('result', '?')} | {s.get('n_checks', '?')} "
            f"| {s.get('n_fail', '?')} | {'/'.join(s.get('families', []))} "
            f"| fidelity-{lid}.json |"
        )
    for lid, note in FIDELITY_COVERAGE_EXCEPTIONS.items():
        lines.append(f"| {lid} | 例外 | — | — | {note} | — |")
    lines += ["", "## 超容差（FAIL）项逐层列出", ""]
    any_fail = False
    for lid, s in summary["per_layer"].items():
        for f in s.get("fails", []):
            any_fail = True
            detail = f.get("summary", "").replace("|", "\\|")
            lines.append(f"- **{lid}** — {f['name']}：{detail}")
    if not any_fail:
        lines.append("无（全部硬判据通过）")
    lines += ["", "## 汇总级检查", ""]
    for c in summary["checks"]:
        lines.append(f"- [{c['status']}] {c['name']}：{c['summary']}")
    (reports_dir / "fidelity-summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return json_path


# ---------------- 构建回报四项 ----------------

def build_obligations(reports_dir: str | Path, store_dir: str | Path) -> dict:
    """构建期回报义务四项汇总（既有制品提取；缺失 = FAIL）。"""
    reports_dir = Path(reports_dir)
    checks: list[dict] = []
    items: dict = {}

    # ① GHFDB/NGHF 整编核实结论（fidelity-thermal__heatflow_density.json）
    thermal_path = reports_dir / "fidelity-thermal__heatflow_density.json"
    try:
        thermal = json.loads(thermal_path.read_text(encoding="utf-8"))
        inc = {
            c["name"]: c.get("summary", "")
            for c in thermal["checks"] if "整编" in c["name"]
        }
        if not inc:
            raise ValueError("报告中无整编核实检查项")
        items["ghfdb_nghf_incorporation"] = {
            "source": thermal_path.name,
            "checks": inc,
        }
        conclusion = next(
            (v for k, v in inc.items() if "结论" in k), ""
        )
        _add_check(
            checks, "回报① GHFDB/NGHF 整编核实结论", "pass",
            f"{len(inc)} 项核实记录；结论：{conclusion[:200]}"
        )
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as e:
        _add_check(checks, "回报① GHFDB/NGHF 整编核实结论", "FAIL", f"{thermal_path.name}: {e}")

    # ② Slab2 重叠像元总量（slab2-merge.json）
    slab2_path = reports_dir / "slab2-merge.json"
    try:
        s2 = json.loads(slab2_path.read_text(encoding="utf-8"))
        keys = ("zones", "covered_3min_pixels", "overlap_3min_pixels",
                "overlap_share_of_coverage")
        missing_keys = [k for k in keys if k not in s2]
        if missing_keys:
            raise ValueError(f"缺字段 {missing_keys}")
        items["slab2_overlap"] = {"source": slab2_path.name, **{k: s2[k] for k in keys}}
        _add_check(
            checks, "回报② Slab2 重叠像元总量", "pass",
            f"{s2['zones']} 带合并：覆盖 3′ 像元 {s2['covered_3min_pixels']}，"
            f"重叠 {s2['overlap_3min_pixels']}"
            f"（{s2['overlap_share_of_coverage']:.2%}，重叠取最浅 depth）"
        )
    except (OSError, ValueError, json.JSONDecodeError) as e:
        _add_check(checks, "回报② Slab2 重叠像元总量", "FAIL", f"{slab2_path.name}: {e}")

    # ③ LITHO1 nc 变量清单（litho1-inventory.json）
    litho_path = reports_dir / "litho1-inventory.json"
    try:
        inv = json.loads(litho_path.read_text(encoding="utf-8"))
        keys = ("n_variables", "ingested", "n_v1_1_candidates")
        missing_keys = [k for k in keys if k not in inv]
        if missing_keys:
            raise ValueError(f"缺字段 {missing_keys}")
        items["litho1_inventory"] = {
            "source": litho_path.name,
            **{k: inv[k] for k in keys},
            "v1_1_note": inv.get("v1_1_note", ""),
        }
        _add_check(
            checks, "回报③ LITHO1 nc 变量清单", "pass",
            f"nc 数据变量 {inv['n_variables']} 项；v1.0 ingest = "
            f"{list(inv['ingested'].values())}；v1.1 候选 "
            f"{inv['n_v1_1_candidates']} 项（清单见 {litho_path.name}）"
        )
    except (OSError, ValueError, json.JSONDecodeError) as e:
        _add_check(checks, "回报③ LITHO1 nc 变量清单", "FAIL", f"{litho_path.name}: {e}")

    # ④ 各层覆盖率登记（cube_manifest.json）
    try:
        manifest = read_manifest(store_dir)
        coverage = {m["id"]: m["coverage"] for m in manifest["layers"]}
        if not coverage:
            raise ValueError("manifest 无层条目")
        by_tier: dict[str, dict] = {}
        for lid, cov in coverage.items():
            for tier, v in cov.items():
                stats = by_tier.setdefault(tier, {"n": 0, "full": 0, "min": 1.0})
                stats["n"] += 1
                stats["full"] += int(v >= 1.0)
                stats["min"] = min(stats["min"], v)
        items["coverage_registry"] = {
            "source": "cube_manifest.json",
            "n_layers": len(coverage),
            "by_tier": by_tier,
            "coverage": coverage,
        }
        tier_str = "；".join(
            f"{t}: {s['full']}/{s['n']} 全覆盖，min {s['min']:.4f}"
            for t, s in sorted(by_tier.items())
        )
        _add_check(
            checks, "回报④ 各层覆盖率登记", "pass",
            f"{len(coverage)} 层逐档登记（{tier_str}；全表见 JSON/manifest）"
        )
    except (OSError, ValueError, KeyError) as e:
        _add_check(checks, "回报④ 各层覆盖率登记", "FAIL", f"cube_manifest: {e}")

    n_fail = sum(1 for c in checks if c["status"] == "FAIL")
    return {
        "result": "FAIL" if n_fail else "PASS",
        "items": items,
        "checks": checks,
    }


def write_obligations_report(obligations: dict, reports_dir: str | Path) -> Path:
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / "build-obligations.json"
    json_path.write_text(
        json.dumps(obligations, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# 构建回报义务汇总（四项）",
        "",
        f"- 结果：**{obligations['result']}**",
        "",
        "| 回报项 | 状态 | 说明 |",
        "|---|---|---|",
    ]
    for c in obligations["checks"]:
        detail = (c["summary"] or "").replace("|", "\\|")
        lines.append(f"| {c['name']} | {c['status']} | {detail} |")
    # 覆盖率全表（各层覆盖率登记；档位列 = 全层档位并集，缺档 = —）
    cov = obligations.get("items", {}).get("coverage_registry", {}).get("coverage")
    if cov:
        tiers = sorted({t for c in cov.values() for t in c})
        lines += [
            "", "## 各层覆盖率登记（cube_manifest）", "",
            "| 层 | " + " | ".join(tiers) + " |",
            "|---|" + "---|" * len(tiers),
        ]
        for lid, c in sorted(cov.items()):
            lines.append(
                f"| {lid} | "
                + " | ".join(f"{c[t]:.4f}" if t in c else "—" for t in tiers)
                + " |"
            )
    (reports_dir / "build-obligations.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return json_path


# ---------------- 集成总报告 ----------------

def build_integration_report(store_dir, contract: Contract,
                             reports_dir: str | Path,
                             crosscheck_src_dirs: dict | None = None) -> dict:
    """总验收单：结构验证（全量）+ 保真汇总 + 构建回报四项 + 物理交叉四组。"""
    reports_dir = Path(reports_dir)

    # 1. 结构验证（全量）——validate_store 为唯一实现，报告落盘 structure-validation.*
    structure = validate_store(store_dir, contract)
    write_report(structure, reports_dir)

    # 2. 保真汇总（既有逐层报告聚合，不重跑）
    fidelity = fidelity_summary(reports_dir, contract, store_dir)
    write_fidelity_summary(fidelity, reports_dir)

    # 3. 构建回报四项
    obligations = build_obligations(reports_dir, store_dir)
    write_obligations_report(obligations, reports_dir)

    # 4. 物理交叉四组（crosscheck 统一装配：只报告不拦截；组执行失败 =
    #    无报告可出，属交付缺失 → FAIL）
    cross_checks = assemble_crosscheck_groups(store_dir, crosscheck_src_dirs)
    crosscheck = {
        "store": str(store_dir),
        "policy": "数值结果只报告不拦截；组执行失败（无报告可出）计 FAIL",
        "n_groups": len(cross_checks),
        "groups": cross_checks,
    }
    write_crosscheck_report(crosscheck, reports_dir)

    result = (
        "PASS"
        if structure["result"] == "PASS"
        and fidelity["result"] == "PASS"
        and obligations["result"] == "PASS"
        and not any(
            c["status"] == "FAIL"
            for g in cross_checks for c in g["checks"]
        )
        else "FAIL"
    )
    report = {
        "result": result,
        "store": str(store_dir),
        "mapping": str(contract.path),
        "structure": {
            "result": structure["result"],
            "passed": structure["passed"],
            "total_checks": structure["total_checks"],
            "failed": structure["failed"],
            "report": "structure-validation.json",
        },
        "fidelity": {
            "result": fidelity["result"],
            "n_reports": fidelity["n_reports"],
            "n_exceptions": fidelity["n_exceptions"],
            "n_layers_with_fails": fidelity["n_layers_with_fails"],
            "n_stale_reports": fidelity["n_stale_reports"],
            "stale_note": (
                "早于最终构建的逐层报告依赖构建确定性（两轮 checksum 实证）"
                if fidelity["n_stale_reports"] else "全部报告不早于最终构建"
            ),
            "report": "fidelity-summary.json",
        },
        "obligations": {
            "result": obligations["result"],
            "report": "build-obligations.json",
        },
        "crosscheck": {
            "policy": crosscheck["policy"],
            "n_groups": crosscheck["n_groups"],
            "report": "physical-crosscheck.json",
        },
    }
    write_integration_report(report, reports_dir)
    return report


def write_integration_report(report: dict, reports_dir: str | Path) -> Path:
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / "integration-report.json"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    s, f, o, x = report["structure"], report["fidelity"], report["obligations"], report["crosscheck"]
    lines = [
        "# 集成验证报告 —— 立方 v1.0 总验收单",
        "",
        f"- 总结果：**{report['result']}**",
        f"- store：`{report['store']}`",
        f"- 契约：`{report['mapping']}`",
        "",
        "| 验收面 | 结果 | 明细 |",
        "|---|---|---|",
        f"| 结构层（全量契约层 × YAML 逐条 + 1° 全集档 + 铁律 + 配准） | **{s['result']}** "
        f"| {s['passed']}/{s['total_checks']} 项通过 → {s['report']} |",
        f"| 保真层（积分量/分位数/众数/计数/距离抽样汇总） | **{f['result']}** "
        f"| {f['n_reports']} 报告 + {f['n_exceptions']} 例外，"
        f"{f['n_layers_with_fails']} 层超容差 → {f['report']} |",
        f"| 构建回报四项（整编核实/Slab2 重叠/LITHO1 清单/覆盖率登记） | **{o['result']}** "
        f"| → {o['report']} |",
        f"| 物理交叉四组（只报告不拦截） | 报告制 "
        f"| {x['n_groups']} 组 → {x['report']} |",
        "",
        "物理交叉与保真超容差明细见各分报告（physical-crosscheck.md / fidelity-summary.md）。",
    ]
    (reports_dir / "integration-report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return json_path
