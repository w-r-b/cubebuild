"""md 表体 ↔ CSV 字段级门禁（映射为现编号）。

判据：手稿表体（8 张主表 + 附录 A–D）与 paper/tables/*.csv 按映射列逐行逐字段比对；
差异必须全部落在「论文面措辞适配」白名单列内，白名单外的任何差异即失败。行数校验、
手稿列 ⊆ CSV 列（CSV 可多「出处」等底账列）。手稿为人工对齐 CSV 的表体，若无门禁，
manifest 漂移致 CSV 重算后 md 不随动也不报警（已知遗留②）。

来源：比对器逻辑承自旧编号映射专项工具（
该专项工具已退役删除）；本门禁更新至现编号——表 2 = 层—档位矩阵
（转表）、表 2–7 顺延为表 3–8、新增附录 A/C/D——并纳入 pytest。归一函数处理显示差异
（U+2212 负号、×10⁻⁵ 上下标、km² 上标、单位串尾标点）。已知论文面差异按处置分两类
（实测重定，豁免面收窄）：
- 附录 A/D「单位」与附录 D「说明」：形态机械，改**凝缩式规则校验**（`condense_unit`／
  `omit_optional`）——列不再整列豁免，仅放行规则覆盖的形态（单位：手稿＝CSV 或 CSV 串
  的「，…」注记段略去并闭合括号；说明：手稿＝CSV 或整格省略），第三值一律命中；
- 其余宽口径措辞列（表 1／表 8／附录 B 的豁免列）：自由措辞差异不可机械判（如「数据集」
  列 24/24 行差异），维持整列豁免（`adapted`）。

**英文稿变体**：同一接缝的第二个比对面——
英文稿表体 ↔ 同一套中文底账 CSV，列级口径三类 strict/numeric/adapted（定义见文末
「表格门禁英文变体」节），附规则化映射（主档逐档、关联层公共前缀省略、数据 DOI 注记归并）。

运行：pytest tests/test_tables_gate.py（轻量，只读 md 与 CSV）。
"""
from __future__ import annotations

import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import NamedTuple

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "paper" / "scripts"))

try:
    import nsd_tables as nt  # noqa: E402
except ImportError:  # 公开仓不含论文侧脚本与产物：缺件即跳过（沿用缺件跳过惯例）
    nt = None

pytestmark = pytest.mark.skipif(
    nt is None, reason="论文侧表格登记不在本工作区（缺 paper/scripts/nsd_tables.py）")

MD = REPO / "paper" / "manuscript" / "SEDC-manuscript-zh.md"
TABLES = REPO / "paper" / "tables"

_SUP = {"⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4", "⁵": "5", "⁶": "6",
        "⁷": "7", "⁸": "8", "⁹": "9", "⁻": "-", "⁺": "+"}


def norm_cell(v: str) -> str:
    """显示差异归一到可比形：负号、上标数字、×10ⁿ 科学计数。"""
    v = v.replace("−", "-").replace("–", "-")
    v = "".join(_SUP.get(ch, ch) for ch in v)
    return v.replace("×10", "e").replace("e-0", "e-").replace("e+0", "e")


def cell_equal(mv: str, cv: str) -> bool:
    if norm_cell(mv) == norm_cell(cv):
        return True
    try:  # 数值同值不同写法（如 0.01 vs 1e-02 类）按数值比
        a, b = float(norm_cell(mv)), float(norm_cell(cv))
    except ValueError:
        return False
    return abs(a - b) <= 1e-6 * max(1.0, abs(b))


def md_table(lines: list[str], after_pattern: str) -> tuple[list[str], list[list[str]]]:
    """返回 after_pattern 之后第一个 markdown 表的 (header, rows)。"""
    i = next(k for k, l in enumerate(lines) if re.search(after_pattern, l))
    block, started = [], False
    for l in lines[i + 1:]:
        if l.startswith("|"):
            started = True
            block.append(l)
        elif started:
            break
    cells = [[c.strip() for c in b.strip().strip("|").split("|")] for b in block]
    header, rows = cells[0], cells[2:]  # cells[1] 为分隔行
    assert all(len(r) == len(header) for r in rows), after_pattern
    return header, rows


def load_csv(name: str) -> tuple[list[str], list[list[str]]]:
    with open(TABLES / f"{name}.csv", encoding="utf-8", newline="") as fh:
        rd = list(csv.reader(fh))
    return rd[0], rd[1:]


def abbrev(cell_md: str, cell_csv: str) -> str:
    """手稿 pb2002 关联层用「、_」省略公共前缀；比对时把 CSV 侧同样省略。"""
    if "、_" not in cell_md:
        return cell_csv
    ids = cell_csv.split("、")
    if len(ids) < 2:
        return cell_csv
    pre = os.path.commonprefix(ids).rstrip("_")
    return "、".join([ids[0]] + [x[len(pre):] for x in ids[1:]])


def norm_data_doi(cell: str) -> str:
    """registry 数据 DOI 为「（…）」注记形时手稿显示「—（凝缩注记）」——只比前缀。"""
    return "—" if cell.startswith("（") else cell


def md_startswith_dash(v: str) -> str:
    return "—" if v.startswith("—") else v


def condense_unit(mv: str, cv: str) -> str:
    """附录 A/D「单位」凝缩式规则：手稿把 category 单位串的「，…」注记段略去并闭合括号。

    仅当手稿恰为该凝缩形时放行（返回凝缩串）；第三值一律照原串比对（命中）。
    """
    head, sep, _ = cv.partition("，")
    condensed = head + "）" if sep and head.count("（") > head.count("）") else head
    return condensed if mv == condensed else cv


def omit_optional(mv: str, cv: str) -> str:
    """附录 D「说明」省略式规则：手稿按需省略底账注记（整格留空）。

    仅当手稿为空时放行；非空的第三值一律照原串比对（命中）。
    """
    return "" if mv == "" else cv


class TableSpec(NamedTuple):
    """一张表的比对规格：手稿定位（名＋正则）、对应 CSV、列映射、论文面措辞适配列。"""

    label: str
    anchor: str
    csv_name: str
    columns: list[tuple[str, str, object]]
    adapted: set[str]


SPECS = [
    TableSpec("表 1", r"\*\*表 1\*\*", "table1-source-datasets",
     [("序号", "序号", None), ("数据集", "数据集", None), ("域组", "域组", None),
      ("原生分辨率/形态", "原生分辨率/形态", None),
      ("立方去向", "立方去向", None), ("DOI", "DOI", None)],
     {"数据集", "原生分辨率/形态"}),
    TableSpec("表 2", r"\*\*表 2\*\*", "table2-tier-matrix",
     [("条目", "条目", None), ("1°", "1°", None), ("30′", "30′", None),
      ("6′", "6′", None), ("3′", "3′", None), ("30″", "30″", None)],
     set()),
    TableSpec("表 3", r"\*\*表 3\*\*", "table3-scale-volume",
     [("类别", "类别", None), ("条目", "条目", None), ("数值", "数值", None)],
     set()),
    TableSpec("表 4", r"\*\*表 4\*\*", "table4-sidecar-inventory",
     [("侧车", "侧车", None), ("源数据集", "源数据集", None), ("格式", "格式", None),
      ("要素数", "要素数", None), ("可见性", "可见性", None),
      ("关联层", "关联层", abbrev)],
     set()),
    TableSpec("表 5", r"\*\*表 5\*\*", "table5-validation-summary",
     [("验证家族", "验证家族", None), ("对象", "对象", None),
      ("判据", "判据", None), ("结果", "结果", None)],
     set()),
    TableSpec("表 6", r"\*\*表 6\*\*", "table6-known-biases",
     [("已知偏差", "已知偏差", None), ("机制 / 实测特征", "机制 / 实测特征", None),
      ("影响与使用者动作", "影响与使用者动作", None)],
     set()),
    TableSpec("表 7", r"\*\*表 7\*\*", "table7-task-quickref",
     [("任务", "任务", None), ("硬约束", "硬约束", None),
      ("操作规则", "操作规则", None)],
     set()),
    TableSpec("表 8", r"\*\*表 8\*\*", "table8-internal-exclusions",
     [("分组", "分组", None), ("涉及数据集", "涉及数据集", None),
      ("层数", "层数", None), ("侧车", "侧车数", None),
      ("排除依据", "依据", None), ("原源获取入口", "读者指引", None)],
     {"分组", "涉及数据集", "排除依据", "原源获取入口"}),
    TableSpec("附录 A", r"^## 附录 A", "appendix-a-layer-registry",
     [("#", "序号", None), ("层 id", "层 id", None), ("域组", "域组", None),
      ("可见性", "可见性", None), ("维度", "维度", None), ("档位", "档位", None),
      ("主档", "主档", None), ("dtype", "dtype", None),
      ("单位", "单位", condense_unit), ("重采样核", "重采样核", None)],
     set()),
    TableSpec("附录 B", r"^## 附录 B", "appendix-b-dataset-licences",
     [("#", "序号", None), ("数据集", "数据集", None), ("版本", "版本", None),
      ("论文 DOI", "论文 DOI", None), ("数据 DOI", "数据 DOI", None),
      ("许可依据", "许可依据", None), ("原源获取指引", "读者原源指引", None)],
     {"数据集", "版本", "许可依据", "原源获取指引"}),
    TableSpec("附录 C", r"^## 附录 C", "appendix-c-layer-coverage",
     [("层 id", "层 id", None), ("可见性", "可见性", None), ("主档", "主档", None),
      ("1°", "1° 覆盖率", None), ("30′", "30′ 覆盖率", None), ("6′", "6′ 覆盖率", None),
      ("3′", "3′ 覆盖率", None), ("30″", "30″ 覆盖率", None)],
     set()),
    TableSpec("附录 D", r"^## 附录 D", "appendix-d-layer-stats",
     [("层 id", "layer_id", None), ("域组", "domain_group", None),
      ("可见性", "visibility", None), ("主档", "home_tier", None),
      ("单位", "unit", condense_unit), ("min", "min", None), ("max", "max", None),
      ("mean", "mean", None), ("p50", "p50", None),
      ("缺测占比", "nan_frac", None), ("说明", "notes", omit_optional)],
     set()),
]


def compare(md_text: str) -> list[str]:
    """逐表比对，返回白名单外的差异清单（空 = PASS）。"""
    lines = md_text.split("\n")
    findings: list[str] = []
    for spec in SPECS:
        tab, mapping, adapted = spec.label, spec.columns, spec.adapted
        mdh, mdr = md_table(lines, spec.anchor)
        ch, cr = load_csv(spec.csv_name)
        for mcol, ccol, _ in mapping:
            assert mcol in mdh, f"{tab}: 手稿无列 {mcol}（有 {mdh}）"
            assert ccol in ch, f"{tab}: CSV 无列 {ccol}（有 {ch}）"
        if len(mdr) != len(cr):
            findings.append(f"{tab}: 行数 手稿 {len(mdr)} ≠ CSV {len(cr)}")
            continue
        mi = {c: mdh.index(c) for c, _, _ in mapping}
        ci = {c: ch.index(c) for _, c, _ in mapping}
        for r, (mrow, crow) in enumerate(zip(mdr, cr), 1):
            for mcol, ccol, fn in mapping:
                if mcol in adapted:
                    continue
                mv, cv = mrow[mi[mcol]], crow[ci[ccol]]
                if fn:
                    cv = fn(mv, cv)
                if mcol == "数据 DOI":  # 附录 B 特例：registry「（…」→手稿「—（…」
                    cv, mv = norm_data_doi(cv), md_startswith_dash(mv)
                if not cell_equal(mv, cv):
                    findings.append(f"{tab}: 行{r}「{mcol}」手稿={mv!r} CSV={cv!r}")
    return findings


def kernel_label_findings(labels: dict[str, str]) -> list[str]:
    """附录 A 核列 vs 注册表标签逐层比对（findings 空 = PASS）。"""
    manifest = json.loads(
        (REPO / "products" / "cube-v1.0.zarr" / "cube_manifest.json").read_text(encoding="utf-8"))
    ch, rows = load_csv("appendix-a-layer-registry")
    ji, jk = ch.index("层 id"), ch.index("重采样核")
    got = {r[ji]: r[jk] for r in rows}
    out = []
    for layer in manifest["layers"]:
        resampling = layer["resampling"]
        expect = labels.get(resampling, resampling)
        if got[layer["id"]] != expect:
            out.append(f"{layer['id']}: 附录 A 核列 {got[layer['id']]!r} "
                       f"≠ KERNEL_LABEL[{resampling!r}]={expect!r}")
    return out


def _kernel_labels() -> dict[str, str]:
    sys.path.insert(0, str(REPO / "paper" / "scripts"))
    import common  # noqa: PLC0415

    return common.KERNEL_LABEL


def test_kernel_labels_match_registry():
    """核列标签与注册表同源（闭合 common.KERNEL_LABEL → 附录 A CSV 段）。

    标签改一处（common.py）而忘重跑生成器时，CSV 会停在旧文本；本条与
    test_current_tables_clean（CSV↔md）合起来完成 common.py → md 的传递链。
    """
    findings = kernel_label_findings(_kernel_labels())
    assert findings == [], (
        "核列标签与注册表不符（改 common.py 后需重跑 make_appendix_abc.py）：\n"
        + "\n".join(findings))


def test_kernel_label_drift_detected():
    """真实基底变异负例：none 标签回退为旧文本 → 命中（提示重跑生成器）。"""
    labels = dict(_kernel_labels())
    labels["none"] = "无（不跨档）"
    findings = kernel_label_findings(labels)
    assert findings and all("'none'" in f for f in findings), findings


def test_current_tables_clean():
    """当前手稿全部表体与 CSV 一致（白名单外零差异）。"""
    findings = compare(MD.read_text(encoding="utf-8"))
    assert findings == [], "md↔CSV 白名单外差异：\n" + "\n".join(findings)


def test_mutated_table_number_detected():
    """真实基底变异负例：改表 2 像元数一格 → 门禁命中。"""
    text = MD.read_text(encoding="utf-8")
    assert "| 64,800 |" in text
    mutated = text.replace("| 64,800 |", "| 64,801 |", 1)
    findings = compare(mutated)
    assert any(f.startswith("表 2:") and "行2" in f for f in findings), findings


def test_mutated_kernel_label_detected():
    """真实基底变异负例：改附录 A 一行重采样核 → 门禁命中。"""
    text = MD.read_text(encoding="utf-8")
    m = re.search(r"^\| 4 \| seismology__gladm35_vsv \|.*\|$", text, flags=re.M)
    assert m, "附录 A 第 4 行定位失败"
    mutated = text[:m.start()] + re.sub(r"[^|]+ \|$", " 众数 |", m.group(0)) + text[m.end():]
    findings = compare(mutated)
    assert any(f.startswith("附录 A:") and "重采样核" in f for f in findings), findings


def test_condensed_unit_third_value_detected():
    """真实基底变异负例：附录 A 行 39「单位」改第三值（非凝缩形）→ 门禁命中。"""
    text = MD.read_text(encoding="utf-8")
    m = re.search(r"^\| 39 \| lithosphere__crustal_age_class \|.*\|$", text, flags=re.M)
    assert m, "附录 A 行 39 定位失败"
    mutated = text[:m.start()] + re.sub(r"\| category（[^|]*） \| 众数 \|$", "| category | 众数 |",
                                        m.group(0)) + text[m.end():]
    findings = compare(mutated)
    assert any(f.startswith("附录 A:") and "行39" in f and "单位" in f for f in findings), findings


def test_omitted_notes_third_value_detected():
    """真实基底变异负例：附录 D pixel_area 行「说明」填第三值（非省略、非原注记）→ 命中。"""
    text = MD.read_text(encoding="utf-8")
    m = re.search(r"^\| derived__pixel_area \| 古地形与派生层 \|.*\|$", text, flags=re.M)
    assert m, "附录 D pixel_area 行定位失败"
    row = m.group(0)
    mutated_row = row[: row.rfind("|", 0, -1) + 1] + " 伪注记 |"
    assert mutated_row != row
    mutated = text[:m.start()] + mutated_row + text[m.end():]
    findings = compare(mutated)
    assert any(f.startswith("附录 D:") and "行45" in f and "说明" in f for f in findings), findings


def test_exclusion_totals_hold():
    """表 8 数字互核：CSV 层明细合计 = 手稿题注「22 层 + 6 内部侧车」。"""
    ch, cr = load_csv("table8-internal-exclusions")
    tot_l = sum(int(r[ch.index("层数")]) for r in cr)
    tot_s = sum(int(r[ch.index("侧车数")]) for r in cr)
    assert (tot_l, tot_s) == (22, 6), (tot_l, tot_s)


# ============================================================================
# 英文稿表体门禁（表格门禁英文变体）
# ============================================================================
# 判据：英文稿（paper/manuscript/SEDC-manuscript-en.md）表体 ↔ 中文底账 CSV。
# 列级口径三类（「自然语言列按映射/豁免处理；数值与结构列与底账严格相等」口径的
# 机械化展开）：
#   strict  —— 结构/数值列（层 id、数值、可见性、档位记号、dtype、DOI、计数）：显示差异
#              归一（负号/上标/科学计数）＋中英标点转写（、，；：→ 半角）后严格相等；
#   numeric —— 含自然语言注记的数值列（如「层 5（全公开）」/「5 layers (all public)」）：
#              数字 token **多重集**相等（缺失/新增/替换任一即命中；不比较顺序，因中英句
#              序合法不同，如「25 层登记、1° 全集档 25 层」↔「25 layers registered, 25
#              layers in the 1° master tier」）；
#   adapted —— 自然语言列（域组名、单位说明、说明列、措辞列）：整列豁免，改由译英保真
#              检查器（数字/引用/层 id/DOI 集合级 diff）与中文门禁两面兜底。
# 特例规则：附录 A/C/D「主档」的「逐档」↔「per tier」；表 4「关联层」沿用中文门禁的
# 公共前缀省略（英文记形 "、_"→", _"）；附录 B「数据 DOI」沿用「（…」→「—」注记归并。

EN_MD = REPO / "paper" / "manuscript" / "SEDC-manuscript-en.md"

_ZH_PUNCT = str.maketrans({"（": " (", "）": ")", "，": ", ", "、": ", ", "；": "; ", "：": ": "})
_NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?(?:e-?\d+)?")


def norm_en(cell: str) -> str:
    """英文稿显示归一：中英标点转写 + 既有显示归一（负号/上标/科学计数）。"""
    return re.sub(r"\s+", " ", norm_cell(cell.translate(_ZH_PUNCT))).strip()


def num_tokens(cell: str) -> list[str]:
    """数字 token 多重集（去千分位；科学计数与上标先经 norm_cell 归一）。"""
    return sorted(t.replace(",", "") for t in _NUM_RE.findall(norm_cell(cell)))


def strict_en_equal(mv: str, cv: str) -> bool:
    a, b = norm_en(mv), norm_en(cv)
    if a == b:
        return True
    try:
        return abs(float(a) - float(b)) <= 1e-6 * max(1.0, abs(float(b)))
    except ValueError:
        return False


def norm_home_tier(cv: str) -> str:
    """主档列：底账「逐档」在英文稿记作「per tier」（按档解析计算，无单一主档）。"""
    return "per tier" if cv == "逐档" else cv


def norm_doi_cell(cv: str) -> str:
    """DOI 列的非 DOI 格：CRUST1.0 摘要引用「EGU2013-2658（摘要）」的中文注记译作 "(abstract)"。"""
    return cv.replace("（摘要）", " (abstract)")


def abbrev_en(mv: str, cv: str) -> str:
    """表 4 关联层：英文稿记形 "、_"→", _" 省略公共前缀；比对时把 CSV 侧同样省略。"""
    if ", _" not in mv:
        return cv.replace("、", ", ")
    ids = cv.split("、")
    if len(ids) < 2:
        return cv
    pre = os.path.commonprefix(ids).rstrip("_")
    return ", ".join([ids[0]] + [x[len(pre):] for x in ids[1:]])


class EnCol(NamedTuple):
    md: str
    csv: str
    kind: str  # strict | numeric | adapted


class EnSpec(NamedTuple):
    label: str
    anchor: str
    csv_name: str
    columns: list[EnCol]


EN_SPECS = [
    EnSpec("表 1", r"\*\*Table 1\.\*\*", "table1-source-datasets",
           [EnCol("No.", "序号", "strict"), EnCol("Dataset", "数据集", "adapted"),
            EnCol("Domain group", "域组", "adapted"),
            EnCol("Native resolution/form", "原生分辨率/形态", "numeric"),
            EnCol("Cube destination", "立方去向", "numeric"), EnCol("DOI", "DOI", "strict")]),
    EnSpec("表 2", r"\*\*Table 2\.\*\*", "table2-tier-matrix",
           [EnCol("Item", "条目", "adapted"), EnCol("1°", "1°", "strict"),
            EnCol("30′", "30′", "strict"), EnCol("6′", "6′", "strict"),
            EnCol("3′", "3′", "strict"), EnCol("30″", "30″", "strict")]),
    EnSpec("表 3", r"\*\*Table 3\.\*\*", "table3-scale-volume",
           [EnCol("Category", "类别", "adapted"), EnCol("Item", "条目", "adapted"),
            EnCol("Value", "数值", "numeric")]),
    EnSpec("表 4", r"\*\*Table 4\.\*\*", "table4-sidecar-inventory",
           [EnCol("Sidecar", "侧车", "strict"), EnCol("Source dataset", "源数据集", "adapted"),
            EnCol("Format", "格式", "adapted"), EnCol("Feature count", "要素数", "numeric"),
            EnCol("Visibility", "可见性", "strict"), EnCol("Related layers", "关联层", "abbrev")]),
    EnSpec("表 5", r"\*\*Table 5\.\*\*", "table5-validation-summary",
           [EnCol("Validation family", "验证家族", "adapted"), EnCol("Object", "对象", "numeric"),
            EnCol("Criterion", "判据", "numeric"), EnCol("Result", "结果", "numeric")]),
    EnSpec("表 6", r"\*\*Table 6\.\*\*", "table6-known-biases",
           [EnCol("Known bias", "已知偏差", "numeric"),
            EnCol("Mechanism / measured signature", "机制 / 实测特征", "numeric"),
            EnCol("Impact and user action", "影响与使用者动作", "numeric")]),
    EnSpec("表 7", r"\*\*Table 7\.\*\*", "table7-task-quickref",
           [EnCol("Task", "任务", "adapted"), EnCol("Hard constraint", "硬约束", "numeric"),
            EnCol("Operating rule", "操作规则", "numeric")]),
    EnSpec("表 8", r"\*\*Table 8\.\*\*", "table8-internal-exclusions",
           [EnCol("Group", "分组", "adapted"), EnCol("Datasets involved", "涉及数据集", "adapted"),
            EnCol("Layers", "层数", "strict"), EnCol("Sidecars", "侧车数", "strict"),
            EnCol("Basis for exclusion", "依据", "adapted"),
            EnCol("Access to the original sources", "读者指引", "adapted")]),
    EnSpec("附录 A", r"^## Appendix A", "appendix-a-layer-registry",
           [EnCol("#", "序号", "strict"), EnCol("Layer id", "层 id", "strict"),
            EnCol("Domain group", "域组", "adapted"), EnCol("Visibility", "可见性", "strict"),
            EnCol("Dimensions", "维度", "numeric"), EnCol("Tiers", "档位", "strict"),
            EnCol("Home tier", "主档", "home_tier"), EnCol("dtype", "dtype", "strict"),
            EnCol("Unit", "单位", "adapted"), EnCol("Resampling kernel", "重采样核", "adapted")]),
    EnSpec("附录 B", r"^## Appendix B", "appendix-b-dataset-licences",
           [EnCol("#", "序号", "strict"), EnCol("Dataset", "数据集", "adapted"),
            EnCol("Version", "版本", "adapted"), EnCol("Paper DOI", "论文 DOI", "strict"),
            EnCol("Data DOI", "数据 DOI", "strict"), EnCol("Licence basis", "许可依据", "adapted"),
            EnCol("Access to the original sources", "读者原源指引", "adapted")]),
    EnSpec("附录 C", r"^## Appendix C", "appendix-c-layer-coverage",
           [EnCol("Layer id", "层 id", "strict"), EnCol("Visibility", "可见性", "strict"),
            EnCol("Home tier", "主档", "home_tier"),
            EnCol("1°", "1° 覆盖率", "strict"), EnCol("30′", "30′ 覆盖率", "strict"),
            EnCol("6′", "6′ 覆盖率", "strict"), EnCol("3′", "3′ 覆盖率", "strict"),
            EnCol("30″", "30″ 覆盖率", "strict")]),
    EnSpec("附录 D", r"^## Appendix D", "appendix-d-layer-stats",
           [EnCol("Layer id", "layer_id", "strict"), EnCol("Domain group", "domain_group", "adapted"),
            EnCol("Visibility", "visibility", "strict"), EnCol("Home tier", "home_tier", "home_tier"),
            EnCol("Unit", "unit", "adapted"), EnCol("min", "min", "strict"),
            EnCol("max", "max", "strict"), EnCol("mean", "mean", "strict"),
            EnCol("p50", "p50", "strict"), EnCol("Missing fraction", "nan_frac", "strict"),
            EnCol("Notes", "notes", "adapted")]),
]


def compare_en(md_text: str) -> list[str]:
    """英文稿逐表比对，返回口径外差异清单（空 = PASS）。"""
    lines = md_text.split("\n")
    findings: list[str] = []
    for spec in EN_SPECS:
        tab = spec.label
        mdh, mdr = md_table(lines, spec.anchor)
        ch, cr = load_csv(spec.csv_name)
        for col in spec.columns:
            assert col.md in mdh, f"{tab}(en): 手稿无列 {col.md}（有 {mdh}）"
            assert col.csv in ch, f"{tab}(en): CSV 无列 {col.csv}（有 {ch}）"
        if len(mdr) != len(cr):
            findings.append(f"{tab}(en): 行数 手稿 {len(mdr)} ≠ CSV {len(cr)}")
            continue
        mi = {c.md: mdh.index(c.md) for c in spec.columns}
        ci = {c.csv: ch.index(c.csv) for c in spec.columns}
        for r, (mrow, crow) in enumerate(zip(mdr, cr), 1):
            for col in spec.columns:
                if col.kind == "adapted":
                    continue
                mv, cv = mrow[mi[col.md]], crow[ci[col.csv]]
                if col.csv in ("DOI", "论文 DOI", "数据 DOI"):
                    cv = norm_doi_cell(cv)
                if col.kind == "home_tier":
                    cv = norm_home_tier(cv)
                elif col.kind == "abbrev":
                    cv = abbrev_en(mv, cv)
                elif col.csv == "数据 DOI":  # 附录 B 特例：registry「（…」→手稿「—（…」
                    cv, mv = norm_data_doi(cv), md_startswith_dash(mv)
                if col.kind == "numeric":
                    if num_tokens(mv) != num_tokens(cv):
                        findings.append(
                            f"{tab}(en): 行{r}「{col.md}」数字 token 不等 "
                            f"手稿={num_tokens(mv)} CSV={num_tokens(cv)}")
                elif not strict_en_equal(mv, cv):
                    findings.append(f"{tab}(en): 行{r}「{col.md}」手稿={mv!r} CSV={cv!r}")
    return findings


# ---------- 英文稿门禁：现稿零差异 + 变异负例 ----------

def test_en_current_tables_clean():
    """英文稿全部表体与底账 CSV 在口径内一致（adapted 列与规则化映射之外零差异）。"""
    findings = compare_en(EN_MD.read_text(encoding="utf-8"))
    assert findings == [], "英文稿 md↔CSV 口径外差异：\n" + "\n".join(findings)


def test_en_mutated_tier_number_detected():
    """英文基底变异负例：表 2 像元数一格（64,800→64,801）→ 门禁命中。"""
    text = EN_MD.read_text(encoding="utf-8")
    assert "| Pixels | 64,800 |" in text
    findings = compare_en(text.replace("| Pixels | 64,800 |", "| Pixels | 64,801 |", 1))
    assert any(f.startswith("表 2(en):") and "行2" in f for f in findings), findings


def test_en_mutated_layer_id_detected():
    """英文基底变异负例：附录 A 层 id 大小写篡改（vsv→VSV）→ 命中（层 id 列 strict）。"""
    text = EN_MD.read_text(encoding="utf-8")
    assert "seismology__gladm35_vsv" in text
    findings = compare_en(text.replace("seismology__gladm35_vsv", "seismology__gladm35_VSV", 1))
    assert any(f.startswith("附录 A(en):") and "Layer id" in f for f in findings), findings


def test_en_mutated_visibility_detected():
    """英文基底变异负例：附录 A 可见性 public→internal → 命中（可见性列 strict）。"""
    text = EN_MD.read_text(encoding="utf-8")
    mutated = re.sub(
        r"^\| 1 \| topography__bedrock_elevation \| [^|]*\| public \|",
        "| 1 | topography__bedrock_elevation | Topography, gravity and magnetics | internal |",
        text, count=1, flags=re.M)
    assert mutated != text
    findings = compare_en(mutated)
    assert any(f.startswith("附录 A(en):") and "Visibility" in f for f in findings), findings


def test_en_missing_number_in_numeric_col_detected():
    """英文基底变异负例：表 3 数值列丢一个数字（13,695）→ 命中（numeric 多重集判据）。"""
    text = EN_MD.read_text(encoding="utf-8")
    assert "| Volume | Full store | 12.79 GB (13,695 files) |" in text
    mutated = text.replace("| Volume | Full store | 12.79 GB (13,695 files) |",
                           "| Volume | Full store | 12.79 GB (files) |", 1)
    findings = compare_en(mutated)
    assert any(f.startswith("表 3(en):") and "数字 token 不等" in f for f in findings), findings


def test_en_mutated_doi_detected():
    """英文基底变异负例：附录 B 数据 DOI 末位篡改（SEMUCB-WM1 行，仅附录 B 出现）→ 命中。"""
    text = EN_MD.read_text(encoding="utf-8")
    assert "10.17611/dp/emc.2023.semucbwm1.1" in text
    findings = compare_en(text.replace("10.17611/dp/emc.2023.semucbwm1.1",
                                       "10.17611/dp/emc.2023.semucbwm1.2", 1))
    assert any(f.startswith("附录 B(en):") and "DOI" in f for f in findings), findings


def test_en_home_tier_mapping_and_mutation():
    """附录 A「逐档→per tier」为登记在案的映射；改回档位记号即命中（第三值不放行）。"""
    text = EN_MD.read_text(encoding="utf-8")
    assert "| per tier | float32 | km² | Analytical formula |" in text
    mutated = text.replace("| per tier | float32 | km² | Analytical formula |",
                           "| 1° | float32 | km² | Analytical formula |", 1)
    assert mutated != text
    findings = compare_en(mutated)
    assert any(f.startswith("附录 A(en):") and "Home tier" in f for f in findings), findings


def test_en_adapted_column_exempt_by_design():
    """adapted 列按 spec 口径整列豁免：改表 1「Dataset」列措辞不产生发现（豁免面显式登记）。"""
    text = EN_MD.read_text(encoding="utf-8")
    mutated = text.replace(
        "| 1 | ETOPO 2022 Bedrock (NOAA National Centers for Environmental Information, 2022) |",
        "| 1 | ETOPO 2022 Bedrock elevation (NOAA NCEI, 2022) |", 1)
    assert mutated != text
    assert compare_en(mutated) == []

# ============================================================================
# NSD 稿表体门禁（表格门禁 NSD 变体）
# ============================================================================
# 判据：NSD 稿表体 ↔ 底账 CSV（列级口径沿用英文变体：strict/numeric/adapted 与
# 规则化映射）；表号↔CSV 键的映射由 `nsd_tables.TABLE_REGISTRY` 登记守护——
# 锚点、CSV 与表号三者同源，表号漂移即命中；另核「首现序＝数字序」（判据②）。
# 注：`nsd_tables` 导入与缺件跳过守卫在模块顶部（守卫作用于整模块）。
NSD_MD = REPO / "paper" / "manuscript" / "SEDC-manuscript-nsd-en.md"


def _nsd_specs() -> list[EnSpec]:
    """按登记表号把英文变体的列规格重锚到 NSD 稿（表体内容与英文稿逐字一致）。"""
    base = {spec.csv_name: spec for spec in EN_SPECS}
    specs: list[EnSpec] = []
    for entry in sorted(nt.TABLE_REGISTRY, key=lambda e: e.number):
        src = base[entry.csv_name]
        specs.append(EnSpec(f"表 {entry.number}", rf"\*\*Table {entry.number}\.\*\*",
                            entry.csv_name, src.columns))
    return specs


def compare_nsd(md_text: str) -> list[str]:
    """NSD 稿逐表比对（列口径与英文变体同一实现，仅锚点/表号按登记映射）。

    编号引用上标（`^N^`）是结构产物，先剥离再取 token——否则胞内上标编号会被计入
    数字多重集（实测假阳性：底账 `Sect. 2.5` ↔ NSD `…subsection)^30^`）。
    """
    md_text = re.sub(r"\^[\d,]+\^", "", md_text)
    lines = md_text.split("\n")
    findings: list[str] = []
    for spec in _nsd_specs():
        tab = spec.label
        mdh, mdr = md_table(lines, spec.anchor)
        ch, cr = load_csv(spec.csv_name)
        for col in spec.columns:
            assert col.md in mdh, f"{tab}(nsd): 手稿无列 {col.md}（有 {mdh}）"
            assert col.csv in ch, f"{tab}(nsd): CSV 无列 {col.csv}（有 {ch}）"
        if len(mdr) != len(cr):
            findings.append(f"{tab}(nsd): 行数 手稿 {len(mdr)} ≠ CSV {len(cr)}")
            continue
        mi = {c.md: mdh.index(c.md) for c in spec.columns}
        ci = {c.csv: ch.index(c.csv) for c in spec.columns}
        for r, (mrow, crow) in enumerate(zip(mdr, cr), 1):
            for col in spec.columns:
                if col.kind == "adapted":
                    continue
                mv, cv = mrow[mi[col.md]], crow[ci[col.csv]]
                if col.csv in ("DOI", "论文 DOI", "数据 DOI"):
                    cv = norm_doi_cell(cv)
                if col.kind == "home_tier":
                    cv = norm_home_tier(cv)
                elif col.kind == "abbrev":
                    cv = abbrev_en(mv, cv)
                elif col.csv == "数据 DOI":
                    cv, mv = norm_data_doi(cv), md_startswith_dash(mv)
                if col.kind == "numeric":
                    if num_tokens(mv) != num_tokens(cv):
                        findings.append(f"{tab}(nsd): 行{r}「{col.md}」数字 token 不等 "
                                        f"手稿={num_tokens(mv)} CSV={num_tokens(cv)}")
                elif not strict_en_equal(mv, cv):
                    findings.append(f"{tab}(nsd): 行{r}「{col.md}」手稿={mv!r} CSV={cv!r}")
    return findings


# 结构差异白名单（逐格登记）：表胞内的节号指针改节名 / author-date 引用改上标编号，
# 使底账侧的节号与年份 token 从胞内消失。多余登记与陈旧登记均判失败（见下测试）。
NSD_STRUCTURAL_CELL_DIFFS: set[tuple[str, int, str]] = {
    ("表 6", 14, "Result"),                        # Sect. 4.4 → the Physical cross-check subsection
    ("表 7", 3, "Mechanism / measured signature"),  # Sect. 2.5 → the Special data channels subsection
    ("表 7", 5, "Mechanism / measured signature"),  # 同上
    ("表 8", 5, "Hard constraint"),                 # Sect. 2.5 → the Special data channels subsection
    ("表 8", 6, "Operating rule"),                  # (Meyer and Pebesma, 2021) → 上标编号
    ("表 7", 5, "Impact and user action"),          # (表 4) → (Table 3)：重编号后与底账「表 4」token 不等
    ("表 7", 7, "Mechanism / measured signature"),  # 润色：去破折号并紧凑化，胞内 6′ 出现次数减一（数字 token 集变）
}


def _registered_nsd(finding: str) -> bool:
    return any(f"{tab}(nsd): 行{r}「{col}」" in finding
               for tab, r, col in NSD_STRUCTURAL_CELL_DIFFS)


def test_nsd_tables_clean():
    """NSD 稿全部主表与底账 CSV 在口径内一致（白名单逐格登记之外零差异）。"""
    findings = compare_nsd(NSD_MD.read_text(encoding="utf-8"))
    unregistered = [f for f in findings if not _registered_nsd(f)]
    assert unregistered == [], "NSD 稿 md↔CSV 白名单外差异：\n" + "\n".join(unregistered)
    assert len(findings) == len(NSD_STRUCTURAL_CELL_DIFFS), (
        f"登记 {len(NSD_STRUCTURAL_CELL_DIFFS)} 格、实测 {len(findings)} 格——"
        f"存在陈旧登记：{sorted(NSD_STRUCTURAL_CELL_DIFFS)}")


def test_nsd_table_numbering_consistent():
    """表号↔CSV 键映射 + 首现序＝数字序（nsd_tables 判据①②）零发现。"""
    findings = nt.numbering_findings(NSD_MD.read_text(encoding="utf-8"))
    assert findings == [], findings
    assert nt.mention_order(NSD_MD.read_text(encoding="utf-8")) == \
        sorted(e.number for e in nt.TABLE_REGISTRY)


def test_nsd_mutated_cell_detected():
    """负例：NSD 稿表体一格（64,800→64,801）→ 命中（表体仍与底账闭链）。"""
    text = NSD_MD.read_text(encoding="utf-8")
    assert "| Pixels | 64,800 |" in text
    findings = compare_nsd(text.replace("| Pixels | 64,800 |", "| Pixels | 64,801 |", 1))
    assert any(f.startswith("表 2(nsd):") and "行2" in f for f in findings), findings


def test_nsd_numbering_drift_detected():
    """负例：把某处表号提及改错（映射错位/首现序漂移）→ 命中。"""
    text = NSD_MD.read_text(encoding="utf-8")
    findings = nt.numbering_findings(text.replace("in Table 5.", "in Table 6.", 1))
    assert any("首现序" in f or "题注表号集合" in f or "孤儿" in f for f in findings), findings


def test_supplementary_tables_match_baseline():
    """Supplementary 表产物与 ESSD 英文稿附录表体逐字段一致（载体迁移零手工转录）。"""
    src = (REPO / "paper" / "manuscript" / "SEDC-manuscript-en.md").read_text(encoding="utf-8")
    lines = src.split("\n")
    anchors = {
        "registry": r"^## Appendix A", "coverage": r"^## Appendix C",
        "stats": r"^## Appendix D", "licences": r"^## Appendix B",
    }
    checked = 0
    for entry in nt.SUPP_REGISTRY:
        if entry.key == "licences" and nt.LICENCE_ROLE == "main":
            continue
        header, rows = md_table(lines, anchors[entry.key])
        path = nt.SUPPLEMENTARY_DIR / entry.filename
        assert path.exists(), f"缺 Supplementary 产物：{path.name}"
        with open(path, encoding="utf-8", newline="") as fh:
            got = list(csv.reader(fh))
        assert got[0] == header, (entry.filename, got[0][:3], header[:3])
        assert got[1:] == rows, f"{entry.filename} 表体与基线附录表不一致"
        checked += 1
    assert checked == (3 if nt.LICENCE_ROLE == "main" else 4), checked


def test_renumber_single_pass_keeps_caption_markers():
    """renumber 机械改写回归题注与正文提及
    按首现序**一次**映射——粗体标记 `**` 不得被吃、编号不得二次映射；且幂等。"""
    text = ("**Table 2.** second caption.\n\n**Table 1.** first caption.\n\n"
            "See Table 3, Table 4, Table 5, Table 6, Table 7 and Table 8.\n")
    out = nt.renumber(text)
    assert out.startswith("**Table 1.** second caption."), out[:60]
    assert "**Table 2.** first caption." in out
    assert "See Table 3, Table 4, Table 5, Table 6, Table 7 and Table 8." in out
    assert nt.caption_numbers(out) == [1, 2]
    assert nt.renumber(out) == out  # 幂等（首现序已＝数字序）
