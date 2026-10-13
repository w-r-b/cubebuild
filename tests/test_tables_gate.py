"""Field-level gate between the md table bodies and the CSV (mapped to the current numbering).

Criterion: compare the manuscript table bodies (8 main tables + Appendices A–D) with
paper/tables/*.csv row by row and field by field along the mapped columns; all differences
must fall within the "paper-facing wording adaptation" whitelist columns, and any difference
outside the whitelist is a failure. Row-count check, and manuscript columns ⊆ CSV columns
(the CSV may carry extra ledger columns such as "source"). The manuscript is a table body
manually aligned to the CSV; without the gate, a manifest drift that triggers a CSV recompute
would leave the md unmoved and unflagged (known leftover ②).

Origin: the comparator logic is inherited from the old-numbering mapping tool (that tool has
been retired and deleted); this gate has been updated to the current numbering — Table 2 =
layer–tier matrix (re-tabled), Tables 2–7 shifted to Tables 3–8, Appendices A/C/D added — and
is brought under pytest. The normalisation functions handle display differences (U+2212 minus
sign, ×10⁻⁵ superscripts/subscripts, km² superscript, trailing punctuation in unit strings).
The known paper-facing differences fall into two classes by disposition (re-determined by
measurement, with a narrowed exemption surface):
- Appendix A/D "Unit" and Appendix D "Notes": the form is mechanical, so they are switched to
  a **condensed-form rule check** (`condense_unit` / `omit_optional`) — the column is no
  longer exempted wholesale, and only the forms covered by the rule are accepted (Unit:
  manuscript = CSV, or the "，…" annotation segment of the CSV string is dropped with the
  parenthesis closed; Notes: manuscript = CSV, or the whole cell is omitted), while any third
  value is always flagged;
- the remaining broad-wording columns (the exempt columns of Table 1 / Table 8 / Appendix B):
  free wording differences cannot be judged mechanically (e.g. all 24/24 rows differ in the
  "Dataset" column), so they remain exempted wholesale (`adapted`).

**English manuscript variant**: the second comparison surface at the same seam — the English
manuscript table bodies ↔ the same set of Chinese ledger CSVs, with column criteria in three
classes strict/numeric/adapted (definitions in the "table gate English variant" section at the
end), plus regularised mappings (per-tier home tier, related-layer common-prefix omission,
data-DOI annotation merging).

Run: pytest tests/test_tables_gate.py (lightweight; reads only md and CSV).
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
except ImportError:
    nt = None

pytestmark = pytest.mark.skipif(
    nt is None, reason="论文侧表格登记不在本工作区（缺 paper/scripts/nsd_tables.py）")

MD = REPO / "paper" / "manuscript" / "SEDC-manuscript-zh.md"
TABLES = REPO / "paper" / "tables"

_SUP = {"⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4", "⁵": "5", "⁶": "6",
        "⁷": "7", "⁸": "8", "⁹": "9", "⁻": "-", "⁺": "+"}


def norm_cell(v: str) -> str:
    """Normalise display differences to a comparable form: minus sign, superscript digits, ×10ⁿ scientific notation."""
    v = v.replace("−", "-").replace("–", "-")
    v = "".join(_SUP.get(ch, ch) for ch in v)
    return v.replace("×10", "e").replace("e-0", "e-").replace("e+0", "e")


def cell_equal(mv: str, cv: str) -> bool:
    if norm_cell(mv) == norm_cell(cv):
        return True
    try:
        a, b = float(norm_cell(mv)), float(norm_cell(cv))
    except ValueError:
        return False
    return abs(a - b) <= 1e-6 * max(1.0, abs(b))


def md_table(lines: list[str], after_pattern: str) -> tuple[list[str], list[list[str]]]:
    """Return the (header, rows) of the first markdown table after after_pattern."""
    i = next(k for k, l in enumerate(lines) if re.search(after_pattern, l))
    block, started = [], False
    for l in lines[i + 1:]:
        if l.startswith("|"):
            started = True
            block.append(l)
        elif started:
            break
    cells = [[c.strip() for c in b.strip().strip("|").split("|")] for b in block]
    header, rows = cells[0], cells[2:]
    assert all(len(r) == len(header) for r in rows), after_pattern
    return header, rows


def load_csv(name: str) -> tuple[list[str], list[list[str]]]:
    with open(TABLES / f"{name}.csv", encoding="utf-8", newline="") as fh:
        rd = list(csv.reader(fh))
    return rd[0], rd[1:]


def abbrev(cell_md: str, cell_csv: str) -> str:
    """The manuscript's pb2002 related-layers column uses "、_" to omit the common prefix; on comparison, abbreviate the CSV side the same way."""
    if "、_" not in cell_md:
        return cell_csv
    ids = cell_csv.split("、")
    if len(ids) < 2:
        return cell_csv
    pre = os.path.commonprefix(ids).rstrip("_")
    return "、".join([ids[0]] + [x[len(pre):] for x in ids[1:]])


def norm_data_doi(cell: str) -> str:
    """When the registry data DOI has the annotation form "（…）", the manuscript shows "—(condensed annotation)"; only the prefix is compared."""
    return "—" if cell.startswith("（") else cell


def md_startswith_dash(v: str) -> str:
    return "—" if v.startswith("—") else v


def condense_unit(mv: str, cv: str) -> str:
    """Condensed-form rule for the "Unit" column in Appendix A/D: the manuscript drops the "，…" annotation segment of the category unit string and closes the parenthesis.

    Only when the manuscript is exactly in this condensed form is it accepted (returning the condensed string); any third value is compared against the original string as-is (and flagged).
    """
    head, sep, _ = cv.partition("，")
    condensed = head + "）" if sep and head.count("（") > head.count("）") else head
    return condensed if mv == condensed else cv


def omit_optional(mv: str, cv: str) -> str:
    """Omission-form rule for the "Notes" column in Appendix D: the manuscript omits the ledger annotation as needed (leaving the whole cell empty).

    Only when the manuscript cell is empty is it accepted; any non-empty third value is compared against the original string as-is (and flagged).
    """
    return "" if mv == "" else cv


class TableSpec(NamedTuple):
    """Comparison spec for one table: manuscript anchor (label + regex), corresponding CSV, column mapping, and paper-facing wording adaptation columns."""

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
    """Compare table by table; return the list of differences outside the whitelist (empty = PASS)."""
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
                if mcol == "数据 DOI":
                    cv, mv = norm_data_doi(cv), md_startswith_dash(mv)
                if not cell_equal(mv, cv):
                    findings.append(f"{tab}: 行{r}「{mcol}」手稿={mv!r} CSV={cv!r}")
    return findings


def kernel_label_findings(labels: dict[str, str]) -> list[str]:
    """Compare the Appendix A kernel column against the registry labels layer by layer (empty findings = PASS)."""
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
    """The kernel-column labels share a single source with the registry (closing the common.KERNEL_LABEL → Appendix A CSV segment chain).

    If a label is changed in one place (common.py) but the generator is not rerun, the CSV stays at the old text; this test together with test_current_tables_clean (CSV↔md) completes the common.py → md propagation chain.
    """
    findings = kernel_label_findings(_kernel_labels())
    assert findings == [], (
        "核列标签与注册表不符（改 common.py 后需重跑 make_appendix_abc.py）：\n"
        + "\n".join(findings))


def test_kernel_label_drift_detected():
    """Real-base mutation negative case: reverting the none label to the old text → flagged (reminding to rerun the generator)."""
    labels = dict(_kernel_labels())
    labels["none"] = "无（不跨档）"
    findings = kernel_label_findings(labels)
    assert findings and all("'none'" in f for f in findings), findings


def test_current_tables_clean():
    """All current manuscript table bodies match the CSV (zero differences outside the whitelist)."""
    findings = compare(MD.read_text(encoding="utf-8"))
    assert findings == [], "md↔CSV 白名单外差异：\n" + "\n".join(findings)


def test_mutated_table_number_detected():
    """Real-base mutation negative case: change one cell of the Table 2 pixel count → the gate flags it."""
    text = MD.read_text(encoding="utf-8")
    assert "| 64,800 |" in text
    mutated = text.replace("| 64,800 |", "| 64,801 |", 1)
    findings = compare(mutated)
    assert any(f.startswith("表 2:") and "行2" in f for f in findings), findings


def test_mutated_kernel_label_detected():
    """Real-base mutation negative case: change the resampling kernel in one Appendix A row → the gate flags it."""
    text = MD.read_text(encoding="utf-8")
    m = re.search(r"^\| 4 \| seismology__gladm35_vsv \|.*\|$", text, flags=re.M)
    assert m, "附录 A 第 4 行定位失败"
    mutated = text[:m.start()] + re.sub(r"[^|]+ \|$", " 众数 |", m.group(0)) + text[m.end():]
    findings = compare(mutated)
    assert any(f.startswith("附录 A:") and "重采样核" in f for f in findings), findings


def test_condensed_unit_third_value_detected():
    """Real-base mutation negative case: change the "Unit" in Appendix A row 39 to a third value (non-condensed form) → the gate flags it."""
    text = MD.read_text(encoding="utf-8")
    m = re.search(r"^\| 39 \| lithosphere__crustal_age_class \|.*\|$", text, flags=re.M)
    assert m, "附录 A 行 39 定位失败"
    mutated = text[:m.start()] + re.sub(r"\| category（[^|]*） \| 众数 \|$", "| category | 众数 |",
                                        m.group(0)) + text[m.end():]
    findings = compare(mutated)
    assert any(f.startswith("附录 A:") and "行39" in f and "单位" in f for f in findings), findings


def test_omitted_notes_third_value_detected():
    """Real-base mutation negative case: fill the "Notes" of the Appendix D pixel_area row with a third value (neither omitted nor the original annotation) → flagged."""
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
    """Table 8 numeric cross-check: CSV layer detail total = manuscript caption "22 layers + 6 internal sidecars"."""
    ch, cr = load_csv("table8-internal-exclusions")
    tot_l = sum(int(r[ch.index("层数")]) for r in cr)
    tot_s = sum(int(r[ch.index("侧车数")]) for r in cr)
    assert (tot_l, tot_s) == (22, 6), (tot_l, tot_s)


# ============================================================================
# ============================================================================
#              layers in the 1° master tier」）；

EN_MD = REPO / "paper" / "manuscript" / "SEDC-manuscript-en.md"

_ZH_PUNCT = str.maketrans({"（": " (", "）": ")", "，": ", ", "、": ", ", "；": "; ", "：": ": "})
_NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?(?:e-?\d+)?")


def norm_en(cell: str) -> str:
    """English-manuscript display normalisation: Chinese-English punctuation transliteration + the existing display normalisation (minus sign / superscripts / scientific notation)."""
    return re.sub(r"\s+", " ", norm_cell(cell.translate(_ZH_PUNCT))).strip()


def num_tokens(cell: str) -> list[str]:
    """Multiset of numeric tokens (thousands separators removed; scientific notation and superscripts first normalised via norm_cell)."""
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
    """Home-tier column: the ledger's tier-range marker is written as "per tier" in the English manuscript (resolved per tier, with no single home tier)."""
    return "per tier" if cv == "逐档" else cv


def norm_doi_cell(cv: str) -> str:
    """Non-DOI cell of the DOI column: the Chinese parenthetical annotation on the CRUST1.0 abstract citation is translated as "(abstract)"."""
    return cv.replace("（摘要）", " (abstract)")


def abbrev_en(mv: str, cv: str) -> str:
    """Table 4 related-layers: the English manuscript writes "、_"→", _" to omit the common prefix; on comparison, abbreviate the CSV side the same way."""
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
    """Compare the English manuscript table by table; return the list of differences outside the criteria (empty = PASS)."""
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
                elif col.csv == "数据 DOI":
                    cv, mv = norm_data_doi(cv), md_startswith_dash(mv)
                if col.kind == "numeric":
                    if num_tokens(mv) != num_tokens(cv):
                        findings.append(
                            f"{tab}(en): 行{r}「{col.md}」数字 token 不等 "
                            f"手稿={num_tokens(mv)} CSV={num_tokens(cv)}")
                elif not strict_en_equal(mv, cv):
                    findings.append(f"{tab}(en): 行{r}「{col.md}」手稿={mv!r} CSV={cv!r}")
    return findings



def test_en_current_tables_clean():
    """All English manuscript table bodies agree with the ledger CSV within the criteria (zero differences beyond the adapted columns and the regularised mappings)."""
    findings = compare_en(EN_MD.read_text(encoding="utf-8"))
    assert findings == [], "英文稿 md↔CSV 口径外差异：\n" + "\n".join(findings)


def test_en_mutated_tier_number_detected():
    """English-base mutation negative case: one cell of the Table 2 pixel count (64,800→64,801) → the gate flags it."""
    text = EN_MD.read_text(encoding="utf-8")
    assert "| Pixels | 64,800 |" in text
    findings = compare_en(text.replace("| Pixels | 64,800 |", "| Pixels | 64,801 |", 1))
    assert any(f.startswith("表 2(en):") and "行2" in f for f in findings), findings


def test_en_mutated_layer_id_detected():
    """English-base mutation negative case: case tampering of an Appendix A layer id (vsv→VSV) → flagged (the layer id column is strict)."""
    text = EN_MD.read_text(encoding="utf-8")
    assert "seismology__gladm35_vsv" in text
    findings = compare_en(text.replace("seismology__gladm35_vsv", "seismology__gladm35_VSV", 1))
    assert any(f.startswith("附录 A(en):") and "Layer id" in f for f in findings), findings


def test_en_mutated_visibility_detected():
    """English-base mutation negative case: Appendix A visibility public→internal → flagged (the visibility column is strict)."""
    text = EN_MD.read_text(encoding="utf-8")
    mutated = re.sub(
        r"^\| 1 \| topography__bedrock_elevation \| [^|]*\| public \|",
        "| 1 | topography__bedrock_elevation | Topography, gravity and magnetics | internal |",
        text, count=1, flags=re.M)
    assert mutated != text
    findings = compare_en(mutated)
    assert any(f.startswith("附录 A(en):") and "Visibility" in f for f in findings), findings


def test_en_missing_number_in_numeric_col_detected():
    """English-base mutation negative case: the Table 3 value column drops one number (13,695) → flagged (numeric multiset criterion)."""
    text = EN_MD.read_text(encoding="utf-8")
    assert "| Volume | Full store | 12.79 GB (13,695 files) |" in text
    mutated = text.replace("| Volume | Full store | 12.79 GB (13,695 files) |",
                           "| Volume | Full store | 12.79 GB (files) |", 1)
    findings = compare_en(mutated)
    assert any(f.startswith("表 3(en):") and "数字 token 不等" in f for f in findings), findings


def test_en_mutated_doi_detected():
    """English-base mutation negative case: the last digit of the Appendix B data DOI is tampered with (SEMUCB-WM1 row, which appears only in Appendix B) → flagged."""
    text = EN_MD.read_text(encoding="utf-8")
    assert "10.17611/dp/emc.2023.semucbwm1.1" in text
    findings = compare_en(text.replace("10.17611/dp/emc.2023.semucbwm1.1",
                                       "10.17611/dp/emc.2023.semucbwm1.2", 1))
    assert any(f.startswith("附录 B(en):") and "DOI" in f for f in findings), findings


def test_en_home_tier_mapping_and_mutation():
    """The Appendix A mapping from the ledger's tier-range marker to "per tier" is on record; reverting to a tier marker is flagged (a third value is not accepted)."""
    text = EN_MD.read_text(encoding="utf-8")
    assert "| per tier | float32 | km² | Analytical formula |" in text
    mutated = text.replace("| per tier | float32 | km² | Analytical formula |",
                           "| 1° | float32 | km² | Analytical formula |", 1)
    assert mutated != text
    findings = compare_en(mutated)
    assert any(f.startswith("附录 A(en):") and "Home tier" in f for f in findings), findings


def test_en_adapted_column_exempt_by_design():
    """adapted columns are exempt for the whole column under the spec criteria: changing the wording of the Table 1 "Dataset" column produces no finding (the exemption surface is explicitly on record)."""
    text = EN_MD.read_text(encoding="utf-8")
    mutated = text.replace(
        "| 1 | ETOPO 2022 Bedrock (NOAA National Centers for Environmental Information, 2022) |",
        "| 1 | ETOPO 2022 Bedrock elevation (NOAA NCEI, 2022) |", 1)
    assert mutated != text
    assert compare_en(mutated) == []

# ============================================================================
# ============================================================================
NSD_MD = REPO / "paper" / "manuscript" / "SEDC-manuscript-nsd-en.md"


def _nsd_specs() -> list[EnSpec]:
    """Re-anchor the English-variant column specs to the NSD manuscript by registered table number (the table body content is verbatim identical to the English manuscript)."""
    base = {spec.csv_name: spec for spec in EN_SPECS}
    specs: list[EnSpec] = []
    for entry in sorted(nt.TABLE_REGISTRY, key=lambda e: e.number):
        src = base[entry.csv_name]
        specs.append(EnSpec(f"表 {entry.number}", rf"\*\*Table {entry.number}\.\*\*",
                            entry.csv_name, src.columns))
    return specs


def compare_nsd(md_text: str) -> list[str]:
    """Compare the NSD manuscript table by table (the column criteria share the same implementation as the English variant; only the anchors / table numbers follow the registered mapping).

    Numbered-citation superscripts (`^N^`) are a structural artefact; strip them before extracting tokens — otherwise in-cell superscript numbers get counted into the numeric multiset (measured false positive: ledger `Sect. 2.5` ↔ NSD `…subsection)^30^`).
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


NSD_STRUCTURAL_CELL_DIFFS: set[tuple[str, int, str]] = {
    ("表 6", 14, "Result"),                        # Sect. 4.4 → the Physical cross-check subsection
    ("表 7", 3, "Mechanism / measured signature"),  # Sect. 2.5 → the Special data channels subsection
    ("表 7", 5, "Mechanism / measured signature"),
    ("表 8", 5, "Hard constraint"),                 # Sect. 2.5 → the Special data channels subsection
    ("表 8", 6, "Operating rule"),
    ("表 7", 5, "Impact and user action"),
    ("表 7", 7, "Mechanism / measured signature"),
}


def _registered_nsd(finding: str) -> bool:
    return any(f"{tab}(nsd): 行{r}「{col}」" in finding
               for tab, r, col in NSD_STRUCTURAL_CELL_DIFFS)


def test_nsd_tables_clean():
    """All NSD manuscript main tables agree with the ledger CSV within the criteria (zero differences beyond the per-cell whitelist entries)."""
    findings = compare_nsd(NSD_MD.read_text(encoding="utf-8"))
    unregistered = [f for f in findings if not _registered_nsd(f)]
    assert unregistered == [], "NSD 稿 md↔CSV 白名单外差异：\n" + "\n".join(unregistered)
    assert len(findings) == len(NSD_STRUCTURAL_CELL_DIFFS), (
        f"登记 {len(NSD_STRUCTURAL_CELL_DIFFS)} 格、实测 {len(findings)} 格——"
        f"存在陈旧登记：{sorted(NSD_STRUCTURAL_CELL_DIFFS)}")


def test_nsd_table_numbering_consistent():
    """Table-number ↔ CSV key mapping + first-mention order = numeric order (nsd_tables criteria ①②): zero findings."""
    findings = nt.numbering_findings(NSD_MD.read_text(encoding="utf-8"))
    assert findings == [], findings
    assert nt.mention_order(NSD_MD.read_text(encoding="utf-8")) == \
        sorted(e.number for e in nt.TABLE_REGISTRY)


def test_nsd_mutated_cell_detected():
    """Negative case: one cell of the NSD manuscript table body (64,800→64,801) → flagged (the table body still forms a closed chain with the ledger)."""
    text = NSD_MD.read_text(encoding="utf-8")
    assert "| Pixels | 64,800 |" in text
    findings = compare_nsd(text.replace("| Pixels | 64,800 |", "| Pixels | 64,801 |", 1))
    assert any(f.startswith("表 2(nsd):") and "行2" in f for f in findings), findings


def test_nsd_numbering_drift_detected():
    """Negative case: corrupt one table-number mention (mapping misalignment / first-mention-order drift) → flagged."""
    text = NSD_MD.read_text(encoding="utf-8")
    findings = nt.numbering_findings(text.replace("in Table 5.", "in Table 6.", 1))
    assert any("首现序" in f or "题注表号集合" in f or "孤儿" in f for f in findings), findings


def test_supplementary_tables_match_baseline():
    """The Supplementary table products match the ESSD English-manuscript appendix table bodies field by field (carrier migration with zero manual transcription)."""
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
    """renumber mechanically rewrites the regression captions and body mentions, mapping **once**
    by first-mention order — the bold markers `**` must not be consumed and numbers must not be
    mapped twice; it is also idempotent."""
    text = ("**Table 2.** second caption.\n\n**Table 1.** first caption.\n\n"
            "See Table 3, Table 4, Table 5, Table 6, Table 7 and Table 8.\n")
    out = nt.renumber(text)
    assert out.startswith("**Table 1.** second caption."), out[:60]
    assert "**Table 2.** first caption." in out
    assert "See Table 3, Table 4, Table 5, Table 6, Table 7 and Table 8." in out
    assert nt.caption_numbers(out) == [1, 2]
    assert nt.renumber(out) == out
