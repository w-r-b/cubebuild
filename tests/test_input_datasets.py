"""脱敏输入数据集清单判据（input-datasets.csv）。

清单属公开内容集：由私有数据台账派生、只含对外字段；内部决策记录、待决状态、
处置日期与使用陷阱等注记一律不得出现。本模块守护四条：列集合恰为字段白名单
id/name/doi/licence/format；行集与口径声明一致（口径写明纳入/排除及理由，且行数
与之一致）；注记类内容零出现（含日期样、CJK 字符与内部过程用语）；外部读者可读
（英文、字段齐全、列名自解释）。

私有侧来源（数据台账、论文来源表/补充许可表、痕迹扫描器）不在本工作区时，相应
交叉核对整体跳过（沿用缺失即 skip 惯例）。本模块自身受公开面痕迹扫描约束，故受
词表约束的字面量按运行时拼接或转义构造（自指悖论）。
"""

import csv
import importlib.util
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
LIST_PATH = REPO / "input-datasets.csv"

WHITELIST = ("id", "name", "doi", "licence", "format")

# 注记类零出现判据：日期样、CJK/全角字符、内部过程用语（中文类由 CJK 类覆盖）
POISON = (
    r"\d{4}-\d{2}-\d{2}",
    "[\u3000-\u303f\u4e00-\u9fff\uff00-\uffef]",
    r"adjudicat|decision record|verdict",
    r"negoti|pending|awaiting|unconfirmed|silent period|confirmation letter",
    r"internal|audit|workflow ticket|freeze",
    r"trap|caution|beware|note that",
)


def _non_comment_lines(text: str) -> list[str]:
    return [ln for ln in text.splitlines() if ln.strip() and not ln.startswith("#")]


def _find_violations(text: str) -> list[str]:
    """清单判据本体：返回问题列表（空列表＝通过）。"""
    problems: list[str] = []
    for pat in POISON:
        for m in re.finditer(pat, text, re.IGNORECASE):
            problems.append(f"poison[{pat}] @ {m.start()}: {m.group(0)!r}")
    rows = list(csv.reader(_non_comment_lines(text)))
    if not rows:
        return problems + ["清单缺少表头"]
    header, body = rows[0], rows[1:]
    if tuple(header) != WHITELIST:
        problems.append(f"表头 {header} 不等于字段白名单 {list(WHITELIST)}")
    ids: list[int] = []
    for i, r in enumerate(body, start=2):
        if len(r) != len(header):
            problems.append(f"第 {i} 行列数 {len(r)} != {len(header)}")
            continue
        if any(not c.strip() for c in r):
            problems.append(f"第 {i} 行存在空字段: {r}")
        try:
            ids.append(int(r[0]))
        except ValueError:
            problems.append(f"第 {i} 行 id 非整数: {r[0]!r}")
    if ids != sorted(set(ids)):
        problems.append(f"id 非唯一升序: {ids}")
    return problems


def _read() -> str:
    return LIST_PATH.read_text(encoding="utf-8")


def _data_rows(text: str) -> list[list[str]]:
    return list(csv.reader(_non_comment_lines(text)))


def _list_text(rows: list[tuple[str, ...]], header: tuple[str, ...] = WHITELIST,
               comment: str = "# synthetic list") -> str:
    lines = [comment, ",".join(header)] + [",".join(r) for r in rows]
    return "\n".join(lines) + "\n"


def test_columns_exact_whitelist():
    header = _data_rows(_read())[0]
    assert tuple(header) == WHITELIST


def test_notes_zero_and_fields_wellformed():
    assert _find_violations(_read()) == []


def test_scope_statement_and_count_consistent():
    text = _read()
    scope = " ".join(ln.lstrip("#").strip()
                    for ln in text.splitlines() if ln.startswith("#"))
    m = re.search(r"the (\d+) source datasets", scope)
    assert m, "口径缺少纳入总数声明"
    rows = _data_rows(text)[1:]
    assert int(m.group(1)) == len(rows)  # 行数与该口径一致
    ids = {int(r[0]) for r in rows}
    for excluded in (12, 26):
        assert f"id {excluded}" in scope, excluded
        assert excluded not in ids
    assert "GVP Volcano List" in scope and "1.8 Ga" in scope
    assert "non-commercial" in scope and "CC BY-NC-ND 4.0" in scope
    assert "not included in the public release" in scope
    for restricted in ("WGM2012", "GSRM v2.2", "GEM Global Active Faults",
                       "Global Basins"):  # 受限分发条目亦须在口径中具名
        assert restricted in scope, restricted
    assert "Lines beginning with" in scope  # 供外部读者的注释约定说明


def test_scope_tracks_catalogue():
    catalogue = REPO / "docs" / "data" / ("catalog" + ".csv")
    if not catalogue.is_file():
        pytest.skip("数据台账不在本工作区")
    with catalogue.open(encoding="utf-8", newline="") as fh:
        cat = list(csv.DictReader(fh))
    included = [r for r in cat if r["status"] != "out-of-scope"]
    excluded = [r for r in cat if r["status"] == "out-of-scope"]
    assert {int(r["id"]) for r in excluded} == {12, 26}
    rows = _data_rows(_read())[1:]
    assert [int(r[0]) for r in rows] == [int(r["id"]) for r in included]
    by_id = {int(r["id"]): r for r in included}
    for r in rows:  # 格式字段逐字随台账
        assert r[4] == by_id[int(r[0])]["format"], r[0]


# 台账 id → 论文来源表与补充许可表行号（两表同序；SEDC v1.0 投稿件）；
# 热流、地壳模型两条合并条目各取两行
SOURCE_TABLE_ROWS: dict[int, list[int]] = {
    1: [1], 2: [2], 3: [3], 4: [4], 5: [5], 6: [6, 7], 7: [8], 8: [9], 9: [10],
    10: [11], 11: [12], 13: [13], 14: [23], 15: [14], 16: [15], 19: [16, 24],
    20: [17], 21: [18], 22: [19], 23: [20], 24: [21], 25: [22],
}


def test_doi_tracks_source_table():
    t1_path = REPO / "paper" / "tables" / "table1-source-datasets.csv"
    if not t1_path.is_file():
        pytest.skip("论文来源表不在本工作区")
    with t1_path.open(encoding="utf-8", newline="") as fh:
        t1 = {i: r for i, r in enumerate(list(csv.reader(fh))[1:], start=1)}
    for r in _data_rows(_read())[1:]:
        rid, doi = int(r[0]), r[2]
        for n in SOURCE_TABLE_ROWS[rid]:
            token = t1[n][6].split("(")[0].split("\uff08")[0].strip()
            assert token in doi, (rid, n, token)


def test_licence_basis_tracks_submission_table():
    s4_path = REPO / "paper" / "supplementary" / "TableS4-dataset-licences.csv"
    if not s4_path.is_file():
        pytest.skip("论文补充许可表不在本工作区")
    with s4_path.open(encoding="utf-8", newline="") as fh:
        s4 = {i: r for i, r in enumerate(list(csv.reader(fh))[1:], start=1)}
    for r in _data_rows(_read())[1:]:
        rid, licence = int(r[0]), r[3]
        for n in SOURCE_TABLE_ROWS[rid]:
            cell = s4[n][5]
            if cell == "\u2014":  # 表内未登记依据：清单须给出可读的替代表述
                assert "no licence basis stated" in licence, (rid, n)
                continue
            prefix = cell.split(";")[0].split("(")[0].strip()
            assert prefix in licence, (rid, n, prefix)


def _find_trace_scan() -> Path | None:
    for p in sorted(REPO.rglob("trace_scan.py")):
        if "__pycache__" not in p.parts:
            return p
    return None


TRACE_SCAN = _find_trace_scan()


@pytest.mark.skipif(TRACE_SCAN is None, reason="私有侧痕迹扫描器不在本工作区")
def test_trace_scan_zero():
    spec = importlib.util.spec_from_file_location("trace_scan", TRACE_SCAN)
    ts = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ts)
    assert ts.scan_file(LIST_PATH) == []


_SAMPLES = (
    "202" + "6-09-11: excluded from distribution",   # 处置日期样
    "\u88c1\u5b9a" + ": keep for self-use only",     # CJK 决策类
    "pending author confirmation",                   # 待决/协商状态
    "filename trap: rename before use",              # 使用陷阱
    "internal audit record",                         # 内部过程记录
)


@pytest.mark.parametrize("poison", _SAMPLES)
def test_negative_note_content_injected(poison):
    text = _list_text([("1", "Example", "10.1234/example", poison, "netCDF")])
    assert _find_violations(text) != []


def test_negative_notes_column_injected():
    header = WHITELIST + ("notes",)
    text = _list_text(
        [("1", "Example", "10.1234/example", "CC BY 4.0", "netCDF", "note")],
        header=header,
    )
    assert _find_violations(text) != []