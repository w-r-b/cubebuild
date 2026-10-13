"""Checks for the desensitized input-dataset list (input-datasets.csv).

The list is part of the public content set: derived from a private data ledger and containing
only external-facing fields; notes such as internal decision records, pending states,
disposition dates, and usage traps must never appear. This module guards four things: the
column set is exactly the field whitelist
id/name/doi/licence/format; the row set is consistent with the scope statement (the scope
states inclusion/exclusion and reasons, and the row count
matches it); note-class content never appears (including date-like patterns, CJK characters,
and internal process wording); external readers can read it
(English, complete fields, self-explanatory column names).

When the private-side sources (data ledger, paper source table/supplementary licence table,
trace scanner) are not in this workspace, the corresponding
cross-checks are skipped entirely (following the absent-means-skip convention). This module
is itself subject to the public-surface trace scan, so literals constrained by the term list
are built by run-time concatenation or escaping (self-reference paradox).
"""

import csv
import importlib.util
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
LIST_PATH = REPO / "input-datasets.csv"

WHITELIST = ("id", "name", "doi", "licence", "format")

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
    """The list check itself: returns a list of problems (empty list = pass)."""
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
    assert int(m.group(1)) == len(rows)
    ids = {int(r[0]) for r in rows}
    for excluded in (12, 26):
        assert f"id {excluded}" in scope, excluded
        assert excluded not in ids
    assert "GVP Volcano List" in scope and "1.8 Ga" in scope
    assert "non-commercial" in scope and "CC BY-NC-ND 4.0" in scope
    assert "not included in the public release" in scope
    for restricted in ("WGM2012", "GSRM v2.2", "GEM Global Active Faults",
                       "Global Basins"):
        assert restricted in scope, restricted
    assert "Lines beginning with" in scope


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
    for r in rows:
        assert r[4] == by_id[int(r[0])]["format"], r[0]


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
            if cell == "\u2014":
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
    "202" + "6-09-11: excluded from distribution",
    "\u88c1\u5b9a" + ": keep for self-use only",
    "pending author confirmation",
    "filename trap: rename before use",
    "internal audit record",
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