"""Criteria tests for the item-by-item journal-guideline compliance mapping table.

A positive case = the real mapping table yields zero findings; negative cases
are injected item by item (missing entry / unregistered entry / illegal status /
missing evidence pointer / to-do item not pointing at a submission requirement)
-- the criteria must catch the various forms of "the table itself being broken".
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "paper" / "scripts"))

try:
    import check_compliance_nsd as cc  # noqa: E402
except ImportError:
    cc = None

pytestmark = pytest.mark.skipif(
    cc is None, reason="论文侧合规检查器不在本工作区（缺 paper/scripts/check_compliance_nsd.py）")

DOC = REPO / "submission-compliance-nsd.md"
REQUIREMENTS = REPO / "submission-requirements-nsd.md"
TODO_KEYS = set()


def _doc() -> str:
    return DOC.read_text(encoding="utf-8")


def _row(text: str, key: str) -> str:
    return next(l for l in text.split("\n") if l.lstrip().startswith(f"| `{key}` "))


def test_compliance_table_clean():
    """Positive case: the mapping table passes the three self-checks (entry closure / status-enum closure / each row carries a pointer)."""
    findings = cc.run_checks(_doc())
    assert findings == [], findings


def test_compliance_item_set_is_the_ten_guideline_faces():
    """Entry set = the ten guideline faces (title / abstract / section set / figures / tables / references / availability / file set / transaction section / revision-round .tex)."""
    rows = cc.parse_rows(_doc())
    assert [r[0].strip("`") for r in rows] == [k for k, _ in cc.REQUIRED_ITEMS]


def test_compliance_missing_item_detected():
    """Negative case: delete one entry -> "missing entry" is caught."""
    text = _doc()
    findings = cc.run_checks(text.replace(_row(text, "title") + "\n", "", 1))
    assert any("缺条目：title" in f for f in findings), findings


def test_compliance_unknown_item_detected():
    """Negative case: insert an unregistered entry (the entry set must be closed) -> caught."""
    text = _doc()
    injected = "| `funding-extra` | 自造要求 | `submission-requirements-nsd.md` | 通过 |\n"
    findings = cc.run_checks(text.replace("## 待办项与前置动作的对应", injected +
                                          "## 待办项与前置动作的对应", 1))
    assert any("未登记条目：funding-extra" in f for f in findings), findings


def test_compliance_illegal_status_detected():
    """Negative case: change the status to a value outside the enum (e.g. "partially passing") -> caught."""
    text = _doc()
    row = _row(text, "tex")
    findings = cc.run_checks(text.replace(row, row.rsplit("|", 2)[0] + "| 部分通过 |", 1))
    assert any("非法" in f for f in findings), findings


def test_compliance_missing_pointer_detected():
    """Negative case: erase the evidence pointer (inline code in backticks) -> caught."""
    text = _doc()
    row = _row(text, "figures")
    stripped = row.split("|")
    stripped[3] = " 无指针的一段说明 "
    findings = cc.run_checks(text.replace(row, "|".join(stripped), 1))
    assert any("缺指针" in f for f in findings), findings


def test_compliance_todo_requires_registered_action():
    """Negative case: an entry has to-do status but its evidence does not point
    at a submission requirement -> "to-do item not pointing" is caught. This
    constructs the to-do scenario itself and does not depend on whether the
    mapping table currently has any to-do rows."""
    text = _doc()
    row = _row(text, "title")
    cells = row.split("|")
    cells[3] = " `paper/manuscript/SEDC-manuscript-nsd-en.md` 某落点 "
    cells[4] = " 待办 "
    findings = cc.run_checks(text.replace(row, "|".join(cells), 1))
    assert any("待办项未指向" in f for f in findings), findings


def test_compliance_todo_set_is_recorded_state():
    """The to-do set = the currently to-do rows of the mapping table = TODO_KEYS (after closing an item this set must be updated explicitly; silently loosening/tightening is forbidden)."""
    statuses = {r[0].strip("`"): r[3] for r in cc.parse_rows(_doc())}
    assert {k for k, v in statuses.items() if v == cc.STATUS_TODO} == TODO_KEYS, statuses
    assert set(statuses.values()) <= set(cc.STATUSES)


def test_compliance_todo_actions_exist_in_requirements():
    """The "§1 action N" referenced by a to-do item must actually be registered (guarding against pointing at a non-existent prerequisite action)."""
    req = REQUIREMENTS.read_text(encoding="utf-8")
    registered = {int(m.group(1)) for m in re.finditer(r"^\|\s*(\d+)\s*\|", req, re.M)}
    assert registered == set(range(1, 10)), registered
    doc = _doc()
    for key in TODO_KEYS:
        row = _row(doc, key)
        for m in re.finditer(r"动作\s*(\d+)(?:\s*[–-]\s*(\d+))?", row):
            lo, hi = int(m.group(1)), int(m.group(2) or m.group(1))
            assert set(range(lo, hi + 1)) <= registered, (key, m.group(0), sorted(registered))
