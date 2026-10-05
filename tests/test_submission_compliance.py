"""期刊指南逐条合规对照表的判据测试。

正例＝真对照表零发现；负例逐条注入（缺条目／未登记条目／非法状态／缺证据指针／待办项
未指向提交要件）——判据须能抓住「表本身失效」的各类形态。
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
except ImportError:  # 公开仓不含论文侧脚本与产物：缺件即跳过（沿用缺件跳过惯例）
    cc = None

pytestmark = pytest.mark.skipif(
    cc is None, reason="论文侧合规检查器不在本工作区（缺 paper/scripts/check_compliance_nsd.py）")

DOC = REPO / "submission-compliance-nsd.md"
REQUIREMENTS = REPO / "submission-requirements-nsd.md"
# 当前登记的四处待办（与提交要件 §1 前置动作一一对应；任一闭环须显式改本集合）；
# 题名回改后新增 title → §1 动作 9。
TODO_KEYS = {"title"}  # 与对照表现存待办集一致；闭环一项就删一项，禁静默放宽/收紧


def _doc() -> str:
    return DOC.read_text(encoding="utf-8")


def _row(text: str, key: str) -> str:
    return next(l for l in text.split("\n") if l.lstrip().startswith(f"| `{key}` "))


def test_compliance_table_clean():
    """正例：对照表过三条自检（条目封闭／状态枚举封闭／每条含指针）。"""
    findings = cc.run_checks(_doc())
    assert findings == [], findings


def test_compliance_item_set_is_the_ten_guideline_faces():
    """条目集合 = 指南十面（题名/摘要/节集/图/表/参考文献/可得性/文件集/事务段/修订轮 .tex）。"""
    rows = cc.parse_rows(_doc())
    assert [r[0].strip("`") for r in rows] == [k for k, _ in cc.REQUIRED_ITEMS]


def test_compliance_missing_item_detected():
    """负例：删一条条目 → 命中「缺条目」。"""
    text = _doc()
    findings = cc.run_checks(text.replace(_row(text, "title") + "\n", "", 1))
    assert any("缺条目：title" in f for f in findings), findings


def test_compliance_unknown_item_detected():
    """负例：插入未登记条目（条目集合须封闭）→ 命中。"""
    text = _doc()
    injected = "| `funding-extra` | 自造要求 | `submission-requirements-nsd.md` | 通过 |\n"
    findings = cc.run_checks(text.replace("## 待办项与前置动作的对应", injected +
                                          "## 待办项与前置动作的对应", 1))
    assert any("未登记条目：funding-extra" in f for f in findings), findings


def test_compliance_illegal_status_detected():
    """负例：状态改成枚举外取值（如「部分通过」）→ 命中。"""
    text = _doc()
    row = _row(text, "tex")
    findings = cc.run_checks(text.replace(row, row.rsplit("|", 2)[0] + "| 部分通过 |", 1))
    assert any("非法" in f for f in findings), findings


def test_compliance_missing_pointer_detected():
    """负例：抹掉证据指针（反引号内联码）→ 命中。"""
    text = _doc()
    row = _row(text, "figures")
    stripped = row.split("|")
    stripped[3] = " 无指针的一段说明 "
    findings = cc.run_checks(text.replace(row, "|".join(stripped), 1))
    assert any("缺指针" in f for f in findings), findings


def test_compliance_todo_requires_registered_action():
    """负例：待办项不指向提交要件（新增未登记待办）→ 命中。"""
    text = _doc()
    row = _row(text, "title")   # 当前唯一待办项
    stripped = row.split("|")
    stripped[3] = " 某个未登记的动作 "
    findings = cc.run_checks(text.replace(row, "|".join(stripped), 1))
    assert any("待办项未指向" in f for f in findings), findings


def test_compliance_todo_set_is_recorded_state():
    """待办集合＝登记现状（当前仅 `title` 因已知偏差保留）；闭环后须显式更新本集合（禁静默放宽/收紧）。"""
    statuses = {r[0].strip("`"): r[3] for r in cc.parse_rows(_doc())}
    assert {k for k, v in statuses.items() if v == cc.STATUS_TODO} == TODO_KEYS, statuses
    assert set(statuses.values()) <= set(cc.STATUSES)


def test_compliance_todo_actions_exist_in_requirements():
    """待办项引用的「§1 动作 N」必须真实登记（防指向不存在的前置动作）。"""
    req = REQUIREMENTS.read_text(encoding="utf-8")
    registered = {int(m.group(1)) for m in re.finditer(r"^\|\s*(\d+)\s*\|", req, re.M)}
    assert registered == set(range(1, 10)), registered
    doc = _doc()
    for key in TODO_KEYS:
        row = _row(doc, key)
        for m in re.finditer(r"动作\s*(\d+)(?:\s*[–-]\s*(\d+))?", row):
            lo, hi = int(m.group(1)), int(m.group(2) or m.group(1))
            assert set(range(lo, hi + 1)) <= registered, (key, m.group(0), sorted(registered))
