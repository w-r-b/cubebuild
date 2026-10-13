"""Criteria for the release runbook and the "zero-residue placeholder string" scan tool.

Two parts:
- Runbook structure: each row of the execution-order table is annotated with
  "manual/automatic" and its precondition; the first-occurrence order of the
  step chain (create private repo -> backfill manuscript -> submit to journal
  -> make public -> Zenodo-GitHub binding -> mint code DOI -> backfill
  CITATION -> zero-residue scan) is strictly increasing; the two handover
  artifacts (freeze-manifest sanitization, public-repo .gitignore) fall within
  the execution order; the hard-constraint sentences are all present
  (the two time points around making public, the code DOI minting constraint,
  same-batch backfill, the manual-step declaration); the scan word registry and
  the tool word registry are locked together (guarding against two-sided
  drift). All the above criteria come with synthetic-artifact negative cases.
- Scan tool: synthetic-artifact baseline (fully backfilled, zero hits) plus
  negative cases (a deliberately retained placeholder must be hit) plus full
  coverage of the four placeholder-string families (including case variants)
  plus fail-closed on missing targets plus the default scope covering the
  manuscript and the four public-repo essentials plus CLI exit codes 0/1/2.

When the private-side artifacts are not present in this workspace (public
repo), the whole module is skipped (following the missing-artifact skip
convention). This module ships with the test suite onto the public face, so
its text is subject to public-face trace scanning.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _find_private(name: str) -> Path | None:
    for p in sorted(REPO.rglob(name)):
        if "__pycache__" not in p.parts:
            return p
    return None


TOOL = _find_private("placeholder_scan.py")
RUNBOOK = _find_private("release-runbook.md")

skip_no_tool = pytest.mark.skipif(TOOL is None, reason="私有侧检索工具不在本工作区")
skip_no_runbook = pytest.mark.skipif(RUNBOOK is None, reason="私有侧发布执行单不在本工作区")
skip_no_pair = pytest.mark.skipif(
    TOOL is None or RUNBOOK is None, reason="私有侧工具或执行单不在本工作区")


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def tool():
    return _load(TOOL, "placeholder_scan")



MANUSCRIPT_SAMPLE = (
    "The public data are published on Zenodo as a single tar package "
    "(DOI to be registered), 10.85 GB in total.\n\n"
    "The construction pipeline cubebuild (v0.1.0) is published on GitHub "
    "(repository URL to be filled in).\n"
)
FILLED_MANUSCRIPT = MANUSCRIPT_SAMPLE.replace(
    "DOI to be registered", "10.5281/zenodo.0000001"
).replace(
    "repository URL to be filled in", "https://github.com/example/cubebuild"
)
ESSENTIALS_SAMPLE = (
    "git clone https://github.com/<account>/cubebuild.git\n"
    'doi: "<to be registered>"\n'
    'authors = [{ name = "<to be filled in>" }]\n'
    "<To be completed by the authors at submission.>\n"
)


def _fake_tree(tmp_path: Path, manuscript_text: str) -> Path:
    """Minimal isomorphic tree of the private repo: manuscript plus the four essentials."""
    root = tmp_path / "root"
    (root / "paper" / "manuscript").mkdir(parents=True)
    (root / "paper" / "manuscript" / "SEDC-manuscript-nsd-en.md").write_text(
        manuscript_text, encoding="utf-8")
    for name in ("README.md", "CITATION.cff", "pyproject.toml", "LICENSE"):
        (root / name).write_text("filled\n", encoding="utf-8")
    return root




@skip_no_tool
def test_scan_filled_sample_is_clean(tool, tmp_path):
    """Baseline: a fully backfilled synthetic artifact yields zero hits (ensuring the negative cases below are triggered by placeholders rather than fixture faults)."""
    f = tmp_path / "manuscript.md"
    f.write_text(FILLED_MANUSCRIPT, encoding="utf-8")
    assert tool.scan_file(f) == []


@skip_no_tool
def test_negative_retained_single_placeholder_hits(tool, tmp_path):
    """Negative case: a single deliberately retained placeholder must be hit, and only that one must be hit."""
    f = tmp_path / "manuscript.md"

    f.write_text(MANUSCRIPT_SAMPLE, encoding="utf-8")
    assert len(tool.scan_file(f)) == 2

    retained = MANUSCRIPT_SAMPLE.replace(
        "repository URL to be filled in", "https://github.com/example/cubebuild")
    f.write_text(retained, encoding="utf-8")
    hits = tool.scan_file(f)
    assert len(hits) == 1
    line_no, token, _text = hits[0]
    assert token == "to be registered"
    assert line_no == 1


@skip_no_tool
def test_scan_detects_all_placeholder_token_families(tool, tmp_path):
    """Each of the four placeholder-string families is detectable (including case variants: the declarative placeholder with angle brackets)."""
    f = tmp_path / "essentials.txt"
    f.write_text(ESSENTIALS_SAMPLE, encoding="utf-8")
    tokens = [tok for _ln, tok, _line in tool.scan_file(f)]
    assert tokens == ["<account>", "to be registered", "to be filled in",
                      "to be completed"]


@skip_no_tool
def test_default_targets_cover_manuscript_and_essentials(tool):
    """Default scope = manuscript plus the four essentials; the targets all exist in the current workspace (a missing artifact is fail-closed)."""
    rels = [p.relative_to(REPO).as_posix() for p in tool.targets("all")]
    assert rels == [
        "paper/manuscript/SEDC-manuscript-nsd-en.md",
        "README.md",
        "CITATION.cff",
        "pyproject.toml",
        "LICENSE",
    ]
    assert [p.name for p in tool.targets("manuscript")] == ["SEDC-manuscript-nsd-en.md"]
    assert tool.report("all")["missing"] == []


@skip_no_tool
def test_missing_target_fails_closed(tool, tmp_path):
    """A missing target must not be treated as zero residue: missing is non-empty and does not appear among the hits."""
    report = tool.report("all", root=tmp_path)
    assert len(report["missing"]) == 5
    assert report["hits"] == []


@skip_no_tool
def test_unknown_scope_rejected(tool):
    """An unknown scope name fails explicitly (fail-closed); it does not silently fall back to the default scope."""
    with pytest.raises(ValueError):
        tool.targets("everything")


@skip_no_tool
def test_cli_exit_codes(tool, tmp_path, monkeypatch):
    """CLI contract: 0 zero residue, 1 hit, 2 missing artifact."""
    monkeypatch.setattr(sys, "argv", ["placeholder_scan.py", "--manuscript"])

    monkeypatch.setattr(tool, "REPO", _fake_tree(tmp_path / "clean", FILLED_MANUSCRIPT))
    assert tool.main() == 0

    monkeypatch.setattr(tool, "REPO", _fake_tree(tmp_path / "hit", MANUSCRIPT_SAMPLE))
    assert tool.main() == 1

    monkeypatch.setattr(tool, "REPO", tmp_path / "empty")
    assert tool.main() == 2



CHAIN_ANCHORS = ("建私有仓", "回填手稿", "投稿提交", "转公开",
                 "Zenodo-GitHub 绑定", "铸造代码 DOI", "回填 CITATION", "零残留检索")
RUNBOOK_CONSTRAINTS = (
    "不得早于", "不得晚于",
    "不支持私有仓", "不能早于",
    "同批",
    "人工步骤", "超出范围",
    "四件要件",
)


def _runbook_rows(text: str) -> list[list[str]]:
    """Step-table rows within the "execution order" section (table rows whose first column is a number)."""
    section = text.split("## 执行序", 1)[1].split("\n## ", 1)[0]
    rows = []
    for line in section.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if cells and re.fullmatch(r"\d+", cells[0]):
            rows.append(cells)
    return rows


def check_steps_annotated(rows: list[list[str]]) -> list[str]:
    """Each step has all five columns, is annotated manual/automatic, and has a non-empty precondition; the two handover artifacts fall within the execution order."""
    problems = []
    for cells in rows:
        if len(cells) != 5:
            problems.append(f"步骤行应恰为五列: {cells}")
        elif cells[2] not in ("人工", "自动"):
            problems.append(f"人工／自动标注缺失: {cells}")
        elif not cells[3]:
            problems.append(f"前置条件缺失: {cells}")
    joined = "\n".join(" | ".join(c) for c in rows)
    if "cube_manifest.json" not in joined:
        problems.append("执行序缺数据侧移交步骤（冻结清单净化）")
    if ".gitignore" not in joined:
        problems.append("执行序缺仓侧移交步骤（公开仓补件）")
    return problems


def check_chain_order(rows: list[list[str]]) -> list[str]:
    """The first-occurrence row number of each anchor in the step chain must be strictly increasing."""
    joined = [" | ".join(c) for c in rows]
    problems = []
    positions = []
    for anchor in CHAIN_ANCHORS:
        idx = next((i for i, row in enumerate(joined) if anchor in row), None)
        if idx is None:
            problems.append(f"执行序缺步骤锚: {anchor}")
        positions.append(idx)
    if problems:
        return problems
    found = [p for p in positions if p is not None]
    if found != sorted(found) or len(set(found)) != len(found):
        problems.append(f"步骤链顺序错乱: {positions}")
    return problems


def check_constraints(text: str) -> list[str]:
    """All hard-constraint sentences are present."""
    return [f"执行单缺约束句: {t}" for t in RUNBOOK_CONSTRAINTS if t not in text]


def check_token_registry(text: str, tokens) -> list[str]:
    """The entry set of the runbook's "scan word registry" line matches the tool word registry (guarding against two-sided drift)."""
    line = next((l for l in text.splitlines() if l.startswith("检索词表")), "")
    if not line:
        return ["执行单缺检索词表行"]
    found = set(re.findall(r"`([^`]+)`", line))
    problems = [f"执行单词表缺: {t}" for t in sorted(set(tokens) - found)]
    problems += [f"执行单词表多出: {t}" for t in sorted(found - set(tokens))]
    return problems



_SAMPLE_ROWS = [
    "| 1 | 建私有仓 | 人工 | 账号就绪 | 建 Private 空仓 |",
    "| 2 | 净化 cube_manifest.json | 人工 | 冻结制品就绪 | 文本级净化 |",
    "| 3 | 补 .gitignore | 人工 | 导出完成 | 写入四行 |",
    "| 4 | 回填手稿两处 | 人工 | 数据 DOI 就绪 | 同批回填 |",
    "| 5 | 投稿提交 | 人工 | 闸门全过 | 记 T0 |",
    "| 6 | 转公开 | 人工 | 提交完成 | 改可见性 |",
    "| 7 | Zenodo-GitHub 绑定 | 人工 | 已开放 | 开启集成 |",
    "| 8 | 铸造代码 DOI | 人工 | 绑定完成 | 发 release |",
    "| 9 | 回填 CITATION | 人工 | 代码 DOI 就绪 | 填 doi |",
    "| 10 | 零残留检索 | 自动 | 回填推送 | 零命中 |",
]


def _sample_runbook(rows: list[str] | None = None) -> str:
    body = "\n".join(_SAMPLE_ROWS if rows is None else rows)
    return (
        "# 示例执行单\n\n## 执行序\n\n"
        "| # | 步骤 | 人工/自动 | 前置条件 | 操作与判据 |\n"
        "|---|---|---|---|---|\n"
        f"{body}\n\n"
        "## 时点与约束\n\n"
        "- 转公开不得早于投稿提交，亦不得晚于提交。\n"
        "- 代码 DOI 不支持私有仓归档，不能早于转公开时点。\n"
        "- 同批回填；人工步骤属超出范围；四件要件。\n\n"
        "## 占位清单\n\n"
        "检索词表（四枚）：`to be filled in`、`to be registered`、`<account>`、"
        "`to be completed`。\n"
    )


@skip_no_runbook
def test_runbook_structure_passes():
    """The real runbook: all four structural criteria pass."""
    text = RUNBOOK.read_text(encoding="utf-8")
    rows = _runbook_rows(text)
    assert len(rows) >= 15
    assert check_steps_annotated(rows) == []
    assert check_chain_order(rows) == []
    assert check_constraints(text) == []


@skip_no_runbook
def test_runbook_negative_examples():
    """Malformed synthetic-artifact negative cases: each of the following four mutations must be caught by its corresponding criterion (the criteria are falsifiable)."""
    base = _sample_runbook()
    assert check_steps_annotated(_runbook_rows(base)) == []
    assert check_chain_order(_runbook_rows(base)) == []
    assert check_constraints(base) == []
    assert check_token_registry(base, ("to be filled in", "to be registered",
                                       "<account>", "to be completed")) == []

    annotated = base.replace("| 人工 |", "| 待定 |", 1)
    assert check_steps_annotated(_runbook_rows(annotated)) != []

    swapped = list(_SAMPLE_ROWS)
    swapped[4], swapped[5] = swapped[5], swapped[4]
    assert check_chain_order(_runbook_rows(_sample_runbook(swapped))) != []

    no_late = base.replace("，亦不得晚于提交。", "。")
    assert check_constraints(no_late) != []

    no_fourth = base.replace("、`to be completed`", "")
    assert check_token_registry(no_fourth, ("to be filled in", "to be registered",
                                            "<account>", "to be completed")) != []


@skip_no_pair
def test_runbook_token_registry_matches_tool(tool):
    """The runbook word registry is locked in agreement with the scan tool's word registry."""
    text = RUNBOOK.read_text(encoding="utf-8")
    assert check_token_registry(text, tool.PLACEHOLDER_TOKENS) == []