"""发布执行单与「占位串零残留」检索工具的判据。

两部分：
- 执行单结构：执行序表逐行标注「人工／自动」与前置条件；步骤链首现序
  （建私有仓 → 回填手稿 → 投稿提交 → 转公开 → Zenodo-GitHub 绑定 →
  铸造代码 DOI → 回填 CITATION → 零残留检索）严格递增；两份移交要件
  （冻结清单净化、公开仓 .gitignore）落在执行序内；硬约束句齐备
  （转公开两侧时点、代码 DOI 铸造约束、同批回填、人工步骤声明）；
  检索词表与工具词表锁定（防双面漂移）。以上判据均带合成件负例。
- 检索工具：合成件基准（全回填零命中）＋负例（人为保留一处占位必须命中）＋
  四类占位串全覆盖（含大小写变体）＋目标缺件 fail-closed＋默认范围覆盖
  手稿与公开仓四件要件＋CLI 退出码 0／1／2。

私有侧制品不在本工作区（公开仓）时整体跳过（沿用缺件跳过惯例）。
本模块随测试套件进入公开面，其文本受公开面痕迹扫描约束。
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


# ---------- 合成件（供检索工具判据与负例使用） ----------

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
    """私有仓最小同构树：手稿＋四件要件。"""
    root = tmp_path / "root"
    (root / "paper" / "manuscript").mkdir(parents=True)
    (root / "paper" / "manuscript" / "SEDC-manuscript-nsd-en.md").write_text(
        manuscript_text, encoding="utf-8")
    for name in ("README.md", "CITATION.cff", "pyproject.toml", "LICENSE"):
        (root / name).write_text("filled\n", encoding="utf-8")
    return root


# ---------- 检索工具判据 ----------


@skip_no_tool
def test_scan_filled_sample_is_clean(tool, tmp_path):
    """基准：全回填合成件零命中（保证下述负例由占位触发而非夹具故障）。"""
    f = tmp_path / "manuscript.md"
    f.write_text(FILLED_MANUSCRIPT, encoding="utf-8")
    assert tool.scan_file(f) == []


@skip_no_tool
def test_negative_retained_single_placeholder_hits(tool, tmp_path):
    """负例：人为保留一处占位，检索必须命中，且只命中保留的那一处。"""
    f = tmp_path / "manuscript.md"

    f.write_text(MANUSCRIPT_SAMPLE, encoding="utf-8")
    assert len(tool.scan_file(f)) == 2  # 两处俱在时两处皆中

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
    """四类占位串逐一可检（含大小写变体：有角括号的声明式占位）。"""
    f = tmp_path / "essentials.txt"
    f.write_text(ESSENTIALS_SAMPLE, encoding="utf-8")
    tokens = [tok for _ln, tok, _line in tool.scan_file(f)]
    assert tokens == ["<account>", "to be registered", "to be filled in",
                      "to be completed"]


@skip_no_tool
def test_default_targets_cover_manuscript_and_essentials(tool):
    """默认范围＝手稿＋四件要件；目标在当前工作区全部存在（缺件即 fail-closed）。"""
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
    """目标缺件不得当作零残留：missing 非空且不出现在命中里。"""
    report = tool.report("all", root=tmp_path)
    assert len(report["missing"]) == 5
    assert report["hits"] == []


@skip_no_tool
def test_unknown_scope_rejected(tool):
    """未知范围名即显式失败（fail-closed），不静默回退默认范围。"""
    with pytest.raises(ValueError):
        tool.targets("everything")


@skip_no_tool
def test_cli_exit_codes(tool, tmp_path, monkeypatch):
    """CLI 契约：0 零残留／1 命中／2 缺件。"""
    monkeypatch.setattr(sys, "argv", ["placeholder_scan.py", "--manuscript"])

    monkeypatch.setattr(tool, "REPO", _fake_tree(tmp_path / "clean", FILLED_MANUSCRIPT))
    assert tool.main() == 0

    monkeypatch.setattr(tool, "REPO", _fake_tree(tmp_path / "hit", MANUSCRIPT_SAMPLE))
    assert tool.main() == 1

    monkeypatch.setattr(tool, "REPO", tmp_path / "empty")
    assert tool.main() == 2


# ---------- 执行单结构判据 ----------

CHAIN_ANCHORS = ("建私有仓", "回填手稿", "投稿提交", "转公开",
                 "Zenodo-GitHub 绑定", "铸造代码 DOI", "回填 CITATION", "零残留检索")
RUNBOOK_CONSTRAINTS = (
    "不得早于", "不得晚于",           # 转公开两侧时点
    "不支持私有仓", "不能早于",        # 代码 DOI 铸造约束
    "同批",                            # 同批回填
    "人工步骤", "超出范围",            # 人工步骤声明
    "四件要件",                        # 回填与检索范围
)


def _runbook_rows(text: str) -> list[list[str]]:
    """「执行序」区内步骤表行（首列为数字的表行）。"""
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
    """每步骤五列齐备、标注人工／自动、前置条件非空；两份移交要件在执行序内。"""
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
    """步骤链每锚的首现行号须严格递增。"""
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
    """硬约束句齐备。"""
    return [f"执行单缺约束句: {t}" for t in RUNBOOK_CONSTRAINTS if t not in text]


def check_token_registry(text: str, tokens) -> list[str]:
    """执行单「检索词表」行的词条集合与工具词表一致（防双面漂移）。"""
    line = next((l for l in text.splitlines() if l.startswith("检索词表")), "")
    if not line:
        return ["执行单缺检索词表行"]
    found = set(re.findall(r"`([^`]+)`", line))
    problems = [f"执行单词表缺: {t}" for t in sorted(set(tokens) - found)]
    problems += [f"执行单词表多出: {t}" for t in sorted(found - set(tokens))]
    return problems


# ---------- 执行单：合成件基准与负例 ----------

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
    """真实执行单：四类结构判据全过。"""
    text = RUNBOOK.read_text(encoding="utf-8")
    rows = _runbook_rows(text)
    assert len(rows) >= 15
    assert check_steps_annotated(rows) == []
    assert check_chain_order(rows) == []
    assert check_constraints(text) == []


@skip_no_runbook
def test_runbook_negative_examples():
    """畸形合成件负例：以下四种变异各自必须被对应判据命中（判据可被证伪）。"""
    base = _sample_runbook()
    assert check_steps_annotated(_runbook_rows(base)) == []
    assert check_chain_order(_runbook_rows(base)) == []
    assert check_constraints(base) == []
    assert check_token_registry(base, ("to be filled in", "to be registered",
                                       "<account>", "to be completed")) == []

    annotated = base.replace("| 人工 |", "| 待定 |", 1)
    assert check_steps_annotated(_runbook_rows(annotated)) != []

    swapped = list(_SAMPLE_ROWS)
    swapped[4], swapped[5] = swapped[5], swapped[4]  # 转公开 抢在 投稿提交 之前
    assert check_chain_order(_runbook_rows(_sample_runbook(swapped))) != []

    no_late = base.replace("，亦不得晚于提交。", "。")
    assert check_constraints(no_late) != []

    no_fourth = base.replace("、`to be completed`", "")
    assert check_token_registry(no_fourth, ("to be filled in", "to be registered",
                                            "<account>", "to be completed")) != []


@skip_no_pair
def test_runbook_token_registry_matches_tool(tool):
    """执行单词表与检索工具词表锁定一致。"""
    text = RUNBOOK.read_text(encoding="utf-8")
    assert check_token_registry(text, tool.PLACEHOLDER_TOKENS) == []