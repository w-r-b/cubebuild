"""套件可收集性判据：导出产物（公开内容集）内测试套件可收集、可运行。

判据（对产物本体，经单一导出入口产到临时目录）：
① 收集零错误——全套收集无 ERROR（收集期缺件不得报错）；
② 运行仅通过与跳过——全套运行零失败零错误（重量级判据：约数分钟，含合成重算项）；
③ 缺件即跳过——依赖被排除面（论文脚本）的模块在产物内整体跳过，且逐模块可辨
   「因缺何件而跳过」（跳过行存在、原因非空、点名缺失件路径）；
④ 守卫不误伤——私有仓内同一批模块的守卫条件不成立，测试照常运行（零触发）。

负例（守卫必须有效）：对产物内某守卫模块做定点手术（AST 定位、结构删块）——
⑤ 去掉受保护导入（还原为裸导入）→ 收集即报错（判据①失败）；
⑥ 只去掉模块级 skipif（受保护导入保留）→ 收集虽过、运行即失败（判据②失败）——
   即收集与运行两条判据互补，缺一不可。

依赖被排除面的模块由**受保护导入名 × paper/scripts 模块名**机械派生（不另
设清单，防第二来源漂移）；派生集合为空即判失败（防空跑假绿）。私有侧导出
机制不在本工作区时整体跳过（沿用缺件跳过惯例——公开仓不带导出自身逻辑）。
"""
from __future__ import annotations

import ast
import importlib.util
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PAPER_SCRIPTS = {p.stem for p in (REPO / "paper" / "scripts").glob("*.py")}


def _find_private(name: str) -> Path | None:
    for p in sorted(REPO.rglob(name)):
        if "__pycache__" not in p.parts:
            return p
    return None


ENTRY = _find_private("export_public.py")

pytestmark = pytest.mark.skipif(
    ENTRY is None, reason="私有侧导出机制不在本工作区（公开仓不带导出自身逻辑）")


@pytest.fixture(scope="module")
def artifact(tmp_path_factory: pytest.TempPathFactory) -> Path:
    spec = importlib.util.spec_from_file_location("export_public", ENTRY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = tmp_path_factory.mktemp("collectability") / "out"
    mod.export_public(out)
    return out


# ---------- 派生与运行助手 ----------

def _guarded_import_names(src: str) -> set[str]:
    """模块级受保护导入（try/except ImportError 块内）引用的模块名集合。"""
    names: set[str] = set()
    for node in ast.parse(src).body:
        if isinstance(node, ast.Try) and any(
                isinstance(h.type, ast.Name) and h.type.id == "ImportError"
                for h in node.handlers):
            for sub in ast.walk(node):
                if isinstance(sub, ast.Import):
                    names.update(a.name.split(".")[0] for a in sub.names)
                elif isinstance(sub, ast.ImportFrom) and sub.level == 0 and sub.module:
                    names.add(sub.module.split(".")[0])
    return names


def _excluded_face_modules(tests_dir: Path) -> list[str]:
    """依赖被排除面的测试模块 = 受保护导入引用 paper/scripts 模块者。"""
    return [p.name for p in sorted(tests_dir.glob("test_*.py"))
            if _guarded_import_names(p.read_text(encoding="utf-8")) & PAPER_SCRIPTS]


def _guard_reasons(src: str) -> list[str]:
    """模块级 skipif 守卫的原因串（AST 取 pytestmark 上 reason 关键字的值）。"""
    reasons: list[str] = []
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "pytestmark" for t in node.targets):
            for call in ast.walk(node.value):
                if isinstance(call, ast.Call):
                    for kw in call.keywords:
                        if kw.arg == "reason" and isinstance(kw.value, ast.Constant):
                            reasons.append(kw.value.value)
    return reasons


def _guard_reasons_by_module(tests_dir: Path, names: list[str]) -> dict[str, list[str]]:
    return {n: _guard_reasons((tests_dir / n).read_text(encoding="utf-8")) for n in names}


def _pytest(cwd: Path, args: list[str], timeout: int = 600) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", *args],
        cwd=cwd, capture_output=True, text=True, timeout=timeout)


def _strip_import_guard(src: str) -> tuple[str, bool]:
    """去掉受保护导入（还原为裸导入）：模块级 try/except ImportError 块 → 其 try 体。"""
    for node in ast.parse(src).body:
        if isinstance(node, ast.Try) and any(
                isinstance(h.type, ast.Name) and h.type.id == "ImportError"
                for h in node.handlers):
            seg = ast.get_source_segment(src, node)
            body = "".join(ast.unparse(s) + "\n" for s in node.body)
            return src.replace(seg, body, 1), True
    return src, False


def _strip_pytestmark(src: str) -> tuple[str, bool]:
    """去掉模块级 skipif 守卫：pytestmark 赋值语句整段移除。"""
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "pytestmark" for t in node.targets):
            seg = ast.get_source_segment(src, node)
            return src.replace(seg + "\n", "", 1), True
    return src, False


# 负例手术对象：单守卫代表模块（守卫形态与其余守卫模块一致；须落在派生集合内）
SURGERY_SUBJECT = "test_translation_gate.py"


def _mutate_module(artifact: Path, tmp_path: Path, name: str, mutate) -> Path:
    """产物副本 + 对 name 模块的定点手术（守卫形态变了即显式失败）。"""
    mut = tmp_path / "mut"
    shutil.copytree(artifact, mut)
    target = mut / "tests" / name
    src = target.read_text(encoding="utf-8")
    mutated, applied = mutate(src)
    assert applied and mutated != src, "守卫块未找到（形态已变，须同步本判据）"
    target.write_text(mutated, encoding="utf-8")
    return mut


# ---------- 判据本体（产物） ----------

def test_artifact_collects_zero_errors(artifact: Path):
    """① 产物内全套收集零错误。"""
    r = _pytest(artifact, ["--collect-only", "-q", "tests/"])
    assert r.returncode == 0, r.stdout + r.stderr
    assert not re.search(r"\d+ errors? during collection", r.stdout), r.stdout


def test_artifact_full_suite_run_only_pass_or_skip(artifact: Path):
    """② 产物内全套运行仅含通过与跳过（重量级：与私有全量套件的合成重算项重叠）。"""
    r = _pytest(artifact, ["-q"], timeout=1800)
    tail = r.stdout[-2000:]
    assert r.returncode == 0, tail + r.stderr
    assert not re.search(r"\d+ failed", r.stdout), tail
    assert not re.search(r"\d+ errors?", r.stdout), tail
    assert "ERROR " not in r.stdout, tail


def test_artifact_excluded_face_modules_skip_with_visible_reason(artifact: Path):
    """③ 依赖被排除面的模块在产物内整体跳过，且逐模块可辨「因缺何件而跳过」。"""
    tests_dir = artifact / "tests"
    names = _excluded_face_modules(tests_dir)
    assert names, "产物内未派生到依赖被排除面的模块（判据空跑）"
    reasons = _guard_reasons_by_module(tests_dir, names)
    for n in names:
        assert reasons[n], f"{n} 缺模块级 skipif 守卫（应跳过而无守卫）"
        for reason in reasons[n]:  # 跳过原因须点名缺失件（文件路径样）
            assert re.search(r"\S+\.py", reason), f"{n} 跳过原因未点名缺失件: {reason!r}"
    r = _pytest(artifact, ["-q", "-rs", *[f"tests/{n}" for n in names]])
    assert r.returncode == 0, r.stdout + r.stderr
    assert "failed" not in r.stdout and "ERROR" not in r.stdout, r.stdout
    assert " skipped" in r.stdout, r.stdout
    for n in names:
        assert re.search(rf"SKIPPED \[\d+\] tests/{re.escape(n)}:\d+: \S", r.stdout), n
        assert any(reason in r.stdout for reason in reasons[n]), (
            f"{n} 的跳过原因未出现在输出（读者不可辨）")


def test_guards_do_not_false_skip_in_private_repo():
    """④ 私有仓（论文侧齐备）内守卫条件不成立：测试照常运行，不予跳过。"""
    names = _excluded_face_modules(REPO / "tests")
    assert names, "本仓未派生到依赖被排除面的模块（判据空跑）"
    reasons = _guard_reasons_by_module(REPO / "tests", names)
    r = _pytest(REPO, ["-q", "-rs", *[f"tests/{n}" for n in names]])
    assert r.returncode == 0, r.stdout + r.stderr
    for n in names:
        assert reasons[n], f"{n} 缺模块级 skipif 守卫"
        assert not any(reason in r.stdout for reason in reasons[n]), (
            f"{n} 的守卫在私有仓触发（误伤）")
    m = re.search(r"(\d+) passed", r.stdout)
    assert m and int(m.group(1)) > 0, r.stdout


# ---------- 负例（守卫必须有效；对产物本体定点手术） ----------

def test_negative_guard_import_stripped_collection_errors(artifact: Path, tmp_path: Path):
    """⑤ 删掉受保护导入 → 缺件环境下收集即报错（判据①失败）。"""
    assert SURGERY_SUBJECT in _excluded_face_modules(artifact / "tests")
    mut = _mutate_module(artifact, tmp_path, SURGERY_SUBJECT, _strip_import_guard)
    r = _pytest(mut, ["--collect-only", "-q", f"tests/{SURGERY_SUBJECT}"])
    assert r.returncode != 0, r.stdout
    assert re.search(r"\d+ errors? during collection", r.stdout), r.stdout


def test_negative_skipif_stripped_run_fails(artifact: Path, tmp_path: Path):
    """⑥ 只去掉模块级 skipif → 收集虽过、运行即失败（判据②失败）。"""
    assert SURGERY_SUBJECT in _excluded_face_modules(artifact / "tests")
    mut = _mutate_module(artifact, tmp_path, SURGERY_SUBJECT, _strip_pytestmark)
    collect = _pytest(mut, ["--collect-only", "-q", f"tests/{SURGERY_SUBJECT}"])
    assert collect.returncode == 0, collect.stdout  # 此类缺陷收集判据抓不到
    run = _pytest(mut, ["-q", f"tests/{SURGERY_SUBJECT}"])
    assert run.returncode != 0, run.stdout
    assert "failed" in run.stdout, run.stdout