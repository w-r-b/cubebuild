"""Suite collectability criteria: the test suite inside the exported artifact
(the public content set) is collectable and runnable.

Criteria (applied to the artifact itself, produced into a temp directory via the
single export entry point):
① Collection has zero errors -- the full suite collects with no ERROR (a missing
   artifact at collection time must not raise an error);
② Execution consists only of passes and skips -- the full suite runs with zero
   failures and zero errors (a heavyweight criterion: several minutes, including
   synthetic recomputation items);
③ A missing artifact means skip -- modules depending on the excluded face (paper
   scripts) are skipped wholesale inside the artifact, and per module it is
   distinguishable "which missing artifact caused the skip" (the skip line
   exists, the reason is non-empty, and it names the missing artifact path);
④ Guards do not cause collateral damage -- inside the private repo the guard
   condition does not hold for the same batch of modules, and the tests run as
   usual (zero triggers).

Negative cases (the guards must be effective): perform targeted surgery on some
guarded module inside the artifact (AST location, structural block removal) --
⑤ remove the protected import (reverting it to a bare import) -> collection
   immediately errors (criterion ① fails);
⑥ remove only the module-level skipif (keeping the protected import) ->
   collection passes but execution immediately fails (criterion ② fails) --
   i.e. the collection and execution criteria are complementary and neither can
   be omitted.

Modules that depend on the excluded face are mechanically derived from the
**protected import name x paper/scripts module name** (no separate manifest, to
prevent a second source from drifting); an empty derived set is a failure (to
prevent a vacuous run from turning falsely green). When the private-side export
mechanism is not present in this workspace, the whole module is skipped
(following the missing-artifact skip convention -- the public repo does not
carry the export logic itself).
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



def _guarded_import_names(src: str) -> set[str]:
    """Set of module names referenced by module-level protected imports (inside try/except ImportError blocks)."""
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
    """Test modules depending on the excluded face = those whose protected imports reference a paper/scripts module."""
    return [p.name for p in sorted(tests_dir.glob("test_*.py"))
            if _guarded_import_names(p.read_text(encoding="utf-8")) & PAPER_SCRIPTS]


def _guard_reasons(src: str) -> list[str]:
    """Reason strings of module-level skipif guards (obtained via AST from the value of the reason keyword on pytestmark)."""
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
    """Remove the protected import (reverting it to a bare import): a module-level try/except ImportError block -> its try body."""
    for node in ast.parse(src).body:
        if isinstance(node, ast.Try) and any(
                isinstance(h.type, ast.Name) and h.type.id == "ImportError"
                for h in node.handlers):
            seg = ast.get_source_segment(src, node)
            body = "".join(ast.unparse(s) + "\n" for s in node.body)
            return src.replace(seg, body, 1), True
    return src, False


def _strip_pytestmark(src: str) -> tuple[str, bool]:
    """Remove the module-level skipif guard: the entire pytestmark assignment statement is removed."""
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "pytestmark" for t in node.targets):
            seg = ast.get_source_segment(src, node)
            return src.replace(seg + "\n", "", 1), True
    return src, False


SURGERY_SUBJECT = "test_translation_gate.py"


def _mutate_module(artifact: Path, tmp_path: Path, name: str, mutate) -> Path:
    """Artifact copy + targeted surgery on module name (an explicit failure if the guard form has changed)."""
    mut = tmp_path / "mut"
    shutil.copytree(artifact, mut)
    target = mut / "tests" / name
    src = target.read_text(encoding="utf-8")
    mutated, applied = mutate(src)
    assert applied and mutated != src, "守卫块未找到（形态已变，须同步本判据）"
    target.write_text(mutated, encoding="utf-8")
    return mut



def test_artifact_collects_zero_errors(artifact: Path):
    """① The full suite inside the artifact collects with zero errors."""
    r = _pytest(artifact, ["--collect-only", "-q", "tests/"])
    assert r.returncode == 0, r.stdout + r.stderr
    assert not re.search(r"\d+ errors? during collection", r.stdout), r.stdout


def test_artifact_full_suite_run_only_pass_or_skip(artifact: Path):
    """② The full suite inside the artifact runs with only passes and skips (heavyweight: overlaps with the synthetic recomputation items of the private full suite)."""
    r = _pytest(artifact, ["-q"], timeout=1800)
    tail = r.stdout[-2000:]
    assert r.returncode == 0, tail + r.stderr
    assert not re.search(r"\d+ failed", r.stdout), tail
    assert not re.search(r"\d+ errors?", r.stdout), tail
    assert "ERROR " not in r.stdout, tail


def test_artifact_excluded_face_modules_skip_with_visible_reason(artifact: Path):
    """③ Modules depending on the excluded face are skipped wholesale inside the artifact, and per module it is distinguishable "which missing artifact caused the skip"."""
    tests_dir = artifact / "tests"
    names = _excluded_face_modules(tests_dir)
    assert names, "产物内未派生到依赖被排除面的模块（判据空跑）"
    reasons = _guard_reasons_by_module(tests_dir, names)
    for n in names:
        assert reasons[n], f"{n} 缺模块级 skipif 守卫（应跳过而无守卫）"
        for reason in reasons[n]:
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
    """④ Inside the private repo (paper side complete) the guard condition does not hold: the tests run as usual and are not skipped."""
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



def test_negative_guard_import_stripped_collection_errors(artifact: Path, tmp_path: Path):
    """⑤ Remove the protected import -> under the missing-artifact environment collection immediately errors (criterion ① fails)."""
    assert SURGERY_SUBJECT in _excluded_face_modules(artifact / "tests")
    mut = _mutate_module(artifact, tmp_path, SURGERY_SUBJECT, _strip_import_guard)
    r = _pytest(mut, ["--collect-only", "-q", f"tests/{SURGERY_SUBJECT}"])
    assert r.returncode != 0, r.stdout
    assert re.search(r"\d+ errors? during collection", r.stdout), r.stdout


def test_negative_skipif_stripped_run_fails(artifact: Path, tmp_path: Path):
    """⑥ Remove only the module-level skipif -> collection passes but execution fails (criterion ② fails)."""
    assert SURGERY_SUBJECT in _excluded_face_modules(artifact / "tests")
    mut = _mutate_module(artifact, tmp_path, SURGERY_SUBJECT, _strip_pytestmark)
    collect = _pytest(mut, ["--collect-only", "-q", f"tests/{SURGERY_SUBJECT}"])
    assert collect.returncode == 0, collect.stdout
    run = _pytest(mut, ["-q", f"tests/{SURGERY_SUBJECT}"])
    assert run.returncode != 0, run.stdout
    assert "failed" in run.stdout, run.stdout