"""Export criteria (artifact-set class) + trace gate: whitelist complete /
excluded faces never present / scripts not in the artifact / idempotent
(including full rebuild) / zero trace hits; negative cases: an excluded face
mistakenly registered, an injected trace, a missing registered entry → must fail.

When the private-side mechanisms (export manifest + entry point) are absent from
this workspace everything is skipped (following the missing-artifact skip
convention — the public repository does not carry the export logic itself). This
module is itself scanned by the same gate, so paths and trace samples are
constructed at runtime (controlled strings are not written literally).
"""

import hashlib
import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _find_entry() -> Path | None:
    for p in sorted(REPO.rglob("export_public.py")):
        if "__pycache__" not in p.parts:
            return p
    return None


ENTRY = _find_entry()
MANIFEST = ENTRY.with_name("export-manifest.yaml") if ENTRY else None

pytestmark = pytest.mark.skipif(
    not (ENTRY and MANIFEST and MANIFEST.is_file()),
    reason="私有侧导出机制不在本工作区（公开仓不带导出自身逻辑）",
)


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("export_public", ENTRY)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _sha_set(root: Path) -> dict:
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def test_whitelist_paths_complete(mod, tmp_path):
    out = tmp_path / "out"
    report = mod.export_public(out)
    for entry in mod.load_manifest()["entries"]:
        assert (out / entry.rstrip("/")).exists(), entry
    assert (out / "cubebuild" / "layer-mapping-v0.yaml").is_file()
    expected_top = {e.rstrip("/").split("/")[0] for e in mod.load_manifest()["entries"]}
    assert {p.name for p in out.iterdir()} == expected_top
    assert report["n_files"] == len(_sha_set(out))


def test_excluded_faces_absent(mod, tmp_path):
    out = tmp_path / "out"
    mod.export_public(out)
    names = [str(p.relative_to(out)) for p in out.rglob("*")]
    for bad in [d.rstrip("/") for d in mod.DENIED] + ["__pycache__"]:
        assert not any(
            n == bad or n.startswith(bad + "/") or bad in Path(n).parts
            for n in names
        ), bad


def test_idempotent_and_rebuild(mod, tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    mod.export_public(a)
    mod.export_public(b)
    assert _sha_set(a) == _sha_set(b)
    stale = a / "cubebuild" / "stale-leftover.txt"
    stale.write_text("stale", encoding="utf-8")
    mod.export_public(a)
    assert not stale.exists()
    assert _sha_set(a) == _sha_set(b)


def test_script_not_in_artifact(mod, tmp_path):
    out = tmp_path / "out"
    mod.export_public(out)
    names = {p.name for p in out.rglob("*")}
    assert "export_public.py" not in names
    assert "export-manifest.yaml" not in names


def test_artifact_trace_free(mod, tmp_path):
    out = tmp_path / "out"
    mod.export_public(out)
    assert mod.scan_artifact(out) == []


def test_negative_denied_path_registration(mod, tmp_path):
    mf = tmp_path / "manifest.yaml"
    mf.write_text("version: 1\nentries:\n  - paper/\n", encoding="utf-8")
    with pytest.raises(mod.ExportViolation):
        mod.export_public(tmp_path / "out", manifest_path=mf)


def test_negative_trace_injected(mod, tmp_path):
    sample = '"""源（磁盘实证 2026-09-' + '12）。"""\n'
    fake = tmp_path / "repo"
    (fake / "cubebuild").mkdir(parents=True)
    (fake / "tests").mkdir()
    (fake / "cubebuild" / "x.py").write_text(sample, encoding="utf-8")
    mf = tmp_path / "manifest.yaml"
    mf.write_text("version: 1\nentries:\n  - cubebuild/\n  - tests/\n", encoding="utf-8")
    with pytest.raises(mod.ExportViolation):
        mod.export_public(tmp_path / "out", manifest_path=mf, repo=fake)


def test_negative_missing_entry(mod, tmp_path):
    mf = tmp_path / "manifest.yaml"
    mf.write_text("version: 1\nentries:\n  - nonexistent-dir/\n", encoding="utf-8")
    with pytest.raises(mod.ExportViolation):
        mod.export_public(tmp_path / "out", manifest_path=mf)


def test_negative_trace_in_suffixless_file(mod, tmp_path):
    sample = "内部过程记录 " + "202" + "6-09-" + "12\n"
    fake = tmp_path / "repo"
    (fake / "cubebuild").mkdir(parents=True)
    (fake / "LICENSE").write_text(sample, encoding="utf-8")
    mf = tmp_path / "manifest.yaml"
    mf.write_text("version: 1\nentries:\n  - cubebuild/\n  - LICENSE\n", encoding="utf-8")
    with pytest.raises(mod.ExportViolation):
        mod.export_public(tmp_path / "out", manifest_path=mf, repo=fake)


# ---------- public-face language: comment stripping + zero-Han gate ----------

def test_artifact_comment_and_docstring_face_han_free(mod, tmp_path):
    """Public-face language criterion: the artifact has no CJK in comment/docstring regions
    (checked for .py and .yaml alike)."""
    out = tmp_path / "out"
    mod.export_public(out)
    assert mod.scan_han_artifact(out) == []


def test_chinese_comments_stripped_english_kept_code_intact(mod, tmp_path):
    """CJK comments are stripped from the artifact; English comments and all code survive."""
    fake = tmp_path / "repo"
    (fake / "cubebuild").mkdir(parents=True)
    src = (
        "# English comment kept\n"
        "X = 1  # 行尾中文注释\n"
        "# 整行中文注释\n"
        "def f():\n"
        '    """English docstring."""\n'
        "    return X\n"
    )
    (fake / "cubebuild" / "x.py").write_text(src, encoding="utf-8")
    mf = tmp_path / "manifest.yaml"
    mf.write_text("version: 1\nentries:\n  - cubebuild/\n", encoding="utf-8")
    out = tmp_path / "out"
    mod.export_public(out, manifest_path=mf, repo=fake)
    art = (out / "cubebuild" / "x.py").read_text(encoding="utf-8")
    assert "English comment kept" in art
    assert mod.ts.HAN_RE.search(art) is None
    assert mod.scan_han_artifact(out) == []


def test_chinese_string_constant_kept_and_not_flagged(mod, tmp_path):
    """Boundary: CJK inside ordinary string constants is neither stripped nor flagged
    (it belongs to the runtime-output / product-metadata face, not the comment face)."""
    fake = tmp_path / "repo"
    (fake / "cubebuild").mkdir(parents=True)
    (fake / "cubebuild" / "x.py").write_text('NOTE = "中文注记"\n', encoding="utf-8")
    mf = tmp_path / "manifest.yaml"
    mf.write_text("version: 1\nentries:\n  - cubebuild/\n", encoding="utf-8")
    out = tmp_path / "out"
    mod.export_public(out, manifest_path=mf, repo=fake)
    assert "中文注记" in (out / "cubebuild" / "x.py").read_text(encoding="utf-8")
    assert mod.scan_han_artifact(out) == []


def test_chinese_yaml_comment_stripped_value_kept(mod, tmp_path):
    """A CJK YAML full-line comment is stripped; a CJK value stays (values are consumed by code)."""
    fake = tmp_path / "repo"
    (fake / "cubebuild").mkdir(parents=True)
    (fake / "cubebuild" / "c.yaml").write_text("# 中文注释\nkey: 中文值\n", encoding="utf-8")
    mf = tmp_path / "manifest.yaml"
    mf.write_text("version: 1\nentries:\n  - cubebuild/\n", encoding="utf-8")
    out = tmp_path / "out"
    mod.export_public(out, manifest_path=mf, repo=fake)
    art = (out / "cubebuild" / "c.yaml").read_text(encoding="utf-8")
    assert "中文注释" not in art and "中文值" in art
    assert mod.scan_han_artifact(out) == []


def test_negative_chinese_docstring_fails_gate(mod, tmp_path):
    """Negative: a CJK docstring is outside the stripping face → the zero-Han gate fails closed
    (docstrings are handled at the source by translation, not by stripping)."""
    fake = tmp_path / "repo"
    (fake / "cubebuild").mkdir(parents=True)
    (fake / "cubebuild" / "x.py").write_text(
        'def f():\n    """中文 docstring。"""\n    return 1\n', encoding="utf-8")
    mf = tmp_path / "manifest.yaml"
    mf.write_text("version: 1\nentries:\n  - cubebuild/\n", encoding="utf-8")
    with pytest.raises(mod.ExportViolation):
        mod.export_public(tmp_path / "out", manifest_path=mf, repo=fake)