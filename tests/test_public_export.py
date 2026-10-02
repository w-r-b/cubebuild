"""导出判据（产物集合类）＋痕迹门禁：白名单齐备／排除面零出现／脚本不在产物内／
幂等（含整体重建）／痕迹零命中；负例：误登记排除面、注入痕迹、登记项缺失 → 必须失败。

私有侧机制（导出清单＋入口）不在本工作区时整体跳过（沿用缺失即 skip 惯例——
公开仓不携带导出自身逻辑）。本模块自身亦经同一门禁扫描，故路径与痕迹样本均
按运行时构造（不直书受控字符串）。
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
    # 清单登记项齐备（登记即判齐备）
    for entry in mod.load_manifest()["entries"]:
        assert (out / entry.rstrip("/")).exists(), entry
    # 关键成员：层契约位于管线包内
    assert (out / "cubebuild" / "layer-mapping-v0.yaml").is_file()
    # 产物顶层 = 白名单顶层（顶层集合由清单派生——清单为唯一来源）
    expected_top = {e.rstrip("/").split("/")[0] for e in mod.load_manifest()["entries"]}
    assert {p.name for p in out.iterdir()} == expected_top
    assert report["n_files"] == len(_sha_set(out))


def test_excluded_faces_absent(mod, tmp_path):
    out = tmp_path / "out"
    mod.export_public(out)
    names = [str(p.relative_to(out)) for p in out.rglob("*")]
    # 排除面清单以导出模块的单一来源为准（DENIED）
    for bad in [d.rstrip("/") for d in mod.DENIED] + ["__pycache__"]:
        assert not any(
            n == bad or n.startswith(bad + "/") or bad in Path(n).parts
            for n in names
        ), bad


def test_idempotent_and_rebuild(mod, tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    mod.export_public(a)
    mod.export_public(b)
    assert _sha_set(a) == _sha_set(b)  # 二次导出与首次逐文件一致
    # 整体重建：预置陈旧文件不得残留
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
    # 合成「未净化的源」：伪造工作仓并注入一条内部痕迹 → 预检必须失败。
    # 样本按运行时拼接构造（本测试文件自身受同一门禁扫描）。
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
    # 无扩展名文本要件（如 LICENSE）同样须被预检覆盖（不得按扩展名漏检）。
    sample = "内部过程记录 " + "202" + "6-09-" + "12\n"
    fake = tmp_path / "repo"
    (fake / "cubebuild").mkdir(parents=True)
    (fake / "LICENSE").write_text(sample, encoding="utf-8")
    mf = tmp_path / "manifest.yaml"
    mf.write_text("version: 1\nentries:\n  - cubebuild/\n  - LICENSE\n", encoding="utf-8")
    with pytest.raises(mod.ExportViolation):
        mod.export_public(tmp_path / "out", manifest_path=mf, repo=fake)