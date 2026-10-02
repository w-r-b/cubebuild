"""开源要件判据：许可、README、依赖声明、引用元数据与 CI 工作流。

五项判据对仓库根目录本体检查（导出为逐字复制，故产物内同源）：
① LICENSE 为 MIT 全文（标志段落齐备）；
② README 六段锚点（定位／安装／复现边界／数据／引用／许可）齐备；复现边界段
   显式声明「约 22 GB 原始数据、部分不可再分发、不可由公开件重建、读者应使用
   公开版」；许可段显式分列代码 MIT 与数据 CC BY 4.0；并声明 CI 覆盖边界
   （完整套件需本地数据，不在 CI 覆盖）；
③ pyproject.toml 可解析：包名、版本（与管线包 __version__ 一致）、requires-python、
   依赖集合（与包内第三方 import 全集逐一对应）、命令行入口、MIT 许可、构建后端；
④ CITATION.cff 可解析：作者（姓名／ORCID／机构）、版本、DOI、MIT 许可，
   并显式分列数据侧 CC BY 4.0；
⑤ CI 工作流存在，范围与 README 声明一致（可编辑安装＋import 冒烟＋命令行冒烟＋
   两份纯算法测试），Python 3.11 单档。

负例（合成件）：许可文本被替换、复现边界句被删、依赖声明缺入口、依赖声明缺版本、
引用元数据缺版本——各自必须失败。私有侧机制（导出入口、痕迹扫描器）不在本工作区时，
相应核对整体跳过（沿用缺件跳过惯例）；本模块自身受公开面痕迹扫描约束，受词表约束的
字面量按运行时拼接构造。
"""

from __future__ import annotations

import ast
import importlib.util
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]

# ---------- 判据本体 ----------

MIT_MARKERS = (
    "MIT License",
    "Permission is hereby granted, free of charge",
    'THE SOFTWARE IS PROVIDED "AS IS"',
    "WITHOUT WARRANTY OF ANY KIND",
)

# README 六段锚点：段名 → 标题关键字（小写包含匹配）
SECTION_ANCHORS = (
    ("定位（管线而非数据集）", ("what this repository is", "positioning")),
    ("安装", ("installation",)),
    ("复现边界", ("reproduction",)),
    ("数据获取", ("data",)),
    ("引用", ("citation",)),
    ("许可", ("licence", "license")),
)

# 复现边界段四事实（必须落在该段正文内）
REPRODUCTION_FACTS = (
    (r"22\s*gb", "约 22 GB 原始数据"),
    (r"cannot be redistributed", "部分原始数据不可再分发"),
    (r"cannot be rebuilt from", "不可由公开件重建"),
    (r"public version", "读者应使用公开版"),
)

# 许可段须显式分列的两侧口径
LICENCE_SEPARATION = ("mit", "cc by 4.0")

# CI 覆盖边界声明（全 README 范围内）
CI_COVERAGE_MARKERS = ("not covered by ci", "full test suite")

# 包内第三方 import → 声明的发行名（PyPI 名）
DIST_BY_IMPORT = {
    "dask": "dask",
    "geopandas": "geopandas",
    "netCDF4": "netCDF4",
    "numpy": "numpy",
    "pandas": "pandas",
    "pyproj": "pyproj",
    "yaml": "PyYAML",
    "rasterio": "rasterio",
    "shapely": "shapely",
    "xarray": "xarray",
    "zarr": "zarr",
}

CI_ESSENTIALS = (
    ("可编辑安装", lambda t: "pip install -e" in t),
    ("import 冒烟", lambda t: "import cubebuild" in t),
    ("命令行冒烟", lambda t: re.search(r"cubebuild\s+--(version|help)", t) is not None),
    ("Python 3.11 单档", lambda t: re.search(r'python-version:\s*["\']?3\.11', t) is not None),
    ("纯算法测试", lambda t: "test_contract.py" in t and "test_pixel_area.py" in t),
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _package_version(root: Path) -> str | None:
    init = root / "cubebuild" / "__init__.py"
    if not init.is_file():
        return None
    m = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', _read(init))
    return m.group(1) if m else None


def check_licence(root: Path) -> list[str]:
    """LICENSE 必须是 MIT 全文（标志段落齐备）。"""
    p = root / "LICENSE"
    if not p.is_file():
        return ["LICENSE 缺失"]
    text = _read(p)
    return [f"LICENSE 缺少 MIT 标志串: {m!r}" for m in MIT_MARKERS if m not in text]


def _sections(text: str) -> dict[str, str]:
    """README → {小写标题: 段正文}（仅二级标题分段）。"""
    sections: dict[str, str] = {}
    cur: str | None = None
    for line in text.splitlines():
        if line.startswith("## "):
            cur = line[3:].strip().lower()
            sections[cur] = ""
        elif cur is not None:
            sections[cur] += line + "\n"
    return sections


def _section_body(sections: dict[str, str], keywords: tuple[str, ...]) -> str | None:
    for head, body in sections.items():
        if any(k in head for k in keywords):
            return body
    return None


def check_readme(root: Path) -> list[str]:
    """README 六段锚点＋复现边界四事实＋许可分列＋CI 覆盖边界。"""
    p = root / "README.md"
    if not p.is_file():
        return ["README.md 缺失"]
    text = _read(p)
    low = text.lower()
    problems: list[str] = []
    sections = _sections(text)
    bodies: dict[str, str] = {}
    for name, keywords in SECTION_ANCHORS:
        body = _section_body(sections, keywords)
        if body is None:
            problems.append(f"README 缺少「{name}」段锚点")
        else:
            bodies[name] = body
    reproduction = bodies.get("复现边界", "").lower()
    for pat, fact in REPRODUCTION_FACTS:
        if not re.search(pat, reproduction):
            problems.append(f"复现边界段缺少事实：{fact}")
    licence = bodies.get("许可", "").lower()
    for token in LICENCE_SEPARATION:
        if token not in licence:
            problems.append(f"许可段未显式分列：{token}")
    for marker in CI_COVERAGE_MARKERS:
        if marker not in low:
            problems.append(f"README 缺少 CI 覆盖边界声明：{marker}")
    return problems


def _dep_name(spec: str) -> str:
    return re.split(r"[<>=!~\[\s;]", spec.strip())[0].lower()


def _third_party_imports(pkg_dir: Path) -> set[str]:
    mods: set[str] = set()
    for f in sorted(pkg_dir.glob("*.py")):
        for node in ast.walk(ast.parse(_read(f))):
            if isinstance(node, ast.Import):
                mods.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                mods.add(node.module.split(".")[0])
    return {m for m in mods if m not in sys.stdlib_module_names and m != "cubebuild"}


def check_pyproject(root: Path) -> list[str]:
    """依赖声明：包名／版本／requires-python／依赖集合／入口／许可／构建后端。"""
    p = root / "pyproject.toml"
    if not p.is_file():
        return ["pyproject.toml 缺失"]
    try:
        data = tomllib.loads(_read(p))
    except tomllib.TOMLDecodeError as exc:
        return [f"pyproject.toml 不可解析: {exc}"]
    problems: list[str] = []
    proj = data.get("project") or {}
    if proj.get("name") != "cubebuild":
        problems.append(f"包名 {proj.get('name')!r} 不是 cubebuild")
    version = _package_version(root)
    if version is None:
        problems.append("管线包 __init__.py 缺 __version__（无法比对版本）")
    elif proj.get("version") != version:
        problems.append(f"版本 {proj.get('version')!r} 与管线包 {version!r} 不一致")
    req = str(proj.get("requires-python") or "")
    if not req:
        problems.append("缺 requires-python")
    elif "3.11" not in req:
        problems.append(f"requires-python {req!r} 未覆盖 CI 的 3.11 档")
    if not str((data.get("build-system") or {}).get("build-backend") or ""):
        problems.append("缺 build-system.build-backend（不可安装）")
    licence = proj.get("license")
    if isinstance(licence, dict):
        licence = licence.get("text")
    if not licence or "MIT" not in str(licence):
        problems.append(f"许可声明 {licence!r} 非 MIT（与代码许可口径不一致）")
    scripts = proj.get("scripts") or {}
    if scripts.get("cubebuild") != "cubebuild.cli:main":
        problems.append(f"命令行入口 {scripts.get('cubebuild')!r} 不是 cubebuild.cli:main")
    deps = [str(d) for d in (proj.get("dependencies") or [])]
    if not deps:
        problems.append("依赖集合为空")
    declared = {_dep_name(d) for d in deps}
    imports = _third_party_imports(root / "cubebuild")
    unknown = sorted(m for m in imports if m not in DIST_BY_IMPORT)
    for mod in unknown:
        problems.append(f"包内 import {mod!r} 未登记到声明名映射")
    if unknown:
        return problems
    expected = {DIST_BY_IMPORT[m].lower() for m in imports}
    for dist in sorted(expected - declared):
        problems.append(f"依赖集合缺 {dist}（包内 import 引用）")
    for dist in sorted(declared - expected):
        problems.append(f"依赖集合多出未使用声明 {dist}")
    return problems


def check_citation(root: Path) -> list[str]:
    """引用元数据：作者（姓名／ORCID／机构）、版本、DOI、MIT 许可。"""
    p = root / "CITATION.cff"
    if not p.is_file():
        return ["CITATION.cff 缺失"]
    try:
        data = yaml.safe_load(_read(p))
    except yaml.YAMLError as exc:
        return [f"CITATION.cff 不可解析: {exc}"]
    if not isinstance(data, dict):
        return ["CITATION.cff 顶层不是映射"]
    problems: list[str] = []
    for key, label in (("cff-version", "cff-version"), ("title", "标题"),
                       ("version", "版本"), ("doi", "DOI 或 DOI 占位")):
        if not data.get(key):
            problems.append(f"缺 {label}")
    if "MIT" not in str(data.get("license") or ""):
        problems.append(f"许可 {data.get('license')!r} 非 MIT（与代码许可口径不一致）")
    if "CC BY 4.0" not in _read(p):
        problems.append("引用元数据未显式分列数据侧许可（CC BY 4.0）")
    authors = data.get("authors")
    if not isinstance(authors, list) or not authors:
        problems.append("缺作者")
    else:
        first = authors[0] if isinstance(authors[0], dict) else {}
        if not (first.get("family-names") or first.get("name")):
            problems.append("作者缺姓名（family-names/name）")
        if not first.get("orcid"):
            problems.append("作者缺 ORCID")
        if not first.get("affiliation"):
            problems.append("作者缺机构")
    return problems


def check_ci(root: Path) -> list[str]:
    """CI 工作流：存在，且范围与 README 声明一致（不跑超范围测试）。"""
    workflows = sorted((root / ".github" / "workflows").glob("*.y*ml"))
    if not workflows:
        return ["缺少 .github/workflows/*.yml"]
    text = "\n".join(_read(f) for f in workflows)
    problems: list[str] = []
    for label, ok in CI_ESSENTIALS:
        if not ok(text):
            problems.append(f"CI 缺要件：{label}")
    if "matrix" in text.lower():
        problems.append("CI 含矩阵配置（应为 3.11 单档）")
    for line in text.splitlines():
        if "pytest" in line and not ("test_contract.py" in line or "test_pixel_area.py" in line):
            problems.append(f"pytest 行超出声明范围: {line.strip()}")
    return problems


# ---------- 本仓本体判据 ----------


def test_licence_mit_full_text():
    assert check_licence(REPO) == []


def test_readme_anchors_boundaries_and_separation():
    assert check_readme(REPO) == []


def test_pyproject_declaration_consistent():
    assert check_pyproject(REPO) == []


def test_citation_metadata_complete():
    assert check_citation(REPO) == []


def test_ci_workflow_scope_matches_readme():
    assert check_ci(REPO) == []


def test_cli_entry_smoke():
    """命令行入口可跑通（模块入口；控制台脚本由 CI 的可编辑安装冒烟覆盖）。"""
    version = _package_version(REPO)
    assert version, "管线包缺 __version__"
    r = subprocess.run(
        [sys.executable, "-m", "cubebuild", "--version"],
        cwd=REPO, capture_output=True, text=True, timeout=300,
    )
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == f"cubebuild {version}"


# ---------- 与私有侧机制的联动 ----------


def _find_private(name: str) -> Path | None:
    for p in sorted(REPO.rglob(name)):
        if "__pycache__" not in p.parts:
            return p
    return None


EXPORT_ENTRY = _find_private("export_public.py")
TRACE_SCAN = _find_private("trace_scan.py")


@pytest.mark.skipif(EXPORT_ENTRY is None, reason="私有侧导出机制不在本工作区（公开仓不带导出自身逻辑）")
def test_exported_artifact_essentials_pass(tmp_path):
    """导出产物内五件要件齐备且自洽（对产物本体检查，非源文件）。"""
    spec = importlib.util.spec_from_file_location("export_public", EXPORT_ENTRY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = tmp_path / "out"
    mod.export_public(out)
    problems = (
        check_licence(out)
        + check_readme(out)
        + check_pyproject(out)
        + check_citation(out)
        + check_ci(out)
    )
    assert problems == []


@pytest.mark.skipif(TRACE_SCAN is None, reason="私有侧痕迹扫描器不在本工作区")
def test_essentials_trace_free():
    """五件要件文本零内部痕迹（LICENSE 无后缀，不在导出预检的扫描扩展名内，此处补齐）。"""
    spec = importlib.util.spec_from_file_location("trace_scan", TRACE_SCAN)
    ts = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ts)
    targets = [REPO / n for n in ("README.md", "LICENSE", "pyproject.toml", "CITATION.cff")]
    targets += sorted((REPO / ".github" / "workflows").glob("*.y*ml"))
    hits = {str(p): found for p in targets for found in [ts.scan_file(p)] if found}
    assert hits == {}


# ---------- 负例（合成件；每条判据须能被证伪） ----------

MIT_SAMPLE = """MIT License

Copyright (c) 2026 <to be filled in>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software.

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
"""

REPRODUCTION_SAMPLE = (
    "The full cube cannot be rebuilt from this repository alone. Construction "
    "requires about 22 GB of raw input data that are not included here, and part "
    "of that raw data cannot be redistributed. Readers should use the public "
    "version of the cube."
)


def _readme_sample(reproduction: str) -> str:
    return f"""# cubebuild

Synthetic README for gate self-tests.

## What this repository is

The construction pipeline only.

## Installation

Requires Python 3.11. The full test suite requires local data and is therefore
not covered by CI.

## Reproduction scope

{reproduction}

## Data

The public version of the cube is archived on Zenodo (DOI: to be registered).

## Citation

See CITATION.cff.

## Licence

Code: MIT. Data: CC BY 4.0.
"""


PYPROJECT_SAMPLE = """[build-system]
requires = ["setuptools>=77"]
build-backend = "setuptools.build_meta"

[project]
name = "cubebuild"
version = "0.1.0"
requires-python = ">=3.11"
license = "MIT"
dependencies = ["numpy"]

[project.scripts]
cubebuild = "cubebuild.cli:main"
"""

CITATION_SAMPLE = """cff-version: "1.2.0"
title: "cubebuild"
version: "0.1.0"
license: MIT
notes: "The code is MIT; the public data cube release is licensed separately under CC BY 4.0."
doi: "<to be registered>"
authors:
  - family-names: "<to be filled in>"
    given-names: "<to be filled in>"
    orcid: "https://orcid.org/<to be filled in>"
    affiliation: "<to be filled in>"
"""

CI_SAMPLE = """name: CI
on: [push, pull_request]
jobs:
  ci:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -e ".[test]"
      - run: python -c "import cubebuild"
      - run: cubebuild --version
      - run: pytest tests/test_contract.py tests/test_pixel_area.py
"""


def _good_root(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "cubebuild").mkdir(parents=True)
    (root / ".github" / "workflows").mkdir(parents=True)
    (root / "cubebuild" / "__init__.py").write_text(
        'import numpy\n\n__version__ = "0.1.0"\n', encoding="utf-8")
    (root / "LICENSE").write_text(MIT_SAMPLE, encoding="utf-8")
    (root / "README.md").write_text(_readme_sample(REPRODUCTION_SAMPLE), encoding="utf-8")
    (root / "pyproject.toml").write_text(PYPROJECT_SAMPLE, encoding="utf-8")
    (root / "CITATION.cff").write_text(CITATION_SAMPLE, encoding="utf-8")
    (root / ".github" / "workflows" / "ci.yml").write_text(CI_SAMPLE, encoding="utf-8")
    return root


def _all_problems(root: Path) -> list[str]:
    return (check_licence(root) + check_readme(root) + check_pyproject(root)
            + check_citation(root) + check_ci(root))


def test_synthetic_root_passes_baseline(tmp_path):
    """合成件基准：齐备时全过（保证下述负例由变异触发，而非基线故障）。"""
    assert _all_problems(_good_root(tmp_path)) == []


def test_negative_licence_text_replaced(tmp_path):
    root = _good_root(tmp_path)
    (root / "LICENSE").write_text(
        "Apache License\nVersion 2.0, January 2004\nhttp://www.apache.org/licenses/\n",
        encoding="utf-8")
    assert check_licence(root) != []


def test_negative_reproduction_boundary_sentence_deleted(tmp_path):
    root = _good_root(tmp_path)
    (root / "README.md").write_text(
        _readme_sample("This section has been emptied."), encoding="utf-8")
    assert check_readme(root) != []


def test_negative_pyproject_missing_entry_point(tmp_path):
    root = _good_root(tmp_path)
    (root / "pyproject.toml").write_text(
        PYPROJECT_SAMPLE.replace(
            '[project.scripts]\ncubebuild = "cubebuild.cli:main"\n', ""),
        encoding="utf-8")
    assert check_pyproject(root) != []


def test_negative_pyproject_missing_version(tmp_path):
    root = _good_root(tmp_path)
    (root / "pyproject.toml").write_text(
        PYPROJECT_SAMPLE.replace('version = "0.1.0"\n', ""), encoding="utf-8")
    assert check_pyproject(root) != []


def test_negative_citation_missing_version(tmp_path):
    root = _good_root(tmp_path)
    (root / "CITATION.cff").write_text(
        CITATION_SAMPLE.replace('version: "0.1.0"\n', ""), encoding="utf-8")
    assert check_citation(root) != []


def test_negative_citation_missing_data_licence(tmp_path):
    root = _good_root(tmp_path)
    (root / "CITATION.cff").write_text(
        CITATION_SAMPLE.replace('notes: "The code is MIT; the public data cube '
                                'release is licensed separately under CC BY 4.0."\n', ""),
        encoding="utf-8")
    assert check_citation(root) != []