"""图件机械门槛「交付格式一致性」判据测试。

接缝：`paper/scripts/check_figures.py` 的逐图判定入口 `check_pdf()`——沿用现有接缝，
不锚定脚本内部调用方式、不锚定具体块号（脆弱）。

反例是**真实夹具**（非人造样本）：git 历史里收口提交的缺陷版 fig07 PDF
（matplotlib 3.11.0 PDF 后端丢弃负数据坐标的集合元素，886 个六边形只写出 226 个），
连同该次输出的直出 PNG。若判据被削弱或阈值被抬高到失去鉴别力，本测试即失败。
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "paper/scripts"))

try:
    import check_figures  # noqa: E402
except ImportError:  # 公开仓不含论文侧脚本与产物：缺件即跳过（沿用缺件跳过惯例）
    check_figures = None

pytestmark = pytest.mark.skipif(
    check_figures is None, reason="论文侧图件检查器不在本工作区（缺 paper/scripts/check_figures.py）")

FIGS = REPO / "paper" / "figures"
# 收口提交：fig07 PDF 丢几何的缺陷版（修复前的交付态）
DEFECT_COMMIT = "be8f768"
DEFECT_FIG = "fig07-triple-scatter"

CURRENT_FIGS = sorted(p.stem for p in FIGS.glob("fig*.pdf"))


@pytest.fixture(scope="module")
def defect_pdf(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """从 git 历史取出缺陷版 fig07 的 PDF 与同次直出 PNG，落成一对同目录夹具。"""
    d = tmp_path_factory.mktemp("defect-fig07")
    for suffix in (".pdf", ".png"):
        blob = subprocess.run(
            ["git", "show", f"{DEFECT_COMMIT}:paper/figures/{DEFECT_FIG}{suffix}"],
            cwd=REPO, capture_output=True, check=True,
        ).stdout
        (d / f"{DEFECT_FIG}{suffix}").write_bytes(blob)
    return d / f"{DEFECT_FIG}.pdf"


@pytest.mark.parametrize("name", CURRENT_FIGS)
def test_figure_passes_delivery_gate(name: str):
    """正例：交付态五图（图号重排后体系；含图 3–4 地图类位图档）
    全部 PASS，不误报。"""
    ok, problems = check_figures.check_pdf(FIGS / f"{name}.pdf")
    assert ok, problems


def test_defect_fixture_flags_delivery_mismatch(defect_pdf: Path):
    """反例：真实缺陷版 fig07 必须 FAIL，并给出最大块差与所在块位置。"""
    ok, problems = check_figures.check_pdf(defect_pdf)
    assert not ok
    assert len(problems) == 1, problems
    msg = problems[0]
    assert "最大块差" in msg and "块(" in msg, msg
    diff, _, _ = check_figures.ink_block_diff(defect_pdf, defect_pdf.with_suffix(".png"))
    # 鉴别力守护：缺陷量级远高于阈值与正常图上限（修复前实测 0.707 / 阈值 0.05 /
    # 正常图 ≤0.023）——阈值若被抬高到接近缺陷量级，本断言即失败
    assert diff > 0.3, f"缺陷量级 {diff:.3f} 过低，判据鉴别力不足"


# ---------- S8 图面记号扫描：正反例 ----------

def _mini_fig(tmp_path: Path, label: str, size: float = 8) -> Path:
    """12cm 宽、默认 8pt、纯矢量、嵌入 TrueType 的最小合规 PDF+PNG 对（其余门槛全过）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12 / 2.54, 2.0))
    fig.text(0.05, 0.5, label, fontsize=size)
    with matplotlib.rc_context({"pdf.fonttype": 42, "ps.fonttype": 42}):
        fig.savefig(tmp_path / "fig-mini.pdf")
    fig.savefig(tmp_path / "fig-mini.png", dpi=150)
    plt.close(fig)
    return tmp_path / "fig-mini.pdf"


def test_symbol_scan_flags_forbidden_forms(tmp_path: Path):
    """反例：`·` 出现在图面 → check_pdf 判 FAIL 并报 S8。"""
    ok, problems = check_figures.check_pdf(_mini_fig(tmp_path, "2θ circular · conservative"))
    assert not ok
    assert any("S8" in p for p in problems), problems


def test_symbol_scan_allows_sanctioned_forms(tmp_path: Path):
    """正例：`&`/`|`/「数字+数字」/「 / 」独立记号不误报。"""
    ok, problems = check_figures.check_pdf(
        _mini_fig(tmp_path, "Counts & density | layers (25+22); records / cell; units (km/s)"))
    assert ok, problems


def test_symbol_scan_flags_nonnumeric_plus(tmp_path: Path):
    """反例：`+` 非数值求和（如 GeoParquet + FileGDB）→ check_pdf 判 FAIL 并报 S8。"""
    ok, problems = check_figures.check_pdf(
        _mini_fig(tmp_path, "Vector sidecars: GeoParquet + FileGDB"))
    assert not ok
    assert any("S8" in p for p in problems), problems


# ---------- 9. NSD 档（Scientific Data 可读性档） ----------
# 判据：页宽转观察项（首轮无版式要求），字号/字体内嵌/交付格式一致性三族硬判据保留。

def test_nsd_profile_drops_width_criterion():
    """页宽判据分档：ESSD 档违白名单即 FAIL；NSD 档同宽不判（可读性档）。"""
    assert check_figures.width_problems(16.0, "essd"), "ESSD 档 16cm 应判 FAIL"
    assert check_figures.width_problems(16.0, "nsd") == [], "NSD 档不判页宽"
    assert check_figures.width_problems(12.0, "essd") == []
    assert check_figures.width_problems(17.7, "nsd") == []


def test_nsd_profile_keeps_readability_criteria(tmp_path: Path):
    """可读性档仍守字号下限：7pt 图件在 NSD 档判 FAIL（S2）。"""
    ok, problems = check_figures.check_pdf(_mini_fig(tmp_path, "too small", size=7), journal="nsd")
    assert not ok
    assert any("最小字号" in p for p in problems), problems


def test_nsd_profile_catches_delivery_mismatch(defect_pdf: Path):
    """可读性档仍守交付格式一致性：真实缺陷版 fig07 在 NSD 档同样 FAIL。"""
    ok, problems = check_figures.check_pdf(defect_pdf, journal="nsd")
    assert not ok
    assert any("最大块差" in p for p in problems), problems


def test_nsd_profile_real_figure_passes():
    """正例：交付态图在 NSD 档 PASS（图 1 单件代表，避免全量重渲染）。"""
    ok, problems = check_figures.check_pdf(FIGS / "fig01-pipeline.pdf", journal="nsd")
    assert ok, problems
