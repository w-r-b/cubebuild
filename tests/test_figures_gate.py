"""Checks for the figure mechanical gate "delivery-format consistency".

Seam: the per-figure decision entry point `check_pdf()` of
`paper/scripts/check_figures.py` — following the existing seam, without anchoring the
script's internal call pattern or specific block numbers (brittle).

The counterexample is a **real fixture** (not a synthetic sample): the defective fig07 PDF
from the closing commit in git history
(matplotlib 3.11.0's PDF backend drops set elements with negative data coordinates, writing
only 226 of 886 hexagons), together with the direct-export PNG from that same run. If the
check is weakened or the threshold is raised until it loses discriminating power, this test
fails.
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
except ImportError:
    check_figures = None

pytestmark = pytest.mark.skipif(
    check_figures is None, reason="论文侧图件检查器不在本工作区（缺 paper/scripts/check_figures.py）")

FIGS = REPO / "paper" / "figures"
DEFECT_COMMIT = "be8f768"
DEFECT_FIG = "fig07-triple-scatter"

CURRENT_FIGS = sorted(p.stem for p in FIGS.glob("fig*.pdf"))


@pytest.fixture(scope="module")
def defect_pdf(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Extract the defective fig07 PDF and its same-run direct-export PNG from git history, materializing them as a pair of fixtures in one directory."""
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
    """Positive case: the five figures in delivery state (system after figure renumbering; including
    the figure 3–4 map-type bitmap profiles)
    all PASS, no false positives."""
    ok, problems = check_figures.check_pdf(FIGS / f"{name}.pdf")
    assert ok, problems


def test_defect_fixture_flags_delivery_mismatch(defect_pdf: Path):
    """Counterexample: the real defective fig07 must FAIL, and report the maximum block difference and its block location."""
    ok, problems = check_figures.check_pdf(defect_pdf)
    assert not ok
    assert len(problems) == 1, problems
    msg = problems[0]
    assert "最大块差" in msg and "块(" in msg, msg
    diff, _, _ = check_figures.ink_block_diff(defect_pdf, defect_pdf.with_suffix(".png"))
    assert diff > 0.3, f"缺陷量级 {diff:.3f} 过低，判据鉴别力不足"



def _mini_fig(tmp_path: Path, label: str, size: float = 8) -> Path:
    """A minimal compliant PDF+PNG pair that is 12cm wide, default 8pt, pure vector, with embedded TrueType (all other gates pass)."""
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
    """Counterexample: a `·` appearing on the figure surface → check_pdf flags FAIL and reports S8."""
    ok, problems = check_figures.check_pdf(_mini_fig(tmp_path, "2θ circular · conservative"))
    assert not ok
    assert any("S8" in p for p in problems), problems


def test_symbol_scan_allows_sanctioned_forms(tmp_path: Path):
    """Positive case: `&`/`|`/"digit+digit"/standalone " / " notation does not raise a false positive."""
    ok, problems = check_figures.check_pdf(
        _mini_fig(tmp_path, "Counts & density | layers (25+22); records / cell; units (km/s)"))
    assert ok, problems


def test_symbol_scan_flags_nonnumeric_plus(tmp_path: Path):
    """Counterexample: `+` as non-numeric summation (e.g. GeoParquet + FileGDB) → check_pdf flags FAIL and reports S8."""
    ok, problems = check_figures.check_pdf(
        _mini_fig(tmp_path, "Vector sidecars: GeoParquet + FileGDB"))
    assert not ok
    assert any("S8" in p for p in problems), problems



def test_nsd_profile_drops_width_criterion():
    """Page-width criterion is profile-split: the ESSD profile FAILs if it violates the whitelist; the NSD profile does not judge the same width (readability profile)."""
    assert check_figures.width_problems(16.0, "essd"), "ESSD 档 16cm 应判 FAIL"
    assert check_figures.width_problems(16.0, "nsd") == [], "NSD 档不判页宽"
    assert check_figures.width_problems(12.0, "essd") == []
    assert check_figures.width_problems(17.7, "nsd") == []


def test_nsd_profile_keeps_readability_criteria(tmp_path: Path):
    """The readability profile still enforces the font-size lower bound: a 7pt figure FAILs under the NSD profile (S2)."""
    ok, problems = check_figures.check_pdf(_mini_fig(tmp_path, "too small", size=7), journal="nsd")
    assert not ok
    assert any("最小字号" in p for p in problems), problems


def test_nsd_profile_catches_delivery_mismatch(defect_pdf: Path):
    """The readability profile still enforces delivery-format consistency: the real defective fig07 also FAILs under the NSD profile."""
    ok, problems = check_figures.check_pdf(defect_pdf, journal="nsd")
    assert not ok
    assert any("最大块差" in p for p in problems), problems


def test_nsd_profile_real_figure_passes():
    """Positive case: a delivery-state figure PASSes under the NSD profile (figure 1 as a single representative, avoiding a full re-render)."""
    ok, problems = check_figures.check_pdf(FIGS / "fig01-pipeline.pdf", journal="nsd")
    assert ok, problems
