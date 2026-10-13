"""Checks for the en↔en fidelity checker.

Fixture strategy:
(1) current pairing (ESSD English manuscript ↔ NSD manuscript + Supplementary artifacts)
    zero difference across the five object classes (regression baseline);
(2) non-empty object self-check (guards against a false-green empty run: a failing extractor
    would let a set-level diff pass silently);
(3) one negative per the five real-base classes (number / layer id / DOI·URL / citation key /
    figure-caption data value) — the injected forms are exactly the defects listed as
    "missing number, layer id tampering, citation key loss, DOI change";
(4) whitelist has no dead entries: recompute after deleting each one, every registered entry
    must be real (guards against exemption-surface rot);
(5) a missing Supplementary carrier is immediately caught (appendix long-table content must
    not vanish from both ends).

Run: pytest tests/test_fidelity_nsd.py (lightweight, reads md and csv only).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "paper" / "scripts"))

try:
    import check_fidelity_nsd as cf  # noqa: E402
except ImportError:
    cf = None

pytestmark = pytest.mark.skipif(
    cf is None, reason="论文侧保真检查器不在本工作区（缺 paper/scripts/check_fidelity_nsd.py）")

BASELINE = REPO / "paper" / "manuscript" / "SEDC-manuscript-en.md"
NSD = REPO / "paper" / "manuscript" / "SEDC-manuscript-nsd-en.md"


@pytest.fixture(scope="module")
def pair() -> tuple[str, str]:
    return BASELINE.read_text(encoding="utf-8"), NSD.read_text(encoding="utf-8")


def test_current_pair_clean(pair):
    """Current pairing has zero differences (structural differences are explicitly registered in STRUCTURAL_EXCEPTIONS)."""
    findings = cf.run_checks(*pair)
    assert findings == [], findings


def test_objects_nonempty(pair):
    """The five object classes are non-empty (guards against a false-green empty run)."""
    essd, nsd = pair
    supp = cf.supplementary_text()
    assert len(cf._side_numbers(essd, False)) > 400
    assert len(cf._side_numbers(nsd, True) | cf.ct.numbers(supp)) > 400
    assert len(cf.ct.layer_ids(essd)) >= 47
    assert cf.ct.layer_ids(nsd) | cf.ct.layer_ids(supp) >= cf.ct.layer_ids(essd)
    assert len(cf.ct.dois_urls(essd)) > 100
    assert {"Figure 1", "Figure 2", "Figure 3", "Figure 4", "Figure 5"} <= set(cf.caption_numbers(essd, False))


def test_mutated_number_detected(pair):
    """Number drift: NSD manuscript Table 2 pixel count 64,800→64,801 → caught."""
    essd, nsd = pair
    assert "| Pixels | 64,800 |" in nsd
    findings = cf.run_checks(essd, nsd.replace("| Pixels | 64,800 |", "| Pixels | 64,801 |", 1))
    assert any(f.startswith("[数字 token]") and "64801" in f for f in findings), findings


def test_mutated_layer_id_detected(pair):
    """Layer id tampering: an NSD manuscript layer id truncated → caught."""
    essd, nsd = pair
    assert "topography__bedrock_elevation" in nsd
    findings = cf.run_checks(essd, nsd.replace("topography__bedrock_elevation",
                                               "topography__bedrock_elev", 1))
    assert any(f.startswith("[层 id]") and "topography__bedrock_elev" in f for f in findings), findings


def test_mutated_doi_detected(pair):
    """DOI change: the last digit of an NSD reference DOI is tampered → caught."""
    essd, nsd = pair
    assert "10.25921/fd45-gt74" in nsd
    findings = cf.run_checks(essd, nsd.replace("10.25921/fd45-gt74", "10.25921/fd45-gt75", 1))
    assert any(f.startswith("[DOI/URL]") and "fd45-gt75" in f for f in findings), findings


def test_mutated_citation_key_detected(pair):
    """Citation key loss: deleting one NSD reference entry → caught (citation-key set + per-entry comparison)."""
    essd, nsd = pair
    assert "46. Neumann, F.," in nsd
    findings = cf.run_checks(essd, nsd.replace("46. Neumann, F.,", "46. Xx, F.,", 1))
    assert any(f.startswith("[引文键]") or f.startswith("[条目]") for f in findings), findings


def test_mutated_figure_caption_value_detected(pair):
    """Figure-caption data-value drift: figure 3 caption −10776→−10777 → caught."""
    essd, nsd = pair
    assert "−10776 to 8354 m" in nsd
    findings = cf.run_checks(essd, nsd.replace("−10776 to 8354 m", "−10777 to 8354 m", 1))
    assert any(f.startswith("[图注 Figure 3 数据值]") for f in findings), findings


def test_author_date_leftover_detected(pair):
    """Mechanical surface of the numeric citation style: an author-date citation left in the NSD body → caught."""
    essd, nsd = pair
    findings = cf.run_checks(essd, nsd.replace("The quality evidence is organized",
                                               "(Reichstein et al., 2019) The quality evidence is organized", 1))
    assert any(f.startswith("[编号体例]") for f in findings), findings


def test_whitelist_entries_all_needed(pair, monkeypatch):
    """The whitelist has no dead entries: recompute after deleting each one, every registered entry must be real."""
    essd, nsd = pair
    dead: list[tuple[str, str, str, str]] = []
    for entry in cf.STRUCTURAL_EXCEPTIONS:
        monkeypatch.setattr(cf, "STRUCTURAL_EXCEPTIONS",
                            [e for e in cf.STRUCTURAL_EXCEPTIONS if e != entry])
        findings = cf.run_checks(essd, nsd)
        if not any(f"[{entry[0]}]" in f and entry[2] in f for f in findings):
            dead.append(entry)
    assert dead == [], f"白名单死条目（豁免不再被需要）：{dead}"


def test_supplementary_carrier_missing_detected(pair, monkeypatch, tmp_path):
    """Supplementary carrier absent: appendix long-table content is missing from both ends on the NSD side → caught."""
    essd, nsd = pair
    monkeypatch.setattr(cf, "SUPPLEMENTARY_DIR", tmp_path)
    findings = cf.run_checks(essd, nsd)
    assert any(f.startswith("[层 id]") for f in findings), findings