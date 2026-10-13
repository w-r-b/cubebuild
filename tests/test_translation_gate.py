"""Tests for the English-translation fidelity criteria of check_translation.py.

Fixture strategy:
① the current Chinese-English pair (Chinese merged manuscript SSOT ↔ English manuscript SSOT)
   has zero differences across the five checks (regression baseline);
② one real-base mutation negative case per each of the five classes (numbers / citation keys /
   layer id / DOI·URL / references section) must be intercepted — the injected form is exactly
   the defect form listed under "new criteria must come with a negative-case regression"
   (missing number or citation, tampered layer id).

Run: pytest tests/test_translation_gate.py (lightweight; reads only the two manuscript mds).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "paper" / "scripts"))

try:
    import check_translation as ct  # noqa: E402
except ImportError:
    ct = None

pytestmark = pytest.mark.skipif(
    ct is None, reason="论文侧译校检查器不在本工作区（缺 paper/scripts/check_translation.py）")

ZH_MD = REPO / "paper" / "manuscript" / "SEDC-manuscript-zh.md"
EN_MD = REPO / "paper" / "manuscript" / "SEDC-manuscript-en.md"


def _pair() -> tuple[str, str]:
    return ZH_MD.read_text(encoding="utf-8"), EN_MD.read_text(encoding="utf-8")



def test_current_pair_clean():
    zh, en = _pair()
    findings = ct.run_checks(zh, en)
    assert findings == [], findings


def test_objects_nonempty():
    """All five object classes are non-empty (guards against a vacuous pass: a broken extractor would make the baseline test falsely green)."""
    zh, en = _pair()
    assert len(ct.numbers(zh)) > 400 and len(ct.numbers(en)) > 400
    assert len(ct.layer_ids(zh)) >= 47 and ct.layer_ids(zh) == ct.layer_ids(en)
    assert len(ct.dois_urls(zh)) > 100 and len(ct.dois_urls(en)) > 100



def test_mutated_number_detected():
    """Number distortion: the English manuscript's Table 2 pixel count 64,800→64,801 → flagged (the same value survives elsewhere, so only the new English-side value is seen)."""
    zh, en = _pair()
    assert "| Pixels | 64,800 |" in en
    findings = ct.run_checks(zh, en.replace("| Pixels | 64,800 |", "| Pixels | 64,801 |", 1))
    assert any(f.startswith("[数字 token]") and "64801" in f for f in findings), findings


def test_mutated_citation_detected():
    """Citation missing/distorted: the English manuscript's year 2026→2027 → flagged."""
    zh, en = _pair()
    assert "(Mancino et al., 2026)" in en
    findings = ct.run_checks(zh, en.replace("(Mancino et al., 2026)", "(Mancino et al., 2027)", 1))
    assert any(f.startswith("[引用键]") and "Mancino" in f for f in findings), findings


def test_mutated_layer_id_detected():
    """Layer id tampering: truncating a layer id in the English manuscript → flagged."""
    zh, en = _pair()
    assert "topography__bedrock_elevation" in en
    findings = ct.run_checks(zh, en.replace("topography__bedrock_elevation",
                                            "topography__bedrock_elev", 1))
    assert any(f.startswith("[层 id]") and "topography__bedrock_elev" in f for f in findings), findings


def test_mutated_doi_detected():
    """DOI distortion: the last digit of the English manuscript's data DOI is tampered with (the same DOI appears once in Table 1 and once in Appendix B; changing one shows the new English-side value)."""
    zh, en = _pair()
    assert "10.25921/fd45-gt74" in en
    findings = ct.run_checks(zh, en.replace("10.25921/fd45-gt74", "10.25921/fd45-gt75", 1))
    assert any(f.startswith("[DOI/URL]") and "fd45-gt75" in f for f in findings), findings


def test_mutated_reference_entry_detected():
    """References-section distortion (an item on the "do not translate" list): volume/page-number tampering → flagged."""
    zh, en = _pair()
    assert "Geochem. Geophys. Geosyst., 4, 1027" in en
    findings = ct.run_checks(zh, en.replace("Geochem. Geophys. Geosyst., 4, 1027",
                                            "Geochem. Geophys. Geosyst., 4, 1028", 1))
    assert any(f.startswith("[参考文献节]") for f in findings), findings


def test_missing_refs_section_detected():
    """Structural absence: the English manuscript has no References section → flagged (structural level, no silent pass)."""
    zh, en = _pair()
    findings = ct.run_checks(zh, en.replace("# References", "# Bibliography", 1))
    assert any(f.startswith("[参考文献节]") and "未同时找到" in f for f in findings), findings