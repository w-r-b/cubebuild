"""NSD numbered-reference conversion and triple verification tests.

Criteria (reference conversion: mechanical conversion (key→number mapping) +
triple verification):
① reference-key set unchanged relative to the baseline; ② every DOI unchanged;
③ numbering = first-occurrence order (body superscript first-occurrence order
   1..N, entry number prefixes 1..N, no orphans). Each negative case must be
   caught — the injected forms "missing number / lost reference key / altered
   DOI / out-of-order numbering".

Also includes the ESSD frozen track's **byte freeze** criterion: relative to
frozen commit 536546a, the ESSD English and Chinese manuscripts allow only the
registered named-exception substitutions (fact-check round errata), and are
otherwise byte-for-byte unchanged (migration derives solely from the ESSD
English manuscript; the frozen track must not be rewritten in passing).

Run: pytest tests/test_nsd_references.py (lightweight; reads md and one git
show).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "paper" / "scripts"))

try:
    import nsd_citations as nc  # noqa: E402
except ImportError:
    nc = None

pytestmark = pytest.mark.skipif(
    nc is None, reason="论文侧引用转换检查器不在本工作区（缺 paper/scripts/nsd_citations.py）")

BASELINE = REPO / "paper" / "manuscript" / "SEDC-manuscript-en.md"
NSD = REPO / "paper" / "manuscript" / "SEDC-manuscript-nsd-en.md"
FROZEN_COMMIT = "536546a"


@pytest.fixture(scope="module")
def pair() -> tuple[str, str]:
    return BASELINE.read_text(encoding="utf-8"), NSD.read_text(encoding="utf-8")


def test_triple_verification_clean(pair):
    """Triple verification with zero findings (reference-key set / per-entry
    DOI / numbering = first-occurrence order)."""
    essd, nsd = pair
    findings = nc.verify(nsd, essd)
    assert findings == [], findings


def test_numbering_is_first_occurrence(pair):
    """Numbering = first-occurrence order: both the body superscript
    first-occurrence sequence and the entry number prefixes are 1..N, and
    first-occurrence order is the assignment order."""
    _, nsd = pair
    body, refs = nc.split_sections(nsd, "# References")
    order = [int(x) for m in nc.SUP_RE.finditer(body) for x in m.group(1).split(",")]
    uniq = list(dict.fromkeys(order))
    assert uniq == list(range(1, len(uniq) + 1)), uniq[:12]
    prefixes = [int(m.group(1)) for m in nc.ENTRY_RE.finditer(refs)]
    assert prefixes == list(range(1, len(prefixes) + 1))
    assert len(uniq) == len(prefixes), (len(uniq), len(prefixes))


def test_entry_counts_match_baseline(pair):
    """Entry count unchanged: NSD numbered entry count = ESSD baseline entry
    count ('99 entries' was a typo; actual 49)."""
    essd, nsd = pair
    base = nc.parse_entries(nc.split_sections(essd, "# References")[1], numbered=False)
    got = nc.parse_entries(nc.split_sections(nsd, "# References")[1], numbered=True)
    assert len(base) == len(got) == 49, (len(base), len(got))


def test_mutated_doi_detected(pair):
    """② every DOI unchanged: tampering the last digit of an NSD entry DOI →
    detected."""
    essd, nsd = pair
    assert "10.1093/gji/ggae270" in nsd
    findings = nc.verify(nsd.replace("10.1093/gji/ggae270", "10.1093/gji/ggae271"), essd)
    assert any(f.startswith("[②DOI]") for f in findings), findings


def test_dropped_entry_detected(pair):
    """① reference-key set unchanged: deleting one NSD entry → detected."""
    essd, nsd = pair
    line = [l for l in nsd.split("\n") if l.startswith("24. Bird, P.")]
    assert len(line) == 1
    findings = nc.verify(nsd.replace(line[0] + "\n", "", 1), essd)
    assert any(f.startswith("[①引用键]") for f in findings), findings


def test_extra_entry_detected(pair):
    """① reverse: NSD gains an entry the baseline lacks → detected (padding /
    numbering insertion surface)."""
    essd, nsd = pair
    findings = nc.verify(nsd.replace("\n# Funding", "\n50. Zzfake, Z. Padding. *J. Test* **1**, 1 (2020)."
                                     " https://doi.org/10.1/fake\n\n# Funding", 1), essd)
    assert any(f.startswith("[①引用键]") or f.startswith("[③编号]") for f in findings), findings


def test_out_of_order_numbering_detected(pair):
    """③ numbering = first-occurrence order: swapping the two first-occurring
    body numbers → detected (out-of-order numbering)."""
    essd, nsd = pair
    assert "^1,2^" in nsd
    findings = nc.verify(nsd.replace("^1,2^", "^2,1^", 1), essd)
    assert any(f.startswith("[③编号]") for f in findings), findings


def test_broken_entry_prefix_detected(pair):
    """③ entry number prefix: changing the first prefix to 2 → detected (missing
    / out-of-order number)."""
    essd, nsd = pair
    assert nsd.count("\n1. Reichstein") == 1
    findings = nc.verify(nsd.replace("\n1. Reichstein", "\n2. Reichstein", 1), essd)
    assert any(f.startswith("[③编号]") for f in findings), findings


def test_orphan_entry_detected(pair):
    """③ no orphans: body citation count ≠ entry count (deleting one body
    citation) → detected."""
    essd, nsd = pair
    findings = nc.verify(nsd.replace("^49^", "", 1), essd)
    assert any(f.startswith("[③编号]") for f in findings), findings


FROZEN_EXCEPTIONS: dict[str, list[tuple[str, str]]] = {
    "SEDC-manuscript-en.md": [
        ("eta is ξ, the radial anisotropy parameter, dimensionless)",
         "eta is the dimensionless anisotropy parameter η)"),
        ("932,509 (including 20,958 debris flows)",
         "932,509 (including 20,958 pyroclastic features)"),
        ("the cubebuild test suite; sha256 checksums of 260 arrays across multiple full builds"
         " | 0 failures, 0 skips; identical checksums | 378 tests passed |",
         "the construction pipeline test suite; sha256 checksums of 260 arrays across multiple full builds"
         " | 0 failures, 0 skips; identical checksums | 357 tests passed |"),
    ],
    "SEDC-manuscript-zh.md": [
        ("eta 即 ξ，径向各向异性参数，无量纲", "eta 为无量纲的各向异性参数 η"),
        ("932,509（含碎屑流 20,958）", "932,509（含火山碎屑要素 20,958）"),
        ("cubebuild 测试套件", "构建管线测试套件"),
        ("378 项通过", "357 项通过"),
    ],
}


def test_essd_baseline_frozen():
    """ESSD frozen-track byte freeze: after applying the registered named
    exceptions, byte-for-byte equal relative to 536546a; any other change
    fails."""
    for name, subs in FROZEN_EXCEPTIONS.items():
        blob = subprocess.run(["git", "show", f"{FROZEN_COMMIT}:paper/manuscript/{name}"],
                              cwd=REPO, capture_output=True, check=True).stdout.decode("utf-8")
        patched = blob
        for old, new in subs:
            hits = blob.count(old)
            assert hits == 1, f"{name}: 具名例外在基线中命中 {hits} 次（须恰 1 次）——{old!r}"
            patched = patched.replace(old, new)
        cur = (REPO / "paper" / "manuscript" / name).read_bytes()
        assert patched.encode("utf-8") == cur, (
            f"{name} 相对冻结基线 {FROZEN_COMMIT} 存在登记例外之外的改动（冻结轨不得改写）")