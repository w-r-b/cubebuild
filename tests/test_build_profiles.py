"""Compile-side acceptance tests (compile side + ESSD regression).

Three groups:
(1) Revision-round .tex delivery self-checks (`check_tex_delivery.py`) — references embedded /
    no internal hyperlinks / single file with zero dependencies / entry count complete;
    negatives injected one by one;
(2) Structural baseline and comparison (`check_pdf_structure.py`) — page count/overfull/zero
    character loss/text-layer garbage/ink out of bounds
   ＋ caption sandwich/caption-body separation/PDF metadata title; negatives recomputed with
    a mutated baseline dict (no recompile needed);
(3) ESSD profile-frozen surface — after the journal profile is dispatched, the font and
    geometry constants of the two ESSD profiles must not drift (the static surface of the
    frozen-track regression; the dynamic surface = essd_regression.sh recompile + structural
    baseline comparison).
Additionally, SI PDF checks (`check_supplementary_pdf.py`).

Run: pytest tests/test_build_profiles.py (lightweight; reads PDF/tex only when artifacts exist).
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "paper" / "scripts"))

try:
    import pymupdf  # noqa: E402
    import check_pdf_structure as cps  # noqa: E402
    import check_supplementary_pdf as csp  # noqa: E402
    import check_tex_delivery as ctd  # noqa: E402
    import make_supplementary_tables as mst  # noqa: E402
    import nsd_tables as nt  # noqa: E402
except ImportError:
    pymupdf = cps = csp = ctd = mst = nt = None

pytestmark = pytest.mark.skipif(
    cps is None or pymupdf is None,
    reason="论文侧编译检查器不在本工作区（缺 paper/scripts/*.py 或 PDF 读件 pymupdf）")

BUILD = REPO / "paper" / "build"
NSD_TEX = BUILD / "SEDC-manuscript-nsd-en.tex"
NSD_MD = REPO / "paper" / "manuscript" / "SEDC-manuscript-nsd-en.md"
BUILD_SH = REPO / "paper" / "scripts" / "build_review_pdf.sh"


def _tex() -> str:
    if not NSD_TEX.exists():
        pytest.skip("NSD 交付件 .tex 未生成（跑 bash paper/scripts/build_review_pdf.sh nsd）")
    return NSD_TEX.read_text(encoding="utf-8")


def test_tex_delivery_clean():
    """Positive case: the NSD-profile delivery .tex passes all checks, and the embedded entry count matches the manuscript source (49)."""
    tex = _tex()
    findings = ctd.run_checks(tex, expected_entries=ctd.count_source_entries(NSD_MD))
    assert findings == [], findings


@pytest.mark.parametrize("inject,expect", [
    ("\\cref{fig:1}", "② 检出内部超链接宏"),
    ("\\ref{tab:1}", "② 检出内部超链接宏"),
    ("\\bibliography{refs}", "① 参考文献未内嵌"),
    ("\\input{extra.tex}", "③ 检出外部文件依赖"),
])
def test_tex_delivery_negatives(inject: str, expect: str):
    """Negative: internal hyperlinks/external .bib/external file dependency each injected once → caught."""
    tex = _tex()
    marker = "\\begin{document}"
    findings = ctd.run_checks(tex.replace(marker, inject + "\n" + marker, 1))
    assert any(f.startswith(expect) for f in findings), findings


def test_tex_delivery_dropped_entries_detected():
    """Negative: reference entries are pruned (10 \\item lines deleted) → "entries not accompanying the delivery" is caught."""
    tex = _tex()
    m = re.search(r"\\section\*?\{References\}", tex)
    head, tail = tex[:m.end()], tex[m.end():]
    items = list(re.finditer(r"\\item\b", tail))
    assert len(items) > 20
    cut = tail[: items[10].start()] + tail[items[10].start():].replace("\\item", "\\i tem", 20)
    findings = ctd.run_checks(head + cut, expected_entries=49)
    assert any("条目数" in f for f in findings), findings


def test_pdf_structure_matches_baseline():
    """Structural baseline comparison: existing artifacts (both ESSD profiles + NSD profile) each PASS (missing artifacts skipped)."""
    baseline = json.loads(cps.DEFAULT_BASELINE.read_text(encoding="utf-8"))
    checked = 0
    for name, pdf in cps.TARGETS.items():
        if not pdf.exists():
            continue
        log = pdf.parent / f"_manuscript-{cps.LOG_SUFFIX[name]}.log"
        got = cps.measure(pdf, log)
        findings = cps.compare(baseline[name], got)
        assert findings == [], f"[{name}] {findings}"
        checked += 1
    assert checked >= 1, "无编译产物可比（先跑 ESSD 回归/NSD 编译）"


def test_structure_baseline_detects_drift():
    """Negative: structural-fingerprint drift (page-count drop, lost characters, text-layer
    garbage, ink out of bounds, sandwich, separation, metadata title)
    is caught item by item by the comparison."""
    base = {"pages": 27, "empty_pages": 0, "ink_edge_pages": 0,
            "garbled": {"\\protect": 0}, "overfull": 0, "missing_chars": 0,
            "caption_head_duplicates": 0, "caption_body_separation": 0, "pdf_title": ""}
    good = dict(base)
    assert cps.compare(base, good) == []
    for key, value, expect in (("pages", 25, "页数"), ("missing_chars", 3, "missing_chars"),
                               ("overfull", 2, "overfull"), ("ink_edge_pages", 1, "ink_edge_pages"),
                               ("caption_head_duplicates", 2, "caption_head_duplicates"),
                               ("caption_body_separation", 3, "caption_body_separation"),
                               ("pdf_title", "Some Title", "pdf_title")):
        bad = dict(base)
        bad[key] = value
        findings = cps.compare(base, bad)
        assert any(expect in f for f in findings), (key, findings)
    garbled = dict(base)
    garbled["garbled"] = {"\\protect": 5}
    assert any("垃圾串" in f for f in cps.compare(base, garbled))


def test_essd_profile_constants_frozen():
    """ESSD profile-frozen surface: after the journal profile is dispatched, the font and geometry constants of zh/en must not drift."""
    sh = BUILD_SH.read_text(encoding="utf-8")
    assert 'zh) MAIN_FONT="DejaVu Sans"; MONO_FONT="DejaVu Sans Mono"; CAPTIONSPACE=10;' in sh, \
        "zh 档字体/题注守卫被改（守卫值 10＝按档实测保留：zh 守卫下 0/0、去守卫致题注跨页 3 处；变更须显式重录）"
    assert 'en) MAIN_FONT="Liberation Sans"; MONO_FONT="Liberation Sans"; CAPTIONSPACE=0;' in sh, \
        "en 档字体/题注守卫被改（守卫值 0＝按档实测不注入：守卫致夹层 3 处；变更须显式重录）"
    assert '-V geometry:"paperwidth=216mm, paperheight=246mm, textwidth=177mm, hcentering, ' \
           'top=10mm, bottom=29mm"' in sh, "版式几何被改（Copernicus 投稿稿几何）"
    assert 'nsd) MAIN_FONT=""; MONO_FONT=""; CAPTIONSPACE=0' in sh, \
        "NSD 档字体/题注守卫被改（字体须走 fontspec 默认＝Computer Modern；" \
        "守卫值 0 ＝不注入（实测零漂移复核），变更须显式重录）"


def test_caption_guard_is_gated_by_profile():
    """Regression sentinel (determined by per-profile measurement): the caption guard must be
    **gated by profile** (value semantics in the build script's case-block comment) — removing
    the gate and injecting the guard unconditionally is caught by this assertion; each profile's
    value (zh=10, en/nsd=0) is pinned verbatim in test_essd_profile_constants_frozen.
    The delivery .tex side separately checks for zero \\needspace (see below)."""
    sh = BUILD_SH.read_text(encoding="utf-8")
    assert 'if [ "$CAPTIONSPACE" -gt 0 ]; then' in sh, \
        "题注守卫失去按档门控（en 档将重新出现夹层）"


def test_nsd_delivery_tex_has_no_caption_guard():
    """Same check on the delivery side: the NSD-profile .tex has zero \\needspace (guard not injected) — artifact-side evidence of "putting the guard back"."""
    tex = _tex()
    assert "\\needspace" not in tex, "NSD 交付件 .tex 出现题注守卫（夹层将复发）"


def test_nsd_delivery_tex_has_header_split():
    """Delivery-side positive check: the long table's "first-page head/continuation-page
    head" split must actually land in the delivery — `\\endfirsthead` and `\\endhead` appear
    exactly once per table (the split has zero effect on external
    rendering; without this assertion a regex mismatch would pass silently)."""
    tex = _tex()
    n_tables = tex.count("\\begin{longtable}")
    assert n_tables >= 1
    assert tex.count("\\endfirsthead") == n_tables, "二分缺失/不完整（\\endfirsthead 计数≠表数）"
    assert tex.count("\\endhead") == n_tables, "续页表头缺失/不完整（\\endhead 计数≠表数）"


def _synthetic_pdf(path: Path, duplicated_head: bool) -> Path:
    """Synthetic single-page PDF: optionally reproduces the "header→caption→header→body" caption-sandwich form."""
    import pymupdf
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    if duplicated_head:
        page.insert_text((72, 80), "No. Dataset Domain group DOI")
    page.insert_text((72, 130), "Table 1. A caption block for the detector test.")
    page.insert_text((72, 180), "No. Dataset Domain group DOI")
    page.insert_text((72, 210), "1 ETOPO 2022 ...")
    doc.save(path)
    doc.close()
    return path


def test_caption_sandwich_detector(tmp_path: Path):
    """Check discriminating power: the sandwich form scores 1, the clean form scores 0 (synthetic artifacts, not depending on a real PDF)."""
    bad = cps.caption_head_duplicates(pymupdf.open(_synthetic_pdf(tmp_path / "bad.pdf", True)))
    good = cps.caption_head_duplicates(pymupdf.open(_synthetic_pdf(tmp_path / "good.pdf", False)))
    assert bad == 1, bad
    assert good == 0, good


def test_essd_en_caption_sandwich_pinned_to_zero():
    """Frozen ESSD English manuscript caption sandwich pinned at **0** (guard removal fixed 3
    registered cases (tables 2/6/7) → 0; before the fix this artifact was the mechanical
    evidence for "the defect class predates the migration").
    If a sandwich reappears in regression, this assertion failing signals "the en-profile guard
    was put back" (per-profile guard switching in the build script's case block).
    The `essd-zh` profile is checked separately via the guard pin values and baseline comparison
    (the guard is retained as a measured decision; see the build script's case block)."""
    pdf, log = cps.TARGETS["essd-en"], BUILD / "_manuscript-en.log"
    if not pdf.exists():
        pytest.skip("ESSD 英文稿编译件不存在")
    got = cps.measure(pdf, log)
    assert got["caption_head_duplicates"] == 0, got


def test_nsd_caption_sandwich_pinned_to_zero():
    """NSD-profile sandwich count pinned at **0** (not written as "equals the baseline" — so it
    can be falsified;
    the benefit side of D1). If a sandwich reappears in regression, this assertion failing
    signals "the guard was put back" or layout degradation."""
    pdf, log = cps.TARGETS["nsd-en"], BUILD / "_manuscript-nsd.log"
    if not pdf.exists():
        pytest.skip("NSD 编译件不存在")
    got = cps.measure(pdf, log)
    assert got["caption_head_duplicates"] == 0, got


def test_nsd_caption_separation_matches_recorded_state():
    """The NSD-profile caption-body separation count equals the re-recorded registered value (the
    **cost** side of D1; not removing the guard makes
    the separation zero but revives the sandwich — both are registered together, single-sided
    optimization is forbidden)."""
    pdf, log = cps.TARGETS["nsd-en"], BUILD / "_manuscript-nsd.log"
    if not pdf.exists():
        pytest.skip("NSD 编译件不存在")
    baseline = json.loads(cps.DEFAULT_BASELINE.read_text(encoding="utf-8"))
    want = baseline["nsd-en"]["caption_body_separation"]
    got = cps.measure(pdf, log)
    assert got["caption_body_separation"] == want, (got, want)


def _synthetic_separation_pdf(path: Path, with_line_number: bool) -> Path:
    """Synthetic single-page PDF: if only the lineno line-number block remains below the caption
    on this page (with_line_number) = table-body-next-page form;
    otherwise the caption is followed by a header row = table-caption-same-page form."""
    import pymupdf
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 80), "Table 1. A source dataset inventory caption block.")
    if with_line_number:
        page.insert_text((40, 700), "14")
    else:
        page.insert_text((72, 130), "No. Dataset Domain group DOI")
    doc.save(path)
    doc.close()
    return path


def test_caption_separation_detector(tmp_path: Path):
    """Check discriminating power: a caption followed by a header row (not separated) scores 0; a
    caption with only the lineno block below (true separation) scores 1
    — if the line-number block is not filtered out, this form is misjudged as "not separated",
    and the separation count would forever be 0 (the check fails)."""
    clean = cps.caption_body_separation(
        pymupdf.open(_synthetic_separation_pdf(tmp_path / "clean.pdf", False)))
    sepa = cps.caption_body_separation(
        pymupdf.open(_synthetic_separation_pdf(tmp_path / "sep.pdf", True)))
    assert clean == 0, clean
    assert sepa == 1, sepa


def test_pdf_metadata_title_matches_manuscript():
    """D6 check: the NSD compiled PDF metadata title = the NSD manuscript H1 (title line)."""
    pdf, log = cps.TARGETS["nsd-en"], BUILD / "_manuscript-nsd.log"
    if not pdf.exists():
        pytest.skip("NSD 编译件不存在")
    title = NSD_MD.read_text(encoding="utf-8").split("\n", 1)[0].lstrip("# ").strip()
    findings = cps.metadata_findings(cps.measure(pdf, log), title)
    assert findings == [], findings


def test_pdf_metadata_missing_is_caught(tmp_path: Path):
    """Negative: the PDF has no title metadata (or the title drifts) → "metadata title ≠ manuscript source title" is caught."""
    import pymupdf
    doc = pymupdf.open()
    doc.new_page(width=595, height=842).insert_text((72, 80), "Table 1. A caption.")
    bare = tmp_path / "bare.pdf"
    doc.save(bare)
    doc.close()
    measured = cps.measure(bare)
    assert cps.metadata_findings(measured, "A global multi-resolution data cube") != []
    with_title = {"pdf_title": "A global multi-resolution data cube"}
    assert cps.metadata_findings(with_title, "A global multi-resolution data cube") == []


def _si_pages() -> list[str]:
    if not mst.SI_PDF.exists():
        pytest.skip("SI PDF 未生成（跑 python paper/scripts/make_supplementary_tables.py）")
    return csp.pdf_pages(mst.SI_PDF)


def test_si_pdf_content_and_captions_match_sources():
    """Positive case: each field of the SI PDF is co-sourced with the csv, captions are co-sourced with submission requirements §4, the main text has no S-table captions, and each table starts on its own page."""
    findings = csp.run_checks(mst.SI_PDF, nt.SUPPLEMENTARY_DIR, mst.supplementary_captions(),
                              NSD_MD.read_text(encoding="utf-8"))
    assert findings == [], findings


def test_si_pdf_mutated_cell_detected(tmp_path: Path):
    """Negative: one csv cell tampered (1.0000→1.0001) → "field absent from the SI PDF" is caught (carrier drift)."""
    pages = _si_pages()
    src = nt.SUPPLEMENTARY_DIR / "TableS2-layer-coverage.csv"
    assert "1.0000" in src.read_text(encoding="utf-8")
    shutil.copytree(nt.SUPPLEMENTARY_DIR, tmp_path / "csv")
    target = tmp_path / "csv" / src.name
    target.write_text(src.read_text(encoding="utf-8").replace("1.0000", "1.0001", 1),
                      encoding="utf-8")
    findings = csp.content_findings("\n".join(pages), tmp_path / "csv")
    assert any("字段未见于 SI PDF" in f for f in findings), findings


def test_si_pdf_caption_drift_detected():
    """Negative: a caption word changed (submission requirements changed but the SI PDF was not regenerated) → caption inconsistency caught."""
    pages = _si_pages()
    drifted = dict(mst.supplementary_captions())
    drifted[1] = drifted[1].replace("Layer registry", "Layer registery", 1)
    findings = csp.caption_findings("\n".join(pages), drifted)
    assert any("题注文本" in f for f in findings), findings


def test_si_pdf_main_md_red_line_detected():
    """Negative: an S-table caption appears in the main text (a journal-spec prohibited item) → caught."""
    md = NSD_MD.read_text(encoding="utf-8")
    assert csp.main_md_findings(md) == []
    assert csp.main_md_findings("**Supplementary Table 1.** Layer registry …") != []


def test_si_pdf_layout_start_of_page():
    """Check discriminating power: two tables' captions on the same page → "not each starting on its own page" caught; each starting on its own page → pass."""
    caps = {1: "C1", 2: "C2"}
    assert csp.layout_findings(["Supplementary Table 1. C1\nSupplementary Table 2. C2"], caps)
    assert csp.layout_findings(["Supplementary Table 1. C1\nrow", "Supplementary Table 2. C2"],
                               caps) == []


def test_si_pdf_generator_records_single_caption_source():
    """Single caption source: both the generating side and the checking side read the submission
    requirements §4 table (writing one copy in each place causes drift); moreover, captions must
    be ready-to-submit English (Chinese source annotations belong in column 4 of this table and
    are not submitted with the delivery)."""
    caps = mst.supplementary_captions()
    assert sorted(caps) == [e.s_number for e in nt.active_supp_entries()]
    cjk = re.compile(r"[\u3000-\u303f\u3400-\u9fff\uff00-\uffef]")
    assert all(len(c) > 30 and not cjk.search(c) for c in caps.values()), \
        f"题注含中文（应移入内部注记列）：{[c for c in caps.values() if cjk.search(c)]}"


def test_nsd_pdf_metadata_author_stays_empty():
    """D5/D6: the PDF metadata author field must be left empty (the author block enters the
    manuscript as body text, without writing the PDF /Author);
    once the field appears it is caught — the "no author in metadata" convention is raised from
    eyeball inspection to a mechanical check."""
    pdf, log = cps.TARGETS["nsd-en"], BUILD / "_manuscript-nsd.log"
    if not pdf.exists():
        pytest.skip("NSD 编译件不存在")
    measured = cps.measure(pdf, log)
    assert measured["pdf_author"] == "", measured
    assert cps.metadata_findings(measured, "A global multi-resolution data cube") != []
    assert cps.metadata_findings({"pdf_title": "X", "pdf_author": "A. Author"}, "X") == \
        ["PDF 元数据作者字段非空：'A. Author'（署名块应走正文、不设 PDF 元数据）"]


def test_si_pdf_font_program_magic_is_truetype():
    """(5) Font-structure check: the font program magic of the fonts used by the SI PDF must be
    TrueType (glyf).
    The negative is expressed via the classifier's discriminating power (CFF/OTF would be
    written as a non-compliant FontFile2 containing CFF, and the glyphs would not render at
    all)."""
    pages = _si_pages()
    assert pages
    assert csp.font_findings(mst.SI_PDF) == [], csp.font_findings(mst.SI_PDF)
    assert csp.font_program_kind(b"OTTO") == "cff"
    assert csp.font_program_kind(b"\x01\x00\x04\x00") == "cff"
    assert csp.font_program_kind(b"\x00\x01\x00\x00") == "truetype"
    assert csp.font_program_kind(b"true") == "truetype"
