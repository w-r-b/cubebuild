"""编译侧判据测试（编译侧 + ESSD 回归）。

三组：
① 修订轮 .tex 交付件自检（`check_tex_delivery.py`）——参考文献内嵌 / 无内部超链接 /
   单文件零依赖 / 条目数齐备；负例逐条注入；
② 结构基线与对照（`check_pdf_structure.py`）——页数/溢出/字符零丢失/文本层垃圾/墨迹越界
   ＋题注夹层/题注-表体分离/PDF 元数据标题；负例用变异基线 dict 复算（不需要重编译）；
③ ESSD 档冻结面——期刊档派发后 ESSD 两档的字体与几何常量不得漂移（冻结轨回归的
   静态面；动态面＝essd_regression.sh 重编译＋结构基线对照）。
另加 SI PDF 判据（`check_supplementary_pdf.py`）。

运行：pytest tests/test_build_profiles.py（轻量；仅当产物存在时读 PDF/tex）。
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
except ImportError:  # 公开仓不含论文侧脚本与产物：缺件即跳过（沿用缺件跳过惯例）
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
    """正例：NSD 档交付件 .tex 过全部判据，且内嵌条目数与稿源一致（49）。"""
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
    """负例：内部超链接/外部 .bib/外部文件依赖各注入一处 → 命中。"""
    tex = _tex()
    marker = "\\begin{document}"
    findings = ctd.run_checks(tex.replace(marker, inject + "\n" + marker, 1))
    assert any(f.startswith(expect) for f in findings), findings


def test_tex_delivery_dropped_entries_detected():
    """负例：参考文献条目被裁（删掉 10 条 \\item）→ 命中「条目未随件」。"""
    tex = _tex()
    m = re.search(r"\\section\*?\{References\}", tex)
    head, tail = tex[:m.end()], tex[m.end():]
    items = list(re.finditer(r"\\item\b", tail))
    assert len(items) > 20
    cut = tail[: items[10].start()] + tail[items[10].start():].replace("\\item", "\\i tem", 20)
    findings = ctd.run_checks(head + cut, expected_entries=49)
    assert any("条目数" in f for f in findings), findings


def test_pdf_structure_matches_baseline():
    """结构基线对照：现存产物（ESSD 两档 + NSD 档）逐件 PASS（缺件跳过）。"""
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
    """负例：结构指纹漂移（页数掉、丢字、文本层垃圾、墨迹越界、夹层、分离、元数据标题）
    逐项被对照捕获。"""
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
    """ESSD 档冻结面：期刊档派发后 zh/en 的字体与几何常量不得漂移。"""
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
    """回归哨兵（按档实测确定）：题注守卫必须
    **按档门控**（取值语义见 build 脚本 case 块注释）——去掉门控使守卫无条件注入即被本断言
    拦下；各档取值（zh=10、en/nsd=0）逐字钉在 test_essd_profile_constants_frozen。
    交付件 .tex 侧另核零 \\needspace（见下）。"""
    sh = BUILD_SH.read_text(encoding="utf-8")
    assert 'if [ "$CAPTIONSPACE" -gt 0 ]; then' in sh, \
        "题注守卫失去按档门控（en 档将重新出现夹层）"
    # 各档取值（zh=10、en/nsd=0）由 test_essd_profile_constants_frozen 逐字钉住，不在此重复。


def test_nsd_delivery_tex_has_no_caption_guard():
    """交付件侧同判据：NSD 档 .tex 零 \\needspace（守卫未注入）——「装回守卫」的产物面证据。"""
    tex = _tex()
    assert "\\needspace" not in tex, "NSD 交付件 .tex 出现题注守卫（夹层将复发）"


def test_nsd_delivery_tex_has_header_split():
    """交付件侧正面判据长表「首页头部/续页
    表头」二分须实际落入交付件——`\\endfirsthead` 与 `\\endhead` 各表恰好一份（二分对外部
    渲染零效应，缺本断言则正则失配会静默通过）。"""
    tex = _tex()
    n_tables = tex.count("\\begin{longtable}")
    assert n_tables >= 1
    assert tex.count("\\endfirsthead") == n_tables, "二分缺失/不完整（\\endfirsthead 计数≠表数）"
    assert tex.count("\\endhead") == n_tables, "续页表头缺失/不完整（\\endhead 计数≠表数）"

# ---------- 题注夹层判据（用户报告「表格看似编码错误」的实因） ----------

def _synthetic_pdf(path: Path, duplicated_head: bool) -> Path:
    """合成单页 PDF：可选地复现「表头→题注→表头→表体」的题注夹层形态。"""
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
    """判据鉴别力：夹层形态计 1、干净形态计 0（合成件，不依赖真实 PDF）。"""
    bad = cps.caption_head_duplicates(pymupdf.open(_synthetic_pdf(tmp_path / "bad.pdf", True)))
    good = cps.caption_head_duplicates(pymupdf.open(_synthetic_pdf(tmp_path / "good.pdf", False)))
    assert bad == 1, bad
    assert good == 0, good


def test_essd_en_caption_sandwich_pinned_to_zero():
    """冻结 ESSD 英文稿题注夹层钉在 **0**（去守卫
    修复登记 3 处（表 2/6/7）→ 0；修复前该件是「缺陷类先于迁移存在」的机械证据）。
    若回归出现夹层，本断言失败即提示「en 档守卫被装回」（守卫按档开关见 build 脚本 case 块）。
    `essd-zh` 档由守卫钉值与基线对照另核（守卫保留为实测决策，见 build 脚本 case 块）。"""
    pdf, log = cps.TARGETS["essd-en"], BUILD / "_manuscript-en.log"
    if not pdf.exists():
        pytest.skip("ESSD 英文稿编译件不存在")
    got = cps.measure(pdf, log)
    assert got["caption_head_duplicates"] == 0, got


def test_nsd_caption_sandwich_pinned_to_zero():
    """NSD 档夹层计数钉在 **0**（不写「等于基线」——以可被证伪；
    D1 的收益面）。若回归出现夹层，本断言失败即提示「守卫被装回」或版面退化。"""
    pdf, log = cps.TARGETS["nsd-en"], BUILD / "_manuscript-nsd.log"
    if not pdf.exists():
        pytest.skip("NSD 编译件不存在")
    got = cps.measure(pdf, log)
    assert got["caption_head_duplicates"] == 0, got


def test_nsd_caption_separation_matches_recorded_state():
    """NSD 档题注-表体分离计数等于重录后的登记值（D1 的**代价**面；不去守卫则
    分离归零而夹层复发——两者同登记，禁单边优化）。"""
    pdf, log = cps.TARGETS["nsd-en"], BUILD / "_manuscript-nsd.log"
    if not pdf.exists():
        pytest.skip("NSD 编译件不存在")
    baseline = json.loads(cps.DEFAULT_BASELINE.read_text(encoding="utf-8"))
    want = baseline["nsd-en"]["caption_body_separation"]
    got = cps.measure(pdf, log)
    assert got["caption_body_separation"] == want, (got, want)


def _synthetic_separation_pdf(path: Path, with_line_number: bool) -> Path:
    """合成单页 PDF：题注下方本页只剩 lineno 行号块（with_line_number）＝表体次页形态；
    否则题注下方接表头行＝表题注同页形态。"""
    import pymupdf
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 80), "Table 1. A source dataset inventory caption block.")
    if with_line_number:
        page.insert_text((40, 700), "14")           # lineno 边距行号（独立数字块）
    else:
        page.insert_text((72, 130), "No. Dataset Domain group DOI")
    doc.save(path)
    doc.close()
    return path


def test_caption_separation_detector(tmp_path: Path):
    """判据鉴别力：题注下接表头（未分离）计 0；题注下只剩 lineno 行号块（真分离）计 1
    ——行号块若不滤除，该形态会被误判为「未分离」，分离计数将永远为 0（判据失效）。"""
    clean = cps.caption_body_separation(
        pymupdf.open(_synthetic_separation_pdf(tmp_path / "clean.pdf", False)))
    sepa = cps.caption_body_separation(
        pymupdf.open(_synthetic_separation_pdf(tmp_path / "sep.pdf", True)))
    assert clean == 0, clean
    assert sepa == 1, sepa


def test_pdf_metadata_title_matches_manuscript():
    """D6 判据：NSD 编译件 PDF 元数据标题 = NSD 稿 H1（题名行）。"""
    pdf, log = cps.TARGETS["nsd-en"], BUILD / "_manuscript-nsd.log"
    if not pdf.exists():
        pytest.skip("NSD 编译件不存在")
    title = NSD_MD.read_text(encoding="utf-8").split("\n", 1)[0].lstrip("# ").strip()
    findings = cps.metadata_findings(cps.measure(pdf, log), title)
    assert findings == [], findings


def test_pdf_metadata_missing_is_caught(tmp_path: Path):
    """负例：PDF 无标题元数据（或标题漂移）→ 命中「元数据标题≠稿源题名」。"""
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

# ---------- SI PDF 判据 ----------

def _si_pages() -> list[str]:
    if not mst.SI_PDF.exists():
        pytest.skip("SI PDF 未生成（跑 python paper/scripts/make_supplementary_tables.py）")
    return csp.pdf_pages(mst.SI_PDF)


def test_si_pdf_content_and_captions_match_sources():
    """正例：SI PDF 逐字段同源于 csv、题注同源于提交要件 §4、主文无 S 表题注、每表一页起排。"""
    findings = csp.run_checks(mst.SI_PDF, nt.SUPPLEMENTARY_DIR, mst.supplementary_captions(),
                              NSD_MD.read_text(encoding="utf-8"))
    assert findings == [], findings


def test_si_pdf_mutated_cell_detected(tmp_path: Path):
    """负例：csv 篡改一格（1.0000→1.0001）→ 命中「字段未见于 SI PDF」（载体漂移）。"""
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
    """负例：题注改一字（提交要件已改而 SI PDF 未重生成）→ 命中题注不一致。"""
    pages = _si_pages()
    drifted = dict(mst.supplementary_captions())
    drifted[1] = drifted[1].replace("Layer registry", "Layer registery", 1)
    findings = csp.caption_findings("\n".join(pages), drifted)
    assert any("题注文本" in f for f in findings), findings


def test_si_pdf_main_md_red_line_detected():
    """负例：主文出现 S 表题注（期刊规范禁止项）→ 命中。"""
    md = NSD_MD.read_text(encoding="utf-8")
    assert csp.main_md_findings(md) == []
    assert csp.main_md_findings("**Supplementary Table 1.** Layer registry …") != []


def test_si_pdf_layout_start_of_page():
    """判据鉴别力：两表题注同页 → 命中「未一页起排」；各起一页 → 通过。"""
    caps = {1: "C1", 2: "C2"}
    assert csp.layout_findings(["Supplementary Table 1. C1\nSupplementary Table 2. C2"], caps)
    assert csp.layout_findings(["Supplementary Table 1. C1\nrow", "Supplementary Table 2. C2"],
                               caps) == []


def test_si_pdf_generator_records_single_caption_source():
    """题注单一来源：生成端与判据端同读提交要件 §4 表（两处各写一份即漂移）；且题注须为
    可直接提交的英文（中文来源注记应在本表第 4 列，不随件提交）。"""
    caps = mst.supplementary_captions()
    assert sorted(caps) == [e.s_number for e in nt.active_supp_entries()]
    cjk = re.compile(r"[\u3000-\u303f\u3400-\u9fff\uff00-\uffef]")
    assert all(len(c) > 30 and not cjk.search(c) for c in caps.values()), \
        f"题注含中文（应移入内部注记列）：{[c for c in caps.values() if cjk.search(c)]}"


def test_nsd_pdf_metadata_author_stays_empty():
    """D5/D6：PDF 元数据作者字段须留空（SSOT 无署名块，署名由投稿系统采集）；
    字段一旦出现即命中——「无署名块」约定从人眼检查升为机械判据。"""
    pdf, log = cps.TARGETS["nsd-en"], BUILD / "_manuscript-nsd.log"
    if not pdf.exists():
        pytest.skip("NSD 编译件不存在")
    measured = cps.measure(pdf, log)
    assert measured["pdf_author"] == "", measured
    assert cps.metadata_findings(measured, "A global multi-resolution data cube") != []  # 题名不符先命中
    assert cps.metadata_findings({"pdf_title": "X", "pdf_author": "A. Author"}, "X") == \
        ["PDF 元数据作者字段非空：'A. Author'（SSOT 无署名块）"]


def test_si_pdf_font_program_magic_is_truetype():
    """⑤ 字体结构判据：SI PDF 所用字体程序魔数须为 TrueType（glyf）。
    负例用分类器鉴别力表达（CFF/OTF 会被写成违规范的 FontFile2 内含 CFF，字形整片不渲染）。"""
    pages = _si_pages()
    assert pages
    assert csp.font_findings(mst.SI_PDF) == [], csp.font_findings(mst.SI_PDF)
    assert csp.font_program_kind(b"OTTO") == "cff"
    assert csp.font_program_kind(b"\x01\x00\x04\x00") == "cff"
    assert csp.font_program_kind(b"\x00\x01\x00\x00") == "truetype"
    assert csp.font_program_kind(b"true") == "truetype"
