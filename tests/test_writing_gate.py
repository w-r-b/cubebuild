"""check_writing.py 的行为测试（随引用两向改版）。

夹具策略：
① 词表回归：修订**前**的 02-methods.md（git ee9477a^ = cedb97b）应命中
   工单动词，修订后零命中——真实历史负例（02 章区域自合并稿切取）；
② 当前合并稿全项零发现（回归基线，S3 已知债快照）；
③ 回指落空：合成最小 manuscript 触发（dangling 节/附录/图表）；
④ 引用两向负例：合成最小 manuscript（正文键无 References 条目 + References 孤儿）；
⑤ 真实基底变异负例：合并稿删一条 References 条目 → 正文键落空；插一条凑数
   条目 → 孤儿。原有的 ee9477a 三向缺口夹具随章末表结构退役（其守卫的
   章末表层缺口在新结构下不存在，改由本组测试守护两向判据的鉴别力）。

运行：pytest tests/test_writing_gate.py（轻量，只读文件与一次 git show）。
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "paper" / "scripts"))

try:
    import check_writing as cw  # noqa: E402
except ImportError:  # 公开仓不含论文侧脚本与产物：缺件即跳过（沿用缺件跳过惯例）
    cw = None

pytestmark = pytest.mark.skipif(
    cw is None, reason="论文侧写作检查器不在本工作区（缺 paper/scripts/check_writing.py）")

MANUSCRIPT = REPO / "paper" / "manuscript"
MERGED_MD = MANUSCRIPT / "SEDC-manuscript-zh.md"
EN_MD = MANUSCRIPT / "SEDC-manuscript-en.md"


def run_checks(mdir: Path) -> list[str]:
    files = cw.load_files(mdir)
    return cw.check_wordlists(files) + cw.check_citations(files) + cw.check_refs(files)


# ---------- ① 词表：真实历史负例（修订前） ----------

PRE_REV = "cedb97b:paper/manuscript/02-methods.md"  # ee9477a^：收口提交的父态


@pytest.fixture(scope="module")
def pre_revision_dir(tmp_path_factory):
    try:
        out = subprocess.run(
            ["git", "show", PRE_REV],
            cwd=REPO, capture_output=True, check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        pytest.skip("git 历史不可用（无 cedb97b 提交）")
    d = tmp_path_factory.mktemp("pre_rev")
    (d / "02-methods.md").write_text(out.stdout.decode("utf-8"), encoding="utf-8")
    return d


def test_pre_revision_hits_internal_words(pre_revision_dir):
    findings = cw.check_wordlists(cw.load_files(pre_revision_dir))
    joined = "\n".join(findings)
    for word in ("烧录", "白名单", "制品实测"):
        assert word in joined, f"修订前 02-methods 应命中内部语域词：{word}"


def test_pre_revision_zero_words_now(tmp_path):
    # 02 章区域自合并稿切取（章标题至下一章标题），修订后应零命中
    assert MERGED_MD.exists()
    text = MERGED_MD.read_text(encoding="utf-8")
    region = text[text.index("# 2 数据源与构建方法"):text.index("# 3 数据立方描述")].rstrip("\n")
    d = tmp_path / "02-methods.md"
    d.write_text(region, encoding="utf-8")
    assert cw.check_wordlists([d]) == []


# ---------- ② 当前合并稿回归基线 ----------
# S3（复核级）已知合法命中快照：已知项之外不得新增。
# 勘误记录：原第二项「独立实现的＋
# 生产实现」（合并稿:220，§4.3 距离场 bullet）随该段表格化改写消解——表 4
# 对应行改用「独立撰写」「生产端算法路径」措辞，命中不再存在，快照缩为一项。

S3_KNOWN = {
    "SEDC-manuscript-zh.md:15",   # 完成（自行完成跨源对齐，复核级）
}


def test_current_manuscript_clean():
    files = cw.load_files(MANUSCRIPT, language="zh")   # 语言过滤：只查中文稿（并置后互不误扫）
    findings = cw.check_wordlists(files) + cw.check_citations(files) + cw.check_refs(files)
    s3 = {f.split("] ", 1)[1].split(" 「", 1)[0] for f in findings if f.startswith("[S3")}
    assert s3 == S3_KNOWN, f"S3 已知债之外的新增命中：{s3 ^ S3_KNOWN}"
    others = [f for f in findings if not f.startswith("[S3")]
    assert others == [], others


# ---------- ③ 回指落空：合成最小 manuscript ----------

def _write(d: Path, name: str, text: str):
    (d / name).write_text(text, encoding="utf-8")


def test_dangling_refs_detected(tmp_path):
    _write(tmp_path, "01-a.md", "# 1 概念\n\n见第 2.3 节与附录 E；详见第 4.9 节；如图 9 所示。\n")
    findings = cw.check_refs(cw.load_files(tmp_path))
    joined = "\n".join(findings)
    assert "第 2.3 节」不存在" in joined
    assert "附录 E」不存在" in joined
    assert "第 4.9 节」不存在" in joined
    assert "图 9」无对应题注" in joined


def test_figtab_guard_no_false_positive(tmp_path):
    # 「欧拉极表 203 行」类名词+计数不得判为图表引用（实测误报案例）
    _write(tmp_path, "01-a.md", "# 1 概念\n\n另附欧拉极表 203 行。\n")
    assert cw.check_refs(cw.load_files(tmp_path)) == []


def test_valid_refs_pass(tmp_path):
    _write(
        tmp_path, "01-a.md",
        "# 1 概念\n\n见第 1 节；数据见表 2（表 2 见下）。\n\n**表 2** 示例表。\n",
    )
    assert cw.check_refs(cw.load_files(tmp_path)) == []


# ---------- ④ 引用两向：合成最小 manuscript ----------

REFS = """# 参考文献（References）

Alpha, A.: One, J., 1, 1–2, https://doi.org/10.1/one, 2001.

Beta, B.: Two, J., 2, 3–4, https://doi.org/10.1/two, 2002.
"""

REFS_CLEAN = """# 参考文献（References）

Alpha, A.: One, J., 1, 1–2, https://doi.org/10.1/one, 2001.
"""

BODY = """# 1 概念

正文引用（Alpha, 2001）。表格引用（Gamma, 2003）。另有正文键无条目（Delta, 2004）。
"""

BODY_CLEAN = """# 1 概念

正文引用（Alpha, 2001）。表格引用（Alpha, 2001）。
"""


def test_citation_diffs_detected(tmp_path):
    _write(tmp_path, "SEDC-manuscript-zh.md", BODY + "\n" + REFS)
    findings = cw.check_citations(cw.load_files(tmp_path))
    joined = "\n".join(findings)
    assert "正文键无References条目] SEDC-manuscript-zh.md：Gamma, 2003" in joined
    assert "正文键无References条目] SEDC-manuscript-zh.md：Delta, 2004" in joined
    assert "References孤儿" in joined              # Beta 收录但正文/表内零出现
    assert "Beta" in joined


def test_citations_consistent_pass(tmp_path):
    _write(tmp_path, "SEDC-manuscript-zh.md", BODY_CLEAN + "\n" + REFS_CLEAN)
    assert cw.check_citations(cw.load_files(tmp_path)) == []


def test_missing_refs_section_reported(tmp_path):
    _write(tmp_path, "01-a.md", "# 1 概念\n\n正文引用（Alpha, 2001）。\n")
    findings = cw.check_citations(cw.load_files(tmp_path))
    assert len(findings) == 1 and "结构缺失" in findings[0]


# ---------- ⑤ 真实基底变异负例：两向判据鉴别力 ----------

def test_real_manuscript_missing_entry_detected(tmp_path):
    # 删一条真实 References 条目（Wilkinson）→ 其正文键 (Wilkinson, 2016) 落空
    text = MERGED_MD.read_text(encoding="utf-8")
    assert text.count("\nWilkinson, M. D.,") == 1
    mutated = "\n".join(l for l in text.split("\n") if not l.startswith("Wilkinson, M. D.,"))
    _write(tmp_path, "SEDC-manuscript-zh.md", mutated)
    joined = "\n".join(cw.check_citations(cw.load_files(tmp_path)))
    assert "正文键无References条目] SEDC-manuscript-zh.md：Wilkinson, 2016" in joined


def test_real_manuscript_padding_entry_detected(tmp_path):
    # 插一条正文零出现的凑数条目 → References 孤儿
    text = MERGED_MD.read_text(encoding="utf-8")
    hdr = "# 参考文献（References）"
    i = text.index(hdr) + len(hdr)
    mutated = text[:i] + "\n\nZzfake, Z.: Padding entry, J. Test, 1, 1–2, 2020." + text[i:]
    _write(tmp_path, "SEDC-manuscript-zh.md", mutated)
    joined = "\n".join(cw.check_citations(cw.load_files(tmp_path)))
    assert "References孤儿] 未在正文/表内出现：Zzfake, Z.: Padding" in joined


# ---------- ⑥ 键提取单元 ----------

def test_entry_key_forms():
    assert cw.entry_key("Mardia, K. V., and Jupp, P. E.: Directional Statistics, Wiley, 2000.") == ("Mardia", "2000")
    assert cw.entry_key("GDAL/OGR contributors: GDAL/OGR Library, 2025.") == ("GDAL/OGR contributors", "2025")
    k, y = cw.entry_key("Global Heat Flow Data Assessment Group, Fuchs, S., and others: Release 2024, 2024.")
    assert k == "Global Heat Flow Data Assessment Group" and y == "2024"


def test_parse_tokens_mixed_forms():
    body = (
        "（Heidbach et al., 2025; Kreemer et al., 2014）"
        "（NOAA National Centers for Environmental Information, 2022）"
        "（见 Ploton et al., 2020）"
        "（0–540 Ma；步长 10 km；EPSG:4326；10–2890 km）"
    )
    toks = cw.parse_tokens(body)
    assert ("Heidbach", "2025") in toks
    assert ("Kreemer", "2014") in toks
    assert ("NOAA National Centers for Environmental Information", "2022") in toks
    assert ("Ploton", "2020") in toks
    assert not any(not t[1].isdigit() for t in toks)  # 年份均为纯数字
    assert all("km" not in t[0] and "°" not in t[0] for t in toks)  # 非引用排除


def test_parse_tokens_requires_comma():
    # 无逗号括注年份（附录 B 版本格「2012（LiMW 2015）」）不是引用，不得计入
    toks = cw.parse_tokens("版本格 2012（LiMW 2015）与正文引用（Alpha, 2001）。")
    assert toks == {("Alpha", "2001")}


# ---------- ⑦ S6 记号零容忍：负例与现稿零命中 ----------

def test_s6_symbols_flagged(tmp_path):
    text = MERGED_MD.read_text(encoding="utf-8")
    d = tmp_path / "SEDC-manuscript-zh.md"
    d.write_text(text.replace("保真：积分量守恒", "保真·积分量守恒", 1), encoding="utf-8")
    findings = cw.check_wordlists([d])
    assert any("S6 记号零容忍" in f for f in findings), findings


def test_s6_current_manuscript_clean():
    assert not any("S6" in f for f in cw.check_wordlists(cw.load_files(MANUSCRIPT)))


# ---------- ⑧ 英文稿与语言参数化 ----------
# 英文侧判据：E 系词表（S 系英文对应）、# References 标题、英文回指体系
# （Sect./Appendix/Fig./Table）；语言过滤保证中英并置互不误扫。

def test_language_filter_isolates():
    """语言过滤：zh 只取中文稿、en 只取英文稿；不过滤（auto）两稿并见。"""
    zh = cw.load_files(MANUSCRIPT, language="zh")
    en = cw.load_files(MANUSCRIPT, language="en")
    assert [f.name for f in zh] == ["SEDC-manuscript-zh.md"]
    # 迁移后 en 语言面含两稿（ESSD 基线 + NSD 稿），按**期刊档**隔离（见 test_profile_isolates）
    assert [f.name for f in en] == ["SEDC-manuscript-en.md", "SEDC-manuscript-nsd-en.md"]
    assert {f.name for f in cw.load_files(MANUSCRIPT)} == {
        "SEDC-manuscript-zh.md", "SEDC-manuscript-en.md", "SEDC-manuscript-nsd-en.md"}


def test_profile_isolates():
    """期刊档隔离：NSD 稿的缺号/孤儿不污染 ESSD 英文稿判据（反之亦然）。"""
    essd = [f for f in cw.load_files(MANUSCRIPT, language="en") if cw.file_profile(f) == "essd"]
    nsd = [f for f in cw.load_files(MANUSCRIPT, language="en") if cw.file_profile(f) == "nsd"]
    assert [f.name for f in essd] == ["SEDC-manuscript-en.md"]
    assert [f.name for f in nsd] == ["SEDC-manuscript-nsd-en.md"]


def test_current_english_manuscript_clean():
    """英文稿全项零发现（E 系词表、引用两向、英文回指体系）。"""
    assert EN_MD.exists(), "英文稿缺失"
    files = cw.load_files(MANUSCRIPT, language="en")
    findings = cw.check_wordlists(files) + cw.check_citations(files) + cw.check_refs(files)
    assert findings == [], findings


def test_english_wordlist_flagged(tmp_path):
    """E 系负例：注入英文导航句 → 命中（E5）。"""
    _write(tmp_path, "SEDC-manuscript-en.md",
           "# 1 Introduction\n\nWe will discuss the design. This section introduces the cube.\n")
    findings = cw.check_wordlists(cw.load_files(tmp_path))
    assert any("E5 导航句" in f for f in findings), findings


def test_english_fullwidth_punct_flagged(tmp_path):
    """E7 负例：英文区段残留顿号/全角逗号 → 命中（TR-014 的机械面）。"""
    _write(tmp_path, "SEDC-manuscript-en.md",
           "# 1 Introduction\n\nThe cube、the store，is one.\n")
    findings = cw.check_wordlists(cw.load_files(tmp_path))
    assert any("E7" in f for f in findings), findings


def test_english_anaphora_dangling_detected(tmp_path):
    """英文回指负例：Sect./Appendix/Fig./Table 落空 → 命中。"""
    _write(tmp_path, "SEDC-manuscript-en.md",
           "# 1 Introduction\n\nSee Sect. 9.9 and Appendix Z; as shown in Fig. 9 and Table 9.\n")
    findings = cw.check_refs(cw.load_files(tmp_path))
    joined = "\n".join(findings)
    assert "Sect. 9.9」不存在" in joined
    assert "Appendix Z」不存在" in joined
    assert "Fig. 9」无对应题注" in joined
    assert "Table 9」无对应题注" in joined


def test_english_valid_refs_pass(tmp_path):
    """英文回指正例：节/附录/表题注齐备 → 零发现。"""
    _write(tmp_path, "SEDC-manuscript-en.md",
           "# 1 Introduction\n\nSee Sect. 1 and Appendix A; Table 1 lists the data.\n\n"
           "**Table 1.** Example.\n\n# Appendix\n\n## Appendix A Registry\n")
    assert cw.check_refs(cw.load_files(tmp_path)) == []


def test_english_citation_twoways(tmp_path):
    """英文引用两向：# References 标题下，正文键缺条目与 References 孤儿各命中。"""
    refs = ("# References\n\nAlpha, A.: One, J., 1, 1–2, https://doi.org/10.1/one, 2001.\n\n"
            "Beta, B.: Two, J., 2, 3–4, https://doi.org/10.1/two, 2002.\n")
    body = "# 1 Introduction\n\nCited (Alpha, 2001) and (Gamma, 2003).\n"
    _write(tmp_path, "SEDC-manuscript-en.md", body + "\n" + refs)
    findings = cw.check_citations(cw.load_files(tmp_path))
    joined = "\n".join(findings)
    assert "正文键无References条目] SEDC-manuscript-en.md：Gamma, 2003" in joined
    assert "References孤儿" in joined and "Beta" in joined


def test_english_citation_see_leadin_parses(tmp_path):
    """英文引述领起（"… see Neumann et al., 2026"）不产生首作者键污染（真实假阳性回归）。"""
    refs = "# References\n\nNeumann, F., Norden, B., and others: The 2024 release, 2026.\n"
    body = "# 1 Introduction\n\nSee the description (for the database description see Neumann et al., 2026).\n"
    _write(tmp_path, "SEDC-manuscript-en.md", body + "\n" + refs)
    assert cw.check_citations(cw.load_files(tmp_path)) == []


def test_mixed_dir_no_crosstalk(tmp_path):
    """中英并置互不误扫：zh 稿含中文违规、en 稿干净 → en 过滤下零发现。"""
    _write(tmp_path, "SEDC-manuscript-zh.md",
           "# 1 概念\n\n本稿 烧录 与 制品实测 只在中文侧。\n\n"
           "# 参考文献（References）\n\nAlpha, A.: One, 1, 2001.\n")
    _write(tmp_path, "SEDC-manuscript-en.md",
           "# 1 Introduction\n\nA clean English body with (Alpha, 2001).\n\n"
           "# References\n\nAlpha, A.: One, 1, 2001.\n")
    zh_findings = cw.check_wordlists(cw.load_files(tmp_path, language="zh"))
    assert any("S1" in f for f in zh_findings), zh_findings
    en_files = cw.load_files(tmp_path, language="en")
    findings = cw.check_wordlists(en_files) + cw.check_citations(en_files) + cw.check_refs(en_files)
    assert findings == [], findings


# ---------- ⑨ NSD 档（写作门禁 NSD 档） ----------
# 判据：具名节集合（规定节齐备、ESSD 遗留节零出现）、编号引用体例（无 author-date 残留、
# 上标编号合式、条目编号前缀 1..N）、图题注 ≤350 词、Supplementary Table 引用可解析。

NSD_MD = MANUSCRIPT / "SEDC-manuscript-nsd-en.md"


@pytest.fixture(scope="session")
def nsd_tmp(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """临时目录（session 级，由 pytest 清理）：NSD 稿副本落此，不污染 manuscript 目录。"""
    return tmp_path_factory.mktemp("nsd-gate")


def _nsd_checks(text: str, tmp: Path) -> list[str]:
    f = tmp / "SEDC-manuscript-nsd-en.md"
    f.write_text(text, encoding="utf-8")
    files = [f]
    return (cw.check_wordlists(files) + cw.check_citations(files) + cw.check_refs(files)
            + cw.check_nsd_structure(files) + cw.check_nsd_citation_style(files)
            + cw.check_nsd_captions(files))


def test_nsd_manuscript_clean(nsd_tmp):
    """NSD 稿全项零发现（词表/回指/具名节/编号引用/题注词数）。"""
    findings = _nsd_checks(NSD_MD.read_text(encoding="utf-8"), nsd_tmp)
    assert findings == [], findings


def test_nsd_missing_section_detected(nsd_tmp):
    """负例：删「Data Overview」具名节 → 命中。"""
    text = NSD_MD.read_text(encoding="utf-8")
    findings = _nsd_checks(text.replace("# Data Overview", "# Cube numbers", 1), nsd_tmp)
    assert any("缺规定节「Data Overview」" in f for f in findings), findings


def test_nsd_forbidden_section_detected(nsd_tmp):
    """负例：ESSD 遗留节名（结论/附录）回插 → 命中。"""
    text = NSD_MD.read_text(encoding="utf-8")
    findings = _nsd_checks(text + "\n# Conclusions\n\nSome text.\n", nsd_tmp)
    assert any("ESSD 遗留节标题" in f for f in findings), findings


def test_nsd_author_date_leftover_detected(nsd_tmp):
    """负例：正文残留 author-date 引用（编号体例未改号）→ 命中。"""
    text = NSD_MD.read_text(encoding="utf-8")
    findings = _nsd_checks(text.replace("The quality evidence is organized",
                                        "(Reichstein et al., 2019) The quality evidence is organized", 1),
                           nsd_tmp)
    assert any("残留 author-date" in f for f in findings), findings


def test_nsd_reference_prefix_gap_detected(nsd_tmp):
    """负例：参考文献条目编号缺号（第 1 条改 2）→ 命中。"""
    text = NSD_MD.read_text(encoding="utf-8")
    findings = _nsd_checks(text.replace("\n1. Reichstein", "\n2. Reichstein", 1), nsd_tmp)
    assert any("参考文献编号前缀非 1..N" in f for f in findings), findings


def test_nsd_citation_out_of_range_detected(nsd_tmp):
    """负例：正文引用编号超出条目数 → 命中。"""
    text = NSD_MD.read_text(encoding="utf-8")
    findings = _nsd_checks(text.replace("^1,2^", "^1,2,99^", 1), nsd_tmp)
    assert any("超出条目数" in f for f in findings), findings


def test_nsd_caption_over_limit_detected(nsd_tmp):
    """负例：图题注超 350 词 → 命中。"""
    text = NSD_MD.read_text(encoding="utf-8")
    findings = _nsd_checks(text.replace("**Figure 4.** Tier zoom comparison",
                                        "**Figure 4.** Tier zoom comparison " + "word " * 360, 1), nsd_tmp)
    assert any("题注 4" in f and "350" in f for f in findings), findings


def test_nsd_section_number_pointer_detected(nsd_tmp):
    """负例：NSD 档出现节号回指（Sect. x，具名节体例禁用）→ 命中。"""
    text = NSD_MD.read_text(encoding="utf-8")
    assert "described in Data Records." in text
    findings = _nsd_checks(text.replace("described in Data Records.", "described in Sect. 2.6.", 1), nsd_tmp)
    assert any("NSD 档不得用节号回指" in f for f in findings), findings


def test_nsd_supplementary_ref_dangling_detected(nsd_tmp):
    """负例：Supplementary Table 引用无对应产物（S9）→ 命中。"""
    text = NSD_MD.read_text(encoding="utf-8")
    findings = _nsd_checks(text.replace("Supplementary Table 1", "Supplementary Table 9", 1), nsd_tmp)
    assert any("Supplementary Table 9「无对应" in f or "Supplementary Table 9」无对应" in f
               for f in findings), findings


def test_essd_profile_default_unchanged():
    """ESSD 默认档回归：英文稿在 ESSD 档仍零发现、中文稿仅存既有 S3 已知债。"""
    en = [f for f in cw.load_files(MANUSCRIPT, language="en") if cw.file_profile(f) == "essd"]
    assert cw.check_wordlists(en) + cw.check_citations(en) + cw.check_refs(en) == []
    assert EN_MD.exists()

