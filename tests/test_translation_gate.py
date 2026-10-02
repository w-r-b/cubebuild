"""check_translation.py 的译英保真判据测试。

夹具策略：
① 现行中英配对（中文合并稿 SSOT ↔ 英文稿 SSOT）五类检查零差异（回归基线）；
② 五类真实基底变异负例各一（数字/引用键/层 id/DOI·URL/参考文献节）必须被拦截——
   注入形态即「新增判据须带负例回归」所列的缺陷形态（数字或引用缺失、层 id 篡改）。

运行：pytest tests/test_translation_gate.py（轻量，只读两稿 md）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "paper" / "scripts"))

try:
    import check_translation as ct  # noqa: E402
except ImportError:  # 公开仓不含论文侧脚本与产物：缺件即跳过（沿用缺件跳过惯例）
    ct = None

pytestmark = pytest.mark.skipif(
    ct is None, reason="论文侧译校检查器不在本工作区（缺 paper/scripts/check_translation.py）")

ZH_MD = REPO / "paper" / "manuscript" / "SEDC-manuscript-zh.md"
EN_MD = REPO / "paper" / "manuscript" / "SEDC-manuscript-en.md"


def _pair() -> tuple[str, str]:
    return ZH_MD.read_text(encoding="utf-8"), EN_MD.read_text(encoding="utf-8")


# ---------- ① 回归基线：现行配对零差异 ----------

def test_current_pair_clean():
    zh, en = _pair()
    findings = ct.run_checks(zh, en)
    assert findings == [], findings


def test_objects_nonempty():
    """五类对象均非空（防空跑通过：提取器失灵时基线测试会假绿）。"""
    zh, en = _pair()
    assert len(ct.numbers(zh)) > 400 and len(ct.numbers(en)) > 400
    # 47 层 id + §5.1 代码示例中的有效性掩膜名（两稿一致）
    assert len(ct.layer_ids(zh)) >= 47 and ct.layer_ids(zh) == ct.layer_ids(en)
    assert len(ct.dois_urls(zh)) > 100 and len(ct.dois_urls(en)) > 100


# ---------- ② 变异负例：五类各一 ----------

def test_mutated_number_detected():
    """数字走样：英文稿表 2 像元数 64,800→64,801 → 命中（同值在别处存活，故只见英侧新值）。"""
    zh, en = _pair()
    assert "| Pixels | 64,800 |" in en
    findings = ct.run_checks(zh, en.replace("| Pixels | 64,800 |", "| Pixels | 64,801 |", 1))
    assert any(f.startswith("[数字 token]") and "64801" in f for f in findings), findings


def test_mutated_citation_detected():
    """引用缺失/走样：英文稿年份 2026→2027 → 命中。"""
    zh, en = _pair()
    assert "(Mancino et al., 2026)" in en
    findings = ct.run_checks(zh, en.replace("(Mancino et al., 2026)", "(Mancino et al., 2027)", 1))
    assert any(f.startswith("[引用键]") and "Mancino" in f for f in findings), findings


def test_mutated_layer_id_detected():
    """层 id 篡改：英文稿层 id 截断 → 命中。"""
    zh, en = _pair()
    assert "topography__bedrock_elevation" in en
    findings = ct.run_checks(zh, en.replace("topography__bedrock_elevation",
                                            "topography__bedrock_elev", 1))
    assert any(f.startswith("[层 id]") and "topography__bedrock_elev" in f for f in findings), findings


def test_mutated_doi_detected():
    """DOI 走样：英文稿数据 DOI 末位篡改（同 DOI 在表 1 与附录 B 各一份，改一份即见英侧新值）。"""
    zh, en = _pair()
    assert "10.25921/fd45-gt74" in en
    findings = ct.run_checks(zh, en.replace("10.25921/fd45-gt74", "10.25921/fd45-gt75", 1))
    assert any(f.startswith("[DOI/URL]") and "fd45-gt75" in f for f in findings), findings


def test_mutated_reference_entry_detected():
    """参考文献节走样（「不译」清单项）：卷页号篡改 → 命中。"""
    zh, en = _pair()
    assert "Geochem. Geophys. Geosyst., 4, 1027" in en
    findings = ct.run_checks(zh, en.replace("Geochem. Geophys. Geosyst., 4, 1027",
                                            "Geochem. Geophys. Geosyst., 4, 1028", 1))
    assert any(f.startswith("[参考文献节]") for f in findings), findings


def test_missing_refs_section_detected():
    """结构缺失：英文稿无 References 节 → 命中（结构级，不静默通过）。"""
    zh, en = _pair()
    findings = ct.run_checks(zh, en.replace("# References", "# Bibliography", 1))
    assert any(f.startswith("[参考文献节]") and "未同时找到" in f for f in findings), findings