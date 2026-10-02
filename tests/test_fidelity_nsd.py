"""en↔en 保真检查器判据测试。

夹具策略：
① 现行配对（ESSD 英文稿 ↔ NSD 稿＋Supplementary 产物）五类对象零差异（回归基线）；
② 对象非空自检（防空跑假绿：提取器失灵时集合级 diff 会静默通过）；
③ 五类真实基底变异负例各一（数字/层 id/DOI·URL/引文键/图注数据值）——注入形态即
   「漏数字、层 id 篡改、引用键丢失、DOI 改动」所列缺陷；
④ 白名单无死条目：逐条删除复算，每条登记项都应确有其事（防豁免面腐化）；
⑤ Supplementary 载体缺席即命中（附录长表内容不得两头不见）。

运行：pytest tests/test_fidelity_nsd.py（轻量，只读 md 与 csv）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "paper" / "scripts"))

try:
    import check_fidelity_nsd as cf  # noqa: E402
except ImportError:  # 公开仓不含论文侧脚本与产物：缺件即跳过（沿用缺件跳过惯例）
    cf = None

pytestmark = pytest.mark.skipif(
    cf is None, reason="论文侧保真检查器不在本工作区（缺 paper/scripts/check_fidelity_nsd.py）")

BASELINE = REPO / "paper" / "manuscript" / "SEDC-manuscript-en.md"
NSD = REPO / "paper" / "manuscript" / "SEDC-manuscript-nsd-en.md"


@pytest.fixture(scope="module")
def pair() -> tuple[str, str]:
    return BASELINE.read_text(encoding="utf-8"), NSD.read_text(encoding="utf-8")


def test_current_pair_clean(pair):
    """现行配对零差异（结构差异已在 STRUCTURAL_EXCEPTIONS 显式登记）。"""
    findings = cf.run_checks(*pair)
    assert findings == [], findings


def test_objects_nonempty(pair):
    """五类对象非空（防空跑通过）。"""
    essd, nsd = pair
    supp = cf.supplementary_text()
    assert len(cf._side_numbers(essd, False)) > 400
    assert len(cf._side_numbers(nsd, True) | cf.ct.numbers(supp)) > 400
    assert len(cf.ct.layer_ids(essd)) >= 47
    assert cf.ct.layer_ids(nsd) | cf.ct.layer_ids(supp) >= cf.ct.layer_ids(essd)
    assert len(cf.ct.dois_urls(essd)) > 100
    assert {"Figure 1", "Figure 2", "Figure 3", "Figure 4", "Figure 5"} <= set(cf.caption_numbers(essd, False))


def test_mutated_number_detected(pair):
    """数字走样：NSD 稿表 2 像元数 64,800→64,801 → 命中。"""
    essd, nsd = pair
    assert "| Pixels | 64,800 |" in nsd
    findings = cf.run_checks(essd, nsd.replace("| Pixels | 64,800 |", "| Pixels | 64,801 |", 1))
    assert any(f.startswith("[数字 token]") and "64801" in f for f in findings), findings


def test_mutated_layer_id_detected(pair):
    """层 id 篡改：NSD 稿层 id 截断 → 命中。"""
    essd, nsd = pair
    assert "topography__bedrock_elevation" in nsd
    findings = cf.run_checks(essd, nsd.replace("topography__bedrock_elevation",
                                               "topography__bedrock_elev", 1))
    assert any(f.startswith("[层 id]") and "topography__bedrock_elev" in f for f in findings), findings


def test_mutated_doi_detected(pair):
    """DOI 改动：NSD 参考文献 DOI 末位篡改 → 命中。"""
    essd, nsd = pair
    assert "10.25921/fd45-gt74" in nsd
    findings = cf.run_checks(essd, nsd.replace("10.25921/fd45-gt74", "10.25921/fd45-gt75", 1))
    assert any(f.startswith("[DOI/URL]") and "fd45-gt75" in f for f in findings), findings


def test_mutated_citation_key_detected(pair):
    """引用键丢失：删 NSD 参考文献一条目 → 命中（引文键集合 + 逐条比对）。"""
    essd, nsd = pair
    assert "46. Neumann, F.," in nsd
    findings = cf.run_checks(essd, nsd.replace("46. Neumann, F.,", "46. Xx, F.,", 1))
    assert any(f.startswith("[引文键]") or f.startswith("[条目]") for f in findings), findings


def test_mutated_figure_caption_value_detected(pair):
    """图注数据值走样：图 3 题注 −10776→−10777 → 命中。"""
    essd, nsd = pair
    assert "−10776 to 8354 m" in nsd
    findings = cf.run_checks(essd, nsd.replace("−10776 to 8354 m", "−10777 to 8354 m", 1))
    assert any(f.startswith("[图注 Figure 3 数据值]") for f in findings), findings


def test_author_date_leftover_detected(pair):
    """编号体例的机械面：NSD 正文残留 author-date 引用 → 命中。"""
    essd, nsd = pair
    findings = cf.run_checks(essd, nsd.replace("The quality evidence is organized",
                                               "(Reichstein et al., 2019) The quality evidence is organized", 1))
    assert any(f.startswith("[编号体例]") for f in findings), findings


def test_whitelist_entries_all_needed(pair, monkeypatch):
    """白名单无死条目：逐条删除复算，每条登记项都必须确有其事。"""
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
    """Supplementary 载体缺席：附录长表内容在 NSD 侧两头不见 → 命中。"""
    essd, nsd = pair
    monkeypatch.setattr(cf, "SUPPLEMENTARY_DIR", tmp_path)   # 空目录
    findings = cf.run_checks(essd, nsd)
    assert any(f.startswith("[层 id]") for f in findings), findings