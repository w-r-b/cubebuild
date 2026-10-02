"""NSD 编号引用转换与三重核验测试。

判据（参考文献转换：机械转换（键→编号映射）＋三重核验）：
① 引用键集合相对基线不变；② 每条 DOI 不变；③ 编号＝首现序（正文上标首现序 1..N、
   条目编号前缀 1..N、无孤儿）。负例须逐条被拦截——「漏数字/引用键丢失/DOI 改动/
   编号错序」注入形态。

另含 ESSD 冻结轨的**字节冻结**判据：ESSD 英文稿与中文稿相对冻结 commit 536546a 只容许
登记在案的具名例外替换（事实核验轮勘误），
其余逐字节不变（迁移以 ESSD 英文稿为唯一派生源，冻结轨不得被顺手改写）。

运行：pytest tests/test_nsd_references.py（轻量，只读 md 与一次 git show）。
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
except ImportError:  # 公开仓不含论文侧脚本与产物：缺件即跳过（沿用缺件跳过惯例）
    nc = None

pytestmark = pytest.mark.skipif(
    nc is None, reason="论文侧引用转换检查器不在本工作区（缺 paper/scripts/nsd_citations.py）")

BASELINE = REPO / "paper" / "manuscript" / "SEDC-manuscript-en.md"
NSD = REPO / "paper" / "manuscript" / "SEDC-manuscript-nsd-en.md"
FROZEN_COMMIT = "536546a"          # ESSD 冻结基线


@pytest.fixture(scope="module")
def pair() -> tuple[str, str]:
    return BASELINE.read_text(encoding="utf-8"), NSD.read_text(encoding="utf-8")


def test_triple_verification_clean(pair):
    """三重核验零发现（引用键集合 / 逐条 DOI / 编号＝首现序）。"""
    essd, nsd = pair
    findings = nc.verify(nsd, essd)
    assert findings == [], findings


def test_numbering_is_first_occurrence(pair):
    """编号＝首现序：正文上标首现序列与条目编号前缀均为 1..N，且首现序即分配序。"""
    _, nsd = pair
    body, refs = nc.split_sections(nsd, "# References")
    order = [int(x) for m in nc.SUP_RE.finditer(body) for x in m.group(1).split(",")]
    uniq = list(dict.fromkeys(order))
    assert uniq == list(range(1, len(uniq) + 1)), uniq[:12]
    prefixes = [int(m.group(1)) for m in nc.ENTRY_RE.finditer(refs)]
    assert prefixes == list(range(1, len(prefixes) + 1))
    assert len(uniq) == len(prefixes), (len(uniq), len(prefixes))


def test_entry_counts_match_baseline(pair):
    """条目数不变：NSD 编号条目数 = ESSD 基线条目数（「99 条」为笔误，实测 49）。"""
    essd, nsd = pair
    base = nc.parse_entries(nc.split_sections(essd, "# References")[1], numbered=False)
    got = nc.parse_entries(nc.split_sections(nsd, "# References")[1], numbered=True)
    assert len(base) == len(got) == 49, (len(base), len(got))


def test_mutated_doi_detected(pair):
    """② 每条 DOI 不变：篡改 NSD 条目 DOI 末位 → 命中。"""
    essd, nsd = pair
    assert "10.1093/gji/ggae270" in nsd
    findings = nc.verify(nsd.replace("10.1093/gji/ggae270", "10.1093/gji/ggae271"), essd)
    assert any(f.startswith("[②DOI]") for f in findings), findings


def test_dropped_entry_detected(pair):
    """① 引用键集合不变：删 NSD 一条目 → 命中。"""
    essd, nsd = pair
    line = [l for l in nsd.split("\n") if l.startswith("24. Bird, P.")]
    assert len(line) == 1
    findings = nc.verify(nsd.replace(line[0] + "\n", "", 1), essd)
    assert any(f.startswith("[①引用键]") for f in findings), findings


def test_extra_entry_detected(pair):
    """① 反向：NSD 增一条基线没有的条目 → 命中（凑数/串号面）。"""
    essd, nsd = pair
    findings = nc.verify(nsd.replace("\n# Funding", "\n50. Zzfake, Z. Padding. *J. Test* **1**, 1 (2020)."
                                     " https://doi.org/10.1/fake\n\n# Funding", 1), essd)
    assert any(f.startswith("[①引用键]") or f.startswith("[③编号]") for f in findings), findings


def test_out_of_order_numbering_detected(pair):
    """③ 编号＝首现序：把正文首现的两个编号对调 → 命中（编号错序）。"""
    essd, nsd = pair
    assert "^1,2^" in nsd
    findings = nc.verify(nsd.replace("^1,2^", "^2,1^", 1), essd)
    assert any(f.startswith("[③编号]") for f in findings), findings


def test_broken_entry_prefix_detected(pair):
    """③ 条目编号前缀：把第 1 条前缀改为 2 → 命中（缺号/乱序）。"""
    essd, nsd = pair
    assert nsd.count("\n1. Reichstein") == 1
    findings = nc.verify(nsd.replace("\n1. Reichstein", "\n2. Reichstein", 1), essd)
    assert any(f.startswith("[③编号]") for f in findings), findings


def test_orphan_entry_detected(pair):
    """③ 无孤儿：正文引用数 ≠ 条目数（删一处正文引用）→ 命中。"""
    essd, nsd = pair
    findings = nc.verify(nsd.replace("^49^", "", 1), essd)
    assert any(f.startswith("[③编号]") for f in findings), findings


# 冻结轨具名例外（事实核验轮）：
# 两处事实误标勘误（GLAD-M35 的 eta 非 ξ；GUM 20,958 为 pyroclastic）＋
# 表 5/表 3「测试与可复现性」行对象-数字同口径同步（357 = 490 − 论文侧 133）。
# 每条例外须在基线中恰命中一次（死条目即失败）；例外之外任何改动（第三处）在
# 施加替换后的逐字节比较中仍被拦截。
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
    """ESSD 冻结轨字节冻结：相对 536546a 施加登记具名例外后逐字节相等；其余改动即失败。"""
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