"""契约加载与铁律前置校验测试（外部行为：契约解析 + 违规即失败指明层）。"""

import pytest

from cubebuild.contract import (
    ContractViolation,
    DEFAULT_CONTRACT_PATH,
    load_contract,
    normalize_resampling,
    parse_native_res_deg,
    validate_contract,
)

MAPPING = DEFAULT_CONTRACT_PATH

_HEADER = """
version: test
status: test
tiers: [1deg, 30min, 6min]
layers:
"""

_VALID_LAYER = """
  - id: test__x
    source: test
    native_res: 0.5deg
    home_tier: 30min
    tiers: [1deg, 30min]
    dtype: float32
    unit: m
    resampling: conservative_area_weighted
    mask: validity
"""


def _write(tmp_path, body):
    p = tmp_path / "contract.yaml"
    p.write_text(_HEADER + body, encoding="utf-8")
    return p


def test_load_real_contract():
    c = load_contract(MAPPING)
    assert len(c.layers) == 47
    # 六项待确认再分发许可的数据集共 12 层为 internal
    # （GLAD-M35×5 + SEMUCB×2 + PB2002×3 + GST1×1 +
    # HFgrid14×1 = 12，枚举明细为准）→ 25 public / 22
    # internal（既有 10 + 新 12）
    assert sum(1 for l in c.layers if l.visibility == "public") == 25
    assert sum(1 for l in c.layers if l.visibility == "internal") == 22
    # 热流部分退出拆分口径：HFgrid14 层（源
    # 栅格再分发）退 internal、密度层（聚合分析产物，不可恢复源记录）留
    # 公开——数据集粒度的部分退出，非全退/全留
    hf = next(l for l in c.layers if l.id == "thermal__hfgrid14_heatflow")
    hd = next(l for l in c.layers if l.id == "thermal__heatflow_density")
    assert hf.visibility == "internal" and hd.visibility == "public"
    validate_contract(c)  # 定稿契约必须整体通过铁律


def test_pixel_area_entry_fields():
    c = load_contract(MAPPING)
    pa = next(l for l in c.layers if l.id == "derived__pixel_area")
    assert pa.tiers == ("1deg", "30min", "6min", "3min", "30sec")
    assert pa.dtype == "float32"
    assert pa.unit == "km2"
    assert pa.resampling == "exact_formula"
    assert pa.dims == ("lat", "lon")
    assert pa.native_res_deg is None  # exact：解析公式无档位约束


def test_iron_rule_rejects_finer_tier_and_names_layer(tmp_path):
    # native 0.5deg 却进 6min(0.1°) 档 → 铁律违规，指明违规层
    p = _write(
        tmp_path,
        """
  - id: test__x
    source: test
    native_res: 0.5deg
    home_tier: 6min
    tiers: [1deg, 30min, 6min]
    dtype: float32
    unit: m
    resampling: conservative_area_weighted
    mask: validity
""",
    )
    with pytest.raises(ContractViolation) as ei:
        validate_contract(load_contract(p))
    assert "test__x" in str(ei.value)
    assert "铁律违规" in str(ei.value)


def test_iron_rule_allows_equal_tier(tmp_path):
    # native 0.1deg 进 6min(0.1°) 档：等宽不算细于，放行
    p = _write(
        tmp_path,
        """
  - id: test__x
    source: test
    native_res: 0.1deg
    home_tier: 6min
    tiers: [1deg, 6min]
    dtype: float32
    unit: m
    resampling: none
    mask: validity
""",
    )
    validate_contract(load_contract(p))


def test_full_set_rule_missing_1deg(tmp_path):
    p = _write(
        tmp_path,
        """
  - id: test__x
    source: test
    native_res: 1deg
    home_tier: 30min
    tiers: [30min]
    dtype: float32
    unit: m
    resampling: none
    mask: validity
""",
    )
    with pytest.raises(ContractViolation) as ei:
        validate_contract(load_contract(p))
    assert "1deg" in str(ei.value)


def test_home_tier_must_be_in_tiers(tmp_path):
    p = _write(
        tmp_path,
        _VALID_LAYER.replace("home_tier: 30min", "home_tier: 6min"),
    )
    with pytest.raises(ContractViolation) as ei:
        validate_contract(load_contract(p))
    assert "home_tier" in str(ei.value)


def test_missing_required_field(tmp_path):
    body = """
  - id: test__x
    source: test
    native_res: 0.5deg
    home_tier: 30min
    tiers: [1deg, 30min]
    dtype: float32
    unit: m
    resampling: conservative_area_weighted
"""
    p = _write(tmp_path, body)  # 缺 mask
    with pytest.raises(ContractViolation) as ei:
        load_contract(p)
    assert "mask" in str(ei.value)


def test_normalize_resampling_strips_annotations():
    assert normalize_resampling("none（原生 1° 即全集档）") == "none"
    assert normalize_resampling("exact_formula（逐档精确计算，不重采样）") == "exact_formula"
    assert (
        normalize_resampling("circular_axial（2θ 变换循环统计；走向无方向性）")
        == "circular_axial"
    )
    assert normalize_resampling("mode") == "mode"


def test_parse_native_res_deg():
    assert parse_native_res_deg("15sec") == pytest.approx(15 / 3600)
    assert parse_native_res_deg("2min") == pytest.approx(1 / 30)
    assert parse_native_res_deg("0.125deg") == 0.125
    assert parse_native_res_deg("point") is None
    assert parse_native_res_deg("vector-polygon") is None
    assert parse_native_res_deg("exact") is None
