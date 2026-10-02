"""层映射表 YAML 契约加载与铁律前置校验。

契约即施工图纸：构建只消费这里解析出的条目，不做自由裁量。
铁律：任何层不得进入细于其原生分辨率的档；1° 为全集档（所有层汇入）。
"""

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from .grids import TIER_ORDER, tier_deg

# 层契约（管线的声明式规格）随包分发的固定位置：
# --mapping 缺省值与既有测试共同引用的唯一位置定义
DEFAULT_CONTRACT_PATH = Path(__file__).resolve().parent / "layer-mapping-v0.yaml"

# 规范允许的重采样核（YAML 条目可带中文括注，取主干 token 归一化）
RESAMPLING_TOKENS = {
    "conservative_area_weighted",
    "mode",
    "circular",
    "circular_axial",
    "sum",
    "none",
    "exact_formula",
}

DTYPES = {"float32", "uint8", "uint32"}

VISIBILITIES = {"public", "internal"}

# native_res 数值形态：如 1deg / 0.5deg / 2min / 15sec；point/vector/exact 无栅格约束
_NATIVE_RES_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(deg|min|sec)\s*$")
_NATIVE_RES_UNIT_DEG = {"deg": 1.0, "min": 1.0 / 60.0, "sec": 1.0 / 3600.0}
# 非数值原生形态（点/矢量/解析公式）：铁律不施加档位约束
_NATIVE_RES_UNCONSTRAINED = {"point", "vector-line", "vector-polygon", "exact"}

# 特殊 home_tier 语义：逐档生成（无单一主档，如 pixel_area 逐档精确计算）
_HOME_TIER_SPECIAL = {"逐档"}

_REQUIRED_FIELDS = (
    "id",
    "source",
    "native_res",
    "home_tier",
    "tiers",
    "dtype",
    "unit",
    "resampling",
    "mask",
)


class ContractViolation(Exception):
    """契约校验失败：violations 为 (位置, 问题描述) 列表，逐条指明违规层。"""

    def __init__(self, violations: list[tuple[str, str]]):
        self.violations = violations
        lines = [f"  [{loc}] {msg}" for loc, msg in violations]
        super().__init__("契约校验失败（共 %d 项）:\n" % len(violations) + "\n".join(lines))


_IRON_RULE_EPS = 1e-12
_TIER_ORDER_SET = set(TIER_ORDER)


def violates_iron_rule(tier: str, native_res_deg: float) -> bool:
    """铁律判断的唯一实现：档位是否细于层原生分辨率（等宽不算细于）。

    未知档位返回 False（其非法性由字段合法性检查负责）。
    """
    if tier not in _TIER_ORDER_SET:
        return False
    return tier_deg(tier) < native_res_deg - _IRON_RULE_EPS


def needs_validity_mask(mask_field: str) -> bool:
    """YAML mask 字段是否要求 uint8 有效性伴生层（判定唯一实现）。

    取值形态（映射表实测）：要求 = "validity" / "validity + landsea（…）"
    / "validity（…）" / "NaN（板片区外）+ validity"（Slab2 形态——ASCII
    token "validity" 出现即要求）；不要求 = "免" / "全球全覆盖，免有效性
    掩膜" / "随高程层" / "自身即掩膜"（均不含 ASCII "validity"）。
    """
    return "validity" in mask_field


@dataclass(frozen=True)
class LayerEntry:
    id: str
    dims: tuple[str, ...]
    source: str
    native_res: str
    native_res_deg: float | None  # None = 无栅格约束（point/vector/exact）
    home_tier: str
    tiers: tuple[str, ...]
    dtype: str
    unit: str
    resampling: str  # 归一化 token
    mask: str  # 原文保留（如 "validity + landsea"、"免"）
    visibility: str
    notes: str = ""


@dataclass(frozen=True)
class Contract:
    path: Path
    version: str
    status: str
    tiers: tuple[str, ...]
    layers: tuple[LayerEntry, ...]
    vectors_sidecar: tuple
    validation_reference: tuple
    not_ingested: tuple


def _strip_annotation(raw: str) -> str:
    """截去中文/英文括注，取主干 token（如 "0.05deg（分区网格）" → "0.05deg"）。"""
    token = raw
    for ch in "（(":
        idx = token.find(ch)
        if idx != -1:
            token = token[:idx]
    return token.strip()


def normalize_resampling(raw: str) -> str:
    """归一化重采样核 token（如 "none（逐档重算）" → "none"）。"""
    return _strip_annotation(raw)


def parse_native_res_deg(raw: str) -> float | None:
    """原生分辨率（度）；point/vector/exact 等非栅格形态返回 None。"""
    token = _strip_annotation(raw)
    if token in _NATIVE_RES_UNCONSTRAINED:
        return None
    m = _NATIVE_RES_RE.match(token)
    if m is None:
        raise ValueError(f"无法解析 native_res: {raw!r}")
    return float(m.group(1)) * _NATIVE_RES_UNIT_DEG[m.group(2)]


def load_contract(path: str | Path) -> Contract:
    """加载并解析层映射表 YAML（字段完整性校验在 validate_contract 中聚合执行）。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"层映射表不存在: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    tiers = tuple(raw["tiers"])
    if "1deg" not in tiers:
        raise ContractViolation([("tiers", "契约档位列表缺少全集档 1deg")])

    layers = []
    for i, item in enumerate(raw.get("layers", [])):
        loc = f"layers[{i}]" if "id" not in item else f"layers[{i}] {item.get('id')}"
        missing = [f for f in _REQUIRED_FIELDS if f not in item]
        if missing:
            raise ContractViolation([(loc, f"缺少必填字段: {', '.join(missing)}")])

        dims = tuple(item.get("dims", ("lat", "lon")))
        native_res = item["native_res"]
        layers.append(
            LayerEntry(
                id=item["id"],
                dims=dims,
                source=item["source"],
                native_res=native_res,
                native_res_deg=parse_native_res_deg(native_res),
                home_tier=item["home_tier"],
                tiers=tuple(item["tiers"]),
                dtype=item["dtype"],
                unit=item["unit"],
                resampling=normalize_resampling(item["resampling"]),
                mask=item["mask"],
                visibility=item.get("visibility", "public"),
                notes=item.get("notes", ""),
            )
        )

    return Contract(
        path=path,
        version=str(raw.get("version", "")),
        status=str(raw.get("status", "")),
        tiers=tiers,
        layers=tuple(layers),
        vectors_sidecar=tuple(raw.get("vectors_sidecar", [])),
        validation_reference=tuple(raw.get("validation_reference", [])),
        not_ingested=tuple(raw.get("not_ingested", [])),
    )


def validate_contract(contract: Contract) -> None:
    """铁律前置校验：违规即聚合抛出 ContractViolation，逐条指明违规层。"""
    violations: list[tuple[str, str]] = []
    contract_tiers = set(contract.tiers)

    seen_ids: set[str] = set()
    for entry in contract.layers:
        loc = f"层 {entry.id}"
        if entry.id in seen_ids:
            violations.append((loc, "层 id 重复"))
        seen_ids.add(entry.id)

        # 档位集合合法性
        unknown = [t for t in entry.tiers if t not in contract_tiers]
        if unknown:
            violations.append((loc, f"包含契约未定义的档位: {unknown}"))
        if "1deg" not in entry.tiers:
            violations.append((loc, "档位集合缺少全集档 1deg（铁律：所有层汇入 1°）"))
        if entry.home_tier not in entry.tiers and entry.home_tier not in _HOME_TIER_SPECIAL:
            violations.append((loc, f"home_tier {entry.home_tier!r} 不在 tiers 集合内"))
        if len(set(entry.tiers)) != len(entry.tiers):
            violations.append((loc, "tiers 存在重复"))

        # 铁律：不进入细于原生分辨率的档
        if entry.native_res_deg is not None:
            for t in entry.tiers:
                if violates_iron_rule(t, entry.native_res_deg):
                    violations.append(
                        (
                            loc,
                            f"铁律违规：档位 {t}（{tier_deg(t):g}°）细于原生分辨率 "
                            f"{entry.native_res}（{entry.native_res_deg:g}°）",
                        )
                    )

        # 字段值域
        if entry.resampling not in RESAMPLING_TOKENS:
            violations.append((loc, f"resampling 归一化后非法: {entry.resampling!r}"))
        if entry.dtype not in DTYPES:
            violations.append((loc, f"dtype 非法: {entry.dtype!r}"))
        if entry.visibility not in VISIBILITIES:
            violations.append((loc, f"visibility 非法: {entry.visibility!r}"))
        if set(entry.dims) - {"lat", "lon", "depth", "time"}:
            violations.append((loc, f"dims 含未知维度: {entry.dims}"))
        elif tuple(entry.dims[-2:]) != ("lat", "lon"):
            violations.append((loc, f"dims 必须以 (lat, lon) 结尾: {entry.dims}"))

    if violations:
        raise ContractViolation(violations)
