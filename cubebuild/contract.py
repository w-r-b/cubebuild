"""Layer-mapping YAML contract loading and iron-rule pre-validation.

The contract is the construction blueprint: the build consumes only the entries
parsed here and exercises no discretion.
Iron rule: no layer may enter a tier finer than its native resolution; 1deg is
the all-inclusive tier (all layers converge there).
"""

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from .grids import TIER_ORDER, tier_deg

DEFAULT_CONTRACT_PATH = Path(__file__).resolve().parent / "layer-mapping-v0.yaml"

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

_NATIVE_RES_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(deg|min|sec)\s*$")
_NATIVE_RES_UNIT_DEG = {"deg": 1.0, "min": 1.0 / 60.0, "sec": 1.0 / 3600.0}
_NATIVE_RES_UNCONSTRAINED = {"point", "vector-line", "vector-polygon", "exact"}

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
    """Contract validation failure: violations is a list of (location, description) pairs, each pinpointing the offending layer."""

    def __init__(self, violations: list[tuple[str, str]]):
        self.violations = violations
        lines = [f"  [{loc}] {msg}" for loc, msg in violations]
        super().__init__("契约校验失败（共 %d 项）:\n" % len(violations) + "\n".join(lines))


_IRON_RULE_EPS = 1e-12
_TIER_ORDER_SET = set(TIER_ORDER)


def violates_iron_rule(tier: str, native_res_deg: float) -> bool:
    """Single implementation of the iron-rule test: whether a tier is finer than the layer's native resolution (equal width does not count as finer).

    An unknown tier returns False (its invalidity is handled by the field-validity check).
    """
    if tier not in _TIER_ORDER_SET:
        return False
    return tier_deg(tier) < native_res_deg - _IRON_RULE_EPS


def needs_validity_mask(mask_field: str) -> bool:
    """Whether the YAML mask field requires a uint8 validity companion layer (single implementation of the test).

    Value forms (measured from the mapping table): a field counts as required iff it
    contains the ASCII token "validity" (the Slab2 form matches as well); every other
    form (exempt / globally fully covered / bundled with the elevation layer / itself
    a mask) contains no ASCII "validity". The exact field values are listed in
    ``cubebuild/layer-mapping-v0.yaml``.
    """
    return "validity" in mask_field


@dataclass(frozen=True)
class LayerEntry:
    id: str
    dims: tuple[str, ...]
    source: str
    native_res: str
    native_res_deg: float | None
    home_tier: str
    tiers: tuple[str, ...]
    dtype: str
    unit: str
    resampling: str
    mask: str
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
    """Strip the parenthetical annotation (full-width or ASCII), keeping the main token — e.g. the annotated form of a resolution token maps to its bare token."""
    token = raw
    for ch in "（(":
        idx = token.find(ch)
        if idx != -1:
            token = token[:idx]
    return token.strip()


def normalize_resampling(raw: str) -> str:
    """Normalize the resampling-kernel token by stripping its annotation (annotated form → bare token)."""
    return _strip_annotation(raw)


def parse_native_res_deg(raw: str) -> float | None:
    """Native resolution (degrees); non-raster forms such as point/vector/exact return None."""
    token = _strip_annotation(raw)
    if token in _NATIVE_RES_UNCONSTRAINED:
        return None
    m = _NATIVE_RES_RE.match(token)
    if m is None:
        raise ValueError(f"无法解析 native_res: {raw!r}")
    return float(m.group(1)) * _NATIVE_RES_UNIT_DEG[m.group(2)]


def load_contract(path: str | Path) -> Contract:
    """Load and parse the layer-mapping YAML (field-completeness validation is done in aggregate by validate_contract)."""
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
    """Iron-rule pre-validation: violations raise a ContractViolation in aggregate, each pinpointing the offending layer."""
    violations: list[tuple[str, str]] = []
    contract_tiers = set(contract.tiers)

    seen_ids: set[str] = set()
    for entry in contract.layers:
        loc = f"层 {entry.id}"
        if entry.id in seen_ids:
            violations.append((loc, "层 id 重复"))
        seen_ids.add(entry.id)

        unknown = [t for t in entry.tiers if t not in contract_tiers]
        if unknown:
            violations.append((loc, f"包含契约未定义的档位: {unknown}"))
        if "1deg" not in entry.tiers:
            violations.append((loc, "档位集合缺少全集档 1deg（铁律：所有层汇入 1°）"))
        if entry.home_tier not in entry.tiers and entry.home_tier not in _HOME_TIER_SPECIAL:
            violations.append((loc, f"home_tier {entry.home_tier!r} 不在 tiers 集合内"))
        if len(set(entry.tiers)) != len(entry.tiers):
            violations.append((loc, "tiers 存在重复"))

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
