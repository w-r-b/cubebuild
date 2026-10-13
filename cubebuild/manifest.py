"""cube_manifest: the cube root-level machine-readable layer registry (JSON, written to the store root directory).

The manifest registers only layers already built into the store; build summaries (pending layers etc.) are emitted as a separate report,
without polluting the product listing. The manifest contains no timestamps — ensuring the artifacts are deterministic and reproducible.
"""

import json
from pathlib import Path

from . import __version__
from .contract import Contract, LayerEntry, needs_validity_mask
from .grids import grid_shape, tier_deg
from .masks import validity_layer_name
from .pixel_area import EARTH_RADIUS_KM, FORMULA

MANIFEST_NAME = "cube_manifest.json"

# category_encoding/rasterization_rule/license_note/nodata_semantics：
PROVENANCE_ATTR_KEYS = (
    "doi",
    "source_url",
    "composite_rule",
    "vertical_datum",
    "derived_from",
    "derivation",
    "known_bias",
    "registration_rule",
    "crs_assignment",
    "source_file",
    "category_encoding",
    "rasterization_rule",
    "license_note",
    "nodata_semantics",
    "count_semantics",
    "product_type",
    "depth_levels",
    "datatree_node",
    "overlap_note",
    "time_levels",
    "time_semantics",
    "value_convention",
    "merge_rule",
    "unit_basis",
    "classification_rule",
    "basement_evidence",
    "usage_note",
)


def build_layer_entry(
    entry: LayerEntry,
    coverage_by_tier: dict[str, float],
    extra: dict | None = None,
) -> dict:
    """A single layer's manifest entry: dims/dtype/unit/tier set/native resolution/source/coverage, etc.

    extra: supplementary provenance fields supplied by the builder (e.g. ETOPO's doi/composite_rule).
    Only extra keys outside the whitelist are allowed — overwriting a canonical field (id/coverage/...) is an assertion failure.
    """
    item = {
        "id": entry.id,
        "dims": list(entry.dims),
        "tiers": list(entry.tiers),
        "home_tier": entry.home_tier,
        "dtype": entry.dtype,
        "unit": entry.unit,
        "resampling": entry.resampling,
        "mask": entry.mask,
        "visibility": entry.visibility,
        "source": entry.source,
        "native_res": entry.native_res,
        "coverage": {t: coverage_by_tier[t] for t in entry.tiers},
        "notes": entry.notes,
    }
    if needs_validity_mask(entry.mask):
        item["validity_mask"] = validity_layer_name(entry.id)
    if extra:
        clash = set(extra) & set(item)
        if clash:
            raise ValueError(f"extras 覆盖 canonical 字段: {sorted(clash)}")
        item.update(extra)
    return item


def build_manifest(
    contract: Contract,
    built_entries: list[LayerEntry],
    coverage: dict[tuple[str, str], float],
    extras: dict[str, dict] | None = None,
    sidecars: list[dict] | None = None,
) -> dict:
    extras = extras or {}
    return {
        "schema": "cube_manifest/1.0",
        "cube": {
            "version": "v1.0",
            "crs": "EPSG:4326",
            "container": "zarr_v3",
            "entry_point": "datatree",
        },
        "mapping": {
            "file": contract.path.name,
            "version": contract.version,
            "status": contract.status,
        },
        "pipeline": {"name": "cubebuild", "version": __version__},
        "grids": {
            t: {
                "nlat": grid_shape(t)[0],
                "nlon": grid_shape(t)[1],
                "cell_size_deg": tier_deg(t),
                "edge_aligned": "-180/-90 与 +180/+90，中心偏移半档",
            }
            for t in contract.tiers
        },
        "conventions": {
            "pixel_area": {
                "formula": FORMULA,
                "earth_radius_km": EARTH_RADIUS_KM,
                "radius_convention": "IUGG authalic (equal-area)",
                "note": "全部像元面积之和 = 4πR²（地表总面积），全球面积守恒自检",
            }
        },
        "layers": [
            build_layer_entry(
                e,
                {t: coverage[(e.id, t)] for t in e.tiers},
                extras.get(e.id),
            )
            for e in built_entries
        ],
        "sidecars": sidecars or [],
    }


def write_manifest(manifest: dict, store_dir: str | Path) -> Path:
    path = Path(store_dir) / MANIFEST_NAME
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return path


def read_manifest(store_dir: str | Path) -> dict:
    path = Path(store_dir) / MANIFEST_NAME
    if not path.exists():
        raise FileNotFoundError(f"store 缺少 {MANIFEST_NAME}: {path}")
    return json.loads(path.read_text(encoding="utf-8"))
