"""Zarr store writer: a Datatree grouped by tier.

Structure: the root node carries cube-level attrs; each tier is a child node
(/1deg, /30min, ...),
the Dataset coordinates within a node are that tier's lat/lon centers, and the
variables are arrays named by layer id.
Same-tier layers share coordinates → strict registration (stackable upon
reading).
"""

import shutil
import warnings
from pathlib import Path

import xarray as xr
from xarray import DataTree

from . import __version__
from .contract import Contract


def build_root_attrs(contract: Contract) -> dict:
    return {
        "title": "地质数据立方 v1.0（构建中骨架）",
        "cube_version": "v1.0",
        "crs": "EPSG:4326",
        "entry_point": "datatree",
        "tiers": "/".join(contract.tiers),
        "mapping_file": contract.path.name,
        "mapping_version": contract.version,
        "mapping_status": contract.status,
        "manifest": "cube_manifest.json",
        "pipeline": f"cubebuild {__version__}",
        "missing_value_convention": (
            "浮点层缺测 = NaN；整型类别层缺测 = 0（类别编码自 1 起）；"
            "带 validity 的层伴生 <layer_id>__validity uint8 掩膜"
            "（1=有效，0=缺测），manifest validity_mask 字段链接"
        ),
    }


def assemble_datatree(
    contract: Contract,
    tier_datasets: dict[str, xr.Dataset],
    group_datasets: dict[tuple[str, str], xr.Dataset] | None = None,
) -> DataTree:
    """Assemble the Datatree: tier nodes + 3D layer-group child nodes.

    group_datasets: {(tier, group_name): Dataset}—3D model layers with mutually
    different depth coordinates
    (289 vs 74 layers) cannot coexist in the tier node (the same Dataset can
    have only
    one set of coordinates), so they go into a child node /{tier}/{group}; the
    group shares that model's depth coordinate.
    """
    nodes: dict[str, xr.Dataset] = {"/": xr.Dataset(attrs=build_root_attrs(contract))}
    for tier in contract.tiers:
        if tier not in tier_datasets:
            continue
        ds = tier_datasets[tier]
        if len(ds.data_vars) == 0:
            ds = xr.Dataset(attrs={"tier": tier})
        nodes[f"/{tier}"] = ds
        for (t, group), gds in sorted((group_datasets or {}).items()):
            if t == tier and len(gds.data_vars) > 0:
                gds.attrs.setdefault("tier", tier)
                nodes[f"/{tier}/{group}"] = gds
    return DataTree.from_dict(nodes)


def write_store(tree: DataTree, out_dir: str | Path) -> Path:
    """Write a Zarr v3 store (overwrite-style: clear the directory first,
    ensuring clean and rerunnable products)."""
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*Consolidated metadata.*")
        tree.to_zarr(out_dir, mode="w", zarr_format=3)
    return out_dir
