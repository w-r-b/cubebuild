"""Zarr store 写入器：按档位分组的 Datatree。

结构：根节点携带立方级 attrs；每个档位一个子节点（/1deg, /30min, ...），
节点内 Dataset 坐标为该档 lat/lon 中心，变量为层 id 命名的数组。
同档层共享坐标 → 严格配准（读入即可堆叠）。
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
    """组装 Datatree：档位节点 + 3D 层组子节点。

    group_datasets：{(tier, group_name): Dataset}——深度坐标互不相同
    （289 vs 74 层）的 3D 模型层不能共存于档位节点（同一 Dataset 只能
    有一套坐标），入子节点 /{tier}/{group}；组内共享该模型 depth 坐标。
    """
    nodes: dict[str, xr.Dataset] = {"/": xr.Dataset(attrs=build_root_attrs(contract))}
    for tier in contract.tiers:
        if tier not in tier_datasets:
            continue
        ds = tier_datasets[tier]
        if len(ds.data_vars) == 0:
            # 骨架阶段也落空档位组，保持五档结构完整
            ds = xr.Dataset(attrs={"tier": tier})
        nodes[f"/{tier}"] = ds
        for (t, group), gds in sorted((group_datasets or {}).items()):
            if t == tier and len(gds.data_vars) > 0:
                gds.attrs.setdefault("tier", tier)
                nodes[f"/{tier}/{group}"] = gds
    return DataTree.from_dict(nodes)


def write_store(tree: DataTree, out_dir: str | Path) -> Path:
    """写 Zarr v3 store（覆盖式：先清空目录，保证产物干净、可复跑）。"""
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        # zarr3 的 consolidated metadata 实验性告警，与产物无关
        warnings.filterwarnings("ignore", message=".*Consolidated metadata.*")
        tree.to_zarr(out_dir, mode="w", zarr_format=3)
    return out_dir
