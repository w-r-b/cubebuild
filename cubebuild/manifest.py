"""cube_manifest：立方根级机器可读层注册表（JSON，写在 store 根目录）。

manifest 只登记已构建进 store 的层；构建摘要（pending 层等）另出报告，
不污染产品清单。manifest 不含时间戳——保证产物确定性可复跑。
"""

import json
from pathlib import Path

from . import __version__
from .contract import Contract, LayerEntry, needs_validity_mask
from .grids import grid_shape, tier_deg
from .masks import validity_layer_name
from .pixel_area import EARTH_RADIUS_KM, FORMULA

MANIFEST_NAME = "cube_manifest.json"

# 构建器可补充的溯源字段白名单（attrs → manifest 条目）。
# 归属本模块：新字段在此登记一次，cli 只透传，不逐字段点名。
# derived_from/derivation/known_bias：派生层登记——来源（高程层 id
# 或源数据集复合描述，如 landsea 冰面口径）、确定性派生方法（含口径语义）、
# 已知偏差（通用字段；landsea 冰区注记已随冰面口径修复退役）。
# registration_rule/crs_assignment/source_file：源网格注册
# 解释（WGM 节点→Voronoi 胞 / EMAG 0..360 卷绕）与 CRS 显式指定。
# category_encoding/rasterization_rule/license_note/nodata_semantics：
# 矢量通道——类别编码表（类别值 → 语义）、栅格化/聚合语义、
# 许可依据、整型类别层 0 = 无类别约定。
# count_semantics：点密度通道——计数基准与守恒语义（WSM 可定位
#   行 100672 / GSRM 三数与记录行基准）。
# composite_rule/license_note：热流——合并去重规则（GHFDB 为主 +
#   NGHF 补 0.1° 像元外点）与双源许可状态注记。
# product_type：层产品性质（观测 vs 模型；HFgrid14 标 model，
#   与密度层互补非冗余）。
# depth_levels/datatree_node/overlap_note：3D 地震通道——深度层数
#   （结构验证对照源契约）、store 内 Datatree 节点路径（深度坐标互不相同
#   的 3D 模型层入档位子节点）、两模型上地幔重叠段有意保留注记。
# time_levels/time_semantics：4D 古高程通道——时间片数（结构验证
#   对照）与 time 轴语义（Ma 标签来源与官方整数惯例对齐注记）。
# source_url：无 DOI 的源以官方资源页 URL 登记（PaleoDEM
#   无正式 DOI，earthbyte 资源页为唯一权威出处）。
# value_convention/merge_rule：Slab2 通道——分量值域与符号约定
#   （depth 正值向下 / dip 0–90 / strike 轴向 0–180）与 27 分区合并
#   规则（重叠取最浅 depth，manifest 登记处）。
# unit_basis：单位确认依据（源文件头/README/论文的哪一行
# 自证了单位；GSRM 应变四层首用，GST1 等"以源文件确认"的层后续同款）。
# classification_rule：陆域类别组——源字面值 → 类别码的映射规则
#   （GUM DD 厚度分级五档分箱/代表值/单位假设；GLiM nd 并入 0）。
# basement_evidence：岩石圈栅格组——gemma_basement_depth 语义实证
#   （crust_top = 结晶基底面的值域/层序/论文三重依据）。
# usage_note：层的配对使用说明（moho_err 与 moho 的 ML 加权
#   配对），非溯源证据。
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
    """单层 manifest 条目：维度/dtype/单位/档位集合/原生分辨率/来源/覆盖率等。

    extra：构建器提供的补充溯源字段（如 ETOPO 的 doi/composite_rule）。
    只允许白名单外的补充键——覆盖 canonical 字段（id/coverage/...）即断言失败。
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
        # 数据层 ↔ 掩膜层链接：伴生 uint8 层名，store 内同档存在
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
    coverage: dict[tuple[str, str], float],  # (layer_id, tier) -> 非缺测像元占比
    extras: dict[str, dict] | None = None,   # layer_id → 补充溯源字段
    sidecars: list[dict] | None = None,      # 已导出侧车的 manifest 记录
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
