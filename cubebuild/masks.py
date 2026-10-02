"""uint8 有效性掩膜伴生机制（供所有带缺测的层复用）。

机制（缺测体系）：
- 数据层（float）缺测 = NaN；伴生掩膜层 <layer_id>__validity 与数据层
  同档生成、同 shape/同坐标，uint8：1 = 有效，0 = 缺测；
- 整型类别层：类别编码自 1 起，0 = 无类别/缺测，
  掩膜判定 = 数据 != 0（data_missing 唯一实现）；
- manifest 层条目 validity_mask 字段链接数据层 ↔ 掩膜层（manifest.py 生成）；
- 结构验证对两者做硬判据「缺测 ⟺ 掩膜 0」两通道一致性检查（validate.py）。

掩膜语义 = 数据层非缺测：浮点层 isfinite；整型类别层非零。部分覆盖
像元（矢量栅格化块内 ≥1 有效子胞即有值）记 1——与保守核「W>0 即
有值」同语义（见 kernels.conservative_overlap_mean / aggregate_mode）。
是否需要伴生层由契约 mask 字段决定（contract.needs_validity_mask 唯一判定）。
"""

import numpy as np
import xarray as xr

from .contract import LayerEntry

VALIDITY_SUFFIX = "__validity"


def data_missing(data, zero_is_missing: bool = True):
    """数据层数组的缺测判定（浮点 NaN / 整型类别 0 的唯一实现）。

    浮点层：NaN；整型层（类别编码自 1 起，如矢量栅格化类别层）：
    0 = 无类别/缺测。构建端（build_validity_mask）与验证端
    （validate.py 两通道一致性硬判据）共用本函数，保证两端口径一致。

    zero_is_missing=False：整型层 0 为真实类的形态（如陆海掩膜
    0=海，契约 mask=自身即掩膜、无 validity 伴生层）——此时整型
    层无缺测。调用方按契约 needs_validity_mask 决定（cli 覆盖率）。
    """
    if np.issubdtype(data.dtype, np.floating):
        return data.isnull()
    if zero_is_missing:
        return data == 0
    return data.isnull()      # 整型 notnull 恒真 → 无缺测


def validity_layer_name(layer_id: str) -> str:
    """数据层 id → 伴生掩膜层名（全库唯一约定）。"""
    return f"{layer_id}{VALIDITY_SUFFIX}"


def build_validity_mask(data: xr.DataArray, entry: LayerEntry) -> xr.DataArray:
    """数据层数组 → uint8 有效性伴生层（惰性：dask 输入 → dask 输出）。

    两通道一致性（缺测 ⟺ 掩膜 0）是结构验证硬判据，
    本函数是掩膜取值的唯一来源（不另行手写）。
    """
    mask = (~data_missing(data)).astype(np.uint8)
    mask.name = validity_layer_name(entry.id)
    mask.attrs = {
        "units": "1",
        "long_name": "有效性掩膜（1=有效，0=缺测）",
        "source": entry.source,
        "validity_of": entry.id,
        "visibility": entry.visibility,
    }
    return mask
