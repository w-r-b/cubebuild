"""确定性校验：对 store 中每个数组按固定顺序计算 sha256。

用于「两次运行产物数值一致」验收：逐数组按 C 序分块读取后哈希，
两次构建的 checksum 集合完全相等即数值确定。
（覆盖任意维数组：首轴按行片读取，字节序与整展平一致。）
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import zarr

_SLAB_ROWS = 1024


def _array_checksum(arr: zarr.Array) -> str:
    """首轴行片流式哈希（1D/2D/3D/4D 通用；字节序 = C 序整展平）。"""
    h = hashlib.sha256()
    if arr.ndim == 0:
        h.update(np.ascontiguousarray(arr[...]).tobytes())
        return h.hexdigest()
    n0 = arr.shape[0]
    for i in range(0, n0, _SLAB_ROWS):
        h.update(np.ascontiguousarray(arr[i : i + _SLAB_ROWS]).tobytes())
    return h.hexdigest()


def store_checksums(store_dir: str | Path) -> dict[str, str]:
    """{数组路径: sha256}，键排序后 JSON 序列化亦确定。"""
    store_dir = Path(store_dir)
    root = zarr.open(store_dir, mode="r")

    def _walk(group, prefix):
        out = {}
        # zarr v3 Group 用 members()（(name, node) 迭代器）；v2 用 items()
        members = group.members() if hasattr(group, "members") else group.items()
        for name, item in sorted(members):
            path = f"{prefix}/{name}"
            if isinstance(item, zarr.Array):
                out[path] = _array_checksum(item)
            else:
                out.update(_walk(item, path))
        return out

    return _walk(root, "")


def checksums_equal(a: dict[str, str], b: dict[str, str]) -> bool:
    return set(a) == set(b) and all(a[k] == b[k] for k in a)


def dumps(checksums: dict[str, str]) -> str:
    return json.dumps(dict(sorted(checksums.items())), indent=2)
