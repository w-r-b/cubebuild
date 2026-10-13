"""Deterministic verification: compute sha256 for every array in the store in a fixed order.

Used for the "two runs produce numerically identical artifacts" acceptance test:
each array is read in C-order chunks and then hashed; the two builds are
numerically deterministic iff their checksum sets are exactly equal.
(Covers arrays of any dimensionality: the first axis is read as row slabs, so the
byte order matches a full C-order flatten.)
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import zarr

_SLAB_ROWS = 1024


def _array_checksum(arr: zarr.Array) -> str:
    """Streaming hash over first-axis row slabs (works for 1D/2D/3D/4D; byte order = full C-order flatten)."""
    h = hashlib.sha256()
    if arr.ndim == 0:
        h.update(np.ascontiguousarray(arr[...]).tobytes())
        return h.hexdigest()
    n0 = arr.shape[0]
    for i in range(0, n0, _SLAB_ROWS):
        h.update(np.ascontiguousarray(arr[i : i + _SLAB_ROWS]).tobytes())
    return h.hexdigest()


def store_checksums(store_dir: str | Path) -> dict[str, str]:
    """{array path: sha256}; JSON serialization is also deterministic once keys are sorted."""
    store_dir = Path(store_dir)
    root = zarr.open(store_dir, mode="r")

    def _walk(group, prefix):
        out = {}
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
