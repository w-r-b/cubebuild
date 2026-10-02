"""结构验证（立方出口单接缝）。

对照三方：Zarr store × cube_manifest × YAML 契约。
范围：全部契约层逐条比对（层不在 manifest / store → FAIL，
不静默跳过）+ manifest↔契约集合互检 + 1° 全集档完整性（含 3D/4D
节点与 internal 层）+ 铁律复检 + 同档层配准 + pixel_area 解析公式核对。
"""

import json
from pathlib import Path

import numpy as np
import xarray as xr

from .contract import Contract, needs_validity_mask, parse_native_res_deg, violates_iron_rule
from .grids import grid_shape, tier_centers, tier_deg
from .manifest import read_manifest
from .masks import data_missing, validity_layer_name
from .pixel_area import EARTH_RADIUS_KM, pixel_area_rows


def _tier_node(tree, tier):
    """取档位节点；不存在返回 None（getitem 支持 '/1deg' 与 '1deg' 两种路径）。"""
    try:
        return tree[tier]
    except KeyError:
        return None


def _layer_node(tree, tier: str, manifest_entry: dict):
    """取层所在节点：3D/4D 层组按 manifest datatree_node 解析
    子节点，2D 层在档位节点本节点；不存在返回 None。

    datatree_node 两种形态：绝对路径（3D 地震，如 /1deg/gladm35，
    单档层）或相对组名（4D 古高程，如 paleodem → /{tier}/paleodem，
    多档层共用组名、各档各自解析）。
    """
    raw = manifest_entry.get("datatree_node")
    path = f"/{tier}" if raw is None else (raw if raw.startswith("/") else f"/{tier}/{raw}")
    try:
        return tree[path]
    except KeyError:
        return None


def _check(checks: list, name: str, ok: bool, detail: str = "") -> None:
    checks.append({"name": name, "status": "pass" if ok else "FAIL", "detail": detail})


def validate_store(store_dir: str | Path, contract: Contract) -> dict:
    """结构验证：返回报告 dict（result=PASS/FAIL）。"""
    checks: list[dict] = []
    store_dir = Path(store_dir)

    # 1. store 可开 + 根 attrs（打开/manifest 读取失败也要出 FAIL 报告，不静默崩溃）
    try:
        tree_ctx = xr.open_datatree(store_dir, engine="zarr", chunks={})
        manifest = read_manifest(store_dir)
    except Exception as e:  # noqa: BLE001 —— 任何打开失败都须落报告
        _fail = {
            "result": "FAIL",
            "store": str(store_dir),
            "mapping": str(contract.path),
            "total_checks": 1,
            "passed": 0,
            "failed": 1,
            "checks": [{"name": "store_openable", "status": "FAIL", "detail": str(e)}],
        }
        return _fail

    with tree_ctx as tree:
        _check(checks, "store_openable", True, f"Datatree 入口打开 {store_dir}")
        _check(
            checks,
            "root_crs",
            str(tree.attrs.get("crs", "")) == "EPSG:4326",
            f"crs={tree.attrs.get('crs')}",
        )

        manifest_layers = {m["id"]: m for m in manifest["layers"]}
        _check(
            checks,
            "manifest_present",
            True,
            f"cube_manifest 登记层数 = {len(manifest_layers)}",
        )

        # manifest ↔ 契约集合互检（双向完备——缺层或多层都 FAIL）
        contract_ids = {e.id for e in contract.layers}
        missing_in_manifest = sorted(contract_ids - set(manifest_layers))
        extra_in_manifest = sorted(set(manifest_layers) - contract_ids)
        _check(
            checks,
            "manifest 与契约层集合一致",
            not missing_in_manifest and not extra_in_manifest,
            f"契约缺登记 {missing_in_manifest or '无'}；"
            f"manifest 多出 {extra_in_manifest or '无'}"
            f"（契约 {len(contract_ids)} 层 / manifest {len(manifest_layers)} 层）",
        )

        # 1° 全集档完整性：全部契约层（含 3D/4D 节点与
        # internal 层）在 1° 档齐汇——逐层解析节点并确认数组存在
        absent_1deg = []
        for entry in contract.layers:
            if entry.id not in manifest_layers:
                absent_1deg.append(f"{entry.id}（未入 manifest）")
                continue
            node = _layer_node(tree, "1deg", manifest_layers[entry.id])
            if node is None or entry.id not in node.ds.data_vars:
                absent_1deg.append(entry.id)
        _check(
            checks,
            "1° 全集档完整性（全部层含 3D/4D 节点与 internal 齐汇）",
            not absent_1deg,
            "; ".join(absent_1deg) or f"{len(contract.layers)} 层全部在 1° 档",
        )

        # 2. 逐层结构比对（manifest 条目 × store 实际 × YAML 契约）
        for entry in contract.layers:
            if entry.id not in manifest_layers:
                _check(
                    checks,
                    f"{entry.id}: manifest 登记",
                    False,
                    "层不在 cube_manifest（全量验证：缺层即 FAIL）",
                )
                continue  # 无 manifest 条目，逐档比对无从谈起
            m = manifest_layers[entry.id]

            # manifest 条目字段与 YAML 契约一致
            field_pairs = [
                ("dims", list(entry.dims), m["dims"]),
                ("dtype", entry.dtype, m["dtype"]),
                ("unit", entry.unit, m["unit"]),
                ("tiers", list(entry.tiers), m["tiers"]),
                ("home_tier", entry.home_tier, m["home_tier"]),
                ("visibility", entry.visibility, m["visibility"]),
                ("resampling", entry.resampling, m["resampling"]),
                ("native_res", entry.native_res, m["native_res"]),
                ("mask", entry.mask, m["mask"]),
            ]
            bad = [f"{k}: YAML={y!r} vs manifest={mm!r}" for k, y, mm in field_pairs if y != mm]
            _check(
                checks,
                f"{entry.id}: manifest 与 YAML 契约一致",
                not bad,
                "; ".join(bad),
            )

            # store 内逐档比对
            for tier in entry.tiers:
                loc = f"{entry.id}@{tier}"
                node = _layer_node(tree, tier, m)
                if node is None or entry.id not in node.ds.data_vars:
                    _check(checks, f"{loc}: 数组存在", False, "store 中缺失")
                    continue
                arr = node.ds[entry.id]

                _check(
                    checks,
                    f"{loc}: dtype",
                    str(arr.dtype) == entry.dtype,
                    f"store={arr.dtype} 契约={entry.dtype}",
                )
                _check(
                    checks,
                    f"{loc}: 维序",
                    arr.dims == entry.dims,
                    f"store={arr.dims} 契约={entry.dims}",
                )
                _check(
                    checks,
                    f"{loc}: 单位",
                    str(arr.attrs.get("units", "")) == entry.unit,
                    f"store={arr.attrs.get('units')!r} 契约={entry.unit!r}",
                )
                exp_shape = grid_shape(tier)
                ok_shape = arr.shape == exp_shape if len(entry.dims) == 2 else (
                    arr.shape[-2:] == exp_shape
                )
                _check(
                    checks,
                    f"{loc}: 网格形状",
                    ok_shape,
                    f"store={arr.shape} 期望 lat,lon={exp_shape}",
                )

                # 坐标配准：与档位中心坐标完全一致
                exp_lat, exp_lon = tier_centers(tier)
                _check(
                    checks,
                    f"{loc}: 坐标配准",
                    bool(
                        np.array_equal(arr.coords["lat"].values, exp_lat)
                        and np.array_equal(arr.coords["lon"].values, exp_lon)
                    ),
                    "lat/lon 中心与档位网格定义一致（edge-aligned，中心偏移半档）",
                )

                # depth 维登记（深度坐标 + 单位 + 层数与 manifest 一致）
                if "depth" in entry.dims:
                    dep = arr.coords["depth"]
                    dep_units = str(dep.attrs.get("units", ""))
                    dep_positive = str(dep.attrs.get("positive", ""))
                    n_dep = int(dep.sizes["depth"])
                    m_dep = m.get("depth_levels")
                    depth_ok = (
                        dep_units == "km"
                        and dep_positive == "down"
                        and bool(np.all(np.diff(dep.values) > 0))
                        and m_dep is not None
                        and n_dep == m_dep
                    )
                    _check(
                        checks,
                        f"{loc}: depth 维（单位 km/正向下/递增/层数 {m_dep}）",
                        depth_ok,
                        f"units={dep_units!r} positive={dep_positive!r} "
                        f"层数={n_dep} manifest={m_dep}",
                    )

                # time 维登记（Ma 标签 + 升序 + 片数与 manifest 一致）
                if "time" in entry.dims:
                    t = arr.coords["time"]
                    t_units = str(t.attrs.get("units", ""))
                    n_t = int(t.sizes["time"])
                    m_t = m.get("time_levels")
                    time_ok = (
                        t_units == "Ma"
                        and bool(np.all(np.diff(t.values.astype(np.float64)) > 0))
                        and m_t is not None
                        and n_t == m_t
                    )
                    _check(
                        checks,
                        f"{loc}: time 维（单位 Ma/升序/片数 {m_t}）",
                        time_ok,
                        f"units={t_units!r} 片数={n_t} manifest={m_t} "
                        f"首末 {t.values[0]:g}/{t.values[-1]:g} Ma",
                    )

            # 有效性掩膜伴生检查（带 validity 的层自动受检）
            if needs_validity_mask(entry.mask):
                mname = validity_layer_name(entry.id)
                for tier in entry.tiers:
                    loc = f"{entry.id}@{tier}"
                    node = _layer_node(tree, tier, m)
                    if (
                        node is None
                        or entry.id not in node.ds.data_vars
                        or mname not in node.ds.data_vars
                    ):
                        _check(checks, f"{loc}: 有效性掩膜存在", False, f"{mname} 或数据层缺失")
                        continue
                    marr = node.ds[mname]
                    arr = node.ds[entry.id]
                    exp_shape = grid_shape(tier)
                    # 3D 层：掩膜与数据层同 shape，末两维 = 档位网格
                    if len(entry.dims) == 2:
                        shape_ok = marr.shape == arr.shape == exp_shape
                    else:
                        shape_ok = marr.shape == arr.shape and arr.shape[-2:] == exp_shape
                    bad_vals = int(((marr != 0) & (marr != 1)).sum().compute())
                    # 两通道一致性（硬判据）：缺测 ⟺ 掩膜 0
                    # （浮点层缺测 = NaN；整型类别层缺测 = 0，入立方）
                    mismatch = int((data_missing(arr) != (marr == 0)).sum().compute())
                    _check(
                        checks,
                        f"{loc}: 掩膜 uint8 且值域 {{0,1}}",
                        marr.dtype == np.uint8 and shape_ok and bad_vals == 0,
                        f"dtype={marr.dtype} shape={marr.shape}（数据层 {arr.shape}）非法值 {bad_vals}",
                    )
                    _check(
                        checks,
                        f"{loc}: 两通道一致（缺测 ⟺ 掩膜 0）",
                        mismatch == 0,
                        f"不一致像元 {mismatch}（期望 0）",
                    )

            # 覆盖率字段
            cov = m.get("coverage", {})
            cov_ok = set(cov) == set(entry.tiers) and all(
                isinstance(v, (int, float)) and 0.0 <= v <= 1.0 for v in cov.values()
            )
            _check(
                checks,
                f"{entry.id}: 覆盖率登记",
                cov_ok,
                json.dumps(cov, ensure_ascii=False),
            )

        # 3. 同档层严格配准（已构建层之间同 shape/同坐标）
        #    3D 层组在档位子节点内独立检查：组内共享该模型 depth
        #    坐标；与档位节点的配准由逐层坐标配准检查覆盖（lat/lon 同网格）。
        for tier in contract.tiers:
            tnode = _tier_node(tree, tier)
            if tnode is None:
                continue
            for node, node_name in [
                (tnode, tier),
                *[(child, f"{tier}/{name}") for name, child in tnode.children.items()],
            ]:
                if len(node.ds.data_vars) == 0:
                    continue
                ref = None
                reg_ok = True
                detail = ""
                for name, arr in node.ds.data_vars.items():
                    if ref is None:
                        ref = (name, arr.shape, arr.coords["lat"].values, arr.coords["lon"].values)
                    else:
                        if (
                            arr.shape != ref[1]
                            or not np.array_equal(arr.coords["lat"].values, ref[2])
                            or not np.array_equal(arr.coords["lon"].values, ref[3])
                        ):
                            reg_ok = False
                            detail = f"{ref[0]} 与 {name} 在 {node_name} 节点不配准"
                _check(checks, f"{node_name}: 同节点层严格配准", reg_ok, detail or "同 shape/同坐标")

    # 4. 铁律复检（对 manifest 登记的档位集合；判断与构建前置校验共用同一实现）
    iron_bad = []
    for m in manifest["layers"]:
        nd = parse_native_res_deg(m["native_res"])
        if nd is None:
            continue
        for t in m["tiers"]:
            if violates_iron_rule(t, nd):
                iron_bad.append(f"{m['id']}@{t}（{tier_deg(t):g}° < 原生 {nd:g}°）")
    _check(
        checks,
        "铁律复检（manifest 档位集合）",
        not iron_bad,
        "; ".join(iron_bad) or "无层进入细于原生分辨率的档",
    )

    # 5. pixel_area 数值与解析公式一致（逐档：整列比对 + 全球面积守恒）
    pa = [e for e in contract.layers if e.id == "derived__pixel_area"]
    if pa and "derived__pixel_area" in manifest_layers:
        with xr.open_datatree(store_dir, engine="zarr", chunks={}) as tree:
            for tier in pa[0].tiers:
                arr = tree[f"/{tier}"]["derived__pixel_area"]
                expect = pixel_area_rows(tier).astype("float32")
                col = arr.values[:, 0]  # 行向量在列上恒定，取首列即可逐行核对
                exact = bool(np.array_equal(col, expect))
                _check(
                    checks,
                    f"derived__pixel_area@{tier}: 数值=解析公式",
                    exact,
                    "首列逐行比对 float32 精确一致" if exact else "存在偏差",
                )
            # 全球面积守恒（各档合计 = 4πR²，float32 累计容差）
            target = 4.0 * np.pi * EARTH_RADIUS_KM**2
            for tier in pa[0].tiers:
                arr = tree[f"/{tier}"]["derived__pixel_area"]
                total = float(arr.sum(dtype="float64").compute())
                rel = abs(total - target) / target
                _check(
                    checks,
                    f"derived__pixel_area@{tier}: 全球面积守恒",
                    rel < 1e-5,
                    f"合计 {total:.6e} km²，相对 4πR² 偏差 {rel:.2e}",
                )

    passed = sum(1 for c in checks if c["status"] == "pass")
    failed = len(checks) - passed
    return {
        "result": "PASS" if failed == 0 else "FAIL",
        "store": str(store_dir),
        "mapping": str(contract.path),
        "total_checks": len(checks),
        "passed": passed,
        "failed": failed,
        "checks": checks,
    }


def write_report(report: dict, reports_dir: str | Path,
                 stem: str = "structure-validation") -> Path:
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / f"{stem}.json"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# 结构验证报告（全量）" if stem == "structure-validation"
        else f"# 结构验证报告（{stem}）",
        "",
        f"- 结果：**{report['result']}**",
        f"- 检查项：{report['passed']}/{report['total_checks']} 通过",
        f"- store：`{report['store']}`",
        f"- 契约：`{report['mapping']}`",
        "",
        "| 检查项 | 状态 | 说明 |",
        "|---|---|---|",
    ]
    for c in report["checks"]:
        detail = (c["detail"] or "").replace("|", "\\|")
        lines.append(f"| {c['name']} | {c['status']} | {detail} |")
    (reports_dir / f"{stem}.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return json_path
