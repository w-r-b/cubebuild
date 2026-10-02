"""端到端管线测试（外部行为：出口产物对照 YAML 契约）。

用缩减档位契约（1deg/30min/6min）跑全链路，30″ 全档由验收构建承担。
"""

import json
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from cubebuild.checksum import checksums_equal, store_checksums
from cubebuild.cli import main
from cubebuild.manifest import read_manifest
from cubebuild.grids import tier_centers

# 缩减档位契约：骨架管线全链路（铁律结构同真实契约）
_CONTRACT = """
version: test
status: test
tiers: [1deg, 30min, 6min]
layers:
  - id: derived__pixel_area
    source: 纬度解析公式（逐档精确计算）
    native_res: exact
    home_tier: 逐档
    tiers: [1deg, 30min, 6min]
    dtype: float32
    unit: km2
    resampling: exact_formula（逐档精确计算，不重采样）
    mask: 免
    notes: 测试契约
"""


@pytest.fixture()
def mini_mapping(tmp_path):
    p = tmp_path / "mini-mapping.yaml"
    p.write_text(_CONTRACT, encoding="utf-8")
    return p


@pytest.fixture(autouse=True)
def _no_sidecars(monkeypatch):
    """管线测试不导出矢量侧车（起构建入口会导出注册侧车，
    其中 limw .gdb 原样传递达 1.2G——侧车导出由 test_vector 与真实构建覆盖）。"""
    import cubebuild.cli

    monkeypatch.setattr(cubebuild.cli, "SIDECARS", {})


def _run_build(tmp_path, mapping, out_name):
    out = tmp_path / out_name
    reports = tmp_path / "reports"
    rc = main(["build", "--mapping", str(mapping), "--out", str(out), "--reports", str(reports)])
    return rc, out, reports


def test_build_skeleton_e2e(tmp_path, mini_mapping):
    rc, out, reports = _run_build(tmp_path, mini_mapping, "cube.zarr")
    assert rc == 0

    # Datatree 入口按档读取
    dt = xr.open_datatree(out, engine="zarr")
    assert set(dt.children) >= {"1deg", "30min", "6min"}
    for tier in ("1deg", "30min", "6min"):
        arr = dt[f"/{tier}"]["derived__pixel_area"]
        assert arr.dtype == "float32"
        assert arr.dims == ("lat", "lon")
        exp_lat, exp_lon = tier_centers(tier)
        np.testing.assert_array_equal(arr["lat"].values, exp_lat)
        np.testing.assert_array_equal(arr["lon"].values, exp_lon)

    # store 根有 manifest，且条目字段齐备
    manifest = read_manifest(out)
    entry = manifest["layers"][0]
    assert entry["id"] == "derived__pixel_area"
    assert entry["dims"] == ["lat", "lon"]
    assert entry["dtype"] == "float32"
    assert entry["unit"] == "km2"
    assert entry["tiers"] == ["1deg", "30min", "6min"]
    assert entry["native_res"] == "exact"
    assert "逐档精确计算" in entry["source"]
    assert entry["coverage"] == {"1deg": 1.0, "30min": 1.0, "6min": 1.0}

    # 验证报告 PASS
    report = json.loads((reports / "structure-validation.json").read_text())
    assert report["result"] == "PASS"
    assert report["failed"] == 0
    # 构建摘要
    summary = json.loads((reports / "build-summary.json").read_text())
    assert summary["built"] == ["derived__pixel_area"]


def test_determinism_two_runs(tmp_path, mini_mapping):
    rc1, out1, _ = _run_build(tmp_path, mini_mapping, "run1.zarr")
    rc2, out2, _ = _run_build(tmp_path, mini_mapping, "run2.zarr")
    assert rc1 == rc2 == 0
    cs1 = store_checksums(out1)
    cs2 = store_checksums(out2)
    assert cs1 and checksums_equal(cs1, cs2)  # 两次运行产物数值一致


def test_iron_rule_blocks_build(tmp_path):
    bad = """
version: test
status: test
tiers: [1deg, 30min, 6min]
layers:
  - id: bad__layer
    source: test
    native_res: 0.5deg
    home_tier: 6min
    tiers: [1deg, 30min, 6min]
    dtype: float32
    unit: m
    resampling: none
    mask: validity
"""
    p = tmp_path / "bad-mapping.yaml"
    p.write_text(bad, encoding="utf-8")
    out = tmp_path / "bad.zarr"
    rc = main(["build", "--mapping", str(p), "--out", str(out), "--reports", str(tmp_path)])
    assert rc == 2  # 铁律前置校验失败，构建即失败
    assert not out.exists()  # 违规产物不落盘


def test_validate_subcommand(tmp_path, mini_mapping):
    rc, out, reports = _run_build(tmp_path, mini_mapping, "cube.zarr")
    assert rc == 0
    rc2 = main(
        [
            "validate",
            "--store", str(out),
            "--mapping", str(mini_mapping),
            "--reports", str(tmp_path / "reports2"),
        ]
    )
    assert rc2 == 0
    report = json.loads((tmp_path / "reports2" / "structure-validation.json").read_text())
    assert report["result"] == "PASS"
