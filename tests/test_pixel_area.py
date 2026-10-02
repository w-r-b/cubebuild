"""pixel_area 解析公式数值测试（外部行为：档位网格 × 公式自洽性）。"""

import numpy as np
import pytest

from cubebuild.grids import TIER_ORDER, grid_shape, tier_deg, tier_edges_lat
from cubebuild.pixel_area import EARTH_RADIUS_KM, build_pixel_area, pixel_area_rows

R = EARTH_RADIUS_KM


@pytest.mark.parametrize("tier", TIER_ORDER)
def test_row_count_matches_grid(tier):
    nlat, nlon = grid_shape(tier)
    assert pixel_area_rows(tier).shape == (nlat,)
    assert nlat * tier_deg(tier) == pytest.approx(180.0)
    assert nlon * tier_deg(tier) == pytest.approx(360.0)


@pytest.mark.parametrize("tier", TIER_ORDER)
def test_global_sum_is_sphere_area(tier):
    # 解析公式的面积守恒：全部像元之和 = 4πR²（地表总面积）
    nlat, nlon = grid_shape(tier)
    total = pixel_area_rows(tier).sum() * nlon
    target = 4.0 * np.pi * R**2
    assert abs(total - target) / target < 1e-12


@pytest.mark.parametrize("tier", TIER_ORDER)
def test_hemisphere_symmetry(tier):
    # A(-φ) = A(φ)：南北半球对称（容差=边缘浮点求值的舍入噪声）
    rows = pixel_area_rows(tier)
    np.testing.assert_allclose(rows, rows[::-1], rtol=1e-9)


@pytest.mark.parametrize("tier", TIER_ORDER)
def test_area_grows_toward_equator(tier):
    rows = pixel_area_rows(tier)
    nlat = len(rows)
    mid = nlat // 2
    # 全档 nlat 为偶数：赤道是行边缘，最大值由跨赤道的一对行取得（数学上相等）
    assert int(np.argmax(rows)) in (mid - 1, mid)
    assert np.all(np.diff(rows[:mid]) > 0)  # 严格递增至赤道行对
    assert np.all(np.diff(rows[mid:]) < 0)  # 过赤道后严格递减


def test_single_row_closed_form():
    # 与闭式公式逐值核对（6min 档任取一行；赤道是行边缘，无行横跨）
    rows = pixel_area_rows("6min")
    edges = tier_edges_lat("6min")
    i = 100
    dlon = np.deg2rad(360.0 / grid_shape("6min")[1])
    expect = R**2 * dlon * (
        np.sin(np.deg2rad(edges[i + 1])) - np.sin(np.deg2rad(edges[i]))
    )
    np.testing.assert_allclose(rows[i], expect, rtol=1e-14)


def test_build_pixel_area_array_form():
    # 构建器输出：dask 惰性、float32、行广播、坐标正确
    import dask.array

    entry_tiers = ("1deg",)
    from cubebuild.contract import LayerEntry

    entry = LayerEntry(
        id="derived__pixel_area",
        dims=("lat", "lon"),
        source="纬度解析公式",
        native_res="exact",
        native_res_deg=None,
        home_tier="逐档",
        tiers=("1deg",),
        dtype="float32",
        unit="km2",
        resampling="exact_formula",
        mask="免",
        visibility="public",
    )
    arr = build_pixel_area(entry, "1deg")
    nlat, nlon = grid_shape("1deg")
    assert arr.shape == (nlat, nlon)
    assert arr.dtype == "float32"
    assert isinstance(arr.data, dask.array.Array)
    rows = pixel_area_rows("1deg")
    vals = arr.values
    np.testing.assert_array_equal(vals[:, 0], rows.astype("float32"))
    np.testing.assert_array_equal(vals[:, -1], vals[:, 0])  # 行内恒定
    assert arr.attrs["units"] == "km2"
