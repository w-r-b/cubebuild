"""测试：轴向 2θ 核解析性质（边界跨越/轴向不变性/分数倍档/对消约定）+
分区读取器契约（gridline 格点/步长/五分量几何/分辨率清单）+ 合并规则
（最浅 depth/并列字典序/计数并集/区外 NaN，合成分区）+ 构建器 attrs +
真实源 e2e（六层入 store、manifest 溯源、保真 PASS、合并回报落盘）。
"""

from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

import cubebuild.slab2 as slab2
from cubebuild.kernels import conservative_overlap_circular_axial, conservative_overlap_mean

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_SLAB2 = (REPO_ROOT / "original data/stress-kinematics/slab2/Slab2Distribute_Mar2018").is_dir()
requires_slab2 = pytest.mark.skipif(not REAL_SLAB2, reason="Slab2 源未集齐")

THREE_MIN = Fraction(1, 20)


# ---------------- 轴向 2θ 核（解析性质） ----------------

def _ring_source(theta_cols, lat_edges=None, lon_res=Fraction(1, 20), valid_cols=None):
    """构造整环单行源：theta_cols 为环列 0..n-1 的轴向角，None=NaN。"""
    n = int(Fraction(360) / lon_res)
    vals = np.full((1, n), np.nan, dtype=np.float32)
    mask = np.zeros((1, n), dtype=bool)
    for c, t in theta_cols.items():
        vals[0, c] = t
        if valid_cols is None or c in valid_cols:
            mask[0, c] = True
    return vals, mask, (np.array([-0.05, 0.05]) if lat_edges is None else lat_edges)


def test_axial_constant_field_exact():
    """常数轴向角：任意面积权下循环均值恒等于该角（位级）。"""
    n = 7200
    vals = np.full((1, n), 77.5, dtype=np.float32)
    mask = np.ones((1, n), dtype=bool)
    v, W = conservative_overlap_circular_axial(
        vals, mask, np.array([-0.05, 0.05]), THREE_MIN, Fraction(1))
    i, j = 90, 100
    assert v[i, j] == np.float32(77.5) and W[i, j] > 0


def test_axial_wrap_boundary_mean():
    """跨 0/180 边界均值：{170,175,5,10} → ~1.45°（日期线界胞半权），朴素均值 90 为反面教材。"""
    vals, mask, le = _ring_source({0: 170.0, 1: 175.0, 2: 5.0, 3: 10.0})
    v, W = conservative_overlap_circular_axial(vals, mask, le, THREE_MIN, Fraction(1))
    # 环列 0 胞 [-180.025,-179.975] 跨日期线：目标 1° 列 0 内仅半权 0.025，
    # 列 1..3 全权 0.05 → 解析期望（手算）：
    c = 0.025 * np.cos(np.deg2rad(340)) + 0.05 * (
        np.cos(np.deg2rad(350)) + np.cos(np.deg2rad(10)) + np.cos(np.deg2rad(20)))
    s = 0.025 * np.sin(np.deg2rad(340)) + 0.05 * (
        np.sin(np.deg2rad(350)) + np.sin(np.deg2rad(10)) + np.sin(np.deg2rad(20)))
    expect = 0.5 * np.degrees(np.arctan2(s, c)) % 180.0
    assert v[90, 0] == pytest.approx(expect, abs=1e-5)
    assert abs(v[90, 0]) < 2.0        # 轴向正确；朴素算术均值为 90（垂直方向，错）


def test_axial_invariance_plus_180():
    """轴向不变性：θ 与 θ+180（如 33.25 与 213.25）→ 输出一致（float 末位内）。"""
    n = 7200
    va, _ = conservative_overlap_circular_axial(
        np.full((1, n), 33.25, dtype=np.float32), np.ones((1, n), dtype=bool),
        np.array([-0.05, 0.05]), THREE_MIN, Fraction(1))
    vb, _ = conservative_overlap_circular_axial(
        np.full((1, n), 213.25, dtype=np.float32), np.ones((1, n), dtype=bool),
        np.array([-0.05, 0.05]), THREE_MIN, Fraction(1))
    i, j = 90, 100
    assert va[i, j] == pytest.approx(vb[i, j], abs=1e-5)


def test_axial_nan_outside_valid_and_w_matches():
    """无有效源 → NaN；W 与保守核同权（几何完全共享）。"""
    n = 7200
    rng = np.random.default_rng(3)
    vals = rng.uniform(0, 180, size=(3, n)).astype(np.float32)
    mask = rng.random((3, n)) > 0.5
    le = np.array([9.9, 10.0, 10.1, 10.2])
    va, Wa = conservative_overlap_circular_axial(vals, mask, le, THREE_MIN, THREE_MIN)
    vm, Wm = conservative_overlap_mean(vals, mask, le, THREE_MIN, THREE_MIN)
    assert np.array_equal(Wa, Wm)
    assert np.array_equal(np.isnan(va), Wm <= 0)
    assert np.array_equal(np.isfinite(va), Wm > 0)


def test_axial_perpendicular_pair_deterministic():
    """等权正交对（θ 与 θ+90，如 {0°, 90°}）：真实轴向均值无定义（双侧角
    45/135 均可），浮点合成矢量为残差级——cos(0)+cos(180°)=0 精确相消，
    sin 残差 1.22e-16 → atan2 择 π/2 → ½=45°。IEEE + numpy deg2rad 下
    逐位确定（残差由常量函数值决定，非随机），核不崩溃不发散即受检。"""
    vals, mask, le = _ring_source({0: 0.0, 1: 90.0})
    # 3′ 源 → 3′ 目标：目标列 0 由源列 0/1 各半权（0.025/0.025）——等权
    v, W = conservative_overlap_circular_axial(vals, mask, le, THREE_MIN, THREE_MIN)
    assert W[1800, 0] > 0
    assert v[1800, 0] == pytest.approx(45.0, abs=1e-6)


def test_axial_fractional_002_source():
    """0.02 分区（分数倍 2.5）→ 3′：常数场位级直出。"""
    lon_res = Fraction(1, 50)
    n = 18000
    vals = np.full((1, n), 15.0, dtype=np.float32)
    mask = np.ones((1, n), dtype=bool)
    v, W = conservative_overlap_circular_axial(
        vals, mask, np.array([-0.01, 0.01]), lon_res, THREE_MIN)
    assert v[1800, 5] == np.float32(15.0) and W[1800, 5] > 0


def test_axial_weighted_two_value_closed_form():
    """两值加权闭式（0.025/0.05 界胞权重）：30°/80° → ½·atan2(Σw·sin2θ, Σw·cos2θ)。"""
    n = 7200
    vals = np.full((1, n), np.nan, dtype=np.float32)
    vals[0, 0], vals[0, 1] = 30.0, 80.0
    mask = np.zeros((1, n), dtype=bool)
    mask[0, 0] = mask[0, 1] = True
    v, _ = conservative_overlap_circular_axial(
        vals, mask, np.array([-0.05, 0.05]), THREE_MIN, Fraction(1, 10))
    # 目标 0.1° 列 0：源列 0（界胞半权 0.025）、列 1（全权 0.05）；列 2 半权但无效
    w0, w1 = 0.025, 0.05
    c = w0 * np.cos(np.deg2rad(60.0)) + w1 * np.cos(np.deg2rad(160.0))
    s = w0 * np.sin(np.deg2rad(60.0)) + w1 * np.sin(np.deg2rad(160.0))
    expect = 0.5 * np.degrees(np.arctan2(s, c)) % 180.0
    assert v[900, 0] == pytest.approx(expect, abs=1e-5)


# ---------------- 分区读取器（合成 grd） ----------------

def _write_grd(path: Path, x, y, z):
    import netCDF4

    with netCDF4.Dataset(path, "w", format="NETCDF4_CLASSIC") as ds:
        ds.createDimension("x", len(x))
        ds.createDimension("y", len(y))
        vx = ds.createVariable("x", "f8", ("x",))
        vy = ds.createVariable("y", "f8", ("y",))
        vz = ds.createVariable("z", "f4", ("y", "x"), fill_value=np.float32(np.nan))
        vx[:] = np.asarray(x, dtype=np.float64)
        vy[:] = np.asarray(y, dtype=np.float64)
        vz[:] = np.asarray(z, dtype=np.float32)


def _make_zone_files(tmp_path, code, x, y, comp_values, date="02.24.18"):
    """写五分量 grd；comp_values: dict comp → 标量或 (ny,nx) 数组。"""
    for comp in slab2.COMPONENTS:
        z = comp_values[comp]
        z = np.broadcast_to(np.asarray(z, dtype=np.float32), (len(y), len(x))).copy()
        _write_grd(tmp_path / f"{code}_slab2_{comp}_{date}.grd", x, y, z)
    return tmp_path


def _lattice(x0, nx, step):
    return x0 + np.arange(nx) * step


def test_loader_sign_and_axial_transform(tmp_path):
    """dep 取负（正向下）；str mod 180；dip/thk/unc 原样。"""
    x = _lattice(10.0, 5, 0.05)
    y = _lattice(20.0, 3, 0.05)
    _make_zone_files(tmp_path, "alu", x, y, {
        "dep": -100.0, "dip": 45.0, "str": 200.0, "thk": 60.0, "unc": 10.0,
    })
    zone = slab2.load_slab2_zone("alu", tmp_path)
    assert zone.lon_res == Fraction(1, 20)
    assert np.all(zone.values["dep"] == 100.0)       # -(-100)
    assert np.all(zone.values["str"] == 20.0)        # 200 mod 180
    assert np.all(zone.values["dip"] == 45.0)
    assert np.all(zone.values["thk"] == 60.0)


def test_loader_rejects_off_lattice_origin(tmp_path):
    """gridline 注册破坏：起点不在步长整数倍格点 → 拒绝。"""
    x = _lattice(10.02, 5, 0.05)
    y = _lattice(20.0, 3, 0.05)
    _make_zone_files(tmp_path, "alu", x, y, {c: 1.0 for c in slab2.COMPONENTS})
    with pytest.raises(ValueError, match="格点"):
        slab2.load_slab2_zone("alu", tmp_path)


def test_loader_rejects_nonuniform_step(tmp_path):
    x = np.array([10.0, 10.05, 10.11, 10.15])        # 步长漂移
    y = _lattice(20.0, 3, 0.05)
    _make_zone_files(tmp_path, "alu", x, y, {c: 1.0 for c in slab2.COMPONENTS})
    with pytest.raises(ValueError, match="步长"):
        slab2.load_slab2_zone("alu", tmp_path)


def test_loader_rejects_component_geometry_mismatch(tmp_path):
    x = _lattice(10.0, 5, 0.05)
    y = _lattice(20.0, 3, 0.05)
    _make_zone_files(tmp_path, "alu", x, y, {c: 1.0 for c in slab2.COMPONENTS})
    _write_grd(tmp_path / "alu_slab2_dip_02.24.18.grd", _lattice(10.0, 6, 0.05), y,
               np.ones((3, 6), dtype=np.float32))
    with pytest.raises(ValueError, match="几何"):
        slab2.load_slab2_zone("alu", tmp_path)


def test_loader_rejects_resolution_list_violation(tmp_path):
    """0.02° 网格挂在非 0.02 分区码（或反之）→ 契约清单拒绝。"""
    x = _lattice(10.0, 5, 0.02)
    y = _lattice(20.0, 3, 0.02)
    _make_zone_files(tmp_path, "alu", x, y, {c: 1.0 for c in slab2.COMPONENTS})
    with pytest.raises(ValueError, match="分辨率清单"):
        slab2.load_slab2_zone("alu", tmp_path)


def test_loader_002_zone_accepted(tmp_path):
    x = _lattice(10.0, 5, 0.02)
    y = _lattice(20.0, 3, 0.02)
    _make_zone_files(tmp_path, "mue", x, y, {c: 1.0 for c in slab2.COMPONENTS})
    zone = slab2.load_slab2_zone("mue", tmp_path)
    assert zone.lon_res == Fraction(1, 50)


# ---------------- 合并规则（合成分区，monkeypatch 分区集） ----------------

@pytest.fixture()
def two_zone_src(tmp_path):
    """两分区：alu 深（dep -300）、cal 浅（dep -100），重叠带 dep 并列无关。

    网格 0.05°，alu x∈[10,10.20]、cal x∈[10.10,10.30]，y 同 [20,20.10]：
    重叠区 x∈[10.10,10.20]；分量值各分区常数可辨。
    """
    ya = _lattice(20.0, 3, 0.05)
    x_alu = _lattice(10.0, 5, 0.05)
    x_cal = _lattice(10.10, 5, 0.05)
    _make_zone_files(tmp_path, "alu", x_alu, ya, {
        "dep": -300.0, "dip": 20.0, "str": 30.0, "thk": 50.0, "unc": 11.0})
    _make_zone_files(tmp_path, "cal", x_cal, ya, {
        "dep": -100.0, "dip": 60.0, "str": 120.0, "thk": 90.0, "unc": 22.0}, date="02.23.18")
    return tmp_path


def _merge_under(tmp_path, zones, monkeypatch):
    monkeypatch.setattr(slab2, "SLAB2_ZONES", tuple(sorted(zones)))
    monkeypatch.setattr(slab2, "_MERGE_CACHE", {})
    return slab2.merge_slab2(tmp_path)


def _pixel(lon, lat):
    j = int(round((lon + 180.0) / 0.05))
    i = int(round((lat + 90.0) / 0.05))
    return i, j


def test_merge_shallowest_depth_wins(two_zone_src, monkeypatch):
    """重叠区取 depth 最浅（cal -100 → 100 km）分区的全部五分量。"""
    m = _merge_under(two_zone_src, ("alu", "cal"), monkeypatch)
    i, j = _pixel(10.15, 20.05)                       # 重叠带（两区节点均覆盖）
    assert m["fields"]["dep"][i, j] == np.float32(100.0)
    assert m["fields"]["dip"][i, j] == np.float32(60.0)
    assert m["fields"]["str"][i, j] == np.float32(120.0)
    assert m["fields"]["thk"][i, j] == np.float32(90.0)
    assert m["fields"]["unc"][i, j] == np.float32(22.0)
    # cal 独占区（[10.25,10.30]：alu 末节点胞 ≤10.225 不可及）
    i2, j2 = _pixel(10.25, 20.05)
    assert m["fields"]["dep"][i2, j2] == np.float32(100.0)
    # alu 独占区（[10.00,10.05]：cal 首节点胞 ≥10.075 不可及）
    i3, j3 = _pixel(10.02, 20.05)
    assert m["fields"]["dep"][i3, j3] == np.float32(300.0)


def test_merge_outside_slab_nan_and_counts(two_zone_src, monkeypatch):
    """板片区外 NaN；计数 3′=2/1、粗档并集基数（非块内求和）。"""
    m = _merge_under(two_zone_src, ("alu", "cal"), monkeypatch)
    i0, j0 = _pixel(50.0, 50.0)
    assert np.isnan(m["fields"]["dep"][i0, j0])
    i, j = _pixel(10.15, 20.05)
    assert m["counts"]["3min"][i, j] == 2
    assert m["counts"]["6min"][i // 2, j // 2] >= 1
    # alu 独占 3′ 像元的 1° 计数 = 1（并集基数）；朴素块内求和会把单一
    # 分区按其覆盖的 3′ 子像元数累计（该 1° 像元内 400 子像元全为 alu
    # → 朴素和 400），与「分区数 ≤27」语义冲突——本核不取该路径
    i4, j4 = _pixel(10.02, 20.05)                     # alu 独占
    assert m["counts"]["3min"][i4, j4] == 1
    assert m["counts"]["1deg"][i4 // 20, j4 // 20] <= 2
    stats = m["stats"]
    assert stats["overlap_3min_pixels"] > 0
    assert stats["count_distribution_3min"].get(2, 0) > 0


def test_merge_tie_breaks_dictionary_order(two_zone_src, monkeypatch):
    """depth 位级并列 → 分区码字典序靠前者（alu）胜出——确定性。"""
    ya = _lattice(20.0, 3, 0.05)
    x_cal = _lattice(10.10, 5, 0.05)
    _make_zone_files(two_zone_src, "cal", x_cal, ya, {   # 覆盖同名文件（同日期）
        "dep": -300.0, "dip": 60.0, "str": 120.0, "thk": 90.0, "unc": 22.0}, date="02.23.18")
    m = _merge_under(two_zone_src, ("alu", "cal"), monkeypatch)
    i, j = _pixel(10.15, 20.05)
    assert m["fields"]["dip"][i, j] == np.float32(20.0)   # alu 的 dip
    assert m["stats"]["depth_tie_pixels"] > 0


def test_merge_002_fractional_zone(tmp_path, monkeypatch):
    """0.02 分区（mue）分数倍 2.5 → 3′：常数场位级直出，覆盖正确。"""
    x = _lattice(10.0, 25, 0.02)
    y = _lattice(20.0, 10, 0.02)
    _make_zone_files(tmp_path, "mue", x, y, {
        "dep": -150.0, "dip": 40.0, "str": 95.0, "thk": 70.0, "unc": 5.0})
    m = _merge_under(tmp_path, ("mue",), monkeypatch)
    i, j = _pixel(10.2, 20.1)
    assert m["fields"]["dep"][i, j] == np.float32(150.0)
    assert m["fields"]["str"][i, j] == np.float32(95.0)
    assert m["counts"]["3min"][i, j] == 1


def test_merge_ring_wrap_dateline(tmp_path, monkeypatch):
    """跨日期线分区（x 179.9..180.1）：嵌入环后场经回绕落 -180 邻域。"""
    x = _lattice(178.0, 9, 0.05)                      # 178..178.4，不跨线
    xw = _lattice(179.9, 5, 0.05)                     # 179.9..180.1 跨 +180
    y = _lattice(20.0, 3, 0.05)
    _make_zone_files(tmp_path, "alu", xw, y, {
        "dep": -1.0, "dip": 1.0, "str": 1.0, "thk": 1.0, "unc": 1.0})
    _make_zone_files(tmp_path, "cal", x, y, {c: 2.0 for c in slab2.COMPONENTS}, date="02.23.18")
    m = _merge_under(tmp_path, ("alu", "cal"), monkeypatch)
    # 180.05 ≡ -179.95 → 环回绕列；dep 取负后为 +1
    i, j = _pixel(-179.95, 20.05)
    assert m["fields"]["dep"][i, j] == np.float32(1.0)
    assert m["counts"]["3min"][i, j] == 1


# ---------------- 构建器 ----------------

def _entry(layer_id, visibility="public"):
    from cubebuild.contract import LayerEntry

    return LayerEntry(
        id=layer_id, dims=("lat", "lon"), source="slab2（27 俯冲带）",
        native_res="0.05deg（分区网格）", native_res_deg=0.05, home_tier="3min",
        tiers=("1deg", "30min", "6min", "3min"), dtype="float32",
        unit="km（正值向下）", resampling="conservative_area_weighted",
        mask="NaN（板片区外）+ validity", visibility=visibility,
    )


def test_builders_attrs_and_dtypes(two_zone_src, monkeypatch):
    monkeypatch.setattr(slab2, "SLAB2_SRC_DIR", two_zone_src)
    m = _merge_under(two_zone_src, ("alu", "cal"), monkeypatch)

    arr = slab2.build_slab2_data_layer(_entry("stress_kinematics__slab2_depth"), "3min")
    assert arr.dtype == np.float32 and arr.shape == (3600, 7200)
    assert arr.attrs["value_convention"].startswith("深度为海平面下距离")
    assert "最浅" in arr.attrs["merge_rule"]
    assert "重叠像元" in arr.attrs["overlap_note"]
    assert arr.attrs["doi"] == slab2.SLAB2_DOI

    coarse = slab2.build_slab2_data_layer(_entry("stress_kinematics__slab2_depth"), "1deg")
    assert coarse.shape == (180, 360)

    from cubebuild.contract import LayerEntry
    ovl = LayerEntry(
        id="stress_kinematics__slab2_overlap_mask", dims=("lat", "lon"),
        source="slab2（合并时派生）", native_res="0.05deg（分区网格）",
        native_res_deg=0.05, home_tier="3min",
        tiers=("1deg", "30min", "6min", "3min"), dtype="uint8",
        unit="count（覆盖该像元的分区数）", resampling="sum", mask="免",
        visibility="public",
    )
    om = slab2.build_slab2_overlap_mask(ovl, "6min")
    assert om.dtype == np.uint8 and om.shape == (1800, 3600)
    assert "并集基数" in om.attrs["count_semantics"]

    p = slab2.write_slab2_merge_report(two_zone_src / "reports", src_dir=two_zone_src)
    import json
    stats = json.loads(p.read_text(encoding="utf-8"))
    assert stats["overlap_3min_pixels"] == m["stats"]["overlap_3min_pixels"]


# ---------------- 真实源 e2e ----------------

_MINI = """\
version: mini-slab2-test
status: test
tiers: [1deg, 30min, 6min, 3min]
layers:
  - id: stress_kinematics__slab2_depth
    source: slab2（27 俯冲带 × dep）
    native_res: 0.05deg（分区网格）
    home_tier: 3min
    tiers: [1deg, 30min, 6min, 3min]
    dtype: float32
    unit: km（正值向下）
    resampling: conservative_area_weighted
    mask: NaN（板片区外）+ validity
    notes: 合并规则：重叠像元取 depth 最浅者所属分区的值
  - id: stress_kinematics__slab2_dip
    source: slab2（27 带 × dip）
    native_res: 0.05deg
    home_tier: 3min
    tiers: [1deg, 30min, 6min, 3min]
    dtype: float32
    unit: degree（0–90）
    resampling: conservative_area_weighted
    mask: NaN + validity
  - id: stress_kinematics__slab2_strike
    source: slab2（27 带 × str）
    native_res: 0.05deg
    home_tier: 3min
    tiers: [1deg, 30min, 6min, 3min]
    dtype: float32
    unit: degree（0–180，轴向数据）
    resampling: circular_axial（2θ 变换循环统计；走向无方向性，不用 360° 循环）
    mask: NaN + validity
  - id: stress_kinematics__slab2_thickness
    source: slab2（27 带 × thk）
    native_res: 0.05deg
    home_tier: 3min
    tiers: [1deg, 30min, 6min, 3min]
    dtype: float32
    unit: km
    resampling: conservative_area_weighted
    mask: NaN + validity
  - id: stress_kinematics__slab2_uncertainty
    source: slab2（27 带 × unc）
    native_res: 0.05deg
    home_tier: 3min
    tiers: [1deg, 30min, 6min, 3min]
    dtype: float32
    unit: km
    resampling: conservative_area_weighted
    mask: NaN + validity
  - id: stress_kinematics__slab2_overlap_mask
    source: slab2（合并时派生）
    native_res: 0.05deg
    home_tier: 3min
    tiers: [1deg, 30min, 6min, 3min]
    dtype: uint8
    unit: count（覆盖该像元的分区数）
    resampling: sum
    mask: 免
"""


@requires_slab2
def test_slab2_e2e(tmp_path, monkeypatch):
    """六层全链路：build（含合并回报）→ 结构验证 PASS → manifest（值域/合并
    规则/计数语义登记）→ 保真六层 PASS（非重叠区与分区源一致）。"""
    import cubebuild.cli
    from cubebuild.manifest import read_manifest

    monkeypatch.setattr(cubebuild.cli, "SIDECARS", {})

    mapping = tmp_path / "mini.yaml"
    mapping.write_text(_MINI, encoding="utf-8")
    out = tmp_path / "cube.zarr"
    reports = tmp_path / "reports"
    rc = cubebuild.cli.main([
        "build", "--mapping", str(mapping), "--out", str(out),
        "--reports", str(reports), "--sidecars", str(tmp_path / "sidecars"),
    ])
    assert rc == 0

    # 合并回报（重叠像元总量写入构建回报）
    import json
    merge_stats = json.loads((reports / "slab2-merge.json").read_text(encoding="utf-8"))
    assert merge_stats["overlap_3min_pixels"] == 15241
    assert merge_stats["covered_3min_pixels"] == 923250
    assert merge_stats["zones"] == 27

    manifest = read_manifest(out)
    entries = {e["id"]: e for e in manifest["layers"]}
    assert len(entries) == 6
    d = entries["stress_kinematics__slab2_depth"]
    assert d["value_convention"].startswith("深度为海平面下距离，正值向下")
    assert "最浅" in d["merge_rule"]
    assert d["validity_mask"] == "stress_kinematics__slab2_depth__validity"
    assert "15241" in d["overlap_note"]
    s = entries["stress_kinematics__slab2_strike"]
    assert "2θ" in s["value_convention"] or "2θ" in s["resampling"] or "轴向" in s["value_convention"]
    om = entries["stress_kinematics__slab2_overlap_mask"]
    assert "并集基数" in om["count_semantics"]
    assert om["resampling"] == "sum"
    # 覆盖率：depth 各档有缺测（板片区外），overlap_mask 全档 1.0（0 为真实计数）
    assert 0.0 < d["coverage"]["3min"] < 1.0
    assert om["coverage"]["3min"] == 1.0

    # 保真：六层全 PASS（分区源与合并层非重叠区一致）
    for lid in list(entries):
        report = slab2.build_slab2_fidelity_report(out, lid, ["1deg", "30min", "6min", "3min"])
        assert report["result"] == "PASS", [
            c for c in report["checks"] if c["status"] == "FAIL"]

    # 3′ strike 值域（轴向）+ 计数互证
    import xarray as xr
    with xr.open_datatree(out, engine="zarr", chunks={}) as dt:
        strike = dt["/3min"].ds["stress_kinematics__slab2_strike"].values
        fin = strike[np.isfinite(strike)]
        assert fin.min() >= 0.0 and fin.max() < 180.0
        cnt = dt["/3min"].ds["stress_kinematics__slab2_overlap_mask"].values
        dep = dt["/3min"].ds["stress_kinematics__slab2_depth"].values
        assert np.array_equal(cnt >= 1, np.isfinite(dep))
