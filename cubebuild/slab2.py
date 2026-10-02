"""Slab2 俯冲板片几何：27 分区合并全球层 + 轴向 2θ strike 核应用。

源几何（磁盘实证，135 个 grd = 27 分区 × 5 分量，逐盘核对）：

- 分区网格：GMT grd（xyz2grd COARDS netCDF），gridline 节点注册——
  节点坐标即 -R 端点、步长均匀、全部落在步长整数倍格点上；
  22 分区 0.05°，5 分区（hin/man/mue/pam/puy）0.02°（catalog 原记
  "0.05°" 不实，已勘误）。五分量同分区共享同一 x/y 几何（逐区断言）。
- 分量符号约定：dep 源为海平面下负值（构建期取负号 → 正值向下；
  mue 含 61 个 z>0 海平面以上节点，最小 −2.67 km，如实保留并注记）；
  dip ∈ [0,90]；str ∈ [0,360] 方位角（轴向数据：σ 与 σ+180° 同义，
  读取期 mod 180 入 [0,180)）；thk/unc ∈ km 正值。
- 有效性：dep 有效 ⊆ 其余四分量有效（9 分区在 fringe 有少量 dep 缺测
  而其余分量有值的节点——以 dep 为合并键，这些节点不参与合并，量级
  记入构建回报）；分区内 70–97% 节点为 NaN（板片窄带，区外即缺测）。

合并设计（合并处置全部落盘注记）：

- 合并基准 = 3′ 主档像元网格（0.05°；0.02 分区 → 3′ 为 2.5 分数倍，
  保守核精确处理）。分区网格按 Voronoi 胞解释嵌入全球经度环
  （纬度仅分区带，核支持区域源），逐分区过保守核到 3′。
- 合并规则：3′ 像元上「dep 覆盖 ≥2 分区」为重叠；重叠像元取 depth
  最浅分区全部五分量（逐分量整组取自同一分区，不跨分区混合）；
  depth 并列（float32 位级相等）取分区码字典序靠前者——确定性。
- overlap_mask：覆盖该像元的分区数（覆盖 = 该分区 dep 分量有效）。
  3′ 档 = 逐分区覆盖计数；粗档 = 所含 3′ 子像元分区覆盖的并集基数
  ——非块内朴素求和：朴素求和会把单一分区按其覆盖的子像元数重复
  累计（1° 档可达数百），与「分区数」语义（≤27）冲突；集合论上
  「分区胞域与粗像元相交 ⟺ 与其某个 3′ 子像元相交」恒成立，故
  并集基数即分区胞域语义下的精确分区数（「求和」按分区轴求和
  实现，注记于 count_semantics）。
- 保真：① store 与核重算逐位一致（同码路径，位级）；② 3′ 档单分区
  像元 store == 该分区保守场（逐位，合并正确性硬判据）；
  ③ 逐分区积分恒等式 Σ v·W = Σ_nodes v·w_node（解析节点面积，独立
  权重路径，捕获嵌入错位）；④ 粗档积分 vs 3′ 合并场解析像元面积；
  ⑤ 值域硬判据；⑥ strike 附加：独立路径逐像元循环均值抽样复算
  （标量级直算，不经 einsum/分块）+ 轴向 vs 360° 循环差异诊断
  （报告制——360° 循环对偏 N-S 走向场塌向垂直方向，即经典反面教材）。
"""

import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import dask.array as da
import numpy as np
import xarray as xr

from .contract import LayerEntry
from .etopo import SourceNotReady
from .fidelity import INTEGRAL_TOL, hist_add, quantiles_from_hist
from .grids import grid_shape, tier_centers, tier_chunks, tier_edges_lat, tier_fraction
from .kernels import conservative_overlap_circular_axial, conservative_overlap_mean
from .pixel_area import EARTH_RADIUS_KM

REPO_ROOT = Path(__file__).resolve().parents[1]
SLAB2_SRC_DIR = REPO_ROOT / "original data/stress-kinematics/slab2/Slab2Distribute_Mar2018"

SLAB2_DOI = "10.1126/science.aat4723"       # Hayes et al. 2018, Science
SOURCE_URL = "https://www.sciencebase.gov/catalog/item/5aa1b00ee4b0b1c392e86467"

COMPONENTS = ("dep", "dip", "str", "thk", "unc")
HOME_TIER = "3min"                          # 合并基准档（0.05° = 步长上界分区）
HOME_LON_RES = Fraction(1, 20)              # 3′ 合并场源经度分辨率
HOME_LON_PHASE = Fraction(1, 40)            # 3′ 合并场中心相位（0.025°）

# 27 分区码（磁盘枚举契约化：多/少一个分区即构建失败）
SLAB2_ZONES = tuple(sorted((
    "alu", "cal", "cam", "car", "cas", "cot", "hal", "hel", "him", "hin",
    "izu", "ker", "kur", "mak", "man", "mue", "pam", "phi", "png", "puy",
    "ryu", "sam", "sco", "sol", "sul", "sum", "van",
)))
# 实测 0.02° 分区（磁盘实证；其余 22 分区 0.05°）
ZONES_002 = frozenset({"hin", "man", "mue", "pam", "puy"})

LAYER_OF_COMP = {
    "dep": "stress_kinematics__slab2_depth",
    "dip": "stress_kinematics__slab2_dip",
    "str": "stress_kinematics__slab2_strike",
    "thk": "stress_kinematics__slab2_thickness",
    "unc": "stress_kinematics__slab2_uncertainty",
}
COMP_OF_LAYER = {v: k for k, v in LAYER_OF_COMP.items()}
OVERLAP_MASK_ID = "stress_kinematics__slab2_overlap_mask"

# 值域硬判据（源值域磁盘实证 + 裕量；dep 下界容纳 mue 61 个海平面以上节点）
VALUE_BOUNDS = {
    "dep": (-3.0, 750.0),    # km，正值向下；实测 [-2.67, 701.16]
    "dip": (0.0, 90.0),      # 实测 [0.009, 81.17]
    "str": (0.0, 180.0),     # 轴向 [0,180)；源 0–360 mod 180 后
    "thk": (0.0, 300.0),     # 实测 [15.0, 197.6]
    "unc": (0.0, 100.0),     # 实测 [1.10, 63.95]
}
# 分位数直方图几何 (lo, hi, bin_width)；str 轴向角分位数无意义，改报循环统计量
HIST_SPEC = {
    "dep": (-10.0, 800.0, 0.5),
    "dip": (0.0, 90.0, 0.1),
    "str": None,
    "thk": (0.0, 250.0, 0.5),
    "unc": (0.0, 100.0, 0.25),
}
QUANTILES = (1, 5, 25, 50, 75, 95, 99)

_LONG_NAME = {
    "dep": "俯冲板片顶面深度（Slab2，海平面下，正值向下）",
    "dip": "俯冲板片倾角（Slab2，0–90°）",
    "str": "俯冲板片走向（Slab2，轴向 0–180°，2θ 循环统计降采样）",
    "thk": "俯冲板片厚度（Slab2）",
    "unc": "俯冲板片深度不确定度（Slab2）",
}
_VALUE_CONVENTION = {
    "dep": "深度为海平面下距离，正值向下（源 grd z 为负值向下，构建期取负号；"
           "mue 含 61 个源 z>0 海平面以上节点，实测最小 −2.67 km，如实保留）",
    "dip": "倾角 0–90°（水平 0，垂直 90）",
    "str": "走向 0–180° 轴向数据（σ 与 σ+180° 同义；源 0–360° 方位角于读取期 "
           "mod 180）；降采样经 2θ 变换循环统计（不用 360° 循环）",
    "thk": "板片厚度，km，正值",
    "unc": "板片深度不确定度，km，正值",
}

_MERGE_RULE = (
    "27 分区合并：3′ 档像元上 dep 覆盖 ≥2 分区为重叠，重叠像元取 depth 最浅"
    "分区的全部五分量（整组取自同一分区）；depth 位级并列取分区码字典序"
    "靠前者（确定性）；合并前统计重叠像元总量入构建回报"
)
_REGISTRATION_RULE = (
    "gridline 节点注册（xyz2grd，节点坐标即 -R 端点）→ 节点代表 Voronoi 胞"
    "（±半步长，经度环接日期线）；22 分区 0.05° + 5 分区（hin/man/mue/pam/"
    "puy）0.02°；分区带嵌入全球经度环后经保守核入 3′ 合并基准档"
)
_COUNT_SEMANTICS = (
    "覆盖该像元的分区数（分区覆盖 = 该分区 dep 分量有效；dep 有效 ⊆ 其余"
    "分量， fringe 量级见构建回报）。3′ 档 = 逐分区覆盖计数之和；粗档 = "
    "所含 3′ 子像元分区覆盖的并集基数（集合论恒等：分区胞域与粗像元相交 "
    "⟺ 与某子像元相交）——非块内朴素求和，否则单一分区按子像元数重复"
    "累计，与「分区数」语义（≤27）冲突；「求和」按分区轴求和实现"
)


# ---------------- 分区读取 ----------------

@dataclass
class Slab2Zone:
    """单分区五分量网格（符号/轴向变换已施加）。"""

    code: str
    lon_res: Fraction            # 1/20（0.05°）或 1/50（0.02°）
    x: np.ndarray                # (nx,) f64 严格递增，步长整数倍格点
    y: np.ndarray                # (ny,) f64 严格递增
    values: dict                 # comp → (ny, nx) f32（dep 已取负、str 已 mod 180）
    valid: dict                  # comp → (ny, nx) bool

    @property
    def lat_edges(self) -> np.ndarray:
        """Voronoi 胞纬度边（节点 ± 半步长；分区远离极点，无钳制必要）。"""
        return _zone_lat_edges(self.y, self.lon_res)


def _zone_lat_edges(y: np.ndarray, lon_res: Fraction) -> np.ndarray:
    step = float(lon_res)     # 经纬同步步长（磁盘实证 dx=dy）
    edges = np.empty(len(y) + 1, dtype=np.float64)
    edges[:-1] = y - step / 2.0
    edges[-1] = y[-1] + step / 2.0
    return edges


def load_slab2_zone(code: str, src_dir: str | Path | None = None) -> Slab2Zone:
    """读单分区 5 分量 grd → 规范化分区（几何契约 + 符号/轴向变换）。

    契约断言：x/y 严格递增、步长均匀且为 0.05/0.02 且与分区分辨率清单一致、
    坐标落在步长整数倍格点（gridline 注册自证）、五分量几何逐位同。
    src_dir 缺省调用期解析（教训：默认参数定义期绑定常量，测试
    monkeypatch 模块属性不影响默认值）。
    """
    import netCDF4

    src_dir = Path(SLAB2_SRC_DIR if src_dir is None else src_dir)
    if not src_dir.is_dir():
        raise SourceNotReady([str(src_dir)])

    def read_component(comp: str):
        matches = sorted(src_dir.glob(f"{code}_slab2_{comp}_*.grd"))
        if len(matches) != 1:
            raise SourceNotReady([f"{src_dir}/{code}_slab2_{comp}_*.grd（{len(matches)} 个）"])
        ds = netCDF4.Dataset(matches[0])
        try:
            x = np.asarray(ds.variables["x"][:], dtype=np.float64)
            y = np.asarray(ds.variables["y"][:], dtype=np.float64)
            z = np.ma.filled(ds.variables["z"][:], np.nan).astype(np.float32)
        finally:
            ds.close()
        return x, y, z

    raw = {c: read_component(c) for c in COMPONENTS}
    x0, y0 = raw["dep"][0], raw["dep"][1]
    for comp in COMPONENTS:
        if not (np.array_equal(raw[comp][0], x0) and np.array_equal(raw[comp][1], y0)):
            raise ValueError(f"{code}/{comp}: 分量几何与 dep 不一致（五分量须共享网格）")

    # 步长与格点对齐（精确值自坐标导出；经纬同步为磁盘实证）
    dx, dy = np.diff(x0), np.diff(y0)
    if np.any(dx <= 0) or np.any(dy <= 0):
        raise ValueError(f"{code}: x/y 非严格递增")
    if not (np.allclose(dx, dx[0], atol=1e-9) and np.allclose(dy, dx[0], atol=1e-9)):
        raise ValueError(f"{code}: 步长不均匀或经纬不同步（dx {dx.min():.6g}..{dx.max():.6g}）")
    step = float(dx[0])
    lon_res = {0.05: Fraction(1, 20), 0.02: Fraction(1, 50)}.get(round(step, 2))
    if lon_res is None or abs(float(lon_res) - step) > 1e-9:
        raise ValueError(f"{code}: 步长 {step}° 非 0.05/0.02")
    if (code in ZONES_002) != (lon_res == Fraction(1, 50)):
        raise ValueError(f"{code}: 实测步长 {step}° 与契约分区分辨率清单不符")
    for arr, name in ((x0, "x"), (y0, "y")):
        if abs(arr[0] / step - round(arr[0] / step)) > 1e-6:
            raise ValueError(f"{code}: {name}[0]={arr[0]:g} 不在步长整数倍格点（gridline 注册被破坏）")
        expect = arr[0] + np.arange(len(arr)) * step
        if np.max(np.abs(arr - expect)) > 1e-6:
            raise ValueError(f"{code}: {name} 偏离规则网格")
    if y0[0] - step / 2 < -90.0 or y0[-1] + step / 2 > 90.0:
        raise ValueError(f"{code}: 分区纬度带越界 [{y0[0]:g}, {y0[-1]:g}]")

    # 符号/轴向变换：dep 取负（正值向下）；str mod 180（轴向化）
    values, valid = {}, {}
    for comp in COMPONENTS:
        z = raw[comp][2]
        if comp == "dep":
            z = -z
        elif comp == "str":
            z = np.mod(z, 180.0).astype(np.float32)
        values[comp] = z
        valid[comp] = np.isfinite(z)
    return Slab2Zone(code=code, lon_res=lon_res, x=x0, y=y0, values=values, valid=valid)


# ---------------- 分区 → 3′ 合并基准档 ----------------

def _embed_ring(zone: Slab2Zone, comp: str) -> tuple[np.ndarray, np.ndarray]:
    """分区分量嵌入全球经度环（纬度仅分区带；核支持区域源）。

    列映射 col = ((x + 180) mod 360) / lon_res（环上无 ±180 重复列），
    分区 x 跨度 < 360 → 映射列连续（可回绕日期线），断言后双段放置。
    返回 (values (ny, ring_n) f32 区外 NaN, valid (ny, ring_n) bool)。
    """
    ring_n = int(Fraction(360) / zone.lon_res)
    cols = (zone.x + 180.0) % 360.0 / float(zone.lon_res)
    cols_int = np.rint(cols).astype(np.int64)
    if np.max(np.abs(cols - cols_int)) > 1e-6:
        raise ValueError(f"{zone.code}: 列映射非整数（格点对齐被破坏）")
    if not np.array_equal((cols_int - cols_int[0]) % ring_n, np.arange(len(cols_int))):
        raise ValueError(f"{zone.code}: 映射列不连续（分区跨度 ≥ 360°？）")
    c0 = int(cols_int[0]) % ring_n
    ny = len(zone.y)
    values = np.full((ny, ring_n), np.nan, dtype=np.float32)
    valid = np.zeros((ny, ring_n), dtype=bool)
    v, m = zone.values[comp], zone.valid[comp]
    end = c0 + len(cols_int)
    if end <= ring_n:
        values[:, c0:end] = v
        valid[:, c0:end] = m
    else:                                   # 日期线回绕：分两段
        k = ring_n - c0
        values[:, c0:] = v[:, :k]
        valid[:, c0:] = m[:, :k]
        values[:, :end - ring_n] = v[:, k:]
        valid[:, :end - ring_n] = m[:, k:]
    return values, valid


def _kernel_for(comp: str):
    """分量 → 降采样核（str 轴向 2θ 循环统计，其余保守平均）。"""
    return conservative_overlap_circular_axial if comp == "str" else conservative_overlap_mean


def zone_home_field(zone: Slab2Zone, comp: str) -> tuple[np.ndarray, np.ndarray]:
    """分区分量 → 3′ 合并基准档场（连续分量保守核 / str 轴向 2θ 循环核）。

    返回 (values float32 (3600, 7200), W float64)——W 即分区覆盖权重，
    合并覆盖判定（overlap_mask / 最浅 depth 选择）用 dep 分量的 W>0。
    """
    values, valid = _embed_ring(zone, comp)
    return _kernel_for(comp)(values, valid, zone.lat_edges, zone.lon_res, tier_fraction(HOME_TIER))


# ---------------- 全球合并（缓存） ----------------

_MERGE_CACHE: dict[str, dict] = {}


def _block_any(cov: np.ndarray, factor: int) -> np.ndarray:
    """3′ 覆盖布尔场 → 粗档任覆盖布尔场（整数倍块）。"""
    m, n = cov.shape
    return cov.reshape(m // factor, factor, n // factor, factor).any(axis=(1, 3))


def merge_slab2(src_dir: str | Path | None = None) -> dict:
    """27 分区 → 全球 3′ 合并场 + 逐档分区计数 + 重叠统计（进程内缓存）。

    src_dir 缺省调用期解析 SLAB2_SRC_DIR（测试 monkeypatch 友好）。

    返回 dict：
      fields: comp → (3600, 7200) f32（板片区外 NaN；重叠取最浅 depth 分区）
      counts: tier → uint8 分区覆盖数（3′ 逐分区计数，粗档并集基数）
      stats:  重叠像元总量/分布/分区对重叠/并列次数/分区表（构建回报数据）
    """
    src_dir = SLAB2_SRC_DIR if src_dir is None else src_dir
    key = str(Path(src_dir).resolve())
    if key in _MERGE_CACHE:
        return _MERGE_CACHE[key]

    nlat, nlon = grid_shape(HOME_TIER)
    fields = {c: np.full((nlat, nlon), np.nan, dtype=np.float32) for c in COMPONENTS}
    counts = {t: np.zeros(grid_shape(t), dtype=np.uint8) for t in ("3min", "6min", "30min", "1deg")}
    best_depth = np.full((nlat, nlon), np.inf, dtype=np.float64)

    cov_indices: dict[str, np.ndarray] = {}     # 分区覆盖扁平索引（分区对重叠统计）
    pair = np.zeros((len(SLAB2_ZONES),) * 2, dtype=np.int64)
    zone_table = []
    n_ties = 0

    for zi, code in enumerate(SLAB2_ZONES):     # 已排序：并列取字典序靠前者
        zone = load_slab2_zone(code, src_dir)
        zf, w_dep = {}, None
        for comp in COMPONENTS:
            v, w = zone_home_field(zone, comp)
            zf[comp] = v
            if comp == "dep":
                w_dep = w
        coverage = w_dep > 0
        counts["3min"] += coverage
        for tier, factor in (("6min", 2), ("30min", 10), ("1deg", 20)):
            counts[tier] += _block_any(coverage, factor).astype(np.uint8)

        depth64 = zf["dep"].astype(np.float64)
        sel = coverage & (depth64 < best_depth)
        n_ties += int((coverage & (depth64 == best_depth) & ~sel).sum())
        best_depth[sel] = depth64[sel]
        for comp in COMPONENTS:
            fields[comp][sel] = zf[comp][sel]

        idx = np.flatnonzero(coverage.ravel())
        for prev, prev_idx in cov_indices.items():
            pair[zi, SLAB2_ZONES.index(prev)] = int(
                np.intersect1d(idx, prev_idx, assume_unique=True).size)
        cov_indices[code] = idx

        zone_table.append({
            "code": code,
            "lon_res_deg": float(zone.lon_res),
            "dep_valid_nodes": int(zone.valid["dep"].sum()),
            # dep 缺测而其余分量有效的节点（不入合并；dep ⊆ 其余分量，磁盘实证）
            "fringe_nodes_beyond_dep": int((zone.valid["dip"] & ~zone.valid["dep"]).sum()),
            "above_sea_level_nodes": int((zone.values["dep"] < 0).sum()),
            "covered_3min_pixels": int(idx.size),
        })

    # 合并场一致性断言：dep 有限 ⟺ 覆盖 ≥1 分区；其余分量有限 ⊆ dep 有限
    covered = counts["3min"] > 0
    if not np.array_equal(np.isfinite(fields["dep"]), covered):
        raise AssertionError("合并场 dep 有限性与分区覆盖不一致")
    for comp in COMPONENTS[1:]:
        if (np.isfinite(fields[comp]) & ~covered).any():
            raise AssertionError(f"合并场 {comp} 在无覆盖像元取有限值")

    dist = {int(k): int(v) for k, v in zip(*np.unique(counts["3min"], return_counts=True))}
    n_overlap = int((counts["3min"] >= 2).sum())
    n_cov = int(covered.sum())
    stats = {
        "zones": len(SLAB2_ZONES),
        "merge_tier": HOME_TIER,
        "covered_3min_pixels": n_cov,
        "overlap_3min_pixels": n_overlap,       # 重叠像元总量
        "overlap_share_of_coverage": n_overlap / n_cov,
        "count_distribution_3min": dist,
        "depth_tie_pixels": int(n_ties),
        "zone_pair_overlap_3min_pixels": {
            f"{SLAB2_ZONES[j]}|{SLAB2_ZONES[i]}": int(pair[i, j])
            for i in range(len(SLAB2_ZONES)) for j in range(i)
            if pair[i, j] > 0
        },
        "zone_table": zone_table,
    }
    out = {"fields": fields, "counts": counts, "stats": stats}
    _MERGE_CACHE[key] = out
    return out


def write_slab2_merge_report(reports_dir: str | Path, src_dir: str | Path | None = None) -> Path:
    """合并回报落盘（重叠像元总量写入构建回报；确定性，无时间戳）。"""
    stats = merge_slab2(src_dir)["stats"]
    path = Path(reports_dir) / "slab2-merge.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


# ---------------- 层构建 ----------------

def _home_source(comp: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """3′ 合并场作为粗档保守核源（值/有效/纬度边，中心相位 0.025°）。"""
    values = merge_slab2()["fields"][comp]
    return values, np.isfinite(values), tier_edges_lat(HOME_TIER)


def _tier_field(comp: str, tier: str) -> np.ndarray:
    """分层数组：3′ 直出合并场；粗档自合并场保守核（str 走轴向核）。"""
    if tier == HOME_TIER:
        return merge_slab2()["fields"][comp]
    src, valid, lat_edges = _home_source(comp)
    values, _W = _kernel_for(comp)(src, valid, lat_edges, HOME_LON_RES, tier_fraction(tier),
                                  lon_phase=HOME_LON_PHASE)
    return values


def _common_attrs(entry: LayerEntry) -> dict:
    m = merge_slab2()["stats"]
    return {
        "units": entry.unit,
        "source": entry.source,
        "doi": SLAB2_DOI,
        "source_url": SOURCE_URL,
        "native_res": entry.native_res,
        "resampling": entry.resampling,
        "visibility": entry.visibility,
        "product_type": "model（俯冲板片几何模型，Hayes et al. 2018）",
        "registration_rule": _REGISTRATION_RULE,
        "merge_rule": _MERGE_RULE,
        "overlap_note": (
            f"3′ 档重叠像元 {m['overlap_3min_pixels']} 个"
            f"（占分区覆盖 {m['overlap_share_of_coverage']:.3%}）；"
            "overlap_mask 记分区数；详见 slab2-merge.json"
        ),
        "source_file": "Slab2Distribute_Mar2018/*.grd（27 分区 × 5 分量）",
    }


def build_slab2_data_layer(entry: LayerEntry, tier: str) -> xr.DataArray:
    """构建五分量数据层（stress_kinematics__slab2_{depth,dip,strike,thickness,uncertainty}）。"""
    comp = COMP_OF_LAYER[entry.id]
    values = _tier_field(comp, tier)
    arr = da.from_array(values, chunks=tier_chunks(tier))
    lat, lon = tier_centers(tier)
    attrs = _common_attrs(entry)
    attrs.update({
        "long_name": _LONG_NAME[comp],
        "value_convention": _VALUE_CONVENTION[comp],
    })
    return xr.DataArray(
        arr, dims=("lat", "lon"), coords={"lat": lat, "lon": lon},
        name=entry.id, attrs=attrs,
    )


def build_slab2_overlap_mask(entry: LayerEntry, tier: str) -> xr.DataArray:
    """构建 stress_kinematics__slab2_overlap_mask（uint8 分区覆盖数，免掩膜）。"""
    values = merge_slab2()["counts"][tier]
    arr = da.from_array(values, chunks=tier_chunks(tier))
    lat, lon = tier_centers(tier)
    attrs = _common_attrs(entry)
    attrs.update({
        "long_name": "分区覆盖计数（覆盖该像元的 Slab2 分区数）",
        "count_semantics": _COUNT_SEMANTICS,
        "source_file": "Slab2Distribute_Mar2018/*.grd（27 分区 × 5 分量，计数为合并派生）",
    })
    return xr.DataArray(
        arr, dims=("lat", "lon"), coords={"lat": lat, "lon": lon},
        name=entry.id, attrs=attrs,
    )


# ---------------- 保真验证（本层自供） ----------------

SLAB2_LAYERS = {
    **{lid: {"kind": "data", "comp": c} for c, lid in LAYER_OF_COMP.items()},
    OVERLAP_MASK_ID: {"kind": "count"},
}


def _cell_areas_3min() -> np.ndarray:
    """3′ 档解析像元行面积（km²）——粗档积分独立权重路径。"""
    edges = tier_edges_lat(HOME_TIER)
    dsin = np.sin(np.deg2rad(edges[1:])) - np.sin(np.deg2rad(edges[:-1]))
    return EARTH_RADIUS_KM**2 * dsin * float(HOME_LON_RES) * np.pi / 180.0


def _node_areas(zone: Slab2Zone) -> np.ndarray:
    """分区解析节点 Voronoi 行面积（km²）——分区积分独立权重路径。"""
    edges = zone.lat_edges
    dsin = np.sin(np.deg2rad(edges[1:])) - np.sin(np.deg2rad(edges[:-1]))
    return EARTH_RADIUS_KM**2 * dsin * float(zone.lon_res) * np.pi / 180.0


def _check(checks: list, name: str, ok: bool, summary: str, **extra) -> None:
    item = {"name": name, "status": "pass" if ok else "FAIL", "summary": summary}
    item.update(extra)
    checks.append(item)


def _report(checks: list, name: str, summary: str, **extra) -> None:
    item = {"name": name, "status": "report", "summary": summary}
    item.update(extra)
    checks.append(item)


def _axial_independent_mean(theta: np.ndarray, w: np.ndarray) -> float:
    """独立路径轴向循环均值（标量级直算，不经 einsum/分块）。"""
    phi = np.deg2rad(2.0 * theta.astype(np.float64))
    return float(0.5 * np.degrees(np.arctan2(np.dot(w, np.sin(phi)), np.dot(w, np.cos(phi)))) % 180.0)


def _independent_pixel_mean(zone: Slab2Zone, comp: str, i: int, j: int,
                            res: float = 0.05) -> float | None:
    """strike 独立路径：单 3′ 像元内分区源节点标量级循环均值。

    节点选取 = Voronoi 胞（± 分区步长/2，非像元半宽——0.02 分区两者不同）
    与像元正面积相交（经度在环坐标下处理 ±180 回绕：候选列含首尾各一格
    延伸）；权重 = Δsin(纬) × Δlon(度)，与核权重相差常数因子（R²·π/180），
    在 atan2 比值中消去。
    """
    step = float(zone.lon_res)
    y0, y1 = -90.0 + i * res, -90.0 + (i + 1) * res
    rows = np.flatnonzero((zone.y + step / 2 > y0) & (zone.y - step / 2 < y1))
    if rows.size == 0:
        return None
    # 环坐标（0..360，自 -180 起）：像元 j 与节点列均折算到此框架
    p0, p1 = j * res, (j + 1) * res
    centers = (zone.x + 180.0) % 360.0          # 节点环坐标
    k_lo = int(np.floor((p0 - step / 2) / step)) - 1
    k_hi = int(np.ceil((p1 + step / 2) / step)) + 1
    wts, ths = [], []
    for r in rows:
        oy0, oy1 = max(y0, zone.y[r] - step / 2), min(y1, zone.y[r] + step / 2)
        wa = np.sin(np.deg2rad(oy1)) - np.sin(np.deg2rad(oy0))
        for k in range(k_lo, k_hi + 1):
            c_lo, c_hi = k * step - step / 2, k * step + step / 2
            ov = min(p1, c_hi) - max(p0, c_lo)
            if ov <= 0:
                continue
            # 环列 k 的分区源节点（k 可能越界一格：±360 回绕）
            col = np.flatnonzero(np.isclose(centers, k * step % 360.0, rtol=0.0, atol=1e-9))
            if col.size != 1:
                continue
            v = zone.values[comp][r, col[0]]
            if np.isfinite(v):
                wts.append(wa * ov)
                ths.append(v)
    if not wts:
        return None
    return _axial_independent_mean(np.array(ths), np.array(wts, dtype=np.float64))


def build_slab2_fidelity_report(store_dir, layer_id: str, tiers: list[str]) -> dict:
    """Slab2 层保真报告（数据层 ①–⑥ / strike 附加独立路径与循环诊断；计数层独立判据组）。"""
    spec = SLAB2_LAYERS[layer_id]
    m = merge_slab2()
    checks: list[dict] = []

    with xr.open_datatree(store_dir, engine="zarr", chunks={}) as tree:
        for tier in tiers:
            node = tree[f"/{tier}"]
            if layer_id not in node.ds.data_vars:
                raise KeyError(f"store {store_dir} 的 {tier} 档不含 {layer_id}（先 build 再 fidelity）")
            store = node.ds[layer_id].values

            if spec["kind"] == "count":
                expect = m["counts"][tier]
                n_bad = int((store != expect).sum())
                _check(checks, f"{tier}: 与合并计数重算逐位一致", n_bad == 0,
                       f"不一致像元 {n_bad}（期望 0）")
                depth = node.ds[LAYER_OF_COMP["dep"]].values
                mism = int(((store >= 1) != np.isfinite(depth)).sum())
                _check(checks, f"{tier}: 覆盖互证（count≥1 ⟺ depth 层有限，跨层硬判据）",
                       mism == 0, f"不一致 {mism} 像元（期望 0）")
                _check(checks, f"{tier}: 值域 [0, {len(SLAB2_ZONES)}]",
                       bool(store.max() <= len(SLAB2_ZONES)),
                       f"max={int(store.max())} / 覆盖占比 {float((store > 0).mean()):.4f}")
                _report(checks, f"{tier}: 覆盖率（报告制）",
                        f"count≥1 占比 {float((store >= 1).mean()):.4f}；"
                        f"重叠（≥2）占比 {float((store >= 2).mean()):.4f}")
                continue

            comp = spec["comp"]
            # ① 同码重算逐位一致 + 有效通道一致性（NaN ⟺ W=0 / 覆盖）
            if tier == HOME_TIER:
                expect = m["fields"][comp]
                ok_mask = m["counts"]["3min"] > 0
                W = None
            else:
                src, valid, lat_edges = _home_source(comp)
                expect, W = _kernel_for(comp)(src, valid, lat_edges, HOME_LON_RES,
                                             tier_fraction(tier), lon_phase=HOME_LON_PHASE)
                ok_mask = W > 0
            eq = (store == expect) | (np.isnan(store) & np.isnan(expect))
            _check(checks, f"{tier}: 与源核重算逐位一致（float32 位级）",
                   int((~eq).sum()) == 0, f"不一致像元 {int((~eq).sum())}（期望 0）")
            mism = int((np.isfinite(store) != ok_mask).sum())
            _check(checks, f"{tier}: 有效位与核 W 复算一致（逐位）", mism == 0,
                   f"不一致 {mism} 像元（期望 0）")

            # ② 值域硬判据
            lo, hi = VALUE_BOUNDS[comp]
            fin = store[np.isfinite(store)]
            _check(checks, f"{tier}: 值域 [{lo:g}, {hi:g}]{'' if comp != 'dep' else '（dep 下界容纳海平面以上 61 节点）'}",
                   bool(fin.size and fin.min() >= lo and fin.max() <= hi),
                   f"min {fin.min() if fin.size else float('nan'):.4g} / max {fin.max() if fin.size else float('nan'):.4g}")

            # ③ 粗档积分守恒（vs 3′ 合并场解析像元面积，独立权重路径）
            #    str 豁免：角度无积分不变量（循环统计不守恒线性和），其粗档
            #    正确性由 ① 位级重算 + 3′ 分区一致性 + 单元解析用例覆盖
            if tier != HOME_TIER and comp != "str":
                w = _cell_areas_3min()
                src_vals = m["fields"][comp]
                src_sum = float(np.dot(
                    np.where(np.isfinite(src_vals), src_vals, 0.0).astype(np.float64).sum(axis=1), w))
                lhs = float(np.dot(store[ok_mask].astype(np.float64), W[ok_mask]))
                rel = abs(lhs - src_sum) / abs(src_sum)
                _check(checks, f"{tier}: 积分守恒（相对源偏差 < 0.1%）", rel < INTEGRAL_TOL,
                       f"相对偏差 {rel:.3e}", rel_dev=rel)

            # ⑤ 分位数/循环统计量（报告制；粗档源 = 3′ 合并场，3′ 源 = 分区节点，见下）
            hist_spec = HIST_SPEC[comp]
            if hist_spec is not None:
                if tier != HOME_TIER:
                    hlo, hhi, bw = hist_spec
                    src_hist = hist_add(np.zeros(int(round((hhi - hlo) / bw)), dtype=np.uint64),
                                       src_vals[np.isfinite(src_vals)], lo=hlo, hi=hhi, bin_width=bw)
                    _quantile_report(checks, tier, comp, src_hist, fin, hist_spec)
                # 3′ 档分位数在分区循环后对节点源报（源直方图在分区循环中累积）
            else:
                phi = np.deg2rad(2.0 * fin.astype(np.float64))
                r_bar = float(np.hypot(np.cos(phi).mean(), np.sin(phi).mean()))
                mean_ax = float(0.5 * np.degrees(np.arctan2(np.sin(phi).mean(), np.cos(phi).mean())) % 180.0)
                _report(checks, f"{tier}: 循环统计量（轴向均值 / R̄，报告制）",
                        f"均值 {mean_ax:.2f}° / R̄ {r_bar:.4f} / 覆盖 {fin.size / store.size:.4f}")

            _report(checks, f"{tier}: 覆盖率（报告制）",
                    f"档 {float(ok_mask.mean()):.4f} vs 3′ 合并场 {float((m['counts']['3min'] > 0).mean()):.4f}")

        # ④ 3′ 分区源一致性（非重叠区与分区源一致）+ 分区积分 + strike 独立路径
        if spec["kind"] == "data":
            comp = spec["comp"]
            store = tree[f"/{HOME_TIER}"].ds[layer_id].values
            count = m["counts"]["3min"]
            rng = np.random.default_rng(0)
            zone_checked = zone_pixels = 0
            hist_spec = HIST_SPEC[comp]
            src_hist = (np.zeros(int(round((hist_spec[1] - hist_spec[0]) / hist_spec[2])), dtype=np.uint64)
                        if hist_spec is not None else None)
            strike_devs: list[float] = []
            strike_sampled = 0

            for code in SLAB2_ZONES:
                zone = load_slab2_zone(code)
                v, W = zone_home_field(zone, comp)
                W_dep = W if comp == "dep" else zone_home_field(zone, "dep")[1]
                sel = (W_dep > 0) & (count == 1)    # 恰被该分区覆盖 → 合并值须逐位等于分区场
                if not sel.any():
                    continue
                zone_checked += 1
                zone_pixels += int(sel.sum())
                n_bad = int((store[sel] != v[sel]).sum())
                _check(checks, f"3′ 分区一致性 {code}（单覆盖像元逐位）",
                       n_bad == 0, f"{code}: 不一致 {n_bad} / {int(sel.sum())} 像元")

                if comp != "str":
                    w_node = _node_areas(zone)
                    src_sum = float(np.dot(
                        np.where(zone.valid[comp], zone.values[comp], 0.0).astype(np.float64).sum(axis=1),
                        w_node))
                    lhs = float(np.sum(np.where(W > 0, v, 0.0).astype(np.float64) * W))
                    rel = abs(lhs - src_sum) / abs(src_sum)
                    _check(checks, f"3′ 分区积分 {code}（Σ v·W = Σ v·w_node，独立权重路径）",
                           rel < INTEGRAL_TOL, f"相对偏差 {rel:.3e}", rel_dev=rel)
                    if hist_spec is not None:
                        hlo, hhi, bw = hist_spec
                        src_hist = hist_add(src_hist, zone.values[comp][zone.valid[comp]],
                                            lo=hlo, hi=hhi, bin_width=bw)
                else:
                    # strike 独立路径：抽样单覆盖像元标量级循环均值（不经 einsum/分块）
                    sel_idx = np.flatnonzero(sel.ravel())
                    take = rng.choice(sel_idx, size=min(16, sel_idx.size), replace=False)
                    devs = []
                    for flat in take:
                        i, j = divmod(int(flat), store.shape[1])
                        indep = _independent_pixel_mean(zone, comp, i, j)
                        if indep is None:
                            continue
                        devs.append(abs((indep - float(store[i, j]) + 90.0) % 180.0 - 90.0))
                    strike_devs.extend(devs)
                    strike_sampled += len(devs)
                    _check(checks, f"3′ strike 独立路径抽样 {code}（标量级循环均值）",
                           bool(devs) and max(devs) < 1e-4,
                           f"最大角偏差 {max(devs) if devs else float('nan'):.3e}°（{len(devs)} 像元）")

            _report(checks, "3′ 分区一致性覆盖说明（报告制）",
                    f"{zone_checked}/{len(SLAB2_ZONES)} 分区含单覆盖像元，共 {zone_pixels} 像元受检")
            if strike_devs:
                _check(checks, "3′ strike 独立路径汇总（全部分区抽样像元）",
                       max(strike_devs) < 1e-4,
                       f"最大角偏差 {max(strike_devs):.3e}°（{strike_sampled} 像元）")
            if hist_spec is not None:
                fin0 = store[np.isfinite(store)]
                _quantile_report(checks, HOME_TIER, comp, src_hist, fin0, hist_spec,
                                 note="（源 = 分区节点直方图，显示保守重采样压缩）")

        # strike 附加：轴向 vs 360° 循环差异诊断（正确性检查，报告制）
        if layer_id == LAYER_OF_COMP["str"]:
            for tier in tiers:
                fin = tree[f"/{tier}"].ds[layer_id].values
                fin = fin[np.isfinite(fin)].astype(np.float64)
                phi2 = np.deg2rad(2.0 * fin)
                mean_ax = 0.5 * np.degrees(np.arctan2(np.sin(phi2).mean(), np.cos(phi2).mean())) % 180.0
                mean_360 = np.degrees(np.arctan2(np.sin(np.deg2rad(fin)).mean(),
                                                 np.cos(np.deg2rad(fin)).mean())) % 360.0
                ns_band = float(((fin < 15.0) | (fin >= 165.0)).mean())
                _report(checks, f"{tier}: 轴向 vs 360° 循环诊断（报告制）",
                        f"轴向全局均值 {mean_ax:.2f}° / 360° 循环全局均值 {mean_360:.2f}° / "
                        f"N-S 走向带（0/180 邻域 ±15°）占比 {ns_band:.3%}——"
                        "360° 循环对偏 N-S 场合成矢量对消后塌向垂直方向（反面教材），"
                        "轴向 2θ 为定稿约定")

    n_fail = sum(1 for c in checks if c["status"] == "FAIL")
    return {
        "layer": layer_id,
        "store": str(store_dir),
        "integral_tolerance": INTEGRAL_TOL,
        "result": "FAIL" if n_fail else "PASS",
        "n_checks": len(checks),
        "checks": checks,
    }


def _quantile_report(checks: list, tier: str, comp: str, src_hist: np.ndarray,
                     tier_values: np.ndarray, hist_spec: tuple, note: str = "") -> None:
    hlo, hhi, bw = hist_spec
    tier_hist = hist_add(np.zeros(int(round((hhi - hlo) / bw)), dtype=np.uint64),
                         tier_values, lo=hlo, hi=hhi, bin_width=bw)
    sq = quantiles_from_hist(src_hist, QUANTILES, lo=hlo, bin_width=bw)
    tq = quantiles_from_hist(tier_hist, QUANTILES, lo=hlo, bin_width=bw)
    _report(checks, f"{tier}: 分位数偏移（报告制）{note}",
            "; ".join(f"q{q:g}:{tq[f'q{q:g}'] - sq[f'q{q:g}']:+.2f}" for q in QUANTILES),
            shift={f"q{q:g}": tq[f"q{q:g}"] - sq[f"q{q:g}"] for q in QUANTILES})
