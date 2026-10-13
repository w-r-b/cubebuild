"""GSRM strain-field channel: v2.2 0.1° global strain grid → four layers, three tiers.

stress_kinematics__gsrm_exx / eyy / exy / vorticity (all internal, primary tier 6min,
1°/30′/6′ conservative-kernel downsampling + validity-mask sidecar).

Source contract (disk-verified; v2.2/GSRM_strain.txt = extracted GSRM_strain.txt.Z,
554872103 B, 6305376 lines):

- Structure: 25-line header (Version 2.2 (2015-12); GEM Foundation
  CC-BY-NC-SA 3.0) + 6300000 complete grid data rows + a 5351-line trailing off-grid block.
- Grid: pixel registration 0.1° — lat -87.45..87.45 (1750 ascending rows) × lon
  -179.95..179.95 (3600 columns), row-major (lat outer loop, lon inner loop), complete
  coverage without duplicates (row-by-row vs. index arithmetic, deviation ≤ 6e-14 = 1 ulp level). Center
  phase 0.05 = source cell edge aligned to -180/±87.5 (conservative kernel lon_phase = 1/20, 6′ tier
  identity mapping). The polar cap |lat|>87.5 is naturally absent from the source grid → each tier's polar rows are NaN, validity
  mask 0 (coverage gap preserved as-is, no extrapolation). All values non-missing (strict
  numeric-token scan of 11 columns finds 0 non-numeric; no NaN/999-type sentinels).
- Columns: lat long exx eyy exy vorticity RL-NLC LL-NLC e1 e2 azi_e1. This group keeps
  the first four columns; e1/e2/azi_e1 are excluded (eigen-decomposition is a deterministic algebraic operation, downstream can
  compute it from the three components); neither the source nor the README gives a semantic definition for RL-NLC/LL-NLC,
  so the v1 mapping does not include them.
- Units: README self-attests "GSRM_strain.txt.Z … units are 1e-9/yr" (the file header has no
  unit line) → exx/eyy/exy = nano-strain/yr, vorticity = nano-rad/yr
  (unit_basis recorded in manifest). Value ranges (disk-verified): exx -12142.4..11638.6 /
  eyy -12146.2..14150.9 / exy -17129.9..10840.5 / vorticity
  -895.326..1210.912.
- Trailing off-grid block (physical rows 6300026..6305376, 5351 rows): 3 zero rows + 1070 index rows
  (field1 = 1.000..1070.000, lon = 0.000, exx = -87.5 constant) each accompanied by 4 zero rows
  (2 zero rows after the last index row). Index rows 801..1070 carry self-consistent strain-tensor values (e1/e2/azi_e1
  agree with the eigen-decomposition of exx/eyy/exy, already verified) but all coordinates are off-grid (integer lat and
  0.000 at 0.05° from the nearest grid center). The v2.1 file of the same name lacks this block (6300025 rows intact) and
  the official v2.2 directory has no README/changelog — judged as a v2.2 build artifact, excluded from the
  cube; the exclusion is contractualized at read time (three assertions: total row count / grid completeness / trailing all-off-grid;
  a source change fails explicitly).

Visibility: internal (file header GEM Foundation CC-BY-NC-SA 3.0 (NC+SA
clauses), does not meet open-license requirements, excluded from the public version; the paper recommends readers combine it themselves).
"""

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import dask.array as da
import numpy as np
import xarray as xr

from .contract import LayerEntry
from .fidelity import INTEGRAL_TOL, fidelity_checks, hist_add
from .gravmag import RasterSource, gravmag_source_stats, hist_bins
from .grids import tier_centers, tier_chunks, tier_fraction
from .kernels import conservative_overlap_mean
from .pixel_area import EARTH_RADIUS_KM
from .points import GSRM_DOI, GSRM_SRC_DIR

GSRM_V22_DIR = GSRM_SRC_DIR / "v2.2"
STRAIN_FILE = "GSRM_strain.txt"

STRAIN_LAYERS = {
    "stress_kinematics__gsrm_exx": {
        "field": "exx", "hist_spec": (-12500.0, 12500.0, 1.0), "unit": "nano-strain/yr",
    },
    "stress_kinematics__gsrm_eyy": {
        "field": "eyy", "hist_spec": (-13000.0, 15000.0, 1.0), "unit": "nano-strain/yr",
    },
    "stress_kinematics__gsrm_exy": {
        "field": "exy", "hist_spec": (-17500.0, 12500.0, 1.0), "unit": "nano-strain/yr",
    },
    "stress_kinematics__gsrm_vorticity": {
        "field": "vorticity", "hist_spec": (-1000.0, 1500.0, 1.0), "unit": "nano-rad/yr",
    },
}

STRAIN_FIELD_OF = {lid: spec["field"] for lid, spec in STRAIN_LAYERS.items()}

HEADER_LINES = 25
GRID_NLAT, GRID_NLON = 1750, 3600        # lat -87.45..87.45 × lon -179.95..179.95
GRID_ROWS = GRID_NLAT * GRID_NLON
TRAILING_ROWS = 5351

SRC_TENTH = Fraction(1, 10)
LON_PHASE = Fraction(1, 20)
LAT_SOUTH_K = 25

_COORD_TOL = 1e-6

UNIT_BASIS = (
    "README 自证 'GSRM_strain.txt.Z - strain rate results at global 0.1deg "
    "grid. units are 1e-9/yr'（文件头无单位行）——exx/eyy/exy = nano-strain/yr"
    "（应变无量纲，1e-9/yr 直读）；vorticity = nano-rad/yr【推断：README 整文件"
    "1e-9/yr 概称未点名 rad，依据 = 涡度角量纲 × 1e-9/yr + 值域 ±1211 与"
    "变形带旋转率 nrad/yr 量级自洽（若为 1e-9 deg/yr 则小 4–5 个量级）】"
)

VALUE_RANGES = {
    "exx": (-12142.4, 11638.6), "eyy": (-12146.2, 14150.9),
    "exy": (-17129.9, 10840.5), "vorticity": (-895.326, 1210.912),
}

_LICENSE_NOTE = (
    "GSRM 组不纳入公开版：文件头 GEM Foundation CC-BY-NC-SA 3.0（NC+SA "
    "条款不符合开放许可要求），论文建议读者自行联用"
)

_REGISTRATION_RULE = (
    "pixel 注册（0.1° 中心 -87.45..87.45 / -179.95..179.95，0.05 相位）＝"
    "源胞边对齐 -180/±87.5（保守核 lon_phase = 0.05，6′ 档恒等映射）；"
    "行主序 lat 升序外循环；尾部 5351 行离格块（v2.2 构建残留：3 零行 + "
    "1070 索引行各随 4 零行，坐标距最近网格中心 0.05°）剔除不入立方，"
    "剔除量读取期契约化；极冠 |lat|>87.5 源网格天然不含，极区行 NaN 保留"
)

_LONG_NAMES = {
    "exx": "东西向正应变率（e 轴向东；张量对角分量）",
    "eyy": "南北向正应变率（n 轴向北；张量对角分量）",
    "exy": "剪切应变率（张量非对角分量；工程剪应变率 = 2·exy——尾部块特征分解验算自洽）",
    "vorticity": "垂向涡度率（绕垂直轴旋转率，正 = 逆时针；刚性板块区依参考系显示）",
}


@dataclass
class StrainSource:
    """Normalized strain source: fields four components (1750, 3600) float32, row 0 = -87.5 side.

    raster(field) reuses RasterSource views for the conservative kernel and source statistics (shared geometry).
    """

    fields: dict[str, np.ndarray]
    lat_edges: np.ndarray               # (1751,) -87.5..87.5
    lon_res: Fraction
    lon_phase: Fraction
    n_trailing: int

    def raster(self, field: str) -> RasterSource:
        v = self.fields[field]
        return RasterSource(
            values=v,
            valid=np.isfinite(v),
            lat_edges=self.lat_edges,
            lon_res=self.lon_res,
            lon_phase=self.lon_phase,
        )



def load_gsrm_strain(src_dir: str | Path = GSRM_V22_DIR) -> StrainSource:
    """Read v2.2/GSRM_strain.txt → StrainSource (trailing off-grid block excluded).

    Contract (three assertions; a source change fails explicitly):
    ① total data rows = 6305351 (25 header rows skipped);
    ② first 6300000 rows = complete regular grid: row i coordinates == (lat[i//3600], lon[i%3600])
       index arithmetic (tolerance 1e-6, absorbs only 1 ulp) — completeness and row-major order together;
    ③ trailing 5351 rows all off-grid (lat/lon > 1e-6 from the nearest grid center) — if a source update
       moves a valid grid point into the trailing block, this assertion catches the silent point loss.
    """
    import pandas as pd

    path = Path(src_dir) / STRAIN_FILE
    if not path.is_file():
        raise FileNotFoundError(f"GSRM v2.2 应变源不存在: {path}（.Z 解压版）")
    nlat, nlon = GRID_NLAT, GRID_NLON
    grid_rows = nlat * nlon
    trailing_rows = TRAILING_ROWS
    df = pd.read_csv(
        path, sep=r"\s+", skiprows=HEADER_LINES, header=None,
        usecols=[0, 1, 2, 3, 4, 5], dtype="float64", engine="c",
    )
    if df.shape != (grid_rows + trailing_rows, 6):
        raise ValueError(
            f"{STRAIN_FILE}: 数据形状 {df.shape} ≠ 期望 "
            f"({grid_rows + trailing_rows}, 6)（源结构漂移——须复核契约）"
        )

    la = df.iloc[:grid_rows, 0].to_numpy()
    lo = df.iloc[:grid_rows, 1].to_numpy()
    r_idx = np.repeat(np.arange(nlat), nlon)
    c_idx = np.tile(np.arange(nlon), nlat)
    if np.abs(la - (-87.45 + 0.1 * r_idx)).max() > _COORD_TOL:
        raise ValueError(f"{STRAIN_FILE}: 网格行纬度偏离 -87.45+0.1k 行主序")
    if np.abs(lo - (-179.95 + 0.1 * c_idx)).max() > _COORD_TOL:
        raise ValueError(f"{STRAIN_FILE}: 网格行经度偏离 -179.95+0.1m 列循环")

    t = df.iloc[grid_rows:].to_numpy()
    if not np.isfinite(t).all():
        raise ValueError(f"{STRAIN_FILE}: 尾部块含非有限值（实证全数值）")
    exp_lat_c = -87.45 + 0.1 * np.arange(nlat)
    exp_lon_c = -179.95 + 0.1 * np.arange(nlon)
    r = np.clip(np.searchsorted(exp_lat_c, t[:, 0]), 1, nlat - 1)
    d_lat = np.minimum(np.abs(t[:, 0] - exp_lat_c[r - 1]), np.abs(t[:, 0] - exp_lat_c[r]))
    c = np.clip(np.searchsorted(exp_lon_c, t[:, 1]), 1, nlon - 1)
    d_lon = np.minimum(np.abs(t[:, 1] - exp_lon_c[c - 1]), np.abs(t[:, 1] - exp_lon_c[c]))
    n_ongrid = int((np.minimum(d_lat, d_lon) <= _COORD_TOL).sum())
    if n_ongrid:
        raise ValueError(
            f"{STRAIN_FILE}: 尾部块含 {n_ongrid} 行网格坐标（实证全离格——"
            "源更新把有效点挪入尾部，须复核剔除规则）"
        )

    values = df.iloc[:grid_rows, 2:6].to_numpy()
    if not np.isfinite(values).all():
        raise ValueError(f"{STRAIN_FILE}: 网格值含非有限数（实证全数值无缺测）")
    fields = {
        name: values[:, j].reshape(nlat, nlon).astype(np.float32)
        for j, name in enumerate(("exx", "eyy", "exy", "vorticity"))
    }
    lat_edges = -90.0 + 0.1 * np.arange(LAT_SOUTH_K, LAT_SOUTH_K + nlat + 1, dtype=np.float64)
    return StrainSource(
        fields=fields,
        lat_edges=lat_edges,
        lon_res=SRC_TENTH,
        lon_phase=LON_PHASE,
        n_trailing=trailing_rows,
    )


_SOURCE_CACHE: dict[str, StrainSource] = {}


def _strain_source(src_dir: str | Path = GSRM_V22_DIR) -> StrainSource:
    """In-process cache (~4 s to parse; four layers × three tiers read once)."""
    key = f"strain:{Path(src_dir).resolve()}"
    if key not in _SOURCE_CACHE:
        _SOURCE_CACHE[key] = load_gsrm_strain(src_dir)
    return _SOURCE_CACHE[key]



def build_gsrm_strain_layer(entry: LayerEntry, tier: str) -> xr.DataArray:
    """Build one of the four GSRM strain layers for this tier (exx/eyy/exy/vorticity, internal)."""
    field = STRAIN_FIELD_OF[entry.id]
    src = _strain_source()
    values, _W = conservative_overlap_mean(
        src.fields[field], np.isfinite(src.fields[field]), src.lat_edges,
        src.lon_res, tier_fraction(tier), lon_phase=src.lon_phase,
    )
    arr = da.from_array(values, chunks=tier_chunks(tier))
    lat, lon = tier_centers(tier)
    attrs = {
        "units": entry.unit,
        "long_name": f"GSRM v2.2 应变率场 {field}——{_LONG_NAMES[field]}",
        "source": entry.source,
        "doi": GSRM_DOI,
        "native_res": entry.native_res,
        "resampling": entry.resampling,
        "visibility": entry.visibility,
        "earth_radius_km": EARTH_RADIUS_KM,
        "product_type": (
            "model（Haines-Holt 方法自 GPS 速度场插值的应变率模型网格，"
            "非观测；v2.2（2015-12 版）无 changelog，与 v2.1 不可混）"
        ),
        "unit_basis": UNIT_BASIS,
        "registration_rule": _REGISTRATION_RULE,
        "source_file": f"v2.2/{STRAIN_FILE}（GSRM_strain.txt.Z 解压版）",
        "license_note": _LICENSE_NOTE,
    }
    return xr.DataArray(
        arr, dims=("lat", "lon"), coords={"lat": lat, "lon": lon},
        name=entry.id, attrs=attrs,
    )




def build_gsrm_strain_fidelity_report(store_dir, layer_id: str, tiers: list[str]) -> dict:
    """Fidelity report (group-isomorphic criteria + source-contract criteria):

    ① store vs. source-kernel recompute bit-identical (float32 bit level, NaN-aware);
    ② valid mask vs. source-mask recompute consistent (polar-cap gap from the same source in both channels);
    ③ integral quantity conserved (< 0.1% hard criterion) + quantile offset (report-mode);
    ④ coverage (report-mode: the tier geometry expected from the source polar-cap gap);
    ⑤ source contract (hard criterion): complete grid 6300000 rows + trailing exclusion 5351 rows + value ranges
       four numbers (the reader already asserts these; here re-reported from contract constants — drift means FAIL).
    """
    spec = STRAIN_LAYERS[layer_id]
    src = _strain_source()
    ras = src.raster(spec["field"])
    hist_spec = spec["hist_spec"]
    src_stats = gravmag_source_stats(ras, hist_spec)

    checks: list[dict] = []
    with xr.open_datatree(store_dir, engine="zarr", chunks={}) as tree:
        for tier in tiers:
            node = tree[f"/{tier}"]
            if layer_id not in node.ds.data_vars:
                raise KeyError(f"store {store_dir} 的 {tier} 档不含 {layer_id}（先 build 再 fidelity）")
            v = node.ds[layer_id].values
            expect, W = conservative_overlap_mean(
                ras.values, ras.valid, ras.lat_edges, ras.lon_res,
                tier_fraction(tier), lon_phase=ras.lon_phase,
            )
            ok = W > 0

            eq = (v == expect) | (np.isnan(v) & np.isnan(expect))
            n_bad = int((~eq).sum())
            checks.append({
                "name": f"{tier}: 与源核重算逐位一致（float32 位级）",
                "status": "pass" if n_bad == 0 else "FAIL",
                "summary": f"不一致像元 {n_bad}（期望 0）",
            })

            mism = int((np.isfinite(v) != ok).sum())
            checks.append({
                "name": f"{tier}: 有效位与源掩膜复算一致（逐位）",
                "status": "pass" if mism == 0 else "FAIL",
                "summary": f"不一致 {mism} 像元（期望 0）",
            })

            tier_stats = {
                "sum_va": float(np.dot(v[ok].astype(np.float64), W[ok])),
            }
            lo, hi, bw = hist_spec
            tier_stats["hist"] = hist_add(
                np.zeros(hist_bins(hist_spec), dtype=np.uint64),
                v[np.isfinite(v)], lo=lo, hi=hi, bin_width=bw,
            )
            checks.extend(fidelity_checks(src_stats, tier_stats, tier, hist_spec=hist_spec, unit=spec["unit"]))

            checks.append({
                "name": f"{tier}: 覆盖率（报告制，极冠缺口）",
                "status": "report",
                "summary": (
                    f"档 {ok.mean():.4f} vs 源（像元计）{ras.valid.mean():.4f}"
                    "（源网格 lat ±87.5 天然不含极冠，|lat|>87.5 档位行 NaN"
                    " 保留，有效性掩膜 0）"
                ),
            })

    shape = src.fields["exx"].shape
    grid_ok = shape == (GRID_NLAT, GRID_NLON) and GRID_NLAT * GRID_NLON == GRID_ROWS == 6300000
    checks.append({
        "name": "源契约：完整网格 == 1750×3600（6300000 行）",
        "status": "pass" if grid_ok else "FAIL",
        "summary": f"加载形状 {shape}（读取器已断言行主序完整网格）",
    })
    checks.append({
        "name": "源契约：尾部离格块剔除 == 5351 行",
        "status": "pass" if src.n_trailing == TRAILING_ROWS == 5351 else "FAIL",
        "summary": (
            f"实际剔除 {src.n_trailing}（v2.2 构建残留：3 零行 + 1070 索引行"
            "各随 4 零行；坐标全离格已断言）"
        ),
    })
    f = src.fields[spec["field"]]
    lo_e, hi_e = VALUE_RANGES[spec["field"]]
    got = (float(f.min()), float(f.max()))
    range_ok = abs(got[0] - lo_e) <= 0.01 and abs(got[1] - hi_e) <= 0.01
    checks.append({
        "name": f"源契约：{spec['field']} 值域 == 磁盘实证四数",
        "status": "pass" if range_ok else "FAIL",
        "summary": (
            f"实测 [{got[0]:g}, {got[1]:g}] vs 契约 [{lo_e:g}, {hi_e:g}]"
            "（容差 0.01 吸收 float32 舍入；源数值重算即 FAIL 提示复核）"
        ),
    })

    n_fail = sum(1 for c in checks if c["status"] == "FAIL")
    return {
        "layer": layer_id,
        "store": str(store_dir),
        "integral_tolerance": INTEGRAL_TOL,
        "result": "FAIL" if n_fail else "PASS",
        "n_checks": len(checks),
        "checks": checks,
    }
