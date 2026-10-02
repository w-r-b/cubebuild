"""热流主题：GHFDB+NGHF 合并去重点密度 + HFgrid14 预测网格。

源契约（磁盘实证）：

- GHFDB R2024 v.2026.03：IHFC_2024_GHFDB_v.2026.03.xlsx，sheet
  "GHFDB R20024 v.2026.03"（源表名自带拼写错误，原样引用）。工作表
  前 6 行为表头（P/C 码、M/R/O、B/S、U score、单位、长名），数据
  91182 行 × 67 列，ID（R24-000001..）唯一。并存的 .txt 为空白分隔
  且 5 条记录含内嵌换行（91200 物理行 ≠ 91182 记录，字段缺失致
  token 数 13–166 不等、列对齐歧义）——解析以 xlsx 为权威（记录数
  = R24-P 计数互证）。全 91182 条记录坐标/热流值可解析（无缺坐标
  行；q_uncertainty "?" 占位转 NaN）。经度 ±180 制（含 2 条 lon=180）。
  许可：CC BY 4.0（txt 头 # Licence 行 + datacite rightsList 自证）。

- NGHF.csv（Lucazeau 2019 SI-S02）：69729 逻辑记录 × 25 列（csv
  模块与 pandas 互证；1 条记录站名含内嵌换行 → 物理行 69731）。2 条
  经度空（YENDI / AV-950）→ 可定位 69727；q 非数值 350（密度层不
  消费 q）。经度 ±180 制（含 180.0）。
- HFgrid14.csv（Lucazeau 2019 SI-S01，"preferred heat flow prediction
  map"）：分号分隔 259200 行 = 360 纬 × 720 经完整 0.5° 全球网格；
  列名 "longiyude"（源文件拼写错误，原样引用）。中心 -89.75..89.75
  步 0.5（北起）与 -179.75..179.75（-180 基准），均为二进制精确的
  0.25 偏移值。HF_pred 全球全覆盖无缺测（-6.0..2977.4 mW/m²，含
  少量物理上非正的模型值——作者原产品保真不动）；sHF_pred 不确定度
  与 Hf_obs（12663 胞观测均值）不 ingest（观测通道由密度层承担）。

GHFDB↔NGHF 整编核实（磁盘数值实证）：
NGHF 可定位 69727 点中 53394（76.6%）与 GHFDB 记录坐标圆整 4 位小数后
重合，其中 43785（82.1%）同坐标 q 值差 ≤0.01 mW/m²（同观测整编
证据）；64541（92.6%）落于 GHFDB 占用 0.1° 像元。结论：GHFDB 整编
了 NGHF 主体但未完全（5186 点在 0.1° 像元层面缺席，来源集中于
Rolandonne2020 等 GHFDB 未收录 compilation）——按规则执行合并
去重，NGHF 不整体转验证参考；被 0.1° 规则滤除的 64541 点保留在
original data/ 供验证参考（density 层的验证通道）。
"""

from fractions import Fraction
from pathlib import Path

import dask.array as da
import numpy as np
import xarray as xr

from .contract import LayerEntry
from .fidelity import INTEGRAL_TOL, fidelity_checks, hist_add
from .gravmag import RasterSource, gravmag_source_stats, hist_bins
from .grids import aggregation_factor, tier_centers, tier_chunks, tier_fraction
from .kernels import aggregate_sum, conservative_overlap_mean
from .pixel_area import EARTH_RADIUS_KM
from .points import (
    HOME_TIER,
    density_dataarray,
    points_to_density,
    wrap_lon180,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
HEATFLOW_SRC_DIR = REPO_ROOT / "original data/thermal/heatflow"
GHFDB_SRC_DIR = HEATFLOW_SRC_DIR / "ghfdb2024/GHFBD-R2024_v.2026-03"
GHFDB_XLSX_NAME = "IHFC_2024_GHFDB_v.2026.03.xlsx"
GHFDB_SHEET = "GHFDB R20024 v.2026.03"          # 源表名拼写（R20024）原样引用
NGHF_SRC_DIR = (
    HEATFLOW_SRC_DIR / "lucazeau2019/2019gc008389-sup-0004-data_set_si-s02"
)
NGHF_CSV_NAME = "NGHF.csv"
HFGRID_SRC_DIR = (
    HEATFLOW_SRC_DIR
    / "lucazeau2019/2019gc008389-sup-0003-data_set_si-s01"
    / "2019GC008389-sup-0003-Data_Set_SI-S01"
)
HFGRID_CSV_NAME = "HFgrid14.csv"

HEATFLOW_DENSITY_ID = "thermal__heatflow_density"
HFGRID14_ID = "thermal__hfgrid14_heatflow"

GHFDB_DOI = "10.5880/fidgeo.2024.014"           # GHFDB R2024（txt 头 Citation 自证）
LUCAZEAU_DOI = "10.1029/2019GC008389"           # Lucazeau 2019, G³

# 源契约常量（磁盘实证的契约化；源变更即读取期显式失败）
GHFDB_EXPECTED_ROWS = 91182
GHFDB_EXPECTED_COLS = 67
NGHF_EXPECTED_ROWS = 69729
NGHF_EXPECTED_COLS = 25
NGHF_EXPECTED_LOCATABLE = 69727
HFGRID_ROWS = 259200                             # 360 × 720
HFGRID_HALF = Fraction(1, 2)                     # 0.5° 源分辨率（精确有理数）

# 合并去重：0.1° 像元容差
PIX_TOL_DEG = 0.1
NGHF_SUPPLEMENT = 5186                           # 不落 GHFDB 像元的 NGHF 可定位点
MERGED_TOTAL = GHFDB_EXPECTED_ROWS + NGHF_SUPPLEMENT   # 96368

# 整编核实期望值（实证；源更新漂移即保真 FAIL 提示复核）
NGHF_EXACT_COORD_MATCHES = 53394                 # 坐标 4 位小数精确重合
NGHF_Q_IDENTICAL = 43785                         # 同坐标 |Δq| ≤ 0.01（可比对 53309）
NGHF_Q_COMPARABLE = 53309                        # 同坐标且双方 q 可解析
NGHF_PIXEL_OVERLAP = 64541                       # 落于 GHFDB 占用 0.1° 像元

COMPOSITE_RULE = (
    "GHFDB 为主（全部 91182 条记录），NGHF 仅补不与任何 GHFDB 记录同落 "
    "0.1° 像元的可定位点（5186 点；像元键 = 左闭 0.1° 分箱，坐标十进制"
    "舍入 6 位后取整——边界点确定性归胞，经度先卷绕 [-180,180)）；构建期"
    "核实 GHFDB 已整编 NGHF 主体（坐标精确重合 76.6%、其中 q 值逐位相同 "
    "82.1%）但未完全，故按本规则合并而非整体转验证参考"
)

COUNT_SEMANTICS = (
    "计数基准 = 合并点集 96368（GHFDB 2024 全部 91182 条记录 + NGHF 补充 "
    "5186 点；NGHF 69729 条中 2 条经度空不可定位、64541 点落于 GHFDB 占用 "
    "0.1° 像元被去重规则滤除——滤除点留 original data/ 供验证参考）；"
    "各档计数总和 = 96368（计数守恒），侧车 heatflow_points_merged 逐行"
    "落格即还原本层"
)

INCORPORATION_VERDICT = (
    "GHFDB 整编了 NGHF 主体但未完全：坐标圆整 4 位小数后重合 53394/69727 "
    "（76.6%），其中 q 值差 ≤0.01 mW/m² 者 43785（82.1%）——同观测整编"
    "证据；0.1° 像元重叠 64541（92.6%），5186 点（7.4%，来源集中于 "
    "Rolandonne2020 等 GHFDB 未收录 compilation）在像元层面缺席。按"
    "规则合并去重（GHFDB 为主 + NGHF 补充），NGHF 不整体转验证参考"
)

_LICENSE_NOTE = (
    "GHFDB CC BY 4.0（txt 头 Licence 行 + datacite 自证）；NGHF/HFgrid14 "
    "为 Lucazeau 2019 SI（Wiley 文章 Free Access，PDF/Crossref/Unpaywall "
    "均未给出具体 CC 授权）——公开发布前须向作者确认或随 GSRM 惯例转 "
    "internal-use-only"
)


# ---------------- 源读取 ----------------

def load_ghfdb(src_dir: str | Path = GHFDB_SRC_DIR):
    """读 GHFDB xlsx → tidy DataFrame（q/q_uncertainty/name/lat/lon/reference/id）。

    契约：工作表 (91182+6) × 67、长名行含所需列且无重复、ID 唯一、
    lat/lon 全可解析（缺坐标即显式失败——与 WSM/GSRM 的空坐标容忍不同，
    本源实证无缺坐标行）。
    """
    import pandas as pd

    path = Path(src_dir) / GHFDB_XLSX_NAME
    if not path.is_file():
        raise FileNotFoundError(f"GHFDB 源文件不存在: {path}")
    raw = pd.read_excel(path, sheet_name=GHFDB_SHEET, header=None)
    if raw.shape != (GHFDB_EXPECTED_ROWS + 6, GHFDB_EXPECTED_COLS):
        raise ValueError(
            f"{GHFDB_XLSX_NAME}: 形状 {raw.shape} ≠ 期望 "
            f"({GHFDB_EXPECTED_ROWS + 6}, {GHFDB_EXPECTED_COLS})（含 6 表头行）"
        )
    names = raw.iloc[5].tolist()
    need = ("q", "q_uncertainty", "name", "lat_NS", "long_EW",
            "publication_reference", "ID")
    for n in need:
        if names.count(n) != 1:
            raise ValueError(f"{GHFDB_XLSX_NAME}: 长名行中 {n!r} 缺失或重复")
    data = raw.iloc[6:].reset_index(drop=True)
    idx = {n: names.index(n) for n in need}
    df = pd.DataFrame({
        "q": pd.to_numeric(data[idx["q"]], errors="coerce"),
        "q_uncertainty": pd.to_numeric(data[idx["q_uncertainty"]], errors="coerce"),
        "name": data[idx["name"]].astype(str),
        "lat": pd.to_numeric(data[idx["lat_NS"]], errors="coerce"),
        "lon": pd.to_numeric(data[idx["long_EW"]], errors="coerce"),
        "reference": data[idx["publication_reference"]].astype(str),
        "id": data[idx["ID"]].astype(str),
    })
    n_bad = int(df[["lat", "lon"]].isna().any(axis=1).sum())
    if n_bad:
        raise ValueError(f"{GHFDB_XLSX_NAME}: {n_bad} 行坐标不可解析（实证无缺坐标，源已变更）")
    if not df["id"].is_unique:
        raise ValueError(f"{GHFDB_XLSX_NAME}: 记录 ID 不唯一")
    return df


def load_nghf(src_dir: str | Path = NGHF_SRC_DIR):
    """读 NGHF.csv → tidy DataFrame（同 GHFDB 列名口径 + source_row 原行号）。

    契约：69729 × 25；坐标/heat-flow 列以 to_numeric 解析（2 条经度空
    → NaN，落格端处理）；keep_default_na=False 保留字面字符串（含
    HFC 码等），空字段再显式转 NaN。
    """
    import pandas as pd

    path = Path(src_dir) / NGHF_CSV_NAME
    if not path.is_file():
        raise FileNotFoundError(f"NGHF 源文件不存在: {path}")
    df = pd.read_csv(path, keep_default_na=False, low_memory=False)
    if df.shape != (NGHF_EXPECTED_ROWS, NGHF_EXPECTED_COLS):
        raise ValueError(
            f"{NGHF_CSV_NAME}: 形状 {df.shape} ≠ 期望 "
            f"({NGHF_EXPECTED_ROWS}, {NGHF_EXPECTED_COLS})"
        )
    tidy = pd.DataFrame({
        "q": pd.to_numeric(df["heat-flow (mW/m2)"], errors="coerce"),
        "q_uncertainty": pd.to_numeric(df["uncertainty hf (mW/m2)"], errors="coerce"),
        "name": df["Name of site"].astype(str),
        "lat": pd.to_numeric(df["latitude"], errors="coerce"),
        "lon": pd.to_numeric(df["longitude"], errors="coerce"),
        "reference": df["reference"].astype(str),
        "source_row": np.arange(len(df), dtype=np.int64),
    })
    n_loc = int((tidy["lat"].notna() & tidy["lon"].notna()).sum())
    if n_loc != NGHF_EXPECTED_LOCATABLE:
        raise ValueError(
            f"{NGHF_CSV_NAME}: 可定位行数 {n_loc} ≠ 期望 {NGHF_EXPECTED_LOCATABLE}"
            "（源更新或空坐标行数变化——须复核 count_semantics）"
        )
    return tidy


def load_hfgrid14(src_dir: str | Path = HFGRID_SRC_DIR) -> RasterSource:
    """读 HFgrid14.csv → 规范化源（行 0 = -90 侧、列 0 = -180 侧）。

    断言（磁盘实证的契约化）：259200 行完整网格（每 (lat,lon) 恰一行；
    中心恰为 0.25 偏移精确值，searchsorted 回读逐位相等即配准自证）、
    HF_pred 全球全覆盖有限（无缺测——作者预测场特性）。
    """
    import pandas as pd

    path = Path(src_dir) / HFGRID_CSV_NAME
    if not path.is_file():
        raise FileNotFoundError(f"HFgrid14 源文件不存在: {path}")
    df = pd.read_csv(path, sep=";")
    if df.shape != (HFGRID_ROWS, 5):
        raise ValueError(f"{HFGRID_CSV_NAME}: 形状 {df.shape} ≠ 期望 (259200, 5)")
    if list(df.columns) != ["longiyude", "latitude", "HF_pred", "sHF_pred", "Hf_obs"]:
        raise ValueError(f"{HFGRID_CSV_NAME}: 列名漂移 {list(df.columns)}")
    la = df["latitude"].to_numpy(dtype=np.float64)
    lo = df["longiyude"].to_numpy(dtype=np.float64)
    pred = pd.to_numeric(df["HF_pred"], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(pred).all():
        raise ValueError(f"{HFGRID_CSV_NAME}: HF_pred 含非有限值（实证全覆盖，源已变更）")

    nlat, nlon = 360, 720
    exp_lat = -90.0 + (np.arange(nlat, dtype=np.float64) + 0.5) * float(HFGRID_HALF)
    exp_lon = -180.0 + (np.arange(nlon, dtype=np.float64) + 0.5) * float(HFGRID_HALF)
    r = np.searchsorted(exp_lat, la)
    c = np.searchsorted(exp_lon, lo)
    r = np.clip(r, 0, nlat - 1)
    c = np.clip(c, 0, nlon - 1)
    if not (np.array_equal(exp_lat[r], la) and np.array_equal(exp_lon[c], lo)):
        raise ValueError(f"{HFGRID_CSV_NAME}: 中心坐标偏离 0.5° 规则网格")
    flat = r.astype(np.int64) * nlon + c.astype(np.int64)
    if np.unique(flat).size != HFGRID_ROWS:
        raise ValueError(f"{HFGRID_CSV_NAME}: (lat,lon) 组合存在重复，非完整网格")

    values = np.full((nlat, nlon), np.nan, dtype=np.float32)
    values.ravel()[flat] = pred.astype(np.float32)
    lat_edges = -90.0 + np.arange(nlat + 1, dtype=np.float64) * float(HFGRID_HALF)
    return RasterSource(
        values=values,
        valid=np.isfinite(values),
        lat_edges=lat_edges,
        lon_res=HFGRID_HALF,
        lon_phase=HFGRID_HALF / 2,                # 0.25° 偏移中心 = 边缘对齐 -180
        attrs={
            "registration_rule": (
                "pixel 注册（0.5° 中心 -89.75..89.75 / -179.75..179.75，"
                "0.25 偏移二进制精确值，searchsorted 回读自证）＝源胞边对齐 "
                "-180/-90（保守核 lon_phase = 0.25，30′ 档恒等映射）；"
                "源北起行序翻转为南起"
            ),
            "source_file": HFGRID_CSV_NAME,
        },
    )


# ---------------- 合并去重 + 整编核实 ----------------

def pixel_keys(lat, lon, res: float = PIX_TOL_DEG) -> np.ndarray:
    """0.1° 像元键（左闭分区；经度先卷绕 [-180, 180)）。

    合并去重规则与整编核实共用的唯一实现（行 0 = -90 侧）。
    边界鲁棒性：坐标先 round((x+off)/res, 6) 再 floor——裸除法
    (x+off)/res 对恰在像元边界的十进制坐标（如 23.3）会产生
    1132.999…/1133.000… 的浮点噪声，非确定落邻胞（实证 69727 点中
    3808 点受扰）；先舍入到 6 位小数使边界点按左闭约定确定性归胞。
    """
    lat = np.asarray(lat, dtype=np.float64)
    lon_w = wrap_lon180(lon)
    nlon = int(round(360.0 / res))
    scale = 1.0 / res
    r = np.floor(np.round((lat + 90.0) * scale, 6)).astype(np.int64)
    c = np.floor(np.round((lon_w + 180.0) * scale, 6)).astype(np.int64)
    return r * nlon + c


_MERGED_CACHE: dict[str, object] = {}
_HOME_CACHE: dict[str, np.ndarray] = {}


def merge_heatflow_points(ghfdb_src: str | Path = GHFDB_SRC_DIR,
                          nghf_src: str | Path = NGHF_SRC_DIR):
    """规则：GHFDB 为主 + NGHF 补充不落 GHFDB 0.1° 像元的可定位点。

    返回统一口径 DataFrame（source/source_row/name/lat/lon/q/
    q_uncertainty/reference）：GHFDB 91182 全记录 + NGHF 5186 补充
    = 96368；计数与 MERGED_TOTAL 契约互证（源漂移即显式失败）。
    进程内缓存（xlsx 读取 ~1 分钟，四档只读一次）。
    """
    import pandas as pd

    key = f"merged:{Path(ghfdb_src).resolve()}|{Path(nghf_src).resolve()}"
    if key in _MERGED_CACHE:
        return _MERGED_CACHE[key]
    g = load_ghfdb(ghfdb_src)
    n = load_nghf(nghf_src)
    g_keys = set(pixel_keys(g["lat"].to_numpy(), g["lon"].to_numpy()).tolist())
    locatable = n["lat"].notna() & n["lon"].notna()
    n_sub = n[locatable]
    keep = ~np.isin(
        pixel_keys(n_sub["lat"].to_numpy(), n_sub["lon"].to_numpy()),
        list(g_keys),
    )
    n_keep = n_sub[np.asarray(keep)].copy()
    if len(n_keep) != NGHF_SUPPLEMENT:
        raise ValueError(
            f"NGHF 补充点数 {len(n_keep)} ≠ 期望 {NGHF_SUPPLEMENT}"
            "（GHFDB/NGHF 源更新——须复核 count_semantics 与整编核实结论）"
        )
    g_part = pd.DataFrame({
        "source": "GHFDB",
        "source_row": np.arange(len(g), dtype=np.int64),
        "name": g["name"].to_numpy(),
        "lat": g["lat"].to_numpy(),
        "lon": g["lon"].to_numpy(),
        "q": g["q"].to_numpy(),
        "q_uncertainty": g["q_uncertainty"].to_numpy(),
        "reference": g["reference"].to_numpy(),
        "id": g["id"].to_numpy(),
    })
    n_part = pd.DataFrame({
        "source": "NGHF",
        "source_row": n_keep["source_row"].to_numpy(),
        "name": n_keep["name"].to_numpy(),
        "lat": n_keep["lat"].to_numpy(),
        "lon": n_keep["lon"].to_numpy(),
        "q": n_keep["q"].to_numpy(),
        "q_uncertainty": n_keep["q_uncertainty"].to_numpy(),
        "reference": n_keep["reference"].to_numpy(),
        "id": None,
    })
    merged = pd.concat([g_part, n_part], ignore_index=True)
    if len(merged) != MERGED_TOTAL:
        raise ValueError(f"合并点集 {len(merged)} ≠ 期望 {MERGED_TOTAL}")
    _MERGED_CACHE[key] = merged
    return merged


def verify_nghf_incorporation(ghfdb_src: str | Path = GHFDB_SRC_DIR,
                              nghf_src: str | Path = NGHF_SRC_DIR) -> dict:
    """整编核实：三项磁盘数值证据（确定性，随源固定）。

    ① 坐标 4 位小数精确重合数（NGHF 坐标 4 位小数、GHFDB 最多 9 位，
       重合 = 同一观测被 GHFDB 收录的强证据）；
    ② 同坐标 q 值一致性（|Δq| ≤ 0.01 视为逐位相同）；
    ③ 0.1° 像元重叠数（合并规则的实际滤除量）。
    """
    g = load_ghfdb(ghfdb_src)
    n = load_nghf(nghf_src)
    ok_n = n["lat"].notna() & n["lon"].notna()

    # ①② 精确坐标匹配（GHFDB 同坐标多记录取最小 |Δq|）
    gmap: dict[tuple, list[int]] = {}
    g_lat = g["lat"].to_numpy()
    g_lon = g["lon"].to_numpy()
    g_q = g["q"].to_numpy()
    for i in range(len(g)):
        gmap.setdefault((round(g_lat[i], 4), round(g_lon[i], 4)), []).append(i)
    n_lat = n.loc[ok_n, "lat"].to_numpy()
    n_lon = n.loc[ok_n, "lon"].to_numpy()
    n_q = n.loc[ok_n, "q"].to_numpy()
    exact = 0
    q_comparable = 0
    q_identical = 0
    for j in range(len(n_lat)):
        hits = gmap.get((round(n_lat[j], 4), round(n_lon[j], 4)))
        if not hits:
            continue
        exact += 1
        gqs = [g_q[i] for i in hits if np.isfinite(g_q[i])]
        if gqs and np.isfinite(n_q[j]):
            q_comparable += 1
            if min(abs(n_q[j] - v) for v in gqs) <= 0.01:
                q_identical += 1

    # ③ 像元重叠
    g_keys = set(pixel_keys(g_lat, g_lon).tolist())
    overlap = int(np.isin(
        pixel_keys(n_lat, n_lon), list(g_keys)
    ).sum())

    return {
        "nghf_locatable": int(ok_n.sum()),
        "exact_coord_matches": exact,
        "q_comparable": q_comparable,
        "q_identical": q_identical,
        "pixel_overlap": overlap,
        "supplement": int(ok_n.sum()) - overlap,
        "verdict": INCORPORATION_VERDICT,
    }


def heatflow_home_density(ghfdb_src: str | Path = GHFDB_SRC_DIR,
                          nghf_src: str | Path = NGHF_SRC_DIR) -> np.ndarray:
    """主档密度：合并点集全 96368 行（全部可定位，构造性保证）。"""
    key = f"home:{Path(ghfdb_src).resolve()}|{Path(nghf_src).resolve()}"
    if key not in _HOME_CACHE:
        df = merge_heatflow_points(ghfdb_src, nghf_src)
        _HOME_CACHE[key] = points_to_density(
            df["lat"].to_numpy(), df["lon"].to_numpy(), HOME_TIER
        )
    return _HOME_CACHE[key]


# ---------------- 层构建 ----------------

def build_heatflow_density(entry: LayerEntry, tier: str) -> xr.DataArray:
    """构建该档 thermal__heatflow_density（uint32 计数层，免掩膜）。"""
    return density_dataarray(
        entry, tier, heatflow_home_density(),
        "热流观测点密度（GHFDB 2024 + NGHF 合并去重点集，每像元记录数）",
        {
            "source_file": f"{GHFDB_XLSX_NAME} + {NGHF_CSV_NAME}（合并去重）",
            "composite_rule": COMPOSITE_RULE,
            "count_semantics": COUNT_SEMANTICS,
            "license_note": _LICENSE_NOTE,
        },
    )


_SOURCE_CACHE: dict[str, RasterSource] = {}


def _hfgrid14_source(src_dir: str | Path = HFGRID_SRC_DIR) -> RasterSource:
    key = f"hfgrid14:{Path(src_dir).resolve()}"
    if key not in _SOURCE_CACHE:
        _SOURCE_CACHE[key] = load_hfgrid14(src_dir)
    return _SOURCE_CACHE[key]


def build_hfgrid14_heatflow(entry: LayerEntry, tier: str) -> xr.DataArray:
    """构建该档 thermal__hfgrid14_heatflow（模型产品，非观测）。"""
    src = _hfgrid14_source()
    values, _W = conservative_overlap_mean(
        src.values, src.valid, src.lat_edges, src.lon_res, tier_fraction(tier),
        lon_phase=src.lon_phase,
    )
    arr = da.from_array(values, chunks=tier_chunks(tier))
    lat, lon = tier_centers(tier)
    attrs = {
        "units": entry.unit,
        "long_name": "地表热流预测（Lucazeau 2019 HFgrid14，作者官方 0.5° 预测网格）",
        "source": entry.source,
        "doi": LUCAZEAU_DOI,
        "native_res": entry.native_res,
        "resampling": entry.resampling,
        "visibility": entry.visibility,
        "earth_radius_km": EARTH_RADIUS_KM,
        "product_type": (
            "model（作者官方预测网格，非观测；sHF_pred/Hf_obs 列不 ingest；"
            "与热流密度层互补非冗余——密度层 = 观测计数通道，"
            "本层 = 连续预测场通道）"
        ),
        "license_note": _LICENSE_NOTE,
    }
    attrs.update(src.attrs)
    return xr.DataArray(
        arr, dims=("lat", "lon"), coords={"lat": lat, "lon": lon},
        name=entry.id, attrs=attrs,
    )


# ---------------- 保真验证（本层自供） ----------------

# HF_pred 值域 -6..2977.4 mW/m²（磁盘实证）→ 直方图几何（-10, 3000, 1）
THERMAL_LAYERS = {
    HEATFLOW_DENSITY_ID: {
        "kind": "density",
        "src_dirs": (GHFDB_SRC_DIR, NGHF_SRC_DIR),
        "expected_total": MERGED_TOTAL,
    },
    HFGRID14_ID: {
        "kind": "raster",
        "src_dir": HFGRID_SRC_DIR,
        "loader": load_hfgrid14,
        "hist_spec": (-10.0, 3000.0, 1.0),
        "unit": "mW/m2",
    },
}


def _incorporation_checks(ghfdb_src, nghf_src) -> list[dict]:
    """整编核实：证据数字硬判据 + 结论报告。"""
    ev = verify_nghf_incorporation(ghfdb_src, nghf_src)
    pairs = (
        ("坐标圆整 4 位小数后重合", ev["exact_coord_matches"], NGHF_EXACT_COORD_MATCHES),
        ("同坐标 q 可比对", ev["q_comparable"], NGHF_Q_COMPARABLE),
        ("同坐标 |Δq| ≤ 0.01", ev["q_identical"], NGHF_Q_IDENTICAL),
        ("0.1° 像元重叠", ev["pixel_overlap"], NGHF_PIXEL_OVERLAP),
        ("NGHF 补充点", ev["supplement"], NGHF_SUPPLEMENT),
    )
    checks = []
    for name, got, want in pairs:
        checks.append({
            "name": f"整编核实: {name} == {want}",
            "status": "pass" if got == want else "FAIL",
            "summary": f"实测 {got}（期望 {want}；源漂移须复核整编结论）",
        })
    checks.append({
        "name": "整编核实结论（报告制）",
        "status": "report",
        "summary": ev["verdict"],
    })
    return checks


# 双形态互补注记（密度层与连续场并列呈现、互补非冗余）
_DUAL_FORM_NOTE = (
    "热流主题双形态互补非冗余：thermal__heatflow_density = 观测计数通道"
    "（GHFDB+NGHF 合并点集，uint32 免掩膜），thermal__hfgrid14_heatflow = "
    "连续预测场通道（作者官方 0.5° 模型网格，product_type=model）——"
    "两报告并列呈现，消费方按 product_type 区分观测与模型"
)


def build_thermal_fidelity_report(store_dir, layer_id: str, tiers: list[str],
                                   src_dir=None) -> dict:
    """保真报告：密度层走计数判据（同构）+ 整编核实（新增）；
    HFgrid14 走重磁组三层判据（逐位/有效位/积分+分位数）。

    src_dir：缺省按层解析（双源密度层给 heatflow 根目录，其下按常量的
    相对结构解析 ghfdb/NGHF 子路径；HFgrid14 给其 SI-S01 目录）。
    """
    spec = THERMAL_LAYERS[layer_id]
    checks: list[dict] = []

    if spec["kind"] == "density":
        if src_dir is None:
            gsrc, nsrc = GHFDB_SRC_DIR, NGHF_SRC_DIR
        else:
            root = Path(src_dir)
            gsrc = root / GHFDB_SRC_DIR.relative_to(HEATFLOW_SRC_DIR)
            nsrc = root / NGHF_SRC_DIR.relative_to(HEATFLOW_SRC_DIR)
        home = heatflow_home_density(gsrc, nsrc)
        values: dict[str, np.ndarray] = {}
        with xr.open_datatree(store_dir, engine="zarr", chunks={}) as tree:
            for tier in tiers:
                node = tree[f"/{tier}"]
                if layer_id not in node.ds.data_vars:
                    raise KeyError(f"store {store_dir} 的 {tier} 档不含 {layer_id}（先 build 再 fidelity）")
                v = node.ds[layer_id].values
                values[tier] = v
                expect = home if tier == HOME_TIER else aggregate_sum(
                    home, aggregation_factor(tier, HOME_TIER)
                )
                n_bad = int((v != expect).sum())
                checks.append({
                    "name": f"{tier}: 与源落格重算逐位一致（uint32 位级）",
                    "status": "pass" if n_bad == 0 else "FAIL",
                    "summary": f"不一致像元 {n_bad}（期望 0）",
                })
                total = int(v.sum(dtype=np.uint64))
                checks.append({
                    "name": f"{tier}: 计数守恒（总和 == 计数基准 {spec['expected_total']}）",
                    "status": "pass" if total == spec["expected_total"] else "FAIL",
                    "summary": f"总和 {total} vs 基准 {spec['expected_total']}",
                })
                bad = int((v < 0).sum())
                checks.append({
                    "name": f"{tier}: dtype uint32 且值域 ≥ 0",
                    "status": "pass" if (v.dtype == np.uint32 and bad == 0) else "FAIL",
                    "summary": f"dtype {v.dtype}，负值 {bad}（期望 0）",
                })
            for tier in tiers:
                if tier == HOME_TIER:
                    continue
                expect = aggregate_sum(values[HOME_TIER], aggregation_factor(tier, HOME_TIER))
                n_bad = int((values[tier] != expect).sum())
                checks.append({
                    "name": f"{tier}: sum 一致性（= 主档 {HOME_TIER} store 值求和聚合）",
                    "status": "pass" if n_bad == 0 else "FAIL",
                    "summary": f"不一致像元 {n_bad}（期望 0）",
                })
            for tier in tiers:
                v = values[tier]
                checks.append({
                    "name": f"{tier}: 密度分布（报告制）",
                    "status": "report",
                    "summary": (
                        f"占用像元 {int((v > 0).sum())}/{v.size}，"
                        f"最大计数 {int(v.max())}"
                    ),
                })
        checks.extend(_incorporation_checks(gsrc, nsrc))
    else:
        src = spec["loader"](src_dir) if src_dir else spec["loader"]()
        hist_spec = spec["hist_spec"]
        unit = spec["unit"]
        src_stats = gravmag_source_stats(src, hist_spec)
        with xr.open_datatree(store_dir, engine="zarr", chunks={}) as tree:
            for tier in tiers:
                node = tree[f"/{tier}"]
                if layer_id not in node.ds.data_vars:
                    raise KeyError(f"store {store_dir} 的 {tier} 档不含 {layer_id}（先 build 再 fidelity）")
                v = node.ds[layer_id].values
                expect, W = conservative_overlap_mean(
                    src.values, src.valid, src.lat_edges, src.lon_res,
                    tier_fraction(tier), lon_phase=src.lon_phase,
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
                checks.extend(fidelity_checks(src_stats, tier_stats, tier, hist_spec=hist_spec, unit=unit))
                checks.append({
                    "name": f"{tier}: 覆盖率（报告制）",
                    "status": "report",
                    "summary": (
                        f"档 {ok.mean():.4f} vs 源（像元计）{src.valid.mean():.4f}"
                        "（HF_pred 全球全覆盖——契约层 validity 掩膜全 1，"
                        "与 landsea 掩膜组合分层由消费方完成）"
                    ),
                })

    # 双形态互补：两报告并列呈现，注记互补非冗余
    checks.append({
        "name": "双形态互补注记（报告制）",
        "status": "report",
        "summary": _DUAL_FORM_NOTE,
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
