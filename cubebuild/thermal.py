"""Heat-flow topic: GHFDB+NGHF merged deduplicated point density + HFgrid14
prediction grid.

Source contract (verified on disk):

- GHFDB R2024 v.2026.03: IHFC_2024_GHFDB_v.2026.03.xlsx, sheet
  "GHFDB R20024 v.2026.03" (the source sheet name has its own typo, quoted
  as-is). The worksheet's
  first 6 rows are the header (P/C code, M/R/O, B/S, U score, unit, long name),
  data
  91182 rows × 67 columns, ID (R24-000001..) unique. The coexisting .txt is
  blank-separated
  and 5 records contain embedded newlines (91200 physical lines ≠ 91182
  records; missing fields make the token count range 13–166 and the column
  alignment ambiguous)—parsing takes the xlsx as authoritative (the record
  count
  = R24-P count, cross-checked). All 91182 records' coordinates/heat-flow
  values are parseable (no missing-coordinate rows; the q_uncertainty "?"
  placeholder becomes NaN). Longitude in ±180 convention (including 2 rows with
  lon=180).
  License: CC BY 4.0 (self-verified via the txt header # Licence line +
  datacite rightsList).

- NGHF.csv (Lucazeau 2019 SI-S02): 69729 logical records × 25 columns
  (cross-verified by the csv module and pandas; 1 record has an embedded
  newline in its site name → 69731 physical lines). 2 longitudes empty
  (YENDI / AV-950) → 69727 locatable; q non-numeric 350 (the density layer does
  not consume q). Longitude in ±180 convention (including 180.0).
- HFgrid14.csv (Lucazeau 2019 SI-S01, "preferred heat flow prediction
  map"): semicolon-separated 259200 rows = 360 latitude × 720 longitude complete
  0.5° global grid;
  the column name "longiyude" (a source-file typo, quoted as-is). Centers
  -89.75..89.75
  step 0.5 (north first) and -179.75..179.75 (-180 basis), all binary-exact
  0.25 offset values. HF_pred covers the globe fully with no missing values
  (-6.0..2977.4 mW/m², including
  a few physically non-positive model values—the author's original product kept
  faithfully); the sHF_pred uncertainty
  and Hf_obs (12663 cells of observed means) are not ingested (the observation
  channel is carried by the density layer).

GHFDB↔NGHF incorporation check (numerical disk verification):
of the 69727 locatable NGHF points, 53394 (76.6%) coincide with GHFDB record
coordinates after rounding to 4 decimals,
of which 43785 (82.1%) have the same-coordinate q difference ≤0.01 mW/m² (same
observation incorporation
evidence); 64541 (92.6%) fall in GHFDB-occupied 0.1° cells. Conclusion: GHFDB
incorporated the bulk of NGHF but not fully (5186 points are
absent at the 0.1° cell level, source concentrated in
Rolandonne2020 and other compilations not included in GHFDB)—the deduplicated
merge is executed by rule, and NGHF is not wholly turned into validation
reference; the 64541 points filtered out by the 0.1° rule are kept in
original data/ for validation reference (the density layer's validation
channel).
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
GHFDB_SHEET = "GHFDB R20024 v.2026.03"
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

GHFDB_DOI = "10.5880/fidgeo.2024.014"
LUCAZEAU_DOI = "10.1029/2019GC008389"           # Lucazeau 2019, G³

GHFDB_EXPECTED_ROWS = 91182
GHFDB_EXPECTED_COLS = 67
NGHF_EXPECTED_ROWS = 69729
NGHF_EXPECTED_COLS = 25
NGHF_EXPECTED_LOCATABLE = 69727
HFGRID_ROWS = 259200                             # 360 × 720
HFGRID_HALF = Fraction(1, 2)

PIX_TOL_DEG = 0.1
NGHF_SUPPLEMENT = 5186
MERGED_TOTAL = GHFDB_EXPECTED_ROWS + NGHF_SUPPLEMENT   # 96368

NGHF_EXACT_COORD_MATCHES = 53394
NGHF_Q_IDENTICAL = 43785
NGHF_Q_COMPARABLE = 53309
NGHF_PIXEL_OVERLAP = 64541

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



def load_ghfdb(src_dir: str | Path = GHFDB_SRC_DIR):
    """Read the GHFDB xlsx → tidy DataFrame
    (q/q_uncertainty/name/lat/lon/reference/id).

    Contract: worksheet (91182+6) × 67, the long-name row contains the required
    columns with no duplicates, ID unique, lat/lon all parseable (a missing
    coordinate fails explicitly—unlike the empty-coordinate tolerance of
    WSM/GSRM, this source is verified to have no missing-coordinate rows).
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
    """Read NGHF.csv → tidy DataFrame (same column naming as GHFDB +
    source_row original row number).

    Contract: 69729 × 25; the coordinate/heat-flow columns are parsed with
    to_numeric (2 empty longitudes → NaN, handled at the grid-assignment end);
    keep_default_na=False preserves literal strings (including HFC codes etc.),
    then empty fields are explicitly converted to NaN.
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
    """Read HFgrid14.csv → normalized source (row 0 = -90 side, column 0 = -180
    side).

    Assertions (contract-ized disk verification): 259200 rows forming a complete
    grid (exactly one row per (lat,lon); the center is exactly a 0.25 offset
    exact value, with searchsorted read-back bit-for-bit equal, i.e.
    registration self-verification), HF_pred finite and globally full-coverage
    (no missing values—a property of the author's prediction field).
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
        lon_phase=HFGRID_HALF / 2,
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



def pixel_keys(lat, lon, res: float = PIX_TOL_DEG) -> np.ndarray:
    """0.1° cell key (left-closed binning; longitude first wrapped to
    [-180, 180)).

    The single implementation shared by the deduplicated merge rule and the
    incorporation check (row 0 = -90 side).
    Boundary robustness: coordinates are first round((x+off)/res, 6) then
    floor—raw division
    (x+off)/res, for decimal coordinates lying exactly on a cell boundary (e.g.
    23.3), produces
    1132.999…/1133.000… floating-point noise and lands non-deterministically in
    the neighboring cell (verified: 3808 of 69727 points affected); rounding to
    6 decimals first makes boundary points land deterministically by the
    left-closed convention.
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
    """Rule: GHFDB primary + NGHF supplementing the locatable points not falling
    into GHFDB 0.1° cells.

    Returns a unified-schema DataFrame (source/source_row/name/lat/lon/q/
    q_uncertainty/reference): GHFDB 91182 all records + NGHF 5186 supplement
    = 96368; the count and MERGED_TOTAL cross-verify by contract (a source drift
    fails explicitly). In-process cache (xlsx reading takes ~1 minute; the four
    tiers read only once).
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
    """Incorporation check: three pieces of numerical disk evidence
    (deterministic, fixed with the source).

    ① the number of exact coordinate matches at 4 decimals (NGHF coordinates to
       4 decimals, GHFDB to at most 9; a match = strong evidence the same
       observation was incorporated by GHFDB);
    ② same-coordinate q-value consistency (|Δq| ≤ 0.01 treated as bit-identical);
    ③ the 0.1° cell overlap count (the actual filtered amount of the merge rule).
    """
    g = load_ghfdb(ghfdb_src)
    n = load_nghf(nghf_src)
    ok_n = n["lat"].notna() & n["lon"].notna()

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
    """Home-tier density: all 96368 merged point-set rows (all locatable,
    guaranteed by construction)."""
    key = f"home:{Path(ghfdb_src).resolve()}|{Path(nghf_src).resolve()}"
    if key not in _HOME_CACHE:
        df = merge_heatflow_points(ghfdb_src, nghf_src)
        _HOME_CACHE[key] = points_to_density(
            df["lat"].to_numpy(), df["lon"].to_numpy(), HOME_TIER
        )
    return _HOME_CACHE[key]



def build_heatflow_density(entry: LayerEntry, tier: str) -> xr.DataArray:
    """Build this tier's thermal__heatflow_density (uint32 count layer,
    mask-free)."""
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
    """Build this tier's thermal__hfgrid14_heatflow (a model product, not an
    observation)."""
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
    """Incorporation check: hard criteria on the evidence numbers + a
    conclusion report."""
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


_DUAL_FORM_NOTE = (
    "热流主题双形态互补非冗余：thermal__heatflow_density = 观测计数通道"
    "（GHFDB+NGHF 合并点集，uint32 免掩膜），thermal__hfgrid14_heatflow = "
    "连续预测场通道（作者官方 0.5° 模型网格，product_type=model）——"
    "两报告并列呈现，消费方按 product_type 区分观测与模型"
)


def build_thermal_fidelity_report(store_dir, layer_id: str, tiers: list[str],
                                   src_dir=None) -> dict:
    """Fidelity report: the density layer goes through count criteria
    (isomorphic) + the incorporation check (newly added);
    HFgrid14 goes through the gravity-magnetics group's three-layer criteria
    (bit-for-bit/valid-flag/integral + quantiles).

    src_dir: resolved by layer by default (the dual-source density layer gets
    the heatflow root directory, under which the ghfdb/NGHF subpaths are
    resolved by the constants' relative structure; HFgrid14 gets its SI-S01
    directory).
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
