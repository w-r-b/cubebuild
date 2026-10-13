"""Point-density channel: points → cell count layers (sum kernel).

The cube does not interpolate point data: all raw points remain in vector
sidecars; the cube only ingests density/count layers—this module delivers
two uint32 density layers (mask-free, 0 means no observation):
- stress_kinematics__wsm_density (public): WSM 2025 stress observation points
- stress_kinematics__gsrm_gps_density (internal): GSRM GPS station observations

WSM source contract (verified on disk):
- WSM_Database_2025.csv: 100842 data rows × 40 columns (named header; LAT
  before LON; LON in ±180 convention). An xlsx with the same content exists
  alongside; the cube consumes the csv.
- 170 rows with empty LAT/LON (TYPE=DIF 77/BO 64/HF 29, QUALITY all Xmi)—
  no coordinates means no cell assignment: the density-layer count basis =
  100672 locatable rows (count_semantics note); the sidecar retains all
  100842 rows as-is.
- AZI=999 is a placeholder value (11095 rows)—unrelated to the density layer;
  sidecar consumers should be aware.

GSRM source contract (verified on disk):
- GPS_ITRF08.gmt: psvelo 9 columns without header lon lat ve vn se sn corr
  site study; 22511 rows all locatable, no empty values. Column 1 longitude
  in 0..360 convention (measured 0.001..359.994)—wrap to [-180, 180) before
  cell assignment.
- The psvelo "0,0,0" uncertainty description in the README matches only the
  vel-at-GPS_*.gmt model files; columns 5–7 of the observation files hold the
  real se/sn/corr (parsed as the actual 9 columns).
- Three counts: 22511 rows / 18440 unique coordinates / 16681 unique station
  codes (psvelo is one row per station per study; station codes are reused
  across studies). The count basis is fixed = observation record rows 22511
  (semantically consistent with WSM record rows; assigning each sidecar row
  to a cell reproduces the density layer)—the three counts and the basis all
  go into manifest count_semantics.
- The 53 GPS_*.gmt reference-frame files duplicate the same stations (each
  22511 rows, same station set; verified that only velocity values differ
  across frame rotations); counting across frames is forbidden—the density
  layer takes the ITRF08 frame only.
"""

from pathlib import Path

import dask.array as da
import numpy as np
import xarray as xr

from .contract import LayerEntry
from .fidelity import INTEGRAL_TOL
from .grids import aggregation_factor, grid_shape, tier_centers, tier_chunks, tier_deg
from .kernels import aggregate_sum

REPO_ROOT = Path(__file__).resolve().parents[1]
WSM_SRC_DIR = REPO_ROOT / "original data/stress-kinematics/wsm2025-stress"
WSM_CSV_NAME = "WSM_Database_2025.csv"
GSRM_SRC_DIR = REPO_ROOT / "original data/stress-kinematics/gsrm-strain"
GSRM_GPS_NAME = "GPS_ITRF08.gmt"
GSRM_POLES_NAMES = ("poles.APM", "poles.IGS08", "poles.NNR", "poles.PA")

WSM_LAYER_ID = "stress_kinematics__wsm_density"
GSRM_LAYER_ID = "stress_kinematics__gsrm_gps_density"
GSRM_DOI = "10.1002/2014GC005407"
HOME_TIER = "3min"

WSM_EXPECTED_ROWS = 100842
WSM_EXPECTED_COLS = 40
WSM_EXPECTED_LOCATABLE = 100672
GSRM_EXPECTED_ROWS = 22511
GSRM_UNIQUE_COORDS = 18440
GSRM_UNIQUE_SITES = 16681

WSM_COUNT_SEMANTICS = (
    "计数基准 = 可定位观测记录行 100672（源 100842 行中 170 行 LAT/LON 空，"
    "TYPE=DIF/BO/HF、QUALITY=Xmi，无坐标不可落格不计数，侧车原样保留全行）；"
    "各档计数总和 = 100672（计数守恒）"
)
GSRM_COUNT_SEMANTICS = (
    "计数基准 = GPS_ITRF08.gmt 观测记录行 22511（三数：22511 行 / 18440 唯一"
    "坐标 / 16681 唯一站码；psvelo 按研究一行一站，站码跨研究复用——与 WSM "
    "记录行计数语义一致，侧车逐行落格即还原本层）；53 个参考系文件同站重复"
    "（各 22511 行同站集，已实证仅速度值随参考系旋转），禁跨系计数，密度层"
    "只取 ITRF08 单系"
)



def load_wsm_points(src_dir: str | Path = WSM_SRC_DIR):
    """Read WSM_Database_2025.csv → DataFrame (all 40 columns as-is).

    Contract: 100842 rows × 40 columns, LAT/LON columns present (handling of
    empty-coordinate rows is described in the module docstring and
    count_semantics—the reader does not filter here; the grid-assignment end
    applies dropna uniformly). NA semantics kept on both sides:
    keep_default_na=False preserves literal strings such as "NA" (pandas would
    otherwise swallow them into NaN—the GSRM poles North America plate code NA
    hits this pitfall), then empty fields are explicitly replace("", NaN) (the
    empty = missing semantics is not lost).
    """
    import pandas as pd

    path = Path(src_dir) / WSM_CSV_NAME
    if not path.is_file():
        raise FileNotFoundError(f"WSM 源文件不存在: {path}")
    df = pd.read_csv(path, low_memory=False, keep_default_na=False)
    df = df.replace("", np.nan)
    if df.shape != (WSM_EXPECTED_ROWS, WSM_EXPECTED_COLS):
        raise ValueError(
            f"{WSM_CSV_NAME}: 形状 {df.shape} ≠ 期望 "
            f"({WSM_EXPECTED_ROWS}, {WSM_EXPECTED_COLS})"
        )
    missing_cols = [c for c in ("LAT", "LON") if c not in df.columns]
    if missing_cols:
        raise ValueError(f"{WSM_CSV_NAME}: 缺坐标列 {missing_cols}")
    return df


def load_gsrm_gps(src_dir: str | Path = GSRM_SRC_DIR):
    """Read GPS_ITRF08.gmt → DataFrame (psvelo 9 named columns).

    Contract: 22511 rows × 9 columns, 9 fields per row (psvelo has no header;
    rows with missing fields would silently become "" under
    keep_default_na=False, so the per-row token count is explicitly validated).
    lon keeps its 0..360 raw value (wrapping is done inside the grid-assigner;
    the sidecar adds a separate lon_180 column). keep_default_na=False:
    literal "NA" in station/study names is preserved as-is (pandas would
    otherwise swallow it into NaN—verified pitfall with the poles North America
    plate code NA).
    """
    import pandas as pd

    path = Path(src_dir) / GSRM_GPS_NAME
    if not path.is_file():
        raise FileNotFoundError(f"GSRM GPS 源文件不存在: {path}")
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) != GSRM_EXPECTED_ROWS:
        raise ValueError(
            f"{GSRM_GPS_NAME}: 行数 {len(lines)} ≠ 期望 {GSRM_EXPECTED_ROWS}"
        )
    bad = [i + 1 for i, ln in enumerate(lines) if len(ln.split()) != 9]
    if bad:
        raise ValueError(f"{GSRM_GPS_NAME}: 非 9 字段行（行号如 {bad[:5]}）")
    df = pd.read_csv(
        path, sep=r"\s+", header=None, keep_default_na=False,
        names=["lon", "lat", "ve", "vn", "se", "sn", "corr", "site", "study"],
    )
    return df


def load_gsrm_poles(src_dir: str | Path = GSRM_SRC_DIR):
    """Read the 4 poles.* Euler-pole tables → merged DataFrame (frame column
    distinguishes frames).

    Format (self-documented in the README + measured on disk): four columns
    lat lon rate(deg/Ma) plate—the column order has lat first, opposite to the
    GPS files which have lon first; the last 3 rows of poles.PA are rotations
    of other reference frames relative to PA (negative rate), kept as-is. The
    plate code "NA" = North America plate (keep_default_na=False prevents
    pandas from swallowing it into NaN; rows with missing fields would likewise
    silently become "", so the per-row token count is explicitly validated).
    """
    import pandas as pd

    src_dir = Path(src_dir)
    frames = []
    for name in GSRM_POLES_NAMES:
        path = src_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"GSRM 欧拉极表不存在: {path}")
        lines = path.read_text(encoding="utf-8").splitlines()
        bad = [i + 1 for i, ln in enumerate(lines) if len(ln.split()) != 4]
        if bad:
            raise ValueError(f"{name}: 非 4 字段行（行号如 {bad[:5]}）")
        p = pd.read_csv(
            path, sep=r"\s+", header=None, keep_default_na=False,
            names=["lat", "lon", "rate_deg_per_myr", "plate"],
        )
        p.insert(0, "frame", name.removeprefix("poles."))
        frames.append(p)
    return pd.concat(frames, ignore_index=True)



def points_to_density(lat, lon, tier: str) -> np.ndarray:
    """Point coordinates → tier count grid (uint32, pure function,
    source-agnostic).

    Semantics: a point falling into cell [−90+r·res, −90+(r+1)·res) ×
    [−180+c·res, −180+(c+1)·res) counts as 1 (row 0 = −90 side, column 0 =
    −180 side, the same row/column math as tier_centers). Longitude is first
    wrapped to [-180, 180)—both the WSM ±180 convention and the GSRM 0..360
    convention apply (180 ≡ −180 same meridian → column 0); the lat=90 pole
    edge falls into the northernmost cell (no such point in the data;
    defensive clip).

    Count conservation: Σgrid = len(lat) (guaranteed by construction; the
    grid-assigner discards nothing).
    """
    nlat, nlon = grid_shape(tier)
    res = tier_deg(tier)
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    if lat.shape != lon.shape:
        raise ValueError(f"lat/lon 形状不一致: {lat.shape} vs {lon.shape}")
    if not (np.isfinite(lat).all() and np.isfinite(lon).all()):
        raise ValueError("坐标含非有限值（NaN/Inf）——调用方须先 dropna")
    if lat.size and (lat.min() < -90.0 or lat.max() > 90.0):
        raise ValueError("纬度出界 [-90, 90]")
    lon_w = wrap_lon180(lon)
    row = np.clip(np.floor((lat + 90.0) / res).astype(np.int64), 0, nlat - 1)
    col = np.clip(np.floor((lon_w + 180.0) / res).astype(np.int64), 0, nlon - 1)
    counts = np.bincount(row * nlon + col, minlength=nlat * nlon)
    return counts.astype(np.uint32).reshape(nlat, nlon)


def wrap_lon180(lon) -> np.ndarray:
    """Wrap longitude to [-180, 180) (unifies the 0..360 ↔ ±180 conventions;
    180 ≡ −180).

    The single implementation shared by the grid-assigner and the GSRM sidecar
    lon_180 column.
    """
    lon = np.asarray(lon, dtype=np.float64)
    return ((lon + 180.0) % 360.0) - 180.0


def _aggregation_factor(tier: str) -> int:
    """Home tier 3′ → target tier aggregation factor (thin wrapper around
    grids.aggregation_factor, module-local idiomatic name)."""
    return aggregation_factor(tier, HOME_TIER)


_HOME_CACHE: dict[str, np.ndarray] = {}


def _home_density(cache_key: str, lat, lon) -> np.ndarray:
    """Home-tier (3′) density grid (in-process cache: the four tiers of the
    same layer are assigned to cells only once)."""
    if cache_key not in _HOME_CACHE:
        _HOME_CACHE[cache_key] = points_to_density(lat, lon, HOME_TIER)
    return _HOME_CACHE[cache_key]


def wsm_locatable_mask(df) -> np.ndarray:
    """WSM table locatable-row mask (LAT/LON non-null) + row-count contract
    validation (single implementation).

    Shared by the density layer (grid assignment) and the sidecar exporter
    (empty geometries)—the two are necessarily consistent. Drift fails
    explicitly: a source update or a change in the number of empty-coordinate
    rows requires reviewing count_semantics.
    """
    mask = (df["LAT"].notna() & df["LON"].notna()).to_numpy()
    n_loc = int(mask.sum())
    if n_loc != WSM_EXPECTED_LOCATABLE:
        raise ValueError(
            f"{WSM_CSV_NAME}: 可定位行数 {n_loc} ≠ 期望 {WSM_EXPECTED_LOCATABLE}"
            "（源更新或空坐标行数变化——须复核 count_semantics）"
        )
    return mask


def wsm_home_density(src_dir: str | Path = WSM_SRC_DIR) -> np.ndarray:
    """WSM home-tier density: locatable record rows (empty-coordinate rows are
    not counted, per count_semantics)."""
    key = f"wsm:{Path(src_dir).resolve()}"
    if key not in _HOME_CACHE:
        df = load_wsm_points(src_dir)
        mask = wsm_locatable_mask(df)
        _HOME_CACHE[key] = points_to_density(
            df.loc[mask, "LAT"].to_numpy(), df.loc[mask, "LON"].to_numpy(), HOME_TIER
        )
    return _HOME_CACHE[key]


def gsrm_home_density(src_dir: str | Path = GSRM_SRC_DIR) -> np.ndarray:
    """GSRM home-tier density: all 22511 observation record rows of
    GPS_ITRF08.gmt."""
    key = f"gsrm:{Path(src_dir).resolve()}"
    if key not in _HOME_CACHE:
        df = load_gsrm_gps(src_dir)
        _HOME_CACHE[key] = points_to_density(
            df["lat"].to_numpy(), df["lon"].to_numpy(), HOME_TIER
        )
    return _HOME_CACHE[key]



def density_dataarray(entry: LayerEntry, tier: str, home: np.ndarray,
                      long_name: str, extra_attrs: dict) -> xr.DataArray:
    """Common density-layer assembly: home tier emitted directly / coarser
    tiers aggregated with sum + attrs template.

    Shared assembly for the point-density channel (reused by WSM/GSRM and the
    heat-flow point set). extra_attrs contains at least
    source_file/count_semantics and may carry whitelisted provenance keys such
    as doi/license_note (passed through to the manifest).
    """
    if tier == HOME_TIER:
        values = home
    else:
        values = aggregate_sum(home, _aggregation_factor(tier))
    arr = da.from_array(values, chunks=tier_chunks(tier))
    lat, lon = tier_centers(tier)
    attrs = {
        "units": entry.unit,
        "long_name": long_name,
        "source": entry.source,
        "native_res": entry.native_res,
        "resampling": entry.resampling,
        "visibility": entry.visibility,
        "nodata_semantics": "0 = 无观测（uint32 计数层，免掩膜）",
        **extra_attrs,
    }
    return xr.DataArray(
        arr, dims=("lat", "lon"), coords={"lat": lat, "lon": lon},
        name=entry.id, attrs=attrs,
    )


def build_wsm_density(entry: LayerEntry, tier: str) -> xr.DataArray:
    """Build this tier's stress_kinematics__wsm_density (uint32 count layer,
    mask-free)."""
    return density_dataarray(
        entry, tier, wsm_home_density(),
        "WSM 2025 应力观测点密度（每像元可定位记录数）",
        {"source_file": WSM_CSV_NAME, "count_semantics": WSM_COUNT_SEMANTICS},
    )


def build_gsrm_gps_density(entry: LayerEntry, tier: str) -> xr.DataArray:
    """Build this tier's stress_kinematics__gsrm_gps_density (uint32,
    internal)."""
    return density_dataarray(
        entry, tier, gsrm_home_density(),
        "GSRM GPS 站点观测密度（GPS_ITRF08.gmt 每像元记录数）",
        {
            "source_file": GSRM_GPS_NAME,
            "count_semantics": GSRM_COUNT_SEMANTICS,
            "doi": GSRM_DOI,
            "license_note": (
                "GSRM 组 internal-use-only：磁盘文件头 GEM Foundation "
                "CC-BY-NC-SA 3.0 与 Kreemer 邮件 CC BY 4.0 确认冲突未决，"
                "公开版不发布"
            ),
        },
    )



POINT_DENSITY_LAYERS = {
    WSM_LAYER_ID: {
        "src_dir": WSM_SRC_DIR,
        "home": wsm_home_density,
        "expected_total": WSM_EXPECTED_LOCATABLE,
    },
    GSRM_LAYER_ID: {
        "src_dir": GSRM_SRC_DIR,
        "home": gsrm_home_density,
        "expected_total": GSRM_EXPECTED_ROWS,
    },
}


def build_point_density_fidelity_report(store_dir, layer_id: str, tiers: list[str],
                                        src_dir=None) -> dict:
    """Fidelity report (count-layer criterion: total conservation):

    ① store bit-for-bit consistent with the recomputation from source (hard
       criterion—recomputation = grid-assigner bincount + sum kernel on the
       same code path, catching write/storage-chain misplacement);
    ② cross-tier sum consistency (hard criterion): coarse-tier store values ==
       sum-aggregation of home-tier store values—using the store itself as
       source, independent of ①;
    ③ count conservation (hard criterion, acceptance): each tier's count total
       == the count basis (WSM locatable 100672 / GSRM 22511);
    ④ dtype uint32 + value range ≥ 0 (hard criterion, coarse-error catcher);
    ⑤ occupied cells / max count / three counts (report-only).
    """
    spec = POINT_DENSITY_LAYERS[layer_id]
    home = spec["home"](src_dir) if src_dir else spec["home"]()

    checks: list[dict] = []
    values: dict[str, np.ndarray] = {}
    with xr.open_datatree(store_dir, engine="zarr", chunks={}) as tree:
        for tier in tiers:
            node = tree[f"/{tier}"]
            if layer_id not in node.ds.data_vars:
                raise KeyError(f"store {store_dir} 的 {tier} 档不含 {layer_id}（先 build 再 fidelity）")
            v = node.ds[layer_id].values
            values[tier] = v

            expect = home if tier == HOME_TIER else aggregate_sum(
                home, _aggregation_factor(tier)
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
            expect = aggregate_sum(values[HOME_TIER], _aggregation_factor(tier))
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

    if layer_id == GSRM_LAYER_ID:
        df = load_gsrm_gps(src_dir) if src_dir else load_gsrm_gps()
        three = (
            len(df),
            int(df.groupby(["lat", "lon"]).ngroups),
            int(df["site"].nunique()),
        )
        expect3 = (GSRM_EXPECTED_ROWS, GSRM_UNIQUE_COORDS, GSRM_UNIQUE_SITES)
        checks.append({
            "name": "GSRM 三数（行/唯一坐标/唯一定位站码）",
            "status": "pass" if three == expect3 else "FAIL",
            "summary": (
                f"{three[0]} 行 / {three[1]} 唯一坐标 / {three[2]} 唯一站码"
                f"（期望 {expect3[0]}/{expect3[1]}/{expect3[2]}；"
                "计数基准 = 观测记录行，见 manifest count_semantics）"
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
