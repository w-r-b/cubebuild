"""点密度通道：点 → 像元计数层（sum 核）。

立方不插值点数据：原始点全部留矢量侧车，立方只收密度/计数
层——本模块交付两个 uint32 密度层（免掩膜，0 即无观测）：
- stress_kinematics__wsm_density（public）：WSM 2025 应力观测点
- stress_kinematics__gsrm_gps_density（internal）：GSRM GPS 站点观测

WSM 源契约（磁盘实证）：
- WSM_Database_2025.csv：100842 数据行 × 40 列（表头具名；LAT 在前
  LON 在后；LON ±180 制）。xlsx 并存同内容，立方消费 csv。
- 170 行 LAT/LON 空（TYPE=DIF 77/BO 64/HF 29，QUALITY 全 Xmi）——
  无坐标不可落格：密度层计数基准 = 可定位 100672 行（count_semantics
  注记），侧车原样保留全 100842 行。
- AZI=999 为占位值（11095 行）——与密度层无关，侧车消费方须知。

GSRM 源契约（磁盘实证）：
- GPS_ITRF08.gmt：psvelo 9 列无表头 lon lat ve vn se sn corr site
  study；22511 行全可定位、无空值。列 1 经度 0..360 制（实测
  0.001..359.994）——落格前卷绕到 [-180, 180)。
- README 的 psvelo「0,0,0」不确定度描述只符合 vel-at-GPS_*.gmt 模型
  文件；观测文件第 5–7 列为真实 se/sn/corr（按实际 9 列解析）。
- 三数：22511 行 / 18440 唯一坐标 / 16681 唯一站码（psvelo 按研究
  一行一站，站码跨研究复用）。计数基准定死 = 观测记录行 22511（与
  WSM 记录行语义一致；侧车逐行落格即还原密度层）——三数与基准均入
  manifest count_semantics。
- 53 个 GPS_*.gmt 参考系文件同站重复（各 22511 行、同站集，已实证
  仅速度值随参考系旋转不同），禁跨系计数——密度层只取 ITRF08 单系。
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
GSRM_DOI = "10.1002/2014GC005407"      # Kreemer et al. 2014（DOI 自证）
HOME_TIER = "3min"

# 源契约常量（磁盘实证的契约化；源变更即读取期显式失败）
WSM_EXPECTED_ROWS = 100842
WSM_EXPECTED_COLS = 40
WSM_EXPECTED_LOCATABLE = 100672        # 100842 − 170 空坐标行
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


# ---------------- 源读取 ----------------

def load_wsm_points(src_dir: str | Path = WSM_SRC_DIR):
    """读 WSM_Database_2025.csv → DataFrame（全 40 列原样）。

    契约：100842 行 × 40 列、LAT/LON 列存在（空坐标行的处理见模块
    docstring 与 count_semantics——读取器不在此过滤，落格端统一 dropna）。
    NA 语义两全：keep_default_na=False 保留字面 "NA" 等字符串（pandas
    默认吞成 NaN——GSRM poles 北美板块码 NA 即踩此坑），空字段再显式
    replace("", NaN)（空 = 缺测的语义不丢）。
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
    """读 GPS_ITRF08.gmt → DataFrame（psvelo 9 列具名）。

    契约：22511 行 × 9 列、逐行 9 字段（psvelo 无表头；缺字段行在
    keep_default_na=False 下会静默变 ""，故逐行 token 数显式校验）。
    lon 保留 0..360 原值（卷绕在落格器内做，侧车另附 lon_180 列）。
    keep_default_na=False：站码/研究名字面 "NA" 原样保留（pandas 默认
    吞成 NaN——poles 北美板块码 NA 实证踩此坑）。
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
    """读 4 个 poles.* 欧拉极表 → 合并 DataFrame（frame 列区分框架）。

    格式（README 自证 + 磁盘实测）：lat lon rate(deg/Ma) plate 四列——
    列序 lat 在前，与 GPS 文件 lon 在前相反；poles.PA 末 3 行为其他
    参考系相对 PA 的旋转（负速率），原样保留。plate 码 "NA" = 北美
    板块（keep_default_na=False 防 pandas 吞成 NaN；缺字段行同样会
    静默变 ""，故逐行 token 数显式校验）。
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


# ---------------- 点 → 像元计数器（通道机制，热流点集复用） ----------------

def points_to_density(lat, lon, tier: str) -> np.ndarray:
    """点坐标 → 档位计数网格（uint32，纯函数，源无关）。

    语义：点落入像元 [−90+r·res, −90+(r+1)·res) × [−180+c·res,
    −180+(c+1)·res) 即计 1（行 0 = −90 侧、列 0 = −180 侧，与
    tier_centers 同一套行/列数学）。经度先卷绕到 [-180, 180)——WSM
    ±180 制与 GSRM 0..360 制均适用（180 ≡ −180 同一子午线 → 列 0）；
    lat=90 极点边缘归最北胞（数据无此点，防御性裁剪）。

    计数守恒：Σ网格 = len(lat)（构造性保证，落格器不做任何丢弃）。
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
    # lon_w ∈ [−180, 180) ⇒ col ∈ [0, nlon)；浮点残差防御性裁剪
    col = np.clip(np.floor((lon_w + 180.0) / res).astype(np.int64), 0, nlon - 1)
    counts = np.bincount(row * nlon + col, minlength=nlat * nlon)
    return counts.astype(np.uint32).reshape(nlat, nlon)


def wrap_lon180(lon) -> np.ndarray:
    """经度卷绕到 [-180, 180)（0..360 制 ↔ ±180 制统一；180 ≡ −180）。

    落格器与 GSRM 侧车 lon_180 列共用的唯一实现。
    """
    lon = np.asarray(lon, dtype=np.float64)
    return ((lon + 180.0) % 360.0) - 180.0


def _aggregation_factor(tier: str) -> int:
    """主档 3′ → 目标档聚合因子（grids.aggregation_factor 薄包装，模块内惯用名）。"""
    return aggregation_factor(tier, HOME_TIER)


_HOME_CACHE: dict[str, np.ndarray] = {}


def _home_density(cache_key: str, lat, lon) -> np.ndarray:
    """主档（3′）密度网格（进程内缓存：同层四档只落格一次）。"""
    if cache_key not in _HOME_CACHE:
        _HOME_CACHE[cache_key] = points_to_density(lat, lon, HOME_TIER)
    return _HOME_CACHE[cache_key]


def wsm_locatable_mask(df) -> np.ndarray:
    """WSM 表可定位行掩码（LAT/LON 非空）+ 行数契约校验（唯一实现）。

    密度层（落格）与侧车导出器（空几何）共用——两处口径必然一致。
    漂移即显式失败：源更新或空坐标行数变化须复核 count_semantics。
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
    """WSM 主档密度：可定位记录行（空坐标行不计数，count_semantics）。"""
    key = f"wsm:{Path(src_dir).resolve()}"
    if key not in _HOME_CACHE:
        df = load_wsm_points(src_dir)
        mask = wsm_locatable_mask(df)
        _HOME_CACHE[key] = points_to_density(
            df.loc[mask, "LAT"].to_numpy(), df.loc[mask, "LON"].to_numpy(), HOME_TIER
        )
    return _HOME_CACHE[key]


def gsrm_home_density(src_dir: str | Path = GSRM_SRC_DIR) -> np.ndarray:
    """GSRM 主档密度：GPS_ITRF08.gmt 全部 22511 观测记录行。"""
    key = f"gsrm:{Path(src_dir).resolve()}"
    if key not in _HOME_CACHE:
        df = load_gsrm_gps(src_dir)
        _HOME_CACHE[key] = points_to_density(
            df["lat"].to_numpy(), df["lon"].to_numpy(), HOME_TIER
        )
    return _HOME_CACHE[key]


# ---------------- 层构建 ----------------

def density_dataarray(entry: LayerEntry, tier: str, home: np.ndarray,
                      long_name: str, extra_attrs: dict) -> xr.DataArray:
    """密度层公共装配：主档直出 / 粗档 sum 聚合 + attrs 模板。

    点密度通道的共享装配（供 WSM/GSRM 与热流点集复用）。
    extra_attrs 至少含 source_file/count_semantics，可带 doi/license_note
    等白名单溯源键（manifest 透传）。
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
    """构建该档 stress_kinematics__wsm_density（uint32 计数层，免掩膜）。"""
    return density_dataarray(
        entry, tier, wsm_home_density(),
        "WSM 2025 应力观测点密度（每像元可定位记录数）",
        {"source_file": WSM_CSV_NAME, "count_semantics": WSM_COUNT_SEMANTICS},
    )


def build_gsrm_gps_density(entry: LayerEntry, tier: str) -> xr.DataArray:
    """构建该档 stress_kinematics__gsrm_gps_density（uint32，internal）。"""
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


# ---------------- 保真验证（本层自供） ----------------

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
    """保真报告（计数层判据：总和守恒）：

    ① store 与源重算逐位一致（硬判据——重算 = 落格器 bincount + sum 核
       同码路径，捕获写入/存储链路错位）；
    ② 跨档 sum 一致性（硬判据）：粗档 store 值 == 主档 store 值的 sum
       聚合——以 store 自身为源，独立于 ①；
    ③ 计数守恒（硬判据，验收）：各档计数总和 == 计数基准
       （WSM 可定位 100672 / GSRM 22511）；
    ④ dtype uint32 + 值域 ≥ 0（硬判据，粗错捕捉器）；
    ⑤ 占用像元/最大计数/三数（报告制）。
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

            # ① 逐位一致（硬判据）
            expect = home if tier == HOME_TIER else aggregate_sum(
                home, _aggregation_factor(tier)
            )
            n_bad = int((v != expect).sum())
            checks.append({
                "name": f"{tier}: 与源落格重算逐位一致（uint32 位级）",
                "status": "pass" if n_bad == 0 else "FAIL",
                "summary": f"不一致像元 {n_bad}（期望 0）",
            })

            # ③ 计数守恒（硬判据）
            total = int(v.sum(dtype=np.uint64))
            checks.append({
                "name": f"{tier}: 计数守恒（总和 == 计数基准 {spec['expected_total']}）",
                "status": "pass" if total == spec["expected_total"] else "FAIL",
                "summary": f"总和 {total} vs 基准 {spec['expected_total']}",
            })

            # ④ dtype + 值域（硬判据）
            bad = int((v < 0).sum())   # uint32 恒 0；防 dtype 漂移后负值
            checks.append({
                "name": f"{tier}: dtype uint32 且值域 ≥ 0",
                "status": "pass" if (v.dtype == np.uint32 and bad == 0) else "FAIL",
                "summary": f"dtype {v.dtype}，负值 {bad}（期望 0）",
            })

        # ② 跨档 sum 一致性（硬判据）：以 store 自身为源
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

        # ⑤ 占用像元/最大计数（报告制）
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

    # GSRM 三数（硬判据：源文件漂移即 FAIL）
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
        "integral_tolerance": INTEGRAL_TOL,   # 计数层无积分量；保留报告器统一字段
        "result": "FAIL" if n_fail else "PASS",
        "n_checks": len(checks),
        "checks": checks,
    }
