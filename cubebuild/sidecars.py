"""矢量/点侧车导出器（侧车导出机制）。

机制：侧车随立方发布（GeoParquet 为主）；立方层只收
栅格化/计数产物，原始矢量/点走侧车。已就位：

- hasterok_gprv：global_gprv.shp → GeoParquet，原始 914 多边形与
  属性表全列保留 + prov_type_code 列（与立方层类别编码一致——
  编码表经此列直接可查，另见 manifest category_encoding）。
- limw_polygons：limw-gis-2015.gdb 以 FileGDB 原样传递（v1 不栅格化，
  v1.1 复评；.gdb 扩展名不可改——GDAL 按扩展名识别，catalog 注记），
  复制后按文件数与字节总量核对完整性。
- wsm2025_points：WSM_Database_2025.csv 全 100842 行 →
  GeoParquet（40 属性列全保留 + 点几何；170 行空坐标几何为空、
  密度层不计数——见该层 count_semantics）。
- gsrm_gps_velocities（internal）：GPS_ITRF08.gmt 全 22511 行
  站点速度 → GeoParquet（psvelo 9 列原值 + lon_180 卷绕列 + 点几何）
  + poles.* 4 框架欧拉极表合并 parquet；其余 52 参考系文件同站集仅
  速度随参考系旋转（已实证），不重复收录。
- heatflow_points_merged：GHFDB 2024 全 91182 条 + NGHF
  补充 5186 条（不落 GHFDB 0.1° 像元）合并去重单一点集 → GeoParquet
  （统一口径列 + source 溯源列 + source_row 回查行号 + 点几何）。
- pb2002_plates：plates（52 板块多边形，unwrap+极帽闭合+窗口
  裁剪标准几何 + plate_code 立方层编码列）/ boundaries（229 边界段，
  极性字节与引用，跨日期线 6 段切分 MultiLineString）/ poles（52 欧拉极）
  三件 parquet 目录侧车。
- gum_unconsolidated：GUM_v1.0 主文件 911551 多边形 +
  GUM_pyroclastics 碎屑流 20958 条合并单表（source 溯源列 +
  lithology_code/thickness_class 立方层编码列，DD 原值列保留）。
- mooney2023_provinces：GubanovMooney_July2022 六构造稳定化
  时代 MultiPolygon（Age/Area/layer/path 全列 + age_code 编码列）。
- global_basins（internal）：evenick2021_global_basins.shp 全 768
  多边形（= 764 盆地，4 跨日期线盆地各 2 拆分片，统计列为分片统计）
  + 全属性列 + basin_type_code 编码列；几何为 make_valid 修复后几何。

后续侧车在此登记 SIDECARS
复用导出框架；公开导出按 visibility 剥离 internal 侧车。

发布布局：sidecars/ 目录与 store 并列；manifest 记相对路径
"sidecars/<name>"。导出状态可复跑（覆盖式重写，内容确定性）。
"""

import shutil
from pathlib import Path

import numpy as np

from .points import (
    GSRM_GPS_NAME,
    GSRM_POLES_NAMES,
    GSRM_SRC_DIR,
    WSM_CSV_NAME,
    WSM_SRC_DIR,
    load_gsrm_gps,
    load_gsrm_poles,
    load_wsm_points,
    wsm_locatable_mask,
    wrap_lon180,
)
from .thermal import (
    GHFDB_EXPECTED_ROWS,
    GHFDB_SRC_DIR,
    GHFDB_XLSX_NAME,
    NGHF_CSV_NAME,
    NGHF_SRC_DIR,
    NGHF_SUPPLEMENT,
    merge_heatflow_points,
)
from .faults import (
    EXPECTED_ROWS as GEM_EXPECTED_ROWS,
    GEM_SRC_DIR,
    GPKG_REL_PATH,
    load_gem_faults,
)
from .pb2002 import (
    BOUNDARIES_NAME,
    EXPECTED_BOUNDARY_SEGMENTS,
    EXPECTED_PLATES,
    EXPECTED_POLES,
    PB2002_SRC_DIR,
    PLATES_NAME,
    POLES_NAME,
    ZONES_NAME,
    load_pb2002_boundaries,
    load_pb2002_plates,
    load_pb2002_poles,
    plate_window_geom,
    split_dateline_pts,
)
from .vector import GPRV_SRC_DIR, GPRV_REL_PATH, load_gprv
from .sediment import (
    BASINS_SHP_NAME,
    BASIN_TYPE_CODES,
    BASINS_SRC_DIR,
    EXPECTED_BASINS,
    EXPECTED_POLYGONS,
    load_global_basins,
)
from .landcat import (
    GUM_DOI,
    GUM_EXPECTED_ROWS,
    GUM_LITHOLOGY_CODES,
    GUM_PANGAEA_DOI,
    GUM_PYRO_EXPECTED_ROWS,
    GUM_PYRO_REL,
    GUM_MAIN_REL,
    GUM_SRC_DIR,
    MOONEY_AGE_CODES,
    MOONEY_DOI,
    MOONEY_REL_PATH,
    MOONEY_SRC_DIR,
    load_gum,
    load_gum_pyroclastics,
    load_mooney_provinces,
    thickness_class,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

LIMW_SRC_DIR = REPO_ROOT / "original data/lithosphere/glim-lithology"
LIMW_GDB_NAME = "limw-gis-2015.gdb"


def _tree_stats(path: Path) -> tuple[int, int]:
    """目录树 (文件数, 字节总量)——.gdb 原样传递的完整性核对基准。"""
    files = [p for p in path.rglob("*") if p.is_file()]
    return len(files), sum(p.stat().st_size for p in files)


def export_hasterok_gprv(out_dir: Path, src_dir: Path = GPRV_SRC_DIR) -> dict:
    """global_gprv.shp → GeoParquet 侧车（含立方层一致的 prov_type_code 列）。"""
    import geopandas as gpd

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gdf, codes = load_gprv(src_dir)
    gdf = gdf.copy()
    gdf["prov_type_code"] = codes
    out = out_dir / "hasterok_gprv.parquet"
    gdf.to_parquet(out)
    return {
        "id": "hasterok_gprv",
        "format": "geoparquet",
        "path": f"sidecars/{out.name}",
        "visibility": "public",
        "source": f"tectonic-provinces（{GPRV_REL_PATH}，914 构造省多边形 main 版）",
        "n_features": len(gdf),
        "columns": list(gdf.columns),
        "notes": (
            "原始多边形与属性表全列保留；prov_type_code = 立方层"
            " lithosphere__hasterok_province_class 类别编码（编码表另见该层 "
            "manifest 条目 category_encoding）；许可依据 Zenodo "
            "10.5281/zenodo.6586972（CC BY 4.0）"
        ),
    }


def export_limw_polygons(out_dir: Path, src_dir: Path = LIMW_SRC_DIR) -> dict:
    """limw .gdb 原样传递（不栅格化、不改扩展名；复制后完整性核对）。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    src = Path(src_dir) / LIMW_GDB_NAME
    if not src.is_dir():
        raise FileNotFoundError(f"LiMW 源目录不存在: {src}")
    n_files, n_bytes = _tree_stats(src)
    dst = out_dir / LIMW_GDB_NAME
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    n_files2, n_bytes2 = _tree_stats(dst)
    if (n_files, n_bytes) != (n_files2, n_bytes2):
        raise RuntimeError(
            f"LiMW .gdb 传递不完整: 源 {n_files} 文件/{n_bytes} B vs "
            f"目标 {n_files2} 文件/{n_bytes2} B"
        )
    return {
        "id": "limw_polygons",
        "format": "filegdb",
        "path": f"sidecars/{LIMW_GDB_NAME}",
        "visibility": "internal",
        "source": "glim-lithology（limw-gis-2015.gdb，Hartmann & Moosdorf 2012 LiMW）",
        "n_files": n_files,
        "size_bytes": n_bytes,
        "notes": (
            "FileGDB 原样传递（v1 不栅格化）；.gdb 扩展名不可改"
            "（GDAL 按扩展名识别）；内含图层 GLiM_export（MultiPolygon）；"
            "visibility=internal——源许可含商业限制，不纳入公开版；同条目"
            " GLiM 栅格层（CC BY 3.0）维持 public 不受本侧车影响"
        ),
    }


def export_wsm_points(out_dir: Path, src_dir: Path = WSM_SRC_DIR) -> dict:
    """WSM 2025 全量点表 → GeoParquet 侧车（40 属性列 + 点几何）。

    空坐标 170 行（QUALITY=Xmi）几何为空、原行保留——立方密度层只计
    可定位 100672 行（层 count_semantics），侧车是全量原始记录。
    应力方向 AZI/机制 TYPE/质量分级 QUALITY 等全部属性字段随行保留。
    """
    import geopandas as gpd

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_wsm_points(src_dir)
    has_xy = wsm_locatable_mask(df)          # 行数契约校验（与密度层同一实现）
    n_loc = int(has_xy.sum())
    geom = np.empty(len(df), dtype=object)
    geom[:] = None
    geom[has_xy] = gpd.points_from_xy(
        df.loc[has_xy, "LON"], df.loc[has_xy, "LAT"]
    )
    gdf = gpd.GeoDataFrame(df, geometry=geom, crs="EPSG:4326")
    out = out_dir / "wsm2025_points.parquet"
    gdf.to_parquet(out)
    return {
        "id": "wsm2025_points",
        "format": "geoparquet",
        "path": f"sidecars/{out.name}",
        "visibility": "public",
        "source": f"wsm2025-stress（{WSM_CSV_NAME}，100842 条应力观测记录）",
        "n_features": len(gdf),
        "n_locatable": n_loc,
        "columns": list(gdf.columns),
        "notes": (
            "全 40 属性列原样保留（应力方向 AZI/机制 TYPE/质量分级 QUALITY/"
            "体制 REGIME/S1-S3 产状等）；AZI=999 为占位值（11095 行），方向"
            "统计消费方须排除；170 行空坐标（TYPE=DIF/BO/HF，QUALITY=Xmi）"
            "几何为空、密度层不计数（见 stress_kinematics__wsm_density 的 "
            "count_semantics：计数基准 = 可定位 100672）"
        ),
    }


def export_gsrm_velocities(out_dir: Path, src_dir: Path = GSRM_SRC_DIR) -> dict:
    """GSRM GPS 站点速度 + 欧拉极表 → 目录侧车（internal）。

    gps_itrf08_velocities.parquet：GPS_ITRF08.gmt 全 22511 行（psvelo 9 列
    原值——lon 0..360 制；另附 lon_180 卷绕列与 EPSG:4326 点几何，几何
    逐行落格即还原密度层）。
    poles.parquet：poles.{APM,IGS08,NNR,PA} 4 框架欧拉极表合并（frame 列；
    lat lon rate(deg/Ma) plate，列序 lat 在前与 GPS 文件相反）。
    """
    import geopandas as gpd

    out_dir = Path(out_dir)
    out_root = out_dir / "gsrm_gps_velocities"
    out_root.mkdir(parents=True, exist_ok=True)
    df = load_gsrm_gps(src_dir)
    lon180 = wrap_lon180(df["lon"].to_numpy())
    gdf = gpd.GeoDataFrame(
        df.assign(lon_180=lon180),
        geometry=gpd.points_from_xy(lon180, df["lat"]),
        crs="EPSG:4326",
    )
    gdf.to_parquet(out_root / "gps_itrf08_velocities.parquet")
    poles = load_gsrm_poles(src_dir)
    poles.to_parquet(out_root / "poles.parquet")
    return {
        "id": "gsrm_gps_velocities",
        "format": "geoparquet+parquet",
        "path": "sidecars/gsrm_gps_velocities",
        "visibility": "internal",
        "source": (
            f"gsrm-strain（{GSRM_GPS_NAME} 观测站点速度 + "
            f"{'/'.join(GSRM_POLES_NAMES)} 欧拉极表）"
        ),
        "n_features": len(gdf),
        "n_poles": len(poles),
        "n_files": 2,
        "columns": list(gdf.columns),
        "notes": (
            "GPS_ITRF08.gmt 全部 22511 行站点速度（psvelo 9 列原值：lon 为 "
            "0..360 制，另附 lon_180 卷绕列与点几何）；其余 52 个参考系文件"
            "同站集仅速度随参考系旋转（已实证），不重复收录；欧拉极表 4 框架"
            "合并入 poles.parquet（frame 列，poles.PA 末 3 行为其他参考系相对"
            " PA 的旋转原样保留）；visibility=internal——源许可 "
            "CC-BY-NC-SA 3.0，不纳入公开版；计数语义见 "
            "stress_kinematics__gsrm_gps_density 的 count_semantics"
        ),
    }


def export_gem_active_faults(out_dir: Path, src_dir: Path = GEM_SRC_DIR) -> dict:
    """GEM 活动断层全量 → GeoParquet 侧车（internal）。

    13696 条 LineString 断裂 + 全部 23 属性列原样保留 + slip_type_code
    编码列（与立方层 stress_kinematics__gem_fault_slip_class 类别编码一致
    ——编码表经此列直接可查，另见 manifest category_encoding）。
    slip_type 缺失 319 条 code=0（层不栅格化，距离场仍计入）。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gdf, codes = load_gem_faults(src_dir)
    gdf = gdf.copy()
    gdf["slip_type_code"] = codes
    out = out_dir / "gem_active_faults.parquet"
    gdf.to_parquet(out)
    n_unclassified = int((codes == 0).sum())
    if len(gdf) != GEM_EXPECTED_ROWS:
        raise RuntimeError(
            f"gem_active_faults 侧车行数 {len(gdf)} ≠ 期望 {GEM_EXPECTED_ROWS}"
        )
    return {
        "id": "gem_active_faults",
        "format": "geoparquet",
        "path": f"sidecars/{out.name}",
        "visibility": "internal",
        "source": (
            f"gem-active-faults（{GPKG_REL_PATH}，harmonized 主版本 13696 断裂）"
        ),
        "n_features": len(gdf),
        "n_unclassified": n_unclassified,
        "columns": list(gdf.columns),
        "notes": (
            "全部 23 属性列原样保留（slip_type/average_dip/average_rake/"
            "net_slip_rate/seismic depths/质量分级等）+ slip_type_code 立方层"
            "编码列（0 = slip_type 缺失的 319 条，层不栅格化、距离场仍计入）；"
            "visibility=internal——源许可 CC BY-SA 4.0（SA 条款不符合开放"
            "许可要求），不纳入公开版，论文建议读者直接使用 GEM GAF-DB"
        ),
    }


def export_heatflow_points_merged(out_dir: Path,
                                  ghfdb_src: str | Path = GHFDB_SRC_DIR,
                                  nghf_src: str | Path = NGHF_SRC_DIR) -> dict:
    """热流合并点集 → GeoParquet 侧车（GHFDB 91182 + NGHF 补充 5186）。

    单一点集（克制原则综合为一），统一口径列：source（GHFDB/NGHF 溯源）/
    source_row（原始文件 0 基行号，回查全字段原始记录）
    name / lat / lon / q / q_uncertainty / reference / id（GHFDB 记录 ID；
    NGHF 无 ID 列为空）+ 点几何。逐行落格即还原
    thermal__heatflow_density（见该层 count_semantics）。
    """
    import geopandas as gpd

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = merge_heatflow_points(ghfdb_src, nghf_src)
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs="EPSG:4326",
    )
    out = out_dir / "heatflow_points_merged.parquet"
    gdf.to_parquet(out)
    n_g = int((gdf["source"] == "GHFDB").sum())
    n_n = int((gdf["source"] == "NGHF").sum())
    if (n_g, n_n) != (GHFDB_EXPECTED_ROWS, NGHF_SUPPLEMENT):
        raise RuntimeError(
            f"合并点集来源计数 ({n_g}, {n_n}) ≠ 期望 "
            f"({GHFDB_EXPECTED_ROWS}, {NGHF_SUPPLEMENT})"
        )
    return {
        "id": "heatflow_points_merged",
        "format": "geoparquet",
        "path": f"sidecars/{out.name}",
        "visibility": "internal",
        "source": (
            f"heatflow（{GHFDB_XLSX_NAME} {GHFDB_EXPECTED_ROWS} 条 + "
            f"{NGHF_CSV_NAME} 补充 {NGHF_SUPPLEMENT} 条，合并去重规则见 "
            "composite_rule）"
        ),
        "n_features": len(gdf),
        "n_ghfdb": n_g,
        "n_nghf": n_n,
        "columns": list(gdf.columns),
        "notes": (
            "GHFDB 为主、NGHF 仅补不与任何 GHFDB 记录同落 0.1° 像元的可定位"
            "点（合并去重规则与整编核实结论见 thermal__heatflow_density 的 "
            "composite_rule / count_semantics）；source 列逐记录溯源，"
            "source_row 回查原始文件全字段记录；NGHF 侧无 ID 列（id 为空）；"
            "visibility=internal——源再分发许可未明示（GHFDB 主体为 "
            "CC BY 4.0，但混合制品含 NGHF 记录本体），不纳入公开版；"
            "密度层为聚合分析产物（不可恢复源记录）维持公开；"
            "侧车逐行落格即还原密度层"
        ),
    }


def export_pb2002_plates(out_dir: Path, src_dir: Path = PB2002_SRC_DIR) -> dict:
    """PB2002 板块三件套 → 目录侧车。

    plates.parquet：52 板块多边形 GeoParquet（plate 两字母码 + plate_code
    立方层编码列 + [-180,180] 窗口标准几何——unwrap/极帽闭合/±360 平移
    裁剪，跨日期线部分正确收回窗口；AN 含南极帽、NA 含北极帽）。
    boundaries.parquet：229 边界段（code 5 字节：左板块-极性-右板块；
    极性 '/' = 右侧俯冲于左之下、'\\' = 反向、'-' = 非俯冲；引用注记；
    跨日期线 6 段切分为 MultiLineString）。
    poles.parquet：52 欧拉极（plate / lat / lon / rate_deg_per_ma CCW /
    引用；PA 为任意参考系零旋转，citation 原样保留）。
    """
    import geopandas as gpd
    from shapely.geometry import LineString, MultiLineString

    out_dir = Path(out_dir)
    out_root = out_dir / "pb2002_plates"
    out_root.mkdir(parents=True, exist_ok=True)

    polys, codes, labels = load_pb2002_plates(src_dir)
    poles = load_pb2002_poles(src_dir)
    plate_set, pole_set = set(labels), set(poles["plate"])
    if plate_set != pole_set:
        raise RuntimeError(
            f"板块码集合 ≠ 欧拉极板块码集合: 差集 "
            f"{sorted(plate_set ^ pole_set)}"
        )
    plates_gdf = gpd.GeoDataFrame(
        {
            "plate": labels,
            "plate_code": codes,
        },
        geometry=[plate_window_geom(p) for p in polys],
        crs="EPSG:4326",
    )
    plates_gdf.to_parquet(out_root / "plates.parquet")

    records = load_pb2002_boundaries(src_dir)
    rows = []
    for r in records:
        parts = split_dateline_pts(r["pts"])
        geom = (
            MultiLineString([LineString(p) for p in parts])
            if len(parts) > 1 else LineString(parts[0])
        )
        rows.append({
            "code": r["code"],
            "plate_left": r["plate_left"],
            "plate_right": r["plate_right"],
            "polarity": r["polarity"],
            "citation": r["citation"],
            "geometry": geom,
        })
    bounds_gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
    bounds_gdf.to_parquet(out_root / "boundaries.parquet")
    poles.to_parquet(out_root / "poles.parquet")

    if len(plates_gdf) != EXPECTED_PLATES:
        raise RuntimeError(
            f"pb2002_plates 侧车板块行数 {len(plates_gdf)} ≠ 期望 {EXPECTED_PLATES}"
        )
    if len(bounds_gdf) != EXPECTED_BOUNDARY_SEGMENTS:
        raise RuntimeError(
            f"pb2002_plates 侧车边界段行数 {len(bounds_gdf)} ≠ 期望 "
            f"{EXPECTED_BOUNDARY_SEGMENTS}"
        )
    if len(poles) != EXPECTED_POLES:
        raise RuntimeError(
            f"pb2002_plates 侧车欧拉极行数 {len(poles)} ≠ 期望 {EXPECTED_POLES}"
        )
    return {
        "id": "pb2002_plates",
        "format": "geoparquet+parquet",
        "path": "sidecars/pb2002_plates",
        "visibility": "internal",
        "source": (
            f"pb2002-plate-boundaries（{PLATES_NAME} 52 板块多边形 + "
            f"{BOUNDARIES_NAME} 229 边界段 + {POLES_NAME} 欧拉极表；"
            f"{ZONES_NAME} 构造区划图为立方层源不经侧车）"
        ),
        "n_features": len(plates_gdf),
        "n_boundaries": len(bounds_gdf),
        "n_poles": len(poles),
        "n_files": 3,
        "columns": list(plates_gdf.columns),
        "notes": (
            "plate_code = 立方层 stress_kinematics__pb2002_plate_id 类别编码"
            "（1..52 字典序，编码表另见该层 manifest 条目 category_encoding）；"
            "多边形为 unwrap（跨日期线短边走短弧）+ 极帽闭合（AN 南极/NA 北极）"
            "+ [-180,180] 窗口裁剪的标准几何，窗口并集面积精确平铺全球"
            "（64800.0 deg²）；MS 摩鹿加海零宽钉形环 make_valid 修复（面积 "
            "Δ=7e-15 deg²）；边界段 5 字节 code 第 3 位极性：'/' = 右侧板块"
            "俯冲于左侧之下、'\\' = 反向、'-' = 非俯冲，6 段跨日期线切分为 "
            "MultiLineString（切点平面线性插值，与源大圆弧偏差亚米级）；"
            "poles 为 PB2002_poles.dat.txt 解析版（.xls 二进制副本不消费，"
            "PA 板块为任意参考系零旋转原样保留）；visibility=internal——"
            "源无明示许可，同组三层+本侧车整组不纳入公开版"
        ),
    }


def export_gum_unconsolidated(out_dir: Path, src_dir: Path = GUM_SRC_DIR) -> dict:
    """GUM 全量（主文件 911551 + 碎屑流 20958）→ GeoParquet 侧车。

    属性全列保留（Symbol/Descriptio/XX/YY/ZZ/AA/DD/Shape_Leng/Shape_Area）
    + source 溯源列（GUM_v1.0 主文件 / GUM_pyroclastics 碎屑流，含碎屑流要素）
    + lithology_code（= 立方层 sediment__gum_
    lithology_class 类别编码；Ic 固结火山碎屑不在 41 类未固结词汇内，
    code=0 不参与栅格化）+ thickness_class（= sediment__gum_thickness_class
    有序分级，DD 原值列同时保留）。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gdf, _lith, _thick = load_gum(src_dir)
    main = gdf.copy()
    main["source"] = "GUM_v1.0"
    pyro = load_gum_pyroclastics(src_dir).copy()
    pyro["source"] = "GUM_pyroclastics"
    import geopandas as gpd
    import pandas as pd
    # 两文件属性同构（字段名实测一致）；OBJECTID 仅源内唯一，与 source
    # 列联合定位原始记录
    merged = gpd.GeoDataFrame(
        pd.concat([main, pyro], ignore_index=True), geometry="geometry", crs="EPSG:4326"
    )
    merged["lithology_code"] = (
        merged["XX"].map({**GUM_LITHOLOGY_CODES, "Ic": 0}).astype("uint8")
    )
    merged["thickness_class"] = merged["DD"].map(thickness_class).astype("uint8")
    out = out_dir / "gum_unconsolidated.parquet"
    merged.to_parquet(out)
    n_main = int((merged["source"] == "GUM_v1.0").sum())
    n_pyro = int((merged["source"] == "GUM_pyroclastics").sum())
    if (n_main, n_pyro) != (GUM_EXPECTED_ROWS, GUM_PYRO_EXPECTED_ROWS):
        raise RuntimeError(
            f"gum_unconsolidated 侧车来源计数 ({n_main}, {n_pyro}) ≠ 期望 "
            f"({GUM_EXPECTED_ROWS}, {GUM_PYRO_EXPECTED_ROWS})"
        )
    return {
        "id": "gum_unconsolidated",
        "format": "geoparquet",
        "path": f"sidecars/{out.name}",
        "visibility": "public",
        "source": (
            f"gum-unconsolidated（{GUM_MAIN_REL} {GUM_EXPECTED_ROWS} 多边形 + "
            f"{GUM_PYRO_REL} 碎屑流 {GUM_PYRO_EXPECTED_ROWS} 条）"
        ),
        "n_features": len(merged),
        "n_main": n_main,
        "n_pyroclastics": n_pyro,
        "columns": list(merged.columns),
        "notes": (
            "全部属性列原样保留（岩性 XX/粒度 YY/矿物 ZZ/年龄 AA/厚度 DD 与"
            " Symbol/Descriptio 等）；source 列区分主文件（未固结沉积物）与"
            " GUM_pyroclastics（火山碎屑：Ic 固结 20424 不属未固结词汇、立方"
            "层不栅格化 code=0，Iy 未固结 534 与主文件词汇一致）；"
            "lithology_code/thickness_class = 立方层 sediment__gum_"
            "lithology_class / sediment__gum_thickness_class 类别编码（编码表"
            "另见该层 manifest 条目 category_encoding/classification_rule）；"
            f"许可：PANGAEA {GUM_PANGAEA_DOI}（CC BY 3.0）；论文 DOI "
            f"{GUM_DOI}（勘误：早前所记 10.1029/2018GC007637 系笔误）"
        ),
    }


def export_mooney2023_provinces(out_dir: Path,
                                src_dir: Path = MOONEY_SRC_DIR) -> dict:
    """Mooney 2023 构造省 → GeoParquet 侧车。

    GubanovMooney_July2022.shp 六行（每构造稳定化时代一 MultiPolygon）
    属性全列保留（Age/Area/layer/path 原样，path 为源工程内路径痕迹）
    + age_code（= 立方层 lithosphere__crustal_age_class 编码，1..6 自老
    至新）。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gdf, codes = load_mooney_provinces(src_dir)
    gdf = gdf.copy()
    gdf["age_code"] = codes
    out = out_dir / "mooney2023_provinces.parquet"
    gdf.to_parquet(out)
    if len(gdf) != 6:
        raise RuntimeError(f"mooney2023_provinces 侧车行数 {len(gdf)} ≠ 期望 6")
    return {
        "id": "mooney2023_provinces",
        "format": "geoparquet",
        "path": f"sidecars/{out.name}",
        "visibility": "public",
        "source": f"crustal-age-mooney2023（{MOONEY_REL_PATH}，6 时代多边形）",
        "n_features": len(gdf),
        "columns": list(gdf.columns),
        "notes": (
            "Age = 构造稳定化时代（Archean/Cenozoic-Mesozoic/Mesoproterozoic/"
            "Neoproterozoic/Paleoproterozoic/Paleozoic，非出露地层时代）；"
            "Area 为测地面积（10⁶ km²，WGS84 椭球逐时代与 pyproj 核对一致）；"
            "age_code = 立方层 lithosphere__crustal_age_class 类别编码"
            "（1..6 自老至新，编码表另见该层 manifest 条目 category_encoding）；"
            "path/layer 列为源工程内痕迹原样保留；许可：作者邮件明示 "
            "CC BY 4.0 可再分发（许可登记）；论文 "
            f"DOI {MOONEY_DOI}（ECM1，Earth-Science Reviews）"
        ),
    }


def export_global_basins(out_dir: Path, src_dir: Path = BASINS_SRC_DIR) -> dict:
    """Global Basins 全量 → GeoParquet 侧车（internal）。

    768 源多边形行（4 跨日期线盆地各 2 拆分片——统计列为分片统计非
    盆地级，保留源粒度）+ 全部 38 属性列原样保留 + basin_type_code
    编码列（= 立方层 sediment__global_basins_class 类别编码）。几何列
    为 make_valid 修复后几何（Komandorskaya 自交修复；与立方层栅格化
    同一几何基，逐行落格即还原类别层）。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gdf, codes, geoms = load_global_basins(src_dir)
    gdf = gdf.copy()
    gdf["basin_type_code"] = codes
    gdf = gdf.set_geometry(geoms)
    out = out_dir / "global_basins.parquet"
    gdf.to_parquet(out)
    n_ubi = gdf["Basin UBI"].nunique()
    if len(gdf) != EXPECTED_POLYGONS or n_ubi != EXPECTED_BASINS:
        raise RuntimeError(
            f"global_basins 侧车行数 {len(gdf)}（盆地 {n_ubi}）≠ 期望 "
            f"{EXPECTED_POLYGONS}（{EXPECTED_BASINS}）"
        )
    return {
        "id": "global_basins",
        "format": "geoparquet",
        "path": f"sidecars/{out.name}",
        "visibility": "internal",
        "source": (
            f"global-basins（{BASINS_SHP_NAME}，768 多边形 = 764 盆地，"
            "4 跨日期线盆地各 2 拆分片）"
        ),
        "n_features": len(gdf),
        "n_basins": n_ubi,
        "columns": list(gdf.columns),
        "notes": (
            "全部属性列原样保留（Basin Name/UBI/Type/Salt/Volcanics/Setting/"
            "Geo Period/Age/沉积厚度与 Moho 统计/Poly Area 等）；Basin UBI 为"
            "盆地唯一键——4 个跨日期线盆地（Ross/Anadyr/Hope/Khatyrka）各为 "
            "2 拆分片（Split Poly=Yes，两片分落日期线两侧），统计列为分片"
            "统计非盆地级（磁盘实证同 UBI 两片数值不同），侧车保留源分片"
            "粒度；几何为 make_valid 修复后几何（Komandorskaya 自交修复，"
            "面积 Δ=3.5e-4 deg²，与立方层栅格化同一几何基）；"
            "basin_type_code = 立方层 sediment__global_basins_class 类别编码"
            f"（{len(BASIN_TYPE_CODES)} 类字典序 1..{len(BASIN_TYPE_CODES)}，"
            "编码表另见该层 manifest 条目 category_encoding）；"
            "visibility=internal——源许可 CC BY-NC-ND 4.0（ND 限衍生"
            "再分发），不纳入公开版，论文建议读者直接联用原源"
        ),
    }


# 侧车注册表：id → 导出器（后续侧车在此登记）；公开导出按 visibility 剥离。
# 调用方（cli）逐侧车调用导出器并自行决定源缺失的跳过策略。
SIDECARS = {
    "hasterok_gprv": export_hasterok_gprv,
    "limw_polygons": export_limw_polygons,
    "wsm2025_points": export_wsm_points,
    "gsrm_gps_velocities": export_gsrm_velocities,
    "heatflow_points_merged": export_heatflow_points_merged,
    "gem_active_faults": export_gem_active_faults,
    "pb2002_plates": export_pb2002_plates,
    "gum_unconsolidated": export_gum_unconsolidated,
    "mooney2023_provinces": export_mooney2023_provinces,
    "global_basins": export_global_basins,
}
