"""Vector/point sidecar exporter (the sidecar export mechanism).

Mechanism: sidecars are published alongside the cube (mostly GeoParquet); the
cube layers only ingest
rasterized/count products, while raw vectors/points go through sidecars. In
place:

- hasterok_gprv: global_gprv.shp → GeoParquet, all original 914 polygons and
  attribute-table columns preserved + a prov_type_code column (consistent with
  the cube-layer class encoding—
  the encoding table is directly retrievable through this column; see also the
  manifest category_encoding).
- limw_polygons: limw-gis-2015.gdb passed through as FileGDB (not rasterized in
  v1, revisited in
  v1.1; the .gdb extension must not be changed—GDAL identifies by extension,
  catalog note), integrity
  checked by file count and total bytes after copying.
- wsm2025_points: WSM_Database_2025.csv all 100842 rows →
  GeoParquet (all 40 attribute columns preserved + point geometry; the 170
  empty-coordinate rows have empty geometry and are not counted by the
  density layer—see that layer's count_semantics).
- gsrm_gps_velocities (internal): GPS_ITRF08.gmt all 22511 rows
  of station velocities → GeoParquet (psvelo 9 raw columns + a lon_180 wrapped
  column + point geometry)
  + a merged parquet of the poles.* Euler-pole tables across 4 frames; the
  other 52 reference-frame files have the same station set with only velocities
  differing by frame rotation (verified), and are not duplicated.
- heatflow_points_merged: GHFDB 2024 all 91182 records + the NGHF
  supplement 5186 records (not falling into GHFDB 0.1° cells) merged and
  deduplicated into a single point set → GeoParquet
  (unified schema columns + a source provenance column + a source_row
  back-reference row number + point geometry).
- pb2002_plates: plates (52 plate polygons, unwrap + polar-cap closure + window
  clipping standard geometry + a plate_code cube-layer encoding
  column) / boundaries (229 boundary segments,
  polarity bytes and citation, 6 date-line-crossing segments split into
  MultiLineString) / poles (52 Euler poles)
  three-parquet directory sidecar.
- gum_unconsolidated: GUM_v1.0 main file 911551 polygons +
  GUM_pyroclastics pyroclastic flows 20958 records merged into a single table
  (a source provenance column +
  lithology_code/thickness_class cube-layer encoding columns, the DD raw-value
  column preserved).
- mooney2023_provinces: GubanovMooney_July2022 six tectonic-stabilization age
  MultiPolygons (all Age/Area/layer/path columns + an age_code encoding
  column).
- global_basins (internal): evenick2021_global_basins.shp all 768
  polygons (= 764 basins, 4 date-line-crossing basins each with 2 split parts,
  the statistics columns are per-part statistics)
  + all attribute columns + a basin_type_code encoding column; the geometry is
  the make_valid-repaired geometry.

Subsequent sidecars register in SIDECARS here to
reuse the export framework; the public export strips internal sidecars by
visibility.

Publish layout: the sidecars/ directory sits alongside the store; the manifest
records the relative path
"sidecars/<name>". The export is rerunnable (overwrite rewrite, deterministic
content).
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
    """Directory tree (file count, total bytes)—the integrity-check baseline for
    the .gdb pass-through."""
    files = [p for p in path.rglob("*") if p.is_file()]
    return len(files), sum(p.stat().st_size for p in files)


def export_hasterok_gprv(out_dir: Path, src_dir: Path = GPRV_SRC_DIR) -> dict:
    """global_gprv.shp → GeoParquet sidecar (including a prov_type_code column
    consistent with the cube layer)."""
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
    """limw .gdb pass-through (no rasterization, no extension change; integrity
    checked after copying)."""
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
    """WSM 2025 full point table → GeoParquet sidecar (40 attribute columns +
    point geometry).

    Empty geometry for the 170 empty-coordinate rows (QUALITY=Xmi), original
    rows kept—the cube density layer counts only the 100672 locatable rows
    (layer count_semantics); the sidecar is the full raw record. All attribute
    fields such as stress direction AZI / mechanism TYPE / quality grade
    QUALITY are kept per row.
    """
    import geopandas as gpd

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_wsm_points(src_dir)
    has_xy = wsm_locatable_mask(df)
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
    """GSRM GPS station velocities + Euler-pole tables → directory sidecar
    (internal).

    gps_itrf08_velocities.parquet: GPS_ITRF08.gmt all 22511 rows (psvelo 9 raw
    columns—lon in 0..360 convention; plus a lon_180 wrapped column and
    EPSG:4326 point geometry; assigning each geometry row to a cell reproduces
    the density layer).
    poles.parquet: poles.{APM,IGS08,NNR,PA} Euler-pole tables for 4 frames
    merged (frame column; lat lon rate(deg/Ma) plate, with lat first, opposite
    to the GPS files).
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
    """GEM active faults full set → GeoParquet sidecar (internal).

    13696 LineString fault traces + all 23 attribute columns preserved + a
    slip_type_code encoding column (consistent with the cube-layer
    stress_kinematics__gem_fault_slip_class class encoding
    —the encoding table is directly retrievable through this column; see also
    the manifest category_encoding). 319 records with missing slip_type have
    code=0 (the layer is not rasterized; the distance field still counts them).
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
    """Heat-flow merged point set → GeoParquet sidecar (GHFDB 91182 + NGHF
    supplement 5186).

    A single point set (combined into one by the restraint principle), with
    unified schema columns: source (GHFDB/NGHF provenance) /
    source_row (0-based row number in the original file, for back-referencing
    all fields of the raw record)
    name / lat / lon / q / q_uncertainty / reference / id (GHFDB record ID;
    empty for NGHF, which has no ID column) + point geometry. Assigning each row
    to a cell reproduces
    thermal__heatflow_density (see that layer's count_semantics).
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
    """PB2002 plate triplet → directory sidecar.

    plates.parquet: 52 plate-polygon GeoParquet (two-letter plate code +
    plate_code cube-layer encoding column + [-180,180] window standard
    geometry—unwrap/polar-cap closure/±360 shift clipping, date-line-crossing
    parts correctly brought back into the window; AN includes the Antarctic cap,
    NA the Arctic cap).
    boundaries.parquet: 229 boundary segments (code 5 bytes: left plate- polarity
    -right plate; polarity '/' = the right side subducts beneath the left,
    '\\' = the reverse, '-' = non-subducting; citation note; 6 date-line-crossing
    segments split into MultiLineString).
    poles.parquet: 52 Euler poles (plate / lat / lon / rate_deg_per_ma CCW /
    citation; PA is an arbitrary-frame zero rotation, citation kept as-is).
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
    """GUM full set (main file 911551 + pyroclastic flows 20958) → GeoParquet
    sidecar.

    All attribute columns preserved (Symbol/Descriptio/XX/YY/ZZ/AA/DD/
    Shape_Leng/Shape_Area)
    + a source provenance column (GUM_v1.0 main file / GUM_pyroclastics
    pyroclastic flows, including pyroclastic features)
    + lithology_code (= cube-layer sediment__gum_
    lithology_class class encoding; Ic consolidated volcaniclastics are outside
    the 41-class unconsolidated vocabulary,
    code=0, not rasterized) + thickness_class (= sediment__gum_thickness_class
    ordered classification; the DD raw-value column is also preserved).
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
    """Mooney 2023 tectonic provinces → GeoParquet sidecar.

    GubanovMooney_July2022.shp six rows (one MultiPolygon per
    tectonic-stabilization age)
    with all attribute columns preserved (Age/Area/layer/path as-is; path is a
    source-project path trace)
    + age_code (= cube-layer lithosphere__crustal_age_class encoding, 1..6 from
    old to new).
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
    """Global Basins full set → GeoParquet sidecar (internal).

    768 source polygon rows (4 date-line-crossing basins each with 2 split
    parts—the statistics columns are per-part statistics, not basin-level; the
    source granularity is kept) + all 38 attribute columns preserved + a
    basin_type_code encoding column (= cube-layer
    sediment__global_basins_class class encoding). The geometry column is the
    make_valid-repaired geometry (Komandorskaya self-intersection repaired; the
    same geometry basis as the cube-layer rasterization, so assigning each row
    to a cell reproduces the class layer).
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
