"""PB2002 板块三通道：plate_id 类别层 + zone_class 构造区类别层
+ boundary_distance 距离场 + pb2002_plates 侧车。

源（磁盘实证）：
- PB2002_plates.dig.txt：52 板块闭合多边形（counterclockwise，末点=首点）。
  解析陷阱（磁盘实证）：① 环坐标横跨日期线两侧（PA 4 次跨界），平面直连
  会把整幅地图横切（朴素平面多边形曾使阿尔卑斯点落入 PA）——须逐环
  unwrap 经度；② AN 环绕南极、NA 环绕北极（净经度缠绕 ±360°），平面
  多边形须补极点闭合顶点，否则 Jordan 内部取到错误一侧（极帽丢失）；
  ③ MS（摩鹿加海）环为源固有的零宽钉形（往复折返自交），make_valid
  修复仅清退零面积残件（面积 Δ=7e-15 deg²，栅格化逐位不变）。
  造山带（13 个）不占板块码位：PB2002 用任意内部线切割造山带闭板块
  （readme「overlay relationship」），故 52 多边形真穷尽覆盖（6′ 胞心
  栅格化实测：未覆盖 0、板块间重叠 0、窗口裁剪并集面积 = 64800.0 deg²
  精确平铺全球）。
- PB2002_boundaries.dig.txt：229 边界段、6048 点；标题 5 字节 = 左板块
  [0:2] + 极性 [2] + 右板块 [3:5]，'/' = 右侧俯冲于左之下、'\\' = 反向、
  '-' = 非俯冲（168/40/21 段）；引用注记字节 27 起（3 段无引用）；
  6 段跨日期线（球面 3D 向量引擎对日期线不敏感）。
- PB2002_poles.dat.txt：52 欧拉极，定宽 (A2,F9.3,F10.3,F9.4) + 引用列
  [32:]；前 3 行 56 列起另带 3 行格式图例（不属引用）。PB2002_poles.xls
  为二进制副本不消费。
- PB2002_tectonic_zones.grd.txt：Kagan, Bird & Jackson (2010, Pure Appl.
  Geophys. 167, 721–741, doi:10.1007/s00024-010-0075-3) 构造区划图
  （五区方案），Bird .GRD 格式（peterbird.name/guide/grd_format.htm）：
  行 1 = lon_min,d_lon,lon_max = 0,0.1,360；行 2 = lat_min,d_lat,lat_max =
  -90,0.1,90；数据从左上角（最大纬度、最小经度）起，行北→南、列西→东；
  1801×3601 节点 = 点样本（论文 2.2 节「assigns tectonic zone integers
  to grid points」），末列 = 首列（日期线卷绕副本，磁盘实证逐位相等；
  极区行全 0）。码表（论文 2.1 节 + PB2002_steps.dat 5819 步分类交叉表
  双重实证：大陆界 CCB/CRB/CTF→1、洋脊/转换 OSR/OTF→2/3、俯冲/汇聚
  SUB/OCB→4）：0=板块内部 1=活动大陆（含全部造山带的大陆部分+大陆板块
  边界）2=慢速扩张脊（<40 mm/a，含转换）3=快速扩张脊（≥40 mm/a，含
  转换）4=海沟（含初生俯冲、外隆与上盘）。

通道（三档 1°/30′/6′，主档 6′）：
- plate_id：主档 6′ 胞心栅格化（unwrap+极帽闭合多边形
  的 ±360 平移副本并 MultiPolygon 一次性烧录，面积降序后烧胜）+ 粗档
  kernels.aggregate_mode 众数聚合。穷尽覆盖免掩膜（板块距离
  由 boundary_distance 承担，plate_id 不产距离场）。
- zone_class：0.1° 节点点样本 → 6′ 胞心最近节点采样（胞心恰与 4 个
  相邻节点等距，取 4 角众数、平票取小编码——等价于确定性平票规则的
  最近邻采样；含经度卷绕与 ±90 边行）；源码位 +1 平移（1=板块内部 …
  5=海沟；0=缺测哨兵，源全网格有值故不出现，validity 恒 1）；粗档众数
  聚合（nodata=0 不参与、平票取小、≥1 有效子胞即有值）。
- boundary_distance：229 边界段复用距离引擎（层级环带候选剪枝
  引理保证不漏候选，三档逐档从矢量源精确重算，IUGG authalic R 球面
  大圆 km）；保真以同款独立航空公式抽样全段暴力比对复核。

许可：PB2002 源（peterbird.name/oldFTP/PB2002/）无明示许可，
仅列文件与引用格式；不纳入公开版。
"""

import json
import os
from pathlib import Path

import dask.array as da
import numpy as np
import xarray as xr

from .contract import LayerEntry
from .faults import distance_grids, independent_min_distance_km, segment_table
from .fidelity import INTEGRAL_TOL
from .grids import aggregation_factor, tier_centers, tier_chunks
from .kernels import aggregate_mode
from .vector import rasterize_polygon_classes

REPO_ROOT = Path(__file__).resolve().parents[1]
PB2002_SRC_DIR = REPO_ROOT / "original data/stress-kinematics/pb2002-plate-boundaries"
PLATES_NAME = "PB2002_plates.dig.txt"
BOUNDARIES_NAME = "PB2002_boundaries.dig.txt"
POLES_NAME = "PB2002_poles.dat.txt"
ZONES_NAME = "PB2002_tectonic_zones.grd.txt"

BIRD2003_DOI = "10.1029/2001GC000252"           # Bird 2003, G3 4(3), 1027
KAGAN2010_DOI = "10.1007/s00024-010-0075-3"     # Kagan et al. 2010, PAGEOPH 167, 721-741
PB2002_URL = "http://peterbird.name/oldFTP/PB2002/"
GRD_FORMAT_URL = "http://peterbird.name/guide/grd_format.htm"

PLATE_ID_LAYER_ID = "stress_kinematics__pb2002_plate_id"
ZONE_CLASS_LAYER_ID = "stress_kinematics__pb2002_zone_class"
BOUNDARY_DISTANCE_LAYER_ID = "stress_kinematics__pb2002_boundary_distance"
HOME_TIER = "6min"
DISTANCE_CHAIN = ("1deg", "30min", "6min")

EXPECTED_PLATES = 52
EXPECTED_WINDING = {"AN": -1, "NA": 1}   # 磁盘实证契约：AN 绕南极、NA 绕北极
EXPECTED_BOUNDARY_SEGMENTS = 229
EXPECTED_BOUNDARY_POINTS = 6048
EXPECTED_POLES = 52
NODATA = 0                          # 整型类别层缺测哨兵（码位自 1 起）

LICENSE_NOTE = (
    "PB2002 源（peterbird.name/oldFTP/PB2002/）无明示许可，仅列文件与引用"
    "格式；不纳入公开版。引用 Bird (2003) "
    f"doi:{BIRD2003_DOI}，区划图另引 Kagan et al. (2010) "
    f"doi:{KAGAN2010_DOI}"
)

# Kagan et al. (2010) 五区码表（源码 0..4；层码位 +1 平移自 1 起）。
ZONE_LABELS = {
    0: "plate-interior",
    1: "active-continent",
    2: "slow-spreading-ridge",
    3: "fast-spreading-ridge",
    4: "trench",
}

# 板块编码：52 个两字母码字典序固定编码 1..52（确定性）。
PLATE_CODES: dict[str, int] = {
    label: i for i, label in enumerate(sorted([
        "AF", "AM", "AN", "AP", "AR", "AS", "AT", "AU", "BH", "BR", "BS",
        "BU", "CA", "CL", "CO", "CR", "EA", "EU", "FT", "GP", "IN", "JF",
        "JZ", "KE", "MA", "MN", "MO", "MS", "NA", "NB", "ND", "NH", "NI",
        "NZ", "OK", "ON", "PA", "PM", "PS", "RI", "SA", "SB", "SC", "SL",
        "SO", "SS", "SU", "SW", "TI", "TO", "WL", "YA",
    ]), start=1)
}


def category_encoding() -> dict[str, str]:
    """板块编码表（码 → 语义），attrs/manifest/sidecar 共用唯一来源。"""
    return {str(c): label for label, c in sorted(PLATE_CODES.items(), key=lambda kv: kv[1])}


def zone_encoding() -> dict[str, str]:
    """构造区编码表（层码 → 语义；层码 = 源码 + 1）。"""
    return {str(c + 1): label for c, label in sorted(ZONE_LABELS.items())}


# ---------------- .dig 通用解析 ----------------

def _parse_dig(path: Path) -> list[dict]:
    """Bird .dig 文件 → [{'title', 'pts'}, ...]（*** end of line segment *** 分段）。"""
    segments: list[dict] = []
    cur: dict | None = None
    for ln in path.read_text(encoding="ascii").splitlines():
        s = ln.strip()
        if not s:
            continue
        if s.startswith("***"):
            if cur is not None:
                segments.append(cur)
            cur = None
            continue
        if cur is None:
            cur = {"title": s, "pts": []}
            continue
        lon, lat = (float(x) for x in s.split(","))
        cur["pts"].append((lon, lat))
    if cur is not None:
        segments.append(cur)
    return segments


# ---------------- 球面环 → 平面多边形（板块通道几何件） ----------------

def _ring_unwind(ring_pts: list[tuple[float, float]]):
    """闭合环 → (unwrap 后环, 净经度缠绕圈数)。

    相邻点经度差 |Δlon|>180 时 ±360 校正使环连续（跨日期线短边走短弧）；
    末点=首点（闭合环），缠绕 = Σadj/360。
    """
    a = np.asarray(ring_pts, dtype=np.float64)
    lon, lat = a[:, 0].copy(), a[:, 1]
    d = np.diff(lon)
    adj = np.zeros_like(d)
    adj[d > 180.0] = -360.0
    adj[d < -180.0] = 360.0
    lon[1:] += np.cumsum(adj)
    winding = int(round(adj.sum() / 360.0))
    return list(zip(lon.tolist(), lat.tolist())), winding


def _polygonal_only(geom):
    """make_valid 结果取面状组分（GeometryCollection 时剔除零面积线/点残件）。"""
    from shapely.ops import unary_union

    if geom.geom_type in ("Polygon", "MultiPolygon"):
        return geom
    parts = [g for g in geom.geoms if g.geom_type in ("Polygon", "MultiPolygon")]
    if not parts:
        raise ValueError("make_valid 后无面状组分——源几何异常")
    return unary_union(parts)


def plate_polygon(ring_pts: list[tuple[float, float]]):
    """板块闭合环 → 平面 shapely 多边形（unwrap + 极点闭合 + MS 型修复）。

    极点闭合：净缠绕 +1（东向绕极）补北极顶点，-1 补南极顶点——环
    绕极帽的平面投影不含极帽，须显式闭到极点（AN/NA 磁盘实证契约）。
    返回 (Polygon/MultiPolygon, winding)。
    """
    from shapely.geometry import Polygon
    from shapely.validation import make_valid

    ring, winding = _ring_unwind(ring_pts)
    if winding == 1:
        ring = ring + [(ring[-1][0], 90.0), (ring[0][0], 90.0)]
    elif winding == -1:
        ring = ring + [(ring[-1][0], -90.0), (ring[0][0], -90.0)]
    elif winding != 0:
        raise ValueError(f"板块环净缠绕 {winding}（>1 圈）：源形态异常")
    poly = Polygon(ring)
    if not poly.is_valid:
        # MS（摩鹿加海）型零宽钉形环：make_valid 清退零面积残件
        # （磁盘实证：面积 Δ=7e-15 deg²，6′ 栅格化逐位不变）
        poly = _polygonal_only(make_valid(poly))
    return poly, winding


def plate_window_geom(poly):
    """正确多边形 → [-180,180] 窗口内标准几何（三平移副本 ∩ 窗口的并集）。

    侧车用：unwrap 后越出 ±180 的部分经 ±360 平移副本收回窗口；只保留
    面状组分（窗口边界的线状残件剔除）。磁盘实证：52 板块窗口裁剪
    并集面积 = 64800.0 deg²（精确平铺全球框）。
    """
    from shapely.affinity import translate
    from shapely.geometry import box
    from shapely.ops import unary_union

    window = box(-180.0, -90.0, 180.0, 90.0)
    parts = []
    for shift in (0.0, 360.0, -360.0):
        clipped = translate(poly, shift).intersection(window)
        for g in (clipped.geoms if clipped.geom_type == "GeometryCollection"
                  else (clipped,)):
            if g.geom_type in ("Polygon", "MultiPolygon") and not g.is_empty:
                parts.append(g)
    if not parts:
        raise ValueError("板块多边形窗口裁剪后无面状组分——几何异常")
    return unary_union(parts)


def split_dateline_pts(pts: list[tuple[float, float]]) -> list[list[tuple[float, float]]]:
    """线段点列 → 跨日期线切分后的各 part（每 part 连续且落 [-180, 180)）。

    unwrap 后逐边检测跨越 180+360k 子午线（边界步长 ≤109 km，单边至多跨
    一条），切点为平面线性插值（与源大圆弧的偏差亚米级——切点在 ±180
    子午线上，两侧坐标经度重合、纬度差为直线/大圆插值差）；各 part 末端
    整体平移进 [-180, 180)。6 段跨日期线边界由此获得标准 GIS 几何。
    """
    up = [pts[0]]
    for lon, lat in pts[1:]:
        x_prev = up[-1][0]
        d = lon - x_prev
        if d > 180.0:
            lon -= 360.0
        elif d < -180.0:
            lon += 360.0
        up.append((lon, lat))

    parts: list[list[tuple[float, float]]] = []
    cur = [up[0]]
    for (x0, y0), (x1, y1) in zip(up, up[1:]):
        lo, hi = min(x0, x1), max(x0, x1)
        for k in range(-3, 4):
            b = 180.0 + 360.0 * k
            if lo <= b < hi:
                t = (b - x0) / (x1 - x0)
                yc = y0 + t * (y1 - y0)
                cur.append((b, yc))
                parts.append(cur)
                cur = [(b, yc)]
        cur.append((x1, y1))
    parts.append(cur)

    out = []
    for p in parts:
        mean_x = sum(x for x, _ in p) / len(p)
        shift = -360.0 * round(mean_x / 360.0)
        out.append([(x + shift, y) for x, y in p])
    return out


# ---------------- 源读取（契约化） ----------------

def load_pb2002_plates(src_dir: str | Path = PB2002_SRC_DIR):
    """读 PB2002_plates.dig.txt → (多边形列表, 编码数组, 板块码列表)。

    契约（磁盘实证的契约化）：52 段、两字母标题唯一、环闭合（末点=首点）、
    AN 绕南极（-1）/NA 绕北极（+1）净缠绕、修复后全部 valid 且非空。
    """
    path = Path(src_dir) / PLATES_NAME
    if not path.is_file():
        raise FileNotFoundError(f"PB2002 板块源文件不存在: {path}")
    segments = _parse_dig(path)
    if len(segments) != EXPECTED_PLATES:
        raise ValueError(f"{PLATES_NAME}: 段数 {len(segments)} ≠ 期望 {EXPECTED_PLATES}")

    labels: list[str] = []
    polys = []
    windings: dict[str, int] = {}
    for seg in segments:
        title = seg["title"].split()[0]
        if len(title) != 2 or not title.isalpha() or not title.isupper():
            raise ValueError(f"{PLATES_NAME}: 板块标题非法: {seg['title']!r}")
        if seg["pts"][0] != seg["pts"][-1]:
            raise ValueError(f"{PLATES_NAME}: 板块 {title} 环未闭合（末点≠首点）")
        poly, winding = plate_polygon(seg["pts"])
        if poly.is_empty or not poly.is_valid:
            raise ValueError(f"{PLATES_NAME}: 板块 {title} 多边形无效")
        labels.append(title)
        polys.append(poly)
        windings[title] = winding
    if len(set(labels)) != EXPECTED_PLATES:
        raise ValueError(f"{PLATES_NAME}: 板块码重复: {sorted(labels)}")
    if set(labels) != set(PLATE_CODES):
        raise ValueError(
            f"{PLATES_NAME}: 板块码集合 ≠ 编码表: 多出 "
            f"{sorted(set(labels) - set(PLATE_CODES))}，缺失 "
            f"{sorted(set(PLATE_CODES) - set(labels))}"
        )
    for plate, want in EXPECTED_WINDING.items():
        if windings.get(plate) != want:
            raise ValueError(
                f"{PLATES_NAME}: 板块 {plate} 净缠绕 {windings.get(plate)} ≠ 期望 "
                f"{want}（极点闭合契约破坏，源变更须复评）"
            )
    codes = np.array([PLATE_CODES[k] for k in labels], dtype=np.uint8)
    return polys, codes, labels


def load_pb2002_boundaries(src_dir: str | Path = PB2002_SRC_DIR) -> list[dict]:
    """读 PB2002_boundaries.dig.txt → 229 段属性记录（含坐标点列）。

    契约：229 段、6048 点、标题 5 字节（左右板块 ∈ 编码表、极性字节
    ∈ {-, /, \\}）、引用注记字节 27 起（可为空）。
    """
    path = Path(src_dir) / BOUNDARIES_NAME
    if not path.is_file():
        raise FileNotFoundError(f"PB2002 边界源文件不存在: {path}")
    segments = _parse_dig(path)
    if len(segments) != EXPECTED_BOUNDARY_SEGMENTS:
        raise ValueError(
            f"{BOUNDARIES_NAME}: 段数 {len(segments)} ≠ 期望 {EXPECTED_BOUNDARY_SEGMENTS}"
        )
    n_pts = sum(len(s["pts"]) for s in segments)
    if n_pts != EXPECTED_BOUNDARY_POINTS:
        raise ValueError(
            f"{BOUNDARIES_NAME}: 总点数 {n_pts} ≠ 期望 {EXPECTED_BOUNDARY_POINTS}"
        )

    records = []
    for seg in segments:
        title = seg["title"]
        code = title[:5]
        left, pol, right = code[:2], code[2], code[3:5]
        if left not in PLATE_CODES or right not in PLATE_CODES:
            raise ValueError(f"{BOUNDARIES_NAME}: 边界标题板块码非法: {title!r}")
        if pol not in ("-", "/", "\\"):
            raise ValueError(f"{BOUNDARIES_NAME}: 边界极性字节非法: {title!r}")
        if len(seg["pts"]) < 2:
            raise ValueError(f"{BOUNDARIES_NAME}: 段 {code} 点数 < 2")
        citation = title[27:].rstrip() if len(title) > 27 else ""
        records.append({
            "code": code,
            "plate_left": left,
            "plate_right": right,
            "polarity": pol,
            "citation": citation,
            "pts": seg["pts"],
        })
    return records


def load_pb2002_poles(src_dir: str | Path = PB2002_SRC_DIR):
    """读 PB2002_poles.dat.txt → pandas DataFrame（52 欧拉极）。

    定宽 (A2,F9.3,F10.3,F9.4)；引用列 [32:]，前 3 行 56 列起为 3 行
    格式图例（不属引用，按行号剔除）。契约：52 行、板块码 ⊆ 编码表且唯一。
    """
    import pandas as pd

    path = Path(src_dir) / POLES_NAME
    if not path.is_file():
        raise FileNotFoundError(f"PB2002 欧拉极源文件不存在: {path}")
    rows = []
    lines = path.read_text(encoding="ascii").splitlines()
    for i, line in enumerate(lines):
        if len(line) < 30:
            raise ValueError(f"{POLES_NAME}: 第 {i+1} 行短于 30 字符: {line!r}")
        plate = line[:2]
        if plate not in PLATE_CODES:
            raise ValueError(f"{POLES_NAME}: 第 {i+1} 行板块码非法: {plate!r}")
        # 引用列 [32:]；前 3 行 56 列起为 3 行格式图例（不属引用，按行号剔除）
        citation = line[32:56].rstrip() if i < 3 else line[32:].rstrip()
        rows.append({
            "plate": plate,
            "lat_deg": float(line[2:11]),
            "lon_deg": float(line[11:21]),
            "rate_deg_per_ma": float(line[21:30]),
            "citation": citation,
        })
    if len(rows) != EXPECTED_POLES:
        raise ValueError(f"{POLES_NAME}: 行数 {len(rows)} ≠ 期望 {EXPECTED_POLES}")
    if len({r["plate"] for r in rows}) != EXPECTED_POLES:
        raise ValueError(f"{POLES_NAME}: 板块码重复")
    return pd.DataFrame(rows)


def load_tectonic_zones(src_dir: str | Path = PB2002_SRC_DIR) -> np.ndarray:
    """读 PB2002_tectonic_zones.grd.txt → (1801, 3600) uint8 源码 0..4。

    Bird .GRD 格式（GRD_FORMAT_URL）：行北→南（首行 = 最大纬度 +90）、列
    西→东（lon 0..359.9）；节点点样本含端点。契约：头两行解析出
    (0, 0.1, 360)/(−90, 0.1, 90)；数值总数 = 1801×3601；末列 = 首列
    （日期线卷绕副本，去重前逐位相等）；值域 ⊆ {0..4}。
    """
    path = Path(src_dir) / ZONES_NAME
    if not path.is_file():
        raise FileNotFoundError(f"PB2002 构造区划源文件不存在: {path}")
    lines = path.read_text(encoding="ascii").splitlines()
    if len(lines) < 3:
        raise ValueError(f"{ZONES_NAME}: 文件过短")
    lon_min, d_lon, lon_max = (float(x) for x in lines[0].split())
    lat_min, d_lat, lat_max = (float(x) for x in lines[1].split())
    if not (lon_min == 0.0 and d_lon == 0.1 and lon_max == 360.0
            and lat_min == -90.0 and d_lat == 0.1 and lat_max == 90.0):
        raise ValueError(
            f"{ZONES_NAME}: 网格头 {lines[0]!r} / {lines[1]!r} ≠ 约定 "
            "(0, 0.1, 360)/(−90, 0.1, 90)（源变更须复评解析）"
        )
    values: list[int] = []
    for ln in lines[2:]:
        values.extend(int(t) for t in ln.split())
    nrows, ncols = 1801, 3601
    if len(values) != nrows * ncols:
        raise ValueError(
            f"{ZONES_NAME}: 数值总数 {len(values)} ≠ {nrows}×{ncols}"
        )
    g = np.array(values, dtype=np.uint8).reshape(nrows, ncols)
    if not np.array_equal(g[:, 0], g[:, -1]):
        raise ValueError(f"{ZONES_NAME}: 末列 ≠ 首列（日期线卷绕契约破坏）")
    bad = sorted(int(v) for v in np.unique(g) if v not in ZONE_LABELS)
    if bad:
        raise ValueError(f"{ZONES_NAME}: 值域越界 {bad}（期望 ⊆ 0..4，源变更须复评）")
    return g[:, :-1]          # 去掉卷绕副本列 → (1801, 3600)，行北→南


# ---------------- 通道一：plate_id（栅格化 + 众数聚合） ----------------

# 进程内缓存（层构建三档复用；测试 fixture 清理）
_PLATE_HOME_CACHE: dict[str, tuple] = {}
_PLATE_MODE_CACHE: dict[tuple, np.ndarray] = {}
_ZONE_HOME_CACHE: dict[str, np.ndarray] = {}
_ZONE_MODE_CACHE: dict[tuple, np.ndarray] = {}


def _plate_render_geoms(polys):
    """正确多边形 → 栅格化用 MultiPolygon（本体 + ±360 平移副本）。

    平移副本使跨日期线部分（unwrap 后 lon 越出 ±180）在 [-180,180] 窗口
    内正确落位；三副本互不重叠（unwrap 后环跨度 ≤360°+ε），单次烧录即可。
    """
    from shapely.affinity import translate
    from shapely.geometry import MultiPolygon

    return [
        MultiPolygon([p, translate(p, 360.0), translate(p, -360.0)])
        for p in polys
    ]


def _home_tier_classes(src_dir: str | Path = PB2002_SRC_DIR):
    """主档 6′ 类别网格 + 重叠/未覆盖计数（进程内缓存：三档只栅格化一次）。"""
    key = f"pb2002:{Path(src_dir).resolve()}"
    cache = _PLATE_HOME_CACHE.get(key)
    if cache is None:
        polys, codes, _labels = load_pb2002_plates(src_dir)
        cache = rasterize_polygon_classes(
            _plate_render_geoms(polys), codes, HOME_TIER
        )
        _PLATE_HOME_CACHE[key] = cache
    return cache


def _tier_classes(tier: str, src_dir: str | Path = PB2002_SRC_DIR) -> np.ndarray:
    if tier == HOME_TIER:
        return _home_tier_classes(src_dir)[0]
    key = (tier, str(Path(src_dir).resolve()))
    cache = _PLATE_MODE_CACHE.get(key)
    if cache is None:
        home = _home_tier_classes(src_dir)[0]
        cache = aggregate_mode(
            home, aggregation_factor(tier, HOME_TIER), nodata=NODATA
        )
        _PLATE_MODE_CACHE[key] = cache
    return cache


RASTERIZATION_RULE = (
    "板块闭合多边形 unwrap（跨日期线短边走短弧）+ 极帽闭合（AN 绕南极/NA "
    "绕北极补极点顶点）+ ±360 平移副本 MultiPolygon 单次烧录；胞心落入"
    "多边形即属该板块（all_touched=False）；MS 摩鹿加海"
    "零宽钉形环 make_valid 修复（面积 Δ=7e-15 deg²，栅格化逐位不变）；"
    "多板块重叠取面积较小者（面积降序后烧胜；实测重叠 0）；粗档 = 主档 "
    "6′ 众数聚合（kernels.aggregate_mode，nodata=0 不参与、平票取小、"
    "≥1 有效子胞即有值）"
)


def build_pb2002_plate_id(entry: LayerEntry, tier: str) -> xr.DataArray:
    """构建该档 stress_kinematics__pb2002_plate_id（uint8 类别层，穷尽覆盖）。"""
    values = _tier_classes(tier)
    lat, lon = tier_centers(tier)
    attrs = {
        "units": entry.unit,
        "long_name": "板块归属（Bird 2003 PB2002，52 板块，像元内主导板块）",
        "source": entry.source,
        "doi": BIRD2003_DOI,
        "source_url": PB2002_URL,
        "license_note": LICENSE_NOTE,
        "native_res": entry.native_res,
        "resampling": entry.resampling,
        "visibility": entry.visibility,
        "category_encoding": json.dumps(
            category_encoding(), ensure_ascii=False, sort_keys=True
        ),
        "rasterization_rule": RASTERIZATION_RULE,
        "source_file": PLATES_NAME,
        # 注意：不可命名 missing_value/_FillValue（zarr 后端按
        # CF 约定当 fill 编码，读回 0 被掩 NaN 且 dtype 升 float32）
        "nodata_semantics": (
            "无缺测：52 多边形穷尽覆盖（6′ 实测未覆盖 0 像元），免掩膜；"
            "类别编码自 1 起（0 保留为缺测哨兵，不出现）"
        ),
    }
    arr = da.from_array(values, chunks=tier_chunks(tier))
    return xr.DataArray(
        arr, dims=("lat", "lon"), coords={"lat": lat, "lon": lon},
        name=entry.id, attrs=attrs,
    )


# ---------------- 通道二：zone_class（节点采样 + 众数聚合） ----------------

def zone_home_grid(zones: np.ndarray) -> np.ndarray:
    """源节点网格 (1801, 3600)（行北→南，源码 0..4）→ 6′ 主档层码 (1800, 3600)。

    6′ 胞心与 4 个相邻源节点等距（源节点 0.1° 相位与档位网格相差半档），
    取 4 角众数、平票取小源码——等价于确定性平票规则的最近节点采样；
    经度列卷绕（±180 相邻）与 ±90 端行天然由节点网格含端点覆盖。
    返回 uint8 层码 1..5（源码 +1；0 = 缺测哨兵，源无缺测不出现）。
    """
    if zones.shape != (1801, 3600):
        raise ValueError(f"源节点网格形状 {zones.shape} ≠ (1801, 3600)")
    # 行：源行 r ↔ 纬度 90−0.1r；6′ 行 i（胞心 −90+0.1(i+0.5)）的
    # 北角节点 = 源行 1799−i、南角节点 = 源行 1800−i
    north = zones[:-1][::-1]          # [i] → 源行 1799−i
    south = zones[1:][::-1]           # [i] → 源行 1800−i
    # 列：源列 c ↔ 经度 0.1c；6′ 列 j（胞心 −180+0.1(j+0.5)）的
    # 西角节点 = 源列 (1800+j) mod 3600、东角节点 = 源列 (1801+j) mod 3600
    j = np.arange(3600)
    wc = (j + 1800) % 3600
    ec = (j + 1801) % 3600
    four = np.stack([north[:, wc], north[:, ec], south[:, wc], south[:, ec]])

    # 4 角众数（平票取小源码），码位 +1 平移后返回
    out = np.zeros((1800, 3600), dtype=np.uint8)
    best = np.zeros((1800, 3600), dtype=np.int64)
    for c in sorted(ZONE_LABELS):
        cnt = (four == c).sum(axis=0)
        take = cnt > best
        out[take] = c
        np.maximum(best, cnt, out=best)
    return (out + 1).astype(np.uint8)


def _zone_home(src_dir: str | Path = PB2002_SRC_DIR) -> np.ndarray:
    """主档 6′ 构造区网格（进程内缓存）。"""
    key = f"pb2002:{Path(src_dir).resolve()}"
    cache = _ZONE_HOME_CACHE.get(key)
    if cache is None:
        cache = zone_home_grid(load_tectonic_zones(src_dir))
        _ZONE_HOME_CACHE[key] = cache
    return cache


def _tier_zones(tier: str, src_dir: str | Path = PB2002_SRC_DIR) -> np.ndarray:
    if tier == HOME_TIER:
        return _zone_home(src_dir)
    key = (tier, str(Path(src_dir).resolve()))
    cache = _ZONE_MODE_CACHE.get(key)
    if cache is None:
        home = _zone_home(src_dir)
        cache = aggregate_mode(
            home, aggregation_factor(tier, HOME_TIER), nodata=NODATA
        )
        _ZONE_MODE_CACHE[key] = cache
    return cache


ZONE_INGESTION_RULE = (
    "源 = Kagan et al. (2010) 五区构造区划图（Bird .GRD 节点点样本，"
    "1801×3601，行北→南、列西→东 0..360，末列为日期线卷绕副本已去重）；"
    "6′ 主档 = 胞心最近节点采样（胞心与 4 相邻节点等距，取 4 角众数、"
    "平票取小源码，含经度卷绕与 ±90 端行）；源码 +1 平移入层（1=板块"
    "内部 … 5=海沟，0 = 缺测哨兵——源全网格有值不出现，validity 恒 1）；"
    "粗档 = 主档众数聚合（kernels.aggregate_mode，nodata=0 不参与、"
    "平票取小、≥1 有效子胞即有值）"
)


def build_pb2002_zone_class(entry: LayerEntry, tier: str) -> xr.DataArray:
    """构建该档 stress_kinematics__pb2002_zone_class（uint8 类别层）。"""
    values = _tier_zones(tier)
    lat, lon = tier_centers(tier)
    attrs = {
        "units": entry.unit,
        "long_name": (
            "构造区分类（Kagan et al. 2010 五区方案：板块内部/活动大陆/"
            "慢速扩张脊/快速扩张脊/海沟，基于 Bird 2003 PB2002）"
        ),
        "source": entry.source,
        "doi": KAGAN2010_DOI,
        "source_url": PB2002_URL,
        "license_note": LICENSE_NOTE,
        "native_res": entry.native_res,
        "resampling": entry.resampling,
        "visibility": entry.visibility,
        "category_encoding": json.dumps(
            zone_encoding(), ensure_ascii=False, sort_keys=True
        ),
        "ingestion_rule": ZONE_INGESTION_RULE,
        "source_file": ZONES_NAME,
        "nodata_semantics": (
            "无缺测：源网格全域有值（0=板块内部为真实类，源码 +1 平移后 "
            "1=板块内部；缺测哨兵 0 不出现），validity 恒 1"
        ),
    }
    arr = da.from_array(values, chunks=tier_chunks(tier))
    return xr.DataArray(
        arr, dims=("lat", "lon"), coords={"lat": lat, "lon": lon},
        name=entry.id, attrs=attrs,
    )


# ---------------- 通道三：boundary_distance（距离引擎复用） ----------------

# 距离金字塔磁盘缓存（内存护栏：.npy mmap 机制；本组三档
# 合计 ~80MB，护栏为机制一致而非内存必需）
_PYRAMID_CACHE_DIR = REPO_ROOT / "products/.build/pb2002-distance"

_PYRAMID_CACHE: dict[str, dict[str, np.ndarray]] = {}


def _distance_pyramid(src_dir: str | Path = PB2002_SRC_DIR) -> dict[str, np.ndarray]:
    """229 边界段 → {tier: float32 全球网格}（进程内缓存：三档只算一次）。

    复用距离引擎（faults.distance_grids）：层级环带候选剪枝，逐档
    从矢量源精确重算；跨日期线段对球面 3D 向量引擎无影响。
    """
    key = f"pb2002:{Path(src_dir).resolve()}"
    if key not in _PYRAMID_CACHE:
        records = load_pb2002_boundaries(src_dir)
        geoms = _boundary_geoms(records)
        segs, _n_drop = segment_table(geoms)
        _PYRAMID_CACHE[key] = distance_grids(
            segs, DISTANCE_CHAIN,
            workers=max(1, os.cpu_count() or 1),
            cache_dir=_PYRAMID_CACHE_DIR,
        )
    return _PYRAMID_CACHE[key]


def _boundary_geoms(records: list[dict]):
    """边界记录 → LineString 列表（引擎与保真共用）。"""
    from shapely.geometry import LineString

    return [LineString(r["pts"]) for r in records]


VALUE_CONVENTION = (
    "像元中心到最近板块边界线段（PB2002 229 段）的球面大圆距离（km，"
    "IUGG authalic R=6371.0071810 km，与像元面积层同球）；逐档从矢量源"
    "精确重算（非跨档降采样）——候选剪枝用层级环带方案（父档中心距离环带 "
    "±2×胞半对角线包含子档全体像元的 argmin，数学引理保证不漏候选），"
    "保真报告以独立航空公式抽样全段暴力比对复核"
)


def build_pb2002_boundary_distance(entry: LayerEntry, tier: str) -> xr.DataArray:
    """构建该档 stress_kinematics__pb2002_boundary_distance（float32，全域有值）。"""
    values = _distance_pyramid()[tier]
    lat, lon = tier_centers(tier)
    attrs = {
        "units": entry.unit,
        "long_name": "到最近板块边界（PB2002）的球面大圆距离",
        "source": entry.source,
        "doi": BIRD2003_DOI,
        "source_url": PB2002_URL,
        "license_note": LICENSE_NOTE,
        "native_res": entry.native_res,
        "resampling": entry.resampling,
        "visibility": entry.visibility,
        "value_convention": VALUE_CONVENTION,
        "source_file": BOUNDARIES_NAME,
        "nodata_semantics": "无缺测：距离场全域有值（免掩膜）",
    }
    arr = da.from_array(values, chunks=tier_chunks(tier))
    return xr.DataArray(
        arr, dims=("lat", "lon"), coords={"lat": lat, "lon": lon},
        name=entry.id, attrs=attrs,
    )


# ---------------- 保真验证（本层自供） ----------------

PB2002_FIDELITY_LAYERS = {
    PLATE_ID_LAYER_ID: {"kind": "plate_class"},
    ZONE_CLASS_LAYER_ID: {"kind": "zone_class"},
    BOUNDARY_DISTANCE_LAYER_ID: {"kind": "distance"},
}

_SAMPLE_SEED = 20260913        # 抽样确定性（复跑同样本）
_N_SAMPLES = 120              # 每档抽样像元数（分层：均匀 + 近场）
_DIST_TOL_KM = 1e-3           # 基础容差 1 m（近场公式异构 ~1e-5 km 量级）
_DIST_REL_TOL = 3e-7          # 远场 float32 存储 ULP 主导：tol = max(1e-3, 3e-7·|d|)


def build_pb2002_fidelity_report(store_dir, layer_id: str, tiers: list[str],
                                  src_dir=None) -> dict:
    """保真报告。

    plate_id 判据（同构 + 穷尽覆盖）：① store 与源重算逐位
    一致；② 粗档 = 主档 store 值众数聚合；③ 值域 ⊆ 编码表；④ 主档
    未覆盖像元 = 0（硬判据，穷尽覆盖）+ 类别分布/重叠总量（报告制）。

    zone_class 判据：①②③④ 同构（编码 1..5）；⑤ 主档类别份额 vs 源节点
    类别份额漂移（报告制，众数采样的类量守恒近似）。

    boundary_distance 判据（通道同构）：⑥ 每档分层随机抽样像元（均匀 +
    近场 <50 km）与独立航空公式全段暴力比对，tol = max(1e-3 km, 3e-7·|d|)
    （硬判据）；⑦ 全域有限且 ≥ 0（硬判据，免掩膜语义）；⑧ 距离分布统计
    （报告制）。
    """
    src_dir = src_dir or PB2002_SRC_DIR
    kind = PB2002_FIDELITY_LAYERS[layer_id]["kind"]
    checks: list[dict] = []

    if kind in ("plate_class", "zone_class"):
        if kind == "plate_class":
            home, overlap_cells, uncovered_cells = _home_tier_classes(src_dir)
            valid_codes = np.array(sorted(PLATE_CODES.values()))
        else:
            home = _zone_home(src_dir)
            valid_codes = np.arange(1, len(ZONE_LABELS) + 1)
            overlap_cells = uncovered_cells = None
        values: dict[str, np.ndarray] = {}
        with xr.open_datatree(store_dir, engine="zarr", chunks={}) as tree:
            for tier in tiers:
                node = tree[f"/{tier}"]
                if layer_id not in node.ds.data_vars:
                    raise KeyError(
                        f"store {store_dir} 的 {tier} 档不含 {layer_id}（先 build 再 fidelity）"
                    )
                values[tier] = node.ds[layer_id].values

        # ① 逐位一致（硬判据）
        for tier in tiers:
            expect = (
                _tier_classes(tier, src_dir) if kind == "plate_class"
                else _tier_zones(tier, src_dir)
            )
            n_bad = int((values[tier] != expect).sum())
            checks.append({
                "name": f"{tier}: 与源栅格化/采样重算逐位一致（uint8 位级）",
                "status": "pass" if n_bad == 0 else "FAIL",
                "summary": f"不一致像元 {n_bad}（期望 0）",
            })

        # ② 粗档 = 主档 store 值众数聚合（硬判据，独立于 ① 的跨档检验）
        for tier in tiers:
            if tier == HOME_TIER:
                continue
            expect = aggregate_mode(
                values[HOME_TIER], aggregation_factor(tier, HOME_TIER),
                nodata=NODATA,
            )
            n_bad = int((values[tier] != expect).sum())
            checks.append({
                "name": f"{tier}: 类别众数一致性（= 主档 {HOME_TIER} store 值众数聚合）",
                "status": "pass" if n_bad == 0 else "FAIL",
                "summary": f"不一致像元 {n_bad}（期望 0）",
            })

        # ③ 值域（硬判据；0 = 缺测哨兵合法——真实源穷尽覆盖不出现，
        # 合成源未覆盖像元由本判据放行、由「穷尽覆盖」判据单独把守）+ ④ 分布
        for tier in tiers:
            v = values[tier]
            bad = int((~np.isin(v, np.append(valid_codes, NODATA))).sum())
            checks.append({
                "name": f"{tier}: 值域 ⊆ 编码表（1..{len(valid_codes)}）",
                "status": "pass" if bad == 0 else "FAIL",
                "summary": f"非法值像元 {bad}（期望 0）",
            })
            counts = np.bincount(v.ravel(), minlength=len(valid_codes) + 1)
            if kind == "plate_class":
                dist = ", ".join(
                    f"{label}={counts[c]}" for label, c in
                    sorted(PLATE_CODES.items(), key=lambda kv: kv[1])
                )
            else:
                dist = ", ".join(
                    f"{ZONE_LABELS[c - 1]}={counts[c]}"
                    for c in range(1, len(ZONE_LABELS) + 1)
                )
            checks.append({
                "name": f"{tier}: 类别分布（报告制）",
                "status": "report",
                "summary": f"覆盖 {counts[1:].sum()}/{v.size}；{dist}",
            })

        if kind == "plate_class":
            # 穷尽覆盖（硬判据）+ 重叠总量（报告制）
            checks.append({
                "name": f"{HOME_TIER}: 穷尽覆盖（未覆盖像元 = 0）",
                "status": "pass" if uncovered_cells == 0 else "FAIL",
                "summary": f"未覆盖像元 {uncovered_cells}（期望 0；52 多边形平铺全球）",
            })
            checks.append({
                "name": f"{HOME_TIER}: 板块重叠总量（报告制）",
                "status": "report",
                "summary": (
                    f"多板块重叠像元 {overlap_cells}（规则：面积较小者胜；"
                    "源为共享数字化边界，预期 0）"
                ),
            })
        else:
            # ⑤ 主档类别份额 vs 源节点份额（报告制）
            src_g = load_tectonic_zones(src_dir)
            src_counts = np.bincount(src_g.ravel(), minlength=5)
            src_share = src_counts / src_g.size
            home_counts = np.bincount(home.ravel(), minlength=6)
            home_share = home_counts[1:] / home.size
            drift = home_share - src_share
            worst = int(np.argmax(np.abs(drift)))
            checks.append({
                "name": f"{HOME_TIER}: 类别份额 vs 源节点份额（报告制）",
                "status": "report",
                "summary": (
                    "; ".join(
                        f"{ZONE_LABELS[c]}: 源 {src_share[c]:.4%} → 层 {home_share[c]:.4%}"
                        for c in range(5)
                    )
                    + f"；最大漂移 {ZONE_LABELS[worst]} {drift[worst]:+.4%}"
                    "（半档相位最近节点采样的类量近似守恒）"
                ),
            })

    else:  # distance
        records = load_pb2002_boundaries(src_dir)
        (segs, n_drop) = segment_table(_boundary_geoms(records))
        rng = np.random.default_rng(_SAMPLE_SEED)
        with xr.open_datatree(store_dir, engine="zarr", chunks={}) as tree:
            for tier in tiers:
                node = tree[f"/{tier}"]
                if layer_id not in node.ds.data_vars:
                    raise KeyError(
                        f"store {store_dir} 的 {tier} 档不含 {layer_id}（先 build 再 fidelity）"
                    )
                v = node.ds[layer_id].values
                lat, lon = tier_centers(tier)

                # ⑦ 全域有限且 ≥ 0（硬判据）
                n_bad = int((~np.isfinite(v)).sum() + (v < 0).sum())
                checks.append({
                    "name": f"{tier}: 全域有限且 ≥ 0（免掩膜语义）",
                    "status": "pass" if n_bad == 0 else "FAIL",
                    "summary": f"违法像元 {n_bad}（期望 0）",
                })

                # ⑥ 分层随机抽样 × 独立航空公式全段暴力比对（硬判据）
                n_flat = v.size
                n_near = _N_SAMPLES // 2
                near_idx = np.nonzero((v < 50.0).ravel())[0]
                picks = [int(rng.integers(0, n_flat)) for _ in range(_N_SAMPLES - n_near)]
                if near_idx.size >= n_near:
                    picks += [int(x) for x in near_idx[rng.integers(0, near_idx.size, n_near)]]
                else:
                    picks += [int(x) for x in near_idx] + [
                        int(rng.integers(0, n_flat)) for _ in range(n_near - near_idx.size)
                    ]
                max_abs = 0.0
                max_ratio = 0.0
                worst = None
                for flat in picks:
                    r, c = divmod(flat, v.shape[1])
                    indep = independent_min_distance_km(
                        float(lat[r]), float(lon[c]), segs
                    )
                    d = abs(indep - float(v[r, c]))
                    tol = max(_DIST_TOL_KM, _DIST_REL_TOL * abs(indep))
                    ratio = d / tol
                    if ratio > max_ratio:
                        max_ratio = ratio
                        worst = (float(lat[r]), float(lon[c]), indep, float(v[r, c]), d, tol)
                    max_abs = max(max_abs, d)
                ok = max_ratio <= 1.0
                checks.append({
                    "name": (
                        f"{tier}: 抽样 {len(picks)} 像元 vs 独立大圆（航空公式全段暴力）"
                        f"一致（tol = max(1e-3 km, 3e-7·d)）"
                    ),
                    "status": "pass" if ok else "FAIL",
                    "summary": (
                        f"max|Δ| = {max_abs:.3e} km，max |Δ|/tol = {max_ratio:.3f}"
                        + (f"；最差点 {worst[:2]}（独立 {worst[2]:.6f} vs store "
                           f"{worst[3]:.6f} km，Δ={worst[4]:.2e}/tol={worst[5]:.2e}）"
                           if worst else "")
                    ),
                })

                # ⑧ 距离分布统计（报告制）
                checks.append({
                    "name": f"{tier}: 距离分布（报告制）",
                    "status": "report",
                    "summary": (
                        f"min {v.min():.3f} / p50 {np.percentile(v, 50):.2f} / "
                        f"mean {v.mean():.2f} / max {v.max():.2f} km；"
                        f"<1 km {float((v < 1).mean()):.4%}，<10 km {float((v < 10).mean()):.3%}，"
                        f"<100 km {float((v < 100).mean()):.3%}"
                    ),
                })

        checks.append({
            "name": "段表零长剔除（报告制）",
            "status": "report",
            "summary": f"连续重复顶点段 {n_drop} 条剔除（不参与距离场）",
        })

    n_fail = sum(1 for c in checks if c["status"] == "FAIL")
    return {
        "layer": layer_id,
        "store": str(store_dir),
        "integral_tolerance": INTEGRAL_TOL,   # 类别/距离层无积分量；报告器统一字段
        "result": "FAIL" if n_fail else "PASS",
        "n_checks": len(checks),
        "checks": checks,
    }
