"""topography__bedrock_elevation —— ETOPO 2022 source adapter + area-weighted conservative kernel.

Source and mosaicking rules (disk-verified + ESSD paper MacFerrin et al. 2025, doi:10.5194/essd-17-1835-2025):
- ETOPO 2022 15s global = 288 15°×15° tiles (12 latitude bands × 24 longitude columns);
- the bed version has only 62 tiles (Greenland/Arctic/Antarctic ice-zone footprint),
  the remaining 226 positions are filled from the ice-surface version
  (in ice-free zones bed ≡ surface: both are bare-earth/seafloor DTM);
- mosaicking rule: bed first, surface fills gaps → global bedrock grid.
- mask ice-surface convention: a surface-first variant — the 62 ice-zone positions
  take bed_crosscheck/ surface tiles, the 226 ice-free positions take the source
  directory's surface tiles → global ice-surface grid (ice_surface_array, consumed
  only by derived__landsea_mask).

Naming convention (paper §2.2 + ngdc directory listing, measured):
ETOPO_2022_v1_15s_[N|S]YY[W|E]XXX_{bed|surface}.tif,
YY = absolute value of the tile's north-edge latitude. Measured correction:
the −15°..0° band is labeled N00 (north edge 0° takes the N prefix), not S00 —
the paper's rule derivation would give S00; go by the measured directory.

Resampling kernel: continuous variables = area-weighted conservative mean. 15s and
all tier resolutions are integer multiples (30″/3′/6′/30′/1° → factors 2/12/24/120/240),
each coarse pixel falls entirely within a single tile, the kernel = exact
integer-multiple block aggregation, with row weights equal to the latitude analytic
area R²·Δλ·(sin φ_hi − sin φ_lo). This kernel analytically conserves the integral
Σ v·A exactly (hence the fidelity check's <0.1% criterion can be tested exactly).
"""

from pathlib import Path

import dask.array as da
import numpy as np
import xarray as xr
from dask import delayed

from .contract import LayerEntry
from .grids import grid_shape, tier_centers, tier_deg
from .kernels import aggregate_weighted
from .pixel_area import EARTH_RADIUS_KM

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SRC_DIR = REPO_ROOT / "original data/topography/etopo2022-bed"
BED_CROSSCHECK_DIR = DEFAULT_SRC_DIR / "bed_crosscheck"

LAT_BANDS = ("N90", "N75", "N60", "N45", "N30", "N15",
             "N00", "S15", "S30", "S45", "S60", "S75")
LON_COLS = tuple(["W%03d" % d for d in range(180, 0, -15)]
                 + ["E%03d" % d for d in range(0, 180, 15)])

TILE_DEG = 15.0
TILE_SHAPE = (3600, 3600)          # 15s × 15°
SRC_RES_DEG = 15.0 / 3600.0        # 1/240°
DOI = "10.25921/fd45-gt74"
NODATA = -99999.0


class SourceNotReady(Exception):
    """Source tiles incomplete (missing surface fill tiles); the layer is skipped at build time with a message."""

    def __init__(self, missing: list[str]):
        self.missing = missing
        n = len(missing)
        sample = ", ".join(missing[:5]) + ("..." if n > 5 else "")
        super().__init__(
            f"ETOPO 源未集齐：缺 {n} 片 surface 瓦片（如 {sample}）；"
            "下载清单见下载台账"
        )


def label_bounds(label: str) -> tuple[float, float]:
    """Tile label → (north-edge latitude, west-edge longitude). E.g. N60W030 → (60, -30)."""
    lat_tok, lon_tok = label[:3], label[3:]
    top = int(lat_tok[1:]) * (1 if lat_tok[0] == "N" else -1)
    left = int(lon_tok[1:]) * (-1 if lon_tok[0] == "W" else 1)
    return float(top), float(left)


def all_labels() -> list[str]:
    return [b + c for b in LAT_BANDS for c in LON_COLS]


_ALL_LABELS = frozenset(all_labels())


def scan_tiles(src_dir: str | Path = DEFAULT_SRC_DIR) -> dict[str, Path]:
    """Scan the source directory → {tile label: file path}. bed first: when bed/surface coexist at the same label, take bed."""
    src_dir = Path(src_dir)
    if not src_dir.is_dir():
        raise FileNotFoundError(f"ETOPO 源目录不存在: {src_dir}")
    chosen = _scan_kind(src_dir, "surface")
    chosen.update(_scan_kind(src_dir, "bed"))
    return chosen


def missing_tiles(src_dir: str | Path = DEFAULT_SRC_DIR) -> list[str]:
    """Tile labels in the 288-position global grid not occupied by bed/surface."""
    have = scan_tiles(src_dir)
    return [lb for lb in all_labels() if lb not in have]


def require_full_coverage(src_dir: str | Path = DEFAULT_SRC_DIR) -> dict[str, Path]:
    """Global-coverage check: a missing tile raises SourceNotReady (listing the missing set)."""
    miss = missing_tiles(src_dir)
    if miss:
        raise SourceNotReady(miss)
    return scan_tiles(src_dir)


def scan_surface_tiles(
    src_dir: str | Path = DEFAULT_SRC_DIR,
    ice_surface_dir: str | Path = BED_CROSSCHECK_DIR,
) -> dict[str, Path]:
    """Global ice-surface tile selection (mask ice-surface convention).

    A switch of the tile-selection rule (not array post-processing): surface-first —
    the 62 ice-zone positions take ice_surface_dir (bed_crosscheck/, surface tiles that
    passed build-level validation), the 226 ice-free positions take src_dir's existing
    surface tiles; a bed tile at the same label is ignored, and when both directories
    have surface at the same label ice_surface_dir wins. The assembly is complementary
    to and independent of scan_tiles (bed-first composite), without interference.
    """
    src_dir, ice_dir = Path(src_dir), Path(ice_surface_dir)
    for d in (src_dir, ice_dir):
        if not d.is_dir():
            raise FileNotFoundError(f"ETOPO 源目录不存在: {d}")
    chosen = _scan_kind(src_dir, "surface")
    chosen.update(_scan_kind(ice_dir, "surface"))
    return chosen


def _scan_kind(src_dir: Path, kind: str) -> dict[str, Path]:
    """Scan a single directory → {tile label: file path}, collecting only the specified kind (bed/surface)."""
    chosen: dict[str, Path] = {}
    for f in sorted(src_dir.glob("ETOPO_2022_v1_15s_*_*.tif")):
        stem = f.name.removesuffix(".tif")          # ..._N60W030_surface
        label, _, k = stem.partition("_15s_")[2].rpartition("_")
        if k == kind and label in _ALL_LABELS:
            chosen[label] = f
    return chosen


def require_surface_coverage(
    src_dir: str | Path = DEFAULT_SRC_DIR,
    ice_surface_dir: str | Path = BED_CROSSCHECK_DIR,
) -> dict[str, Path]:
    """Global ice-surface coverage check: a missing tile raises SourceNotReady (listing the missing set)."""
    have = scan_surface_tiles(src_dir, ice_surface_dir)
    miss = [lb for lb in all_labels() if lb not in have]
    if miss:
        raise SourceNotReady(miss)
    return have


def row_areas_km2(lat_top: float, nrows: int, res_deg: float) -> np.ndarray:
    """Per-row pixel areas (km²): A = R²·Δλ_rad·(sin φ_hi − sin φ_lo), Δλ = res_deg."""
    edges = lat_top - np.arange(nrows + 1, dtype=np.float64) * res_deg
    rad = np.deg2rad(edges)
    dlon = np.deg2rad(res_deg)
    return EARTH_RADIUS_KM**2 * dlon * (np.sin(rad[:-1]) - np.sin(rad[1:]))


def read_tile_validated(path: Path) -> np.ndarray:
    """Read a tile → float32 array; any nodata pixel fails explicitly.

    ETOPO tiles are globally valid (disk-verified for the 62 bed tiles; once surface tiles
    arrive they undergo the same validation, and if they contain nodata the build crashes
    rather than silently producing biased values). The build and fidelity statistics share
    this reader, keeping the two paths' nodata policies consistent.
    """
    import rasterio

    with rasterio.open(path) as src:
        if src.shape != TILE_SHAPE:
            raise ValueError(f"{path.name}: 形状 {src.shape} ≠ {TILE_SHAPE}")
        data = src.read(1).astype(np.float32)
        nodata = src.nodata if src.nodata is not None else NODATA
    if np.any(data == nodata):
        raise ValueError(f"{path.name}: 含 nodata 像元，与全球全覆盖预期不符")
    return data


def read_tile_aggregated(path: Path, tier: str, lat_top: float) -> np.ndarray:
    """Read a single tile and aggregate to the target tier: output that tile's sub-grid block at the tier (float32).

    factor = tier resolution / 15s (integer; the 15° tile splits integrally at every tier).
    Output row-order convention: row 0 = the tile's southernmost row (global grid row 0 = −90°).
    rasterio reads row 0 = the tile's north edge, so after aggregation a single vertical flip
    aligns to the global convention — omitting the flip would reverse the within-tile latitude
    gradient: the integral would be off ~1e-3 and false seams would appear at 15° boundaries
    (caught empirically by the fidelity harness, which is exactly its design purpose).
    """
    factor = int(round(tier_deg(tier) / SRC_RES_DEG))
    if abs(tier_deg(tier) / SRC_RES_DEG - factor) > 1e-9:
        raise ValueError(f"档位 {tier} 相对 15s 非整数倍，保守核不适用")
    data = read_tile_validated(path)
    w = row_areas_km2(lat_top, TILE_SHAPE[0], SRC_RES_DEG)
    out = aggregate_weighted(data, w, factor)
    return out[::-1].astype(np.float32)


def _assemble_tiles(tiles: dict[str, Path], tier: str) -> da.Array:
    """Tile-selection result → the tier's global dask array (blocks = 15° tile boundaries, float32).

    bed-first (elevation_array) and surface-first (ice_surface_array) share this assembly
    step: row 0 = −90°, LAT_BANDS runs north to south → reversed.
    """
    nlat, nlon = grid_shape(tier)
    ot = TILE_SHAPE[0] // int(round(tier_deg(tier) / SRC_RES_DEG))
    on = TILE_SHAPE[1] // int(round(tier_deg(tier) / SRC_RES_DEG))
    if nlat % ot or nlon % on:
        raise ValueError(f"档位 {tier}: 瓦块 ({ot},{on}) 不整除全球网格")

    blocks = []
    for band in reversed(LAT_BANDS):
        row_blocks = []
        for col in LON_COLS:
            label = band + col
            row_top, _ = label_bounds(label)
            blk = delayed(read_tile_aggregated, pure=True)(
                tiles[label], tier, row_top
            )
            row_blocks.append(
                da.from_delayed(blk, shape=(ot, on), dtype=np.float32)
            )
        blocks.append(row_blocks)
    return da.block(blocks)


def elevation_array(tier: str, src_dir: str | Path = DEFAULT_SRC_DIR) -> da.Array:
    """The tier's bedrock-elevation dask array (bed-first composite, blocks = 15° tile boundaries, float32).

    The elevation layer shares this entry point with the slope/relief derived layers: pure
    delayed with the same args and keys, so when writing the same tier the dask scheduler
    computes the shared subgraph only once. The land-sea mask (ice-surface convention)
    instead goes through ice_surface_array, independent of this entry point.
    """
    return _assemble_tiles(require_full_coverage(src_dir), tier)


def ice_surface_array(
    tier: str,
    src_dir: str | Path = DEFAULT_SRC_DIR,
    ice_surface_dir: str | Path = BED_CROSSCHECK_DIR,
) -> da.Array:
    """The tier's global ice-surface elevation dask array (mask ice-surface convention).

    Tile selection = surface-first (the 62 ice-zone positions from bed_crosscheck/, the 226
    ice-free positions from the build source directory's surface); the assembly step is
    shared with elevation_array (_assemble_tiles). Consumed only by derived__landsea_mask
    thresholding; the elevation layer (bare-earth composite convention) and slope/relief are
    unaffected.
    """
    return _assemble_tiles(
        require_surface_coverage(src_dir, ice_surface_dir), tier
    )


def build_etopo_bedrock(entry: LayerEntry, tier: str) -> xr.DataArray:
    """Build topography__bedrock_elevation for the given tier (dask-lazy, blocks = 15° tile boundaries)."""
    arr = elevation_array(tier)

    lat, lon = tier_centers(tier)
    return xr.DataArray(
        arr,
        dims=("lat", "lon"),
        coords={"lat": lat, "lon": lon},
        name=entry.id,
        attrs={
            "units": entry.unit,
            "long_name": "基岩高程（ETOPO 2022 bed 版，冰区基岩/冰下地形）",
            "source": entry.source,
            "doi": DOI,
            "native_res": entry.native_res,
            "resampling": entry.resampling,
            "composite_rule": (
                "62 bed tiles（冰区足迹，bed 优先）+ 226 surface tiles 补齐"
                "（无冰区 bed≡surface，ETOPO 为裸地 DTM）；288 位全球网格"
            ),
            "vertical_datum": "EGM2008 geoid (EPSG:3855)",
            "visibility": entry.visibility,
        },
    )
