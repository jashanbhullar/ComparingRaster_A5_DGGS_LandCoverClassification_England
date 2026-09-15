"""Single source of truth for paths, CRS, bands, A5 levels and the class schema."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# --- AOI ladder: london -> england -> uk. Change only this to scale up. -------
AOI_NAME = "london"

AOI_OSM_RELATION = {
    "london": 175342,
    "england": 58447,
    "uk": 62149,
}

# Where each AOI boundary comes from. Both public AOIs are small checked-in
# GeoJSON files, so a run does not require a large OSM or LSOA source dataset.
AOI_SOURCE = {
    "london": ("geojson", ROOT / "aoi" / "london.geojson"),
    "england": ("geojson", ROOT / "aoi" / "england.geojson"),
    "uk": ("lsoa", ""),
}

# Boundary detail below the pixel size is dropped so cutlines stay cheap.
AOI_SIMPLIFY_DEG = 1e-4

# --- CRS ---------------------------------------------------------------------
# EPSG:4326 everywhere; area and distance calculations must account for geographic distortion.
CRS = "EPSG:4326"

# Option B grid: the degree steps in x and y are chosen so that ground pixels are
# square at the AOI centroid latitude, instead of using one degree step for both.
# Resolved per AOI by raster.grid_spec() and written to processed/<aoi>/grid.json.
# TARGET_GROUND_RES_M = None keeps the source's native ground resolution.
TARGET_GROUND_RES_M: float | None = None
RESAMPLING = "bilinear"

# --- Sentinel-2 --------------------------------------------------------------
BANDS = [
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B09",
    "B11",
    "B12",
]
BAND_RESOLUTION_M = {
    "B01": 60,
    "B02": 10,
    "B03": 10,
    "B04": 10,
    "B05": 20,
    "B06": 20,
    "B07": 20,
    "B08": 10,
    "B8A": 20,
    "B09": 60,
    "B11": 20,
    "B12": 20,
}
PERIODS = ["2022-01-01_2023-01-01", "2023-01-01_2024-01-01"]
PERIOD = PERIODS[0]

STAC_URL = "https://stac.earthgenome.org/"
STAC_COLLECTION = "sentinel2-temporal-mosaics"

# --- A5 ----------------------------------------------------------------------
# L18 (~495 m2) is the finest level that still tiles the AOI completely from pixel
# centres: L19 cells are smaller than a pixel, so 23% of them get no pixel at all.
A5_LEVEL = 18
A5_LEVEL_COARSE = 17

# --- Land cover schema -------------------------------------------------------
# ESA WorldCover v200 code -> analysis class id/name.
WORLDCOVER_TO_CLASS = {
    10: 1,  # tree cover
    20: 2,  # shrubland
    30: 3,  # grassland
    40: 4,  # cropland
    50: 5,  # built-up
    60: 6,  # bare / sparse vegetation
    70: 0,  # snow and ice -> unused in the UK
    80: 7,  # permanent water
    90: 8,  # herbaceous wetland
    95: 8,  # mangroves -> wetland
    100: 2,  # moss and lichen -> shrubland
}
CLASS_NAMES = {
    0: "nodata",
    1: "tree_cover",
    2: "shrubland",
    3: "grassland",
    4: "cropland",
    5: "built_up",
    6: "bare",
    7: "water",
    8: "wetland",
}
NODATA_CLASS = 0

# --- Run control -------------------------------------------------------------
SEED = 42
FORCE_REBUILD = False
BLOCK_SIZE = 1024
DUCKDB_MEMORY_LIMIT = "4GB"
A5_PREDICTION_BATCH_ROWS = 250_000
A5_COMPARE_BATCH_ROWS = 250_000
A5_CELL_BUCKETS = 64
RF_N_JOBS = 1

# --- Sampling and training ---------------------------------------------------
SAMPLES_PER_CLASS_PER_TILE = 5000
A5_MAX_TRAIN_ROWS = 50_000
# The England thesis comparison uses the separately persisted 400,000-row run.
# The public pipeline produces the retrained prediction; the 400k run is a
# separately persisted thesis comparison artefact and is not part of this package.
A5_PRIMARY_PREDICTION_NAME = "pred_a5_retrained"
A5_B08_MEAN_EPSILON = 1e-6
# Spatial block side in degrees (~3.5 km) used for a leakage-free train/test split.
SPLIT_BLOCK_DEG = 0.05
TEST_FRACTION = 0.25
# England has sufficient support for every analysis class, including shrubland.
EXCLUDE_CLASSES: set[int] = set()

# --- Paths -------------------------------------------------------------------
DATA = ROOT / "data"

RAW = DATA / "raw"
SENTINEL_DIR = RAW / "sentinel"
SENTINEL_DOWNLOADS = SENTINEL_DIR / "downloads"
SENTINEL_MANIFEST = SENTINEL_DIR / "sentinel_downloads.txt"
BOUNDARIES = RAW / "boundaries"
WORLDCOVER = RAW / "worldcover"
OSM_PBF = RAW / "uk.osm.pbf"
LSOA_GPKG = RAW / "lsoa_uk.gpkg"

INTERIM = DATA / "interim"
TILE_INVENTORY = INTERIM / "tile_inventory.parquet"


def scratch(aoi: str = AOI_NAME) -> Path:
    """Disk-backed scratch space; /tmp is tmpfs here and would be held in RAM."""
    path = INTERIM / aoi / "scratch"
    path.mkdir(parents=True, exist_ok=True)
    return path


DB_PATH = DATA / "dggs.duckdb"


def interim(aoi: str = AOI_NAME) -> Path:
    return INTERIM / aoi


def processed(aoi: str = AOI_NAME) -> Path:
    return DATA / "processed" / aoi


def aoi_geojson(aoi: str = AOI_NAME) -> Path:
    # GPKG, not GeoJSON: a dissolved England boundary exceeds OGR's GeoJSON object limit.
    return processed(aoi) / "aoi.gpkg"


def grid_json(aoi: str = AOI_NAME) -> Path:
    return processed(aoi) / "grid.json"


def mosaic_vrt(period: str = PERIOD, aoi: str = AOI_NAME) -> Path:
    return processed(aoi) / f"mosaic_{period}.vrt"


def tiles_stacked(aoi: str = AOI_NAME) -> Path:
    return interim(aoi) / "tiles_stacked"


def tiles_labels(aoi: str = AOI_NAME) -> Path:
    return interim(aoi) / "tiles_labels"


def tiles_pred(aoi: str = AOI_NAME) -> Path:
    return interim(aoi) / "tiles_pred"


def manifests(aoi: str = AOI_NAME) -> Path:
    return interim(aoi) / "manifests"


def a5_dataset(level: int = A5_LEVEL, aoi: str = AOI_NAME) -> Path:
    return processed(aoi) / f"a5_l{level}.parquet"


def models(aoi: str = AOI_NAME) -> Path:
    return processed(aoi) / "models"


def results(aoi: str = AOI_NAME) -> Path:
    return processed(aoi) / "results"


def band_path(tile: str, period: str, band: str) -> Path:
    return SENTINEL_DOWNLOADS / f"{tile}_{period}_{band}.tif"


def stack_path(tile: str, period: str = PERIOD, aoi: str = AOI_NAME) -> Path:
    return tiles_stacked(aoi) / f"{tile}_{period}_stack.tif"


def band_index(band: str) -> int:
    """1-based band index of `band` in a stacked tile."""
    return BANDS.index(band) + 1
