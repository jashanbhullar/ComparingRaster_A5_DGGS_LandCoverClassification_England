"""Sentinel-2 tile inventory, footprints, AOI selection and the EPSG:4326 raster build."""

from __future__ import annotations

import math
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import geopandas as gpd
import pandas as pd
import rasterio
import requests
from pyproj import Geod
from rasterio.warp import transform_bounds
from shapely.geometry import box

from . import config
from .io import Manifest, atomic, checkpoint, checkpoint_file


def _parse_manifest() -> pd.DataFrame:
    df = pd.read_csv(config.SENTINEL_MANIFEST, sep="\t")
    ids = df["item_id"].str.extract(r"^(?P<tile>[^_]+)_(?P<period>.+)$")
    df = pd.concat([df, ids], axis=1).rename(columns={"asset_key": "band"})
    df["path"] = [
        str(config.band_path(t, p, b))
        for t, p, b in zip(df["tile"], df["period"], df["band"], strict=True)
    ]
    return df[["tile", "period", "band", "path", "asset_href"]]


def load_inventory(*, force: bool = False) -> pd.DataFrame:
    """One row per (tile, band, period) with local presence and the server's size.

    The remote size is fetched once and cached: opening a truncated GeoTIFF succeeds,
    so a size comparison is the only cheap way to detect a partial download.
    """

    def build() -> pd.DataFrame:
        df = _parse_manifest()
        stats = [Path(p) for p in df["path"]]
        df["exists"] = [p.exists() for p in stats]
        df["size_bytes"] = [p.stat().st_size if p.exists() else 0 for p in stats]
        df["remote_bytes"] = _remote_sizes(df["asset_href"].tolist())
        df["intact"] = df["exists"] & (df["size_bytes"] == df["remote_bytes"])
        return df

    return checkpoint(config.TILE_INVENTORY, build, force=force)


def _remote_sizes(hrefs: list[str]) -> list[int]:
    session = requests.Session()

    def head(href: str) -> int:
        try:
            resp = session.head(href, timeout=30, allow_redirects=True)
            return int(resp.headers.get("Content-Length", -1))
        except requests.RequestException:
            return -1

    with ThreadPoolExecutor(16) as pool:
        return list(pool.map(head, hrefs))


def repair_tiles(
    tiles: list[str], period: str = config.PERIOD, *, dry_run: bool = False
) -> list[str]:
    """Re-download only the assets whose local size disagrees with the server."""
    inv = load_inventory()
    broken = inv[
        inv["tile"].isin(tiles)
        & (inv["period"] == period)
        & inv["band"].isin(config.BANDS)
        & ~inv["intact"]
    ]
    if broken.empty or dry_run:
        print(f"repair: {len(broken)} asset(s) need re-download")
        return broken["path"].tolist()

    session = requests.Session()
    for path, href, expected in zip(
        broken["path"], broken["asset_href"], broken["remote_bytes"], strict=True
    ):
        target = Path(path)
        with atomic(target) as tmp, session.get(href, stream=True, timeout=300) as resp:
            resp.raise_for_status()
            with tmp.open("wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
            if tmp.stat().st_size != expected:
                raise OSError(f"{target.name}: got {tmp.stat().st_size}, expected {expected}")
        print(f"   repaired {target.name} ({expected / 1e6:.0f} MB)")
    load_inventory(force=True)
    return broken["path"].tolist()


def load_footprints(*, force: bool = False) -> gpd.GeoDataFrame:
    """Tile footprint in EPSG:4326, read once per (tile, period) from a 10 m band."""
    path = config.INTERIM / "tile_footprints.geojson"

    def build() -> gpd.GeoDataFrame:
        inv = load_inventory()
        ref = inv[(inv["band"] == "B02") & inv["exists"]]
        rows = []
        for tile, period, tif in zip(ref["tile"], ref["period"], ref["path"], strict=True):
            with rasterio.open(tif) as src:
                bounds = transform_bounds(src.crs, config.CRS, *src.bounds, densify_pts=21)
            rows.append({"tile": tile, "period": period, "geometry": box(*bounds)})
        return gpd.GeoDataFrame(rows, crs=config.CRS)

    return checkpoint(path, build, force=force)


def tiles_for_aoi(aoi_gdf: gpd.GeoDataFrame, period: str = config.PERIOD) -> list[str]:
    fp = load_footprints()
    fp = fp[fp["period"] == period]
    hit = fp[fp.intersects(aoi_gdf.geometry.iloc[0])]
    return sorted(hit["tile"].unique().tolist())


def verify_tiles(tiles: list[str], period: str = config.PERIOD) -> pd.DataFrame:
    """Check every band of the selected tiles. Size is checked as well as openability:
    a truncated GeoTIFF still opens and only fails deep inside the file."""
    inv = load_inventory()
    sel = inv[
        inv["tile"].isin(tiles) & (inv["period"] == period) & inv["band"].isin(config.BANDS)
    ].copy()
    ok, note = [], []
    for path, exists, intact in zip(sel["path"], sel["exists"], sel["intact"], strict=True):
        if not exists:
            ok.append(False)
            note.append("missing")
            continue
        if not intact:
            ok.append(False)
            note.append("truncated")
            continue
        try:
            with rasterio.open(path) as src:
                src.read(1, window=rasterio.windows.Window(0, 0, 1, 1))
            ok.append(True)
            note.append("")
        except Exception as exc:  # unreadable file must be re-downloaded
            ok.append(False)
            note.append(str(exc)[:120])
    sel["ok"] = ok
    sel["note"] = note
    return sel


# --- EPSG:4326 grid (Option B: square ground pixels at the AOI centroid) ------

_GEOD = Geod(ellps="WGS84")


def _degree_lengths_m(lat: float) -> tuple[float, float]:
    """Ground length of one degree of longitude and of latitude at `lat`, in metres."""
    lon_m = _GEOD.inv(0.0, lat, 1.0, lat)[2]
    lat_m = _GEOD.inv(0.0, lat - 0.5, 0.0, lat + 0.5)[2]
    return lon_m, lat_m


def grid_spec(aoi: str = config.AOI_NAME, *, force: bool = False) -> dict:
    """Resolve and cache the EPSG:4326 pixel size for this AOI."""

    def build() -> dict:
        from .aoi import load_aoi

        poly = load_aoi(aoi).geometry.iloc[0]
        lat = float(poly.centroid.y)

        tiles = tiles_for_aoi(load_aoi(aoi))
        with rasterio.open(config.band_path(tiles[0], config.PERIOD, "B02")) as src:
            src_crs = src.crs.to_string()
            src_res = float(src.res[0])
        if src_crs != "EPSG:3857":
            raise ValueError(f"Expected Web Mercator sources, got {src_crs}")
        # Web Mercator is conformal: a pixel of `src_res` map units spans
        # src_res * cos(lat) metres on the ground, in both directions.
        native_m = src_res * math.cos(math.radians(lat))

        ground_m = config.TARGET_GROUND_RES_M or native_m
        lon_m, lat_m = _degree_lengths_m(lat)
        return {
            "aoi": aoi,
            "crs": config.CRS,
            "centroid_lat": lat,
            "source_crs": src_crs,
            "source_res_m": src_res,
            "native_ground_res_m": native_m,
            "ground_res_m": ground_m,
            "x_deg": ground_m / lon_m,
            "y_deg": ground_m / lat_m,
            "resampling": config.RESAMPLING,
        }

    return checkpoint(config.grid_json(aoi), build, force=force)


def _stack_tile(tile: str, period: str, grid: dict, aoi: str, out: Path) -> None:
    """Band order in the output is exactly config.BANDS; COGs cannot carry descriptions."""
    from .aoi import load_aoi

    fp = load_footprints()
    fp = fp[(fp["tile"] == tile) & (fp["period"] == period)].geometry.iloc[0]
    extent = fp.intersection(load_aoi(aoi).geometry.iloc[0]).bounds

    band_vrt = out.with_suffix(".bands.vrt")
    subprocess.run(
        [
            "gdalbuildvrt",
            "-q",
            "-separate",
            "-resolution",
            "highest",
            "-r",
            grid["resampling"],
            str(band_vrt),
            *[str(config.band_path(tile, period, b)) for b in config.BANDS],
        ],
        check=True,
    )
    try:
        subprocess.run(
            [
                "gdalwarp",
                "-q",
                "-multi",
                "-wo",
                "NUM_THREADS=ALL_CPUS",
                "-t_srs",
                config.CRS,
                "-tr",
                repr(grid["x_deg"]),
                repr(grid["y_deg"]),
                "-tap",
                "-te",
                *[repr(v) for v in extent],
                "-r",
                grid["resampling"],
                "-cutline",
                str(config.aoi_geojson(aoi)),
                "-srcnodata",
                "0",
                "-dstnodata",
                "0",
                "-ot",
                "UInt16",
                "-of",
                "COG",
                "-co",
                "COMPRESS=ZSTD",
                "-co",
                "BLOCKSIZE=512",
                "-co",
                "NUM_THREADS=ALL_CPUS",
                "-co",
                "BIGTIFF=YES",
                str(band_vrt),
                str(out),
            ],
            check=True,
        )
    finally:
        band_vrt.unlink(missing_ok=True)


def build_stacks(
    tiles: list[str],
    period: str = config.PERIOD,
    aoi: str = config.AOI_NAME,
    *,
    force: bool = False,
) -> list[Path]:
    """One 12-band EPSG:4326 COG per tile, clipped to the AOI. Resumable per tile."""
    grid = grid_spec(aoi)
    manifest = Manifest(f"stacks_{period}", aoi)
    paths = []
    for tile in tiles:
        out = config.stack_path(tile, period, aoi)
        checkpoint_file(
            out, lambda tmp, t=tile: _stack_tile(t, period, grid, aoi, tmp), force=force
        )
        if not manifest.done(tile):
            with rasterio.open(out) as src:
                manifest.mark(tile, shape=src.shape, bytes=out.stat().st_size)
        paths.append(out)
    return paths


def build_mosaic(
    tiles: list[str],
    period: str = config.PERIOD,
    aoi: str = config.AOI_NAME,
    *,
    force: bool = False,
) -> Path:
    """Virtual mosaic over the per-tile stacks; no pixels are duplicated on disk."""
    out = config.mosaic_vrt(period, aoi)
    stacks = [str(config.stack_path(t, period, aoi)) for t in tiles]

    def build(tmp: Path) -> None:
        subprocess.run(["gdalbuildvrt", "-q", "-overwrite", str(tmp), *stacks], check=True)

    return checkpoint_file(out, build, force=force)
