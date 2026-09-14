"""ESA WorldCover reference labels, aligned to the stacked tile grid."""

from __future__ import annotations

import math
import subprocess
from pathlib import Path

import numpy as np
import rasterio
import requests
from rasterio.enums import Resampling
from rasterio.warp import reproject

from . import config
from .io import Manifest, atomic, checkpoint_file

WORLDCOVER_URL = (
    "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
    "ESA_WorldCover_10m_2021_v200_{name}_Map.tif"
)


def _tile_name(lat: int, lon: int) -> str:
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    return f"{ns}{abs(lat):02d}{ew}{abs(lon):03d}"


def tiles_for_bounds(bounds: tuple[float, float, float, float]) -> list[str]:
    """WorldCover tiles (3 deg, named by SW corner) covering `bounds`."""
    minx, miny, maxx, maxy = bounds
    lats = range(int(math.floor(miny / 3) * 3), int(math.floor(maxy / 3) * 3) + 1, 3)
    lons = range(int(math.floor(minx / 3) * 3), int(math.floor(maxx / 3) * 3) + 1, 3)
    return [_tile_name(la, lo) for la in lats for lo in lons]


def download(name: str) -> Path | None:
    """Fetch one WorldCover tile unless already present. Returns None for tiles that
    do not exist: WorldCover is only published where there is land."""
    out = config.WORLDCOVER / f"ESA_WorldCover_10m_2021_v200_{name}_Map.tif"
    if out.exists():
        try:
            with rasterio.open(out):
                return out
        except rasterio.RasterioIOError:
            out.unlink()

    url = WORLDCOVER_URL.format(name=name)
    with requests.get(url, stream=True, timeout=300) as resp:
        if resp.status_code == 404:
            print(f"skip  WorldCover {name} (no land)")
            return None
        resp.raise_for_status()
        with atomic(out) as tmp, tmp.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
    print(f"build {out.name} ({out.stat().st_size / 1e6:.0f} MB)")
    return out


def source_vrt(bounds: tuple[float, float, float, float], aoi: str = config.AOI_NAME) -> Path:
    """Single VRT over the WorldCover tiles covering `bounds`."""
    paths = [p for p in (download(n) for n in tiles_for_bounds(bounds)) if p is not None]
    if not paths:
        raise RuntimeError(f"No WorldCover tiles available for bounds {bounds}")
    out = config.interim(aoi) / "worldcover.vrt"

    def build(tmp: Path) -> None:
        subprocess.run(
            ["gdalbuildvrt", "-q", "-overwrite", str(tmp), *[str(p) for p in paths]], check=True
        )

    return checkpoint_file(out, build)


def _class_lut() -> np.ndarray:
    lut = np.full(256, config.NODATA_CLASS, dtype="uint8")
    for code, cls in config.WORLDCOVER_TO_CLASS.items():
        lut[code] = cls
    return lut


def _build_tile_labels(stack: Path, vrt: Path, out: Path) -> None:
    with rasterio.open(stack) as src:
        profile = src.profile
        shape = (src.height, src.width)
        valid = src.read(config.band_index("B04")) != src.nodata

    raw = np.zeros(shape, dtype="uint8")
    with rasterio.open(vrt) as wc:
        reproject(
            source=rasterio.band(wc, 1),
            destination=raw,
            src_transform=wc.transform,
            src_crs=wc.crs,
            dst_transform=profile["transform"],
            dst_crs=profile["crs"],
            resampling=Resampling.nearest,
        )

    labels = _class_lut()[raw]
    labels[~valid] = config.NODATA_CLASS

    profile.update(
        driver="COG",
        dtype="uint8",
        count=1,
        nodata=config.NODATA_CLASS,
        compress="ZSTD",
        blocksize=512,
    )
    for key in ("tiled", "blockxsize", "blockysize", "interleave", "photometric"):
        profile.pop(key, None)
    with rasterio.open(out, "w", **profile) as dst:
        dst.write(labels, 1)


def build_labels(
    tiles: list[str],
    period: str = config.PERIOD,
    aoi: str = config.AOI_NAME,
    *,
    force: bool = False,
) -> list[Path]:
    """One uint8 class raster per tile, pixel-aligned with the stack. Resumable per tile."""
    from .aoi import load_aoi

    vrt = source_vrt(tuple(load_aoi(aoi).total_bounds), aoi)
    manifest = Manifest(f"labels_{period}", aoi)
    paths = []
    for tile in tiles:
        stack = config.stack_path(tile, period, aoi)
        out = config.tiles_labels(aoi) / f"{tile}_{period}_labels.tif"
        checkpoint_file(out, lambda tmp, s=stack: _build_tile_labels(s, vrt, tmp), force=force)
        if not manifest.done(tile):
            manifest.mark(tile, bytes=out.stat().st_size)
        paths.append(out)
    return paths


def build_mosaic(
    tiles: list[str],
    period: str = config.PERIOD,
    aoi: str = config.AOI_NAME,
    *,
    force: bool = False,
) -> Path:
    """Virtual mosaic over the per-tile label rasters."""
    out = config.processed(aoi) / f"labels_{period}.vrt"
    parts = [str(config.tiles_labels(aoi) / f"{t}_{period}_labels.tif") for t in tiles]

    def build(tmp: Path) -> None:
        subprocess.run(["gdalbuildvrt", "-q", "-overwrite", str(tmp), *parts], check=True)

    return checkpoint_file(out, build, force=force)


def class_counts(paths: list[Path]) -> dict[str, int]:
    counts = np.zeros(256, dtype="int64")
    for p in paths:
        with rasterio.open(p) as src:
            for _, window in src.block_windows(1):
                counts += np.bincount(src.read(1, window=window).ravel(), minlength=256)
    return {
        config.CLASS_NAMES[i]: int(counts[i]) for i in sorted(config.CLASS_NAMES) if counts[i] > 0
    }
