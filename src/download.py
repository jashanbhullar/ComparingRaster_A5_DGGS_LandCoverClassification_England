"""Discover and download the Sentinel-2 assets required by the pipeline."""

from __future__ import annotations

import csv
from pathlib import Path

import rasterio
import requests
from pystac_client import Client

from . import config
from .aoi import load_aoi
from .io import atomic


def _manifest_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def discover_manifest(aoi: str = config.AOI_NAME, *, force: bool = False) -> Path:
    """Discover Sentinel-2 assets intersecting `aoi` and persist their URLs."""
    path = config.SENTINEL_MANIFEST
    if path.exists() and not (force or config.FORCE_REBUILD):
        print(f"skip  {path.name} (exists)")
        return path

    bounds = tuple(float(value) for value in load_aoi(aoi).total_bounds)
    catalog = Client.open(config.STAC_URL)
    search = catalog.search(
        collections=[config.STAC_COLLECTION],
        bbox=bounds,
        max_items=None,
        limit=1000,
        sortby=["+id"],
    )

    rows: list[dict[str, str]] = []
    for item in search.items():
        for band in config.BANDS:
            asset = item.assets.get(band)
            if asset is not None and asset.href:
                rows.append({"item_id": item.id, "asset_key": band, "asset_href": asset.href})
    if not rows:
        raise RuntimeError("STAC search returned no configured Sentinel-2 band assets")

    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic(path) as tmp:
        with tmp.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh, fieldnames=["item_id", "asset_key", "asset_href"], delimiter="\t"
            )
            writer.writeheader()
            writer.writerows(sorted(rows, key=lambda row: (row["item_id"], row["asset_key"])))
    print(f"build {path.name} ({len(rows)} asset URLs)")
    return path


def download_assets(
    period: str = config.PERIOD,
    *,
    aoi: str = config.AOI_NAME,
    force: bool = False,
) -> list[Path]:
    """Download the configured bands for `period`, resuming from local files."""
    manifest = discover_manifest(aoi, force=force)
    rows = [
        row
        for row in _manifest_rows(manifest)
        if row["asset_key"] in config.BANDS and row["item_id"].endswith(f"_{period}")
    ]
    if not rows:
        raise RuntimeError(f"Manifest contains no Sentinel-2 assets for period {period}")

    downloaded: list[Path] = []
    session = requests.Session()
    for row in rows:
        item_id = row["item_id"]
        tile = item_id.split("_", 1)[0]
        target = config.band_path(tile, period, row["asset_key"])
        if target.exists() and not force:
            try:
                with rasterio.open(target):
                    downloaded.append(target)
                    continue
            except rasterio.RasterioIOError:
                target.unlink()

        with session.get(row["asset_href"], stream=True, timeout=300) as response:
            response.raise_for_status()
            with atomic(target) as tmp:
                with tmp.open("wb") as fh:
                    for chunk in response.iter_content(chunk_size=1 << 20):
                        if chunk:
                            fh.write(chunk)
        downloaded.append(target)
        print(f"build {target.name}")
    print(f"sentinel: {len(downloaded)} asset(s) ready for {period}")
    return downloaded


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aoi", default=config.AOI_NAME, choices=config.AOI_SOURCE)
    parser.add_argument("--period", default=config.PERIOD, choices=config.PERIODS)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    download_assets(args.period, aoi=args.aoi, force=args.force)
