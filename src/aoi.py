"""AOI boundaries from a checked-in GeoJSON or local geospatial sources."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import geopandas as gpd

from . import config
from .io import checkpoint


def _extract_relation(relation_id: int, out_path: Path) -> None:
    """Cut one OSM relation out of the local .pbf and write it as a polygon layer."""
    with tempfile.TemporaryDirectory() as tmpdir:
        pbf = Path(tmpdir) / "relation.osm.pbf"
        subprocess.run(
            ["osmium", "getid", "-r", "-t", str(config.OSM_PBF), f"r{relation_id}", "-o", str(pbf)],
            check=True,
        )
        subprocess.run(
            [
                "ogr2ogr",
                "-f",
                "GPKG",
                "-t_srs",
                config.CRS,
                str(out_path),
                str(pbf),
                "multipolygons",
            ],
            check=True,
        )


def _extract_lsoa(prefix: str, out_path: Path) -> None:
    """Dissolve the coastline-clipped ONS LSOA polygons into one land boundary."""
    layer = gpd.list_layers(config.LSOA_GPKG)["name"].iloc[0]
    where = f"LSOA21CD LIKE '{prefix}%'" if prefix else None
    gdf = gpd.read_file(config.LSOA_GPKG, layer=layer, where=where, columns=["LSOA21CD"])
    merged = gdf.to_crs(config.CRS).geometry.union_all().simplify(config.AOI_SIMPLIFY_DEG)
    gpd.GeoDataFrame(geometry=[merged], crs=config.CRS).to_file(
        out_path, driver="GPKG", layer="aoi"
    )


def load_aoi(aoi: str = config.AOI_NAME, *, force: bool = False) -> gpd.GeoDataFrame:
    """Return the dissolved AOI polygon in EPSG:4326, built from the configured source."""
    kind, key = config.AOI_SOURCE[aoi]
    raw = Path(key) if kind == "geojson" else config.BOUNDARIES / f"{aoi}_{kind}.gpkg"

    def build() -> gpd.GeoDataFrame:
        if not raw.exists():
            if kind == "geojson":
                raise FileNotFoundError(
                    f"AOI GeoJSON is missing: {raw}. "
                    "Add the small AOI boundary file before running the pipeline."
                )
            raw.parent.mkdir(parents=True, exist_ok=True)
            if kind == "osm":
                _extract_relation(int(key), raw)
            else:
                _extract_lsoa(str(key), raw)
        gdf = gpd.read_file(raw).to_crs(config.CRS)
        merged = gdf.geometry.union_all().simplify(config.AOI_SIMPLIFY_DEG)
        return gpd.GeoDataFrame({"aoi": [aoi]}, geometry=[merged], crs=config.CRS)

    return checkpoint(config.aoi_geojson(aoi), build, force=force)
