"""Storage and query benchmarks: raster vs A5 Parquet."""

from __future__ import annotations

import json
import time

import numpy as np
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.windows import from_bounds

from . import a5, config
from .io import checkpoint


def register_views(aoi: str = config.AOI_NAME) -> None:
    """DuckDB holds views over the Parquet, never a copy of it."""
    con = a5.connect(str(config.DB_PATH))
    for level in (config.A5_LEVEL, config.A5_LEVEL_COARSE):
        root = config.a5_dataset(level, aoi)
        if not any(root.rglob("*.parquet")):
            print(f"skip  a5_{aoi}_l{level} view (no parquet)")
            continue
        con.execute(f"""
            CREATE OR REPLACE VIEW a5_{aoi}_l{level} AS
            SELECT * FROM read_parquet('{root.as_posix()}/**/*.parquet', hive_partitioning=1)
        """)
    con.close()


def _time(fn, repeats: int = 3) -> tuple[float, object]:
    best, result = float("inf"), None
    for _ in range(repeats):
        started = time.perf_counter()
        result = fn()
        best = min(best, time.perf_counter() - started)
    return best, result


def run(aoi: str = config.AOI_NAME, *, force: bool = False) -> dict:
    """Four paired queries, each answered from the raster and from the A5 Parquet."""

    def build() -> dict:
        con = a5.connect()
        from .aoi import load_aoi

        aoi_geometry = load_aoi(aoi).geometry.iloc[0]
        representative_point = aoi_geometry.representative_point()
        level = config.A5_LEVEL
        root = config.a5_dataset(level, aoi).as_posix()
        src_a5 = f"read_parquet('{root}/**/*.parquet', hive_partitioning=1)"
        mosaic = config.mosaic_vrt(aoi=aoi)
        out: dict = {"level": level, "queries": {}}

        lon, lat = representative_point.x, representative_point.y
        cell = int(a5.lonlat_to_cell(np.array([lon]), np.array([lat]), level)[0])

        t_a5, _ = _time(
            lambda: con.execute(f"SELECT B04, B08 FROM {src_a5} WHERE cell_id = {cell}").fetchall()
        )
        t_r, _ = _time(lambda: list(rasterio.open(mosaic).sample([(lon, lat)], indexes=[4, 8])))
        out["queries"]["point_lookup"] = {"a5_s": t_a5, "raster_s": t_r}

        # Local neighbourhood: k-ring on the DGGS versus a bbox window on the raster.
        ring = a5.grid_disk(cell, 8)
        ids = ",".join(str(int(c)) for c in ring)
        t_a5, rows = _time(
            lambda: con.execute(
                f"SELECT AVG(B08) FROM {src_a5} WHERE cell_id IN ({ids})"
            ).fetchall()
        )
        half = 0.0025
        bounds = (lon - half, lat - half, lon + half, lat + half)

        def raster_window():
            with rasterio.open(mosaic) as src:
                w = from_bounds(*bounds, transform=src.transform)
                return src.read(config.band_index("B08"), window=w).mean()

        t_r, _ = _time(raster_window)
        out["queries"]["neighbourhood"] = {"a5_s": t_a5, "raster_s": t_r, "n_cells": len(ring)}

        # Zonal statistics over the configured AOI polygon.
        poly = aoi_geometry

        def zonal_a5():
            # a5_geometry_to_cells returns only the polygon's boundary cells, so
            # containment has to be tested against each cell's centre point.
            return con.execute(f"""
                SELECT COUNT(*) AS n, AVG(B08) AS mean_b08
                FROM {src_a5}
                WHERE ST_Within(a5_cell_to_point(cell_id),
                                ST_GeomFromText('{poly.wkt}'))
            """).fetchall()

        def zonal_raster():
            with rasterio.open(mosaic) as src:
                arr, _ = rio_mask(src, [poly], crop=True, indexes=[config.band_index("B08")])
            return float(arr[arr > 0].mean())

        t_a5, _ = _time(zonal_a5, repeats=2)
        t_r, _ = _time(zonal_raster, repeats=2)
        out["queries"]["zonal_aoi"] = {"a5_s": t_a5, "raster_s": t_r}

        # Full-dataset aggregate: mean NIR per reference class.
        t_a5, _ = _time(
            lambda: con.execute(f"SELECT label, AVG(B08) FROM {src_a5} GROUP BY label").fetchall(),
            repeats=2,
        )

        def raster_groupby():
            sums = np.zeros(9)
            counts = np.zeros(9)
            labels_vrt = config.processed(aoi) / f"labels_{config.PERIOD}.vrt"
            with rasterio.open(mosaic) as src, rasterio.open(labels_vrt) as lsrc:
                for _, window in lsrc.block_windows(1):
                    lab = lsrc.read(1, window=window).ravel()
                    nir = src.read(config.band_index("B08"), window=window).ravel()
                    keep = lab > 0
                    np.add.at(sums, lab[keep], nir[keep])
                    np.add.at(counts, lab[keep], 1)
            return sums / np.maximum(counts, 1)

        t_r, _ = _time(raster_groupby, repeats=1)
        out["queries"]["group_by_class"] = {"a5_s": t_a5, "raster_s": t_r}

        tiles = set(_aoi_tiles(aoi))
        out["storage_bytes"] = {
            "source_geotiffs": sum(
                p.stat().st_size
                for p in config.SENTINEL_DOWNLOADS.glob(f"*_{config.PERIOD}_*.tif")
                if p.name.split("_")[0] in tiles
            ),
            "stacked_cogs": sum(p.stat().st_size for p in config.tiles_stacked(aoi).glob("*.tif")),
            **{
                f"a5_l{lv}": sum(
                    p.stat().st_size for p in config.a5_dataset(lv, aoi).rglob("*.parquet")
                )
                for lv in (17, 18, 19)
            },
        }
        return out

    return checkpoint(config.results(aoi) / "benchmarks.json", build, force=force)


def _aoi_tiles(aoi: str) -> list[str]:
    from .aoi import load_aoi
    from .raster import tiles_for_aoi

    return tiles_for_aoi(load_aoi(aoi))


def report(bench: dict) -> str:
    lines = ["| query | raster (s) | A5 Parquet (s) | speedup |", "| --- | --- | --- | --- |"]
    for name, q in bench["queries"].items():
        lines.append(
            f"| {name} | {q['raster_s']:.4f} | {q['a5_s']:.4f} | {q['raster_s'] / q['a5_s']:.2f}x |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    register_views()
    print(json.dumps(run(), indent=2))
