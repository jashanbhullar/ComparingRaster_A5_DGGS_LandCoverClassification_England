"""Raster to A5 DGGS: pixel centres are indexed to cells and aggregated to one row
per cell, with each band kept as its own column."""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import rasterio
from rasterio.windows import Window

from . import a5, config
from .io import Manifest, atomic, checkpoint, checkpoint_file


def _source_signature(paths: list[Path]) -> list[dict]:
    return [
        {"path": path.as_posix(), "size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
        for path in paths
    ]


def _dependencies_changed(output: Path, dependencies: list[Path]) -> bool:
    signature_path = output.with_suffix(f"{output.suffix}.sources.json")
    if not output.exists() or not signature_path.exists():
        return True
    return json.loads(signature_path.read_text()) != _source_signature(dependencies)


def _record_dependencies(output: Path, dependencies: list[Path]) -> None:
    signature_path = output.with_suffix(f"{output.suffix}.sources.json")
    with atomic(signature_path) as tmp:
        tmp.write_text(json.dumps(_source_signature(dependencies), indent=2))


def _blocks(height: int, width: int, size: int) -> list[Window]:
    return [
        Window(c, r, min(size, width - c), min(size, height - r))
        for r in range(0, height, size)
        for c in range(0, width, size)
    ]


def _pixel_lonlat(transform, window: Window) -> tuple[np.ndarray, np.ndarray]:
    rows = np.arange(window.row_off, window.row_off + window.height) + 0.5
    cols = np.arange(window.col_off, window.col_off + window.width) + 0.5
    lon = transform.c + cols * transform.a
    lat = transform.f + rows * transform.e
    return np.repeat(lat, cols.size), np.tile(lon, rows.size)


def _pixel_table(
    stack: Path,
    label_path: Path,
    level: int,
    out: Path,
    aoi: str,
    bands: list[str] | None = None,
) -> int:
    """Stream every valid pixel of one tile to `out` as (cell_id, bands..., label)."""
    bands = bands or config.BANDS
    keep = [config.BANDS.index(b) for b in bands]
    schema = pa.schema(
        [("cell_id", pa.uint64())] + [(b, pa.uint16()) for b in bands] + [("label", pa.uint8())]
    )
    written = 0
    with (
        rasterio.open(stack) as src,
        rasterio.open(label_path) as lsrc,
        pq.ParquetWriter(out, schema, compression="zstd") as writer,
    ):
        for window in _blocks(src.height, src.width, config.BLOCK_SIZE):
            values = src.read(window=window).reshape(len(config.BANDS), -1).T
            valid = values.sum(axis=1) > 0
            if not valid.any():
                continue
            labels = lsrc.read(1, window=window).ravel()
            lat, lon = _pixel_lonlat(src.transform, window)
            cell_id = a5.lonlat_to_cell(lon[valid], lat[valid], level, aoi=aoi)

            arrays = [pa.array(cell_id.astype("uint64"))]
            arrays += [pa.array(values[valid][:, i]) for i in keep]
            arrays.append(pa.array(labels[valid]))
            writer.write_table(pa.Table.from_arrays(arrays, schema=schema))
            written += int(valid.sum())
    return written


def _aggregate(pixels: Path, out: Path, tile: str, period: str, aoi: str) -> None:
    """One row per cell: pixel count, per-band mean, majority label. Sorted by cell_id
    so Parquet min/max statistics can prune range scans."""
    means = ", ".join(f"AVG({b})::FLOAT AS {b}" for b in config.BANDS)
    con = a5.connect(aoi=aoi)
    con.execute(f"""
        COPY (
            SELECT
                cell_id,
                COUNT(*)::USMALLINT AS n_pixels,
                {means},
                mode(label) AS label,
                '{tile}' AS tile,
                '{period}' AS period
            FROM read_parquet('{pixels.as_posix()}')
            GROUP BY cell_id
            ORDER BY cell_id
        ) TO '{out.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 1000000)
    """)


def build_dataset(
    tiles: list[str],
    level: int = config.A5_LEVEL,
    period: str = config.PERIOD,
    aoi: str = config.AOI_NAME,
    *,
    force: bool = False,
) -> Path:
    """Hive-partitioned A5 Parquet: period=<P>/tile=<T>/part-0.parquet. Resumable per tile."""
    root = config.a5_dataset(level, aoi)
    manifest = Manifest(f"a5_l{level}_{period}", aoi)
    for tile in tiles:
        out = root / f"period={period}" / f"tile={tile}" / "part-0.parquet"
        label_path = config.tiles_labels(aoi) / f"{tile}_{period}_labels.tif"
        stack = config.stack_path(tile, period, aoi)

        def build(tmp: Path, s=stack, lp=label_path, t=tile) -> None:
            with tempfile.TemporaryDirectory(dir=config.scratch(aoi)) as tmpdir:
                pixels = Path(tmpdir) / "pixels.parquet"
                started = time.perf_counter()
                n = _pixel_table(s, lp, level, pixels, aoi)
                _aggregate(pixels, tmp, t, period, aoi)
                print(f"      {t}: {n:,} px indexed in {time.perf_counter() - started:.1f}s")

        checkpoint_file(out, build, force=force)
        if not manifest.done(tile):
            manifest.mark(tile, bytes=out.stat().st_size)
    return root


def read(level: int = config.A5_LEVEL, aoi: str = config.AOI_NAME):
    """A DuckDB relation over the partitioned dataset; no data is copied."""
    root = config.a5_dataset(level, aoi)
    return a5.connect(aoi=aoi).sql(
        f"SELECT * FROM read_parquet('{root.as_posix()}/**/*.parquet', hive_partitioning=1)"
    )


def cells_sql(
    level: int = config.A5_LEVEL,
    aoi: str = config.AOI_NAME,
    *,
    where: str | None = None,
) -> str:
    """Deduplicated cells: tile overlaps are merged with a pixel-count weighted mean."""
    root = config.a5_dataset(level, aoi).as_posix()
    means = ", ".join(f"(SUM({b} * n_pixels) / SUM(n_pixels))::FLOAT AS {b}" for b in config.BANDS)
    source = f"read_parquet('{root}/**/*.parquet', hive_partitioning=1)"
    if where:
        source = f"(SELECT * FROM {source} WHERE {where})"
    return f"""
        SELECT
            cell_id,
            SUM(n_pixels)::INTEGER AS n_pixels,
            {means},
            mode(label) AS label,
            arg_max(tile, n_pixels) AS tile
        FROM {source}
        GROUP BY cell_id
    """


def _cell_feature_parts(level: int, aoi: str, force: bool) -> list[Path]:
    directory = config.interim(aoi) / f"parts_cells_l{level}"
    parts = []
    bands = ", ".join(config.BANDS)
    for bucket in range(config.A5_CELL_BUCKETS):
        part = directory / f"bucket={bucket:03d}.parquet"

        def build(tmp: Path, b=bucket) -> None:
            where = f"hash(cell_id) % {config.A5_CELL_BUCKETS} = {b}"
            a5.connect(aoi=aoi).execute(f"""
                COPY (
                    SELECT cell_id, {bands}, label
                    FROM ({cells_sql(level, aoi, where=where)})
                    ORDER BY cell_id
                ) TO '{tmp.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
            """)

        checkpoint_file(part, build, force=force)
        parts.append(part)
    return parts


def sample_cells(
    level: int = config.A5_LEVEL,
    aoi: str = config.AOI_NAME,
    *,
    force: bool = False,
) -> pd.DataFrame:
    """Stratified per-class sample per tile, mirroring the raster sampling design."""
    from .classify import block_id, is_test

    path = config.interim(aoi) / f"train_samples_a5_l{level}.parquet"

    def build() -> pd.DataFrame:
        excluded = ", ".join(str(c) for c in {config.NODATA_CLASS, *config.EXCLUDE_CLASSES})
        per_class = config.SAMPLES_PER_CLASS_PER_TILE
        root = config.a5_dataset(level, aoi) / f"period={config.PERIOD}"
        con = a5.connect(aoi=aoi)
        frames = []
        for part in sorted(root.glob("tile=*/part-0.parquet")):
            df_tile = con.execute(
                f"""
                SELECT sampled.* EXCLUDE (rn),
                       ST_X(a5_cell_to_point(cell_id)) AS lon,
                       ST_Y(a5_cell_to_point(cell_id)) AS lat
                FROM (
                    SELECT *, row_number() OVER (
                        PARTITION BY label ORDER BY hash(cell_id)
                    ) AS rn
                    FROM read_parquet('{part.as_posix()}')
                    WHERE label NOT IN ({excluded})
                ) sampled
                WHERE rn <= {per_class}
            """
            ).fetchdf()
            frames.append(df_tile)
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        df["block"] = block_id(df["lon"].to_numpy(), df["lat"].to_numpy())
        df["test"] = is_test(df["block"].to_numpy())
        df["level"] = level
        return df

    return checkpoint(path, build, force=force)


def predict_cells(
    bundle: dict,
    level: int = config.A5_LEVEL,
    aoi: str = config.AOI_NAME,
    *,
    name: str = "a5",
    force: bool = False,
) -> Path:
    """Class prediction for every cell in the dataset, written as sorted Parquet."""
    from . import features

    path = config.processed(aoi) / f"pred_{name}_l{level}.parquet"

    def build(tmp: Path) -> None:
        schema = pa.schema([("cell_id", pa.uint64()), ("label", pa.uint8()), ("pred", pa.uint8())])
        with pq.ParquetWriter(tmp, schema, compression="zstd") as writer:
            for part in _cell_feature_parts(level, aoi, force):
                for batch in pq.ParquetFile(part).iter_batches(
                    batch_size=config.A5_PREDICTION_BATCH_ROWS
                ):
                    df = batch.to_pandas()
                    pred = bundle["model"].predict(features.compute(df[config.BANDS].to_numpy()))
                    writer.write_table(
                        pa.Table.from_arrays(
                            [
                                pa.array(df["cell_id"].to_numpy(), type=pa.uint64()),
                                pa.array(df["label"].to_numpy(), type=pa.uint8()),
                                pa.array(pred.astype("uint8"), type=pa.uint8()),
                            ],
                            schema=schema,
                        )
                    )

    return checkpoint_file(path, build, force=force)


def _index_single_band(raster: Path, level: int, out: Path, aoi: str) -> None:
    """Stream one band of a raster to (cell_id, value) pairs at `level`."""
    schema = pa.schema([("cell_id", pa.uint64()), ("value", pa.uint8())])
    with (
        rasterio.open(raster) as src,
        pq.ParquetWriter(out, schema, compression="zstd") as writer,
    ):
        for window in _blocks(src.height, src.width, config.BLOCK_SIZE):
            values = src.read(1, window=window).ravel()
            valid = values != src.nodata
            if not valid.any():
                continue
            lat, lon = _pixel_lonlat(src.transform, window)
            cell_id = a5.lonlat_to_cell(lon[valid], lat[valid], level, aoi=aoi)
            writer.write_table(
                pa.Table.from_arrays(
                    [pa.array(cell_id.astype("uint64")), pa.array(values[valid])], schema=schema
                )
            )


def _tile_parts(
    name: str,
    tiles: list[str],
    level: int,
    aoi: str,
    make_part,
    force: bool,
    dependencies_for_tile=None,
) -> str:
    """Build one intermediate per tile so a long AOI run resumes at a tile boundary."""
    directory = config.interim(aoi) / f"parts_{name}_l{level}"
    parts = []
    for tile in tiles:
        part = directory / f"{tile}.parquet"
        dependencies = dependencies_for_tile(tile) if dependencies_for_tile else []
        stale = bool(dependencies) and _dependencies_changed(part, dependencies)
        checkpoint_file(part, lambda tmp, t=tile: make_part(t, tmp), force=force or stale)
        if dependencies and (force or stale):
            _record_dependencies(part, dependencies)
        parts.append(part.as_posix())
    return ", ".join(f"'{p}'" for p in parts)


def _concat_parquet(parts: list[Path], out: Path) -> None:
    writer = None
    try:
        for part in parts:
            parquet = pq.ParquetFile(part)
            if writer is None:
                writer = pq.ParquetWriter(out, parquet.schema_arrow, compression="zstd")
            for batch in parquet.iter_batches(batch_size=config.A5_PREDICTION_BATCH_ROWS):
                writer.write_table(pa.Table.from_batches([batch]))
    finally:
        if writer is not None:
            writer.close()


def aggregate_raster_to_cells(
    tiles: list[str],
    name: str,
    raster_for_tile,
    level: int = config.A5_LEVEL,
    aoi: str = config.AOI_NAME,
    *,
    force: bool = False,
) -> Path:
    """Majority value per A5 cell for a single-band raster — used to bring the raster
    predictions onto the cell grid so both arms can be scored over identical ground."""
    path = config.processed(aoi) / f"{name}_l{level}.parquet"
    source_paths = [raster_for_tile(tile) for tile in tiles]
    source_stale = _dependencies_changed(path, source_paths)
    tile_parts = [
        config.interim(aoi) / f"parts_{name}_l{level}" / f"{tile}.parquet" for tile in tiles
    ]
    bucket_dir = config.interim(aoi) / f"parts_{name}_agg_l{level}"
    bucket_parts = [
        bucket_dir / f"bucket={bucket:03d}.parquet" for bucket in range(config.A5_CELL_BUCKETS)
    ]
    aggregation_stale = not all(part.exists() for part in bucket_parts) or (
        all(part.exists() for part in tile_parts)
        and max(part.stat().st_mtime_ns for part in tile_parts)
        > min(part.stat().st_mtime_ns for part in bucket_parts)
    )
    stale = source_stale or aggregation_stale

    def build(tmp: Path) -> None:
        sources = _tile_parts(
            name,
            tiles,
            level,
            aoi,
            lambda t, out: _index_single_band(raster_for_tile(t), level, out, aoi),
            force or source_stale,
            lambda t: [raster_for_tile(t)],
        )
        built_bucket_parts = []
        for bucket in range(config.A5_CELL_BUCKETS):
            part = bucket_dir / f"bucket={bucket:03d}.parquet"

            def build_bucket(bucket_tmp: Path, b=bucket) -> None:
                a5.connect(aoi=aoi).execute(f"""
                    COPY (
                        SELECT cell_id, mode(value) AS value, COUNT(*)::INTEGER AS n_pixels
                        FROM read_parquet([{sources}])
                        WHERE hash(cell_id) % {config.A5_CELL_BUCKETS} = {b}
                        GROUP BY cell_id ORDER BY cell_id
                    ) TO '{bucket_tmp.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
                """)

            checkpoint_file(part, build_bucket, force=force or stale)
            built_bucket_parts.append(part)
        _concat_parquet(built_bucket_parts, tmp)

    result = checkpoint_file(path, build, force=force or stale)
    if force or stale:
        _record_dependencies(path, source_paths)
    return result


def build_cell_stats(
    tiles: list[str],
    level: int = config.A5_LEVEL,
    period: str = config.PERIOD,
    aoi: str = config.AOI_NAME,
    *,
    force: bool = False,
) -> Path:
    """Per-cell information loss: label purity and within-cell reflectance spread."""
    path = config.interim(aoi) / f"cell_stats_l{level}.parquet"

    def make_part(tile: str, out: Path) -> None:
        _pixel_table(
            config.stack_path(tile, period, aoi),
            config.tiles_labels(aoi) / f"{tile}_{period}_labels.tif",
            level,
            out,
            aoi,
            bands=["B08"],
        )

    def build(tmp: Path) -> None:
        sources = _tile_parts(
            "stats",
            tiles,
            level,
            aoi,
            make_part,
            force,
            lambda tile: [
                config.stack_path(tile, period, aoi),
                config.tiles_labels(aoi) / f"{tile}_{period}_labels.tif",
            ],
        )
        bucket_dir = config.interim(aoi) / f"parts_cell_stats_agg_l{level}"
        bucket_parts = []
        for bucket in range(config.A5_CELL_BUCKETS):
            part = bucket_dir / f"bucket={bucket:03d}.parquet"

            def build_bucket(bucket_tmp: Path, b=bucket) -> None:
                a5.connect(aoi=aoi).execute(f"""
                    COPY (
                        WITH px AS (
                            SELECT cell_id, label, B08 FROM read_parquet([{sources}])
                            WHERE hash(cell_id) % {config.A5_CELL_BUCKETS} = {b}
                        ),
                        by_label AS (SELECT cell_id, label, COUNT(*) AS c FROM px GROUP BY 1, 2),
                        purity AS (
                            SELECT cell_id,
                                   (MAX(c)::DOUBLE / SUM(c)) AS label_purity,
                                   SUM(c)::INTEGER AS n_pixels
                            FROM by_label GROUP BY 1
                        ),
                        spread AS (
                            SELECT cell_id, stddev_pop(B08) AS b08_std, AVG(B08) AS b08_mean
                            FROM px GROUP BY 1
                        )
                        SELECT p.cell_id, p.n_pixels, p.label_purity, s.b08_std, s.b08_mean
                        FROM purity p JOIN spread s USING (cell_id)
                        ORDER BY p.cell_id
                    ) TO '{bucket_tmp.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
                """)

            checkpoint_file(part, build_bucket, force=force)
            bucket_parts.append(part)
        _concat_parquet(bucket_parts, tmp)

    return checkpoint_file(path, build, force=force)
