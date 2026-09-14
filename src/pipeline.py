"""End-to-end driver. The AOI is the only thing that changes between runs."""

from __future__ import annotations

import argparse
import gc
import time

from . import aoi as aoi_mod
from . import bench, classify, config, dggs, download, labels, raster, report
from .io import checkpoint


def run(aoi: str = config.AOI_NAME, levels: tuple[int, ...] = (config.A5_LEVEL,)) -> dict:
    started = time.perf_counter()

    def stage(name: str) -> float:
        print(f"\n== {name} [{time.perf_counter() - started:.0f}s]", flush=True)
        return time.perf_counter()

    stage("AOI and tiles")
    boundary = aoi_mod.load_aoi(aoi)
    download.download_assets(aoi=aoi)
    tiles = raster.tiles_for_aoi(boundary)
    checked = raster.verify_tiles(tiles)
    if not checked["ok"].all():
        raster.repair_tiles(tiles)
        checked = raster.verify_tiles(tiles)
        if not checked["ok"].all():
            raise RuntimeError(f"{(~checked['ok']).sum()} source files still unusable")
    print(f"   {len(tiles)} tiles, {checked['size_bytes'].sum() / 1e9:.1f} GB source")

    stage("raster stacks")
    raster.build_stacks(tiles, aoi=aoi)
    raster.build_mosaic(tiles, aoi=aoi)

    stage("reference labels")
    label_paths = labels.build_labels(tiles, aoi=aoi)
    labels.build_mosaic(tiles, aoi=aoi)
    print("  ", labels.class_counts(label_paths))

    stage("raster classification")
    samples_df = classify.sample_pixels(tiles, aoi=aoi)
    raster_model = None

    def get_raster_model() -> dict:
        nonlocal raster_model
        if raster_model is None:
            if samples_df is None:
                raster_model = classify.load_model(aoi)
            else:
                raster_model = classify.train(samples_df, aoi=aoi)
        return raster_model

    raster_metrics = checkpoint(
        config.results(aoi) / "raster_metrics.json",
        lambda: classify.evaluate(get_raster_model(), samples_df),
    )
    pred_paths = [config.tiles_pred(aoi) / f"{tile}_{config.PERIOD}_pred.tif" for tile in tiles]
    if all(path.exists() for path in pred_paths):
        classify.build_predictions({}, tiles, aoi=aoi)
    else:
        classify.build_predictions(get_raster_model(), tiles, aoi=aoi)
    classify.build_prediction_mosaic(tiles, aoi=aoi)
    samples_df = None
    if raster_model is not None:
        del raster_model
        raster_model = None
    gc.collect()
    print(
        f"   raster OA={raster_metrics['overall_accuracy']:.4f}"
        f" macroF1={raster_metrics['macro_f1']:.4f}"
    )

    results = {}
    for level in levels:
        stage(f"A5 level {level}")
        dggs.build_dataset(tiles, level=level, aoi=aoi)
        cell_samples = dggs.sample_cells(level, aoi)
        cell_model = classify.train(cell_samples, aoi=aoi, name=f"a5_l{level}")
        del cell_samples
        gc.collect()
        dggs.predict_cells(cell_model, level, aoi, name="a5_retrained")
        del cell_model
        gc.collect()
        dggs.predict_cells(classify.load_model(aoi), level, aoi, name="a5_transfer")
        gc.collect()
        dggs.aggregate_raster_to_cells(
            tiles,
            "pred_raster_on_cells",
            lambda t: config.tiles_pred(aoi) / f"{t}_{config.PERIOD}_pred.tif",
            level=level,
            aoi=aoi,
        )
        dggs.build_cell_stats(tiles, level=level, aoi=aoi)
        results[level] = report.compare(level, aoi)
        scores = results[level]["scores"]
        for name, score in scores.items():
            print(
                f"   {name:28s} OA={score['overall_accuracy']:.4f} macroF1={score['macro_f1']:.4f}"
            )

    stage("benchmarks")
    bench.register_views(aoi)
    print(bench.report(bench.run(aoi)))

    print(f"\ndone in {(time.perf_counter() - started) / 60:.1f} min")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--aoi", default=config.AOI_NAME)
    parser.add_argument("--levels", default=str(config.A5_LEVEL))
    args = parser.parse_args()
    run(args.aoi, tuple(int(x) for x in args.levels.split(",")))
