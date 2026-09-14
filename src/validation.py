"""England result validation and thesis-ready sensitivity analyses."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio

from . import a5, config, features
from .classify import block_id, is_test
from .io import checkpoint
from .report import _paired_sql, _score_counts


def _raster_population_confusion(aoi: str) -> np.ndarray:
    counts = np.zeros((max(config.CLASS_NAMES) + 1,) * 2, dtype="int64")
    for prediction_path in sorted(config.tiles_pred(aoi).glob("*_pred.tif")):
        tile = prediction_path.name.split("_")[0]
        label_path = config.tiles_labels(aoi) / f"{tile}_{config.PERIOD}_labels.tif"
        with rasterio.open(prediction_path) as predictions, rasterio.open(label_path) as labels:
            for _, window in predictions.block_windows(1):
                predicted = predictions.read(1, window=window).ravel()
                observed = labels.read(1, window=window).ravel()
                keep = (observed != config.NODATA_CLASS) & ~np.isin(
                    observed, list(config.EXCLUDE_CLASSES)
                )
                np.add.at(counts, (observed[keep], predicted[keep]), 1)
    return counts


def audit_raster_majority(aoi: str = "england", *, force: bool = False) -> dict:
    """Test whether low cell accuracy is already present before A5 aggregation."""

    def build() -> dict:
        pixel_counts = _raster_population_confusion(aoi)
        con = a5.connect(aoi=aoi)
        level = config.A5_LEVEL
        high_purity_counts = np.zeros_like(pixel_counts)
        high_purity_disagreements = 0
        high_purity_transfer_correct = 0
        samples: list[dict] = []

        for bucket in range(config.A5_CELL_BUCKETS):
            where = f"hash(cell_id) % {config.A5_CELL_BUCKETS} = {bucket}"
            reader = con.execute(_paired_sql(level, aoi, where=where)).fetch_record_batch(
                config.A5_COMPARE_BATCH_ROWS
            )
            for batch in reader:
                frame = batch.to_pandas()
                frame = frame[frame["label_purity"] >= 0.95]
                if frame.empty:
                    continue
                observed = frame["label"].to_numpy(dtype="int64")
                predicted = frame["pred_raster"].to_numpy(dtype="int64")
                np.add.at(high_purity_counts, (observed, predicted), 1)
                disagreement = frame["pred_raster"] != frame["label"]
                transfer_correct = frame["pred_a5_transfer"] == frame["label"]
                target = frame[disagreement & transfer_correct]
                high_purity_disagreements += int(disagreement.sum())
                high_purity_transfer_correct += int((disagreement & transfer_correct).sum())
                if len(samples) < 40 and not target.empty:
                    remaining = 40 - len(samples)
                    samples.extend(
                        target.head(remaining)[
                            [
                                "cell_id",
                                "label",
                                "pred_raster",
                                "pred_a5",
                                "pred_a5_transfer",
                                "n_pixels",
                                "label_purity",
                                "lon",
                                "lat",
                            ]
                        ].to_dict("records")
                    )

        return {
            "pixel_population": _score_counts(pixel_counts),
            "high_purity_cells": _score_counts(high_purity_counts),
            "high_purity_raster_errors": high_purity_disagreements,
            "high_purity_errors_corrected_by_transfer": high_purity_transfer_correct,
            "sample_transfer_wins": samples,
            "interpretation": (
                "If full-population pixel accuracy is close to cell-majority accuracy, "
                "the low result precedes A5 aggregation and is not primarily a join defect."
            ),
        }

    path = config.results(aoi) / "raster_majority_audit.json"
    return checkpoint(path, build, force=force)


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2))


def _metric_summary(counts: np.ndarray) -> tuple[float, float, float]:
    total = counts.sum()
    if total == 0:
        return 0.0, 0.0, 0.0
    observed = (counts.sum(axis=1) + counts.sum(axis=0)) > 0
    counts = counts[np.ix_(observed, observed)]
    tp = np.diag(counts).astype(float)
    support = counts.sum(axis=1).astype(float)
    predicted = counts.sum(axis=0).astype(float)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted != 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support != 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(tp),
        where=(precision + recall) != 0,
    )
    return float(tp.sum() / total), float(f1.mean()), float((f1 * support).sum() / total)


def spatial_block_bootstrap(
    level: int = config.A5_LEVEL,
    aoi: str = "england",
    *,
    n_boot: int = 1000,
    force: bool = False,
) -> dict:
    """Bootstrap corrected cell metrics by complete spatial test blocks."""

    def build() -> dict:
        con = a5.connect(aoi=aoi)
        block_counts: dict[int, np.ndarray] = {}
        for bucket in range(config.A5_CELL_BUCKETS):
            where = f"hash(cell_id) % {config.A5_CELL_BUCKETS} = {bucket}"
            reader = con.execute(_paired_sql(level, aoi, where=where)).fetch_record_batch(
                config.A5_COMPARE_BATCH_ROWS
            )
            for batch in reader:
                frame = batch.to_pandas()
                test = is_test(block_id(frame["lon"].to_numpy(), frame["lat"].to_numpy()))
                frame = frame.loc[test]
                if frame.empty:
                    continue
                blocks = block_id(frame["lon"].to_numpy(), frame["lat"].to_numpy())
                labels = frame["label"].to_numpy(dtype="int64")
                predictions = {
                    "raster_aggregated_to_cells": frame["pred_raster"].to_numpy(dtype="int64"),
                    "a5_retrained": frame["pred_a5"].to_numpy(dtype="int64"),
                    "a5_transfer": frame["pred_a5_transfer"].to_numpy(dtype="int64"),
                }
                for block in np.unique(blocks):
                    mask = blocks == block
                    counts = block_counts.setdefault(
                        int(block),
                        np.zeros(
                            (3, max(config.CLASS_NAMES) + 1, max(config.CLASS_NAMES) + 1),
                            dtype="int64",
                        ),
                    )
                    for index, name in enumerate(predictions):
                        np.add.at(counts[index], (labels[mask], predictions[name][mask]), 1)

        names = ("raster_aggregated_to_cells", "a5_retrained", "a5_transfer")
        blocks = np.stack([block_counts[key] for key in sorted(block_counts)])
        rng = np.random.default_rng(config.SEED)
        draws = rng.integers(0, len(blocks), size=(n_boot, len(blocks)))
        metrics = np.empty((n_boot, len(names), 3), dtype="float64")
        for draw_index, draw in enumerate(draws):
            total = blocks[draw].sum(axis=0)
            for method_index in range(len(names)):
                metrics[draw_index, method_index] = _metric_summary(total[method_index])

        def interval(values: np.ndarray) -> list[float]:
            return [float(x) for x in np.quantile(values, [0.025, 0.975])]

        return {
            "seed": config.SEED,
            "n_boot": n_boot,
            "n_blocks": int(len(blocks)),
            "metrics": {
                name: {
                    "overall_accuracy_ci": interval(metrics[:, index, 0]),
                    "macro_f1_ci": interval(metrics[:, index, 1]),
                    "weighted_f1_ci": interval(metrics[:, index, 2]),
                }
                for index, name in enumerate(names)
            },
            "differences_vs_raster": {
                name: {
                    "overall_accuracy_ci": interval(metrics[:, index, 0] - metrics[:, 0, 0]),
                    "macro_f1_ci": interval(metrics[:, index, 1] - metrics[:, 0, 1]),
                    "weighted_f1_ci": interval(metrics[:, index, 2] - metrics[:, 0, 2]),
                }
                for index, name in enumerate(names[1:], start=1)
            },
        }

    return checkpoint(config.results(aoi) / f"bootstrap_l{level}.json", build, force=force)


def score_external_a5_prediction(
    prediction_path: Path,
    level: int = config.A5_LEVEL,
    aoi: str = "england",
    *,
    force: bool = False,
) -> dict:
    """Score an additional cell prediction Parquet on the shared held-out cells."""
    output = config.results(aoi) / f"{prediction_path.stem}_metrics.json"

    def build() -> dict:
        con = a5.connect(aoi=aoi)
        labels_path = config.processed(aoi) / (
            f"{config.A5_PRIMARY_PREDICTION_NAME}_l{level}.parquet"
        )
        counts = np.zeros((max(config.CLASS_NAMES) + 1,) * 2, dtype="int64")
        for bucket in range(config.A5_CELL_BUCKETS):
            where = f"hash(a.cell_id) % {config.A5_CELL_BUCKETS} = {bucket}"
            query = f"""
                SELECT a.label, p.pred, ST_X(pt) AS lon, ST_Y(pt) AS lat
                FROM read_parquet('{labels_path}') a
                JOIN read_parquet('{prediction_path}') p USING (cell_id),
                LATERAL (SELECT a5_cell_to_point(a.cell_id) AS pt)
                WHERE a.label != {config.NODATA_CLASS} AND {where}
            """
            reader = con.execute(query).fetch_record_batch(config.A5_COMPARE_BATCH_ROWS)
            for batch in reader:
                frame = batch.to_pandas()
                test = is_test(block_id(frame["lon"].to_numpy(), frame["lat"].to_numpy()))
                frame = frame.loc[test]
                np.add.at(
                    counts,
                    (
                        frame["label"].to_numpy(dtype="int64"),
                        frame["pred"].to_numpy(dtype="int64"),
                    ),
                    1,
                )
        score = _score_counts(counts)
        score.update({"level": level, "n_test_cells": int(counts.sum()), "n_train": None})
        return score

    return checkpoint(output, build, force=force)


def raster_pixel_bootstrap(
    aoi: str = "england", *, n_boot: int = 1000, force: bool = False
) -> dict:
    """Bootstrap the stratified raster test sample by its complete spatial blocks."""

    def build() -> dict:
        import joblib
        import pandas as pd

        samples_path = config.interim(aoi) / f"train_samples_raster_{config.PERIOD}.parquet"
        samples = pd.read_parquet(samples_path)
        test = samples[samples["test"]].copy()
        bundle = joblib.load(config.models(aoi) / "raster_rf.joblib")
        test["pred"] = bundle["model"].predict(features.compute(test[config.BANDS].to_numpy()))
        blocks = block_id(test["lon"].to_numpy(), test["lat"].to_numpy())
        labels = test["label"].to_numpy(dtype="int64")
        predictions = test["pred"].to_numpy(dtype="int64")
        unique_blocks = np.unique(blocks)
        block_counts = []
        for block in unique_blocks:
            mask = blocks == block
            counts = np.zeros((max(config.CLASS_NAMES) + 1,) * 2, dtype="int64")
            np.add.at(counts, (labels[mask], predictions[mask]), 1)
            block_counts.append(counts)
        block_counts = np.stack(block_counts)
        rng = np.random.default_rng(config.SEED)
        draws = rng.integers(0, len(block_counts), size=(n_boot, len(block_counts)))
        values = np.array([_metric_summary(block_counts[draw].sum(axis=0)) for draw in draws])
        return {
            "seed": config.SEED,
            "n_boot": n_boot,
            "n_blocks": int(len(block_counts)),
            "overall_accuracy_ci": [float(x) for x in np.quantile(values[:, 0], [0.025, 0.975])],
            "macro_f1_ci": [float(x) for x in np.quantile(values[:, 1], [0.025, 0.975])],
            "weighted_f1_ci": [float(x) for x in np.quantile(values[:, 2], [0.025, 0.975])],
        }

    return checkpoint(config.results(aoi) / "raster_bootstrap.json", build, force=force)
