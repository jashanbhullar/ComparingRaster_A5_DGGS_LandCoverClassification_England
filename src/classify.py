"""Sampling, training, inference and metrics. Shared by the raster and A5 arms."""

from __future__ import annotations

import hashlib
import subprocess
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import rasterio
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from . import config, features
from .io import Manifest, checkpoint, checkpoint_file


def block_id(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Coarse spatial block, so overlapping samples cannot straddle the train/test split."""
    bx = np.floor(lon / config.SPLIT_BLOCK_DEG).astype("int32")
    by = np.floor(lat / config.SPLIT_BLOCK_DEG).astype("int32")
    return bx.astype("int64") * 100_000 + by


def is_test(blocks: np.ndarray) -> np.ndarray:
    """Deterministic block-wise hold-out; identical for the raster and A5 arms."""
    out = np.empty(blocks.shape, dtype=bool)
    for i, b in enumerate(blocks):
        digest = hashlib.md5(f"{config.SEED}:{b}".encode()).hexdigest()
        out[i] = (int(digest[:8], 16) % 1000) < config.TEST_FRACTION * 1000
    return out


def _sample_tile(tile: str, period: str, aoi: str, rng: np.random.Generator) -> pd.DataFrame:
    stack = config.stack_path(tile, period, aoi)
    label_path = config.tiles_labels(aoi) / f"{tile}_{period}_labels.tif"

    with rasterio.open(label_path) as src:
        labels = src.read(1)
        transform = src.transform

    rows, cols, classes = [], [], []
    for cls in np.unique(labels):
        if cls == config.NODATA_CLASS or cls in config.EXCLUDE_CLASSES:
            continue
        r, c = np.nonzero(labels == cls)
        take = min(config.SAMPLES_PER_CLASS_PER_TILE, r.size)
        pick = rng.choice(r.size, size=take, replace=False)
        rows.append(r[pick])
        cols.append(c[pick])
        classes.append(np.full(take, cls, dtype="uint8"))
    if not rows:
        return pd.DataFrame()

    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    y = np.concatenate(classes)

    with rasterio.open(stack) as src:
        bands = np.stack([src.read(i + 1)[rows, cols] for i in range(len(config.BANDS))], axis=1)

    lon, lat = rasterio.transform.xy(transform, rows, cols)
    df = pd.DataFrame(bands, columns=config.BANDS)
    df["label"] = y
    df["lon"] = np.asarray(lon, dtype="float64")
    df["lat"] = np.asarray(lat, dtype="float64")
    df["tile"] = tile
    return df


def sample_pixels(
    tiles: list[str],
    period: str = config.PERIOD,
    aoi: str = config.AOI_NAME,
    *,
    force: bool = False,
) -> pd.DataFrame:
    """Stratified per-class sample per tile, with the spatial block split attached."""
    path = config.interim(aoi) / f"train_samples_raster_{period}.parquet"

    def build() -> pd.DataFrame:
        rng = np.random.default_rng(config.SEED)
        df = pd.concat([_sample_tile(t, period, aoi, rng) for t in tiles], ignore_index=True)
        df = df[df[config.BANDS].to_numpy().sum(axis=1) > 0].reset_index(drop=True)
        df["block"] = block_id(df["lon"].to_numpy(), df["lat"].to_numpy())
        df["test"] = is_test(df["block"].to_numpy())
        df["seed"] = config.SEED
        return df

    return checkpoint(path, build, force=force)


def to_xy(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    x = features.compute(df[config.BANDS].to_numpy())
    return x, df["label"].to_numpy()


def train(
    df: pd.DataFrame,
    aoi: str = config.AOI_NAME,
    *,
    name: str = "raster",
    max_train_rows: int | None = None,
    force: bool = False,
):
    """Random forest on the training blocks; the model artefact records its own seed."""
    path = config.models(aoi) / f"{name}_rf.joblib"

    def build():
        train_df = df[~df["test"]]
        cap = config.A5_MAX_TRAIN_ROWS if max_train_rows is None else max_train_rows
        if name.startswith("a5_") and len(train_df) > cap:
            print(
                f"   {name}: cap training rows {len(train_df):,} -> {cap:,}",
                flush=True,
            )
            train_df = train_df.sample(n=cap, random_state=config.SEED)
        x, y = to_xy(train_df)
        model = RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            n_jobs=config.RF_N_JOBS,
            random_state=config.SEED,
        )
        started = time.perf_counter()
        model.fit(x, y)
        return {
            "model": model,
            "features": features.FEATURE_NAMES,
            "classes": model.classes_.tolist(),
            "seed": config.SEED,
            "n_train": len(train_df),
            "fit_seconds": round(time.perf_counter() - started, 1),
        }

    return checkpoint(path, build, force=force)


def load_model(aoi: str = config.AOI_NAME, *, name: str = "raster") -> dict:
    return joblib.load(config.models(aoi) / f"{name}_rf.joblib")


def evaluate(bundle: dict, df: pd.DataFrame) -> dict:
    test_df = df[df["test"]]
    x, y = to_xy(test_df)
    pred = bundle["model"].predict(x)
    classes = sorted(np.unique(np.concatenate([y, pred])).tolist())
    precision, recall, f1, support = precision_recall_fscore_support(
        y, pred, labels=classes, zero_division=0
    )
    return {
        "n_test": int(len(test_df)),
        "overall_accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "per_class": {
            config.CLASS_NAMES[c]: {
                "user_accuracy": float(p),
                "producer_accuracy": float(r),
                "f1": float(f),
                "support": int(s),
            }
            for c, p, r, f, s in zip(classes, precision, recall, f1, support, strict=True)
        },
        "labels": [config.CLASS_NAMES[c] for c in classes],
        "confusion_matrix": confusion_matrix(y, pred, labels=classes).tolist(),
    }


def predict_tile(bundle: dict, stack: Path, out: Path) -> None:
    """Windowed inference; a full 12-band tile is never held in memory."""
    model = bundle["model"]
    with rasterio.open(stack) as src:
        profile = src.profile
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

        result = np.full((src.height, src.width), config.NODATA_CLASS, dtype="uint8")
        for _, window in src.block_windows(1):
            block = src.read(window=window)
            flat = block.reshape(block.shape[0], -1).T
            valid = flat.sum(axis=1) > 0
            if not valid.any():
                continue
            pred = model.predict(features.compute(flat[valid]))
            out_block = np.full(flat.shape[0], config.NODATA_CLASS, dtype="uint8")
            out_block[valid] = pred
            rs, re = int(window.row_off), int(window.row_off + window.height)
            cs, ce = int(window.col_off), int(window.col_off + window.width)
            result[rs:re, cs:ce] = out_block.reshape(int(window.height), int(window.width))

    with rasterio.open(out, "w", **profile) as dst:
        dst.write(result, 1)


def build_predictions(
    bundle: dict,
    tiles: list[str],
    period: str = config.PERIOD,
    aoi: str = config.AOI_NAME,
    *,
    force: bool = False,
) -> list[Path]:
    """One prediction COG per tile. Resumable per tile."""
    manifest = Manifest(f"pred_{period}", aoi)
    paths = []
    for tile in tiles:
        stack = config.stack_path(tile, period, aoi)
        out = config.tiles_pred(aoi) / f"{tile}_{period}_pred.tif"
        checkpoint_file(out, lambda tmp, s=stack: predict_tile(bundle, s, tmp), force=force)
        if not manifest.done(tile):
            manifest.mark(tile, bytes=out.stat().st_size)
        paths.append(out)
    return paths


def build_prediction_mosaic(
    tiles: list[str],
    period: str = config.PERIOD,
    aoi: str = config.AOI_NAME,
    *,
    force: bool = False,
) -> Path:
    out = config.processed(aoi) / f"pred_raster_{period}.vrt"
    parts = [str(config.tiles_pred(aoi) / f"{t}_{period}_pred.tif") for t in tiles]

    def build(tmp: Path) -> None:
        subprocess.run(["gdalbuildvrt", "-q", "-overwrite", str(tmp), *parts], check=True)

    return checkpoint_file(out, build, force=force)
