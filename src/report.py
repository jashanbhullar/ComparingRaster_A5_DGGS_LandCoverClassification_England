"""Like-for-like comparison of the raster and A5 arms over identical ground."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from . import a5, config
from .classify import block_id, is_test
from .io import checkpoint


def mcnemar(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    """Exact-ish McNemar with continuity correction on the paired disagreements."""
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    if b + c == 0:
        return {"b": b, "c": c, "statistic": 0.0, "p_value": 1.0}
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    return {"b": b, "c": c, "statistic": float(stat), "p_value": float(chi2.sf(stat, 1))}


def paired_table(level: int = config.A5_LEVEL, aoi: str = config.AOI_NAME) -> pd.DataFrame:
    """One row per A5 cell: reference label, A5 prediction, raster prediction aggregated
    to the same cell. This is the only fair basis for comparing the two arms."""
    con = a5.connect()
    processed = config.processed(aoi).as_posix()
    df = con.execute(f"""
        SELECT
            a.cell_id,
            a.label,
            a.pred AS pred_a5,
            t.pred AS pred_a5_transfer,
            r.value AS pred_raster,
            s.n_pixels,
            s.label_purity,
            s.b08_std,
            s.b08_mean,
            ST_X(pt) AS lon,
            ST_Y(pt) AS lat
        FROM read_parquet(
            '{processed}/{config.A5_PRIMARY_PREDICTION_NAME}_l{level}.parquet'
        ) a
        JOIN read_parquet('{processed}/pred_raster_on_cells_l{level}.parquet') r
          USING (cell_id)
        JOIN read_parquet(
            '{config.interim(aoi).as_posix()}/cell_stats_l{level}.parquet') s
          USING (cell_id)
        LEFT JOIN read_parquet('{processed}/pred_a5_transfer_l{level}.parquet') t
          USING (cell_id),
        LATERAL (SELECT a5_cell_to_point(a.cell_id) AS pt)
        WHERE a.label != {config.NODATA_CLASS}
          AND a.label NOT IN ({", ".join(str(c) for c in config.EXCLUDE_CLASSES)})
    """).fetchdf()
    df["block"] = block_id(df["lon"].to_numpy(), df["lat"].to_numpy())
    df["test"] = is_test(df["block"].to_numpy())
    return df


def _paired_sql(level: int, aoi: str, *, where: str | None = None) -> str:
    processed = config.processed(aoi).as_posix()
    retrained = f"read_parquet('{processed}/{config.A5_PRIMARY_PREDICTION_NAME}_l{level}.parquet')"
    raster = f"read_parquet('{processed}/pred_raster_on_cells_l{level}.parquet')"
    stats = f"read_parquet('{config.interim(aoi).as_posix()}/cell_stats_l{level}.parquet')"
    transfer = f"read_parquet('{processed}/pred_a5_transfer_l{level}.parquet')"
    if where:
        retrained = f"(SELECT * FROM {retrained} WHERE {where})"
        raster = f"(SELECT * FROM {raster} WHERE {where})"
        stats = f"(SELECT * FROM {stats} WHERE {where})"
        transfer = f"(SELECT * FROM {transfer} WHERE {where})"
    excluded = ""
    if config.EXCLUDE_CLASSES:
        values = ", ".join(str(value) for value in config.EXCLUDE_CLASSES)
        excluded = f" AND a.label NOT IN ({values})"
    return f"""
        SELECT
            a.cell_id,
            a.label,
            a.pred AS pred_a5,
            t.pred AS pred_a5_transfer,
            r.value AS pred_raster,
            s.n_pixels,
            s.label_purity,
            s.b08_std,
            s.b08_mean,
            ST_X(pt) AS lon,
            ST_Y(pt) AS lat
                FROM {retrained} a
                JOIN {raster} r USING (cell_id)
                JOIN {stats} s USING (cell_id)
                LEFT JOIN {transfer} t USING (cell_id),
        LATERAL (SELECT a5_cell_to_point(a.cell_id) AS pt)
                WHERE a.label != {config.NODATA_CLASS}{excluded}
    """


def _empty_confusion() -> np.ndarray:
    return np.zeros((max(config.CLASS_NAMES) + 1, max(config.CLASS_NAMES) + 1), dtype="int64")


def _add_confusion(counts: np.ndarray, y: np.ndarray, pred: np.ndarray) -> None:
    np.add.at(counts, (y.astype("int64"), pred.astype("int64")), 1)


def _score_counts(counts: np.ndarray) -> dict:
    support = counts.sum(axis=1)
    predicted = counts.sum(axis=0)
    observed = np.flatnonzero(support + predicted)
    if observed.size == 0:
        return {
            "overall_accuracy": 0.0,
            "macro_f1": 0.0,
            "per_class_f1": {},
            "confusion_matrix": [],
            "labels": [],
        }
    sub = counts[np.ix_(observed, observed)]
    tp = np.diag(sub).astype("float64")
    support = sub.sum(axis=1).astype("float64")
    predicted = sub.sum(axis=0).astype("float64")
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted != 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support != 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom != 0)
    total = sub.sum()
    return {
        "overall_accuracy": float(tp.sum() / total) if total else 0.0,
        "macro_f1": float(f1.mean()),
        "per_class_f1": {
            config.CLASS_NAMES[int(c)]: float(v) for c, v in zip(observed, f1, strict=True)
        },
        "confusion_matrix": sub.tolist(),
        "labels": [config.CLASS_NAMES[int(c)] for c in observed],
    }


def _score(y: np.ndarray, pred: np.ndarray) -> dict:
    classes = sorted(np.unique(np.concatenate([y, pred])).tolist())
    return {
        "overall_accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "per_class_f1": {
            config.CLASS_NAMES[c]: float(f)
            for c, f in zip(
                classes,
                f1_score(y, pred, average=None, labels=classes, zero_division=0),
                strict=True,
            )
        },
        "confusion_matrix": confusion_matrix(y, pred, labels=classes).tolist(),
        "labels": [config.CLASS_NAMES[c] for c in classes],
    }


def compare(level: int = config.A5_LEVEL, aoi: str = config.AOI_NAME, *, force: bool = False):
    """Score every arm on the same held-out cells and test the differences."""

    output = config.results(aoi) / f"comparison_l{level}.json"
    inputs = [
        config.processed(aoi) / f"{config.A5_PRIMARY_PREDICTION_NAME}_l{level}.parquet",
        config.processed(aoi) / f"pred_a5_transfer_l{level}.parquet",
        config.processed(aoi) / f"pred_raster_on_cells_l{level}.parquet",
        config.interim(aoi) / f"cell_stats_l{level}.parquet",
    ]
    stale = output.exists() and any(
        path.stat().st_mtime > output.stat().st_mtime for path in inputs
    )

    def build() -> dict:
        con = a5.connect(aoi=aoi)
        arms = ("raster_aggregated_to_cells", "a5_retrained", "a5_transfer")
        pred_columns = {
            "raster_aggregated_to_cells": "pred_raster",
            "a5_retrained": "pred_a5",
            "a5_transfer": "pred_a5_transfer",
        }
        counts = {name: _empty_confusion() for name in arms}
        significance_counts = {"a5_retrained": [0, 0], "a5_transfer": [0, 0]}
        n_cells = n_test_cells = pure_cells = 0
        n_pixels_sum = purity_sum = b08_std_sum = b08_cv_sum = 0.0
        b08_cv_count = 0
        mixed_total = np.zeros(max(config.CLASS_NAMES) + 1, dtype="int64")
        mixed_count = np.zeros(max(config.CLASS_NAMES) + 1, dtype="int64")
        purity_band_totals = {"pure": 0, "0.75-1.0": 0, "<0.75": 0}
        purity_band_correct = {"pure": 0, "0.75-1.0": 0, "<0.75": 0}

        for bucket in range(config.A5_CELL_BUCKETS):
            where = f"hash(cell_id) % {config.A5_CELL_BUCKETS} = {bucket}"
            reader = con.execute(_paired_sql(level, aoi, where=where)).fetch_record_batch(
                config.A5_COMPARE_BATCH_ROWS
            )
            for batch in reader:
                df = batch.to_pandas()
                purity = df["label_purity"].to_numpy()
                labels = df["label"].to_numpy().astype("int64")
                n_cells += len(df)
                n_pixels_sum += float(df["n_pixels"].sum())
                purity_sum += float(purity.sum())
                pure_cells += int((purity == 1.0).sum())
                b08_std_sum += float(df["b08_std"].sum())
                means = df["b08_mean"].where(
                    df["b08_mean"].abs() > config.A5_B08_MEAN_EPSILON,
                    np.nan,
                )
                cv = df["b08_std"] / means
                b08_cv_sum += float(cv.sum(skipna=True))
                b08_cv_count += int(cv.notna().sum())
                np.add.at(mixed_total, labels, 1)
                np.add.at(mixed_count, labels, (purity < 1.0).astype("int64"))

                test_mask = is_test(block_id(df["lon"].to_numpy(), df["lat"].to_numpy()))
                if not test_mask.any():
                    continue
                n_test_cells += int(test_mask.sum())
                y = labels[test_mask]
                base_correct = df.loc[test_mask, "pred_raster"].to_numpy() == y
                for name in arms:
                    pred = df.loc[test_mask, pred_columns[name]].to_numpy().astype("int64")
                    _add_confusion(counts[name], y, pred)
                    if name in significance_counts:
                        correct = pred == y
                        significance_counts[name][0] += int(np.sum(base_correct & ~correct))
                        significance_counts[name][1] += int(np.sum(~base_correct & correct))
                for band, mask in _purity_bands(df.loc[test_mask]):
                    band_correct = df.loc[test_mask, "pred_a5"].to_numpy()[mask] == y[mask]
                    purity_band_totals[band] += int(mask.sum())
                    purity_band_correct[band] += int(band_correct.sum())

        scores = {name: _score_counts(counts[name]) for name in arms}
        significance = {
            name: mcnemar_from_counts(*values) for name, values in significance_counts.items()
        }
        info_loss = {
            "n_cells": int(n_cells),
            "n_test_cells": int(n_test_cells),
            "mean_pixels_per_cell": float(n_pixels_sum / n_cells) if n_cells else 0.0,
            "pure_cell_fraction": float(pure_cells / n_cells) if n_cells else 0.0,
            "mean_label_purity": float(purity_sum / n_cells) if n_cells else 0.0,
            "mean_within_cell_b08_std": float(b08_std_sum / n_cells) if n_cells else 0.0,
            "mean_within_cell_b08_cv": float(b08_cv_sum / b08_cv_count) if b08_cv_count else 0.0,
            "mixed_cell_fraction_by_class": {
                config.CLASS_NAMES[c]: float(mixed_count[c] / mixed_total[c])
                for c in np.flatnonzero(mixed_total)
            },
            "accuracy_by_purity": {
                band: float(purity_band_correct[band] / purity_band_totals[band])
                if purity_band_totals[band]
                else 0.0
                for band in purity_band_totals
            },
        }
        return {"level": level, "scores": scores, "mcnemar": significance, "info_loss": info_loss}

    return checkpoint(output, build, force=force or stale)


def mcnemar_from_counts(b: int, c: int) -> dict:
    if b + c == 0:
        return {"b": b, "c": c, "statistic": 0.0, "p_value": 1.0}
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    return {"b": b, "c": c, "statistic": float(stat), "p_value": float(chi2.sf(stat, 1))}


def _purity_bands(test: pd.DataFrame):
    purity = test["label_purity"].to_numpy()
    yield "pure", purity == 1.0
    yield "0.75-1.0", (purity >= 0.75) & (purity < 1.0)
    yield "<0.75", purity < 0.75


CLASS_COLOURS = {
    0: "#ffffff",
    1: "#006400",
    2: "#ffbb22",
    3: "#ffff4c",
    4: "#f096ff",
    5: "#fa0000",
    6: "#b4b4b4",
    7: "#0064c8",
    8: "#0096a0",
}


def study_area_figure(aoi: str = "england"):
    """Plot the AOI and intersecting Sentinel-2 tile footprints for the thesis."""
    import geopandas as gpd
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    boundary = gpd.read_file(config.aoi_geojson(aoi)).to_crs(config.CRS)
    footprints = gpd.read_file(config.INTERIM / "tile_footprints.geojson").to_crs(config.CRS)
    footprints = footprints[footprints.intersects(boundary.geometry.iloc[0])]

    fig, ax = plt.subplots(figsize=(7.2, 8.5))
    boundary.plot(ax=ax, color="#dce8d5", edgecolor="#21352b", linewidth=0.8)
    footprints.boundary.plot(ax=ax, color="#b44b35", linewidth=0.55, alpha=0.8)
    ax.set_title("England study area and intersecting Sentinel-2 tiles")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(
        0.02,
        0.02,
        f"{len(footprints)} MGRS tiles",
        transform=ax.transAxes,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85},
    )
    fig.tight_layout()
    path = config.results(aoi) / "thesis_study_area.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def cell_map(
    bounds: tuple[float, float, float, float],
    level: int = config.A5_LEVEL_COARSE,
    aoi: str = config.AOI_NAME,
):
    """Cell geometries for a small extent, regenerated on demand from `cell_id`."""
    import geopandas as gpd
    from shapely import wkb

    processed = config.processed(aoi).as_posix()
    minx, miny, maxx, maxy = bounds
    # a5_geometry_to_cells only covers a polygon's boundary, so select by cell centre.
    df = (
        a5.connect(aoi=aoi)
        .execute(f"""
        WITH cells AS (
            SELECT a.cell_id, a.label, a.pred AS pred_a5, r.value AS pred_raster,
                   a5_cell_to_point(a.cell_id) AS pt
            FROM read_parquet('{processed}/pred_a5_retrained_l{level}.parquet') a
            JOIN read_parquet('{processed}/pred_raster_on_cells_l{level}.parquet') r
              USING (cell_id)
        )
        SELECT cell_id, label, pred_a5, pred_raster,
               ST_AsWKB(a5_cell_to_geometry(cell_id)) AS geom
        FROM cells
        WHERE ST_X(pt) BETWEEN {minx} AND {maxx}
          AND ST_Y(pt) BETWEEN {miny} AND {maxy}
    """)
        .fetchdf()
    )
    return gpd.GeoDataFrame(
        df.drop(columns=["geom"]),
        geometry=[wkb.loads(bytes(g)) for g in df["geom"]],
        crs=config.CRS,
    )


def figures(aoi: str = config.AOI_NAME) -> list:
    import json

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    out_dir = config.results(aoi)
    written = []

    results = {
        lv: json.loads((out_dir / f"comparison_l{lv}.json").read_text())
        for lv in (config.A5_LEVEL, config.A5_LEVEL_COARSE)
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    arms = ["raster_aggregated_to_cells", "a5_retrained", "a5_transfer"]
    short = ["raster", "A5 retrained", "A5 transfer"]
    width = 0.35
    for ax, metric in zip(axes[:2], ["overall_accuracy", "macro_f1"], strict=True):
        for i, lv in enumerate(results):
            vals = [results[lv]["scores"][a][metric] for a in arms]
            ax.bar(np.arange(len(arms)) + i * width, vals, width, label=f"L{lv}")
        ax.set_xticks(np.arange(len(arms)) + width / 2)
        ax.set_xticklabels(short, fontsize=9)
        ax.set_title(metric.replace("_", " "))
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)

    ax = axes[2]
    bands = ["pure", "0.75-1.0", "<0.75"]
    for i, lv in enumerate(results):
        vals = [results[lv]["info_loss"]["accuracy_by_purity"][b] for b in bands]
        ax.bar(np.arange(len(bands)) + i * width, vals, width, label=f"L{lv}")
    ax.set_xticks(np.arange(len(bands)) + width / 2)
    ax.set_xticklabels(bands)
    ax.set_title("A5 accuracy by within-cell label purity")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = out_dir / "m8_comparison.png"
    fig.savefig(path, dpi=110)
    written.append(path)

    # Central London: raster prediction, A5 prediction and their disagreement.
    gdf = cell_map((-0.16, 51.49, -0.09, 51.53), config.A5_LEVEL_COARSE, aoi)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, column, title in zip(
        axes,
        ["pred_raster", "pred_a5", None],
        ["Raster prediction on cells", "A5 prediction", "Disagreement"],
        strict=True,
    ):
        if column is None:
            colours = np.where(gdf["pred_raster"] == gdf["pred_a5"], "#eeeeee", "#d62728")
        else:
            colours = [CLASS_COLOURS[int(v)] for v in gdf[column]]
        gdf.plot(ax=ax, color=colours, linewidth=0)
        ax.set_title(title)
        ax.set_axis_off()
    axes[0].legend(
        handles=[
            mpatches.Patch(color=CLASS_COLOURS[c], label=config.CLASS_NAMES[c])
            for c in sorted(set(gdf["pred_a5"]) | set(gdf["pred_raster"]))
        ],
        loc="lower left",
        fontsize=8,
        ncol=3,
    )
    fig.tight_layout()
    path = out_dir / "m8_central_london_cells.png"
    fig.savefig(path, dpi=110)
    written.append(path)
    return written


def thesis_figures(aoi: str = "england") -> list:
    """Build overview figures from cached result artefacts."""
    import json

    import geopandas as gpd
    import matplotlib
    import rasterio
    from matplotlib.colors import ListedColormap
    from rasterio.enums import Resampling

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = config.results(aoi)
    comparison = json.loads((out_dir / f"comparison_l{config.A5_LEVEL}.json").read_text())
    benchmarks = json.loads((out_dir / "benchmarks.json").read_text())
    written = []

    scores = comparison["scores"]
    arms = ["raster_aggregated_to_cells", "a5_retrained", "a5_transfer"]
    labels = ["Raster majority\non A5 cells", "A5 retrained", "Raster model\non A5 features"]
    colours = ["#4c78a8", "#f2a541", "#2a9d8f"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, metric, title in zip(
        axes,
        ["overall_accuracy", "macro_f1"],
        ["Overall accuracy", "Macro F1"],
        strict=True,
    ):
        values = [scores[arm][metric] for arm in arms]
        bars = ax.bar(labels, values, color=colours)
        ax.bar_label(bars, labels=[f"{value:.1%}" for value in values], padding=3)
        ax.set_title(title)
        ax.set_ylim(0, 0.85)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Like-for-like A5 level {config.A5_LEVEL} comparison over held-out cells")
    fig.tight_layout()
    path = out_dir / "thesis_accuracy_comparison.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    written.append(path)

    class_labels = scores[arms[0]]["labels"]
    x = np.arange(len(class_labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 5))
    for index, (arm, label, colour) in enumerate(zip(arms, labels, colours, strict=True)):
        values = [scores[arm]["per_class_f1"][name] for name in class_labels]
        ax.bar(x + (index - 1) * width, values, width, label=label.replace("\n", " "), color=colour)
    ax.set_xticks(x, [name.replace("_", " ") for name in class_labels], rotation=25, ha="right")
    ax.set_ylabel("F1 score")
    ax.set_ylim(0, 0.9)
    ax.set_title("Class-level performance on held-out A5 cells")
    ax.legend(frameon=False, ncol=3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = out_dir / "thesis_per_class_f1.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    written.append(path)

    query_names = ["point_lookup", "neighbourhood", "zonal_lsoa", "group_by_class"]
    query_labels = ["Point lookup", "Neighbourhood", "LSOA zonal", "Group by class"]
    raster_times = [benchmarks["queries"][name]["raster_s"] for name in query_names]
    a5_times = [benchmarks["queries"][name]["a5_s"] for name in query_names]
    fig, ax = plt.subplots(figsize=(10, 5))
    y = np.arange(len(query_names))
    ax.barh(y - 0.18, raster_times, 0.36, label="Raster", color=colours[0])
    ax.barh(y + 0.18, a5_times, 0.36, label="A5 Parquet", color=colours[2])
    ax.set_yticks(y, query_labels)
    ax.set_xscale("log")
    ax.set_xlabel("Median query time (seconds, log scale)")
    ax.set_title("The faster representation depends on the query")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = out_dir / "thesis_query_benchmarks.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    written.append(path)

    prediction_path = config.processed(aoi) / f"pred_raster_{config.PERIOD}.vrt"
    with rasterio.open(prediction_path) as src:
        scale = max(src.width / 1400, src.height / 1400, 1)
        out_height = max(1, round(src.height / scale))
        out_width = max(1, round(src.width / scale))
        prediction = src.read(
            1,
            out_shape=(out_height, out_width),
            resampling=Resampling.nearest,
            masked=True,
        )
        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]

    boundary = gpd.read_file(config.aoi_geojson(aoi)).to_crs(config.CRS)
    class_ids = sorted(config.CLASS_NAMES)
    cmap = ListedColormap([CLASS_COLOURS[class_id] for class_id in class_ids])
    fig, ax = plt.subplots(figsize=(7.5, 9))
    ax.imshow(
        prediction,
        cmap=cmap,
        vmin=min(class_ids) - 0.5,
        vmax=max(class_ids) + 0.5,
        extent=extent,
        interpolation="nearest",
    )
    boundary.boundary.plot(ax=ax, color="#252525", linewidth=0.5)
    handles = [
        matplotlib.patches.Patch(
            color=CLASS_COLOURS[class_id], label=config.CLASS_NAMES[class_id].replace("_", " ")
        )
        for class_id in class_ids
        if class_id != config.NODATA_CLASS and class_id not in config.EXCLUDE_CLASSES
    ]
    ax.legend(handles=handles, loc="lower left", frameon=True, fontsize=8, ncol=2)
    period_label = config.PERIOD.replace("_", " to ")
    ax.set_title(f"Random-forest land-cover prediction, England\nSentinel-2 {period_label}")
    ax.set_axis_off()
    fig.tight_layout()
    path = out_dir / "thesis_england_prediction_map.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    written.append(path)

    return written


def thesis_detail_figures(aoi: str = "england") -> list:
    """Build the remaining detailed figures from cached result artefacts."""
    import json

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    out_dir = config.results(aoi)
    raster_metrics = json.loads((out_dir / "raster_metrics.json").read_text())
    comparison = json.loads((out_dir / f"comparison_l{config.A5_LEVEL}.json").read_text())
    written = []

    matrix = np.asarray(raster_metrics["confusion_matrix"], dtype="float64")
    row_totals = matrix.sum(axis=1, keepdims=True)
    matrix = np.divide(matrix, row_totals, out=np.zeros_like(matrix), where=row_totals != 0)
    class_labels = raster_metrics["labels"]
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(matrix, cmap="Blues", norm=Normalize(0, 1))
    fig.colorbar(image, ax=ax, label="Share of reference class")
    ax.set_xticks(
        range(len(class_labels)),
        [x.replace("_", " ") for x in class_labels],
        rotation=35,
        ha="right",
    )
    ax.set_yticks(range(len(class_labels)), [x.replace("_", " ") for x in class_labels])
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Reference class")
    ax.set_title("Corrected raster confusion matrix")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, f"{matrix[row, col]:.0%}", ha="center", va="center", fontsize=7)
    fig.tight_layout()
    path = out_dir / "thesis_raster_confusion_final.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    written.append(path)

    bands = ["pure", "0.75-1.0", "<0.75"]
    values = [comparison["info_loss"]["accuracy_by_purity"][band] for band in bands]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(["Pure", "0.75 to <1", "<0.75"], values, color=["#2a9d8f", "#f2a541", "#d95f59"])
    ax.bar_label(bars, labels=[f"{value:.1%}" for value in values], padding=3)
    ax.set_ylim(0, 0.9)
    ax.set_ylabel("A5-retrained accuracy")
    ax.set_title("Accuracy falls as within-cell label purity falls")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = out_dir / "thesis_purity_accuracy_final.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    written.append(path)

    # Small fixed windows keep geometry generation representative and bounded.
    regions = {
        "Urban": (-0.16, 51.49, -0.14, 51.51),
        "Agricultural": (-1.52, 52.18, -1.50, 52.20),
        "Forest/upland": (-2.90, 53.20, -2.88, 53.22),
        "Coastal/water": (-1.42, 50.77, -1.40, 50.79),
        "Wetland": (0.82, 52.71, 0.84, 52.73),
    }
    fig, axes = plt.subplots(5, 3, figsize=(12, 18), constrained_layout=True)
    columns = [
        ("label", "Reference"),
        ("pred_raster", "Raster majority"),
        ("pred_a5", "A5 retrained"),
    ]
    for row, (region, bounds) in enumerate(regions.items()):
        gdf = cell_map(bounds, config.A5_LEVEL, aoi)
        for col, (field, title) in enumerate(columns):
            ax = axes[row, col]
            if gdf.empty:
                ax.text(0.5, 0.5, "No cells in window", ha="center", va="center")
            else:
                colours = [CLASS_COLOURS[int(value)] for value in gdf[field]]
                gdf.plot(ax=ax, color=colours, linewidth=0)
            ax.set_title(f"{region}: {title}", fontsize=9)
            ax.set_axis_off()
        axes[row, 2].text(
            1.02,
            0.5,
            f"{len(gdf):,} cells",
            transform=axes[row, 2].transAxes,
            rotation=90,
            va="center",
            fontsize=8,
        )
    path = out_dir / "thesis_local_comparisons_final.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    written.append(path)
    return written
