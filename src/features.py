"""Spectral features. Shared by the raster arm and the A5 arm — this is what makes
the two experiments comparable."""

from __future__ import annotations

import numpy as np

from . import config

INDEX_NAMES = ["ndvi", "ndwi", "mndwi", "ndbi", "bsi"]
FEATURE_NAMES = [*config.BANDS, *INDEX_NAMES]


def _norm_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    total = a + b
    return np.divide(a - b, total, out=np.zeros_like(total, dtype="float32"), where=total != 0)


def compute(bands: np.ndarray) -> np.ndarray:
    """`bands` is (n, len(config.BANDS)) in config.BANDS order; returns (n, len(FEATURE_NAMES))."""
    x = bands.astype("float32", copy=False)
    col = {b: x[:, i] for i, b in enumerate(config.BANDS)}
    indices = np.column_stack(
        [
            _norm_diff(col["B08"], col["B04"]),
            _norm_diff(col["B03"], col["B08"]),
            _norm_diff(col["B03"], col["B11"]),
            _norm_diff(col["B11"], col["B08"]),
            _norm_diff(col["B11"] + col["B04"], col["B08"] + col["B02"]),
        ]
    )
    return np.column_stack([x, indices])
