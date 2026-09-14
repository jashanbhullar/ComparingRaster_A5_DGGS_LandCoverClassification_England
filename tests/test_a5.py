from __future__ import annotations

import numpy as np

from src import a5


def test_lonlat_to_cell_preserves_input_order() -> None:
    lon = np.linspace(-5.7, 1.7, 50_000)
    lat = np.linspace(50.0, 55.8, 50_000)

    batch = a5.lonlat_to_cell(lon, lat, level=18)
    reversed_batch = a5.lonlat_to_cell(lon[::-1], lat[::-1], level=18)

    np.testing.assert_array_equal(batch, reversed_batch[::-1])
