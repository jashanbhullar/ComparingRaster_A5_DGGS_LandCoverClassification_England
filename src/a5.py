"""All A5 DuckDB extension calls live here, so an API change is a one-file fix."""

from __future__ import annotations

import os

import duckdb
import numpy as np
import pandas as pd

from . import config

_CON: duckdb.DuckDBPyConnection | None = None
_CON_AOI: str | None = None


def _configure(con: duckdb.DuckDBPyConnection, aoi: str | None = None) -> None:
    scratch = config.scratch(aoi or config.AOI_NAME) / "duckdb"
    scratch.mkdir(parents=True, exist_ok=True)
    temp_dir = scratch.as_posix().replace("'", "''")
    memory_limit = os.environ.get("DUCKDB_MEMORY_LIMIT", config.DUCKDB_MEMORY_LIMIT)
    con.execute("SET preserve_insertion_order = false;")
    con.execute(f"SET temp_directory = '{temp_dir}';")
    con.execute(f"SET memory_limit = '{memory_limit}';")


def connect(path: str | None = None, aoi: str | None = None) -> duckdb.DuckDBPyConnection:
    """Return a DuckDB connection with `spatial` and `a5` loaded (cached in-memory)."""
    global _CON, _CON_AOI
    if path is not None:
        con = duckdb.connect(path)
    else:
        requested_aoi = aoi or config.AOI_NAME
        if _CON is not None:
            if _CON_AOI == requested_aoi:
                return _CON
            _CON.close()
            _CON = None
        con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial;")
    con.execute("INSTALL a5 FROM community; LOAD a5;")
    _configure(con, aoi)
    if path is None:
        _CON = con
        _CON_AOI = aoi or config.AOI_NAME
    return con


def lonlat_to_cell(
    lon: np.ndarray,
    lat: np.ndarray,
    level: int = config.A5_LEVEL,
    aoi: str | None = None,
) -> np.ndarray:
    con = connect(aoi=aoi)
    df = pd.DataFrame(
        {
            "row_id": np.arange(len(lon), dtype="int64"),
            "lon": np.asarray(lon, dtype="float64"),
            "lat": np.asarray(lat, dtype="float64"),
        }
    )
    con.register("_pts", df)
    out = con.execute(
        f"SELECT a5_lonlat_to_cell(lon, lat, {level}) AS cell_id FROM _pts ORDER BY row_id"
    ).fetch_arrow_table()
    con.unregister("_pts")
    return out.column("cell_id").to_numpy(zero_copy_only=False)


def cell_to_lonlat(cell_ids: np.ndarray) -> pd.DataFrame:
    con = connect()
    con.register("_cells", pd.DataFrame({"cell_id": cell_ids}))
    out = con.execute(
        "SELECT ST_X(pt) AS lon, ST_Y(pt) AS lat FROM ("
        " SELECT a5_cell_to_point(cell_id) AS pt FROM _cells)"
    ).fetchdf()
    con.unregister("_cells")
    return out


def cell_area_m2(level: int) -> float:
    """Mean A5 cell area at `level`, in square metres."""
    con = connect()
    return con.execute(f"SELECT a5_cell_area({level})").fetchone()[0]


def level_area_table(levels: range | list[int]) -> pd.DataFrame:
    return pd.DataFrame({"level": list(levels), "area_m2": [cell_area_m2(x) for x in levels]})


def grid_disk(cell_id: int, k: int = 1) -> np.ndarray:
    con = connect()
    sql = f"SELECT UNNEST(a5_grid_disk({cell_id}::UBIGINT, {k})) AS cell_id"
    return con.execute(sql).fetchdf()["cell_id"].to_numpy()
