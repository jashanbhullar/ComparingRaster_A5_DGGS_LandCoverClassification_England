"""Checkpointing: never recompute an artefact that is already on disk."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from . import config


def _load(path: Path) -> Any:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        import pandas as pd

        return pd.read_parquet(path)
    if suffix in {".geojson", ".gpkg", ".fgb"}:
        import geopandas as gpd

        return gpd.read_file(path)
    if suffix == ".json":
        return json.loads(path.read_text())
    if suffix == ".joblib":
        import joblib

        return joblib.load(path)
    raise ValueError(f"No loader for {path.suffix}; use checkpoint_file instead")


def _save(obj: Any, path: Path) -> None:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        obj.to_parquet(path, index=False)
    elif suffix == ".geojson":
        obj.to_file(path, driver="GeoJSON")
    elif suffix == ".gpkg":
        # Explicit layer name: the atomic temp filename contains dots, which GPKG rejects.
        obj.to_file(path, driver="GPKG", layer="data")
    elif suffix == ".fgb":
        obj.to_file(path)
    elif suffix == ".json":
        path.write_text(json.dumps(obj, indent=2, default=str))
    elif suffix == ".joblib":
        import joblib

        joblib.dump(obj, path)
    else:
        raise ValueError(f"No saver for {path.suffix}; use checkpoint_file instead")


@contextmanager
def atomic(path: Path):
    """Yield a temp path next to `path`; rename onto `path` only on success."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.tmp{os.getpid()}{path.suffix}")
    if tmp.exists():
        tmp.unlink()
    try:
        yield tmp
        if not tmp.exists():
            raise FileNotFoundError(f"build did not write {tmp}")
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()


def checkpoint(path: Path, build: Callable[[], Any], *, force: bool = False) -> Any:
    """Return the artefact at `path`, else build it, save atomically, return it."""
    path = Path(path)
    if path.exists() and not (force or config.FORCE_REBUILD):
        print(f"skip  {path.name} (exists)")
        return _load(path)
    started = time.perf_counter()
    obj = build()
    with atomic(path) as tmp:
        _save(obj, tmp)
    print(f"build {path.name} in {time.perf_counter() - started:.1f}s")
    return obj


def checkpoint_file(path: Path, build: Callable[[Path], None], *, force: bool = False) -> Path:
    """Like `checkpoint` but `build(tmp_path)` writes the file itself (rasters, VRTs)."""
    path = Path(path)
    if path.exists() and not (force or config.FORCE_REBUILD):
        print(f"skip  {path.name} (exists)")
        return path
    started = time.perf_counter()
    with atomic(path) as tmp:
        build(tmp)
    print(f"build {path.name} in {time.perf_counter() - started:.1f}s")
    return path


class Manifest:
    """Append-only record of completed units of work, for resumable loops."""

    def __init__(self, name: str, aoi: str = config.AOI_NAME) -> None:
        self.path = config.manifests(aoi) / f"{name}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._done: set[str] = set()
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    self._done.add(json.loads(line)["key"])

    def done(self, key: str) -> bool:
        return key in self._done

    def mark(self, key: str, **info: Any) -> None:
        with self.path.open("a") as fh:
            fh.write(json.dumps({"key": key, "ts": time.time(), **info}, default=str) + "\n")
        self._done.add(key)

    def pending(self, keys: Iterable[str]) -> list[str]:
        return [k for k in keys if not self.done(k)]
