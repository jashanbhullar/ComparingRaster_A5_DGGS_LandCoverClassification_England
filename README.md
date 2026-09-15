# Raster and A5 DGGS Land-Cover Classification

This repository contains the reusable code and tests for comparing raster-based land-cover classification with an A5 discrete global grid representation.

The workflow is designed for an AOI configured in `src/config.py` and supports a staged, checkpointed processing pipeline. It uses Sentinel-2 temporal mosaics, ESA WorldCover labels, administrative boundaries, DuckDB with the spatial and A5 extensions, rasterio, GeoPandas, Parquet, and scikit-learn.

## Scope

The public package contains:

- Python source modules for AOI preparation, raster processing, A5 conversion, feature generation, classification, validation, reporting, and benchmarking.
- The A5 row-order regression test.
- Python dependency metadata and the locked environment.
- This setup and structure documentation.

Large input datasets, downloaded source imagery, generated intermediate files, model files, reports, figures, logs, notebooks, and local databases are intentionally excluded. They must be obtained or generated separately under the applicable provider licences and access conditions.

## Folder structure

```text
ComparingRaster_A5_DGGS_LandCoverClassification_England/
├── README.md
├── folder_structure.txt
├── pyproject.toml
├── uv.lock
├── .python-version
├── src/
│   ├── a5.py          A5 conversion wrapper and ordered coordinate conversion
│   ├── aoi.py         Area-of-interest preparation
│   ├── bench.py       Storage and query benchmark helpers
│   ├── classify.py    Sampling, training, inference, and metrics
│   ├── config.py      CRS, bands, classes, paths, and run settings
│   ├── dggs.py        A5 construction and aggregation
│   ├── features.py    Shared spectral features
│   ├── io.py          Checkpoints and atomic writes
│   ├── labels.py      Reference-label preparation
│   ├── pipeline.py    Staged processing entry point
│   ├── raster.py      Raster alignment and mosaics
│   ├── report.py      Paired metrics and figures
│   └── validation.py  Bootstrap and validation analyses
└── tests/
 └── test_a5.py     A5 coordinate-order regression test
```

## Requirements

- Python 3.13 or a compatible version supported by `pyproject.toml`.
- `uv` for environment and dependency management.
- GDAL/rasterio-compatible system libraries.
- GDAL command-line tools (`gdalbuildvrt` and `gdalwarp`) for raster stages.
- Quarto and a LaTeX installation only if the separate thesis document is being built.
- Network access for the Sentinel-2 STAC catalogue, Sentinel-2 assets, and ESA WorldCover.
- Network access on first use so DuckDB can install the `spatial` and community `a5` extensions.

## Setup

```bash
uv sync
```

Run the A5 ordering regression test:

```bash
uv run pytest tests/test_a5.py
```

Run formatting and lint checks:

```bash
uv run ruff check src tests
```

## Required local input

The London and England configurations use small checked-in boundary files at
`aoi/london.geojson` and `aoi/england.geojson`. Each must contain a polygon or
multipolygon with CRS metadata; the pipeline converts it to EPSG:4326. No OSM
PBF, LSOA GeoPackage, Sentinel imagery, or WorldCover raster needs to be placed
in `data/` for these runs.

The UK configuration still uses an ONS LSOA GeoPackage at
`data/raw/lsoa_uk.gpkg` unless its source is changed in `src/config.py`.

## Public demo run

The lightweight public demo runs the A5 coordinate-order regression test and does
not require the large geospatial datasets:

```bash
uv run pytest tests/test_a5.py
```

The complete pipeline is a data-backed run rather than a small demo. It requires
network access and substantial storage; downloaded imagery and generated
artefacts are checkpointed under `data/` and reused on later runs. For London,
run:

```bash
uv run python -m src.pipeline --aoi london --levels 18
```

The pipeline discovers the configured AOI's Sentinel-2 assets from the Earth
Genome STAC catalogue using the AOI GeoJSON, writes
`data/raw/sentinel/sentinel_downloads.txt`, and downloads the 12 configured
bands for the selected period. Downloads are resumable and written atomically.
To prepare imagery without running later
stages, use:

```bash
uv run python -m src.download --aoi london
```

The first run requires network access and enough storage for the source
imagery. Existing manifests and readable rasters are reused; incomplete files
are detected and repaired before processing continues.

## Workflow

Before running the pipeline, configure the AOI and input locations in `src/config.py` and provide the required source datasets. Processing stages use checkpoints and atomic writes where applicable, so completed work can be reused after interruption.

## Reproducibility

The workflow uses a common EPSG:4326 coordinate system, fixed random seed settings, explicit spatial-block validation, and shared feature calculations for the raster and A5 pathways. The A5 conversion wrapper preserves input row identity by assigning an explicit row identifier and ordering converted results by that identifier.

The repository provides code and configuration. Reproducing complete England-scale metrics additionally requires the source datasets, external extensions, sufficient storage, and the generated artefacts produced by the pipeline.

## Data and attribution

The principal inputs are Sentinel-2 temporal mosaics, ESA WorldCover v200, Office for National Statistics boundaries, and the DuckDB A5 community extension. Obtain and cite each resource according to its provider's current terms. Do not commit raw downloads or generated large data products to this repository.
