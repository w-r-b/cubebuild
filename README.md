# cubebuild

`cubebuild` is the open-source construction pipeline of the **Solid Earth Data Cube (SEDC) v1.0**. It builds the cube; it is not the cube. The cube data are released separately — see [Data](#data).

The pipeline is driven by a single declarative layer contract (`cubebuild/layer-mapping-v0.yaml`) and produces a tiered Zarr v3 store (1°, 30′, 6′, 3′ and 30″ tiers), the vector sidecars, a machine-readable manifest (`cube_manifest.json`) and structural validation reports.

## What this repository is

This repository contains the construction pipeline only:

- the `cubebuild` Python package, with the `cubebuild` command-line entry point;
- the layer contract that drives every build (`cubebuild/layer-mapping-v0.yaml`);
- the test suite;
- the input-dataset inventory (`input-datasets.csv`).

It does not contain the data cube itself or the raw input data.

## Installation

Requires Python 3.11 or newer. Install from source:

```bash
git clone https://github.com/w-r-b/cubebuild.git
cd cubebuild
pip install -e .
```

To run the tests as well, install the test extra:

```bash
pip install -e ".[test]"
```

The full test suite requires the raw input data on a local disk and is therefore not covered by CI; CI runs the data-independent subset (`tests/test_contract.py` and `tests/test_pixel_area.py`).

## Reproduction scope

The full cube cannot be rebuilt from this repository alone. Construction requires about 22 GB of raw input data that are not included here, and part of that raw data cannot be redistributed by the authors. Readers who want to use the cube should use the public version (see [Data](#data)) instead of re-running the pipeline from scratch.

## Data

The public version of SEDC v1.0 is archived on Zenodo as a single tar package (DOI: to be registered). It contains 25 layers in the public Zarr v3 store and 4 vector sidecars. The full cube — 47 layers and 10 vector sidecars — is not publicly released: the redistribution licences of the remaining source datasets are unconfirmed or restrictive.

`input-datasets.csv` lists the input datasets with their sources, DOIs and licences.

## Citation

Citation metadata for the software is provided in `CITATION.cff`. If you use the cube data, please cite the dataset release — see [Data](#data) — as well as the original input datasets listed in `input-datasets.csv`.

## Licence

The code and the data are licensed separately:

- **Code** (this repository): MIT — see `LICENSE`.
- **Data** (the public SEDC v1.0 release): CC BY 4.0.

Licences of the original input datasets differ per dataset; see `input-datasets.csv`.