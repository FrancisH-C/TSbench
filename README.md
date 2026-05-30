# TSbench

[![Tests](https://github.com/FrancisH-C/TSbench/actions/workflows/python-package.yml/badge.svg)](https://github.com/FrancisH-C/TSbench/actions/workflows/python-package.yml)
[![Docs](https://github.com/FrancisH-C/TSbench/actions/workflows/docs.yml/badge.svg)](https://francish-c.github.io/TSbench/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

A Python framework for benchmarking time series forecasting and generation models.

TSbench enables a unified *generate-and-benchmark* workflow where models serve as both data generators and forecasters, enabling closed-loop evaluation against known data generating processes (DGPs). This is a methodology well-established in econometrics but lacking a dedicated open-source tool — existing benchmark frameworks only compare forecasters on fixed real datasets, making it impossible to isolate model error from data complexity.

**Key features:**

- **Dual generate/forecast models** — the same model (ARMA, GARCH, etc.) generates synthetic data *and* forecasts it, enabling evaluation where the true DGP is known
- **Multivariate GARCH family** — VEC-GARCH, VEC-SPD-GARCH, and DCC-GARCH implementations rare in the Python ecosystem
- **Structured time series storage** — `(ID, timestamp, dim)` MultiIndex Parquet-backed data layer with metadata tracking
- **Cross-language integration** — R model wrappers (rGARCH via rpy2) for direct Python/R comparison
- **Reproducible experiments** — seeded RNG (PCG64), configurable pipeline, parallel processing via joblib

## Quick Start

Install TSbench:

```shell
python -m pip install .
```

Generate and forecast with an ARMA model:

```python
from numpy.random import Generator, PCG64
from TSbench import TSmodels

# Define an ARMA(1,1) model with a reproducible seed
model = TSmodels.ARMA(
    lag=1,
    rg=Generator(PCG64(1234)),
    dim_label=["price"],
    feature_label=["value"],
)

# Generate 100 time steps of synthetic data
data = model.generate(100)
print(data.head())

# Train the model on its own generated data
model.set_data(data=model.loader.get_df())
model.train()

# Forecast the next 5 steps
forecast = model.forecast(T=5)
print(forecast)
```

## Documentation

Full documentation is available at **[francish-c.github.io/TSbench](https://francish-c.github.io/TSbench/)**.

Example notebooks:

- [TSdata usage](notebooks/TSdata/example_TSdata.ipynb) — loading, storing, and querying time series data
- [TSmodels usage](notebooks/TSmodels/example_TSmodels.ipynb) — defining, generating, training, and forecasting
- [Experiment pipeline](notebooks/experiment/example_experiment.ipynb) — configuring and running a full experiment

## Installation Options

```shell
python -m pip install .[test]    # with test dependencies
python -m pip install .[docs]    # with documentation dependencies
python -m pip install .[R]       # with R support (Linux only)
python -m pip install .[all]     # everything
python -m pip install -e .[all]  # editable install for development
```

### Virtual Environment Setup

It is strongly recommended to use a virtual environment.

**Linux**

```shell
python -m pip install virtualenv
python -m virtualenv $HOME/.venv/TSbench
source $HOME/.venv/TSbench/bin/activate
```

**Windows (conda)**

```shell
conda create -n TSbench python=3.10 anaconda
conda activate TSbench
```

**Jupyter integration**

```shell
python -m pip install ipykernel ipython
python -m ipykernel install --name TSbench --user
```

### R Integration (Linux Only)

TSbench can use R models (rGARCH) via `rpy2`. This requires R and the following CRAN packages: `rugarch`, `rmgarch`, `MTS`, `jsonlite`.

**Ubuntu:**

```shell
R -e 'lib <- Sys.getenv("R_LIBS_USER"); dir.create(lib, recursive=TRUE, showWarnings=FALSE); install.packages(c("jsonlite","rugarch","rmgarch","MTS"), lib=lib, repos="https://cloud.r-project.org")'
python -m pip install .[R]
```

**Arch Linux:** install the build toolchain, then either pull the packages from the AUR (`r-rugarch`, `r-rmgarch`, `r-mts`, `r-jsonlite`) or use the manual install below.

```shell
sudo pacman -S gcc-fortran tcl tk
```

**Manual install (any distro):** install R, then from a shell:

```shell
R -e 'install.packages(c("jsonlite","rugarch","rmgarch","MTS"), repos="https://cloud.r-project.org")'
python -m pip install .[R]
```

If you already have an R library and the install fails with `package 'X' was found, but >= Y is required`, your existing CRAN packages are stale. Refresh them in your user library first:

```shell
R -e 'update.packages(ask=FALSE, repos="https://cloud.r-project.org", lib.loc=.libPaths()[1])'
```

To remove R support: `python -m pip uninstall rpy2`.

## Running Tests

```shell
python -m pytest -x -s          # basic tests
python -m pytest --run-R             # include R-dependent tests
python -m pytest --run-all           # all tests
python -m pytest --run-performance   # performance tests
```

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Citation

If you use TSbench in your research, please cite it:

```bibtex
@software{huot-chantal_tsbench,
  author = {Huot-Chantal, Francis and Bastin, Fabian},
  title = {{TSbench}: Time Series Benchmark in Python},
  url = {https://github.com/FrancisH-C/TSbench},
  license = {MIT}
}
```

See [CITATION.cff](CITATION.cff) for the full citation metadata.

## License

[MIT](LICENSE)
