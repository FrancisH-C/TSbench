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

Set up a virtual environment and install TSbench:

```shell
python -m venv .venv
source .venv/bin/activate
python -m pip install .
```

Install from the package root. The bracketed extras are optional and can be combined (e.g. `.[test,docs]`):

```shell
python -m pip install .[test]    # + test suite dependencies (pytest)
python -m pip install .[docs]    # + Sphinx documentation toolchain
python -m pip install .[all]     # everything (test + docs + R)
python -m pip install -e .[all]  # editable install for development
```

For virtual environment setup (including Windows/conda and a Jupyter kernel), see the [installation guide](https://francish-c.github.io/TSbench/installation.html).

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

## R Integration (Linux Only)

TSbench can use R models (rGARCH) via `rpy2`. This requires R (≥ 4.5.0) and the CRAN packages `rugarch`, `rmgarch`, `MTS`, `jsonlite`. On Ubuntu, the default repositories ship an older R, so a recent R must be installed from CRAN first.

The quick path, once a recent R is installed:

```shell
R -e 'install.packages(c("jsonlite","rugarch","rmgarch","MTS"), repos="https://cloud.r-project.org")'
python -m pip install .[R]
```

For the complete step-by-step guide on Ubuntu (plus Arch/other distros) see [Installing R support (rpy2)](https://francish-c.github.io/TSbench/installation.html).

## Running Tests

For development, install the test dependencies, then run the suite from the package root:

```shell
python -m pip install -e .[test]

python -m pytest -x -s                # default suite (skips R + performance)
python -m pytest --run-R              # also run R-dependent tests
python -m pytest --run-performance    # also run performance tests
python -m pytest --run-all            # everything
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
