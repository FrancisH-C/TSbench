# Description

Alpha version. Usability updates coming soon.

The objective of this package is to define experiments to benchmark
models on a given dataset by calculating various metrics. It provides
the class `Model` and every subclasses defining the right methods can
then be used in experiments.

## Demonstration

[![Presentation](https://img.youtube.com/vi/s0gMqWn-nXo/0.jpg)](https://www.youtube.com/watch?v=s0gMqWn-nXo)

# Installation

The installation has been tested with Python from 3.10.0. See
[file:installation_supplement.md](installation_supplement.md) for more
information about installation.

``` shell
pip install -e .
```

You can choose to add extra dependencies by adding one of the following

- \[default\]
- \[test\]
- \[R\]
- \[all\]

## R integration

This section is used for `R` integration directly into TSbench. `rpy2`
is the bridge to use `R` in Python. However, **it is not supported on
windows**.

If this integration doesn't work for you, see the documentation for how
to use outputs from external packages into TSbench.

### Prerequisites

- `R`: You can find details here <https://www.r-project.org/>.
- `fortran:` `gfortran` or `gcc-fortran` [depending on the
  distribution](https://gcc.gnu.org/wiki/GFortranDistros)

Make sure you can run the commands correctly (by settings `$PATH` and
restarting the terminal).

### Installation and post-installation

``` shell
python -m pip install -e .[R]
python setup_R.py
```
