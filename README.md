# Description

Alpha version. Usability updates coming soon.

The objective of this package is to define experiments to benchmark
models on a given dataset by calculating various metrics. It provides
the class `Model` and every subclasses defining the right methods can
then be used in experiments.

## Demonstration

[![Presentation](https://img.youtube.com/vi/s0gMqWn-nXo/0.jpg)](https://www.youtube.com/watch?v=s0gMqWn-nXo)

# Quick start

## Prerequisites

### Privileges

Admin privileges are **necessary** for installation of `Python` and `R`.
The rest of the installation is tested with console `run as admin` on
Windows, but it should be optional.

### Install Python

Tested on `Python` version 3.10.5. Might work on `Python` \>=3.8.0

### Install R (optional)

You can find details here <https://www.r-project.org/>, or install from
<https://utstat.toronto.edu/cran/>. Make sure that console run the
command `R` (by settings path correctly).

### For Windows

There may be an issue with the locale when working on Windows in French.
Changing the language to English (US) will solve the issue if you
encounter `UTF-8 codec error`.

### For Linux

Some compilers may be needed. For example :

-   `fortran` (`gfortran` or `gcc-fortran` depending on the
    distribution)

## Create a virtual environment (optional, but suggested)

You can do as follows prior to the installation:

-   Option 1 : Create a virtual environment using `virtualenv`

``` shell
virtualenv -p python3 TSbench
. TSbench/bin/activate
```

-   Option 2 : Create a virtual environment using `conda`

``` shell
conda create --name TSbench
conda activate TSbench
```

You can add the environment to Jupyter with the following:

``` shell
pip install ipykernel
pip install ipython
ipython kernel install --name testWithR --user
```

Caution: Don't forget to reload the terminal!

## Installation

Make sure to clone this repo and change current directory to the TSbench
directory.

### Install all dependencies

This will install the `rpy2` `python` package, as well as some `R`
packages.

``` shell
pip install -e ."[all]"
```

### Install without R

``` shell
pip install -e ."[noR]"
```

### Bare minimum installation

``` shell
pip install -e .
```

## Run tests

### WIth R

``` shell
python -m pytest --R
```

### Without R

``` shell
python -m pytest
```
