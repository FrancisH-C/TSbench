# Description

Alpha version. Usability updates coming soon.

The objective of this package is to define experiments to benchmark
models on a given dataset by calculating various metrics. It provides
the class `Model` and every subclasses defining the right methods can
then be used in experiments.

## Demonstration

[![Presentation](https://img.youtube.com/vi/s0gMqWn-nXo/0.jpg)](https://www.youtube.com/watch?v=s0gMqWn-nXo)

# Installation

There is 2 types of installation : with `R` supported or without it.
`rpy2` is the bridge to use `R` in Python. However, **it is not
supported on windows**.

## Virtual environment

Here is an example using `virtualenv`. For details, see the official
documentation <https://virtualenv.pypa.io/en/latest/>.

### Create

``` shell
python -m pip install virtualenv
python -m virtualenv -p python3 $HOME/.venv/TSbench
```

### Activate

On every shell if you want to work within the virtual environment you
must first activate it. To activate the virtual environment you run the
following :

``` shell
source $HOME/.venv/TSbench/bin/activate
```

### Add to jupyter kernels

You can add the environment to Jupyter with the following:

``` shell
python -m pip install ipykernel
python -m pip install ipython
python -m ipykernel install --name TSbench --user
```

### Alternative : virtual environment with `conda`

``` shell
conda create --name TSbench
conda activate TSbench
```

## Installation for LInux

### Install Python

The latest version has been tested with Python 3.10.5. It is expected to
work from version 3.8.0

### Install R (for R support)

Use your favorite package manager. You can find details here
<https://www.r-project.org/>.

Make sure that terminal runs the command `R` everywhere (by settings
path correctly and restarting the terminal).

### Install gcc-Fortran (for R support)

Used to compile R packages. Install `fortran` (`gfortran` or
`gcc-fortran` depending on the distribution)

### Installation using script

Make sure to clone this repository and change current directory to the
`TSbench` directory.

-   With R support

    ``` shell
    python installation.py
    ```

-   Without R support

    ``` shell
    python installation.py --no-R
    ```

## Installation for Windows

### Install `conda`

<https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html>

### Install

Make sure to clone this repository and change current directory to the
`TSbench` directory.

``` shell
python -m pip install -e ."[noR]"
```

## Possible issues

More information

See [file:installation_supplement.md](installation_supplement.md) for
more information

Locale settings

There might be an issue with the locale when working on Windows in
French. Changing the language to English (US) will solve the issue if
you encounter `UTF-8 codec error`.

# Usage

## Run tests

``` shell
python -m pytest
python -m pytest --R # to include R tests
```
