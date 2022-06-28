# Description

Alpha version. Usability updates coming soon.

The objective of this package is to define experiments to benchmark
models on a given dataset by calculating various metrics. It provides
the class `Model` and every subclasses defining the right methods can
then be used in experiments.

## Demonstration

[![Presentation](https://img.youtube.com/vi/s0gMqWn-nXo/0.jpg)](https://www.youtube.com/watch?v=s0gMqWn-nXo)

# The idea

The `datatype` refers type of the data which informs about the structure
of the data. A given `datatype` as the exact same `datafeature` which is
the name their features (observations). `Datatype` is a collection of
multiple categories of input from different `ID`.

In `pandas` term, `ID` is the column with the same name and
`datafeature` is the name of the rest of the columns. The `datatype` is
store in a sequence of `filenames`.

First example,

``` example
datatype = "simulated_returns"
ID = ["ARMA1", "ARMA2", "RandomForest"]
feature = ["returns"]
```

Second example,

``` example
datatype = "TSX"
split = ["20160104", "20160105"]
ID = ["ABX", "BMO", "HXT"]
feature = ["open", "close", "high", "low"]
```

Note that the data is separated on two files (days) because of the
quantity of data. Otherwise only one file would be needed

# Installation

## Prerequisites

### Privileges

Admin privileges are **necessary** for installation of `Python` and `R`.
The rest of the installation is tested with console "`run as admin`" on
Windows, but it should be optional.

### Install Python

Tested on `Python` version 3.10.5. Might work on `Python` \>=3.8.0

### For Windows user

There may be an issue with the locale when working on Windows in French.
Changing the language to English (US) will solve the issue if you
encounter `UTF-8 codec error`.

### Create a virtual environment (optional, but suggested)

-   Option 1 : Create a virtual environment using `virtualenv`

``` shell
python -m pip install virtualenv
python -m virtualenv -p python3 $HOME/.venv/TSbench
```

Whenever you want to activate your virtual envrionment to work with

``` shell
source $HOME/.venv/TSbench/bin/activate
```

-   Option 2 : Create a virtual environment using `conda`

Install `conda` following the instructions
<https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html>

``` shell
conda create --name TSbench
conda activate TSbench
```

You can add the environment to Jupyter with the following:

``` shell
python -m pip install ipykernel
python -m pip install ipython
python -m ipykernel install --name TSbench --user
```

### R specific prerequisites

1.  Install R

    You can find details here <https://www.r-project.org/>, or install
    from <https://utstat.toronto.edu/cran/>. Make sure that console run
    the command `R` (by settings path correctly).

2.  For Linux

    Install `fortran` (`gfortran` or `gcc-fortran` depending on the
    distribution)

## Installation without R dependencies

Make sure to clone this repository and change current directory to the
TSbench directory.

``` shell
python -m pip install -e ."[noR]"
```

## Installation with R dependencies

### Scripted installation

Make sure to clone this repository and change current directory to the
TSbench directory.

``` shell
./installation
```

### Manual installation

1.  Install package without `R` dependencies

    ``` shell
    pytyon -m pip install -e ."[noR]"
    ```

2.  Install `rpy2`

    ``` shell
    pytyon -m pip install rpy["all"]
    ```

3.  Create the `$HOME/.Renviron` file with writable library path. Here
    is an example of such a file :

    ``` shell
    echo "R_HOME_USER = ${HOME}/.config/R
    R_LIBS_USER = ${HOME}/.config/R/packages
    R_PROFILE_USER = ${HOME}/.config/R/.Rprofile
    R_HISTFILE = ${HOME}/.config/R/history.Rhistory" >> $HOME/.Renviron
    ```

4.  Install the necessary R packages using R.

    Read the prompt and write "yes" when ask to use personal library

    ``` r
    install.packages("rugarch", repos="https://cloud.r-project.org")
    install.packages("rmgarch", repos="https://cloud.r-project.org")
    install.packages("MTS", repos="https://cloud.r-project.org")
    ```

# Usage

## Run tests

``` shell
python -m pytest
python -m pytest --R # to include R tests
```
