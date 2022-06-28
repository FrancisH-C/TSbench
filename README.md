# Description

Alpha version. Usability updates coming soon.

The objective of this package is to define experiments to benchmark
models on a given dataset by calculating various metrics. It provides
the class `Model` and every subclasses defining the right methods can
then be used in experiments.

## Demonstration

[![Presentation](https://img.youtube.com/vi/s0gMqWn-nXo/0.jpg)](https://www.youtube.com/watch?v=s0gMqWn-nXo)

# Installation

## Prerequisites

### Privileges

Admin privileges are **necessary** for installation of `Python` and `R`.
The rest of the installation is tested with console "`run as admin`" on
Windows, but it should be optional.

### Install Python

The latest version has been tested with Python 3.10.5. It is expected to work from version 3.8.0

### For Windows user

There may be an issue with the locale when working on Windows in French.
Changing the language to English (US) will solve the issue if you
encounter `UTF-8 codec error`.

### Create a virtual environment (optional, but suggested)

-   Option 1 : Create a virtual environment using `virtualenv`

## Testé avec powershell et anaconda3. Python 3.9
## ~ est un raccourci Linux. Sous PowerShell, cela ne fonctionne pas.
## On peut utiliser $HOME
## J'ai le répertoire Scripts, pas bin
## La commande ipython n'a pas fonctionné.
## Installation réussie avec python -e .
## Pour l'heure, les tests avec R n'ont pas fonctionné. Je suis en train d'installer RStudio avec Anaconda

``` shell
pip install virtualenv
python3 -m virtualenv -p python3 ~/.venv/TSbench
. ~/.venv/TSbench/bin/activate
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
pip install ipykernel
pip install ipython
ipython kernel install --name "~/.venv/TSbench" --user
```

### R specific prerequisites

1.  Install R

    You can find details here <https://www.r-project.org/>, or install
    from <https://utstat.toronto.edu/cran/>. Make sure that console run
    the command `R` (by settings path correctly).

    Set the R environment variables

    ``` shell
    echo "R_HOME_USER = ${HOME}/.config/R2
    R_LIBS_USER = ${HOME}/.config/R2/packages
    R_PROFILE_USER = ${HOME}/.config/R2/.Rprofile
    R_HISTFILE = ${HOME}/.config/R2/history.Rhistory" > ~/.Renviron
    ```

2.  For Linux

    Install `fortran` (`gfortran` or `gcc-fortran` depending on the
    distribution)

## Installation without R dependencies

Make sure to clone this repository and change current directory to the
TSbench directory.

``` shell
pip install -e ."[noR]"
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
    pip install -e ."[noR]"
    ```

2.  Install `rpy2`

    ``` shell
    pip install rpy["all"]
    ```

3.  Create the `~/.Renviron` file with writable library path. Here is an
    example of such a file :

    ``` shell
    echo "R_HOME_USER = ${HOME}/.config/R
    R_LIBS_USER = ${HOME}/.config/R/packages
    R_PROFILE_USER = ${HOME}/.config/R/.Rprofile
    R_HISTFILE = ${HOME}/.config/R/history.Rhistory" > ~/.Renviron
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
