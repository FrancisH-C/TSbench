# Table of Contents [[TOC_3_gh]{.smallcaps}]{.tag tag-name="TOC_3_gh"} {#table-of-contents}

-   [Dependencies](#dependencies)
    -   [Support for R](#support-for-r)
    -   [Test dependencies](#test-dependencies)
    -   [All dependencies](#all-dependencies)
-   [Windows](#windows)
    -   [Virtual environment in Python](#virtual-environment-in-python)
        -   [Create](#create)
        -   [Activate](#activate)
        -   [Add to jupyter kernels](#add-to-jupyter-kernels)
    -   [Troubleshooting](#troubleshooting)
        -   [`PowerShell` `virtualenv` not
            working](#powershell-virtualenv-not-working)
        -   [Locale settings](#locale-settings)
-   [Linux](#linux)
    -   [Virtual environment in
        Python](#virtual-environment-in-python-1)
        -   [Create](#create-1)
        -   [Activate](#activate-1)
        -   [Add to jupyter kernels](#add-to-jupyter-kernels-1)
    -   [R integration](#r-integration)
        -   [Automatic setup](#automatic-setup)
        -   [Remove](#remove)
        -   [Manual setup](#manual-setup)

# Dependencies

The installation has been tested with `Python 3.10.0`.

## Support for R

Install R from <https://cran.r-project.org/bin/windows/base/>, you can
find details here <https://www.r-project.org/>.

As of now, `rpy2` **is not supported on windows**.
<https://rpy2.github.io/doc/latest/html/overview.html#install-installation>
It may work in the future or with some specific docker configuration.
See the documentation for how to use outputs from external packages into
TSbench.

To installs `rpy2` (support for R in Python)

``` shell
pip install .[R]
```

## Test dependencies

To install `pytest` and `setuptools`.

``` shell
pip install .[test]
```

## All dependencies

To install all of the above.

``` shell
pip install .[all]
```

# Windows

## Virtual environment in Python

One way to setup a viratual envrionement in Python is to use `conda`.
Here is a simple example, for details, see the official documentation
<https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>

The following assumes the complete installation of Anaconda
<https://docs.anaconda.com/free/anaconda/install/windows/>

### Create

``` shell
conda create -n TSbench python=3.10 anaconda
```

### Activate

On every terminal, if you want to work within the virtual environment
you must first activate it. To activate the virtual environment you run
the following

``` shell
conda activate TSbench
```

### Add to jupyter kernels

You can add the environment to `Jupyter` with the following:

``` shell
python -m pip install ipykernel
python -m pip install ipython
python -m ipykernel install --name TSbench --user
```

## Troubleshooting

### `PowerShell` `virtualenv` not working

Set the following

``` ps
set-executionpolicy remotesigned
```

### Locale settings

There might be an issue with the locale when working on Windows in
French. Changing the language to English (US) will solve the issue if
you encounter `UTF-8 codec error`{.verbatim}.

# Linux

## Virtual environment in Python

One way to setup a viratual envrionement in Python is to use
`virtualenv`{.verbatim}. Here is a simple example, for details, see the
official documentation <https://docs.python.org/3/library/venv.html>.

### Create

``` shell
python -m pip install virtualenv
python -m virtualenv -p python3 $HOME/.venv/TSbench
```

### Activate

On every terminal, if you want to work within the virtual environment
you must first activate it. To activate the virtual environment you run
the following :

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

## R integration

### Automatic setup

If the post-installation setup for `R`{.verbatim} doesn\'t work, make
sure the prerequisites are met. Make sure every command can be run by
the terminal everywhere (by setting the `$PATH`{.verbatim} and
restarting the terminal).

### Remove

If you want to remove `R`{.verbatim} integration simply remove rpy2
using

``` shell
python -m pip uninstall rpy2
```

### Manual setup

If you don\'t like `setup_R.py`{.verbatim} or if you don\'t use the
integration with rpy2, you can use a custom R setup. Here are presented
the steps based on `Setup_R.py`{.verbatim} as a starting point.

1.  Create the `$HOME/.Renviron`{.verbatim} file with writable library
    path.

    ``` shell
    R_HOME_USER=$HOME/.config/R
    R_LIBS_USER=$HOME/.config/R/packages
    R_PROFILE_USER=$HOME/.config/R/
    R_HISTFILE=$HOME/.config/R/history.Rhistory

    mkdir -p $R_LIBS_USER
    mkdir -p $R_PROFILE_USER

    echo "R_HOME_USER = $R_HOME_USER
    R_LIBS_USER = $R_LIBS_USER
    R_PROFILE_USER = $R_PROFILE_USER
    R_HISTFILE = $R_HISTFILE" >> $HOME/.Renviron
    ```

2.  Install `R` packages using `R`

    ``` {.r org-language="R"}
    install.packages("rugarch", repos="https://cloud.r-project.org")
    install.packages("rmgarch", repos="https://cloud.r-project.org")
    install.packages("MTS", repos="https://cloud.r-project.org")
    ```
