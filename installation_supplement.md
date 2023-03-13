---
---

# Windows

## virtualenv

Here is an example using `virtualenv`. For details, see the official
documentation <https://docs.python.org/3/library/venv.html>.

### Create

``` shell
python -m pip install virtualenv
python -m virtualenv TSbench
```

### Activate

On every shell if you want to work within the virtual environment you
must first activate it. To activate the virtual environment you run the
following :

- PowerShell

  1.  Start Windows PowerShell with the "Run as Administrator"

  2.  set-executionpolicy remotesigned

  virtualenv –python C:.exe venv C:\\ \<venv\>.bat

  ``` ps
  TSbench\Scripts\Activate.ps1
  ```

- cmd.exe

      TSbench\Scripts\activate.bat

### Add to jupyter kernels

You can add the environment to Jupyter with the following:

``` shell
python -m pip install ipykernel
python -m pip install ipython
python -m ipykernel install --name TSbench --user
```

## Information for R

Install R from <https://cran.r-project.org/bin/windows/base/>, you can
find details here <https://www.r-project.org/>.

As of now, `rpy2` **is not supported on windows**.
<https://rpy2.github.io/doc/latest/html/overview.html#install-installation>
It may work in the future or some specific docker configuration.

## Locale settings

There might be an issue with the locale when working on Windows in
French. Changing the language to English (US) will solve the issue if
you encounter `UTF-8 codec error`.

# Linux

## virtualenv

Here is an example using `virtualenv`. For details, see the official
documentation <https://docs.python.org/3/library/venv.html>.

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

## R integration

### Automatic setup

If the post-installation setup for `R` doesn't work, make sure the
prerequisites are met. Make sure every command can be run by the
terminal everywhere (by setting the `$PATH`).

### Remove

If you want to remove `R` integration, you can re-install or simply
remove rpy2 using

``` shell
python -m pip uninstall rpy2
```

### Manual setup

If you don't like `setup_R.py` or if you don't use the integration with
rpy2, you can use a custom R setup.

1.  Create the `$HOME/.Renviron` file with writable library path. Here
    is an example used in `Setup_R.py`

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

2.  Install `R` packages using R

    ``` r
    install.packages("rugarch", repos="https://cloud.r-project.org")
    install.packages("rmgarch", repos="https://cloud.r-project.org")
    install.packages("MTS", repos="https://cloud.r-project.org")
    ```
