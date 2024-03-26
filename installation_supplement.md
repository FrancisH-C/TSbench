# Table of Contents <span class="tag" tag-name="TOC_3_gh"><span class="smallcaps">TOC_3_gh</span></span>

- [Dependencies](#dependencies)
  - [Test dependencies](#test-dependencies)
  - [All dependencies](#all-dependencies)
  - [R support](#r-support)
    - [Prerequisites](#prerequisites)
    - [Automatic setup](#automatic-setup)
- [Windows](#windows)
  - [Virtual environment in Python](#virtual-environment-in-python)
    - [Create](#create)
    - [Activate](#activate)
    - [Add to jupyter kernels](#add-to-jupyter-kernels)
  - [Troubleshooting](#troubleshooting)
    - [`PowerShell` `virtualenv` not
      working](#powershell-virtualenv-not-working)
    - [Issue with Locale Settings in Windows (French
      Language)](#issue-with-locale-settings-in-windows-french-language)
- [Linux](#linux)
  - [Virtual environment in Python](#virtual-environment-in-python-1)
    - [Create](#create-1)
    - [Activate](#activate-1)
    - [Add to jupyter kernels](#add-to-jupyter-kernels-1)
  - [More information for R support](#more-information-for-r-support)
    - [Manual setup](#manual-setup)
    - [Remove](#remove)

# Dependencies

The installation has been tested with `Python 3.10.0`.

## Test dependencies

To install `pytest` and `setuptools`:

``` shell
python3 -m pip install .[test]
```

## All dependencies

To install all dependencies:

``` shell
python3 -m pip install .[all]
```

## R support

\`rpy2\` serves as the bridge to use \`R\` in Python, thus TSbench. As
of now, `rpy2` **is not supported on windows**. It may work in the
future or with some specific docker configuration. Alternatively, see
the documentation for how to use outputs from external packages into
TSbench.

`rpy2` serves as the bridge between `Python` and `R`, enabling its use
within TSbench. Please note that, currently, `rpy2` is **not supported
on Windows**. For more details, you can refer to the installation guide
at this link:
<https://rpy2.github.io/doc/latest/html/overview.html#install-installation>.

It may become compatible in the future or with specific `Docker`
configurations. Alternatively, you can refer to the documentation for
instructions on how to incorporate outputs from external packages into
TSbench.

### Prerequisites

- `R` For further details, please visit the official R website:
  <https://www.r-project.org/>.
- `Fortran` Please check your specific distribution's documentation for
  more information. For additional guidance, you can visit the following
  link: <https://gcc.gnu.org/wiki/GFortranDistros>.

Ensure that you can run the commands correctly by settings the `$PATH`
variable and restarting the terminal.

### Automatic setup

``` shell
python3 -m pip install .[R]
python3 setup_R.py
```

# Windows

## Virtual environment in Python

One way to set up a virtual environment in `Python` is by using `conda`.
For detailed instructions, please refer to the official documentation:
<https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>

Please note that the following steps assume you have completed the full
installation of Anaconda. If you haven't installed Anaconda yet, you can
find the installation guide here: Anaconda Installation Guide for
Windows. <https://docs.anaconda.com/free/anaconda/install/windows/>

### Create

``` shell
conda create -n TSbench python=3.10 anaconda
```

### Activate

When working within the virtual environment, you need to activate it in
every terminal session. To activate the virtual environment, run the
following command:

``` shell
conda activate TSbench
```

### Add to jupyter kernels

You can add the environment to `Jupyter` by executing the following
command:

``` shell
python3 -m pip install ipykernel
python3 -m pip install ipython
python3 -m ipykernel install --name TSbench --user
```

## Troubleshooting

### `PowerShell` `virtualenv` not working

Use the following command

``` ps
set-executionpolicy remotesigned
```

This command allows the execution of locally created scripts while
requiring downloaded scripts to be signed by a trusted publisher.

### Issue with Locale Settings in Windows (French Language)

If you are encountering issues related to locale settings while working
on Windows in French, changing the language to English (US) can resolve
the problem, if you encounter a "UTF-8 codec error".

# Linux

## Virtual environment in Python

One way to set up a virtual environment in `Python` is by using
`virutalenv`. For detailed instructions, please refer to the official
documentation: <https://docs.python.org/3/library/venv.html>.

### Create

``` shell
python -m pip install virtualenv
python -m virtualenv $HOME/.venv/TSbench
```

### Activate

When working within the virtual environment, you need to activate it in
every terminal session. To activate the virtual environment, run the
following command:

``` shell
source $HOME/.venv/TSbench/bin/activate
```

### Add to jupyter kernels

You can add the environment to `Jupyter` by executing the following
command:

``` shell
python -m pip install ipykernel
python -m pip install ipython
python -m ipykernel install --name TSbench --user
```

## More information for R support

### Manual setup

If you don't like `setup_R.py` or if you don't use the integration with
rpy2, you can use a custom R setup. Here are presented the steps based
on `Setup_R.py` as a starting point.

If you prefer not to use `Setup_R.py`, you have the option to set a
custom `R` setup. Below, we outline the steps, using ~Setup_R.py as a
reference point to get started.

1.  Create the \$HOME/.Renviron file and ensure it has a writable
    library path:

    ``` shell
	R_HOME_USER=$HOME/.config/R
	R_PROFILE_USER=$HOME/.config/R/
	R_LIBS_USER=$HOME/.local/share/R/library
	R_HISTFILE=$HOME/.local/share/R/history

    mkdir -p $R_LIBS_USER
    mkdir -p $R_PROFILE_USER

    echo "R_HOME_USER = $R_HOME_USER
    R_LIBS_USER = $R_LIBS_USER
    R_PROFILE_USER = $R_PROFILE_USER
    R_HISTFILE = $R_HISTFILE" >> $HOME/.Renviron
    ```
2.  Make verification
	```r
	.libPaths()
	```

	It should output
	```r
	[1] $HOME/.local/share/R/library" "/usr/lib/R/"
	```

3.  Install the necessary R packages using the R environment:

    ``` r
    install.packages("rugarch", repos="https://cloud.r-project.org")
    install.packages("rmgarch", repos="https://cloud.r-project.org")
    install.packages("MTS", repos="https://cloud.r-project.org")
    install.packages("jsonlite", repos="https://cloud.r-project.org")
    ```

4.  If you still have trouble (packages with non-zero return code), maybe you might be missing dependencies.

	For example, for some distribution, you will need to install `TCL/TK` with the package manager.
	
### Remove

To remove `R` integration, simply uninstall `rpy2` using:

``` shell
python -m pip uninstall rpy2
```
