<!-- markdown-toc start -->
# Table of Contents

- [Description](#description)
    - [Demonstration](#demonstration)
- [Quick Start](#quick-start)
- [Installation Information](#installation-information)
    - [Virtual Environment in Python](#virtual-environment-in-python)
        - [Windows](#windows)
        - [Linux](#linux)
    - [TSbench and R](#tsbench-and-r)
        - [Limitation](#limitation)
        - [Install R](#install-r)
        - [Install rpy2](#install-rpy2)
        - [CRAN packages](#cran-packages)
        - [Remove R from TSbench](#remove-r-from-tsbench)
    - [Installation Options](#installation-options)
    - [Troubleshooting for Windows](#troubleshooting-for-windows)
        - [`PowerShell` `virtualenv` not working](#powershell-virtualenv-not-working)
        - [Issue with Locale Settings in Windows (French Language)](#issue-with-locale-settings-in-windows-french-language)
- [Run Tests](#run-tests)
- [```](#)

<!-- markdown-toc end -->

# Description

Alpha version. Usability updates coming soon.
khe purpose of this package is to define experiments for benchmarking
models on a specified dataset by calculating various metrics. It
includes the `Model` class, and any subclasses that define the
appropriate methods can be utilized in experiments.


## Demonstration

Watch the accompanying video for a comprehensive presentation of the
package in action.

[![Presentation](https://img.youtube.com/vi/s0gMqWn-nXo/0.jpg)](https://www.youtube.com/watch?v=s0gMqWn-nXo)

# Quick Start

To perform the basic installation, navigate to the TSbench directory
using the command line and execute the following command:

``` shell
python -m pip install .
```

Refer to the [next section](#installation-information) for detailed
information on:
- [Virtual Environment in Python](#virtual-environment-in-python)
- [TSbench and R](#tsbench-and-r)
- [Installation Options](#installation-options)
- [Troubleshooting for Windows](#troubleshooting-for-windows)

# Installation Information
The installation has been tested with `Python 3.10.0`.
## Virtual Environment in Python
It is strongly recommended to use a virtual environment for your Python
installation. Some installation errors may occur otherwise due to path localization.

### Windows

One way to set up a virtual environment in `Python` is by using `conda`.
For detailed instructions, please refer to the official documentation:
<https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>
[fortran](https://gcc.gnu.org/wiki/GFortranDistros)

Please note that the following steps assume you have completed the full
installation of Anaconda. If you haven't installed Anaconda yet, you can
find the installation guide here: Anaconda Installation Guide for
Windows. <https://docs.anaconda.com/free/anaconda/install/windows/>

#### Create

``` shell
conda create -n TSbench python=3.10 anaconda
```

#### Activate

When working within the virtual environment, you need to activate it in
every terminal session. To activate the virtual environment, run the
following command:

``` shell
conda activate TSbench
```

#### Add to jupyter kernels

You can add the environment to `Jupyter` by executing the following
command:

``` shell
python3 -m pip install ipykernel
python3 -m pip install ipython
python3 -m ipykernel install --name TSbench --user
```

### Linux

One way to set up a virtual environment in `Python` is by using
`virtualenv`. For detailed instructions, please refer to the official
documentation: <https://docs.python.org/3/library/venv.html>.

#### Create

``` shell
python -m pip install virtualenv
python -m virtualenv $HOME/.venv/TSbench
```

#### Activate

When working within the virtual environment, you need to activate it in
every terminal session. To activate the virtual environment, run the
following command:

``` shell
source $HOME/.venv/TSbench/bin/activate
```

#### Add to jupyter kernels

You can add the environment to `Jupyter` by executing the following
command:

``` shell
python -m pip install ipykernel
python -m pip install ipython
python -m ipykernel install --name TSbench --user
```


## TSbench and R

### Limitation

TSbench and R is supported only on Linux. Moreover, installing `R` and
CRAN packages requires root previleges.

### Install R

For information on how to install `R` and its packages please visit the official R website:
<https://www.r-project.org/>. Ensure that you can run the `R` commands correctly by settings the `$PATH`
variable and restarting the terminal

### Install rpy2

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

### CRAN packages

Ensure that you can load the list of required packages
```
rugarch
rmgarch
mts
jsonlite
```

For installation, follow the instruction on [the official R
website](https://www.r-project.org/). The packages require system
dependencies handled by the package manager. Here are two common examples from
the documentation on how to install CRAN pacakges.

#### Ubuntu

Install using the pacakge manager as detailed https://cloud.r-project.org/bin/linux/ubuntu/

``shell
sudo add-apt-repository ppa:c2d4u.team/c2d4u4.0+
sudo apt install --no-install-recommends r-cran-rugarch r-cran-rmgarch r-cran-mts r-cran-jsonlite
``

``` shell
python3 -m pip install .[R]
```

#### Archlinux

###### Option 1. Using the AUR install the following packages
```
r-rugarch
r-rmgarch
r-mts
r-jsonlite
```

###### Option 2. The Python setup
1. Install dependencies
	```shell
	sudo pacman -S gcc-fortran tcl tk
	```

2. This step is optional and not recommended for a first attempt.
	Change the default R directories to avoid prompts during installation
	```python
	from setup_R import R_config, R_directories
	R_config()
	R_directories()
	```

3. Automatic setup
	``` shell
	python3 -m pip install .[R]
	python3 scripts/archlinux/setup_R.py
	```

#### Manual setup

For advanced users.

1. Using your package manager, install `r` and `r-dev` and all the dependencies for all the required packages.
   This depend of the Linux distribution.

2. This step is optional and not recommended for a first attempt. Create the \$HOME/.Renviron file and ensure it has a existent writable
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

3. Make verification
	```r
	.libPaths()
	```

	It should output
	```r
	[1] $HOME/.local/share/R/library" "/usr/lib/R/"
	```

4.  Install the necessary R packages using the R environment:

   ``` r
   install.packages("rugarch", repos="https://cloud.r-project.org")
   install.packages("rmgarch", repos="https://cloud.r-project.org")
   install.packages("MTS", repos="https://cloud.r-project.org")
   install.packages("jsonlite", repos="https://cloud.r-project.org")
   ```

If you have installation errors (packages with non-zero return code), you are probably missing dependencies.
You can search for the pacakage specific dependencies.

### Remove R from TSbench

To remove `R` integration, simply uninstall `rpy2` using:
``` shell
python -m pip uninstall rpy2
```

## Installation Options

- To install with `test` dependencies:
	``` shell
	python3 -m pip install .[test]
	```

- To install with `R` dependencies:
	``` shell
	python3 -m pip install .[R]
	```

- To install all dependencies:
	``` shell
	python3 -m pip install .[all]
	```


## Troubleshooting for Windows
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


# Run Tests

To run the tests
``` shell
python3 -m pytest -x -s
```

To run the tests with R
``` shell
python3 -m pytest --R
```
=======
