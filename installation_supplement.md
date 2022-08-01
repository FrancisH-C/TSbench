---
author: Francis Huot-Chantal
---

# Manual installation on Linux

1.  Install the package and its dependencies

    ``` shell
    python -m pip install -e ."[all]"
    ```

2.  Change the default `R` configuration

    Create the `$HOME/.Renviron` file with writable library path. Here
    is an example

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

3.  Install `R` packages

    -   Using R

        ``` r
        install.packages("rugarch", repos="https://cloud.r-project.org")
        install.packages("rmgarch", repos="https://cloud.r-project.org")
        install.packages("MTS", repos="https://cloud.r-project.org")
        ```

    -   Or, Using rpy2

        ``` python
        from rpy2.robjects.packages import importr
        utils = importr("utils")
        utils.install_packages("rugarch", repos="https://cloud.r-project.org")
        utils.install_packages("rmgarch", repos="https://cloud.r-project.org")
        utils.install_packages("MTS", repos="https://cloud.r-project.org")
        ```

# Windows information for R

Install from <https://cran.r-project.org/bin/windows/base/>, you can
find details here <https://www.r-project.org/>.
