#!/usr/env python
"""
Install required R packages assuming the dependencies are met.
"""

import os
from pathlib import Path
import importlib.util

home = Path.home()
r_environ = os.path.join(home, ".Renviron")
r_home = os.path.join(home, ".config/R/")
profile = os.path.join(home, ".config/R/.Rprofile")
libs = os.path.join(home, ".local/share/R/library/")
histfile = os.path.join(home, ".local/share/R/history/")


def install_R_package():
    if importlib.util.find_spec("rpy2") is not None:
        from rpy2.robjects.packages import importr
    else:
        print("R packages not installed")
        exit()
    utils = importr("utils")
    utils.install_packages("jsonlite", repos="https://cloud.r-project.org")
    utils.install_packages("rugarch", repos="https://cloud.r-project.org")
    utils.install_packages("rmgarch", repos="https://cloud.r-project.org")
    utils.install_packages("MTS", repos="https://cloud.r-project.org")


def R_config():
    with open(r_environ, "a") as f:
        f.write(
            f"R_HOME_USER = {r_home}\n"
            + f"R_LIBS_USER = {libs}\n"
            + f"R_PROFILE_USER = {profile}\n"
            + f"R_HISTFILE = {histfile}"
        )


def R_directories():
    os.makedirs(r_home, exist_ok=True)
    os.makedirs(libs, exist_ok=True)


if __name__ == "__main__":
    install_R_package()
