#!/usr/bin/env python

import os
from pathlib import Path
import importlib.util

home = Path.home()
r_environ = os.path.join(home, ".Renviron")
r_home = os.path.join(home, ".config/R/")
profile = os.path.join(home, ".config/R/")
libs = os.path.join(home, ".config/R/packages/")
histfile = os.path.join(home, ".config/R/history/")


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
            f"R_HOME_USER = {r_home}\nR_LIBS_USER = {libs}\nR_PROFILE_USER = {profile}\nR_HISTFILE = {histfile}"
        )


def R_directories():
    os.makedirs(r_home, exist_ok=True)
    os.makedirs(profile, exist_ok=True)
    os.makedirs(libs, exist_ok=True)
    os.makedirs(histfile, exist_ok=True)


if __name__ == "__main__":
    R_directories()
    R_config()
    install_R_package()
