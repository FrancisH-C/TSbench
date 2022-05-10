#!/usr/bin/env python

def install_R_package():
    try:
        from rpy2.robjects.packages import importr
        utils = importr("utils")
        utils.install_packages("rugarch", repos="https://cloud.r-project.org")
        utils.install_packages("rmgarch", repos="https://cloud.r-project.org")
        utils.install_packages("MTS", repos="https://cloud.r-project.org")
    except Exception:
        print("R packages not installed")


if __name__ == "__main__":
    install_R_package()
