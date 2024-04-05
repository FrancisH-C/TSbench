#!/usr/bin/env python

import argparse
import subprocess
import sys

from scripts import setup_R


def installation_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="""
    Installation script:
    1. Install the package and its dependencies
    2. Change the default R configuration (with R flag)
    3. Install R packages (with R flag)
    - Generate data from defined simulation models
    - Split data according to a pattern
    - Train a list of model on given data
    - Make comparative measurements of trained models on given data""",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Global parameters
    parser.add_argument(
        "--R",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="""Install with R dependency (Default) or not.""",
    )
    return parser


if __name__ == "__main__":
    parser = installation_parser()
    args = parser.parse_args()
    if args.R:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", ".[all]"])
        setup_R.R_directories()
        setup_R.R_config()
        setup_R.install_R_package()
    else:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", ".[noR]"])
