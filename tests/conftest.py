"""Conftest to skip certain marks by default, unless a flag is specified."""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-all", action="store_true", default=False, help="run all tests"
    )
    parser.addoption("--R", action="store_true", default=False, help="run R tests")
    parser.addoption(
        "--performance",
        action="store_true",
        default=False,
        help="run performance tests",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "R: mark tests to run R")
    config.addinivalue_line("markers", "performance: mark tests to run performance")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-all"):
        # --run--all given in cli: run all
        return

    # List of options to run.
    option_flags = ["--performance", "--R"]
    # Check, for every option flag, if it is specified run it. Otherwise skip it.
    for option in option_flags:
        if not config.getoption(option):
            skip_option = pytest.mark.skip(reason=f"need the {option} option to run")
            for item in items:
                if option[2:] in item.keywords:
                    item.add_marker(skip_option)
