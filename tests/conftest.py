"""Conftest to skip certain marks by default, unless a flag is specified."""

import pytest


OPTIONAL_MARKERS = {
    "--run-R": "R",
    "--run-performance": "performance",
}


def pytest_addoption(parser):
    parser.addoption(
        "--run-all", action="store_true", default=False, help="run all tests"
    )
    parser.addoption(
        "--run-R", action="store_true", default=False, help="run R tests"
    )
    parser.addoption(
        "--run-performance",
        action="store_true",
        default=False,
        help="run performance tests",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "R: mark tests to run R")
    config.addinivalue_line("markers", "performance: mark tests to run performance")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-all"):
        # --run-all given in cli: run all
        return

    # Check, for every option flag, if it is specified run it. Otherwise skip it.
    for option, marker in OPTIONAL_MARKERS.items():
        if not config.getoption(option):
            skip_option = pytest.mark.skip(reason=f"need the {option} option to run")
            for item in items:
                if marker in item.keywords:
                    item.add_marker(skip_option)