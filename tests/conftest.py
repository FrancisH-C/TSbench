# content of conftest.py

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

    skip_R = pytest.mark.skip(reason="need --R option to run")
    skip_performance = pytest.mark.skip(reason="need --performance option to run")

    for item in items:
        if "R" in item.keywords:
            item.add_marker(skip_R)
        if "performance" in item.keywords:
            item.add_marker(skip_performance)
