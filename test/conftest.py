# content of conftest.py

import pytest


def pytest_addoption(parser):
    parser.addoption("--R", action="store_true", default=False, help="run R tests")


def pytest_configure(config):
    config.addinivalue_line("markers", "R: mark test as R to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--R"):
        # --R given in cli: do not skip R tests
        return
    skip_R = pytest.mark.skip(reason="need --R option to run")
    for item in items:
        if "R" in item.keywords:
            item.add_marker(skip_R)
