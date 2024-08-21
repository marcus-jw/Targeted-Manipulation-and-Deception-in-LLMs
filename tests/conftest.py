import os

import pytest


def pytest_runtest_setup(item):
    if "local_only" in item.keywords and "GITHUB_ACTIONS" in os.environ:
        pytest.skip("Skipping local-only test")


def pytest_addoption(parser):
    parser.addoption("--gpus", action="store", default=None, help="A custom option to be used in tests")


@pytest.fixture
def gpus(request):
    return request.config.getoption("--gpus")
