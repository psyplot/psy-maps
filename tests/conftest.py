"""pytest configuration file for psy-maps."""

import os.path as osp

import pytest

# SPDX-FileCopyrightText: 2016-2024 University of Lausanne
# SPDX-FileCopyrightText: 2020-2021 Helmholtz-Zentrum Geesthacht
# SPDX-FileCopyrightText: 2021-2024 Helmholtz-Zentrum hereon GmbH
#
# SPDX-License-Identifier: LGPL-3.0-only


try:
    # make sure we import QtWebEngineWidgets at the start
    import psyplot_gui.compat.qtcompat  # noqa: F401
except ImportError:
    pass


def pytest_addoption(parser):
    group = parser.getgroup("psyplot", "psyplot specific options")
    group.addoption(
        "--ref",
        help="Create reference figures instead of running the tests",
        action="store_true",
    )


def pytest_configure(config):
    if config.getoption("ref"):
        import unittest

        unittest.TestLoader.testMethodPrefix = "ref"


@pytest.fixture
def regular_test_file():
    return osp.join(osp.dirname(__file__), "test-t2m-u-v.nc")
