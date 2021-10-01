try:
    # make sure we import QtWebEngineWidgets at the start
    import psyplot_gui.compat.qtcompat
except ImportError:
    pass

def pytest_addoption(parser):
    group = parser.getgroup("psyplot", "psyplot specific options")
    group.addoption(
        '--ref', help='Create reference figures instead of running the tests',
        action='store_true')


def pytest_configure(config):
    if config.getoption('ref'):
        import unittest
        unittest.TestLoader.testMethodPrefix = 'ref'
