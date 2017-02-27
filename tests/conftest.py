
def pytest_addoption(parser):
    group = parser.getgroup("psyplot", "psyplot specific options")
    group.addoption('--no-removal', help='Do not remove created test files',
                    action='store_true')
    group.addoption(
        '--ref', help='Create reference figures instead of running the tests',
        action='store_true')


def pytest_configure(config):
    if config.getoption('no_removal'):
        import _base_testing
        _base_testing.remove_temp_files = False
    if config.getoption('ref'):
        import unittest
        unittest.TestLoader.testMethodPrefix = 'ref'
