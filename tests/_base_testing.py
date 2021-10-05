import os
import unittest
import sys
import subprocess as spr
from unittest import TestCase
import numpy as np

test_dir = os.path.dirname(__file__)

ref_dir = os.getenv(
    "PSYPLOT_REFERENCES", os.path.join(test_dir, "reference_figures")
)
output_dir = os.getenv(
    "PSYPLOT_TESTFIGURES", os.path.join(test_dir, "test_figures")
)

remove_temp_files = True

# check if the seaborn version is smaller than 0.8 (without actually importing
# it), due to https://github.com/mwaskom/seaborn/issues/966
# If so, disable the import of it when import psyplot.project
try:
    sns_version = spr.check_output(
        [sys.executable, '-c', 'import seaborn; print(seaborn.__version__)'])
except spr.CalledProcessError:  # seaborn is not installed
    sns_version = None
else:
    sns_version = sns_version.decode('utf-8')


class PsyPlotTestCase(TestCase):
    """Base class for testing the psyplot package. It only provides some
    useful methods to compare figures"""

    longMessage = True

    plot_type = None

    grid_type = None

    ncfile = os.path.join(test_dir, 'test-t2m-u-v.nc')

    @classmethod
    def tearDownClass(cls):
        import psyplot
        from psyplot.config.rcsetup import defaultParams
        psyplot.rcParams.update(
            **{key: val[0] for key, val in defaultParams.items()})

    @classmethod
    def create_dirs(cls):
        if not os.path.exists(ref_dir):
            os.makedirs(ref_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        cls.odir = output_dir

    def get_ref_file(self, identifier):
        """
        Gives the name of the reference file for a test

        This staticmethod gives combines the given `plot_type`, `identifier`
        and `grid_type` to form the name of a reference figure

        Parameters
        ----------
        identifier: str
            The unique identifier for the plot (usually the formatoption name)

        Returns
        -------
        str
            The basename of the reference file"""
        identifiers = ['test']
        if self.plot_type is not None:
            identifiers.append(self.plot_type)
        identifiers.append(identifier)
        if self.grid_type is not None:
            identifiers.append(self.grid_type)
        return "_".join(identifiers) + '.png'

    def compare_figures(self, fname, tol=5, **kwargs):
        """Saves and compares the figure to the reference figure with the same
        name"""
        import matplotlib.pyplot as plt
        from matplotlib.testing.compare import compare_images
        plt.savefig(os.path.join(self.odir, fname), **kwargs)
        results = compare_images(
            os.path.join(ref_dir, fname), os.path.join(self.odir, fname),
            tol=tol)
        self.assertIsNone(results, msg=results)

    def assertAlmostArrayEqual(self, actual, desired, rtol=1e-07, atol=0,
                               msg=None, **kwargs):
        """Asserts that the two given arrays are almost the same

        This method uses the :func:`numpy.testing.assert_allclose` function
        to compare the two given arrays.

        Parameters
        ----------
        actual : array_like
            Array obtained.
        desired : array_like
            Array desired.
        rtol : float, optional
            Relative tolerance.
        atol : float, optional
            Absolute tolerance.
        equal_nan : bool, optional.
            If True, NaNs will compare equal.
        err_msg : str, optional
            The error message to be printed in case of failure.
        verbose : bool, optional
            If True, the conflicting values are appended to the error message.
        """
        try:
            np.testing.assert_allclose(actual, desired, rtol=rtol, atol=atol,
                                       err_msg=msg or '', **kwargs)
        except AssertionError as e:
            self.fail(e.message)


class TestBase2D:
    """Test :class:`psyplot.plotter.baseplotter.BasePlotter` class without time
    and vertical dimension"""

    def _label_test(self, key, label_func, has_time=False):
        return super(TestBase2D, self)._label_test(
            key, label_func, has_time=has_time)

    def __getattribute__(self, attr):
        """Hack to disable the creation of reference figures"""
        ret = super().__getattribute__(attr)
        if callable(ret) and attr.startswith("ref_"):

            @unittest.skip("Reference figures are created from the base class")
            def skip_func(self):
                pass

            skip_func.__name__ = attr
            return skip_func()
        return ret
