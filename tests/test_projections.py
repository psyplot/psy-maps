"""Test the support of different CF-conformal projections"""
import pytest
import os.path as osp
import glob
import cartopy.crs as ccrs
import psyplot.data as psyd
import _base_testing as bt


projection_mapping = {
    'albers_conical_equal_area': ccrs.AlbersEqualArea,
    'azimuthal_equidistant': ccrs.AzimuthalEquidistant,
    'geostationary': ccrs.Geostationary,
    'lambert_azimuthal_equal_area': ccrs.LambertAzimuthalEqualArea,
    'lambert_conformal_conic': ccrs.LambertConformal,
    'lambert_cylindrical_equal_area': ccrs.LambertCylindrical,
    'latitude_longitude': ccrs.PlateCarree,
    'mercator': ccrs.Mercator,
    #'oblique_mercator',  # not available for cartopy
    'orthographic': ccrs.Orthographic,
    'polar_stereographic': ccrs.SouthPolarStereo,
    'rotated_latitude_longitude': ccrs.RotatedPole,
    'sinusoidal': ccrs.Sinusoidal,
    'stereographic': ccrs.Stereographic,
    'transverse_mercator': ccrs.TransverseMercator,
    #'vertical_perspective',  # not available for cartopy
}


@pytest.fixture(params=list(projection_mapping))
def grid(request):
    return request.param


@pytest.fixture
def grid_ds(grid):
    return psyd.open_dataset(osp.join(bt.test_dir, 'grids', grid + '.nc'))


@pytest.fixture
def grid_projection(grid):
    return projection_mapping[grid]


def test_grid_plotting(grid_ds, grid, grid_projection):
    with grid_ds.psy.plot.mapplot() as sp:
        assert len(sp) == 1
        plotter = sp.plotters[0]
        assert isinstance(plotter.transform.projection, grid_projection)
        assert plotter.plot._kwargs.get('transform') is \
            plotter.transform.projection
        assert isinstance(plotter.projection.projection, grid_projection)