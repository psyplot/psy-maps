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
def open_grid_ds():
    open_datasets = []
    def _grid_ds(grid):
        ds = psyd.open_dataset(osp.join(bt.test_dir, 'grids', grid + '.nc'))
        open_datasets.append(ds)
        return ds
    yield _grid_ds
    for ds in open_datasets:
        ds.close()


@pytest.fixture
def grid_projection(grid):
    return projection_mapping[grid]


def test_grid_plotting(open_grid_ds, grid, grid_projection):
    grid_ds = open_grid_ds(grid)
    with grid_ds.psy.plot.mapplot() as sp:
        assert len(sp) == 1
        plotter = sp.plotters[0]
        assert isinstance(plotter.transform.projection, grid_projection)
        assert plotter.plot._kwargs.get('transform') is \
            plotter.transform.projection
        assert isinstance(plotter.projection.projection, grid_projection)


@pytest.mark.parametrize(
    "grid,clon", [("rotated_latitude_longitude", 11),
                  (osp.join("..", "test-t2m-u-v"), 0)])
def test_clon_centering(open_grid_ds, grid, clon):
    pytest.importorskip("cartopy", minversion="0.18")
    grid_ds = open_grid_ds(grid)
    with grid_ds.psy.plot.mapplot(projection='ortho') as sp:
        plotter = sp.plotters[0]
        assert plotter.clon.clon == pytest.approx(clon, abs=1)

@pytest.mark.parametrize(
    "grid,clat", [("rotated_latitude_longitude", 51),
                  (osp.join("..", "test-t2m-u-v"), 0)])
def test_clat_centering(open_grid_ds, grid, clat):
    pytest.importorskip("cartopy", minversion="0.18")
    grid_ds = open_grid_ds(grid)
    with grid_ds.psy.plot.mapplot(projection='ortho') as sp:
        plotter = sp.plotters[0]
        assert plotter.clat.clat == pytest.approx(clat, abs=1)


def test_rotated_pole_transform(open_grid_ds):
    """Test if the lon coordinate is correctly interpreted as PlateCarree

    See https://github.com/psyplot/psy-maps/issues/9"""
    grid_ds = open_grid_ds('rotated_latitude_longitude')
    with grid_ds.psy.plot.mapplot(decoder=dict(x={'lon'}, y={'lat'})) as sp:
        plotter = sp.plotters[0]
        assert isinstance(plotter.transform.projection, ccrs.PlateCarree)
        assert isinstance(plotter.ax.projection, ccrs.RotatedPole)


def test_rotated_pole_extent(open_grid_ds):
    grid_ds = open_grid_ds('rotated_latitude_longitude-australasia')
    with grid_ds.psy.plot.mapplot(name='t2m') as sp:
        plotter = sp.plotters[0]
        assert isinstance(plotter.ax.projection, ccrs.RotatedPole)
        lonmin, lonmax = plotter.ax.get_extent()[:2]
        assert lonmax - lonmin < 200
