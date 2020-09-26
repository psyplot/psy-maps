import six
import re
import warnings
from abc import abstractproperty
from difflib import get_close_matches
import copy
from itertools import starmap, chain, repeat
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cf
from cartopy.mpl.gridliner import Gridliner
import matplotlib as mpl
import matplotlib.ticker as ticker
from matplotlib.colors import BoundaryNorm
import numpy as np
from psyplot import rcParams
from psyplot.compat.pycompat import map
from psyplot.docstring import docstrings
from psyplot.data import InteractiveList, _infer_interval_breaks
from psyplot.plotter import (
    Formatoption, START, DictFormatoption, END, BEFOREPLOTTING)
import psy_simple.plotters as psyps
import psy_simple.base
from psy_maps.boxes import lonlatboxes
from psy_simple.colors import FixedBoundaryNorm


wrap_proj_types = (ccrs._RectangularProjection,
                   ccrs._WarpedRectangularProjection,
                   ccrs.InterruptedGoodeHomolosine,
                   ccrs.Mercator)


@docstrings.get_sections(base='shiftdata')
def shiftdata(lonsin, datain, lon_0):
    """
    Shift longitudes (and optionally data) so that they match map projection
    region.
    Only valid for cylindrical/pseudo-cylindrical global projections and data
    on regular lat/lon grids. longitudes and data can be 1-d or 2-d, if 2-d
    it is assumed longitudes are 2nd (rightmost) dimension.

    Parameters
    ----------
    lonsin
        original 1-d or 2-d longitudes.
    datain
        original 1-d or 2-d data
    lon_0
        center of map projection region

    References
    ----------
    This function is copied and taken from the
    :class:`mpl_toolkits.basemap.Basemap` class. The only difference is that
    we do not mask values outside the map projection region
    """
    lonsin = np.asarray(lonsin)
    if lonsin.ndim not in [1, 2]:
        raise ValueError('1-d or 2-d longitudes required')
    if datain is not None:
        # if it's a masked array, leave it alone.
        if not np.ma.isMA(datain):
            datain = np.asarray(datain)
        if datain.ndim not in [1, 2]:
            raise ValueError('1-d or 2-d data required')
    # 2-d data.
    if lonsin.ndim == 2:
        raise NotImplementedError(
            "Shifting of 2D-data is currently not supported")
    # 1-d data.
    elif lonsin.ndim == 1:
        lonsin = np.where(lonsin > lon_0+180, lonsin-360, lonsin)
        lonsin = np.where(lonsin < lon_0-180, lonsin+360, lonsin)
        londiff = np.abs(lonsin[0:-1]-lonsin[1:])
        londiff_sort = np.sort(londiff)
        thresh = 360.-londiff_sort[-2]
        itemindex = len(lonsin) - np.where(londiff >= thresh)[0]
        if itemindex:
            # check to see if cyclic (wraparound) point included
            # if so, remove it.
            if np.abs(lonsin[0]-lonsin[-1]) < 1.e-4:
                hascyclic = True
                lonsin_save = lonsin.copy()
                lonsin = lonsin[1:]
                if datain is not None:
                    datain_save = datain.copy()
                    datain = datain[1:]
            else:
                hascyclic = False
            lonsin = np.roll(lonsin, itemindex-1)
            if datain is not None:
                datain = np.roll(datain, itemindex-1)
            # add cyclic point back at beginning.
            if hascyclic:
                lonsin_save[1:] = lonsin
                lonsin_save[0] = lonsin[-1]-360.
                lonsin = lonsin_save
                if datain is not None:
                    datain_save[1:] = datain
                    datain_save[0] = datain[-1]
                    datain = datain_save
    if datain is not None:
        return lonsin, datain
    else:
        return lonsin


def degree_format():
    if mpl.rcParams['text.usetex']:
        return r'${%g\/^{\circ}\/%s}$'
    else:
        return u'%g\N{DEGREE SIGN}%s'


def format_lons(x, pos):
    fmt_string = degree_format()
    x = x - 360 if x > 180 else x
    if x == 0:
        return fmt_string % (x, '')
    return fmt_string % (abs(x), 'W' if x < 0 else 'E')


def format_lats(x, pos):
    fmt_string = degree_format()
    if x == 0:
        return fmt_string % (abs(x), '')
    return fmt_string % (abs(x), 'S' if x < 0 else 'N')


class Transpose(Formatoption):
    """Transpose the data before plotting

    This formatoption can be used in case the data order of the variable
    dimensions is (x, y) instead of (y, x)"""

    priority = START

    group = 'axes'

    name = 'Transpose (switch) x- and y-dimensions'

    def update(self, value):
        for i, (raw, data) in enumerate(zip(
                self.iter_raw_data, self.iter_data)):
            base_var = next(raw.psy.iter_base_variables)
            is_unstructured = raw.psy.decoder.is_unstructured(base_var)
            if is_unstructured and value:
                decoder = copy.copy(data.psy.decoder)
                xcoord = data.psy.get_coord('x').name
                ycoord = data.psy.get_coord('y').name
                decoder.x.add(ycoord)
                decoder.y.add(xcoord)
                self.set_decoder(decoder, i)
            elif is_unstructured:
                self.set_decoder(raw.psy.decoder, i)
            elif value:
                if data.ndim > 2:
                    new_dims = data.dims[:-2] + data.dims[-2:][::-1]
                else:
                    new_dims = data.dims[::-1]
                self.set_data(data.transpose(*new_dims), i)


lon_formatter = ticker.FuncFormatter(format_lons)
lat_formatter = ticker.FuncFormatter(format_lats)


@docstrings.get_sections(base=
    'ProjectionBase', sections=['Possible types', 'See Also'])
class ProjectionBase(Formatoption):
    """
    Base class for formatoptions that uses cartopy.crs.CRS instances

    Possible types
    --------------
    cartopy.crs.CRS
        A cartopy projection instance (e.g. :class:`cartopy.crs.PlateCarree`)
    str
        A string specifies the projection instance to use. The centered
        longitude and latitude are determined by the :attr:`clon` and
        :attr:`clat` formatoptions.
        Possible strings are (each standing for the specified projection)

        =========== =======================================
        cf          try to decode the CF-conventions
        cyl         :class:`cartopy.crs.PlateCarree`
        robin       :class:`cartopy.crs.Robinson`
        moll        :class:`cartopy.crs.Mollweide`
        geo         :class:`cartopy.crs.Geostationary`
        northpole   :class:`cartopy.crs.NorthPolarStereo`
        southpole   :class:`cartopy.crs.SouthPolarStereo`
        ortho       :class:`cartopy.crs.Orthographic`
        stereo      :class:`cartopy.crs.Stereographic`
        near        :class:`cartopy.crs.NearsidePerspective`
        rotated     :class:`cartopy.crs.RotatedPole`
        =========== =======================================

        The special case ``'cf'`` tries to decode the CF-conventions in the
        data. If this is not possible, we assume a standard lat-lon projection
        (``'cyl'``)

    See Also
    --------
    `Grid-mappings of cf-conventions <http://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#appendix-grid-mappings>`__
    """

    projections = {
        'cyl': ccrs.PlateCarree,
        'robin': ccrs.Robinson,
        'moll': ccrs.Mollweide,
        'geo': ccrs.Geostationary,
        'northpole': ccrs.NorthPolarStereo,
        'southpole': ccrs.SouthPolarStereo,
        'ortho': ccrs.Orthographic,
        'stereo': ccrs.Stereographic,
        'near': ccrs.NearsidePerspective,
        'rotated': ccrs.RotatedPole,
        }

    projection_kwargs = dict(
        chain(zip(projections.keys(), repeat(['central_longitude']))))
    projection_kwargs['ortho'] = ['central_longitude', 'central_latitude']
    projection_kwargs['stereo'] = ['central_longitude', 'central_latitude']
    projection_kwargs['near'] = ['central_longitude', 'central_latitude']
    projection_kwargs['rotated'] = ['pole_longitude', 'pole_latitude']

    # CRS that are supported to be interpreted by CF-conventions
    supported_crs = [
        'albers_conical_equal_area',
        'azimuthal_equidistant',
        'geostationary',
        'lambert_azimuthal_equal_area',
        'lambert_conformal_conic',
        'lambert_cylindrical_equal_area',
        'latitude_longitude',
        'mercator',
        #'oblique_mercator',  # not available for cartopy
        'orthographic',
        'polar_stereographic',
        'rotated_latitude_longitude',
        'sinusoidal',
        'stereographic',
        'transverse_mercator',
        #'vertical_perspective',  # not available for cartopy
        ]

    @property
    def cf_projection(self):
        data = next(self.iter_data)
        if 'grid_mapping' in data.attrs:
            try:
                crs = data[data.attrs['grid_mapping']]
            except KeyError:
                try:
                    crs = data.psy.base[data.attrs['grid_mapping']]
                except KeyError:
                    warnings.warn(
                        "Grid mapping variable %(grid_mapping)s specified but "
                        "impossible to find!" % data.attrs)
                    return
            if 'grid_mapping_name' not in crs.attrs:
                warnings.warn(
                    "Grid mapping variable %(grid_mapping)s specified but "
                    "without 'grid_mapping_name' attribute!" % data.attrs)
                return
            elif crs.attrs['grid_mapping_name'] not in self.supported_crs:
                warnings.warn(
                    "Grid mapping %(grid_mapping_name)s not supported!" %
                    crs.attrs)
                return
            else:
                func = getattr(
                    self, crs.attrs['grid_mapping_name'] + '_from_cf')
                try:
                    return func(crs)
                except Exception:
                    self.logger.debug(
                        "Failed to get CRS from crs variable %s",
                        crs.name, exc_info=True)
                    warnings.warn(
                        f"Failed to get CRS from crs variable {crs.name}")

    def albers_conical_equal_area_from_cf(self, crs):
        kwargs = dict(
            central_longitude=crs.longitude_of_central_meridian,
            central_latitude=crs.latitude_of_projection_origin,
            standard_parallels=crs.standard_parallel
            )
        try:
            iter(kwargs['standard_parallels'])
        except TypeError:
            kwargs['standard_parallels'] = [kwargs['standard_parallels']]
        if getattr(crs, 'false_easting'):
            kwargs['false_easting'] = crs.false_easting
        if getattr(crs, 'false_northing'):
            kwargs['false_northing'] = crs.false_northing
        return ccrs.AlbersEqualArea(**kwargs)

    def azimuthal_equidistant_from_cf(self, crs):
        kwargs = dict(
            central_longitude=crs.longitude_of_projection_origin,
            central_latitude=crs.latitude_of_projection_origin,
            )
        if getattr(crs, 'false_easting'):
            kwargs['false_easting'] = crs.false_easting
        if getattr(crs, 'false_northing'):
            kwargs['false_northing'] = crs.false_northing
        return ccrs.AzimuthalEquidistant(**kwargs)

    def geostationary_from_cf(self, crs):
        kwargs = dict(
            central_longitude=crs.longitude_of_projection_origin,
            satellite_height=crs.perspective_point_height,
            sweep_axis=crs.sweep_angle_axis,
            )
        if getattr(crs, 'false_easting'):
            kwargs['false_easting'] = crs.false_easting
        if getattr(crs, 'false_northing'):
            kwargs['false_northing'] = crs.false_northing
        return ccrs.Geostationary(**kwargs)

    def lambert_azimuthal_equal_area_from_cf(self, crs):
        kwargs = dict(
            central_longitude=crs.longitude_of_projection_origin,
            central_latitude=crs.latitude_of_projection_origin,
            )
        if getattr(crs, 'false_easting'):
            kwargs['false_easting'] = crs.false_easting
        if getattr(crs, 'false_northing'):
            kwargs['false_northing'] = crs.false_northing
        return ccrs.LambertAzimuthalEqualArea(**kwargs)

    def lambert_conformal_conic_from_cf(self, crs):
        kwargs = dict(
            central_longitude=crs.longitude_of_central_meridian,
            central_latitude=crs.latitude_of_projection_origin,
            standard_parallels=crs.standard_parallel,
            )
        try:
            iter(kwargs['standard_parallels'])
        except TypeError:
            kwargs['standard_parallels'] = [kwargs['standard_parallels']]
        if getattr(crs, 'false_easting'):
            kwargs['false_easting'] = crs.false_easting
        if getattr(crs, 'false_northing'):
            kwargs['false_northing'] = crs.false_northing
        return ccrs.LambertConformal(**kwargs)

    def lambert_cylindrical_equal_area_from_cf(self, crs):
        kwargs = dict(
            central_longitude=crs.longitude_of_central_meridian
            )
        return ccrs.LambertCylindrical(**kwargs)

    def latitude_longitude_from_cf(self, crs):
        return ccrs.PlateCarree()

    def mercator_from_cf(self, crs):
        kwargs = dict(
            central_longitude=crs.longitude_of_projection_origin,
            )
        if hasattr(crs, 'scale_factor_at_projection_origin'):
            kwargs['scale_factor'] = crs.scale_factor_at_projection_origin
        if getattr(crs, 'false_easting'):
            kwargs['false_easting'] = crs.false_easting
        if getattr(crs, 'false_northing'):
            kwargs['false_northing'] = crs.false_northing
        return ccrs.Mercator(**kwargs)

    def orthographic_from_cf(self, crs):
        kwargs = dict(
            central_longitude=crs.longitude_of_projection_origin,
            central_latitude=crs.latitude_of_projection_origin,
            )
        return ccrs.Orthographic(**kwargs)

    def polar_stereographic_from_cf(self, crs):
        kwargs = dict(
            central_longitude=crs.straight_vertical_longitude_from_pole)
        if crs.latitude_of_projection_origin == -90:
            return ccrs.SouthPolarStereo(**kwargs)
        else:
            return ccrs.NorthPolarStereo(**kwargs)

    def rotated_latitude_longitude_from_cf(self, crs):
        args = [crs.grid_north_pole_longitude,
                crs.grid_north_pole_latitude]
        if hasattr(crs, 'north_pole_grid_longitude'):
            args.append(crs.north_pole_grid_longitude)
        return ccrs.RotatedPole(*args)

    def sinusoidal_from_cf(self, crs):
        try:
            kwargs = dict(
                central_longitude=crs.longitude_of_projection_origin,
                )
        except AttributeError:
            kwargs = dict(
                central_longitude=crs.longitude_of_central_meridian,
                )
        if getattr(crs, 'false_easting'):
            kwargs['false_easting'] = crs.false_easting
        if getattr(crs, 'false_northing'):
            kwargs['false_northing'] = crs.false_northing
        return ccrs.Sinusoidal(**kwargs)

    def stereographic_from_cf(self, crs):
        kwargs = dict(
            central_latitude=crs.latitude_of_projection_origin,
            central_longitude=crs.longitude_of_projection_origin,
            scale_factor=crs.scale_factor_at_projection_origin
            )
        if getattr(crs, 'false_easting'):
            kwargs['false_easting'] = crs.false_easting
        if getattr(crs, 'false_northing'):
            kwargs['false_northing'] = crs.false_northing
        return ccrs.Stereographic(**kwargs)

    def transverse_mercator_from_cf(self, crs):
        kwargs = dict(
            central_longitude=crs.longitude_of_central_meridian,
            central_latitude=crs.latitude_of_projection_origin,
            scale_factor=crs.scale_factor_at_central_meridian,
            )
        if getattr(crs, 'false_easting'):
            kwargs['false_easting'] = crs.false_easting
        if getattr(crs, 'false_northing'):
            kwargs['false_northing'] = crs.false_northing
        return ccrs.TransverseMercator(**kwargs)

    def set_projection(self, value, *args, **kwargs):
        if value == 'cf':
            ret = self.cf_projection
            if ret is not None:
                return ret
        if isinstance(value, ccrs.CRS):
            return value
        else:
            value = 'cyl' if value == 'cf' else value
            return self.projections[value](**self.get_kwargs(
                value, *args, **kwargs))

    def get_kwargs(self, value, clon=None, clat=None):
        ret = {}
        keys = self.projection_kwargs[value]
        if 'central_longitude' in keys:
            ret['central_longitude'] = self.clon.clon if clon is None else clon
        if 'central_latitude' in keys:
            ret['central_latitude'] = self.clat.clat if clat is None else clat
        if 'pole_longitude' in keys:  # rotated pole
            ret['pole_longitude'] = self.clon.clon if clon is None else clon
            ret['pole_longitude'] -= 180
        if 'pole_latitude' in keys:  # rotated pole
            ret['pole_latitude'] = self.clat.clat if clat is None else clat
        self.logger.debug("Setting projection with %s", ret)
        return ret


class Projection(ProjectionBase):
    """
    Specify the projection for the plot

    This formatoption defines the projection of the plot

    Possible types
    --------------
    %(ProjectionBase.possible_types)s

    See Also
    --------
    %(ProjectionBase.see_also)s

    Warnings
    --------
    An update of the projection clears the axes!
    """

    # the axes has to be cleared completly if the projection is updated
    priority = START

    #: an update of this formatoption requires that the axes is cleared
    requires_clearing = True

    name = 'Projection of the plot'

    dependencies = ['clon', 'clat']

    connections = ['transform']

    def __init__(self, *args, **kwargs):
        super(Projection, self).__init__(*args, **kwargs)
        self.projection = None

    def initialize_plot(self, value, clear=True):
        """Initialize the plot and set the projection for the axes
        """
        self.projection = self.set_projection(value)
        if self.plotter.cleared:
            self.ax.projection = self.projection
            self.ax.clear()

    def update(self, value):
        """Update the formatoption

        Since this formatoption requires clearing, this method does nothing.
        Everything is done in the :meth:`initialize_plot` method.
        """
        pass


@docstrings.get_sections(base='BoxBase')
class BoxBase(Formatoption):
    """
    Abstract base class for specifying a longitude-latitude box

    Possible types
    --------------
    str
        A pattern that matches any of the keys in the :attr:`psyplot.rcParams`
        ``'extents.boxes'`` item (contains user-defined longitude-latitude
        boxes) or the :attr:`psyplot.plotter.boxes.lonlatboxes` dictionary
        (contains longitude-latitude boxes of different countries and
        continents)
    [lonmin, lonmax, latmin, latmax]
        The surrounding longitude-latitude that shall be used. Values can be
        either a float or a string as above

    See Also
    --------
    LonLatBox, MapExtent
    """

    def lola_from_pattern(self, s):
        """
        Calculate the longitude-latitude box based upon a pattern

        This method uses the psyplot.rcParams ``'extents.boxes'`` item to find
        longitude that match `s` and takes the surrounding box.

        Parameters
        ----------
        s: str
            The pattern to use for the keys in the
            :attr:`psyplot.plotter.maps.lonlatboxes` dictionary and the
            ``'extents.boxes'`` item in the :attr:`psyplot.rcParams`

        Returns
        -------
        float: lonmin, lonmax, latmin, latmax or None
            The surrounding longitude-latitude box of all items in
            ``psyplot.rcParams['extents.boxes']`` whose key match `s` if there
            was any match. Otherwise None is returned
        """
        patt = re.compile(s)
        boxes = np.array([
            box for key, box in chain(*map(
                six.iteritems, [lonlatboxes, rcParams['lonlatbox.boxes']]))
            if patt.search(key)])
        if len(boxes) == 0:
            similar_keys = get_close_matches(s, rcParams['lonlatbox.boxes'])
            message = "Did not find any matches for %s!" % s
            if similar_keys:
                message += " Maybe you mean on of " + ', '.join(
                    similar_keys)
            warnings.warn(message, RuntimeWarning)
            return
        return [boxes[:, 0].min(), boxes[:, 1].max(),
                boxes[:, 2].min(), boxes[:, 3].max()]


docstrings.keep_types('BoxBase.possible_types', 'str', 'str')


class CenterLon(BoxBase):
    """
    Set the center longitude of the plot

    Parameters
    ----------
    None
        Let the :attr:`lonlatbox` formatoption determine the center
    float
        Specifiy the center manually
    %(BoxBase.possible_types.str)s
    """

    priority = START

    name = 'Longitude of the center of the plot'

    requires_clearing = True

    dependencies = ['lonlatbox']

    def update(self, value):
        self.lon_mean = float(np.mean(self.lonlatbox.lonlatbox[:2]))
        if value is not None:
            if isinstance(value, six.string_types):
                box = self.lola_from_pattern(value)
                if box is not None:
                    self.clon = float(np.mean(box[:2]))
                else:
                    value = None
            else:
                self.clon = value
        if value is None:
            lonmin, lonmax = self.lonlatbox.lonlatbox[:2]
            if lonmax - lonmin > 350:
                self.clon = 0.0
            else:
                self.clon = self.lon_mean


class CenterLat(BoxBase):
    """
    Set the center latitude of the plot

    Parameters
    ----------
    None
        Let the :attr:`lonlatbox` formatoption determine the center
    float
        Specifiy the center manually
    %(BoxBase.possible_types.str)s
    """

    priority = START

    name = 'Latitude of the center of the plot'

    requires_clearing = True

    dependencies = ['lonlatbox']

    def update(self, value):
        self.lat_mean = float(np.mean(self.lonlatbox.lonlatbox[2:]))
        if value is not None:
            if isinstance(value, six.string_types):
                box = self.lola_from_pattern(value)
                if box is not None:
                    self.clat = float(np.mean(box[2:]))
                else:
                    value = None
            else:
                self.clat = value
        if value is None:
            latmin, latmax = self.lonlatbox.lonlatbox[2:]
            if latmax - latmin > 170:
                self.clat = 0
            else:
                self.clat = self.lat_mean


@docstrings.get_sections(base='LonLatBox')
class LonLatBox(BoxBase):
    """
    Set the longitude-latitude box of the data shown

    This formatoption extracts the data that matches the specified box.

    Possible types
    --------------
    None
        Use the full data
    %(BoxBase.possible_types)s

    Notes
    -----
    - For only specifying the region of the plot, see the :attr:`map_extent`
      formatoption
    - If the coordinates are two-dimensional (e.g. for a circumpolar grid),
      than the data is not extracted but values outside the specified
      longitude-latitude box are set to NaN

    See Also
    --------
    map_extent"""

    priority = START

    name = 'Longitude-Latitude box of the data'

    requires_clearing = True

    dependencies = ['transform', 'transpose']

    @property
    def lonlatbox_transformed(self):
        value = np.asarray(self.lonlatbox)
        transformed = self.transform.projection.transform_points(
            ccrs.PlateCarree(), value[:2], value[2:])[..., :2]
        value[:2] = transformed[..., 0]
        value[2:] = transformed[..., 1]
        if value[0] == value[1] and isinstance(self.transform.projection,
                                               ccrs.PlateCarree):
            value[1] += 360
        return value

    def data_dependent(self, data, set_data=True):
        if isinstance(data, InteractiveList):
            data = data[0]
        decoder = data.psy.decoder
        lon, lat = self._get_lola(data, decoder)
        new_lonlatbox = self.calc_lonlatbox(lon, lat)
        update = self.data_lonlatbox != new_lonlatbox
        if not update and set_data and self.value is not None:
            self.update(self.value)
        elif update:
            self.logger.debug(
                "Reinitializing because lonlatbox of new data %s does not "
                "match the old one %s", new_lonlatbox, self.data_lonlatbox)
        return update

    def update(self, value):
        if isinstance(self.data, InteractiveList):
            for i, arr in enumerate(self.data):
                decoder = self.get_decoder(i)
                self.data[i] = self.update_array(
                    value, arr, decoder, next(six.itervalues(
                        self.raw_data[i].psy.base_variables)))
        else:
            arr = self.data
            arr = self.update_array(
                    value, arr, self.decoder, next(six.itervalues(
                        self.raw_data.psy.base_variables)))
            self.data = arr

    def _get_lola(self, data, decoder):
        """Get longitude and latitde informations from the given data array"""
        lon_da = decoder.get_x(data, data.coords)
        lat_da = decoder.get_y(data, data.coords)
        lon, lat = self.to_degree(lon_da.attrs.get('units'), lon_da.values,
                                  lat_da.values)
        data_shape = data.shape[-2:] if not decoder.is_unstructured(data) \
            else (data.shape[-1], )
        is_combined = data_shape != data.shape
        if lon.shape == data_shape and lat.shape == data_shape:
            vals = data.values[0 if is_combined else slice(None)]
            lon = np.where(np.isnan(vals), np.nan, lon)
            lat = np.where(np.isnan(vals), np.nan, lat)
        return lon, lat

    def update_array(self, value, data, decoder, base_var=None):
        """Update the given `data` array"""
        lon, lat = self._get_lola(data, decoder)
        self.data_lonlatbox = self.calc_lonlatbox(lon, lat)
        if value is None:
            self.lonlatbox = self.data_lonlatbox
            return data
        else:
            if isinstance(value, six.string_types):
                value = self.lola_from_pattern(value)
                if value is None:
                    self.lonlatbox = self.data_lonlatbox
                    return data
            value = list(value)
            for i, v in enumerate(value):
                if isinstance(v, six.string_types):
                    value[i] = self.lola_from_pattern(v)[i]
            is_unstructured = decoder.is_unstructured(
                base_var if base_var is not None else data)
            is_rectilinear = lon.ndim == 1 and not is_unstructured
            shift = isinstance(self.transform.projection, ccrs.PlateCarree)
            if is_rectilinear and shift:
                data = data.copy(True)
                lon, data.values = self.shiftdata(
                    lon, data.values, np.mean(value[:2]))
                lon_name = decoder.get_x(data, data.coords).name
                data[lon_name] = (data[lon_name].dims, lon,
                                  data[lon_name].attrs)
            elif is_unstructured and shift:
                # make sure that we are inside the map extent
                ret = self.transform.projection.transform_points(
                    self.transform.projection, lon, lat)
                lon = ret[..., 0]
                lat = ret[..., 1]
            self.lonlatbox = value
            value = self.lonlatbox_transformed
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', 'invalid value encountered',
                                        RuntimeWarning)
                if is_rectilinear and (lat[1:] < lat[:-1]).all():
                    lat_values = value[3:1:-1]
                else:
                    lat_values = value[2:]
                if is_rectilinear:
                    kwargs = dict(zip(
                        data.dims[-2:], starmap(
                            slice, [lat_values, value[:2]])))
                    ret = data.sel(**kwargs)
                else:
                    ret = self.mask_outside(data.copy(True), lon, lat, *value,
                                            is_unstructured=is_unstructured)
            lon, lat = self._get_lola(ret, decoder)
            self.data_lonlatbox = self.calc_lonlatbox(lon, lat)
            return ret

    def to_degree(self, units=None, *args):
        """Converts arrays with radian units to degree

        Parameters
        ----------
        units: str
            if ``'radian'``, the arrays in ``*args`` will be converted
        ``*args``
            numpy arrays

        Returns
        -------
        list of np.ndarray
            returns the arrays provided with ``*args``

        Notes
        -----
        if `units` is ``'radian'``, a copy of the array will be returned"""
        args = list(args)
        if units == 'radian' and isinstance(self.transform.projection,
                                            ccrs.PlateCarree):
            for i, array in enumerate(args):
                args[i] = array * 180. / np.pi
        return args

    def mask_outside(self, data, lon, lat, lonmin, lonmax, latmin, latmax,
                     is_unstructured=False):
        data.values = data.values.copy()
        ndim = 2 if not is_unstructured else 1
        if lonmax < lonmin:
            lonmax += 360
            lon[lon < 0] += 360
        mask = np.any([lon < lonmin, lon > lonmax, lat < latmin,
                       lat > latmax], axis=0)
        if data.ndim > ndim:
            if is_unstructured:
                data = data.psy[:, np.where(~mask)[0]]
            else:
                for i, arr in enumerate(data.values):
                    arr[mask] = np.nan
                    data.values[i, :] = arr
        else:
            if is_unstructured:
                data = data.psy[np.where(~mask)[0]]
            else:
                data.values[mask] = np.nan
        return data

    def calc_lonlatbox(self, lon, lat):
        if isinstance(self.transform.projection, ccrs.PlateCarree):
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', 'invalid value encountered',
                                        RuntimeWarning)
                lon = lon[np.all([lon >= -180, lon <= 360], axis=0)]
                lat = lat[np.all([lat >= -90, lat <= 90], axis=0)]
                return [lon.min(), lon.max(), lat.min(), lat.max()]
        else:
            if lon.ndim == 1:
                lon, lat = np.meshgrid(lon, lat)
            points = ccrs.PlateCarree().transform_points(
                self.transform.projection, lon, lat)
            lon = points[..., 0]
            lat = points[..., 1]
            lon_0 = self.transform.projection.proj4_params.get('lon_0')
            if lon_0 is not None:
                lon = np.where(lon > lon_0 + 180, lon - 360, lon)
                lon = np.where(lon < lon_0 - 180, lon + 360, lon)
            return [lon.min(), lon.max(), lat.min(), lat.max()]

    def shiftdata(self, lonsin, datain, lon_0):
        """
        Shift the data such that it matches the region we want to show

        Parameters
        ----------
        %(shiftdata.parameters)s

        Notes
        -----
        `datain` can also be multiple fields stored in a three-dimensional
        array. Then we shift all fields along the first dimension
        """
        # shiftdata does not work properly if we do not give the bounds of
        # the array
        lonsin = lonsin.copy()
        if lonsin.ndim == 1:
            get_centers = True
            lonsin = _infer_interval_breaks(lonsin)
        else:
            get_centers = False
        if datain.ndim == 2:
            lonsin, datain = shiftdata(lonsin, datain, lon_0)
        elif datain.ndim == 3:
            lon_save = lonsin.copy()
            for i, a in enumerate(datain):
                lonsin, datain[i] = shiftdata(lon_save, a, lon_0)
        if get_centers:
            lonsin = np.mean([lonsin[1:], lonsin[:-1]], axis=0)
        return lonsin, datain


class MapExtent(BoxBase):
    """
    Set the extent of the map

    Possible types
    --------------
    None
        The map extent is specified by the data (i.e. by the :attr:`lonlatbox`
        formatoption)
    'global'
        The whole globe is shown
    'data'
        The extent is set to the longitude-latitute box defined by the
        :attr:`lonlatbox` formatoption
    %(BoxBase.possible_types)s

    Notes
    -----
    This formatoption sets the extent of the plot. For choosing the region for
    the data, see the :attr:`lonlatbox` formatoption

    See Also
    --------
    lonlatbox"""

    dependencies = ['lonlatbox', 'plot', 'vplot']

    name = 'Longitude-Latitude box of the plot'

    priority = END

    update_after_plot = True

    @property
    def coords(self):
        arr = next(self.iter_data)
        return [arr.psy.get_coord('x'),
                arr.psy.get_coord('y')]

    def update(self, value):
        set_global = False
        if isinstance(value, six.string_types):
            if value == 'global':
                set_global = True
            elif value == 'data':
                value = self.lonlatbox.lonlatbox
            else:
                value = self.lola_from_pattern(value)
        elif value is None:
            # use autoscale
            self.ax.autoscale()
            return
        else:
            value = list(value)
            for i, v in enumerate(value):
                if isinstance(v, six.string_types):
                    value[i] = self.lola_from_pattern(v)[i]
        # Since the set_extent method does not always work and the data limits
        # are not always correctly set, we test here whether the wished
        # extent (the value) is almost global. If so, we set it to a global
        # value
        if not set_global:
            with self.ax.hold_limits():
                self.ax.set_global()
                try:
                    x1, x2, y1, y2 = self.ax.get_extent(ccrs.PlateCarree())
                except ValueError:
                    # extent could not be guessed, we let cartopy do everything
                    x_rng = 360
                    y_rng = 180
                else:
                    x_rng = 360. if x1 == x2 else x2 - x1
                    y_rng = y2 - y1
            proj = self.ax.projection
        if set_global or ((value[1] - value[0]) / x_rng > 0.95 and
                          (value[3] - value[2]) / y_rng > 0.95):
            self.logger.debug("Setting to global extent...")
            self.ax.set_global()
            return
        elif (isinstance(proj,
                         (ccrs.Orthographic, ccrs.Stereographic)) and
              np.abs(np.diff(value[:2])) > 350):
            # HACK: to make sure, that we don't get troubles with the
            # transformation, we don't go until 360
            value[:2] = 0, 359.9999
            self.ax.set_extent(value, crs=ccrs.PlateCarree())
        else:
            try:
                self.ax.set_extent(value, crs=ccrs.PlateCarree())
            except ValueError:
                self.logger.debug(
                    "Failed to set_extent with lonlatbox %s", value,
                    exc_info=True)


class MapBackground(psy_simple.base.BackgroundColor):

    __doc__ = psy_simple.base.BackgroundColor.__doc__

    def update(self, value):
        super().update(value)
        if cartopy.__version__ < "0.18":
            self.ax.background_patch.set_facecolor(self.ax.patch.get_facecolor())
        else:
            self.ax.patch.set_facecolor(self.ax.patch.get_facecolor())


class ClipAxes(Formatoption):
    """
    Clip the part outside the latitudes of the map extent

    Possible types
    --------------
    None
        Clip if all longitudes are shown (i.e. the extent goes from -180 to
        180) and the projection is orthographic or stereographic
    bool
        True, clip, else, don't

    Notes
    -----
    If the plot is clipped. You might need to update with `replot=True`!
    """

    _orig_path = None

    priority = BEFOREPLOTTING

    connections = ['lonlatbox', 'map_extent']

    def draw_circle(self):
        import matplotlib.path as mpath
        self._orig_path = self._orig_path or self.ax.patch.get_path()
        theta = np.linspace(0, 2*np.pi, 100)
        center, radius = [0.5, 0.5], 0.5
        verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        circle = mpath.Path(verts * radius + center)
        self.ax.set_boundary(circle, transform=self.ax.transAxes)

    def update(self, value):
        if value:
            self.draw_circle()
        elif value is None:
            proj = self.ax.projection
            extent = self.map_extent.value or self.lonlatbox.lonlatbox
            if extent == 'data':
                extent = self.lonlatbox.lonlatbox
            if (isinstance(proj, (ccrs.Orthographic, ccrs.Stereographic)) and
                    extent != 'global' and np.abs(np.diff(extent[:2])) > 350):
                self.draw_circle()
            else:
                self.remove()
        else:
            self.remove()

    def remove(self):
        if self._orig_path is not None:
            try:
                self.ax.set_boundary(self._orig_path,
                                     transform=self.ax.transAxes)
            except Exception:
                pass
            del self._orig_path


class Transform(ProjectionBase):
    """
    Specify the coordinate system of the data

    This formatoption defines the coordinate system of the data (usually we
    expect a simple latitude longitude coordinate system)

    Possible types
    --------------
    %(ProjectionBase.possible_types)s

    See Also
    --------
    %(ProjectionBase.see_also)s
    """

    priority = START

    name = 'Coordinate system of the data'

    connections = ['plot', 'vplot']

    @property
    def cf_projection(self):
        data = next(self.iter_data)
        xcoord = data.psy.get_coord('x')
        if xcoord.attrs.get('standard_name') == 'longitude':
            return ccrs.PlateCarree()
        return super().cf_projection

    def update(self, value):
        if value == 'cf':
            self.projection = self.cf_projection
            if self.projection is None:
                self.projection = ccrs.PlateCarree()
        elif value == 'cyl':
            self.projection = ccrs.PlateCarree()
        else:
            self.projection = self.set_projection(value, 0, 0)
        for key in self.connections:
            try:
                getattr(self, key)._kwargs['transform'] = self.projection
            except AttributeError:
                pass


class LSM(Formatoption):
    """
    Draw the continents

    Possible types
    --------------
    bool
        True: draw the coastlines with a line width of 1
        False: don't draw anything
    float
        Specifies the linewidth of the coastlines
    str
        The resolution of the land-sea mask (see the
        :meth:`cartopy.mpl.geoaxes.GeoAxesSubplot.coastlines` method. Must be
        one of ``('110m', '50m', '10m')``.
    list [str or bool, float]
        The resolution and the linewidth
    dict
        A dictionary with any of the following keys

        coast
            The color for the coastlines
        land
            The fill color for the continents
        ocean
            The fill color for the oceans
        res
            The resolution (see above)
        linewidth
            The linewidth of the coastlines (see above)"""

    name = 'Land-Sea mask'

    lsm = None

    dependencies = ['background']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.draw_funcs = {
            ('coast', ): self.draw_coast,
            ('coast', 'land'): self.draw_land_coast,
            ('land', ): self.draw_land,
            ('ocean', ): self.draw_ocean,
            ('coast', 'land', 'ocean'): self.draw_all,
            ('land', 'ocean'): self.draw_land_ocean,
            ('coast', 'ocean'): self.draw_ocean_coast,
            }
        self.artists = []

    @property
    def lsm(self):
        try:
            return self.artists[0]
        except ValueError:
            return None

    @lsm.setter
    def lsm(self, val):
        self.artists.append(val)

    def draw_all(self, land, ocean, coast, res='110m', linewidth=1):
        land_feature = cf.LAND.with_scale(res)
        if land is None:
            land = land_feature._kwargs.get('facecolor')
        if ocean is None:
            ocean = cf.OCEAN._kwargs.get('facecolor')
        self.lsm = self.ax.add_feature(
            land_feature, facecolor=land, edgecolor=coast, linewidth=linewidth)
        # draw the coast above the plot
        self.lsm = self.ax.add_feature(
            land_feature, facecolor='none', edgecolor=coast,
            linewidth=linewidth, zorder=1)
        if cartopy.__version__ < "0.18":
            self.ax.background_patch.set_facecolor(ocean)
        else:
            self.ax.patch.set_facecolor(ocean)

    def draw_land(self, land, res='110m'):
        if cartopy.__version__ < "0.18":
            ocean = self.ax.background_patch.get_facecolor()
        else:
            ocean = self.ax.patch.get_facecolor()
        self.draw_all(land, ocean, 'face', res, 0.0)

    def draw_coast(self, coast, res='110m', linewidth=1.0):
        if coast is None:
            coast = 'k'
        self.lsm = self.ax.coastlines(res, color=coast, linewidth=linewidth)

    def draw_ocean(self, ocean, res='110m'):
        self.draw_ocean_coast(ocean, None, res, 0.0)

    def draw_land_coast(self, land, coast, res='110m', linewidth=1.0):
        if cartopy.__version__ < "0.18":
            ocean = self.ax.background_patch.get_facecolor()
        else:
            ocean = self.ax.patch.get_facecolor()
        self.draw_all(land, ocean, coast, res, linewidth)

    def draw_ocean_coast(self, ocean, coast, res='110m', linewidth=1.0):
        ocean_feature = cf.OCEAN.with_scale(res)
        if ocean is None:
            ocean = cf.OCEAN._kwargs.get('facecolor')
        self.lsm = self.ax.add_feature(
            ocean_feature, facecolor=ocean, edgecolor=coast,
            linewidth=linewidth)
        # draw the coast above the plot
        self.lsm = self.ax.add_feature(
            ocean_feature, facecolor='none', edgecolor=coast,
            linewidth=linewidth, zorder=1)

    def draw_land_ocean(self, land, ocean, res='110m'):
        self.draw_all(land, ocean, None, res, 0.0)

    def update(self, value):
        self.remove()
        # to make sure, we have a dictionary
        value = self.validate(value)
        keys = tuple(sorted({'land', 'ocean', 'coast'}.intersection(value)))
        if keys:
            self.draw_funcs[keys](**value)

    def remove(self):
        for artist in self.artists:
            try:
                artist.remove()
            except ValueError:
                pass
        self.artists.clear()
        try:
            self.background.update(self.background.value)
        except Exception:
            pass

    def get_fmt_widget(self, parent, project):
        from psy_maps.widgets import LSMFmtWidget
        return LSMFmtWidget(parent, self, project)


class StockImage(Formatoption):
    """
    Display a stock image on the map

    This formatoption uses the :meth:`cartopy.mpl.geoaxes.GeoAxes.stock_img`
    method to display a downsampled version of the Natural Earth shaded relief
    raster on the map

    Possible types
    --------------
    bool
        If True, the image is displayed
    """

    image = None

    priority = END

    name = 'Display Natural Earth shaded relief raster'

    connections = ['plot']

    def update(self, value):
        if value and self.image is None:
            self.image = self.ax.stock_img()
            # display below the plot
            try:
                mappable = self.plot.mappable
            except AttributeError:
                pass
            else:
                try:
                    self.image.zorder = mappable.zorder - 0.1
                except AttributeError:
                    self.image.zorder = (
                        mappable.collections[0].zorder - 0.1)
        elif not value and self.image is not None:
            self.remove()

    def remove(self):
        if self.image is None:
            return
        self.image.remove()
        del self.image


class GridColor(Formatoption):
    """
    Set the color of the grid

    Possible types
    --------------
    None
        Choose the default line color
    color
        Any valid color for matplotlib (see the :func:`matplotlib.pyplot.plot`
        documentation)

    See Also
    --------
    grid_settings, grid_labels, grid_labelsize, xgrid, ygrid"""

    name = 'Color of the latitude-longitude grid'

    connections = ['xgrid', 'ygrid']

    def update(self, value):
        if value is not None:
            for connection in self.connections:
                getattr(self, connection)._kwargs['color'] = value
        else:
            for connection in self.connections:
                getattr(self, connection)._kwargs.pop('color', None)


class GridLabels(Formatoption):
    """
    Display the labels of the grid

    Possible types
    --------------
    None
        Grid labels are draw if possible
    bool
        If True, labels are drawn and if this is not possible, a warning is
        raised

    See Also
    --------
    grid_color, grid_settings, grid_labelsize, xgrid, ygrid"""

    name = 'Labels of the latitude-longitude grid'

    dependencies = ['projection', 'transform']

    connections = ['xgrid', 'ygrid']

    def update(self, value):
        if value is None or value:
            # initialize a gridliner to see if we can draw the tick labels
            test_value = True
            try:
                Gridliner(
                    self.ax, self.ax.projection, draw_labels=test_value)
            except TypeError as e:  # labels cannot be drawn
                if value:
                    warnings.warn(str(e), RuntimeWarning)
                value = False
            else:
                value = True
        for connection in self.connections:
            getattr(self, connection)._kwargs['draw_labels'] = value


class GridSettings(DictFormatoption):
    """
    Modify the settings of the grid explicitly

    Possible types
    --------------
    dict
        Items may be any key-value-pair of the
        :class:`matplotlib.collections.LineCollection` class

    See Also
    --------
    grid_color, grid_labels, grid_labelsize, xgrid, ygrid"""

    children = ['grid_labels', 'grid_color']
    connections = ['xgrid', 'ygrid']

    name = 'Line properties of the latitude-longitude grid'

    def set_value(self, value, validate=True, todefault=False):
        if todefault:
            for connection in self.connections:
                for key in self.value:
                    getattr(self, connection)._kwargs.pop(key, None)
        super(GridSettings, self).set_value(value, validate, todefault)

    def update(self, value):
        for connection in self.connections:
            getattr(self, connection)._kwargs.update(value)


class GridLabelSize(Formatoption):
    """
    Modify the size of the grid tick labels

    Possible types
    --------------
    %(fontsizes)s

    See Also
    --------
    grid_color, grid_labels, xgrid, ygrid, grid_settings"""

    dependencies = ['xgrid', 'ygrid']

    name = 'Label size of the latitude-longitude grid'

    def update(self, value):
        for fmto in map(lambda key: getattr(self, key), self.dependencies):
            try:
                gl = fmto._gridliner
            except AttributeError:
                continue
            gl.xlabel_style['size'] = value
            gl.ylabel_style['size'] = value
            try:
                texts = [t[-1] for t in gl.label_artists]
            except AttributeError:  # cartopy < 0.18
                texts = chain(gl.xlabel_artists, gl.ylabel_artists)
            for text in texts:
                text.set_size(value)


@docstrings.get_sections(base='GridBase', sections=['Possible types', 'See Also'])
class GridBase(psyps.DataTicksCalculator):
    """
    Abstract base class for x- and y- grid lines

    Possible types
    --------------
    None
        Don't draw gridlines (same as ``False``)
    bool
        True: draw gridlines and determine position automatically
        False: don't draw gridlines
    %(DataTicksCalculator.possible_types)s
    int
        Specifies how many ticks to use with the ``'rounded'`` option. I.e. if
        integer ``i``, then this is the same as ``['rounded', i]``.

    See Also
    --------
    grid_color, grid_labels"""

    dependencies = ['transform', 'grid_labels', 'grid_color', 'grid_settings',
                    'projection', 'lonlatbox', 'map_extent']

    connections = ['plot']

    @abstractproperty
    def axis(self):
        """The axis string"""
        pass

    def __init__(self, *args, **kwargs):
        super(GridBase, self).__init__(*args, **kwargs)
        self._kwargs = {}

    def update(self, value):
        self.remove()
        if value is None or value is False:
            return
        if value is True:
            loc = None
        elif isinstance(value[0], six.string_types):
            loc = ticker.FixedLocator(self.calc_funcs[value[0]](*value[1:]))
        elif isinstance(value, tuple):
            steps = 11 if len(value) == 2 else value[3]
            loc = ticker.FixedLocator(
                np.linspace(value[0], value[1], steps, endpoint=True))
        else:
            loc = ticker.FixedLocator(value)
        try:
            mappable = self.plot.mappable
        except AttributeError:
            zorder = 0.9
        else:
            try:
                zorder = mappable.zorder
            except AttributeError:
                    zorder = mappable.collections[0].zorder
        self._gridliner = self.ax.gridlines(
            crs=ccrs.PlateCarree(), zorder=zorder + 0.1,
            **self.get_kwargs(loc))
        self._modify_gridliner(self._gridliner)
        self._disable_other_axis()

    def get_kwargs(self, loc):
        return dict(chain(self._kwargs.items(), [(self.axis + 'locs', loc)]))

    def _disable_other_axis(self):
        label_positions = {'x': ['bottom', 'top'], 'y': ['left', 'right']}
        other_axis = 'y' if self.axis == 'x' else 'x'
        setattr(self._gridliner, other_axis + 'lines', False)
        for pos in label_positions[other_axis]:
            if cartopy.__version__ < "0.18":  # cartopy < 0.18
                setattr(self._gridliner, other_axis + 'labels_' + pos, False)
            else:
                setattr(self._gridliner, pos + '_labels', False)

    def _modify_gridliner(self, gridliner):
        """Modify the formatting of the given `gridliner` before drawing"""
        if cartopy.__version__ < "0.18":  # cartopy < 0.18
            gridliner.xlabels_top = False
            gridliner.ylabels_right = False
        else:
            gridliner.top_labels = False
            gridliner.right_labels = False
        gridliner.yformatter = lat_formatter
        gridliner.xformatter = lon_formatter

    def get_fmt_widget(self, parent, project):
        from psy_maps.widgets import GridFmtWidget
        return GridFmtWidget(parent, self, project)

    def remove(self):
        if not hasattr(self, '_gridliner'):
            return
        gl = self._gridliner
        try:
            artists = chain(gl.xline_artists, gl.yline_artists,
                            [t[-1] for t in gl.label_artists])
        except AttributeError:  # cartopy < 0.17
            artists = chain(gl.xline_artists, gl.yline_artists,
                            gl.xlabel_artists, gl.ylabel_artists)
        for artist in artists:
            artist.remove()
        if gl in self.ax._gridliners:
            self.ax._gridliners.remove(gl)
        del self._gridliner

    def _round_min_max(self, vmin, vmax):
        exp = np.floor(np.log10(abs(vmax - vmin)))
        return psyps.round_to_05([vmin, vmax], exp, mode='s')


class XGrid(GridBase):
    """
    Draw vertical grid lines (meridians)

    This formatoption specifies at which longitudes to draw the meridians.

    Possible types
    --------------
    %(GridBase.possible_types)s

    See Also
    --------
    ygrid, %(GridBase.see_also)s"""

    dependencies = GridBase.dependencies + ['clon']

    name = 'Meridians'

    @property
    def array(self):
        decoder = self.decoder
        coord = decoder.get_x(self.data, self.data.coords)
        arr = np.unique(decoder.get_plotbounds(coord, ignore_shape=True))
        if hasattr(coord, 'units') and coord.units == 'radian':
            arr *= 180. / np.pi
        arr = ccrs.PlateCarree().transform_points(
            self.transform.projection, arr, np.zeros(arr.shape))[..., 0]
        return arr

    axis = 'x'


class YGrid(GridBase):
    """
    Draw horizontal grid lines (parallels)

    This formatoption specifies at which latitudes to draw the parallels.

    Possible types
    --------------
    %(GridBase.possible_types)s

    See Also
    --------
    xgrid, %(GridBase.see_also)s"""

    name = 'Parallels'

    @property
    def array(self):
        decoder = self.decoder
        coord = decoder.get_y(self.data, self.data.coords)
        arr = np.unique(decoder.get_plotbounds(coord, ignore_shape=True))
        if hasattr(coord, 'units') and coord.units == 'radian':
            arr *= 180. / np.pi
        arr = ccrs.PlateCarree().transform_points(
            self.transform.projection, arr, np.zeros(arr.shape))[..., 1]
        return arr

    axis = 'y'


class MapPlot2D(psyps.Plot2D):
    __doc__ = psyps.Plot2D.__doc__
    # fixes the plot of unstructured unstructured data on round projections

    connections = psyps.Plot2D.connections + ['transform', 'lonlatbox']

    dependencies = psyps.Plot2D.dependencies + ['clip']

    @property
    def array(self):
        ret = super(MapPlot2D, self).array.astype(float)
        xcoord = self.xcoord
        if xcoord.ndim == 2 and isinstance(self.transform.projection,
                                           ccrs.PlateCarree):
            lon = xcoord.values
            lat = self.ycoord.values
            ret[np.any([lon <= -200, lon >= 400, lat <= -120, lat >= 120],
                       axis=0)] = np.nan
        return ret

    def _get_xy_pcolormesh(self):
        """Use the bounds for non-global circumpolar plots"""
        interp_bounds = self.interp_bounds.value
        if interp_bounds is None and self.decoder.is_circumpolar(
                self.raw_data):
            lola = self.lonlatbox
            if (isinstance(self.transform.projection, ccrs.PlateCarree) and
                    np.abs(np.diff(lola.data_lonlatbox[:2])[0]) < 355):
                return self.xbounds, self.ybounds
        return super(MapPlot2D, self)._get_xy_pcolormesh()

    def _contourf(self):
        t = self.ax.projection
        if isinstance(t, ccrs.CRS) and not isinstance(t, ccrs.Projection):
            raise ValueError('invalid transform:'
                             ' Spherical contouring is not supported - '
                             ' consider using PlateCarree/RotatedPole.')
        elif self.decoder.is_unstructured(self.raw_data):
            warnings.warn('Filled contour plots of unstructured data are not '
                          'correctly warped around!')
        return super(MapPlot2D, self)._contourf()

    @property
    def cell_nodes_x(self):
        xbounds = super(MapPlot2D, self).cell_nodes_x
        if isinstance(self.transform.projection, ccrs.PlateCarree):
            xbounds = xbounds.copy()
            xbounds[xbounds - xbounds.min(axis=1, keepdims=True) > 180] -= 360
        return xbounds

    def _polycolor(self):
        from matplotlib.collections import PolyCollection
        self.logger.debug('Retrieving data')
        arr = self.array
        cmap = self.cmap.get_cmap(arr)
        if hasattr(self, '_plot'):
            self.logger.debug('Updating plot')
            self._plot.update(dict(cmap=cmap, norm=self.bounds.norm))
            if hasattr(self, '_wrapped_plot'):
                self._wrapped_plot.update(dict(cmap=cmap,
                                               norm=self.bounds.norm))
        else:
            self.logger.debug('Retrieving bounds')
            xb = self.cell_nodes_x
            yb = self.cell_nodes_y
            wrap_proj_types = (ccrs._RectangularProjection,
                               ccrs._WarpedRectangularProjection,
                               ccrs.InterruptedGoodeHomolosine,
                               ccrs.Mercator)
            t = self.transform.projection
            proj = self.ax.projection
            wrapped_arr = None
            ### HACK: since the matplotlib transformation is very slow because
            # it transforms every single polygon, we transform the coordinates
            # already here and do not specify the `transform` property for the
            # `PolyCollection` down below.
            # However, in order to wrap the boundaries correctly, we have to
            # identify the corresponding grid cells and then use the standard
            # matplotlib transform.
            if isinstance(t, wrap_proj_types) and 'lon_0' in proj.proj4_params:
                # We adopt and copy some code from the methodology of cartopy
                # _pcolormesh_patched method of the geoaxes. As such, we
                # transform the coordinates using standard matplotlib
                # transformation and check if the hypotenuse is larger than
                # half of the available x-limit of the ax.projection
                # By using the geodetic version of `t`, we place the center
                # longitude to `lon_0` and cells at the boundary do have a
                # distance of 180 degrees to lon_0
                trans_to_data = t.as_geodetic()._as_mpl_transform(self.ax) - \
                    self.ax.transData
                # now we identify the cells at the x-boundary by
                # checking the hypotenuse
                coords = np.column_stack((xb.flat, yb.flat)).astype(float, copy=False)
                transformed_pts = trans_to_data.transform(coords)
                transformed_pts = transformed_pts.reshape(
                    xb.shape + (transformed_pts.shape[-1], ))

                with np.errstate(invalid='ignore'):
                    edge_lengths = np.hypot(
                        np.diff(transformed_pts[..., 0], axis=1),
                        np.diff(transformed_pts[..., 1], axis=1)
                    )
                    mask = (
                        (edge_lengths > abs(self.ax.projection.x_limits[1] -
                                            self.ax.projection.x_limits[0]) / 2) |
                        np.isnan(edge_lengths)
                    ).any(1)

                # if so, we create a wrapped collection that is transformed in
                # the standard matplotlib way.
                if mask.any():
                    wrapped_arr = arr[mask]
                    arr = arr[~mask]
                    xb_wrap = xb[mask]
                    yb_wrap = yb[mask]
                    xb = xb[~mask]
                    yb = yb[~mask]
            self.logger.debug('Making plot with %i cells', arr.size)
            transformed = proj.transform_points(
                t.as_geodetic(), xb.ravel(), yb.ravel())[..., :2].reshape(
                    xb.shape + (2, ))
            self._plot = PolyCollection(
                transformed, array=arr.ravel(),
                norm=self.bounds.norm, rasterized=True, cmap=cmap,
                edgecolors='none', antialiaseds=False)
            self.logger.debug('Adding collection to axes')
            self.ax.add_collection(self._plot)
            if cartopy.__version__ <= "0.18":
                self._plot.set_clip_path(self.ax.outline_patch)
            else:
                self._plot.set_clip_path(self.ax.spines["geo"])
            if wrapped_arr is not None:
                self.logger.debug('Making wrapped plot with %i cells',
                                  wrapped_arr.size)
                self._wrapped_plot = PolyCollection(
                    np.dstack([xb_wrap, yb_wrap]), array=wrapped_arr.ravel(),
                    norm=self.bounds.norm, rasterized=True, cmap=cmap,
                    transform=t, zorder=self._plot.zorder - 0.1,
                    edgecolors='none', antialiaseds=False)
                self.logger.debug('Adding wrapped collection to axes')
                self.ax.add_collection(self._wrapped_plot)
                if cartopy.__version__ <= "0.18":
                    self._wrapped_plot.set_clip_path(self.ax.outline_patch)
                else:
                    self._wrapped_plot.set_clip_path(self.ax.spines["geo"])
        self.logger.debug('Done.')

    def remove(self, *args, **kwargs):
        super(MapPlot2D, self).remove(*args, **kwargs)
        if hasattr(self, '_wrapped_plot'):
            self._wrapped_plot.remove()
            del self._wrapped_plot

    def add2format_coord(self, x, y):
        x, y = self.transform.projection.transform_point(
            x, y, self.ax.projection)
        # shift if necessary
        if isinstance(self.transform.projection, ccrs.PlateCarree):
            coord = self.xcoord
            if coord.min() >= 0 and x < 0:
                x -= 360
            elif coord.max() <= 180 and x > 180:
                x -= 360
            if 'rad' in coord.attrs.get('units', '').lower():
                x = np.deg2rad(x)
                y = np.deg2rad(y)
        return super(MapPlot2D, self).add2format_coord(x, y)


class MapDataGrid(psyps.DataGrid):

    __doc__ = psyps.DataGrid.__doc__ + '\n' + docstrings.dedent("""
    See Also
    --------
    xgrid
    ygrid""")

    def update(self, value):
        self.remove()
        if value is None:
            return
        xb = self.cell_nodes_x
        yb = self.cell_nodes_y
        if isinstance(self.transform.projection, ccrs.PlateCarree):
            xb[xb - xb.min(axis=1, keepdims=True) > 180] -= 360
        t = self.transform.projection
        proj = self.ax.projection
        # HACK: We remove the cells at the boundary of the map projection
        xb_wrap = None
        if isinstance(t, wrap_proj_types) and 'lon_0' in proj.proj4_params:
            # See the :meth:`MapPlot2D._polycolor` method for a documentation
            # of the steps
            trans_to_data = t.as_geodetic()._as_mpl_transform(self.ax) - \
                self.ax.transData

            coords = np.column_stack((xb.flat, yb.flat)).astype(float, copy=False)
            transformed_pts = trans_to_data.transform(coords)
            transformed_pts = transformed_pts.reshape(
                xb.shape + (transformed_pts.shape[-1], ))

            with np.errstate(invalid='ignore'):
                edge_lengths = np.hypot(
                    np.diff(transformed_pts[..., 0], axis=1),
                    np.diff(transformed_pts[..., 1], axis=1)
                )
                mask = (
                    (edge_lengths > abs(self.ax.projection.x_limits[1] -
                                        self.ax.projection.x_limits[0]) / 2) |
                    np.isnan(edge_lengths)
                ).any(1)

            if mask.any():
                xb_wrap = xb[mask]
                yb_wrap = yb[mask]
                xb = xb[~mask]
                yb = yb[~mask]
        orig_shape = xb.shape
        transformed = proj.transform_points(t.as_geodetic(), xb.ravel(),
                                            yb.ravel())
        xb = transformed[..., 0].reshape(orig_shape)
        yb = transformed[..., 1].reshape(orig_shape)
        # We insert nan values in the flattened edges arrays rather than
        # plotting the grid cells separately as it considerably speeds-up code
        # execution.
        n = len(xb)
        xb = np.c_[xb, xb[:, :1], [[np.nan]] * n].ravel()
        yb = np.c_[yb, yb[:, :1], [[np.nan]] * n].ravel()
        if isinstance(value, dict):
            self._artists = self.ax.plot(xb, yb, **value)
        else:
            self._artists = self.ax.plot(xb, yb, value)
        if xb_wrap is not None:
            n = len(xb_wrap)
            xb_wrap = np.c_[xb_wrap, xb_wrap[:, :1], [[np.nan]] * n].ravel()
            yb_wrap = np.c_[yb_wrap, yb_wrap[:, :1], [[np.nan]] * n].ravel()
            if isinstance(value, dict):
                self._artists.extend(self.ax.plot(xb_wrap, yb_wrap,
                                                  transform=t, **value))
            else:
                self._artists.extend(self.ax.plot(xb_wrap, yb_wrap, value,
                                                  transform=t))


class MapDensity(psyps.Density):
    """
    Change the density of the arrows

    Possible types
    --------------
    %(Density.possible_types)s"""

    def _set_quiver_density(self, value):
        if all(val == 1.0 for val in value):
            self.plot._kwargs.pop('regrid_shape', None)
        elif self.decoder.is_unstructured(self.raw_data):
            warnings.warn("Quiver plot of unstructered data does not support "
                          "the density keyword!", RuntimeWarning)
        elif self.decoder.is_circumpolar(self.raw_data):
            warnings.warn("Quiver plot of circumpolar data does not support "
                          "the density keyword!", RuntimeWarning)
        else:
            shape = self.data.shape[-2:]
            value = map(int, [value[0]*shape[0], value[1]*shape[1]])
            self.plot._kwargs['regrid_shape'] = tuple(value)

    def _unset_quiver_density(self):
        self.plot._kwargs.pop('regrid_shape', None)


class MapVectorColor(psyps.VectorColor):

    __doc__ = psyps.VectorColor.__doc__

    def _maybe_ravel(self, arr):
        # no need to ravel the data for quiver plots
        return np.asarray(arr)


class MapVectorPlot(psyps.VectorPlot):

    __doc__ = psyps.VectorPlot.__doc__

    dependencies = psyps.VectorPlot.dependencies + [
        'lonlatbox', 'transform', 'clon', 'clat', 'clip', 'transpose']

    def set_value(self, value, *args, **kwargs):
        # stream plots for circumpolar grids is not supported
        if (value == 'stream' and self.raw_data is not None and
                self.decoder.is_circumpolar(self.raw_data[0])):
            warnings.warn('Cannot make stream plots of circumpolar data!')
            value = 'quiver'
        super(MapVectorPlot, self).set_value(value, *args, **kwargs)

    set_value.__doc__ = psyps.VectorPlot.set_value.__doc__

    def _get_data(self):
        data = self.data
        base_data = self.raw_data
        base_var = next(base_data.psy.iter_base_variables)
        u, v = data[-2:].values
        x = base_data.psy.decoder.get_x(base_var, coords=data.coords)
        y = base_data.psy.decoder.get_y(base_var, coords=data.coords)
        x = x.values * 180. / np.pi if x.attrs.get('units') == 'radian' else \
            x.values
        y = y.values * 180. / np.pi if y.attrs.get('units') == 'radian' else \
            y.values
        # we transform here manually because the transform keyword does not
        # work with unstructered grids and it sometimes shifts the plot data
        # for quiver and stream plots
        transform = self.transform.projection
        if x.ndim == 1 and u.ndim == 2:
            x, y = np.meshgrid(x, y)
        if self.transpose.value and x.shape != u.shape:
            x = x.T
            y = y.T
        u, v = self.ax.projection.transform_vectors(transform, x, y, u, v)
        ret = self.ax.projection.transform_points(transform, x, y)
        x = ret[..., 0]
        y = ret[..., 1]
        self._kwargs.pop('transform', None)  # = self.ax.projection
        return x, y, u, v

    def _stream_plot(self):
        if self.decoder.is_circumpolar(self.raw_data):
            warnings.warn('Cannot make stream plots of circumpolar data!')
            return
        # update map extent such that it fits to the data limits (necessary
        # because streamplot scales the density based upon it). This however
        # does not work for matplotlib 1.5.0
        if not (mpl.__version__ == '1.5.0' and self.color.colored):
            self.ax.set_extent(self.lonlatbox.lonlatbox,
                               crs=ccrs.PlateCarree())
        else:
            self.ax.set_global()
        # Note that this method uses a bug fix through the
        # :class:`psyplot.plotter.colors.FixedColorMap` class
        # and one through the :class:`psyplot.plotter.colors.FixedBoundaryNorm`
        # class
        x, y, u, v = self._get_data()
        norm = self._kwargs.get('norm')
        if isinstance(norm, BoundaryNorm):
            self._kwargs['norm'] = FixedBoundaryNorm(
                norm.boundaries, norm.Ncmap, norm.clip)
        self._plot = self.ax.streamplot(x, y, u, v, **self._kwargs)

    def add2format_coord(self, x, y):
        x, y = self.transform.projection.transform_point(
            x, y, self.ax.projection)
        # shift if necessary
        if isinstance(self.transform.projection, ccrs.PlateCarree):
            coord = self.xcoord
            if coord.min() >= 0 and x < 0:
                x -= 360
            elif coord.max() <= 180 and x > 180:
                x -= 360
        return super(MapVectorPlot, self).add2format_coord(x, y)


class CombinedMapVectorPlot(MapVectorPlot):

    __doc__ = MapVectorPlot.__doc__

    def update(self, *args, **kwargs):
        self._kwargs['zorder'] = 2
        super(CombinedMapVectorPlot, self).update(*args, **kwargs)


class MapPlotter(psyps.Base2D):
    """Base plotter for visualizing data on a map
    """

    #: Boolean that is True if coordinates with units in radian should be
    #: converted to degrees
    convert_radian = True

    _rcparams_string = ['plotter.maps.']

    background = MapBackground('background')

    transpose = Transpose('transpose')
    projection = Projection('projection')
    transform = Transform('transform')
    clon = CenterLon('clon')
    clat = CenterLat('clat')
    lonlatbox = LonLatBox('lonlatbox')
    lsm = LSM('lsm')
    stock_img = StockImage('stock_img')
    grid_color = GridColor('grid_color')
    grid_labels = GridLabels('grid_labels')
    grid_labelsize = GridLabelSize('grid_labelsize')
    grid_settings = GridSettings('grid_settings')
    xgrid = XGrid('xgrid')
    ygrid = YGrid('ygrid')
    map_extent = MapExtent('map_extent')
    datagrid = MapDataGrid('datagrid', index_in_list=0)
    clip = ClipAxes('clip')

    @classmethod
    def _get_sample_projection(cls):
        """Returns None. May be subclassed to retumeshrn a projection that
        can be used when creating a subplot"""
        return ccrs.PlateCarree()

    @property
    def ax(self):
        """Axes instance of the plot"""
        if self._ax is None:
            import matplotlib.pyplot as plt
            plt.figure()
            self._ax = plt.axes(projection=getattr(
                self.projection, 'projection', self._get_sample_projection()))
        return self._ax

    @ax.setter
    def ax(self, value):
        self._ax = value


class FieldPlotter(psyps.Simple2DBase, MapPlotter, psyps.BasePlotter):
    """Plotter for 2D scalar fields on a map
    """

    _rcparams_string = ['plotter.fieldplotter']

    levels = psyps.ContourLevels('levels', cbounds='bounds')
    try:
        interp_bounds = psyps.InterpolateBounds('interp_bounds')
    except AttributeError:
        pass
    plot = MapPlot2D('plot')


class VectorPlotter(MapPlotter, psyps.BaseVectorPlotter, psyps.BasePlotter):
    """Plotter for visualizing 2-dimensional vector data on a map

    See Also
    --------
    psyplot.plotter.simple.SimpleVectorPlotter:
        for a simple version of drawing vector data
    FieldPlotter: for plotting scaler fields
    CombinedPlotter: for combined scalar and vector fields
    """
    _rcparams_string = ['plotter.vectorplotter']
    plot = MapVectorPlot('plot')
    density = MapDensity('density')
    color = MapVectorColor('color')


class CombinedPlotter(psyps.CombinedBase, FieldPlotter, VectorPlotter):
    """Combined 2D plotter and vector plotter on a map

    See Also
    --------
    psyplot.plotter.simple.CombinedSimplePlotter:
        for a simple version of this class
    FieldPlotter, VectorPlotter"""
    plot = MapPlot2D('plot', index_in_list=0)
    vplot = CombinedMapVectorPlot('vplot', cmap='vcmap', bounds='vbounds',
                                  index_in_list=1)
    density = MapDensity('density', plot='vplot', index_in_list=1)
    xgrid = XGrid('xgrid', index_in_list=1)
    ygrid = YGrid('ygrid', index_in_list=1)
