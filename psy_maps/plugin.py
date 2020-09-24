"""psy-simple psyplot plugin

This module defines the rcParams for the psy-simple plugin"""
import six
import yaml
from psyplot.config.rcsetup import RcParams
from matplotlib.rcsetup import validate_path_exists
from psy_simple.plugin import (
    try_and_error, validate_none, validate_str, validate_float,
    validate_bool_maybe_none, validate_fontsize,
    validate_color, validate_dict, BoundsValidator, bound_strings,
    ValidateInStrings, validate_bool, BoundsType, DictValValidator)
from psy_maps import __version__ as plugin_version


def get_versions(requirements=True):
    if requirements:
        import cartopy
        return {'version': plugin_version,
                'requirements': {'cartopy': cartopy.__version__}}
    else:
        return {'version': plugin_version}


def patch_prior_1_0(plotter_d, versions):
    """Patch psy_maps plotters for versions smaller than 1.0

    Before psyplot 1.0.0, the plotters in the psy_maps package where part of
    the psyplot.plotter.maps module. This has to be corrected"""
    plotter_d['cls'] = ('psy_maps.plotters', plotter_d['cls'][1])


#: patches to apply when loading a project
patches = {
    ('psyplot.plotter.maps', 'MapPlotter'): patch_prior_1_0,
    ('psyplot.plotter.maps', 'VectorPlotter'): patch_prior_1_0,
    ('psyplot.plotter.maps', 'FieldPlotter'): patch_prior_1_0,
    ('psyplot.plotter.maps', 'CombinedPlotter'): patch_prior_1_0,
    }


# -----------------------------------------------------------------------------
# ------------------------- validation functions ------------------------------
# -----------------------------------------------------------------------------


def validate_grid(val):
    if isinstance(val, tuple) and len(val) in [2, 3]:
        return val
    try:
        return validate_bool_maybe_none(val)
    except ValueError:
        return BoundsValidator(BoundsType)(val)


def validate_lsm(val):
    res_validation = ValidateInStrings('lsm', ['110m', '50m' ,'10m'])
    if not val:
        val = {}
    elif isinstance(val, dict):
        invalid = set(val).difference(
            ['coast', 'land', 'ocean', 'res', 'linewidth'])
        if invalid:
            raise ValueError(f"Invalid keys for lsm: {invalid}")
    else:
        # First try, if it's a bool, if yes, use 110m
        # then try, if it's a valid resolution
        # then try, if it's a float (i.e. the linewidth)
        # then try if it's a tuple [res, lw]
        try:
            validate_bool(val)
        except (ValueError, TypeError):
            pass
        else:
            val = '110m'
        try:
            val = res_validation(val)
        except (ValueError, TypeError):
            pass
        else:
            if not isinstance(val, str):
                val = '110m'
            val = {'res': val, 'linewidth': 1.0, 'coast': 'k'}
        try:
            val = validate_float(val)
        except (ValueError, TypeError):
            pass
        else:
            val = {'res': '110m', 'linewidth': val, 'coast': 'k'}
    if not isinstance(val, dict):
        try:
            res, lw = val
        except (ValueError, TypeError):
            raise ValueError(f"Invalid lsm configuration: {val}")
        else:
            val = {'res': res, 'linewidth': lw}
    val = dict(val)
    for key, v in val.items():
        if key in ['coast', 'land', 'ocean']:
            val[key] = validate_color(v)
        elif key == 'res':
            val[key] = res_validation(v)
        else:
            val[key] = validate_float(v)  # linewidth
    # finally set black color if linewidth is in val
    if 'linewidth' in val:
        val.setdefault('coast', 'k')
    return val


class ProjectionValidator(ValidateInStrings):

    def __call__(self, val):
        if isinstance(val, six.string_types):
            return ValidateInStrings.__call__(self, val)
        return val  # otherwise we skip the validation


def validate_dict_yaml(s):
    if isinstance(s, dict):
        return s
    validate_path_exists(s)
    if s is not None:
        with open(s) as f:
            return yaml.load(f)


def validate_lonlatbox(value):
    validate = try_and_error(validate_float, validate_str)
    try:
        return validate_none(value)
    except (TypeError, ValueError):
        try:
            return validate_str(value)
        except (TypeError, ValueError):
            if len(value) != 4:
                raise ValueError("Need 4 values for longitude-latitude box, "
                                 "got %i" % len(value))
            return list(map(validate, value))


# -----------------------------------------------------------------------------
# ------------------------------ rcParams -------------------------------------
# -----------------------------------------------------------------------------


#: the :class:`~psyplot.config.rcsetup.RcParams` for the psy-simple plugin
rcParams = RcParams(defaultParams={

    # -------------------------------------------------------------------------
    # ----------------------- Registered plotters -----------------------------
    # -------------------------------------------------------------------------

    'project.plotters': [

        {'maps': {
             'module': 'psy_maps.plotters',
             'plotter_name': 'MapPlotter',
             'plot_func': False,
             'summary': 'The data objects visualized on a map'},
         'mapplot': {
             'module': 'psy_maps.plotters',
             'plotter_name': 'FieldPlotter',
             'prefer_list': False,
             'default_slice': 0,
             'default_dims': {'x': slice(None), 'y': slice(None)},
             'summary': 'Plot a 2D scalar field on a map'},
         'mapvector': {
             'module': 'psy_maps.plotters',
             'plotter_name': 'VectorPlotter',
             'prefer_list': False,
             'default_slice': 0,
             'default_dims': {'x': slice(None), 'y': slice(None)},
             'summary': 'Plot a 2D vector field on a map',
             'example_call': "filename, name=[['u_var', 'v_var']], ..."},
         'mapcombined': {
             'module': 'psy_maps.plotters',
             'plotter_name': 'CombinedPlotter',
             'prefer_list': True,
             'default_slice': 0,
             'default_dims': {'x': slice(None), 'y': slice(None)},
             'summary': ('Plot a 2D scalar field with an overlying vector '
                         'field on a map'),
             'example_call': (
                 "filename, name=[['my_variable', ['u_var', 'v_var']]], ...")},
         }, validate_dict],

    # -------------------------------------------------------------------------
    # --------------------- Default formatoptions -----------------------------
    # -------------------------------------------------------------------------
    # MapBase
    'plotter.maps.transpose': [
        False, validate_bool, "Transpose the input data before plotting"],
    'plotter.maps.lonlatbox': [
        None, validate_lonlatbox,
        'fmt key to define the longitude latitude box of the data'],
    'plotter.maps.map_extent': [
        None, validate_lonlatbox,
        'fmt key to define the extent of the map plot'],
    'plotter.maps.clip': [
        None, validate_bool_maybe_none,
        'fmt key to define clip the axes outside the latitudes'],
    'plotter.maps.clon': [
        None, try_and_error(validate_none, validate_float, validate_str),
        'fmt key to specify the center longitude of the projection'],
    'plotter.maps.clat': [
        None, try_and_error(validate_none, validate_float, validate_str),
        'fmt key to specify the center latitude of the projection'],
    # TODO: Implement the drawing of shape files on a map
    # 'plotter.maps.lineshapes': [None, try_and_error(
    #     validate_none, validate_dict, validate_str, validate_stringlist)],
    'plotter.maps.grid_labels': [
        None, validate_bool_maybe_none,
        'fmt key to draw labels of the lat-lon-grid'],
    'plotter.maps.grid_labelsize': [
        12.0, validate_fontsize,
        'fmt key to modify the fontsize of the lat-lon-grid labels'],
    'plotter.maps.grid_color': [
        'k', try_and_error(validate_none, validate_color),
        'fmt key to modify the color of the lat-lon-grid'],
    'plotter.maps.grid_settings': [
        {}, validate_dict,
        'fmt key for additional line properties for the lat-lon-grid'],
    'plotter.maps.xgrid': [
        True, validate_grid, 'fmt key for drawing meridians on the map'],
    'plotter.maps.ygrid': [
        True, validate_grid, 'fmt key for drawing parallels on the map'],
    'plotter.maps.projection': [
        'cf', ProjectionValidator(
            'projection', ['cf', 'northpole', 'ortho', 'southpole', 'moll', 'geo',
                           'robin', 'cyl', 'stereo', 'near', 'rotated'],
            True),
        'fmt key to define the projection of the plot'],
    'plotter.maps.transform': [
        'cf', ProjectionValidator(
            'projection', ['cf', 'northpole', 'ortho', 'southpole', 'moll', 'geo',
                           'robin', 'cyl', 'stereo', 'near', 'rotated'],
            True),
        'fmt key to define the native projection of the data'],
    'plotter.maps.lsm': [
        True, validate_lsm,
        'fmt key to draw a land sea mask'],
    'plotter.maps.stock_img': [
        False, validate_bool, 'fmt key to draw a stock_img on the map'],

    # -------------------------------------------------------------------------
    # ---------------------------- Miscallaneous ------------------------------
    # -------------------------------------------------------------------------

    # yaml file that holds definitions of lonlatboxes
    'lonlatbox.boxes': [
        {}, validate_dict_yaml,
        'longitude-latitude boxes that shall be accessible for the lonlatbox, '
        'map_extent, etc. keywords. May be a dictionary or the path to a '
        'yaml file'],

    })


rcParams._deprecated_map['plotter.maps.plot.min_circle_ratio'] = (
    'plotter.plot2d.plot.min_circle_ratio', 0.05)

rcParams.update_from_defaultParams()
