# -*- coding: utf-8 -*-
#
# psyplot documentation build configuration file, created by
# sphinx-quickstart on Mon Jul 20 18:01:33 2015.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

import sphinx
import inspect
import os
import os.path as osp
import sys
import re
import six
import subprocess as spr
from itertools import chain
from collections import defaultdict
import warnings
import psyplot
from psyplot.plotter import Formatoption, Plotter
import psy_maps
from autodocsumm import AutoSummClassDocumenter

# automatically import all plotter classes
psyplot.rcParams['project.auto_import'] = True
# include links to the formatoptions in the documentation of the
# :attr:`psyplot.project.ProjectPlotter` methods
Plotter.include_links(True)

warnings.filterwarnings('ignore', message="axes.color_cycle is deprecated")
warnings.filterwarnings(
    'ignore', message=("This has been deprecated in mpl 1.5,"))
warnings.filterwarnings('ignore', message="invalid value encountered in ")
warnings.filterwarnings('ignore', message=r"\s*examples.directory")
warnings.filterwarnings('ignore', message='numpy.dtype size changed')

# -- General configuration ------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosummary',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'matplotlib.sphinxext.plot_directive',
    'IPython.sphinxext.ipython_console_highlighting',
    'IPython.sphinxext.ipython_directive',
    'psyplot.sphinxext.extended_napoleon',
    'autodocsumm',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

if not osp.exists(osp.join(osp.dirname(__file__), 'api')):
    spr.check_call(['bash', 'apigen.bash'])

napoleon_use_admonition_for_examples = True

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

autodoc_default_options = {
    'show_inheritance': True,
    'autosummary': True,
}

autoclass_content = 'both'

not_document_data = ['psy_maps.plugin.defaultParams',
                     'psy_maps.plugin.rcParams']

ipython_savefig_dir = os.path.join(os.path.dirname(__file__), '_static')

# General information about the project.
project = 'psy-maps'
copyright = psy_maps.__copyright__
author = psy_maps.__author__

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = re.match('\d+\.\d+\.\d+', psy_maps.__version__).group()
# The full version, including alpha/beta/rc tags.
release = psy_maps.__version__

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = None

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True


# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets)
# here, relative to this directory. They are copied after the builtin
# static files, so a file named "default.css" will overwrite the builtin
# "default.css".
html_static_path = ['_static']

# otherwise, readthedocs.org uses their theme by default, so no need to specify

# Output file base name for HTML help builder.
htmlhelp_basename = 'psy-mapsdoc'

# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
    # Additional stuff for the LaTeX preamble.
    'preamble': '\setcounter{tocdepth}{10}'
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
  (master_doc, 'psy-maps.tex', 'psy-maps Documentation',
   author, 'manual'),
]


# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'psy-maps', 'psy-maps Documentation',
     [author], 1)
]


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
  (master_doc, 'psy-maps', 'psy-maps Documentation',
   author, 'psy-maps', 'The psyplot plugin for simple visualizations',
   'Miscellaneous'),
]


# -- Options for Epub output ----------------------------------------------

# Bibliographic Dublin Core info.
epub_title = project
epub_author = author
epub_publisher = author
epub_copyright = copyright

# A list of files that should not be packed into the epub file.
epub_exclude_files = ['search.html']

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    'pandas': ('http://pandas.pydata.org/pandas-docs/stable/', None),
    'numpy': ('https://docs.scipy.org/doc/numpy/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'sphinx': ('http://www.sphinx-doc.org/en/stable/', None),
    'xarray': ('http://xarray.pydata.org/en/stable/', None),
    'cartopy': ('https://scitools.org.uk/cartopy/docs/latest/', None),
    'mpl_toolkits': ('https://matplotlib.org/basemap/', None),
    'psyplot': ('https://psyplot.github.io/psyplot/', None),
    'psy_simple': ('https://psyplot.github.io/psy-simple/', None),
    'psy_reg': (
        'https://psyplot.readthedocs.io/projects/psy-reg/en/latest/', None),
    "python": ('https://docs.python.org/3/', None),
}

def group_formatoptions(app, what, name, obj, section, parent):
    if inspect.isclass(obj) and issubclass(obj, Formatoption):
        return 'Formatoption classes'
    elif inspect.isclass(obj) and issubclass(obj, Plotter):
        return 'Plotter classes'
    elif (inspect.isclass(parent) and issubclass(parent, Plotter) and
          isinstance(obj, Formatoption)):
        return obj.groupname


class PlotterAutoClassDocumenter(AutoSummClassDocumenter):
    """A ClassDocumenter that includes all the formatoption of a plotter"""

    priority = AutoSummClassDocumenter.priority + 0.1

    def filter_members(self, *args, **kwargs):
        ret = super(AutoSummClassDocumenter, self).filter_members(
            *args, **kwargs)
        if issubclass(self.object, Plotter):
            fmt_members = defaultdict(set)
            all_fmt = set(self.object._get_formatoptions())
            for i, (mname, member, isattr) in enumerate(ret):
                if isinstance(member, Formatoption):
                    fmt_members[member.group].add((mname, member, isattr))
                    all_fmt.remove(mname)
            for fmt in all_fmt:
                fmto = getattr(self.object, fmt)
                fmt_members[fmto.group].add((fmt, fmto, True))
            ret.extend(
                (tup for tup in chain(*map(sorted, fmt_members.values()))
                 if tup not in ret))
        return ret


def setup(app):
    app.add_autodocumenter(PlotterAutoClassDocumenter)
    app.connect('autodocsumm-grouper', group_formatoptions)
    return {'version': sphinx.__display_version__, 'parallel_read_safe': True}
