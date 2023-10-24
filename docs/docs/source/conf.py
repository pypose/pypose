# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os, sys, re
import pypose_sphinx_theme

proj_root = os.path.abspath(os.path.join(__file__, "..", "..", ".."))
sys.path.insert(0, proj_root)


# -- Project information -----------------------------------------------------

project = 'PyPose'
copyright = '2022, PyPose Contributors'
author = 'PyPose Contributors'

# set by release script
RELEASE = os.environ.get('RELEASE', False)

# The full version, including alpha/beta/rc tags
def find_version(file_path: str) -> str:
    version_file = open(file_path).read()
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if not version_match:
        raise RuntimeError(f"Unable to find version string in {file_path}")
    return version_match.group(1)

version = find_version(os.path.join(proj_root, "pypose/_version.py"))

release = "main"
if RELEASE:
    version = '.'.join(version.split('.')[:2])
    html_title = " ".join((project, version, "documentation"))
    release = version


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinxcontrib.katex',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- Options for Latex rendering ---------------------------------------------
_PREAMBLE = r"""
    \usepackage{amsmath}
    \usepackage{amsfonts}
"""
latex_elements = {
    'preamble': _PREAMBLE,
}
# katex_prerender = True


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pypose_sphinx_theme"
html_theme_path = [pypose_sphinx_theme.get_html_theme_path()]
html_theme_options = {
    'pytorch_project': 'docs',
    'collapse_navigation': False,
    'display_version': True,
    'logo_only': False,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Disable displaying type annotations, these can be very verbose
autodoc_typehints = 'none'
