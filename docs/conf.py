"""
Sphinx configuration file for normix documentation.
"""

# Configuration file for the Sphinx documentation builder.
# This file only contains a selection of the most common options.

# -- Project information -----------------------------------------------------

project = 'normix'
copyright = '2024, normix developers'
author = 'normix developers'

# The full version, including alpha/beta/rc tags
release = '1.0.0'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings.
extensions = [
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'nbsphinx',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# -- Theme options -----------------------------------------------------------

html_theme_options = {
    'prev_next_buttons_location': 'bottom',
    'navigation_depth': 3,
}

# GitHub link in the top-right corner
html_context = {
    'display_github': True,
    'github_user': 'xshi19',
    'github_repo': 'normix',
    'github_version': 'main',
    'conf_py_path': '/docs/',
}

# -- Napoleon settings -------------------------------------------------------

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True

# -- nbsphinx settings ------------------------------------------------------

# Do not execute notebooks during build (use pre-executed outputs)
nbsphinx_execute = 'never'

# Link to the notebook on GitHub from each page
nbsphinx_prolog = r"""
{% set docname = env.doc2path(env.docname, base=None) %}

.. raw:: html

    <div class="admonition note">
    <p>This page was generated from a Jupyter notebook. You can
    <a href="https://github.com/xshi19/normix/blob/main/{{ docname }}">
    view it on GitHub</a> or download and run it locally.</p>
    </div>
"""

