"""
Sphinx configuration file for normix documentation.
"""

import os
import sys

sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------

project = 'normix'
copyright = '2024–2026, normix developers'
author = 'normix developers'
release = '0.2.0'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'sphinx.ext.intersphinx',
    'nbsphinx',
]

templates_path = ['_templates']

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'pdfs', 'plans',
                    'references', 'tech_notes', 'design', 'notebooks']

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'

html_static_path = ['_static']

html_theme_options = {
    'prev_next_buttons_location': 'bottom',
    'navigation_depth': 3,
}

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

# -- autodoc settings --------------------------------------------------------

autodoc_member_order = 'bysource'
autodoc_typehints = 'description'

# -- intersphinx settings ----------------------------------------------------

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'jax': ('https://jax.readthedocs.io/en/latest/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
}

# -- nbsphinx settings ------------------------------------------------------

nbsphinx_execute = 'never'

nbsphinx_prolog = r"""
{% set docname = env.doc2path(env.docname, base=None) %}

.. raw:: html

    <div class="admonition note">
    <p>This page was generated from a Jupyter notebook. You can
    <a href="https://github.com/xshi19/normix/blob/main/{{ docname }}">
    view it on GitHub</a> or download and run it locally.</p>
    </div>
"""
