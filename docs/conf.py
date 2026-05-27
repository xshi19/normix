"""
Sphinx configuration file for normix documentation.
"""

import os
import sys

DOCS_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(DOCS_DIR, '..')))

# -- Project information -----------------------------------------------------

project = 'normix'
copyright = '2024–2026, normix developers'
author = 'normix developers'
release = '0.2.2'

# -- General configuration ---------------------------------------------------

extensions = [
    'myst_nb',  # includes myst_parser; do not list myst_parser separately
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'nbsphinx',
]

templates_path = ['_templates']

# myst-nb registers ``.md`` → myst-nb parser; do not map ``.md`` to ``markdown``.
source_suffix = {
    '.rst': 'restructuredtext',
}

exclude_patterns = [
    '_build',
    'Thumbs.db',
    '.DS_Store',
    'pdfs',
    'plans',
    'references',
    'tech_notes',
    'design',
    'notebooks',
    'ARCHITECTURE.md',
    'investigations',
    'reviews',
    'archive',
]

# Enable after docstring cross-reference cleanup (post Phase 1).
nitpicky = False
nitpick_ignore = [
    ('py:class', 'jax.Array'),
    ('py:class', 'jaxtyping.Array'),
    ('py:class', 'Array'),
    ('py:class', 'Optional'),
    ('py:class', 'Union'),
    ('py:class', 'Callable'),
    ('py:class', 'Mapping'),
    ('py:class', 'Sequence'),
    ('py:class', 'Iterable'),
    ('py:class', 'TypeVar'),
    ('py:class', 'Any'),
    ('py:class', 'Literal'),
    ('py:class', 'Protocol'),
    ('py:obj', 'jax.Array'),
]

# -- MyST settings -----------------------------------------------------------

myst_enable_extensions = [
    'dollarmath',
    'amsmath',
    'colon_fence',
    'deflist',
]

myst_heading_anchors = 3

# -- myst-nb settings --------------------------------------------------------

nb_execution_mode = os.environ.get('NB_EXECUTION_MODE', 'cache')
nb_execution_timeout = 900
nb_execution_raise_on_error = True
nb_execution_cache_path = os.path.join(DOCS_DIR, '_build', '.jupyter_cache')
nb_merge_streams = True

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_book_theme'

html_static_path = ['_static']

html_css_files = ['normix.css']

html_theme_options = {
    'repository_url': 'https://github.com/xshi19/normix',
    'use_repository_button': True,
    'use_issues_button': True,
    'home_page_in_toc': True,
    'show_navbar_depth': 2,
    'navigation_with_keys': True,
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

# -- nbsphinx settings (legacy notebooks; removed in Phase 4) ----------------

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
