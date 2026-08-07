"""
Sphinx configuration file for normix documentation.
"""

import os
import sys
from importlib.metadata import version as _pkg_version

DOCS_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(DOCS_DIR, '..')))

# -- Project information -----------------------------------------------------

project = 'normix'
copyright = '2024–2026, normix developers'
author = 'normix developers'
# Derived from package metadata so the site header can never drift from the
# installed version (docs.yml runs `uv sync`, so normix is always installed).
release = _pkg_version('normix')
version = '.'.join(release.split('.')[:2])

# -- General configuration ---------------------------------------------------

extensions = [
    'myst_nb',  # includes myst_parser; do not list myst_parser separately
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx_design',
    'sphinx_togglebutton',
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
]

# Fail unresolved cross-refs loudly (review roadmap W7). Content pages use
# fully-qualified {py:*} / {doc} / {ref} roles; the allowlist covers autodoc
# noise (Napoleon type prose, private helpers, optional plotting deps) and a
# few external symbols whose inventories omit the name we cite.
nitpicky = True
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
    # Private mixin used in return annotations; not a public API page.
    ('py:class', 'normix.mixtures.marginal._UnivariateNormalMixtureMixin'),
    ('py:meth', 'normix.mixtures.marginal._UnivariateNormalMixtureMixin.ppf'),
    # Optional plotting extra (matplotlib not always inventoriable).
    ('py:class', 'matplotlib.figure.Figure'),
    ('py:class', 'matplotlib.axes._axes.Axes'),
    # External / inventory gaps.
    ('py:func', 'scipy.special.gammaincinv'),
    ('py:func', 'numpy.ndarray.ravel'),
    ('py:func', 'jax.custom_jvp'),
    ('py:func', 'jnp.where'),
]
# Napoleon often lexes Parameters prose as types; private helpers appear in
# inherited docstrings. Regex keeps the allowlist small while still failing
# on wrong public module paths (the W1 class of bug).
nitpick_ignore_regex = [
    (r'py:.*', r'.*\._[A-Za-z].*'),          # private members
    (r'py:class', r'optional|callable|scalar|array|samples|dist|dicts?|scipy|ls|color\}|scipy\}|x_plot|1-D.*|n$|data array|dicts with keys.*|\{.*|\}|θ|→|−|K$|JIT-able|convex|convergence|iteration|initial|target|stats|used|must|and|not|e\.g\.|method=.*|\'.*\''),
    (r'py:meth', r'^(log_prob|ppf|fit|from_expectation|quantile_table|to_gig|__call__)$'),
    (r'py:meth', r'^[A-Za-z]+\.from_expectation$'),  # short class.method in docstrings
    (r'py:meth', r'^JointNormalMixture\.'),
    (r'py:class', r'^(Gamma|InverseGamma|InverseGaussian|VarianceGamma|NormalInverseGamma|NormalInverseGaussian|GeneralizedHyperbolic|JointGeneralizedHyperbolic|JointVarianceGamma|JointNormalInverseGamma|JointNormalInverseGaussian|JointNormalMixture|NormalMixture|NormalMixtureEta|FactorNormalMixture|FactorMixtureStats|EMResult|RiskMeasure)$'),
    (r'py:class', r'^normix\.Univariate'),   # top-level re-export path
    (r'py:data', r'^normix\.utils\.constants\.'),
    (r'py:attr', r'^_'),
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
    'github_version': 'master',
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
    # Canonical host moved off jax.readthedocs.io (inventory redirects).
    'jax': ('https://docs.jax.dev/en/latest/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'equinox': ('https://docs.kidger.site/equinox/', None),
}

# linkcheck: GitHub auto-redirects /issues/N → /pull/N for PRs; treat as OK.
linkcheck_allowed_redirects = {
    r'https://github\.com/xshi19/normix/issues/\d+':
        r'https://github\.com/xshi19/normix/pull/\d+',
}

# myst-nb hide-input / hide-output prompts (sphinx-togglebutton)
nb_code_prompt_show = 'Show code cell {type}'
nb_code_prompt_hide = 'Hide code cell {type}'
