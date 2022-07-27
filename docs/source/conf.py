# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information


import sys, os

sys.path.insert(0, os.path.abspath('../../'))


# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

master_doc = "index"
source_suffix = ['.rst', '.md']

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", ".ipynb_checkpoints"]

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

project = 'aggmap'
copyright = '2022, Shen Wan Xiang'
author = 'Shen Wan Xiang'
release = '1.1.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    'sphinx.ext.napoleon',
    #    'bokeh.sphinxext.bokeh_plot',
    #"sphinx_gallery.gen_gallery",
]

#extensions = ['sphinx.ext.autodoc', 'sphinx.ext.napoleon']



# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    "python": ("https://docs.python.org/{.major}".format(sys.version_info), None),
    "numpy": ("https://docs.scipy.org/doc/numpy/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference", None),
    "matplotlib": ("https://matplotlib.org/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "sklearn": ("http://scikit-learn.org/stable/", None),
    "bokeh": ("http://bokeh.pydata.org/en/latest/", None),
}



sphinx_gallery_conf = {
    # path to your examples scripts
    "examples_dirs": "../examples",
    "ignore_pattern": r"(.*torus.*|inverse_transform.*)\.py",
    # path where to save gallery generated examples
    "gallery_dirs": "auto_examples",
    "plot_gallery": False,  # Turn off running the examples for now
    "reference_url": {
        "python": "https://docs.python.org/{.major}".format(sys.version_info),
        "numpy": "https://docs.scipy.org/doc/numpy/",
        "scipy": "https://docs.scipy.org/doc/scipy/reference",
        "matplotlib": "https://matplotlib.org/",
        "pandas": "https://pandas.pydata.org/pandas-docs/stable/",
        "sklearn": "http://scikit-learn.org/stable/",
        "bokeh": "http://bokeh.pydata.org/en/latest/",
    },
}


templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output


html_theme = "sphinx_rtd_theme"
#html_theme = 'alabaster'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
