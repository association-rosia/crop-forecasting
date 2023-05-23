import os
from os.path import join
import sys

curdir = os.path.abspath(os.curdir)
sys.path.insert(0, join(curdir, os.pardir, os.pardir))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Crop forecasting'
copyright = '2023, Baptiste URGELL, Louis REBERGA'
author = 'Baptiste URGELL, Louis REBERGA'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc']

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'karma_sphinx_theme'
html_static_path = ['_static']


rst_prolog = """
.. |project_name| replace:: Crop forecasting
.. image:: _static/banner.jpg
"""
