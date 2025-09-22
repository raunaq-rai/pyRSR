import os
import sys
sys.path.insert(0, os.path.abspath("../.."))  # ensure cosmicdawn/ is visible

project = "cosmicdawn"
author = "Raunaq Rai"
release = "0.1"

extensions = [
    "sphinx.ext.autodoc",        # pull in docstrings
    "sphinx.ext.napoleon",       # support NumPy/Google style docstrings
    "sphinx_autodoc_typehints",  # show type hints inline
    "sphinx.ext.viewcode",       # add [source] links
    "sphinx.ext.mathjax",        # LaTeX equations
    "myst_parser",               # optional: markdown support
]

html_theme = "sphinx_rtd_theme"

