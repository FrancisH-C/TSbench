#!/usr/bin/env python
# -*- coding: utf-8 -*-
import io
import os

from setuptools import find_packages, setup

try:
    app_dir = os.path.abspath(os.path.dirname(__file__))
    with io.open(os.path.join(app_dir, "README.md"), encoding="utf-8") as f:
        LONG_DESCRIPTION = "\n" + f.read()
except FileNotFoundError:
    LONG_DESCRIPTION = ""

install_requires = ['tqdm',
                    'numpy',
                    'randomgen',
                    'ESRNN',
                    'pandas',
                    'torch',
                    'matplotlib',
                    'statsmodels',
                    'scikit-learn',
                    'sphinx',
                    'furo',
                    'ipywidgets',
                    'sphinx',
                    'furo',
                    'nbsphinx',
                    'ipywidgets']

extras_require = {
    'test': ['pytest'],
    'setup': ['setuptools'],
}

extras_require['noR'] = list(
    set(x for lst in extras_require.values() for x in lst))

extras_require['R'] = ['rpy2[all]']

extras_require['all'] = list(
    set(x for lst in extras_require.values() for x in lst))

VERSION = "0.1.0"
setup(
    name="TSbench",
    version=VERSION,
    description="",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author="Francis Huot-Chantal",
    author_email="",
    url="https://github.com/FrancisH-C/TSbench.git",
    python_requires=">=3.8.0",
    extras_require=extras_require,
    scripts=['setup_R.py'],
    packages=find_packages(),
    install_requires=install_requires,
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Operating System :: OS Independent",
    ],
)
