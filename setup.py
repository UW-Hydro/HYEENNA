#!/usr/bin/env python

try:
    from setuptools import setup
except:
    from distutils.core import setup

setup(name='hyeenna',
      version='0.0.1',
      description='Hydrologic Entropy Estimators based on Nearest Neighbor Approximations',
      url='https://github.com/arbennett/HYEENNA',
      author='Andrew Bennett',
      author_email='bennett.andr@gmail.com',
      packages=['hyeenna'],
      install_requires=['xarray', 'scikit-learn', 'numpy',
                        'pandas', 'joblib', 'ipython'],
      keywords=['hydrology', 'climate', 'information theory', 'statistics'],
      tests_require=['pytest'],)
