import os
from setuptools import setup, find_packages

setup(name='dragg',
      license='MIT',
      version='1.1',
      author='Aisling Pigott and Cory Mosiman',
      author_email='aisling.pigott@colorado.edu',
      packages=find_packages(),
      scripts=["dragg/main.py"],
      install_requires=[
        'redis',
        'pathos',
        'cvxpy',
        'numpy',
        'collections',
        'itertools',
        'pandas',
        'prettytable',
        'datetime',
        'plotly',
        'sys',
        'toml',
        'random',
        'names'],
      py_modules=['dragg'],
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        ],
     )
