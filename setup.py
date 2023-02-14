import os
from setuptools import setup, find_packages

setup(name='dragg',
      license='MIT',
      version='2.3.0',
      author='Aisling Pigott and Cory Mosiman',
      author_email='aisling.pigott@colorado.edu',
      packages=find_packages(),
      scripts=[],
      install_requires=[
        'redis',
        'pathos',
        'cvxpy',
        'numpy',
        'pandas',
        'prettytable',
        'plotly',
        'toml',
        'names'],
      py_modules=['dragg'],
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        ],
     )
