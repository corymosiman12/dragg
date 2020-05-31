import os

with open('./requirements.txt') as f:
    required = f.read().splitlines()

from setuptools import setup, find_packages

setup(name='dragg',
      license='MIT',
      version='0',
      packages=find_packages(),
      scripts=["dragg/main.py"],
      install_requires=required,
     )
