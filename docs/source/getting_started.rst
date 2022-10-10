Getting Started
===============

Installation
------------

DRAGG is available on PyPI and can be downloaded via pip.

.. code-block:: console

   $ pip install dragg

Dependencies
^^^

DRAGG utilizes the Redis database to communicate between individual HEMS and the Aggregator. Therefore Redis is necessary to run DRAGG. While Redis is only available for Linux based operating systems we have included a Dockerfile in /deploy for Windows installations. 