Getting Started
===============

Installation
------------

DRAGG is available on PyPI and can be downloaded via pip.

.. code-block:: console

   $ pip install dragg

Dependencies
^^^^^^^^^^^^

DRAGG utilizes the Redis database to communicate between individual HEMS and the Aggregator. Therefore Redis is necessary to run DRAGG. While Redis is only available for Unix based operating systems we have included a `Dockerfile` in `/deploy` for Windows installations. 

Running a Simulation
^^^^^^^^^^^^^^^^^^^^

1. Configure your simulation via `data/config.toml`
2. Run the 

