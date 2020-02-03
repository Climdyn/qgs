
General Information
===================

qgs is a Python 2-layer `quasi-geostrophic`_ channel model
on a `beta-plane`_ for the atmosphere, coupled to a simple land or
`shallow-water`_ ocean component beneath.

Installation
------------

The easiest way to install is through `Anaconda`_.

First install Anaconda and clone the repository: ::

    git clone

Then install and activate the Python3 Anaconda environment: ::

    conda env create -f environment.yml
    conda activate qgs

You can then perform a test by running the script ::

    python qgs_rp.py

to see if everything run smoothly (should take less than a minute).

To build the documentation, please run (with the conda environment still activated): ::

    cd documentation
    make html


You may need to install `make`_ if it is not already present on your system.
Once built, the documentation is available `here <../index.html>`_.

Usage
-----

qgs can be used by editing and running the script found in the `main folder <./>`_: ::

    python qgs_rp.py

or ::

    python qgs_maooam.py

Examples
--------

Another nice way is through the use of Jupyter notebooks.
Simple examples can be found int the `notebooks folder <./notebooks>`_.
For instance, running ::

    conda activate qgs
    cd notebooks
    jupyter-notebook

will lead you to your favorite browser where you can load and run the examples.

Dependencies
------------

qgs needs mainly:

* `Numpy`_ for numeric support
* `Numba`_ for code acceleration

Check the yaml file `environment.yml <./environment.yml>`_ for the full list of dependencies.

Forthcoming developments
------------------------

* Coming soon (mostly technical developments)

  + Inner products sparse representation

* Scientific development (short-to-mid-term developments)

  + Dynamical equilibrium temperature equations
  + Non-autonomous equation (seasonality, etc...)
  + Heat exchange schemes when using land model version
    (using a model derived from MAOOAM in `Li et al. <https://doi.org/10.1007/s13351-018-8012-y>`_)

* Technical mid-term developments

  + Dimensionally robust Parameter class operation
  + Visualisation tools, e.g. based on the `movie-script`_ package

* Long-term development track

  + `Julia`_ core (integrators, parallelization)
  + Quartic :math:`T^4` temperature tendencies
  + Active advection
  + True quasi-geostrophic ocean when using ocean model version
  + Salinity in the ocean

* Very long-term (may never happen)

  + Symbolic inner products (using e.g. `Simpy`_)

    - Arbitrary spatial mode basis of functions
    - Automatic on-the-fly inner product calculation (numeric or analytic if possible)
    - Symbolic PDE equation specification

  + Multi-domain Galerkin expansion with boundary condition solver

Other atmospheric models in Python
----------------------------------

Non-exhaustive list:

* `Q-GCM <http://q-gcm.org/>`_: A mid-latitude grid based ocean-atmosphere model like MAOOAM. Code in Fortran,
  interface is in Python.
* `pyqg <https://github.com/pyqg/pyqg>`_: A pseudo-spectral python solver for quasi-geostrophic systems.
* `Isca <https://execlim.github.io/IscaWebsite/index.html>`_: Research GCM written in Fortran and largely
  configured with Python scripts, with internal coding changes required for non-standard cases.

.. _quasi-geostrophic: https://en.wikipedia.org/wiki/Quasi-geostrophic_equations
.. _shallow-water: https://en.wikipedia.org/wiki/Shallow_water_equations
.. _MAOOAM: https://github.com/Climdyn/MAOOAM
.. _Numba: https://numba.pydata.org/
.. _Numpy: https://numpy.org/
.. _multiprocessing: https://docs.python.org/3.7/library/multiprocessing.html#module-multiprocessing
.. _tangent linear model: http://glossary.ametsoc.org/wiki/Tangent_linear_model
.. _Anaconda: https://www.anaconda.com/
.. _movie-script: https://github.com/jodemaey/movie-script
.. _Julia: https://julialang.org/
.. _Simpy: https://www.sympy.org/
.. _make: https://www.gnu.org/software/make/
.. _beta-plane: https://en.wikipedia.org/wiki/Beta_plane
