
General Information
===================

qgs is a Python implementation of an atmospheric model for midlatitudes.  It
models the dynamics of a 2-layer `quasi-geostrophic`_ (QG) channel atmosphere
on a `beta-plane`_, coupled to a simple land or `shallow-water`_ ocean
component. 

Installation
------------

.. note::

    qgs is presently compatible with Linux and Mac OS.

    **It is not compatible with Windows for the moment**, but a Windows compatible version will be released soon.

The easiest way to install is through `Anaconda`_.

First install Anaconda and clone the repository: ::

    git clone https://github.com/Climdyn/qgs.git

Then install and activate the Python3 Anaconda environment: ::

    conda env create -f environment.yml
    conda activate qgs

You can then perform a test by running the script ::

    python qgs_rp.py

to see if everything runs smoothly (this should take less than a minute).

Activating DifferentialEquations.jl optional support
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In addition to the qgs builtin Runge-Kutta integrator, the qgs model can alternatively be integrated with a package called `DifferentialEquations.jl <https://github.com/SciML/DifferentialEquations.jl>`_ written in `Julia <https://julialang.org/>`_, and available through the
`diffeqpy <https://github.com/SciML/diffeqpy>`_ python package.
The diffeqpy package first installation step is done by Anaconda in the qgs environment but then you must `install Julia <https://julialang.org/downloads/>`_ and follow the final manual installation instruction found in the `diffeqpy README <https://github.com/SciML/diffeqpy>`_.

These can be summed up as opening a terminal and doing: ::

    conda activate qgs
    python

and then inside the Python command line interface do: ::

    >>> import diffeqpy
    >>> diffeqpy.install()

which will then finalize the installation. An :ref:`files/examples/diffeq:Example of DiffEqPy usage` notebook using this package is available in the documentation.

Documentation
-------------

To build the documentation, please run (with the conda environment activated): ::

    cd documentation
    make html


You may need to install `make`_ if it is not already present on your system.
Once built, the documentation is available `here <../index.html>`_.

Usage
-----

qgs can be used by editing and running the script found in the main folder: ::

    python qgs_rp.py

or ::

    python qgs_maooam.py

Examples
--------

Another nice way to run the model is through the use of Jupyter notebooks.
Simple examples can be found in the `notebooks folder <../../../../notebooks>`_.
For instance, running ::

    conda activate qgs
    cd notebooks
    jupyter-notebook

will lead you to your favorite browser where you can load and run the examples.

Dependencies
------------

qgs needs mainly:

* `Numpy`_ for numeric support
* `sparse`_ for sparse multidimensional arrays support
* `Numba`_ for code acceleration

Check the yaml file `environment.yml <../../../../environment.yml>`_ for the full list of dependencies.

Forthcoming developments
------------------------

* Scientific development (short-to-mid-term developments)

    + Dynamical equilibrium temperature equations
    + Non-autonomous equation (seasonality, etc...)
    + Quartic T⁴ temperature tendencies

* Technical mid-term developments

    + Dimensionally robust Parameter class operation
    + Windows OS support
    + Symbolic inner products (using e.g. `Sympy`_)

        - Arbitrary spatial mode basis of functions
        - Automatic on-the-fly inner product calculation (numeric or analytic if possible)
        - Symbolic PDE equation specification

    + Visualisation tools, e.g. based on the `movie-script`_ package

* Long-term development track

    + Active advection
    + True quasi-geostrophic ocean when using ocean model version
    + Salinity in the ocean

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
.. _Sympy: https://www.sympy.org/
.. _make: https://www.gnu.org/software/make/
.. _beta-plane: https://en.wikipedia.org/wiki/Beta_plane
.. _sparse: https://sparse.pydata.org/
