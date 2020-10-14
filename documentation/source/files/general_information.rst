
General Information
===================

qgs is a Python implementation of an atmospheric model for midlatitudes.  It
models the dynamics of a 2-layer `quasi-geostrophic`_ (QG) channel atmosphere
on a `beta-plane`_, coupled to a simple land or `shallow-water`_ ocean
component.

Statement of need
-----------------

Reduced-order spectral quasi-geostrophic models of the atmosphere with a large number of modes offer a good representations of the dry atmospheric dynamics :cite:`gi-OB1989`.
The dynamics thus obtained allow to identify typical features of the atmospheric circulation, such as blocked and zonal circulation regimes, and low-frequency variability.
However, these models are less often considered in literature than other toy models, despite their demonstration of more realistic behavior.

qgs aims to popularize these systems by providing a fast and easy-to-use Python framework for researchers and teachers to integrate this kind of model.

The choice to use Python was specifically made to facilitate its use in `Jupyter <https://jupyter.org/>`_ notebooks and the multiple recent machine learning libraries that are available in this
language.

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

Contributing to qgs
-------------------

If you want to contribute actively to the roadmap detailed above, please contact directly the authors.

In addition, if you have made changes that you think will be useful to others, please feel free to suggest these as a pull request on the `qgs Github repository <https://github.com/Climdyn/qgs>`_.

A review of your pull request will follow with possibly suggestions of changes before merging it in the master branch.
Please consider the following guidelines before submitting:

* Before submitting a pull request, double check that the branch to be merged contains only changes you wish to add to the master branch. This will save time in reviewing the code.
* For any changes to the core model files, please run the tests found in the folder `model_test <../../../../model_test>`_ to ensure that the model tensors are still valid.
* For substantial additions of code, including a test case in the folder `model_test <../../../../model_test>`_ is recommended.
* Please do not make changes to existing test cases.
* Please document the new functionalities in the documentation. Code addition without documentation addition will not be accepted. The documentation is done with `sphinx`_ and follows the Numpy conventions. Please take a look to the actual code to get an idea about how to document the code.
* If your addition can be considered as a tool not directly related to the core of the model, please develop it in the toolbox folder.
* The team presently maintaining qgs is not working full-time on it, so please be patient as the review of the submission may take some times.

For more information about git, Github and the pull request framework, a good source of information is the `contributing guide <https://mitgcm.readthedocs.io/en/latest/contributing/contributing.html>`_ of the `MITgcm <https://github.com/MITgcm/MITgcm>`_.

Other atmospheric models in Python
----------------------------------

Non-exhaustive list:

* `Q-GCM <http://q-gcm.org/>`_: A mid-latitude grid based ocean-atmosphere model like MAOOAM. Code in Fortran,
  interface is in Python.
* `pyqg <https://github.com/pyqg/pyqg>`_: A pseudo-spectral python solver for quasi-geostrophic systems.
* `Isca <https://execlim.github.io/IscaWebsite/index.html>`_: Research GCM written in Fortran and largely
  configured with Python scripts, with internal coding changes required for non-standard cases.

References
----------

.. bibliography:: model/ref.bib
    :keyprefix: gi-

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
.. _sphinx: https://www.sphinx-doc.org/en/master/
