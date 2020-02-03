
Quasi-Geostrophic Spectral model (qgs)
======================================

General Information
-------------------

qgs is a Python 2-layer [quasi-geostrophic](https://en.wikipedia.org/wiki/Quasi-geostrophic_equations) channel model 
on a [beta-plane](https://en.wikipedia.org/wiki/Beta_plane) for the atmosphere, coupled to a simple land or 
[shallow-water](https://en.wikipedia.org/wiki/Shallow_water_equations) ocean component beneath.

About
-----

(c) 2020 Jonathan Demaeyer and Lesley De Cruz

Part of the code comes from the Python MAOOAM implementation by Maxime Tondeur and Jonathan Demaeyer.

See [LICENSE.txt](./LICENSE.txt) for license information.

Installation
------------

The easiest way to install is through [Anaconda](https://www.anaconda.com/).

First install Anaconda and clone the repository:

    git clone 

Then install and activate the Python3 Anaconda environment:

    conda env create -f environment.yml
    conda activate qgs

You can then perform a test by running the script

    python qgs_rp.py
    
to see if everything run smoothly (should take less than a minute).

To build the documentation, please run (with the conda environment still activated):

    cd documentation
    make html

You may need to install [make](https://www.gnu.org/software/make/) if it is not already present on your system.
Once built, the documentation is available [here](./documentation/build/html/index.html).

Usage
-----

qgs can be used by editing and running the script `qgs_rp.py` and `qgs_maooam.py` found in the [main folder](./).


Examples
--------

Another nice way is through the use of Jupyter notebooks. 
Simple examples can be found int the [notebooks folder](./notebooks).
For instance, running 

    conda activate qgs
    cd notebooks
    jupyter-notebook
    
will lead you to your favorite browser where you can load and run the examples.

Documentation
-------------

Coming soon...

Dependencies
------------

qgs needs mainly:

   * [Numpy](https://numpy.org/) for numeric support
   * [Numba](https://numba.pydata.org/) for code acceleration
   
Check the yaml file [environment.yml](./environment.yml) for the dependencies.

Forthcoming developments
------------------------

* Coming soon (mostly technical developments)
    + Inner products sparse representation 
* Scientific development (short-to-mid-term developments)
    + Dynamical equilibrium temperature equations
    + Non-autonomous equation (seasonality, etc...)
    + Heat exchange schemes when using land model version 
      (using a model derived from MAOOAM in [Li et al.](https://doi.org/10.1007/s13351-018-8012-y))
* Technical mid-term developments
    + Dimensionally robust Parameter class operation
    + Visualisation tools, e.g. based on the [movie-script](https://github.com/jodemaey/movie-script) package
* Long-term development track
    + [Julia](https://julialang.org/) core (integrators, parallelization)
    + Quartic T⁴ temperature tendencies
    + Active advection
    + True quasi-geostrophic ocean when using ocean model version
    + Salinity in the ocean
* Very long-term (may never happen)
    + Symbolic inner products (using e.g. [Simpy](https://www.sympy.org/))
        - Arbitrary spatial mode basis of functions
        - Automatic on-the-fly inner product calculation (numeric or analytic if possible)
        - Symbolic PDE equation specification
    + Multi-domain Galerkin expansion with boundary condition solver

Other atmospheric models in Python
----------------------------------

Non-exhaustive list:

* [Q-GCM](http://q-gcm.org/): A mid-latitude grid based ocean-atmosphere model like MAOOAM. Code in Fortran,
                                interface is in Python.
* [pyqg](https://github.com/pyqg/pyqg): A pseudo-spectral python solver for quasi-geostrophic systems.
* [Isca](https://execlim.github.io/IscaWebsite/index.html): Research GCM written in Fortran and largely
            configured with Python scripts, with internal coding changes required for non-standard cases.
            
            
