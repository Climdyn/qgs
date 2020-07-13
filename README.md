
Quasi-Geostrophic Spectral model (qgs)
======================================

[![DOI](https://zenodo.org/badge/246609584.svg)](https://zenodo.org/badge/latestdoi/246609584)
[![Documentation Status](https://readthedocs.org/projects/qgs/badge/?version=latest)](https://qgs.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

General Information
-------------------

qgs is a Python implementation of an atmospheric model for midlatitudes.  It models the dynamics of
a 2-layer [quasi-geostrophic](https://en.wikipedia.org/wiki/Quasi-geostrophic_equations) channel
atmosphere on a [beta-plane](https://en.wikipedia.org/wiki/Beta_plane), coupled to a simple land or
[shallow-water](https://en.wikipedia.org/wiki/Shallow_water_equations) ocean component. 

About
-----

(c) 2020 Jonathan Demaeyer and Lesley De Cruz

Part of the code comes from the Python [MAOOAM](https://github.com/Climdyn/MAOOAM) implementation by Maxime Tondeur and Jonathan Demaeyer.

See [LICENSE.txt](./LICENSE.txt) for license information.

Installation
------------

> **__Note:__** qgs is presently compatible with Linux and Mac OS.
> **It is not compatible with Windows for the moment**, but a Windows compatible version will be released soon.

The easiest way to install is through [Anaconda](https://www.anaconda.com/).

First install Anaconda and clone the repository:

    git clone https://github.com/Climdyn/qgs.git

Then install and activate the Python3 Anaconda environment:

    conda env create -f environment.yml
    conda activate qgs

You can then perform a test by running the script

    python qgs_rp.py
    
to see if everything runs smoothly (this should take less than a minute).

#### Activating DifferentialEquations.jl optional support

In addition to the qgs builtin Runge-Kutta integrator, the qgs model can alternatively be integrated with a package called [DifferentialEquations.jl](https://github.com/SciML/DifferentialEquations.jl) written in [Julia](https://julialang.org/), and available through the 
[diffeqpy](https://github.com/SciML/diffeqpy) python package.
The diffeqpy package first installation step is done by Anaconda in the qgs environment but then you must [install Julia](https://julialang.org/downloads/) and follow the final manual installation instruction found in the [diffeqpy README](https://github.com/SciML/diffeqpy).

These can be summed up as opening a terminal and doing:
```
conda activate qgs
python
```
and then inside the Python command line interface do:

```
>>> import diffeqpy
>>> diffeqpy.install()
```
which will then finalize the installation. An example of a notebook using this package is available in the documentation and on [readthedocs](https://qgs.readthedocs.io/en/latest/files/examples/diffeq.html).


Documentation
-------------

To build the documentation, please run (with the conda environment activated):

    cd documentation
    make html

You may need to install [make](https://www.gnu.org/software/make/) if it is not already present on your system.
Once built, the documentation is available [here](./documentation/build/html/index.html).

The documentation is also available online on read the docs: [https://qgs.readthedocs.io/](https://qgs.readthedocs.io/)

Usage
-----

qgs can be used by editing and running the script `qgs_rp.py` and `qgs_maooam.py` found in the [main folder](./).


Examples
--------

Another nice way to run the model is through the use of Jupyter notebooks. 
Simple examples can be found in the [notebooks folder](./notebooks).
For instance, running 

    conda activate qgs
    cd notebooks
    jupyter-notebook
    
will lead you to your favorite browser where you can load and run the examples.

Dependencies
------------

qgs needs mainly:

   * [Numpy](https://numpy.org/) for numeric support
   * [sparse](https://sparse.pydata.org/) for sparse multidimensional arrays support
   * [Numba](https://numba.pydata.org/) for code acceleration
   
Check the yaml file [environment.yml](./environment.yml) for the dependencies.

Forthcoming developments
------------------------

* Scientific development (short-to-mid-term developments)
    + Dynamical equilibrium temperature equations
    + Non-autonomous equation (seasonality, etc...)
    + Quartic T⁴ temperature tendencies
* Technical mid-term developments
    + Dimensionally robust Parameter class operation
    + Windows OS support
    + Symbolic inner products (using e.g. [Simpy](https://www.sympy.org/))
        - Arbitrary spatial mode basis of functions
        - Automatic on-the-fly inner product calculation (numeric or analytic if possible)
        - Symbolic PDE equation specification
    + Visualisation tools, e.g. based on the [movie-script](https://github.com/jodemaey/movie-script) package
* Long-term development track
    + Active advection
    + True quasi-geostrophic ocean when using ocean model version
    + Salinity in the ocean

Other atmospheric models in Python
----------------------------------

Non-exhaustive list:

* [Q-GCM](http://q-gcm.org/): A mid-latitude grid based ocean-atmosphere model like MAOOAM. Code in Fortran,
                                interface is in Python.
* [pyqg](https://github.com/pyqg/pyqg): A pseudo-spectral python solver for quasi-geostrophic systems.
* [Isca](https://execlim.github.io/IscaWebsite/index.html): Research GCM written in Fortran and largely
            configured with Python scripts, with internal coding changes required for non-standard cases.
            
            
