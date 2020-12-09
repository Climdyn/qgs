
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

The easiest way to run qgs is to use an appropriate environment created through [Anaconda](https://www.anaconda.com/).

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

qgs can be used by editing and running the script `qgs_rp.py` and `qgs_maooam.py` found in the main folder.


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
    + Symbolic inner products (using e.g. [Sympy](https://www.sympy.org/))
        - Arbitrary spatial mode basis of functions
        - Automatic on-the-fly inner product calculation (numeric or analytic if possible)
        - Symbolic PDE equation specification
    + Visualisation tools, e.g. based on the [movie-script](https://github.com/jodemaey/movie-script) package
* Long-term development track
    + Active advection
    + True quasi-geostrophic ocean when using ocean model version
    + Salinity in the ocean
    
Contributing to qgs
-------------------

If you want to contribute actively to the roadmap detailed above, please contact the authors.

In addition, if you have made changes that you think will be useful to others, please feel free to suggest these as a pull request on the [qgs Github repository](https://github.com/Climdyn/qgs).

A review of your pull request will follow with possibly suggestions of changes before merging it in the master branch.
Please consider the following guidelines before submitting:
* Before submitting a pull request, double check that the branch to be merged contains only changes you wish to add to the master branch. This will save time in reviewing the code.
* For any changes to the core model files, please run the tests found in the folder [model_test](./model_test) to ensure that the model tensors are still valid.
* For substantial additions of code, including a test case in the folder [model_test](./model_test) is recommended.
* Please do not make changes to existing test cases.
* Please document the new functionalities in the documentation. Code addition without documentation addition will not be accepted. 
The documentation is done with [sphinx](https://www.sphinx-doc.org/en/master/) and follows the Numpy conventions. Please take a look to the actual code to get an idea about how to document the code.
* If your addition can be considered as a tool not directly related to the core of the model, please develop it in the toolbox folder.
* The team presently maintaining qgs is not working full-time on it, so please be patient as the review of the submission may take some time.

For more information about git, Github and the pull request framework, a good source of information is the [contributing guide](https://mitgcm.readthedocs.io/en/latest/contributing/contributing.html) of the [MITgcm](https://github.com/MITgcm/MITgcm).


Other atmospheric models in Python
----------------------------------

Non-exhaustive list:

* [Q-GCM](http://q-gcm.org/): A mid-latitude grid based ocean-atmosphere model like MAOOAM. Code in Fortran,
                                interface is in Python.
* [pyqg](https://github.com/pyqg/pyqg): A pseudo-spectral python solver for quasi-geostrophic systems.
* [Isca](https://execlim.github.io/IscaWebsite/index.html): Research GCM written in Fortran and largely
            configured with Python scripts, with internal coding changes required for non-standard cases.
            
            
