
Quasi-Geostrophic Spectral model (qgs)
======================================


[![PyPI version](https://badge.fury.io/py/qgs.svg)](https://badge.fury.io/py/qgs)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/qgs.svg)](https://pypi.org/project/qgs/)
[![DOI](https://zenodo.org/badge/246609584.svg)](https://zenodo.org/badge/latestdoi/246609584)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.02597/status.svg)](https://doi.org/10.21105/joss.02597)
[![Documentation Status](https://readthedocs.org/projects/qgs/badge/?version=latest)](https://qgs.readthedocs.io/en/latest/?badge=latest)
[![tests](https://github.com/Climdyn/qgs/actions/workflows/checks.yml/badge.svg?branch=master)](https://github.com/Climdyn/qgs/actions/workflows/checks.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

General Information
-------------------

qgs is a Python implementation of an atmospheric model for midlatitudes.  It models the dynamics of
a 2-layer [quasi-geostrophic](https://en.wikipedia.org/wiki/Quasi-geostrophic_equations) channel
atmosphere on a [beta-plane](https://en.wikipedia.org/wiki/Beta_plane), coupled to a simple land or
[shallow-water](https://en.wikipedia.org/wiki/Shallow_water_equations) ocean component. 

![](https://github.com/Climdyn/qgs/blob/master/misc/figs/readme.gif?raw=true)

> **You can try qgs online !**  
> Simply click on one of the following links to access an introductory tutorial:
> [![Open in colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Climdyn/qgs/blob/master/notebooks/introduction_qgs.ipynb)
> [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Climdyn/qgs/master?filepath=notebooks/introduction_qgs.ipynb)
> [<img src="https://deepnote.com/buttons/launch-in-deepnote-small.svg">](https://deepnote.com/launch?name=MyProject&url=https://github.com/Climdyn/qgs/tree/master/notebooks/introduction_qgs.ipynb)


About
-----

(c) 2020-2025 qgs Developers and Contributors

Part of the code originates from the Python [MAOOAM](https://github.com/Climdyn/MAOOAM) implementation by Maxime Tondeur and Jonathan Demaeyer.

See [LICENSE.txt](https://raw.githubusercontent.com/Climdyn/qgs/master/LICENSE.txt) for license information.

**Please cite the code description article if you use (a part of) this software for a publication:**

* Demaeyer J., De Cruz, L. and Vannitsem, S. , (2020). qgs: A flexible Python framework of reduced-order multiscale climate models. 
  *Journal of Open Source Software*, **5**(56), 2597,   [https://doi.org/10.21105/joss.02597](https://doi.org/10.21105/joss.02597).

Please consult the qgs [code repository](http://www.github.com/Climdyn/qgs) for updates.


Installation
------------

#### With pip

The easiest way to install and run qgs is to use [pip](https://pypi.org/).
Type in a terminal

    pip install qgs

and you are set!

Additionally, you can clone the repository

    git clone https://github.com/Climdyn/qgs.git

and perform a test by running the script

    python qgs/qgs_rp.py

to see if everything runs smoothly (this should take less than a minute).

> **Note:** 
> With the pip installation, in order to be able to generate the movies with the diagnostics, 
> you need to install separately [ffmpeg](https://ffmpeg.org/).

#### With Anaconda

The second easiest way to install and run qgs is to use an appropriate environment created through [Anaconda](https://www.anaconda.com/).

First install Anaconda and clone the repository:

    git clone https://github.com/Climdyn/qgs.git

Then install and activate the Python3 Anaconda environment:

    conda env create -f environment.yml
    conda activate qgs

You can then perform a test by running the script

    python qgs_rp.py
    
to see if everything runs smoothly (this should take less than a minute).

#### Note for Windows and MacOS users

Presently, qgs is compatible with Windows and MacOS but users wanting to use qgs inside their Python scripts must guard the main script with a

```
if __name__ == "__main__":
```

clause and add the following lines below

```
  from multiprocessing import freeze_support
  freeze_support()
```

About this usage, see for example the main scripts `qgs_rp.py` and `qgs_maooam.py` in the root folder.
Note that the Jupyter notebooks are not concerned by this recommendation and work perfectly well on both operating systems.

> **Why?** These lines are required to make the multiprocessing library works with these operating systems. See [here](https://docs.python.org/3.8/library/multiprocessing.html) for more details, 
> and in particular [this section](https://docs.python.org/3.8/library/multiprocessing.html#the-spawn-and-forkserver-start-methods).


#### Activating DifferentialEquations.jl optional support

In addition to the qgs builtin Runge-Kutta integrator, the qgs model can alternatively be integrated with a package called [DifferentialEquations.jl](https://github.com/SciML/DifferentialEquations.jl) written in [Julia](https://julialang.org/), and available through the
[diffeqpy](https://github.com/SciML/diffeqpy) Python package.
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

For more advanced usages, please read the [User Guides](https://qgs.readthedocs.io/en/latest/files/user_guide.html).

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
   * [Sympy](https://www.sympy.org/) for symbolic manipulation of inner products
   
Check the yaml file [environment.yml](https://raw.githubusercontent.com/Climdyn/qgs/master/environment.yml) for the dependencies.

Forthcoming developments
------------------------

* Scientific development (short-to-mid-term developments)
    + Non-autonomous equation (seasonality, etc...)
    + Energy diagnostics
* Technical midterm developments
    + Vectorization of the tensor computation
* Long-term development track
    + Active advection
    + True quasi-geostrophic ocean when using ocean model version
    + Salinity in the ocean 
    + Numerical basis of function
  
Contributing to qgs
-------------------

If you want to contribute actively to the roadmap detailed above, please contact the main authors.

In addition, if you have made changes that you think will be useful to others, please feel free to suggest these as a pull request on the [qgs Github repository](https://github.com/Climdyn/qgs).

More information and guidance about how to do a pull request for qgs can be found in the documentation [here](https://qgs.readthedocs.io/en/latest/files/general_information.html#contributing-to-qgs).

Other atmospheric models in Python
----------------------------------

Non-exhaustive list:

* [Q-GCM](http://q-gcm.org/): A mid-latitude grid based ocean-atmosphere model like MAOOAM. Code in Fortran,
                                interface is in Python.
* [pyqg](https://github.com/pyqg/pyqg): A pseudo-spectral Python solver for quasi-geostrophic systems.
* [Isca](https://execlim.github.io/IscaWebsite/index.html): Research GCM written in Fortran and largely
            configured with Python scripts, with internal coding changes required for non-standard cases.
            
            
