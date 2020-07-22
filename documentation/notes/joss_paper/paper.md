---
title: 'qgs: A flexible Python framework of reduced-order multiscale quasi-geopstrophic spectral models'
tags:
  - Python
  - Numba
  - Idealized atmospheric model
  - Coupled model
  - Mid-latitude climate variability
authors:
  - name: Jonathan Demaeyer
    orcid: 0000-0002-5098-404X 
    affiliation: 1
  - name: Lesley De Cruz
    orcid: 0000-0003-4458-8953
    affiliation: 1
  - name: Stéphane Vannitsem 
    orcid: 0000-0002-1734-1042
    affiliation: 1
affiliations:
 - name: Institut Royal Météorologique de Belgique, Avenue Circulaire, 3, 1180 Brussels, Belgium
   index: 1

date: 17 July 2020
bibliography: joss.bib

---

# Summary

`qgs` is a a Python implementation of a set of idealized reduced-order models representing atmospheric mid-latitudes variability. 
It consists of a spectral two-layers quasi-geostrophic atmosphere on a beta-plane, coupled either to a simple land surface or to a shallow-water ocean.

* In the case of coupling with an ocean, it reproduces Modular Arbitrary-Order Ocean-Atmosphere Model (MAOOAM) the model proposed in @DDV2016. In @VDDG2015, this model version was shown to reproduce a 
low-frequency variability (LFV) typical of the ocean-atmosphere coupling. 
This coupling consists in both mechanical and heat exchange interactions between the two components. 
The model has already been used in several context, in particular for data assimilation analyses [@PBBCDSY2019; @TCVB2020], 
and predictability studies [@VSD2019; VD2020]
* In the case of a land surface coupling, it emulates the model proposed in @RP1982 and @CT1987 with a simple thermal relaxation toward a climatological temperature and a mechanical coupling due to the 
friction between the land and the atmosphere. It can also emulate the model proposed in @LHHBD2018, with mechanical coupling and heat exchange. In addition, the number of dynamical spectral modes can be fixed by the users as for the MAOOAM model.

In the `qgs` framework, the partial differential equations (PDEs) ruling the time evolution of its fields are projected on a basis of functions defined on its 
spatial domain. 
This kind of decomposition transforms the PDEs into a set of ordinary differential equations (ODEs) which can then be solved with the usual integration techniques.
Presently in `qgs`, the functions of the basis are chosen amongst the orthogonal Fourier modes compatible with the boundary conditions of each subcomponents of the system, namely the atmosphere and the ocean or the land surface. 
In future developments plan, the user will be able to specify the basis of functions for each component, depending on the specific boundary conditions needed.

The model implementation consists in submodules to set up the model's parameters and compute the model's tensor of tendencies terms, see for details @DDV2016.
This tensor is used by the code to compute the tendencies function and its Jacobian matrix. These functions can then be fed to the `qgs` built-in Runge-Kutta integrator or 
to another integrator implemented by the user. For instance, an example of the usage of the Julia DifferentialEquations.jl [@RN2017] integration package through the Python diffeqpy [@diffeqpy] package is provided.

The model implementation use Numpy [@vCV2011; @O2006] and SciPy [@scipy] for arrays and computations support, as well as Numba [@numba] and sparse [@sparse] to extensively accelerate the tensor products computation used to compute the tendencies.

# Statement of need

Often in atmospheric and climate sciences, researches and developments are first conducted with a simple idealized systems like the Lorenz-$N$ models ($N \in \{63, 84, 96\}$) [@L63; @L84; @L96] which are toy models of atmospheric variability. 
The two first models are heavily truncated systems (3-variable) describing the very large synoptic scale dynamics of the single-component atmosphere, that neglect the interaction with other components of the climate system and with smaller scales.
The third one is based on heuristic assumptions that lead to unrealistic features like spatial anti-correlation. 
Truncated spectral quasi-geostrophic models of the atmosphere offer better representations of the dry atmospheric dynamics. The dynamics hence obtained allow to 
identify typical features of the atmospheric circulation, such as blocked and zonal circulation regimes, low-frequency variability, etc...
However, these latter idealized models are less considered in the literature, despite their demonstration of realistic behavior.

`qgs` aims to popularize these systems by providing a fast and easy-to-use Python framework for researcher and teacher to integrate this kind of models. 
For an efficient handling of the model by users, its documentation is conceived such that the its equations and parameters are explained and linked to the code.
In the future, its development will be done in a modular fashion which allows to connect the atmosphere to various other subsystems and use it with built-in and external toolboxes.

The choice of Python was specifically done to facilitate its use in Jupyter [@jupyter] notebooks and the multiple recent machine learning libraries that are available in this 
language.

# State of the field

Other software might interest the reader in need for an easy-to-use idealized atmospheric model.

* MAOOAM: The Modular arbitrary-order ocean-atmosphere model, a coupled ocean-atmosphere model included in `qgs` [@MAOOAM]. 
          Code available in Lua, Fortran and Python.
* q-gcm: A mid-latitude grid based ocean-atmosphere model like MAOOAM. Code in Fortran, interface is in Python [@qgcm].
* pyqg: A pseudo-spectral python solver for quasi-geostrophic systems [@pyqg].
* Isca: A research General Circulation Model (GCM) written in Fortran and largely
        configurable with Python scripts, with internal coding changes required for non-standard cases [@Isca].

The mechanically coupled atmosphere-land version of `qgs` was used recently to test new ideas using response theory to adapt statistical postprocessing schemes to a model change [@DV2020].
 
# Acknowledgements

This research has been partly supported by EUMETNET (Postprocessing module of the NWP Cooperation Programme).

# References
