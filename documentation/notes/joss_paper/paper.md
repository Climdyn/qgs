---
title: 'qgs: A flexible Python framework of reduced-order multiscale climate models'
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
 - name: Royal Meteorological Institute of Belgium, Avenue Circulaire, 3, 1180 Brussels, Belgium
   index: 1

date: 17 July 2020
bibliography: joss.bib

---

# Summary

`qgs` is a a Python implementation of a set of idealized reduced-order models representing atmospheric mid-latitude variability. 
It consists of a spectral two-layer quasi-geostrophic (hence the name`qgs`) atmosphere on a beta-plane, coupled either to a simple land surface or to a shallow-water ocean.

* In the case where it is coupled to an ocean, it reproduces the Modular Arbitrary-Order Ocean-Atmosphere Model (MAOOAM), described in @DDV2016. In @VDDG2015, a 36-variable configuration of this model was shown to reproduce a 
low-frequency variability (LFV) typical of the coupled ocean-atmosphere system. 
This coupling consists in both mechanical and heat exchange interactions between the two components. 
The model has already been used in several different contexts, in particular for data assimilation analyses [@PBBCDSY2019; @TCVB2020], 
and predictability studies [@VSD2019; @VD2020]
* In the case of a land surface coupling, it emulates the model proposed in @RP1982 and @CT1987 with a simple thermal relaxation toward a climatological temperature and a mechanical coupling due to the 
friction between the land and the atmosphere. It can also emulate the model proposed in @LHHBD2018, with mechanical coupling and heat exchange. In addition, the number of dynamical spectral modes can be configured by the user, as is the case for the MAOOAM model.

In the `qgs` framework, the partial differential equations (PDEs) that govern the time evolution of its fields are projected on a basis of functions defined on its 
spatial domain. 
This kind of decomposition transforms the PDEs into a set of ordinary differential equations (ODEs) which can then be solved with the usual integration techniques.
Presently in `qgs`, the functions of the basis are chosen amongst the orthogonal Fourier modes compatible with the boundary conditions of each subcomponent of the system, namely the atmosphere and the ocean or the land surface. 
A future development is planned that will enable the user to specify the basis of functions for each component, depending on the required boundary conditions.

The model implementation consists of submodules to set up the model's parameters and to compute the tensor that defines the coefficients in the tendencies of the model variables; more details can be found in @DDV2016.
This tensor is used by the code to compute the tendencies function and its Jacobian matrix. These functions can then be fed to the `qgs` built-in Runge-Kutta integrator or 
to another integrator implemented by the user. As an example, the usage of the Julia DifferentialEquations.jl [@RN2017] integration package through the Python diffeqpy [@diffeqpy] package is provided.
Technical details about this implementation can be found in the *Code Description* section of the included documentation.

The model implementation uses Numpy [@vCV2011; @O2006] and SciPy [@scipy] for arrays and computations support, as well as Numba [@numba] and sparse [@sparse] to considerably accelerate the tensor products computation used to compute the tendencies.

# Statement of need

In atmospheric and climate sciences, research and development is often first conducted with a simple idealized system like the Lorenz-$N$ models ($N \in \{63, 84, 96\}$) [@L63; @L84; @L96] which are toy models of atmospheric variability. 
The first two models are heavily truncated systems (3-variable) describing the very large synoptic-scale dynamics of the single-component atmosphere, that neglect the interaction with other components of the climate system and with smaller scales.
The third one is based on heuristic assumptions that lead to unrealistic features like spatial anti-correlation. 

Truncated spectral quasi-geostrophic models of the atmosphere offer better representations of the dry atmospheric dynamics [@V2017]. The dynamics thus obtained allow to 
identify typical features of the atmospheric circulation, such as blocked and zonal circulation regimes, and low-frequency variability.
However, these models are less often considered in literature, despite their demonstration of realistic behavior.

`qgs` aims to popularize these systems by providing a fast and easy-to-use Python framework for researchers and teachers to integrate this kind of model. 
For an efficient handling of the model by users, its documentation is conceived such that its equations and parameters are explained and linked to the code.
In the future, its development will be done in a modular fashion which allows to connect the atmosphere to various other subsystems and use it with built-in and external toolboxes.

The choice to use Python was specifically made to facilitate its use in Jupyter [@jupyter] notebooks and the multiple recent machine learning libraries that are available in this 
language.

# State of the field

Other software might interest the reader in need for an easy-to-use idealized atmospheric model.

* MAOOAM: The Modular Arbitrary-Order Ocean-Atmosphere Model, a coupled ocean-atmosphere model included in `qgs` [@MAOOAM]. 
          Code available in Lua, Fortran and Python.
* q-gcm: A mid-latitude grid-based ocean-atmosphere model like MAOOAM. Code in Fortran, interface in Python [@qgcm].
* pyqg: A pseudo-spectral Python solver for quasi-geostrophic systems [@pyqg].
* Isca: A research General Circulation Model (GCM) written in Fortran and largely
        configurable with Python scripts, with internal coding changes required for non-standard cases [@Isca].

The mechanically coupled atmosphere-land version of `qgs` was recently used to test new ideas using response theory to adapt statistical postprocessing schemes to a model change [@DV2020].
 
# Acknowledgements

This research has been partly supported by EUMETNET (Postprocessing module of the NWP Cooperation Programme).

# References
