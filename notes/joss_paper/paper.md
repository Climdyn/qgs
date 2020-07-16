---
title: 'qgs: A quasi-geostrophic spectral model'
tags:
  - Python
  - Numba
  - Idealized atmospheric model
  - Coupled model
  - Mid-latitude climate variability
authors:
  - name: Jonathan Demaeyer^[Corresponding author.]
    orcid: 0000-0002-5098-404X 
    affiliation: "1, 2"
  - name: Lesley De Cruz
    orcid: 0000-0003-4458-8953
    affiliation: 1
  - name: Stéphane Vannitsem 
    orcid: 0000-0002-1734-1042
    affiliation: "1, 2"
affiliations:
 - name: Institut Royal Météorologique de Belgique, Avenue Circulaire, 3, 1180 Brussels, Belgium
   index: 1
 - name: European Meteorological Network (EUMETNET), Avenue Circulaire, 3, 1180 Brussels, Belgium
   index: 2 

date: 16 July 2020
bibliography: joss.bib

---

# Summary

`qgs` is a a Python implementation of an idealized atmospheric model which displays a typical mid-latitudes variability. 
It consists in a spectral 2-layer quasi-geostrophic atmosphere on a beta-plane, coupled to a simple land or shallow-water ocean component.
* In the case of an ocean component, it reproduces the model MAOOAM proposed in [@DDV2016]. In [@VDDG2015], this model version was shown to reproduce a 
low-frequency variability (LFV) typical of the ocean-atmosphere coupling. 
This coupling consists in both mechanical and heat exchange interactions between the two components.
* In the case of a land component, it can reproduce the model proposed in [@RP1982] and [@CT1987] with a mechanical coupling due to the 
friction between the land and the atmosphere. It can also reproduce the model proposed in [@LHHBD2018], with mechanical coupling and heat exchange.

`qgs` being a spectral model means that the partial differential equations (PDEs) ruling the time evolution of its fields are decomposed on a basis of functions defined on its 
spatial domain. 
This kind of decomposition transforms the PDEs into a set of ordinary differential equations (ODEs) which can then be solved with the usual integration techniques.
Presently in `qgs`, the functions of the basis are chosen amongst the orthogonal eigenfunctions of the Laplacian operator, but 
according to the future developments plan, the user will be able to specify the basis of functions for each component, allowing to specify its boundary conditions.

The model implementation consists in submodules to set up the model's parameters and compute the model's tensor of tendencies terms.
This tensor is used by the code to compute the tendencies function and its Jacobian matrix. These functions can then be fed to the `qgs` built-in Runge-Kutta integrator or 
to external one. For instance, an example of the diffeqpy integration package usage is provided.

The model implementation use Numpy [@vCV2011; O2006] and SciPy [@scipy] for arrays and computations support, as well as Numba [@numba] and sparse [@sparse] to extensively accelerate the tensor products computation used to compute the tendencies.

# Statement of need

Frequently in geoscience, methods are tested and researchs are conducted with a Lorenz-$N$ model ($N \in \{63, 84, 96\}$) [@L63; @L84; @L96] which are toy models of atmospheric variability. 
Some of the aforementioned models are obtained by truncating heavily the dynamical equations of the atmosphere, and such heavy truncation of the dynamics can lead to non-realistic dynamical properties.
On the other hand, in a broader point of view, less truncated spectral quasi-geostrophic models of the atmosphere offer also a diverse set of models for the dry atmospheric dynamics. The dynamics hence obtained allow to 
identify typical features of the atmospheric circulation, such as blocked and zonal circulation regimes, low-frequency variability, etc...
However, these latter idealized models are less considered in the literature, despite their exhibit of realistic behaviours.

`qgs` aims to change that by providing a fast and easy-to-use Python framework for researcher and teacher to integrate this kind of models. 
For an efficient handling of the model by users, its documentation is conceived such that the its equations and parameters are explained and linked to the code.
In the future, its development will be done in a modular fashion which allows to connect the atmosphere to various other subsystems and use it with built-in and external toolboxes.

The choice of Python was specifically done to facilitate its use in Jupyter [@jupyter] notebooks and the multiple recent machine learning libraries that are available in this 
language.

# State of the field

Other software might interest the reader in need for an easy-to-use idealized atmospheric model.

* MAOOAM [@MAOOAM]: The Modular arbitrary-order ocean-atmosphere model, a coupled ocean-atmosphere model included in `qgs`. 
                    Code available in Lua, Fortran and Python.
* q-gcm [@qgcm]: A mid-latitude grid based ocean-atmosphere model like MAOOAM. Code in Fortran, interface is in Python.
* pyqg [@pyqg]: A pseudo-spectral python solver for quasi-geostrophic systems.
* Isca [@Isca]: Research General Circulation Model (GCM) written in Fortran and largely
                configured with Python scripts, with internal coding changes required for non-standard cases.

The mechanically coupled atmosphere-land version of `qgs` was used recently to test new ideas using response theory to adapt statistical postprocessing schemes to a model change in [@DV2020].
 
# Acknowledgements

This research has been supported by EUMETNET (Postprocessing module of the NWP Cooperation Programme).

# References
