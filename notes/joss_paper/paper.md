---
title: 'qgs: A Quasi-geostrophic Spectral Model'
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

date: 20 July 2020
bibliography: joss.bib

---

# Summary

`qgs` is a a Python implementation of an idealized atmospheric model which displays a typical mid-latitudes variability. 
It consists in a spectral 2-layer quasi-geostrophic atmosphere on a beta-plane, coupled to a simple land or shallow-water ocean component.
* In the case of an ocean component, it reproduces the model proposed in [@DDV2016]. In [@VDDG2015], this model version was shown to reproduce a 
low-frequency variability (LFV) typical of the ocean-atmosphere coupling. 
This coupling consists in both mechanical and heat exchange interactions between the two components.
* In the case of a land component, it can reproduce the model proposed in [@RP1982] and [@CT1987] with a mechanical coupling due to the 
friction between the land and the atmosphere. It can also reproduce the model proposed in [@Lietal2017], with mechanical coupling and heat exchange.

`qgs` being a spectral model means that the partial differential equations (PDEs) ruling the time evolution of its fields are decomposed on a basis of spatial functions. 
This kind of decomposition are known as Galerkin decomposition and allow to transform the PDEs into a set of ordinary differential equations (ODEs) which can then be solved 
with the usual integration techniques.

The model implementation consists in submodules to set up the model's parameters and compute the model's tensor of tendencies terms.
This tensor is used by the code to compute the tendencies function and its Jacobian matrix. These functions can then be fed to the `qgs` built-in Runge-Kutta integrator or 
to external one. For instance, an example of the diffeqpy integration package usage is provided.

The model implementation use Numba extensively to accelerate the tensor products computation used to compute the tendencies.

# Statement of need

Frequently in geoscience, methods are tested and researchs are conducted with a Lorenz-$N$ model ($N \in \{63, 84, 96\}$)  which are toy models of atmospheric variability. 
Some of the aforementioned models are obtained by truncating heavily the dynamical equations of the atmosphere, and such heavy truncation of the dynamics can lead to non-realistic dynamical properties.
On the other hand, in a broader point of view, less truncated spectral quasi-geostrophic models of the atmosphere offer also a diverse set of models for the dry atmospheric dynamics. The dynamics hence obtained allow to 
identify typical features of the atmospheric circulation, such as blocked and zonal circulation regimes, low-frequency variability, etc...
However, these latter idealized models are less considered in the literature, despite their exhibit of realistic behaviours.

`qgs` aims to change that by providing a fast and easy-to-use Python framework for researcher and teacher to integrate this kind of models. 
Its development will be done in a modular fashion which allows to connect the atmosphere to various other subsystems and use it with built-in and external toolboxes.

The choice of Python was specifically done to facilitate its use in Jupyter notebooks and the multiple recent machine learning libraries that are available in this 
language.

# State of the field

# Acknowledgements

# References
