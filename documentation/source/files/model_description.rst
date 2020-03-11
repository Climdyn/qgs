
Model Description
=================

qgs is a Python 2-layer `quasi-geostrophic`_ channel model
on a `beta-plane`_ for the atmosphere, coupled to a simple land or
`shallow-water`_ ocean component beneath.
Presently it possesses two operation modes:

* A QG atmosphere coupled to an topography through friction only. It is described in details in the articles:

  * Reinhold, B. B., & Pierrehumbert, R. T. (1982). *Dynamics of weather regimes: Quasi-stationary waves and blocking*.
    Monthly Weather Review, **110** (9), 1105-1145.
    `doi:10.1175/1520-0493(1982)110%3C1105:DOWRQS%3E2.0.CO;2 <https://doi.org/10.1175/1520-0493(1982)110%3C1105:DOWRQS%3E2.0.CO;2>`_
  * Cehelsky, P., & Tung, K. K. (1987). *Theories of multiple equilibria and weather regimes—A critical reexamination.
    Part II: Baroclinic two-layer models*. Journal of the atmospheric sciences, **44** (21), 3282-3303.
    `doi:10.1175/1520-0469(1987)044%3C3282%3ATOMEAW%3E2.0.CO%3B2 <https://doi.org/10.1175/1520-0469(1987)044%3C3282%3ATOMEAW%3E2.0.CO%3B2>`_

* A QG atmosphere coupled to a closed-basin shallow-water ocean through friction and heat exchange.
  This configuration is the model `MAOOAM`_ by De Cruz, Demaeyer and Vannitsem; exhibiting oceanic
  low-frequency variability. It is described in details in the articles:

  * Vannitsem, S., Demaeyer, J., De Cruz, L., & Ghil, M. (2015). *Low-frequency variability and heat
    transport in a low-order nonlinear coupled ocean–atmosphere model*. Physica D: Nonlinear Phenomena, **309**, 71-85.
    `doi:10.1016/j.physd.2015.07.006 <https://doi.org/10.1016/j.physd.2015.07.006>`_

  * De Cruz, L., Demaeyer, J. and Vannitsem, S. (2016). *The Modular Arbitrary-Order Ocean-Atmosphere Model: MAOOAM v1.0*,
    Geosci. Model Dev., **9**, 2793-2808. `doi:10.5194/gmd-9-2793-2016 <https://doi.org/10.5194/gmd-9-2793-2016>`_

The shallow-water ocean can in principle be used alone, but it is not implemented for the moment.
More developments are yet to come, see the Forthcoming developments section below.

Components description
----------------------

.. toctree::
    :maxdepth: 2

    model/atmosphere
    model/ocean

Model versions description
--------------------------

.. toctree::
    :maxdepth: 2

    model/oro_model
    model/maooam_model

.. _quasi-geostrophic: https://en.wikipedia.org/wiki/Quasi-geostrophic_equations
.. _shallow-water: https://en.wikipedia.org/wiki/Shallow_water_equations
.. _MAOOAM: https://github.com/Climdyn/MAOOAM
.. _beta-plane: https://en.wikipedia.org/wiki/Beta_plane
