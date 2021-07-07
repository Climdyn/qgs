
Oceanic component
=================

The oceanic component is a `shallow-water`_ active oceanic layer superimposed on a deep ocean layer at rest.
The dynamics is given by the reduced-gravity quasi-geostrophic vorticity equation.

Therefore, the equation of motion for the streamfunction :math:`\psi_\text{o}` of the ocean
layer reads :cite:`oc-P2011` :cite:`oc-DDV2016`

.. math::

    \frac{\partial}{\partial t} \left( \nabla^2 \psi_\text{o} -
    \frac{\psi_\text{o}}{L_\text{R}^2} \right) + J(\psi_\text{o}, \nabla^2
    \psi_\text{o}) + \beta \frac{\partial \psi_\text{o}}{\partial x} = -r \nabla^2 \psi_\text{o}.

:math:`L_\text{R} = \sqrt{g' \, h }/ f_0` (:attr:`~.params.QgParams.LR`) is the `reduced Rossby deformation radius`_
where :math:`g'` (:attr:`~.params.OceanicParams.gp`) is the reduced gravity, :math:`h` is the depth of the layer (:attr:`~.params.OceanicParams.h`),
and :math:`f_0` is the Coriolis parameter (:attr:`~.params.ScaleParams.f0`).
:math:`r` (:attr:`~.params.OceanicParams.r`) is the friction at the bottom of the active ocean layer.

References
----------

.. bibliography:: ref.bib
    :labelprefix: OC-
    :keyprefix: oc-

.. _MAOOAM: https://github.com/Climdyn/MAOOAM
.. _reduced Rossby deformation radius: https://en.wikipedia.org/wiki/Rossby_radius_of_deformation
.. _shallow-water: https://en.wikipedia.org/wiki/Shallow_water_equations
