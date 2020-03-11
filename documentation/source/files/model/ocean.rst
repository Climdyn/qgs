
Oceanic component
=================

The oceanic component is a `shallow-water`_ ocean with a flat bottom orography.

Therefore, the equation of motion for the streamfunction :math:`\psi_\text{o}` of the ocean
layer reads :cite:`oc-P2011` :cite:`oc-DDV2016`

.. math::

    \frac{\partial}{\partial t} \left( \nabla^2 \psi_\text{o} -
    \frac{\psi_\text{o}}{L_\text{R}^2} \right) + J(\psi_\text{o}, \nabla^2
    \psi_\text{o}) + \beta \frac{\partial \psi_\text{o}}{\partial x} = -r \nabla^2 \psi_\text{o}.

:math:`L_\text{R}` is the `reduced Rossby deformation radius`_ (:attr:`~params.params.QgParams.LR`) and :math:`r`
(:attr:`~params.params.OceanicParams.r`) the friction at the bottom of the active ocean layer.

References
----------

.. bibliography:: ref.bib
    :labelprefix: OC-
    :keyprefix: oc-

.. _shallow-water: https://en.wikipedia.org/wiki/Shallow_water_equations
.. _MAOOAM: https://github.com/Climdyn/MAOOAM
.. _reduced Rossby deformation radius: https://en.wikipedia.org/wiki/Rossby_radius_of_deformation
