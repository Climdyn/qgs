
Atmospheric component
=====================

The atmospheric component is a two-layer `quasi-geostrophic`_ (QG) atmosphere in
the `beta-plane`_ approximation. The atmospheric component is an extension of
the QG model, first developed by :cite:`ac-CS1980` and further refined by
:cite:`ac-RP1982` and :cite:`ac-CT1987`.

.. figure:: figures/atmoro.png
    :scale: 70%
    :align: center

    Sketch of the atmospheric model layers with a simple `orography`_.
    The domain (:math:`\beta`-plane) `zonal and meridional`_ coordinates are labeled as the :math:`x` and
    :math:`y` variables.

The equations of motion for the atmospheric streamfunction
fields :math:`\psi^1_\text{a}` at 250 hPa and :math:`\psi^3_\text{a}` at 750 hPa, and
the vertical velocity :math:`\omega = \text{d}p/\text{d}t`, read

.. math::

    \frac{\partial}{\partial t}  \overbrace{\left(\nabla^2 \psi^1_{\rm a}\right) }^{\text{vorticity}}+ \overbrace{J(\psi^1_{\rm a}, \nabla^2 \psi^1_{\rm a})}^{\text{horizontal advection}} + \overbrace{\beta \frac{\partial \psi^1_{\rm a}}{\partial x}}^{\beta\text{-plane approximation} \\ \text{of the Coriolis force}}
    & = \overbrace{-k'_d \nabla^2 (\psi^1_{\rm a}-\psi^3_{\rm a})}^{\text{friction}} + \overbrace{\frac{f_0}{\Delta p} \omega}^{\text{vertical stretching}} \nonumber \\
    \frac{\partial}{\partial t} \left( \nabla^2 \psi^3_{\rm a} \right) + \, \ J(\psi^3_{\rm a}, \nabla^2 \psi^3_{\rm a}) \, \ + \qquad \beta \frac{\partial \psi^3_{\rm a}}{\partial x} \qquad
    & = +k'_d \nabla^2 (\psi^1_{\rm a}-\psi^3_{\rm a}) - \quad \ \frac{f_0}{\Delta p}  \omega \nonumber \\

where :math:`\nabla^2` is the horizontal Laplacian.
The Coriolis parameter :math:`f` is linearized around a value :math:`f_0` (:attr:`~.params.ScaleParams.f0`) evaluated at
latitude :math:`\phi_0` (:attr:`~.params.ScaleParams.phi0_npi`), :math:`f = f_0 + \beta y`, with
:math:`\beta=\text{d}f/\text{d}y` (:attr:`~.params.ScaleParams.beta`). The parameter :math:`k'_d`
(:attr:`~.params.AtmosphericParams.kdp`) quantify the friction between the two atmospheric layers,
and :math:`\Delta p = 500` hPa (:attr:`~.params.ScaleParams.deltap`) is the pressure difference between the atmospheric layers.
Finally, :math:`J` is the Jacobian :math:`J(S, G) = \partial_x S\, \partial_y G - \partial_y S\, \partial_x G`.


References
----------

.. bibliography:: ref.bib
    :labelprefix: AC-
    :keyprefix: ac-

.. _quasi-geostrophic: https://en.wikipedia.org/wiki/Quasi-geostrophic_equations
.. _MAOOAM: https://github.com/Climdyn/MAOOAM
.. _beta-plane: https://en.wikipedia.org/wiki/Beta_plane
.. _orography: https://en.wikipedia.org/wiki/Orography
.. _zonal and meridional: https://en.wikipedia.org/wiki/Zonal_and_meridional_flow

