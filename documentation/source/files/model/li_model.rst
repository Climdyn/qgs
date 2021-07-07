
Model with an orography and heat exchanges
==========================================

In the :ref:`files/model/oro_model:Model with an orography and a temperature profile`, the radiative equilibrium temperature field at the middle of the atmosphere
is specified by a given profile :math:`\theta^\star` (:attr:`~.params.AtmosphericTemperatureParams.thetas`) and the system relaxes to
this profile due to the `Newtonian cooling`_.

In Li et al. :cite:`li-LHHBD2018`, another scheme for the temperature is proposed, based on the
mechanism used for the :ref:`files/model/maooam_model:Coupled ocean-atmosphere model (MAOOAM)` and considering the radiative and heat exchanges between the atmosphere and the
ground. As in MAOOAM, this mechanism is the one proposed in :cite:`li-BB1998` and depicted in the :ref:`files/model/maooam_model:Temperature equations` section of the MAOOAM documentation,
with the ocean being replaced by the ground (with an orography).

The equations for this model thus read:


.. math::

    &\frac{\partial}{\partial t}  \left(\nabla^2 \psi_{\rm a}\right) + J(\psi_{\rm a}, \nabla^2 \psi_{\rm a}) + J(\theta_{\rm a}, \nabla^2 \theta_{\rm a}) + \frac{1}{2} J(\psi_{\rm a} - \theta_{\rm a}, f_0 \, h/H_{\rm a}) + \beta \frac{\partial \psi_{\rm a}}{\partial x} = - \frac{k_d}{2} \nabla^2 (\psi_{\rm a} - \theta_{\rm a}) \\
    &\frac{\partial}{\partial t} \left( \nabla^2 \theta_{\rm a} \right) + J(\psi_{\rm a}, \nabla^2 \theta_{\rm a}) + J(\theta_{\rm a}, \nabla^2 \psi_{\rm a}) - \frac{1}{2} J(\psi_{\rm a} - \theta_{\rm a}, f_0 \, h/H_{\rm a}) + \beta \frac{\partial \theta_{\rm a}}{\partial x} \nonumber \\
    & \qquad \qquad \qquad \qquad \qquad \qquad = - 2 \, k'_d \nabla^2 \theta_{\rm a} + \frac{k_d}{2} \nabla^2 (\psi_{\rm a} - \theta_{\rm a}) + \frac{f_0}{\Delta p}  \omega.

and

.. math::

    \gamma_\text{a} \left( \frac{\partial T_\text{a}}{\partial t} + J(\psi_\text{a}, T_\text{a}) -\sigma \omega \frac{p}{R}\right) &= -\lambda (T_\text{a}-T_\text{o}) + \epsilon_\text{a} \sigma_\text{B} T_\text{o}^4 - 2 \epsilon_\text{a} \sigma_\text{B} T_\text{a}^4 + R_\text{a} \\
    \gamma_\text{g} \, \frac{\partial T_\text{g}}{\partial t} &= -\lambda (T_\text{g}-T_\text{a}) -\sigma_\text{B} T_\text{g}^4 + \epsilon_\text{a} \sigma_\text{B} T_\text{a}^4 + R_\text{g}

As in MAOOAM, the temperature fields are expanded around their equilibrium profile to yield quadratic equations for the deviations to these profiles:

.. math::

    \gamma_{\rm a} \Big( \frac{\partial \delta T_{\rm a}}{\partial t} + J(\psi_{\rm a}, \delta T_{\rm a} )- \sigma \omega \frac{\delta p}{R}\Big) &= -\lambda (\delta T_{\rm a}- \delta T_{\rm g}) +4 \sigma_B T_{{\rm g},0}^3 \delta T_{\rm g} - 8 \epsilon_{\rm a} \sigma_B T_{{\rm a},0}^3 \delta T_{\rm a} + \delta R_{\rm a} \nonumber \\
    \gamma_{\rm g} \frac{\partial \delta T_{\rm g}}{\partial t}  &= -\lambda (\delta T_{\rm g}- \delta T_{\rm a}) -4 \sigma_B T_{{\rm g},0}^3 \delta T_{\rm g} + 4 \epsilon_{\rm a} \sigma_B T_{{\rm a},0}^3 \delta T_{\rm a} + \delta R_{\rm g}. \nonumber

and the ideal gas relation and the vertical discretization of the hydrostatic relation at 500 hPa allows to write the spatially dependent atmospheric temperature anomaly :math:`\delta T_\text{a} = 2f_0\;\theta_\text{a} /R` where :math:`R` (:attr:`~.QgParams.rr`) is
the ideal gas constant.

Ordinary differential equations
-------------------------------

All the modes of this model version are expanded on the set of Fourier modes :math:`F_i` detailed in the section :ref:`files/model/oro_model:Projecting the equations on a set of basis functions`:

.. math::

    \psi_{\rm a}(x,y) & = & \sum_{i=1}^{n_{\mathrm{a}}} \, \psi_{{\rm a},i} \, F_i(x,y) \\
    \theta_{\rm a}(x,y) & = & \sum_{i=1}^{n_{\mathrm{a}}} \, \theta_{{\rm a},i} \, F_i(x,y) \\
    \delta T_{\rm a}(x,y) & = & \sum_{i=1}^{n_{\mathrm{a}}} \, \delta T_{{\rm a},i} \, F_i(x,y) \\
    \delta T_{\rm g}(x,y) & = & \sum_{i=1}^{n_{\mathrm{a}}} \, \delta T_{{\rm g},i} \, F_i(x,y).


and as in MAOOAM, the fields, parameters and variables are non-dimensionalized
by dividing time by :math:`f_0^{-1}` (:attr:`~.params.ScaleParams.f0`), distance by
the characteristic length scale :math:`L` (:attr:`~.params.ScaleParams.L`), pressure by the difference :math:`\Delta p` (:attr:`~.params.ScaleParams.deltap`),
temperature by :math:`f_0^2 L^2/R`, and streamfunction by :math:`L^2 f_0`. As a result of this non-dimensionalization, the
fields :math:`\theta_{\rm a}` and :math:`\delta T_{\rm a}` can be identified: :math:`2 \theta_{\rm a} \equiv \delta T_{\rm a}`.

The equations of the system of ordinary differential equations for this model thus read:

.. math::

  \dot\psi_{{\rm a},i} & = & - a_{i,i}^{-1} \sum_{j,m = 1}^{n_{\mathrm{a}}} b_{i, j, m} \left(\psi_{{\rm a},j}\, \psi_{{\rm a},m} + \theta_{{\rm a},j}\, \theta_{{\rm a},m}\right) - \frac{a_{i,i}^{-1}}{2} \sum_{j,m = 1}^{n_{\mathrm{a}}} g_{i, j, m} \, h_m \left(\psi_{{\rm a},j}-\theta_{{\rm a},j}\right) \nonumber \\
  & & \qquad \qquad \qquad \qquad - \beta\, a_{i,i}^{-1} \, \sum_{j=1}^{n_{\mathrm{a}}} \, c_{i, j} \, \psi_{{\rm a},j} - \frac{k_d}{2} \left(\psi_{{\rm a},i} - \theta_{{\rm a},i}\right) \\
  \dot\theta_{{\rm a},i} & = & - a_{i,i}^{-1} \sum_{j,m = 1}^{n_{\mathrm{a}}} b_{i, j, m} \left(\psi_{{\rm a},j}\, \theta_{{\rm a},m} + \theta_{{\rm a},j}\, \psi_{{\rm a},m}\right) + \frac{a_{i,i}^{-1}}{2} \sum_{j,m = 1}^{n_{\mathrm{a}}} g_{i, j, m} \, h_m \left(\psi_{{\rm a},j}-\theta_{{\rm a},j}\right) \nonumber \\
  & & \qquad \qquad \qquad \qquad - \beta\, a_{i,i}^{-1} \, \sum_{j=1}^{n_{\mathrm{a}}} \, c_{i, j} \, \theta_{{\rm a},j} + \frac{k_d}{2} \left(\psi_{{\rm a},i} - \theta_{{\rm a},i}\right) - 2 \, k'_d \, \theta_{{\rm a},i} + a_{i,i}^{-1} \, \omega_i \\
  \dot\theta_{\rm{a},i} & = & - \sum_{j,m = 1}^{n_{\mathrm{a}}} g_{i, j, m} \, \psi_{{\rm a},j}\, \theta_{{\rm a},m} +  \frac{\sigma}{2}\, \omega_i - \left(\lambda'_{\rm a} + S_{B,{\rm a}} \right)  \, \theta_{\rm{a},i} + \left(\frac{\lambda'_{\rm a}}{2}+ S_{B, {\rm g}}\right) \, \delta T_{{\rm g},i} + C'_{\text{a},i} \\
  \dot\delta T_{{\rm g},i} & = & - \left(\lambda'_{\rm g}+ s_{B,{\rm g}}\right) \, \delta T_{{\rm g},i} + \left(2 \,\lambda'_{\rm g} + s_{B,{\rm a}}\right) \, \theta_{{\rm a},i} + C'_{{\rm g},i}

where the parameters values have been replaced by their non-dimensional ones and we have also defined
:math:`G = - L^2/L_R^2` (:attr:`~.params.QgParams.G`),
:math:`\lambda'_{{\rm a}} = \lambda/(\gamma_{\rm a} f_0)` (:attr:`~.params.QgParams.Lpa`),
:math:`\lambda'_{{\rm g}} = \lambda/(\gamma_{\rm g} f_0)` (:attr:`~.params.QgParams.Lpgo`),
:math:`S_{B,{\rm a}} = 8\,\epsilon_{\rm a}\, \sigma_B \, T_{{\rm a},0}^3 / (\gamma_{\rm a} f_0)` (:attr:`~.params.QgParams.LSBpa`),
:math:`S_{B,{\rm g}} = 2\,\epsilon_{\rm a}\, \sigma_B \, T_{{\rm a},0}^3 / (\gamma_{\rm a} f_0)` (:attr:`~.params.QgParams.LSBpgo`),
:math:`s_{B,{\rm a}} = 8\,\epsilon_{\rm a}\, \sigma_B \, T_{{\rm a},0}^3 / (\gamma_{\rm g} f_0)` (:attr:`~.params.QgParams.sbpa`),
:math:`s_{B,{\rm g}} = 4\,\sigma_B \, T_{{\rm a},0}^3 / (\gamma_{\rm g} f_0)` (:attr:`~.params.QgParams.sbpgo`),
:math:`C'_{{\rm a},i} = R C_{{\rm a},i} / (2 \gamma_{\rm a} L^2 f_0^3)` (:attr:`~.params.QgParams.Cpa`),
:math:`C'_{{\rm g},i} = R C_{{\rm g},i} /   (\gamma_{\rm g} L^2 f_0^3)` (:attr:`~.params.QgParams.Cpgo`).

The coefficients :math:`a_{i,j}`, :math:`g_{i, j, m}`, :math:`b_{i, j, m}` and :math:`c_{i, j}` are the inner products of the Fourier modes :math:`F_i`:

.. math::

  a_{i,j} & = & \frac{n}{2\pi^2}\int_0^\pi\int_0^{2\pi/n} F_i(x,y)\, \nabla^2 F_j(x,y)\, \mathrm{d} x \, \mathrm{d} y = - \delta_{ij} \, a_i^2 \\
  g_{i, j, m} & = & \frac{n}{2\pi^2}\int_0^\pi\int_0^{2\pi/n} F_i(x,y)\, J\left(F_j(x,y), F_m(x,y)\right) \, \mathrm{d} x \, \mathrm{d} y \\
  b_{i, j, m} & = & \frac{n}{2\pi^2}\int_0^\pi\int_0^{2\pi/n} F_i(x,y)\, J\left(F_j(x,y), \nabla^2 F_m(x,y)\right) \, \mathrm{d} x \, \mathrm{d} y \\
  c_{i, j} & = & \frac{n}{2\pi^2}\int_0^\pi\int_0^{2\pi/n} F_i(x,y)\, \frac{\partial}{\partial x} F_j(x,y) \, \mathrm{d} x \, \mathrm{d} y

These inner products are computed according to formulas found in :cite:`om-CT1987` and stored in an object derived from the :class:`~.inner_products.base.AtmosphericInnerProducts` class.

The vertical velocity :math:`\omega_i` can be eliminated, leading to the final equations

.. math::

  \dot\psi_{{\rm a},i} & = & - a_{i,i}^{-1} \sum_{j,m = 1}^{n_{\mathrm{a}}} b_{i, j, m} \left(\psi_{{\rm a},j}\, \psi_{{\rm a},m} + \theta_{{\rm a},j}\, \theta_{{\rm a},m}\right) - \frac{a_{i,i}^{-1}}{2} \sum_{j,m = 1}^{n_{\mathrm{a}}} g_{i, j, m} \, h_m \left(\psi_{{\rm a},j}-\theta_{{\rm a},j}\right) \nonumber \\
  & & \qquad \qquad \qquad \qquad - \beta\, a_{i,i}^{-1} \, \sum_{j=1}^{n_{\mathrm{a}}} \, c_{i, j} \, \psi_{{\rm a},j} - \frac{k_d}{2} \left(\psi_{{\rm a},i} - \theta_{{\rm a},i}\right) \\
  \dot\theta_{{\rm a},i} & = & \frac{\sigma/2}{a_{i,i} \,\sigma/2  - 1}  \left\{ - \sum_{j,m = 1}^{n_{\mathrm{a}}} b_{i, j, m} \left(\psi_{{\rm a},j}\, \theta_{{\rm a},m} + \theta_{{\rm a},j}\, \psi_{{\rm a},m}\right)  + \frac{a_{i,i}^{-1}}{2} \sum_{j,m = 1}^{n_{\mathrm{a}}} g_{i, j, m} \, h_m \left(\psi_{{\rm a},j}-\theta_{{\rm a},j}\right) \right. \nonumber  \\
  & & \qquad \qquad \qquad \qquad - \beta\, \, \sum_{j=1}^{n_{\mathrm{a}}} \, c_{i, j} \, \theta_{{\rm a},j} + \left. \frac{k_d}{2} \, a_{i,i} \left(\psi_{{\rm a},i} - \theta_{{\rm a},i}\right)  -2 \, k'_d \, a_{i,i} \, \theta_{{\rm a},i} \right\} \nonumber \\
  & & + \frac{1}{a_{i,i} \,\sigma/2  - 1} \left\{ \sum_{j,m = 1}^{n_{\mathrm{a}}} g_{i, j, m} \, \psi_{{\rm a},j}\, \theta_{{\rm a},m}  + \left(\lambda'_{\rm a} + S_{B,{\rm a}} \right)  \, \theta_{\rm{a},i} \right. \nonumber \\
  & & \qquad \qquad \qquad \qquad - \left.\left(\frac{\lambda'_{\rm a}}{2}+ S_{B, {\rm g}}\right) \, \delta T_{{\rm g},i} - C'_{\text{a},i} \right\} \\
  \dot\delta T_{{\rm g},i} & = & - \left(\lambda'_{\rm g}+ s_{B,{\rm g}}\right) \, \delta T_{{\rm g},i} + \left(2 \,\lambda'_{\rm g} + s_{B,{\rm a}}\right) \, \theta_{{\rm a},i}  +  C'_{{\rm g},i}

that are implemented by means of a tensorial contraction:

.. math::

    \frac{\text{d}\eta_i}{\text{d}t} = \sum_{j, k=0}^{3 n_\mathrm{a}} \mathcal{T}_{i,j,k} \; \eta_j \; \eta_k

with :math:`\boldsymbol{\eta} = (1, \psi_{{\rm a},1}, \ldots, \psi_{{\rm a},n_\mathrm{a}}, \theta_{{\rm a},1}, \ldots, \theta_{{\rm a},n_\mathrm{a}}, \delta T_{{\rm g},1}, \ldots, \delta T_{{\rm g},n_\mathrm{a}})`, as described in the :ref:`files/technical_description:Code Description`. Note that :math:`\eta_0 \equiv 1`.
The tensor :math:`\mathcal{T}`, which fully encodes the bilinear system of ODEs above, is computed and stored in the :class:`~.tensors.qgtensor.QgsTensor`.

Example
-------

An example about how to setup the model to use this model version is shown in :ref:`files/examples/Lietal:Atmospheric model with heat exchange - Li et al. model version (2017)`.

References
----------

.. bibliography:: ref.bib
    :labelprefix: LI-
    :keyprefix: li-

.. _Newtonian cooling: https://en.wikipedia.org/wiki/Newton%27s_law_of_cooling
