
Coupled ocean-atmosphere model (MAOOAM)
=======================================

This model version is composed of a two-layer quasi-geostrophic :ref:`files/model/atmosphere:Atmospheric component`
coupled both thermally and mechanically to a shallow-water :ref:`files/model/ocean:Oceanic component`.
The coupling between the two components includes wind forcings, radiative and heat exchanges.

.. figure:: figures/model_variables_duck.png
    :scale: 50%
    :align: center

    Sketch of the ocean-atmosphere coupled model layers. The atmosphere is here a periodic channel while the ocean is a closed basin.
    The domain (:math:`\beta`-plane) `zonal and meridional`_ coordinates are labeled as the :math:`x` and
    :math:`y` variables.

The evolution equations for the atmospheric and oceanic streamfunctions defined in the sections above read

.. math::

    \frac{\partial}{\partial t}  \left(\nabla^2 \psi^1_{\rm a}\right)+ J(\psi^1_{\rm a}, \nabla^2 \psi^1_{\rm a})+ \beta \frac{\partial \psi^1_{\rm a}}{\partial x}
    & = -k'_d \nabla^2 (\psi^1_{\rm a}-\psi^3_{\rm a})+ \frac{f_0}{\Delta p} \omega \nonumber \\
    \frac{\partial}{\partial t} \left( \nabla^2 \psi^3_{\rm a} \right) + J(\psi^3_{\rm a}, \nabla^2 \psi^3_{\rm a}) + \beta \frac{\partial \psi^3_{\rm a}}{\partial x}
    & = +k'_d \nabla^2 (\psi^1_{\rm a}-\psi^3_{\rm a}) - \frac{f_0}{\Delta p}  \omega \nonumber - k_d \nabla^2 \left(\psi^3_{\rm a} - \psi_{\rm o}\right) \\
    \frac{\partial}{\partial t} \left( \nabla^2 \psi_\text{o} - \frac{\psi_\text{o}}{L_\text{R}^2} \right) + J(\psi_\text{o}, \nabla^2 \psi_\text{o}) + \beta \frac{\partial \psi_\text{o}}{\partial x}
    & = -r \nabla^2 \psi_\text{o} +\frac{C}{\rho_{\rm o} h} \nabla^2 (\psi^3_\text{a}-\psi_\text{o}).\nonumber

where :math:`\rho_{\rm o}` is the density of the ocean's water and :math:`h` is the depth of its layer (:attr:`~.params.OceanicParams.h`).
The rightmost term of the last equation represents the impact of the wind stress on the ocean, and is modulated
by the coefficient of the mechanical ocean-atmosphere coupling, :math:`d = C/(\rho_{\rm o} h)` (:attr:`~.params.OceanicParams.d`).

As we did for the :ref:`files/model/oro_model:Model with an orography and a temperature profile`, we rewrite these equations in terms of the `barotropic`_ streamfunction :math:`\psi_{\rm a}` and `baroclinic`_ streamfunction :math:`\theta_{\rm a}`:

.. math::

    &\frac{\partial}{\partial t}  \left(\nabla^2 \psi_{\rm a}\right) + J(\psi_{\rm a}, \nabla^2 \psi_{\rm a}) + J(\theta_{\rm a}, \nabla^2 \theta_{\rm a}) + \beta \frac{\partial \psi_{\rm a}}{\partial x} = - \frac{k_d}{2} \nabla^2 (\psi_{\rm a} - \theta_{\rm a} - \psi_{\rm o}) \\
    &\frac{\partial}{\partial t} \left( \nabla^2 \theta_{\rm a} \right) + J(\psi_{\rm a}, \nabla^2 \theta_{\rm a}) + J(\theta_{\rm a}, \nabla^2 \psi_{\rm a}) + \beta \frac{\partial \theta_{\rm a}}{\partial x} \nonumber \\
    & \qquad \qquad \qquad \qquad \qquad \qquad = - 2 \, k'_d \nabla^2 \theta_{\rm a} + \frac{k_d}{2} \nabla^2 (\psi_{\rm a} - \theta_{\rm a} - \psi_{\rm o}) + \frac{f_0}{\Delta p}  \omega \nonumber \\
    &\frac{\partial}{\partial t} \left( \nabla^2 \psi_\text{o} - \frac{\psi_\text{o}}{L_\text{R}^2} \right) + J(\psi_\text{o}, \nabla^2 \psi_\text{o}) + \beta \frac{\partial \psi_\text{o}}{\partial x} = -r \nabla^2 \psi_\text{o} + d \, \nabla^2 (\psi_\text{a}- \theta_\text{a}-\psi_\text{o}).\nonumber

Temperature equations
---------------------

The temperature :math:`T_\text{o}` in the ocean is a passively advected scalar interacting with the atmospheric temperature through radiative and heat exchange, according to a scheme detailed in :cite:`mao-BB1998`:

.. math::

    \gamma_\text{o} \left( \frac{\partial T_\text{o}}{\partial t} + J(\psi_\text{o}, T_\text{o}) \right) = -\lambda (T_\text{o}-T_\text{a}) -\sigma_\text{B} T_\text{o}^4 + \epsilon_\text{a} \sigma_\text{B} T_\text{a}^4 + R_\text{o}

and the time evolution of the atmospheric temperature :math:`T_\text{a}` obeys a similar equation:

.. math::

    \gamma_\text{a} \left( \frac{\partial T_\text{a}}{\partial t} + J(\psi_\text{a}, T_\text{a}) -\sigma \omega \frac{p}{R}\right) = -\lambda (T_\text{a}-T_\text{o}) + \epsilon_\text{a} \sigma_\text{B} T_\text{o}^4 - 2 \epsilon_\text{a} \sigma_\text{B} T_\text{a}^4 + R_\text{a}

where :math:`\gamma_\text{a}` (:attr:`.AtmosphericTemperatureParams.gamma`) and :math:`\gamma_\text{o}`
(:attr:`.OceanicTemperatureParams.gamma`) are the heat capacities of the
atmosphere and the active ocean layer. :math:`\lambda` (:attr:`~.AtmosphericTemperatureParams.hlambda`) is the heat transfer coefficient at the
ocean-atmosphere interface.
:math:`\sigma` (:attr:`~.params.AtmosphericParams.sigma`) is the static stability of the atmosphere, taken to be constant.
The quartic terms represent the long-wave
radiation fluxes between the ocean, the atmosphere, and outer space, with
:math:`\epsilon_\text{a}` (:attr:`~.AtmosphericTemperatureParams.eps`)  the emissivity of the grey-body atmosphere and
:math:`\sigma_\text{B}` (:attr:`~.QgParams.sb`) the Stefan--Boltzmann constant. By decomposing the
temperatures as :math:`T_\text{a} = T_{\text{a},0} + \delta T_\text{a}` and :math:`T_\text{o} = T_{\text{o},0} + \delta T_\text{o}`, the quartic terms are
linearized around spatially uniform and constant temperatures :math:`T_{\text{a},0}` (:attr:`.AtmosphericTemperatureParams.T0`) and
:math:`T_{\text{o},0}` (:attr:`.OceanicTemperatureParams.T0`), as detailed in Appendix B of :cite:`mao-VDDG2015`. :math:`R_\text{a}`
and :math:`R_\text{o}` are the short-wave radiation fluxes entering the atmosphere
and the ocean that are also decomposed as :math:`R_\text{a}=R_{\text{a}, 0} + \delta R_\text{a}` and :math:`R_\text{o} = R_{\text{o}, 0} + \delta R_\text{o}`.
It results in these evolution equations for the temperature anomalies:

.. math::

    \gamma_{\rm a} \Big( \frac{\partial \delta T_{\rm a}}{\partial t} + J(\psi_{\rm a}, \delta T_{\rm a} )- \sigma \omega \frac{p}{R}\Big) &= -\lambda (\delta T_{\rm a}- \delta T_{\rm o}) +4 \sigma_B T_{{\rm o},0}^3 \delta T_{\rm o} - 8 \epsilon_{\rm a} \sigma_B T_{{\rm a},0}^3 \delta T_{\rm a} + \delta R_{\rm a} \nonumber \\
    \gamma_{\rm o} \Big( \frac{\partial \delta T_{\rm o}}{\partial t} + J(\psi_{\rm o}, \delta T_{\rm o})\Big) &= -\lambda (\delta T_{\rm o}- \delta T_{\rm a}) -4 \sigma_B T_{{\rm o},0}^3 \delta T_{\rm o} + 4 \epsilon_{\rm a} \sigma_B T_{{\rm a},0}^3 \delta T_{\rm a} + \delta R_{\rm o}. \nonumber

The hydrostatic relation in pressure coordinates is :math:`(\partial \Phi/\partial p)
= -1/\rho_\text{a}` with the geopotential height :math:`\Phi = f_0\;\psi_\text{a}` and :math:`\rho_\text{a}` the dry air density. The ideal gas relation :math:`p=\rho_\text{a} R T_\text{a}`
and the vertical discretization of the hydrostatic relation at 500 hPa allows to write the spatially dependent atmospheric temperature anomaly :math:`\delta T_\text{a} = 2f_0\;\theta_\text{a} /R` where :math:`R` (:attr:`~.QgParams.rr`) is
the ideal gas constant.

.. figure:: figures/energybalance.png
    :scale: 30%
    :align: center

    Sketch of the energy balance of :cite:`mao-BB1998`. It underlies the radiative and heat exchange scheme in the model.

.. figure:: figures/energybalance_detail.png
    :scale: 30%
    :align: center

    Actual values of the energy flux between the ground and the atmosphere :cite:`mao-TFK2009`.

Set of basis functions
----------------------

The present model solves the equations above by projecting them onto a basis of functions, to obtain a
system of `ordinary differential equations`_ (ODE). This procedure is sometimes called a `Galerkin expansion`_.
This basis being finite, the resolution of the model is automatically truncated at the characteristic length of the
highest-resolution function of the basis.

The atmospheric set of basis functions :math:`F_i` is described in the section :ref:`files/model/oro_model:Projecting the equations on a set of basis functions`.

The oceanic set of basis functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Both oceanic fields :math:`\psi_{\rm o}` and :math:`\delta T_{\rm o}` are defined in a closed basin with no-flux boundary
conditions (:math:`\partial \cdot_{\rm o} /\partial x \equiv 0` at the meridional boundaries and
:math:`\partial \cdot_{\rm o}/\partial y \equiv 0` at the zonal boundaries).

These fields are projected on Fourier modes respecting these boundary conditions:

.. math::

    \phi_{H_{\rm o},P_{\rm o}} (x, y) = 2\sin(\frac{H_{\rm o} n}{2}x)\, \sin(P_{\rm o} y)

with integer values of :math:`H_{\rm o}`, :math:`P_{\rm o}`.
Again, :math:`x` and :math:`y` are the horizontal adimensionalized coordinates defined above.

To easily manipulate these functions and the coefficients of the fields
expansion, we number the basis functions along increasing values of :math:`H_{\rm o}` and then :math:`P_{\rm o}`.
It allows to write the set as :math:`\left\{ \phi_i(x,y); 1 \leq i \leq n_\text{o}\right\}` where :math:`n_{\mathrm{o}}`
(:attr:`~.params.QgParams.nmod` [1]) is the number of modes of the spectral expansion in the ocean.

For example, the model derived in :cite:`mao-VDDG2015` can be specified by setting :math:`H_{\rm o} \in \{1,2\}`; :math:`P_{\rm o} \in \{1,4\}` and the set of basis functions is

.. math::

    \phi_1(x,y) & = &  2\, \sin(\frac{n}{2} x)\, \sin(y), \nonumber \\
    \phi_2(x,y) & = &  2\, \sin(\frac{n}{2} x)\, \sin(2 y), \nonumber \\
    \phi_3(x,y) & = &  2\, \sin(\frac{n}{2} x)\, \sin(3 y), \nonumber \\
    \phi_4(x,y) & = &  2\, \sin(\frac{n}{2} x)\, \sin(4 y), \nonumber \\
    \phi_5(x,y) & = &  2\, \sin(n x)\, \sin(y), \nonumber \\
    \phi_6(x,y) & = &  2\, \sin(n x)\, \sin(2 y), \nonumber \\
    \phi_7(x,y) & = &  2\, \sin(n x)\, \sin(3 y), \nonumber \\
    \phi_8(x,y) & = &  2\, \sin(n x)\, \sin(4 y), \nonumber

such that

.. math::

    \nabla^2 \phi_i(x,y) = -m^2_i  \,\phi_i(x,y)

with eigenvalues :math:`m_i^2 = P_{{\rm o},i}^2 + n^2 \, H_{{\rm o},i}^2/4`.
These Fourier modes are also orthonormal with respect to the inner product

.. math::

    \frac{n}{2\pi^2}\int_0^\pi\int_0^{2\pi/n} \phi_i(x,y)\, \phi_j(x,y)\, \mathrm{d} x \, \mathrm{d} y = \delta_{ij}

where :math:`\delta_{ij}` is the `Kronecker delta`_. Note however that the atmospheric and oceanic basis :math:`F_i` and
:math:`\phi_i` are not orthonormal to each other.

.. figure:: figures/visualise_basisfunctions_ocean.png
    :align: center

    The first 8 basis functions :math:`\phi_i` evaluated on the nondimensional domain of the model.

Fields expansion
----------------

The fields of the model can expanded on these sets of basis functions according to

.. math::

     \psi_\text{a} (x,y) &= \sum_{i=1}^{n_\text{a}} \; \psi_{\text{a},i} \, F_i(x,y), \\
     \theta_\text{a}(x,y) &=\sum_{i=1}^{n_\text{a}} \theta_{\text{a},i} \; F_i(x,y), \\
     \delta T_\text{a}(x,y) &=\sum_{i=1}^{n_\text{a}} \delta T_{\text{a},i} \; F_i(x,y), \\
     &= 2 \frac{f_0}{R} \sum_{i=1}^{n_\text{a}} \theta_{\text{a},i} \; F_i(x,y), \nonumber\\
     \psi_\text{o}(x,y) &= \sum_{j=1}^{n_\text{o}} \psi_{\text{o},j} \; (\phi_j(x,y) \; -\; \overline{\phi_j}), \\
     \delta T_\text{o}(x,y) &= \sum_{j=1}^{n_\text{o}} \delta T_{\text{o},j} \; \phi_j(x,y).

In the expansion for :math:`\psi_\text{o}`, a term :math:`\overline{\phi_j}` is added to the oceanic
basis function :math:`\phi_j` in order to get a zero spatial
average. This is required to guarantee mass conservation in the ocean, but otherwise does not affect the dynamics. Indeed,
it can be added a posteriori when plotting the field
:math:`\psi_\text{o}`. This term is non-zero for odd :math:`P_\text{o}` and
:math:`H_\text{o}`:

.. math::
    \overline{\phi_j} &= \frac{n}{2\pi^2} \int _0^{\pi }\int _0^{\frac{2 \pi }{n}}\phi_j(x,y) \,\text{d}x \,\text{d}y  \\
                   &= 2\frac{((-1)^{H_\text{o}} - 1) ((-1)^{P_\text{o}} - 1)}{H_\text{o} P_\text{o} \pi^2}.\nonumber

The mass conservation is automatically satisfied for :math:`\psi_\text{a}`,
as the spatial averages of the atmospheric basis functions :math:`F_i` are zero.

Furthermore, the short-wave radiation or insolation is determined by

.. math::

    \delta R_\text{a}(x,y) = \sum_{i=1}^{n_\text{a}} \, C_{\text{a},i} \, F_i, \\
    \delta R_\text{o}(x,y) = \sum_{i=1}^{n_\text{a}} \, C_{\text{o},i} \, F_i.

which we project on the same atmospheric basis of function to maintain consistency and allow meridional gradients.
These decompositions are stored in the parameters :attr:`.AtmosphericTemperatureParams.C` and :attr:`.OceanicTemperatureParams.C` and
can be set using the functions :attr:`.AtmosphericTemperatureParams.set_insolation` and :attr:`.OceanicTemperatureParams.set_insolation`.

The vertical velocity :math:`\omega(x,y)` have also to be decomposed:

.. math::

    \omega(x,y) = \sum_{i=1}^{n_{\mathrm{a}}} \, \omega_i \, F_i(x,y) .

Ordinary differential equations
-------------------------------

The fields, parameters and variables are non-dimensionalized
by dividing time by :math:`f_0^{-1}` (:attr:`~.params.ScaleParams.f0`), distance by
the characteristic length scale :math:`L` (:attr:`~.params.ScaleParams.L`), pressure by the difference :math:`\Delta p` (:attr:`~.params.ScaleParams.deltap`),
temperature by :math:`f_0^2 L^2/R`, and streamfunction by :math:`L^2 f_0`. As a result of this non-dimensionalization, the
fields :math:`\theta_{\rm a}` and :math:`\delta T_{\rm a}` can be identified: :math:`2 \theta_{\rm a} \equiv \delta T_{\rm a}`.

The ordinary differential equations of the truncated model are:

.. math::

  \dot\psi_{{\rm a},i} & = & - a_{i,i}^{-1} \sum_{j,m = 1}^{n_{\mathrm{a}}} b_{i, j, m} \left(\psi_{{\rm a},j}\, \psi_{{\rm a},m} + \theta_{{\rm a},j}\, \theta_{{\rm a},m}\right) - \beta\, a_{i,i}^{-1} \, \sum_{j=1}^{n_{\mathrm{a}}} \, c_{i, j} \, \psi_{{\rm a},j} \nonumber \\
  & & \qquad \qquad \qquad \qquad - \frac{k_d}{2} \left(\psi_{{\rm a},i} - \theta_{{\rm a},i}\right) + \frac{k_d}{2} \, a_{i,i}^{-1} \, \sum_{j = 1}^{n_{\mathrm{o}}} d_{i,j} \, \psi_{{\rm o},j} \\
  \dot\theta_{{\rm a},i} & = & - a_{i,i}^{-1} \sum_{j,m = 1}^{n_{\mathrm{a}}} b_{i, j, m} \left(\psi_{{\rm a},j}\, \theta_{{\rm a},m} + \theta_{{\rm a},j}\, \psi_{{\rm a},m}\right) - \beta\, a_{i,i}^{-1} \, \sum_{j=1}^{n_{\mathrm{a}}} \, c_{i, j} \, \theta_{{\rm a},j}  \nonumber  \\
  & & \qquad \qquad \qquad \qquad + \frac{k_d}{2} \left(\psi_{{\rm a},i} - \theta_{{\rm a},i}\right) - \frac{k_d}{2}  \, a_{i,i}^{-1} \, \sum_{j = 1}^{n_{\mathrm{o}}} d_{i,j} \, \psi_{{\rm o},j} - 2 \, k'_d \, \theta_{{\rm a},i} + a_{i,i}^{-1} \, \omega_i \\
  \dot\theta_{\rm{a},i} & = & - \sum_{j,m = 1}^{n_{\mathrm{a}}} g_{i, j, m} \, \psi_{{\rm a},j}\, \theta_{{\rm a},m} +  \frac{\sigma}{2}\, \omega_i - \left(\lambda'_{\rm a} + S_{B,{\rm a}} \right)  \, \theta_{\rm{a},i} \nonumber \\
  & & \qquad \qquad \qquad \qquad  + \left(\frac{\lambda'_{\rm a}}{2}+ S_{B, {\rm o}}\right) \sum_{j=1}^{n_{\mathrm{o}}} \, s_{i, j} \, \delta T_{{\rm o},j} + C'_{\text{a},i} \\
  \dot\psi_{{\rm o},i} & = & \frac{1}{\left(M_{i,i} + G\right)} \, \left\{ - \sum_{j,m = 1}^{n_{\mathrm{o}}} \, C_{i,j,k} \, \psi_{{\rm o},j} \, \psi_{{\rm o},k} - \beta \, \sum_{j = 1}^{n_{\mathrm{o}}} \, N_{i,j} \, \psi_{{\rm o}, j} - (d + r) \, \sum_{j = 1}^{n_{\mathrm{o}}} \, M_{i,j} \, \psi_{{\rm o},j} \right. \nonumber \\
  & & \qquad \qquad \qquad \qquad + \left. d \, \sum_{j = 1}^{n_{\mathrm{a}}} \, K_{i,j} \, \left(\psi_{{\rm a}, j} - \theta_{{\rm a}, j}\right)\right\} \\
  \dot\delta T_{{\rm o},i} & = & - \sum_{j,m = 1}^{n_{\mathrm{o}}} \, O_{i,j,m} \, \psi_{{\rm o},j} \, \delta T_{{\rm o},m} - \left(\lambda'_{\rm o}+ s_{B,{\rm o}}\right) \, \delta T_{{\rm o},i} + \left(2 \,\lambda'_{\rm o} + s_{B,{\rm a}}\right) \, \sum_{j=1}^{n_{\mathrm{a}}} \, W_{i,j} \, \theta_{{\rm a},j} + \sum_{j=1}^{n_{\mathrm{a}}} \, W_{i,j} \, C'_{{\rm o},j}

where the parameters values have been replaced by their non-dimensional ones and we have also defined
:math:`G = - L^2/L_R^2` (:attr:`~.params.QgParams.G`),
:math:`\lambda'_{{\rm a}} = \lambda/(\gamma_{\rm a} f_0)` (:attr:`~.params.QgParams.Lpa`),
:math:`\lambda'_{{\rm o}} = \lambda/(\gamma_{\rm o} f_0)` (:attr:`~.params.QgParams.Lpgo`),
:math:`S_{B,{\rm a}} = 8\,\epsilon_{\rm a}\, \sigma_B \, T_{{\rm a},0}^3 / (\gamma_{\rm a} f_0)` (:attr:`~.params.QgParams.LSBpa`),
:math:`S_{B,{\rm o}} = 2\,\epsilon_{\rm a}\, \sigma_B \, T_{{\rm a},0}^3 / (\gamma_{\rm a} f_0)` (:attr:`~.params.QgParams.LSBpgo`),
:math:`s_{B,{\rm a}} = 8\,\epsilon_{\rm a}\, \sigma_B \, T_{{\rm a},0}^3 / (\gamma_{\rm o} f_0)` (:attr:`~.params.QgParams.sbpa`),
:math:`s_{B,{\rm o}} = 4\,\sigma_B \, T_{{\rm a},0}^3 / (\gamma_{\rm o} f_0)` (:attr:`~.params.QgParams.sbpgo`),
:math:`C'_{{\rm a},i} = R C_{{\rm a},i} / (2 \gamma_{\rm a} L^2 f_0^3)` (:attr:`~.params.QgParams.Cpa`),
:math:`C'_{{\rm o},i} = R C_{{\rm o},i} /   (\gamma_{\rm o} L^2 f_0^3)` (:attr:`~.params.QgParams.Cpgo`).

The coefficients :math:`a_{i,j}`, :math:`g_{i, j, m}`, :math:`b_{i, j, m}` and :math:`c_{i, j}` are the inner products of the Fourier modes :math:`F_i`:

.. math::

  a_{i,j} & = & \frac{n}{2\pi^2}\int_0^\pi\int_0^{2\pi/n} F_i(x,y)\, \nabla^2 F_j(x,y)\, \mathrm{d} x \, \mathrm{d} y = - \delta_{ij} \, a_i^2 \\
  g_{i, j, m} & = & \frac{n}{2\pi^2}\int_0^\pi\int_0^{2\pi/n} F_i(x,y)\, J\left(F_j(x,y), F_m(x,y)\right) \, \mathrm{d} x \, \mathrm{d} y \\
  b_{i, j, m} & = & \frac{n}{2\pi^2}\int_0^\pi\int_0^{2\pi/n} F_i(x,y)\, J\left(F_j(x,y), \nabla^2 F_m(x,y)\right) \, \mathrm{d} x \, \mathrm{d} y \\
  c_{i, j} & = & \frac{n}{2\pi^2}\int_0^\pi\int_0^{2\pi/n} F_i(x,y)\, \frac{\partial}{\partial x} F_j(x,y) \, \mathrm{d} x \, \mathrm{d} y

and the coefficients :math:`M_{i,j}`, :math:`O_{i, j, m}`, :math:`C_{i, j, m}` and :math:`N_{i, j}` are the inner products of the Fourier modes :math:`\phi_i`:

.. math::

  M_{i,j} & = & \frac{n}{2\pi^2}\int_0^\pi\int_0^{2\pi/n} \phi_i(x,y)\, \nabla^2 \phi_j(x,y)\, \mathrm{d} x \, \mathrm{d} y = - \delta_{ij} \, m_i^2 \\
  O_{i, j, m} & = & \frac{n}{2\pi^2}\int_0^\pi\int_0^{2\pi/n} \phi_i(x,y)\, J\left(\phi_j(x,y), \phi_m(x,y)\right) \, \mathrm{d} x \, \mathrm{d} y \\
  C_{i, j, m} & = & \frac{n}{2\pi^2}\int_0^\pi\int_0^{2\pi/n} \phi_i(x,y)\, J\left(\phi_j(x,y), \nabla^2 \phi_m(x,y)\right) \, \mathrm{d} x \, \mathrm{d} y \\
  N_{i, j} & = & \frac{n}{2\pi^2}\int_0^\pi\int_0^{2\pi/n} \phi_i(x,y)\, \frac{\partial}{\partial x} \phi_j(x,y) \, \mathrm{d} x \, \mathrm{d} y.

The coefficients involved in the ocean-atmosphere interactions :math:`W_{i,j}`, :math:`K_{i, j}`, :math:`d_{i, j}` and :math:`s_{i, j}` are the inner products between the Fourier modes :math:`\phi_i` and :math:`F_i`:

.. math::

  d_{i,j} & = & \frac{n}{2\pi^2}\int_0^\pi\int_0^{2\pi/n} F_i(x,y)\, \nabla^2 \phi_j(x,y)\, \mathrm{d} x \, \mathrm{d} y \\
  K_{i,j} & = & \frac{n}{2\pi^2}\int_0^\pi\int_0^{2\pi/n} \phi_i(x,y)\, \nabla^2 F_j(x,y)\, \mathrm{d} x \, \mathrm{d} y \\
  W_{i, j} & = & \frac{n}{2\pi^2}\int_0^\pi\int_0^{2\pi/n} \phi_i(x,y)\, F_j(x,y) \, \mathrm{d} x \, \mathrm{d} y = s_{j, i}


These inner products are computed according to formulas detailed in :cite:`mao-DDV2016` and stored in objects derived from the :class:`~.inner_products.base.AtmosphericInnerProducts` and
:class:`~.inner_products.base.OceanicInnerProducts` classes.

The vertical velocity :math:`\omega_i` can be eliminated, leading to the final equations

.. math::

  \dot\psi_{{\rm a},i} & = & - a_{i,i}^{-1} \sum_{j,m = 1}^{n_{\mathrm{a}}} b_{i, j, m} \left(\psi_{{\rm a},j}\, \psi_{{\rm a},m} + \theta_{{\rm a},j}\, \theta_{{\rm a},m}\right) - \beta\, a_{i,i}^{-1} \, \sum_{j=1}^{n_{\mathrm{a}}} \, c_{i, j} \, \psi_{{\rm a},j} \nonumber \\
  & & \qquad \qquad \qquad \qquad - \frac{k_d}{2} \left(\psi_{{\rm a},i} - \theta_{{\rm a},i}\right) + \frac{k_d}{2}  \, a_{i,i}^{-1} \, \sum_{j = 1}^{n_{\mathrm{o}}} d_{i,j} \, \psi_{{\rm o},j} \\
  \dot\theta_{{\rm a},i} & = & \frac{\sigma/2}{a_{i,i} \,\sigma/2  - 1}  \left\{ - \sum_{j,m = 1}^{n_{\mathrm{a}}} b_{i, j, m} \left(\psi_{{\rm a},j}\, \theta_{{\rm a},m} + \theta_{{\rm a},j}\, \psi_{{\rm a},m}\right) - \beta\, \, \sum_{j=1}^{n_{\mathrm{a}}} \, c_{i, j} \, \theta_{{\rm a},j} \right. \nonumber  \\
  & & \qquad \qquad \qquad \qquad + \left. \frac{k_d}{2} \, a_{i,i} \left(\psi_{{\rm a},i} - \theta_{{\rm a},i}\right)  - \frac{k_d}{2}  \, \sum_{j = 1}^{n_{\mathrm{o}}} d_{i,j} \, \psi_{{\rm o},j} -2 \, k'_d \, a_{i,i} \, \theta_{{\rm a},i} \right\} \nonumber \\
  & & + \frac{1}{a_{i,i} \,\sigma/2  - 1} \left\{ \sum_{j,m = 1}^{n_{\mathrm{a}}} g_{i, j, m} \, \psi_{{\rm a},j}\, \theta_{{\rm a},m}  + \left(\lambda'_{\rm a} + S_{B,{\rm a}} \right)  \, \theta_{\rm{a},i} \right. \nonumber \\
  & & \qquad \qquad \qquad \qquad - \left.\left(\frac{\lambda'_{\rm a}}{2}+ S_{B, {\rm o}}\right) \sum_{j=1}^{n_{\mathrm{o}}} \, s_{i, j} \, \delta T_{{\rm o},j} - C'_{\text{a},i} \right\} \\
  \dot\psi_{{\rm o},i} & = & \frac{1}{\left(M_{i,i} + G\right)} \, \left\{ - \sum_{j,m = 1}^{n_{\mathrm{o}}} \, C_{i,j,k} \, \psi_{{\rm o},j} \, \psi_{{\rm o},k} - \beta \, \sum_{j = 1}^{n_{\mathrm{o}}} \, N_{i,j} \, \psi_{{\rm o}, j} - (d + r) \, \sum_{j = 1}^{n_{\mathrm{o}}} \, M_{i,j} \, \psi_{{\rm o},j} \right. \nonumber \\
  & & \qquad \qquad \qquad \qquad + \left. d \, \sum_{j = 1}^{n_{\mathrm{a}}} \, K_{i,j} \, \left(\psi_{{\rm a}, j} - \theta_{{\rm a}, j}\right)\right\} \\
  \dot\delta T_{{\rm o},i} & = & - \sum_{j,m = 1}^{n_{\mathrm{o}}} \, O_{i,j,m} \, \psi_{{\rm o},j} \, \delta T_{{\rm o},m} - \left(\lambda'_{\rm o}+ s_{B,{\rm o}}\right) \, \delta T_{{\rm o},i} + \left(2 \,\lambda'_{\rm o} + s_{B,{\rm a}}\right) \, \sum_{j=1}^{n_{\mathrm{a}}} \, W_{i,j} \, \theta_{{\rm a},j}  + \sum_{j=1}^{n_{\mathrm{a}}} \, W_{i,j} \, C'_{{\rm o},j}

that are implemented by means of a tensorial contraction:

.. math::

    \frac{\text{d}\eta_i}{\text{d}t} = \sum_{j, k=0}^{2 (n_\mathrm{a}+n_\mathrm{o})} \mathcal{T}_{i,j,k} \; \eta_j \; \eta_k

with :math:`\boldsymbol{\eta} = (1, \psi_{{\rm a},1}, \ldots, \psi_{{\rm a},n_\mathrm{a}}, \theta_{{\rm a},1}, \ldots, \theta_{{\rm a},n_\mathrm{a}}, \psi_{{\rm o},1}, \ldots, \psi_{{\rm o},n_\mathrm{o}}, \delta T_{{\rm o},1}, \ldots, \delta T_{{\rm o},n_\mathrm{o}})`, as described in the :ref:`files/technical_description:Code Description`. Note that :math:`\eta_0 \equiv 1`.
The tensor :math:`\mathcal{T}`, which fully encodes the bilinear system of ODEs above, is computed and stored in the :class:`~.tensors.qgtensor.QgsTensor`.

Example
-------

An example about how to setup the model to use this model version is shown in :ref:`files/examples/DDV:Recovering the result of De Cruz, Demaeyer and Vannitsem (2016)`.


MAOSOAM model version
---------------------

qgs allows one to specify the basis of functions over which the PDE equations of the model are projected.
Therefore it is easy to reproduce the model proposed in :cite:`mao-VSD2019`, where the atmosphere and the ocean are subject to the same periodic channel boundary conditions.

.. figure:: figures/model_variables_maosoam_duck.png
    :scale: 50%
    :align: center

    Sketch of the ocean-atmosphere coupled model layers for the MAOSOAM model version, where both the atmosphere and the ocean are periodic channels.
    The domain (:math:`\beta`-plane) `zonal and meridional`_ coordinates are labeled as the :math:`x` and
    :math:`y` variables.

The basis of functions in the ocean is thus defined as:

.. math::

   \begin{aligned}
       \phi_{1}(x, y)&= F_{1} (x, y) = \sqrt{2}\cos(y),\\
       \phi_{2}(x, y)&= F_{2} (x, y) = 2 \cos (n x) \sin (y),\\
       &\vdots
   \end{aligned}

while the basis of functions is the same between the atmosphere and ocean.

Since these boundary conditions mimick the situation in the southern hemisphere, this model is called the Modular Arbitrary-Order Southern Ocean-Atmosphere Model (MAOSOAM).
This change in basis does not alter the ordinary differential equations shown above.

An example of how to setup this model version is given in :ref:`files/examples/VSPD:Recovering the result of Vannitsem, Solé-Pomies and De Cruz (2019)`.

Dynamical temperatures and quartic temperature tendencies model version
-----------------------------------------------------------------------

It is possible to calculate solutions of the model without linearizing the quartic long-wave radiation terms of the temperature equations for :math:`T_\text{a}` and
:math:`T_\text{o}`.
Instead, these radiation terms can be expanded onto the set of basis functions, including the 0-th order reference temperatures :math:`T_{\text{a}, 0}` and :math:`T_{\text{o}, 0}`.

This requires the introduction of a new basis function, which is the
constant basis: ‘:math:`1`’. Thus the atmospheric and oceanic basis
functions are now expanded as follow:

.. math::

   \begin{aligned}
       \phi_0(x, y)&=1,\\
       \phi_{1}(x, y)&=2 \sin \left(\frac{n}{2} x\right) \sin (y),\\
       \phi_{2}(x, y)&=2 \sin (n x) \sin (y),\\
       &\vdots
   \end{aligned}
   \qquad
   \begin{aligned}
       F_0(x, y)&=1,\\
       F_{1}(x, y)&=\sqrt{2}\cos(y),\\
       F_{2}(x, y)&=2 \cos (n x) \sin (y),\\
       &\vdots
   \end{aligned}

The temperature fields are then expanded onto these functions:

.. math:: T_\text{o}(x, y)=\sum_{j=0}^{n_\text{o}}T_{\text{o}, j}\,\phi_j(x, y)\qquad T_\text{a}(x, y)=\sum_{j=0}^{n_\text{a}}T_{\text{a}, j}\, F_j(x, y) = T_{\text{a}, 0} \, F_0(x, y) + 2 \frac{f_{0}}{R} \sum_{i=1}^{n_{\mathrm{a}}} \theta_{\mathrm{a}, i} F_{i}(x, y) .

Likewise, the short-wave radiation fluxes are expanded as

.. math:: R_\text{o}=R_{\text{o}, 0} + \delta R_\text{o} = \sum_{j=0}^{n_\text{o}}C_{\text{o}, j}\, F_j(x, y)\qquad R_\text{a}=R_{\text{a}, 0} + \delta R_\text{a} = \sum_{j=0}^{n_\text{a}}C_{\text{a}, j}\, F_j(x, y),

where we are thus identifying :math:`R_{\text{o}, 0}=C_{\text{o}, 0}` and :math:`R_{\text{a}, 0}=C_{\text{a}, 0}`.

Consequences in the ocean
^^^^^^^^^^^^^^^^^^^^^^^^^

This modified expansion involves re-writing the ocean temperature equations as:

.. math::

   &\gamma_{\mathrm{o}}\Bigg(\frac{\partial }{\partial t} \sum_{j=0}^{n_\text{o}}T_{\text{o}, j}\,\phi_j(x, y)+J\left(\sum_{m=1}^{n_\text{o}} \psi_{\mathrm{o}, m}\,\phi_m(x, y), \sum_{j=0}^{n_\text{o}}T_{\text{o}, j}\,\phi_j(x, y)\right) \Bigg)=\\
   &\qquad \qquad \qquad \qquad -\lambda\left(\sum_{j=0}^{n_\text{o}}T_{\text{o}, j}\,\phi_j(x, y)-\sum_{j=0}^{n_\text{a}}T_{\text{a}, j}\,F_j(x, y)\right)\\
   &\qquad \qquad \qquad \qquad -\sigma_{B} \left(\sum_{j=0}^{n_\text{o}}T_{\text{o}, j}\,\phi_j(x, y)\right)^{4}
   +\epsilon_{\mathrm{a}} \sigma_{B} \left(\sum_{j=0}^{n_\text{a}}T_{\text{a}, j}\, F_j(x, y)\right)^4+R_{\mathrm{o}}


This equation is then simplified and non-dimensionalized as in the linearized case.
This leads to the ordinary differential equation for the ocean temperature with quartic terms:

.. math::

   &\sum_{j=0}^{n_\text{o}}U_{i, j}\,\dot{T}_{\text{o}, j}=
   -\sum_{j,k=1}^{n_\text{o}}O_{i,j,m}\,\psi_{\text{o}, j}\,T_{\text{o}, m}
   -\sum_{j=0}^{n_\text{o}}\lambda'_\text{o}\, U_{i,j}T_{\text{o}, j}\\
   &\qquad \qquad \qquad -s_{B, \text{o}}\sum_{j,k,l,m=0}^{n_\text{o}}V_{i,j,k,l,m}\, T_{\text{o}, j}T_{\text{o},k}T_{\text{o},l}T_{\text{o},m}+2\lambda_\text{o}'\sum_{j=0}^{n_\text{a}}W_{i,j}\,\theta_{\text{a},j}\\
   &\qquad \qquad \qquad +s_{B, \text{a}} \sum_{j,k,l,m=0}^{n_\text{a}}Z_{i,j,k,l,m}\,\theta_{\text{a},j}\theta_{\text{a},k}\theta_{\text{a},l}\theta_{\text{a},m}
   +\sum_{j=0}^{n_\text{o}}W_{i,j}\, C'_{\text{o},j}

where here :math:`s_{B,\text{o}}=\sigma_B f_0^5 L^6/(\gamma_\text{o} R^3)` (:attr:`~.params.QgParams.T4sbpgo`), and
:math:`s_{B, \text{a}}=16 \varepsilon_\text{a} \sigma_B f_0^5 L^6/(\gamma_\text{o} R^3)` (:attr:`~.params.QgParams.T4sbpa`).

The coefficients :math:`U_{i, j}`, :math:`V_{i,j,k,l,m}`, and
:math:`Z_{i,j,k,l,m}` are the inner products of the basis functions :math:`\phi_{i}` and are defined as:

.. math:: U_{i, j}=\frac{n}{2 \pi^{2}} \int_{0}^{\pi} \int_{0}^{2 \pi / n} \phi_{i}(x, y)~\phi_{j}(x, y)\, \mathrm{d} x \, \mathrm{d} y

.. math:: V_{i, j, k, l, m}=\frac{n}{2 \pi^{2}} \int_{0}^{\pi} \int_{0}^{2 \pi / n} \phi_{i}(x, y)~\phi_{j}(x, y)~\phi_{k}(x, y)~\phi_{l}(x, y)~\phi_{m}(x, y)\, \mathrm{d} x \, \mathrm{d} y

.. math:: Z_{i, j, k, l, m}=\frac{n}{2 \pi^{2}} \int_{0}^{\pi} \int_{0}^{2 \pi / n} \phi_{i}(x, y)~F_{j}(x, y)~F_{k}(x, y)~F_{l}(x, y)~F_{m}(x, y)\, \mathrm{d} x \, \mathrm{d} y,

These inner products are stored in objects derived from the :class:`~.inner_products.base.OceanicInnerProducts` classes.

The equation corresponding to the oceanic barotropic streamfunction :math:`\psi_\text{o}(x, y)=\sum_{j=1}^{n_\text{o}}\psi_{\text{o}, j}\, \phi_j(x, y)`
is unchanged by the modified expansion and is given by:

.. math::

  \sum_{j=1}^{n_\text{o}} \left(M_{i,j} + G\, U_{i,j}\right) \dot\psi_{{\rm o},j} & = & - \sum_{j,m = 1}^{n_{\mathrm{o}}} \, C_{i,j,k} \, \psi_{{\rm o},j} \, \psi_{{\rm o},k} - \beta \, \sum_{j = 1}^{n_{\mathrm{o}}} \, N_{i,j} \, \psi_{{\rm o}, j} - (d + r) \, \sum_{j = 1}^{n_{\mathrm{o}}} \, M_{i,j} \, \psi_{{\rm o},j}  \nonumber \\
  & & \qquad \qquad \qquad \qquad + d \, \sum_{j = 1}^{n_{\mathrm{a}}} \, K_{i,j} \, \left(\psi_{{\rm a}, j} - \theta_{{\rm a}, j}\right) \\


Consequences in the atmosphere
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The atmosphere temperature equation now reads:

.. math::

   &\gamma_{\mathrm{a}}\left(\frac{\partial}{\partial t}\sum_{j=0}^{n_\text{a}}T_{\text{a}, j}\, F_j(x, y)+J\left(\sum_{m=1}^{n_\text{a}} \psi_{\mathrm{a}, m}\, F_m(x, y), \sum_{j=0}^{n_\text{a}}T_{\text{a}, j}\,F_j(x, y)\right)-\sigma \omega \frac{p}{R}\right)=\\
   &\qquad \qquad \qquad \qquad -\lambda\left(\sum_{j=0}^{n_\text{a}}T_{\text{a}, j}\, F_j(x, y)-\sum_{j=0}^{n_\text{o}}T_{\text{o}, j}\,\phi_j(x, y)\right)\\
   &\qquad \qquad \qquad \qquad +\epsilon_{\mathrm{a}} \sigma_{B} \left(\sum_{j=0}^{n_\text{o}}T_{\text{o}, j}\,\phi_j(x, y)\right)^4-2 \epsilon_{\mathrm{a}} \sigma_{B} \left(\sum_{j=0}^{n_\text{a}}T_{\text{a}, j}\, F_j(x, y)\right)^4+R_{\mathrm{a}}.

The atmospheric baroclinic streamfunction is calculated in a similar manner to the ocean temperature equation, where the identity :math:`T_{\text{a}, 0} \equiv 2 \frac{f_{0}}{R} \theta_{\mathrm{a}, 0}` is used:

.. math::

   &\sum_{j=0}^{n_\text{a}}u_{i, j}\,\dot{\theta}_{\text{a}, j}=
   -\sum_{j,k=1}^{n_\text{a}}g_{i,j,m}\,\psi_{\text{a}, j}\,\theta_{\text{a}, m}+\frac{\sigma}{2}\sum_{j=1}^{n_\text{a}}u_{i,j}\,\omega_j-\lambda'_\text{a}\sum_{j=0}^{n_\text{a}}u_{i,j}\,\theta_{\text{a}, j}\\
   &\qquad \qquad \qquad -S_{B,\text{a}}\sum_{j,k,l,m=0}^{n_\text{a}}z_{i,j,k,l,m}\,\theta_{\text{a}, j}\theta_{\text{a},k}\theta_{\text{a},l}\theta_{\text{a},m}+\frac{\lambda'_\text{a}}{2}\sum_{j=0}^{n_\text{o}}s_{i,j}\, T_{\text{o}, j}\\
   &\qquad \qquad \qquad +S_{B, \text{o}}\sum_{j,k,l,m=0}^{n_\text{o}}v_{i,j,k,l,m}\, T_{\text{o}, j}T_{\text{o},k}T_{\text{o},l}T_{\text{o},m}+\sum_{j=0}^{n_\text{a}}u_{i,j}\, C'_{\text{a},j}

where :math:`S_{B,\text{a}}=16\varepsilon_\text{a}\sigma_B L^6 f_0^5/(\gamma_\text{a}R^3)` (:attr:`~.params.QgParams.T4LSBpa`) and
:math:`S_{B, \text{o}}=\varepsilon_\text{a} \sigma_B L^6 f_0^5/(2\gamma_\text{a}R)` (:attr:`~.params.QgParams.T4LSBpgo`).

The coefficients :math:`u_{i, j}`, :math:`v_{i,j,k,l,m}`, and
:math:`z_{i,j,k,l,m}` are the corresponding inner products of the basis functions :math:`F_{i}`:

.. math:: u_{i, j}=\frac{n}{2 \pi^{2}} \int_{0}^{\pi} \int_{0}^{2 \pi / n} F_{i}(x, y)~F_{j}(x, y)\, \mathrm{d} x \mathrm{~d} y

.. math:: v_{i, j, k, l, m}=\frac{n}{2 \pi^{2}} \int_{0}^{\pi} \int_{0}^{2 \pi / n} F_{i}(x, y)~\phi_{j}(x, y)~\phi_{k}(x, y)~\phi_{l}(x, y)~\phi_{m}(x, y)\, \mathrm{d} x \mathrm{~d} y

.. math:: z_{i, j, k, l, m}=\frac{n}{2 \pi^{2}} \int_{0}^{\pi} \int_{0}^{2 \pi / n} F_{i}(x, y)~F_{j}(x, y)~F_{k}(x, y)~F_{l}(x, y)~F_{m}(x, y)\, \mathrm{d} x \mathrm{~d} y

These inner products are stored in objects derived from the :class:`~.inner_products.base.AtmosphericInnerProducts`.

Besides the two equations for the temperatures, the equations for the atmospheric barotropic and the baroclinic streamfunctions:

.. math:: \psi_\text{a}(x, y)=\sum_{j=1}^{n_\text{a}}\psi_{\text{a}, j}\, F_j(x, y)\qquad \theta_\text{a}(x, y)=\sum_{j=1}^{n_\text{a}} \theta_{\text{a}, j}\, F_j(x, y),

derived in the linearized case are still valid here:

.. math::

  \sum_{j=1}^{n_\text{a}} a_{i,j} \, \dot\psi_{{\rm a},j} & = & - \sum_{j,m = 1}^{n_{\mathrm{a}}} b_{i, j, m} \left(\psi_{{\rm a},j}\, \psi_{{\rm a},m} + \theta_{{\rm a},j}\, \theta_{{\rm a},m}\right) - \beta\, \sum_{j=1}^{n_{\mathrm{a}}} \, c_{i, j} \, \psi_{{\rm a},j} \nonumber \\
  & & \qquad \qquad \qquad \qquad - \frac{k_d}{2} \sum_{j=1}^{n_\text{a}} a_{i,j} \left(\psi_{{\rm a},j} - \theta_{{\rm a},j}\right) + \frac{k_d}{2} \, \sum_{j = 1}^{n_{\mathrm{o}}} d_{i,j} \, \psi_{{\rm o},j} \\
  \sum_{j=1}^{n_\text{a}} a_{i,j} \, \dot\theta_{{\rm a},j} & = & - \sum_{j,m = 1}^{n_{\mathrm{a}}} b_{i, j, m} \left(\psi_{{\rm a},j}\, \theta_{{\rm a},m} + \theta_{{\rm a},j}\, \psi_{{\rm a},m}\right) - \beta\, \sum_{j=1}^{n_{\mathrm{a}}} \, c_{i, j} \, \theta_{{\rm a},j}  \nonumber  \\
  & & \qquad \qquad \qquad \qquad + \frac{k_d}{2} \sum_{j=1}^{n_\text{a}} a_{i,j} \left(\psi_{{\rm a},j} - \theta_{{\rm a},j}\right) - \frac{k_d}{2}  \, \sum_{j = 1}^{n_{\mathrm{o}}} d_{i,j} \, \psi_{{\rm o},j} \nonumber \\
  & & \qquad \qquad \qquad \qquad - 2 \, k'_d \, \sum_{j=1}^{n_\text{a}} a_{i,j} \, \theta_{{\rm a},j} + \sum_{j=1}^{n_\text{a}} u_{i,j} \, \omega_j \\

As with the MAOOAM model with linearized temperature described above, the vertical velocity is then eliminated to provide a single equation
for the atmospheric baroclinic streamfunction.
The resulting closed baroclinic streamfunction equation, with :math:`T_{\text{a}, 0} \equiv 2 \frac{f_{0}}{R} \theta_{\mathrm{a}, 0}`, is:

.. math::

   \frac{\sigma}{2}\sum_{j=1}^{n_\text{a}} a_{i, j}~\dot{\theta}_{\text{a}, j} - \sum_{j=0}^{n_\text{a}} u_{i, j}~\dot{\theta}_{\text{a}, j}&=\frac{\sigma}{2}\bigg\{- \sum_{j,m = 1}^{n_{\mathrm{a}}} b_{i, j, m} \left(\psi_{{\rm a},j}\, \theta_{{\rm a},m} + \theta_{{\rm a},j}\, \psi_{{\rm a},m}\right) - \beta\, \sum_{j=1}^{n_{\mathrm{a}}} \, c_{i, j} \, \theta_{{\rm a},j} \\
   & + \frac{k_d}{2} \sum_{j=1}^{n_\text{a}} a_{i,j} \left(\psi_{{\rm a},j} - \theta_{{\rm a},j}\right) - \frac{k_d}{2}  \, \sum_{j = 1}^{n_{\mathrm{o}}} d_{i,j} \, \psi_{{\rm o},j} - 2 \, k'_d \, \sum_{j=1}^{n_\text{a}} a_{i,j} \, \theta_{{\rm a},j} \bigg\} \\
   &+\bigg\{\sum_{j,k=1}^{n_\text{a}}g_{i,j,m}\,\psi_{\text{a}, j}\,\theta_{\text{a}, m}+\lambda'_\text{a}\sum_{j=0}^{n_\text{a}}u_{i,j}\,\theta_{\text{a}, j}\\
   &+S_{B,\text{a}}\sum_{j,k,l,m=0}^{n_\text{a}}z_{i,j,k,l,m}\,\theta_{\text{a}, j}\theta_{\text{a},k}\theta_{\text{a},l}\theta_{\text{a},m}-\frac{\lambda'_\text{a}}{2}\sum_{j=0}^{n_\text{o}}s_{i,j}\, T_{\text{o}, j}\\
   &-S_{B, \text{o}}\sum_{j,k,l,m=0}^{n_\text{o}}v_{i,j,k,l,m}\, T_{\text{o}, j}T_{\text{o},k}T_{\text{o},l}T_{\text{o},m}-\sum_{j=0}^{n_\text{a}}u_{i,j}\, C'_{\text{a},j}\bigg\}

These equations are implemented by means of a tensor contraction:

.. math:: \frac{\mathrm{d} \eta_{i}}{\mathrm{d} t}=\sum_{j, k, l, m, n=0}^{2\left(n_{\mathrm{a}}+n_{\mathrm{o}}+1\right)} \mathcal{T}_{i, j, k, l, m} \,\eta_{j} \eta_{k}\eta_{l}\eta_{m}

with
:math:`\eta=(1, \psi_{\text{a},1},\dots,\psi_{\text{a},n_\text{a}},\theta_{\text{a}, 0},\theta_{\text{a}, 1},\dots,\theta_{\text{a}, n_\text{a}},\psi_{\text{o}, 1},\dots,\psi_{\text{o}, n_o},T_{o, 0},T_{\text{o}, 1},\dots,T_{\text{o}, n_o})`,
which is described in more detail in the :ref:`files/technical_description:Code Description` page, see in particular the section :ref:`files/technical_description:Special case with the quartic temperature scheme`.
This specific temperature scheme can be activated by setting the parameter :attr:`.QgParams.T4` to :code:`True` when instantiating the model's :class:`.QgParams` parameters object.

Dynamical temperature and linearized temperature tendencies model version
-------------------------------------------------------------------------

As an alternative to the quartic temperature tendencies, it is also possible to use dynamic zero-th order reference temperatures while preserving the linearization of the temperature perturbations.
This differs from the linearised version of the model, described in the section :ref:`files/model/maooam_model:Temperature equations`, as here the
reference temperatures :math:`T_{{\rm o}, 0}` and :math:`T_{{\rm a}, 0}` are considered as spatially uniform but not constant in time.
These temperatures are therefore calculated at every time step.

In this version, the radiative quartic terms in the equations described in the previous section are
linearized by setting the indices :math:`j, k` and :math:`l` to zero, hence dropping the higher-order dependencies in the perturbations.
The linearized ocean temperature equation is reduced to the following equation:

.. math::

   &\sum_{j=0}^{n_\text{o}}U_{i, j}\,\dot{T}_{\text{o}, j}=
   -\sum_{j,k=1}^{n_\text{o}}O_{i,j,m}\,\psi_{\text{o}, j}\,T_{\text{o}, m}
   -\sum_{j=0}^{n_\text{o}}\lambda'_\text{o}\, U_{i,j}T_{\text{o}, j}\\
   &\qquad \qquad \qquad - 4 \, s_{B, \text{o}}\sum_{m=0}^{n_\text{o}}V_{i,0,0,0,m}\, T_{\text{o}, 0}^3\, T_{\text{o},m}+2\lambda_\text{o}'\sum_{j=0}^{n_\text{a}}W_{i,j}\,\theta_{\text{a},j}\\
   &\qquad \qquad \qquad + 4 \, s_{B, \text{a}} \sum_{m=0}^{n_\text{a}}Z_{i,0,0,0,m}\,\theta_{\text{a},0}^3\,\theta_{\text{a},m}
   +\sum_{j=0}^{n_\text{o}}W_{i,j}\, C'_{\text{o},j} ,

and similarly for the atmospheric temperature:

.. math::

   \frac{\sigma}{2}\sum_{j=1}^{n_\text{a}} a_{i, j}~\dot{\theta}_{\text{a}, j} - \sum_{j=0}^{n_\text{a}} u_{i, j}~\dot{\theta}_{\text{a}, j}&=\frac{\sigma}{2}\bigg\{- \sum_{j,m = 1}^{n_{\mathrm{a}}} b_{i, j, m} \left(\psi_{{\rm a},j}\, \theta_{{\rm a},m} + \theta_{{\rm a},j}\, \psi_{{\rm a},m}\right) - \beta\, \sum_{j=1}^{n_{\mathrm{a}}} \, c_{i, j} \, \theta_{{\rm a},j} \\
   & + \frac{k_d}{2} \sum_{j=1}^{n_\text{a}} a_{i,j} \left(\psi_{{\rm a},j} - \theta_{{\rm a},j}\right) - \frac{k_d}{2}  \, \sum_{j = 1}^{n_{\mathrm{o}}} d_{i,j} \, \psi_{{\rm o},j} - 2 \, k'_d \, \sum_{j=1}^{n_\text{a}} a_{i,j} \, \theta_{{\rm a},j} \bigg\} \\
   &+\bigg\{\sum_{j,k=1}^{n_\text{a}}g_{i,j,m}\,\psi_{\text{a}, j}\,\theta_{\text{a}, m}+\lambda'_\text{a}\sum_{j=0}^{n_\text{a}}u_{i,j}\,\theta_{\text{a}, j}\\
   &+ 4 \, S_{B,\text{a}}\sum_{m=0}^{n_\text{a}}z_{i,0,0,0,m}\,\theta_{\text{a},0}^3\,\theta_{\text{a}, m}-\frac{\lambda'_\text{a}}{2}\sum_{j=0}^{n_\text{o}}s_{i,j}\, T_{\text{o}, j}\\
   &- 4 \, S_{B, \text{o}}\sum_{m=0}^{n_\text{o}}v_{i,0,0,0,m}\, T_{\text{o}, 0}^3\, T_{\text{o},m}-\sum_{j=0}^{n_\text{a}}u_{i,j}\, C'_{\text{a},j}\bigg\}.

On the other hand, the equations for :math:`\psi_{\rm a}` and :math:`\psi_{\rm o}` are unchanged.
This particular temperature scheme can be activated by setting the parameter :attr:`.QgParams.dynamic_T` to :code:`True` when instantiating the model's :class:`.QgParams` parameters object.

References
----------

.. bibliography:: ref.bib
    :keyprefix: mao-

.. _zonal and meridional: https://en.wikipedia.org/wiki/Zonal_and_meridional_flow
.. _Kronecker delta: https://en.wikipedia.org/wiki/Kronecker_delta
.. _ordinary differential equations: https://en.wikipedia.org/wiki/Ordinary_differential_equation
.. _Galerkin expansion: https://en.wikipedia.org/wiki/Galerkin_method
.. _baroclinic: https://en.wikipedia.org/wiki/Baroclinity
.. _barotropic: https://en.wikipedia.org/wiki/Barotropic_fluid
