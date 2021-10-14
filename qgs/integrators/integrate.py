"""
    Integrate module
    ================

    Module with the function to integrate the ordinary differential equations

    .. math:: \\dot{\\boldsymbol{x}} = \\boldsymbol{f}(t, \\boldsymbol{x})

    of the model and its linearized version.

    Description of the module functions
    -----------------------------------

    Two main functions:

    * :obj:`integrate_runge_kutta`
    * :obj:`integrate_runge_kutta_tgls`

"""


# TODO : - test the provided functions with try before proceeding

from numba import njit
import numpy as np
from qgs.functions.util import reverse


def integrate_runge_kutta(f, t0, t, dt, ic=None, forward=True, write_steps=1, b=None, c=None, a=None):
    """
    Integrate the ordinary differential equations (ODEs)

    .. math:: \\dot{\\boldsymbol{x}} = \\boldsymbol{f}(t, \\boldsymbol{x})

    with a specified `Runge-Kutta method`_. The function :math:`\\boldsymbol{f}` should
    be a `Numba`_ jitted function. This function must have a signature ``f(t, x)`` where ``x`` is
    the state value and ``t`` is the time.

    .. _Runge-Kutta method: https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
    .. _Numba: https://numba.pydata.org/

    Parameters
    ----------
    f: callable
        The `Numba`_-jitted function :math:`\\boldsymbol{f}`.
        Should have the signature``f(t, x)`` where ``x`` is the state value and ``t`` is the time.
    t0: float
        Initial time of the time integration. Corresponds to the initial condition.
        Important if the ODEs are non-autonomous.
    t: float
        Final time of the time integration. Corresponds to the final condition.
        Important if the ODEs are non-autonomous.
    dt: float
        Timestep of the integration.
    ic: None or ~numpy.ndarray(float), optional
        Initial (or final) conditions of the system. Can be a 1D or a 2D array:

        * 1D: Provide a single initial condition.
          Should be of shape (`n_dim`,) where `n_dim` = :math:`\\mathrm{dim}(\\boldsymbol{x})`.
        * 2D: Provide an ensemble of initial condition.
          Should be of shape (`n_traj`, `n_dim`) where `n_dim` = :math:`\\mathrm{dim}(\\boldsymbol{x})`,
          and where `n_traj` is the number of initial conditions.

        If `None`, use a zero initial condition. Default to `None`.
        If the `forward` argument is `False`, it specifies final conditions.
    forward: bool, optional
        Whether to integrate the ODEs forward or backward in time. In case of backward integration, the
        initial condition `ic` becomes a final condition. Default to forward integration.
    write_steps: int, optional
        Save the state of the integration in memory every `write_steps` steps. The other intermediary
        steps are lost. It determines the size of the returned objects. Default is 1.
        Set to 0 to return only the final state.
    b: None or ~numpy.ndarray, optional
        Vector of coefficients :math:`b_i` of the `Runge-Kutta method`_ .
        If `None`, use the classic RK4 method coefficients. Default to `None`.
    c: None or ~numpy.ndarray, optional
        Matrix of coefficients :math:`c_{i,j}` of the `Runge-Kutta method`_ .
        If `None`, use the classic RK4 method coefficients. Default to `None`.
    a: None or ~numpy.ndarray, optional
        Vector of coefficients :math:`a_i` of the `Runge-Kutta method`_ .
        If `None`, use the classic RK4 method coefficients. Default to `None`.

    Returns
    -------
    time, traj: ~numpy.ndarray
        The result of the integration:

        * **time:** Time at which the state of the system was saved. Array of shape (`n_step`,) where
          `n_step` is the number of saved states of the integration.
        * **traj:** Saved dynamical system states. 3D array of shape (`n_traj`, `n_dim`, `n_steps`). If `n_traj` = 1,
          a 2D array of shape (`n_dim`, `n_steps`) is returned instead.

    Examples
    --------

    >>> from numba import njit
    >>> import numpy as np
    >>> from qgs.integrators.integrate import integrate_runge_kutta
    >>> a = 0.25
    >>> F = 16.
    >>> G = 3.
    >>> b = 6.
    >>> # Lorenz 84 example
    >>> @njit
    ... def fL84(t, x):
    ...     xx = -x[1] ** 2 - x[2] ** 2 - a * x[0] + a * F
    ...     yy = x[0] * x[1] - b * x[0] * x[2] - x[1] + G
    ...     zz = b * x[0] * x[1] + x[0] * x[2] - x[2]
    ...     return np.array([xx, yy, zz])
    >>> # no ic
    >>> # write_steps is 1 by default
    >>> tt, traj = integrate_runge_kutta(fL84, t0=0., t=10., dt=0.1)  # 101 steps
    >>> print(traj.shape)
    (3, 101)
    >>> # 1 ic
    >>> ic = 0.1 * np.random.randn(3)
    >>> tt, traj = integrate_runge_kutta(fL84, t0=0., t=10., dt=0.1, ic=ic)  # 101 steps
    >>> print(ic.shape)
    (3,)
    >>> print(traj.shape)
    (3, 101)
    >>> # 4 ic
    >>> ic = 0.1 * np.random.randn(4, 3)
    >>> tt, traj = integrate_runge_kutta(fL84, t0=0., t=10., dt=0.1, ic=ic)  # 101 steps
    >>> print(ic.shape)
    (4, 3)
    >>> print(traj.shape)
    (4, 3, 101)
    """

    if ic is None:
        i = 1
        while True:
            ic = np.zeros(i)
            try:
                x = f(0., ic)
            except:
                i += 1
            else:
                break

        i = len(f(0., ic))
        ic = np.zeros(i)

    if len(ic.shape) == 1:
        ic = ic.reshape((1, -1))

    # Default is RK4
    if a is None and b is None and c is None:
        c = np.array([0., 0.5, 0.5, 1.])
        b = np.array([1./6, 1./3, 1./3, 1./6])
        a = np.zeros((len(c), len(b)))
        a[1, 0] = 0.5
        a[2, 1] = 0.5
        a[3, 2] = 1.

    if forward:
        time_direction = 1
    else:
        time_direction = -1

    time = np.concatenate((np.arange(t0, t, dt), np.full((1,), t)))

    recorded_traj = _integrate_runge_kutta_jit(f, time, ic, time_direction, write_steps, b, c, a)

    if write_steps > 0:
        if forward:
            if time[::write_steps][-1] == time[-1]:
                return time[::write_steps], np.squeeze(recorded_traj)
            else:
                return np.concatenate((time[::write_steps], np.full((1,), t))), np.squeeze(recorded_traj)
        else:
            rtime = reverse(time[::-write_steps])
            if rtime[0] == time[0]:
                return rtime, np.squeeze(recorded_traj)
            else:
                return np.concatenate((np.full((1,), t0), rtime)), np.squeeze(recorded_traj)
    else:
        return time[-1], np.squeeze(recorded_traj)


@njit
def _integrate_runge_kutta_jit(f, time, ic, time_direction, write_steps, b, c, a):

    n_traj = ic.shape[0]
    n_dim = ic.shape[1]

    s = len(b)

    if write_steps == 0:
        n_records = 1
    else:
        tot = time[::write_steps]
        n_records = len(tot)
        if tot[-1] != time[-1]:
            n_records += 1

    recorded_traj = np.zeros((n_traj, n_dim, n_records))
    if time_direction == -1:
        directed_time = reverse(time)
    else:
        directed_time = time

    for i_traj in range(n_traj):
        y = ic[i_traj].copy()
        k = np.zeros((s, n_dim))
        iw = 0
        for ti, (tt, dt) in enumerate(zip(directed_time[:-1], np.diff(directed_time))):

            if write_steps > 0 and np.mod(ti, write_steps) == 0:
                recorded_traj[i_traj, :, iw] = y
                iw += 1

            k.fill(0.)
            for i in range(s):
                y_s = y + dt * a[i] @ k
                k[i] = f(tt + c[i] * dt, y_s)
            y_new = y + dt * b @ k
            y = y_new

        recorded_traj[i_traj, :, -1] = y

    return recorded_traj[:, :, ::time_direction]


@njit
def _tangent_linear_system(fjac, t, xs, x, adjoint):
    if adjoint:
        return fjac(t, xs).transpose() @ x
    else:
        return fjac(t, xs) @ x


# a function that return always zero
@njit
def _zeros_func(t, x):
    return np.zeros_like(x)


def integrate_runge_kutta_tgls(f, fjac, t0, t, dt, ic=None, tg_ic=None,
                               forward=True, adjoint=False, inverse=False, boundary=None,
                               write_steps=1, b=None, c=None, a=None):
    """Integrate simultaneously the ordinary differential equations (ODEs)

    .. math:: \dot{\\boldsymbol{x}} = \\boldsymbol{f}(t, \\boldsymbol{x})

    and its tangent linear model, i.e. the linearized ODEs

    .. math :: \dot{\\boldsymbol{\delta x}} = \\boldsymbol{\mathrm{J}}(t, \\boldsymbol{x}) \cdot \\boldsymbol{\delta x}

    where :math:`\\boldsymbol{\mathrm{J}} = \\frac{\partial \\boldsymbol{f}}{\partial \\boldsymbol{x}}` is the
    Jacobian matrix of :math:`\\boldsymbol{f}`, with a specified `Runge-Kutta method`_.
    To solve this equation, one has to integrate simultaneously both ODEs.

    The function :math:`\\boldsymbol{f}` and :math:`\\boldsymbol{J}` should
    be `Numba`_ jitted functions. These functions must have a signature ``f(t, x)`` and ``J(t, x)`` where ``x`` is
    the state value and ``t`` is the time.

    .. _Runge-Kutta method: https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
    .. _Numba: https://numba.pydata.org/
    .. _fundamental matrix of solutions: https://en.wikipedia.org/wiki/Fundamental_matrix_(linear_differential_equation)

    Parameters
    ----------
    f: callable
        The `Numba`_-jitted function :math:`\\boldsymbol{f}`.
        Should have the signature``f(t, x)`` where ``x`` is the state value and ``t`` is the time.
    fjac: callable
        The `Numba`_-jitted Jacobian :math:`\\boldsymbol{J}`.
        Should have the signature``J(t, x)`` where ``x`` is the state value and ``t`` is the time.
    t0: float
        Initial time of the time integration. Corresponds to the initial conditions.
    t: float
        Final time of the time integration. Corresponds to the final conditions.
    dt: float
        Timestep of the integration.
    ic: None or ~numpy.ndarray(float), optional
        Initial (or final) conditions of the ODEs :math:`\dot{\\boldsymbol{x}} = \\boldsymbol{f}(t, \\boldsymbol{x})`.
        Can be a 1D or a 2D array:

        * 1D: Provide a single initial condition.
          Should be of shape (`n_dim`,) where `n_dim` = :math:`\mathrm{dim}(\\boldsymbol{x})`.
        * 2D: Provide an ensemble of initial condition.
          Should be of shape (`n_traj`, `n_dim`) where `n_dim` = :math:`\mathrm{dim}(\\boldsymbol{x})`,
          and where `n_traj` is the number of initial conditions.

        If `None`, use a zero initial condition. Default to `None`.
        If the `forward` argument is `False`, it specifies final conditions.
    tg_ic: None or ~numpy.ndarray(float), optional
        Initial (or final) conditions of the linear ODEs
        :math:`\dot{\\boldsymbol{\delta x}} = \\boldsymbol{\mathrm{J}}(t, \\boldsymbol{x}) \cdot \\boldsymbol{\delta x}`. \n
        Can be a 1D, a 2D or a 3D array:

        * 1D: Provide a single initial condition. This initial condition of the linear ODEs will be the same used for each
          initial condition `ic` of the ODEs :math:`\dot{\\boldsymbol{x}} = \\boldsymbol{f}(t, \\boldsymbol{x})`
          Should be of shape (`n_dim`,) where `n_dim` = :math:`\mathrm{dim}(\\boldsymbol{x})`.
        * 2D: Two sub-cases:

            + If `tg_ic.shape[0]`=`ic.shape[0]`, assumes that each initial condition `ic[i]` of :math:`\dot{\\boldsymbol{x}} = \\boldsymbol{f}(t, \\boldsymbol{x})`,
              correspond to a different initial condition `tg_ic[i]`.
            + Else, assumes and integrate an ensemble of `n_tg_traj` initial condition of the linear ODEs for each
              initial condition of :math:`\dot{\\boldsymbol{x}} = \\boldsymbol{f}(t, \\boldsymbol{x})`.
        * 3D: An array of shape (`n_traj`, `n_tg_traj`, `n_dim`) which provide an ensemble of `n_tg_ic` initial conditions
          specific to each of the `n_traj` initial conditions of :math:`\dot{\\boldsymbol{x}} = \\boldsymbol{f}(t, \\boldsymbol{x})`.

        If `None`, use the identity matrix as initial condition, returning the `fundamental matrix of solutions`_ of the
        linear ODEs.
        Default to `None`.
        If the `forward` argument is `False`, it specifies final conditions.
    forward: bool, optional
        Whether to integrate the ODEs forward or backward in time. In case of backward integration, the
        initial condition `ic` becomes a final condition. Default to forward integration.
    adjoint: bool, optional
        Wheter to integrate the tangent :math:`\dot{\\boldsymbol{\delta x}} = \\boldsymbol{\mathrm{J}}(t, \\boldsymbol{x}) \cdot \\boldsymbol{\delta x}`
        or the adjoint linear model :math:`\dot{\\boldsymbol{\delta x}} = \\boldsymbol{\mathrm{J}}^T(t, \\boldsymbol{x}) \cdot \\boldsymbol{\delta x}`.
        Integrate the tangent model by default.
    inverse: bool, optional
        Wheter or not to invert the Jacobian matrix
        :math:`\\boldsymbol{\mathrm{J}}(t, \\boldsymbol{x}) \\rightarrow \\boldsymbol{\mathrm{J}}^{-1}(t, \\boldsymbol{x})`.
        `False` by default.
    boundary: None or callable, optional
        Allow to add a inhomogeneous term to linear ODEs:
        :math:`\dot{\\boldsymbol{\delta x}} = \\boldsymbol{\mathrm{J}}(t, \\boldsymbol{x}) \cdot \\boldsymbol{\delta x} + \Psi(t, \\boldsymbol{x})`.
        The boundary :math:`\Psi` should have the same signature as :math:`\\boldsymbol{\mathrm{J}}`, i.e.  ``func(t, x)``.
        If `None`, don't add anything (homogeneous case). `None` by default.
    write_steps: int, optional
        Save the state of the integration in memory every `write_steps` steps. The other intermediary
        steps are lost. It determines the size of the returned objects. Default is 1.
        Set to 0 to return only the final state.
    b: None or ~numpy.ndarray, optional
        Vector of coefficients :math:`b_i` of the `Runge-Kutta method`_ .
        If `None`, use the classic RK4 method coefficients. Default to `None`.
    c: None or ~numpy.ndarray, optional
        Matrix of coefficients :math:`c_{i,j}` of the `Runge-Kutta method`_ .
        If `None`, use the classic RK4 method coefficients. Default to `None`.
    a: None or ~numpy.ndarray, optional
        Vector of coefficients :math:`a_i` of the `Runge-Kutta method`_ .
        If `None`, use the classic RK4 method coefficients. Default to `None`.

    Returns
    -------
    time, traj, tg_traj: ~numpy.ndarray
        The result of the integration:

        * **time:** Time at which the state of the system was saved. Array of shape (`n_step`,) where
          `n_step` is the number of saved states of the integration.
        * **traj:** Saved states of the ODEs. 3D array of shape (`n_traj`, `n_dim`, `n_steps`). If `n_traj` = 1,
          a 2D array of shape (`n_dim`, `n_steps`) is returned instead.
        * **tg_traj:** Saved states of the linear ODEs.
          Depending on the input initial conditions of both ODEs,
          it is at maximum a 4D array of shape (`n_traj`, `n_tg_traj `n_dim`, `n_steps`).
          If one of the dimension is 1, it is squeezed.


    Examples
    --------

    >>> from numba import njit
    >>> import numpy as np
    >>> from qgs.integrators.integrate import integrate_runge_kutta_tgls
    >>> a = 0.25
    >>> F = 16.
    >>> G = 3.
    >>> b = 6.
    >>> # Lorenz 84 example
    >>> @njit
    ... def fL84(t, x):
    ...     xx = -x[1] ** 2 - x[2] ** 2 - a * x[0] + a * F
    ...     yy = x[0] * x[1] - b * x[0] * x[2] - x[1] + G
    ...     zz = b * x[0] * x[1] + x[0] * x[2] - x[2]
    ...     return np.array([xx, yy, zz])
    >>> @njit
    ... def DfL84(t, x):
    ...    return np.array([[     -a        , -2. * x[1], -2. * x[2]],
    ...                      [x[1] - b * x[2], -1. + x[0], -b * x[0]],
    ...                      [b * x[1] + x[2],  b * x[0], -1. + x[0]]])
    >>> # 4 ic, no tg_ic (fundamental matrix computation of an ensemble of ic)
    >>> ic = 0.1 * np.random.randn(4, 3)
    >>> tt, traj, dtraj = integrate_runge_kutta_tgls(fL84, DfL84, t0=0., t=10., dt=0.1,
    ...                                              ic=ic, write_steps=1) # 101 steps
    >>> print(ic.shape)
    (4, 3)
    >>> print(traj.shape)
    (4, 3, 101)
    >>> print(dtraj.shape)
    (4, 3, 3, 101)
    >>> # 1 ic, 1 tg_ic (one ic and its tg_ic)
    >>> ic = 0.1 * np.random.randn(3)
    >>> tg_ic = 0.001 * np.random.randn(3)
    >>> tt, traj, dtraj = integrate_runge_kutta_tgls(fL84, DfL84, t0=0., t=10., dt=0.1,
    ...                                              ic=ic, tg_ic=tg_ic) # 101 steps
    >>> print(ic.shape)
    (3,)
    >>> print(tg_ic.shape)
    (3,)
    >>> print(traj.shape)
    (3, 101)
    >>> print(dtraj.shape)
    (3, 101)
    >>> # 4 ic, 1 same tg_ic (an ensemble of ic with the same tg_ic)
    >>> ic = 0.1 * np.random.randn(4, 3)
    >>> tt, traj, dtraj = integrate_runge_kutta_tgls(fL84, DfL84, t0=0., t=10., dt=0.1,
    ...                                              ic=ic, tg_ic=tg_ic) # 101 steps
    >>> print(ic.shape)
    (4, 3)
    >>> print(tg_ic.shape)
    (3,)
    >>> print(traj.shape)
    (4, 3, 101)
    >>> print(dtraj.shape)
    (4, 3, 101)
    >>> # 1 ic, 4 tg_ic (an ic with an ensemble of tg_ic in its tangent space)
    >>> ic = 0.1 * np.random.randn(3)
    >>> tg_ic = 0.001 * np.random.randn(4, 3)
    >>> tt, traj, dtraj = integrate_runge_kutta_tgls(fL84, DfL84, t0=0., t=10., dt=0.1,
    ...                                              ic=ic, tg_ic=tg_ic) # 101 steps
    >>> print(ic.shape)
    (3,)
    >>> print(tg_ic.shape)
    (4, 3)
    >>> print(traj.shape)
    (3, 101)
    >>> print(dtraj.shape)
    (4, 3, 101)
    >>> # 2 ic, same 4 tg_ic (an ensemble of 2 ic, both with the same ensemble
    >>> # of tg_ic in their tangent space)
    >>> ic = 0.1 * np.random.randn(2, 3)
    >>> tg_ic = 0.001 * np.random.randn(4, 3)
    >>> tt, traj, dtraj = integrate_runge_kutta_tgls(fL84, DfL84, t0=0., t=10., dt=0.1,
    ...                                              ic=ic, tg_ic=tg_ic) # 101 steps
    >>> print(ic.shape)
    (2, 3)
    >>> print(tg_ic.shape)
    (4, 3)
    >>> print(traj.shape)
    (2, 3, 101)
    >>> print(dtraj.shape)
    (2, 4, 3, 101)
    >>> # 2 ic, 4 different tg_ic (an ensemble of 2 ic, with different ensemble
    >>> # of tg_ic in their tangent space)
    >>> ic = 0.1 * np.random.randn(2, 3)
    >>> tg_ic = 0.001 * np.random.randn(2, 4, 3)
    >>> tt, traj, dtraj = integrate_runge_kutta_tgls(fL84, DfL84, t0=0., t=10., dt=0.1,
    ...                                              ic=ic, tg_ic=tg_ic) # 101 steps
    >>> print(ic.shape)
    (2, 3)
    >>> print(tg_ic.shape)
    (2, 4, 3)
    >>> print(traj.shape)
    (2, 3, 101)
    >>> print(dtraj.shape)
    (2, 4, 3, 101)

    """

    if ic is None:
        i = 1
        while True:
            ic = np.zeros(i)
            try:
                x = f(0., ic)
            except:
                i += 1
            else:
                break

        i = len(f(0., ic))
        ic = np.zeros(i)

    if len(ic.shape) == 1:
        ic = ic.reshape((1, -1))

    n_traj = ic.shape[0]

    if tg_ic is None:
        tg_ic = np.eye(ic.shape[1])

    tg_ic_sav = tg_ic.copy()

    if len(tg_ic.shape) == 1:
        tg_ic = tg_ic.reshape((1, -1, 1))
        ict = tg_ic.copy()
        for i in range(n_traj-1):
            ict = np.concatenate((ict, tg_ic))
        tg_ic = ict
    elif len(tg_ic.shape) == 2:
        if tg_ic.shape[0] == n_traj:
            tg_ic = tg_ic[..., np.newaxis]
        else:
            tg_ic = tg_ic[np.newaxis, ...]
            tg_ic = np.swapaxes(tg_ic, 1, 2)
            ict = tg_ic.copy()
            for i in range(n_traj-1):
                ict = np.concatenate((ict, tg_ic))
            tg_ic = ict
    elif len(tg_ic.shape) == 3:
        if tg_ic.shape[1] != ic.shape[1]:
            tg_ic = np.swapaxes(tg_ic, 1, 2)

    # Default is RK4
    if a is None and b is None and c is None:
        c = np.array([0., 0.5, 0.5, 1.])
        b = np.array([1./6, 1./3, 1./3, 1./6])
        a = np.zeros((len(c), len(b)))
        a[1, 0] = 0.5
        a[2, 1] = 0.5
        a[3, 2] = 1.

    if forward:
        time_direction = 1
    else:
        time_direction = -1

    time = np.concatenate((np.arange(t0, t, dt), np.full((1,), t)))

    if boundary is None:
        boundary = _zeros_func

    inv = 1.
    if inverse:
        inv *= -1.

    recorded_traj, recorded_fmatrix = _integrate_runge_kutta_tgls_jit(f, fjac, time, ic, tg_ic,
                                                                      time_direction, write_steps,
                                                                      b, c, a, adjoint, inv, boundary)

    if len(tg_ic_sav.shape) == 2:
        if recorded_fmatrix.shape[1:3] != tg_ic_sav.shape:
            recorded_fmatrix = np.swapaxes(recorded_fmatrix, 1, 2)

    elif len(tg_ic_sav.shape) == 3:
        if tg_ic_sav.shape[1] != ic.shape[1]:
            if recorded_fmatrix.shape[:3] != tg_ic_sav.shape:
                recorded_fmatrix = np.swapaxes(recorded_fmatrix, 1, 2)

    if write_steps > 0:
        if forward:
            if time[::write_steps][-1] == time[-1]:
                return time[::write_steps], np.squeeze(recorded_traj), np.squeeze(recorded_fmatrix)
            else:
                return np.concatenate((time[::write_steps], np.full((1,), t))), np.squeeze(recorded_traj),\
                       np.squeeze(recorded_fmatrix)
        else:
            rtime = reverse(time[::-write_steps])
            if rtime[0] == time[0]:
                return rtime, np.squeeze(recorded_traj), np.squeeze(recorded_fmatrix)
            else:
                return np.concatenate((np.full((1,), t0), rtime)), np.squeeze(recorded_traj),\
                       np.squeeze(recorded_fmatrix)

    else:
        return time[-1], np.squeeze(recorded_traj), np.squeeze(recorded_fmatrix)


@njit
def _integrate_runge_kutta_tgls_jit(f, fjac, time, ic, tg_ic, time_direction, write_steps, b, c, a,
                                    adjoint, inverse, boundary):

    n_traj = ic.shape[0]
    n_dim = ic.shape[1]

    s = len(b)

    if write_steps == 0:
        n_records = 1
    else:
        tot = time[::write_steps]
        n_records = len(tot)
        if tot[-1] != time[-1]:
            n_records += 1
    recorded_traj = np.zeros((n_traj, n_dim, n_records))
    recorded_fmatrix = np.zeros((n_traj, tg_ic.shape[1], tg_ic.shape[2], n_records))
    if time_direction == -1:
        directed_time = reverse(time)
    else:
        directed_time = time

    for i_traj in range(n_traj):
        y = ic[i_traj].copy()
        fm = tg_ic[i_traj].copy()
        recorded_traj[i_traj, :, 0] = ic[i_traj]
        recorded_fmatrix[i_traj, :, :, 0] = tg_ic[i_traj]
        k = np.zeros((s, n_dim))
        km = np.zeros((s, tg_ic.shape[1], tg_ic.shape[2]))
        iw = 0
        for ti, (tt, dt) in enumerate(zip(directed_time[:-1], np.diff(directed_time))):

            if write_steps > 0 and np.mod(ti, write_steps) == 0:
                recorded_traj[i_traj, :, iw] = y
                recorded_fmatrix[i_traj, :, :, iw] = fm
                iw += 1

            k.fill(0.)
            km.fill(0.)
            for i in range(s):
                y_s = y + dt * a[i] @ k
                k[i] = f(tt + c[i] * dt, y_s)
                km_s = fm.copy()
                for j in range(len(a[i])):
                    km_s += dt * a[i, j] * km[j]
                hom = inverse * _tangent_linear_system(fjac, tt + c[i] * dt, y_s, km_s, adjoint)
                inhom = boundary(tt + c[i] * dt, y_s)
                km[i] = (hom.T + inhom.T).T
            y_new = y + dt * b @ k
            fm_new = fm.copy()
            for j in range(len(b)):
                fm_new += dt * b[j] * km[j]
            y = y_new
            fm = fm_new

        recorded_traj[i_traj, :, -1] = y
        recorded_fmatrix[i_traj, :, :, -1] = fm

    return recorded_traj[:, :, ::time_direction], recorded_fmatrix[:, :, :, ::time_direction]


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from scipy.integrate import odeint

    @njit
    def f(t, x):
        return - np.array([1., 2., 3.]) * x

    def fr(x, t):
        return f(t, x)

    ic = np.random.randn(6).reshape(2, 3)

    time, r = integrate_runge_kutta(f, 0., 10., 0.01, ic=ic, write_steps=3)

    t = np.arange(0., 10., 0.01)
    t = np.concatenate((t[::3], np.full((1,), 10.)))
    rl = list()
    for i in range(ic.shape[0]):
        rl.append(odeint(fr, ic[i], t).T)

    plt.figure()
    for i in range(ic.shape[0]):
        p, = plt.plot(time, r[i, 0])
        c = p.get_color()
        plt.plot(t, rl[i][0], color=c, ls='--')
        for j in range(1, ic.shape[1]):
            p, = plt.plot(time, r[i, j], color=c)
            plt.plot(t, rl[i][j], color=c, ls='--')

    timet, rt = integrate_runge_kutta(f, 0., 10., 0.01, ic=ic, forward=False, write_steps=3)
    rlt = list()
    for i in range(ic.shape[0]):
        rlt.append(odeint(fr, ic[i], reverse(t)).T)

    plt.figure()
    for i in range(ic.shape[0]):
        p, = plt.plot(timet, rt[i, 0])
        c = p.get_color()
        plt.plot(t, reverse(rlt[i][0]), color=c, ls='--')
        for j in range(1, ic.shape[1]):
            p, = plt.plot(timet, rt[i, j], color=c)
            plt.plot(t, reverse(rlt[i][j]), color=c, ls='--')

    tt, re = integrate_runge_kutta(f, 0., 10., 0.01, ic=ic, write_steps=0)
    print(tt)
    print(r[0, :, -1], re[0])
    plt.show(block=False)

    a = 0.25
    F = 16.
    G = 3.
    b = 6.


    @njit
    def DfL84(t, x):
        return np.array([[     -a        , -2. * x[1], -2. * x[2]],
                         [x[1] - b * x[2], -1. + x[0], -b * x[0]],
                         [b * x[1] + x[2],  b * x[0], -1. + x[0]]])

    @njit
    def fL84(t, x):
        xx = -x[1] ** 2 - x[2] ** 2 - a * x[0] + a * F
        yy = x[0] * x[1] - b * x[0] * x[2] - x[1] + G
        zz = b * x[0] * x[1] + x[0] * x[2] - x[2]
        return np.array([xx, yy, zz])

    def fL84r(x, t):
        return fL84(t, x)

    tt, ic_L84 = integrate_runge_kutta(fL84, 0., 10000., 0.01, write_steps=0)

    print(ic_L84)

    ttt1, irkt1, fm_irkt1 = integrate_runge_kutta_tgls(fL84, DfL84, 0., 0.01, 0.001, ic=ic_L84, write_steps=1)
    bttt1, birkt1, bfm_irkt1 = integrate_runge_kutta_tgls(fL84, DfL84, 0., 0.01, 0.001, ic=irkt1[:, -1],
                                                          forward=False, write_steps=1, adjoint=True, inverse=True)

    plt.figure()
    for i in range(len(ic_L84)):
        for j in range(len(ic_L84)):
            p, = plt.plot(ttt1, fm_irkt1[i, j])
            c = p.get_color()
            plt.plot(bttt1, bfm_irkt1[i, j], ls='--', color=c)

    plt.figure()
    for i in range(len(ic_L84)):
        p, = plt.plot(ttt1, irkt1[i])
        c = p.get_color()
        plt.plot(bttt1, birkt1[i], ls='--', color=c)

    vec = np.random.randn(3)
    vec = vec/np.linalg.norm(vec)
    for i in range(1,13):
        lam = 2.**(-i)

        dy = lam * vec

        tx, dx = integrate_runge_kutta(fL84, 0., 0.001, 0.001, ic=ic_L84, write_steps=0)
        txp, dxp = integrate_runge_kutta(fL84, 0., 0.001, 0.001, ic=ic_L84+dy, write_steps=0)

        dy1 = dxp - dx

        txtl, dxtl, dytl = integrate_runge_kutta_tgls(fL84, DfL84, 0., 0.001, 0.001, ic=ic_L84, tg_ic=dy, write_steps=0)

        print('Perturbation size: ', dy @ dy)
        print('Resulting difference in trajectory: (eps ~ 2^-'+str(i)+')')
        print('diff: ', dy1 @ dy1)
        print('tl: ', dytl @ dytl)
        print('ratio: ', (dy1 @ dy1) / (dytl @ dytl))
