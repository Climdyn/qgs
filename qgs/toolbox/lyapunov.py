
"""
    Lyapunov module
    =================

    Module with the classes of multi-thread the computation of the various
    `Lyapunov vectors`_ and `exponents`_. Integrate using the `Runge-Kutta method`_
    defined in the :mod:`~.integrators.integrate` module.
    See :cite:`lyap-KP2012` for more details on the Lyapunov vectors theoretical framework.

    Module classes
    --------------

    * :class:`LyapunovsEstimator` to estimate the Backward and Forward Lyapunov Vectors (BLVs and FLVs) along a trajectory
    * :class:`CovariantLyapunovsEstimator` to estimate the Covariant Lyapunov Vectors (CLVs) along a trajectory
    
    .. _Lyapunov vectors: https://en.wikipedia.org/wiki/Lyapunov_vector
    .. _exponents: https://en.wikipedia.org/wiki/Lyapunov_exponent
    .. _Runge-Kutta method: https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
    .. _Numba: https://numba.pydata.org/


    References
    ----------

    .. bibliography:: ../model/ref.bib
        :labelprefix: LYAP-
        :keyprefix: lyap-
"""

from numba import njit
import numpy as np
import qgs.integrators.integrate as integrate
from qgs.functions.util import normalize_matrix_columns, solve_triangular_matrix, reverse

import multiprocessing


# TODO: change the usage of np.squeeze in the return of the estimators. Use specific shape descriptors instead.

class LyapunovsEstimator(object):
    """Class to compute the Forward and Backward `Lyapunov vectors`_ and `exponents`_ along a trajectory of a dynamical system

    .. math:: \\dot{\\boldsymbol{x}} = \\boldsymbol{f}(t, \\boldsymbol{x})

    with a set of :class:`LyapProcess` and a specified `Runge-Kutta method`_.
    The tangent linear model must also be provided. I.e. one must provide the linearized ODEs

    .. math :: \\dot{\\boldsymbol{\\delta x}} = \\boldsymbol{\\mathrm{J}}(t, \\boldsymbol{x}) \\cdot \\boldsymbol{\\delta x}

    where :math:`\\boldsymbol{\\mathrm{J}} = \\frac{\\partial \\boldsymbol{f}}{\\partial \\boldsymbol{x}}` is the
    Jacobian matrix of :math:`\\boldsymbol{f}`.
    The method used to compute the Lyapunov vectors is the one introduced by
    Benettin et al. :cite:`lyap-BGGS1980`.


    Parameters
    ----------
    num_threads: None or int, optional
        Number of :class:`LyapProcess` workers (threads) to use. If `None`, use the number of machine's
        cores available. Default to `None`.
    b: None or ~numpy.ndarray, optional
        Vector of coefficients :math:`b_i` of the `Runge-Kutta method`_ .
        If `None`, use the classic RK4 method coefficients. Default to `None`.
    c: None or ~numpy.ndarray, optional
        Matrix of coefficients :math:`c_{i,j}` of the `Runge-Kutta method`_ .
        If `None`, use the classic RK4 method coefficients. Default to `None`.
    a: None or ~numpy.ndarray, optional
        Vector of coefficients :math:`a_i` of the `Runge-Kutta method`_ .
        If `None`, use the classic RK4 method coefficients. Default to `None`.
    number_of_dimensions: None or int, optional
        Allow to hardcode the dynamical system dimension. If `None`, evaluate the dimension from the
        callable :attr:`func`. Default to `None`.

    Attributes
    ----------
    num_threads: int
        Number of :class:`LyapProcess` workers (threads) to use.
    b: ~numpy.ndarray
        Vector of coefficients :math:`b_i` of the `Runge-Kutta method`_ .
    c: ~numpy.ndarray
        Matrix of coefficients :math:`c_{i,j}` of the `Runge-Kutta method`_ .
    a: ~numpy.ndarray
        Vector of coefficients :math:`a_i` of the `Runge-Kutta method`_ .
    n_dim: int
        Dynamical system dimension.
    n_vec: int
        The number of Lyapunov vectors to compute.
    n_traj: int
        The number of trajectories (initial conditions) computed at the last estimation
        performed by the estimator.
    n_records: int
        The number of saved states of the last estimation performed by the estimator.
    ic: ~numpy.ndarray
        Store the estimator initial conditions.
    func: callable
        Last function :math:`\\boldsymbol{f}` used by the estimator.
    func_jac: callable
        Last Jacobian matrix function :math:`\\boldsymbol{J}` used by the estimator.
    """

    def __init__(self, num_threads=None, b=None, c=None, a=None, number_of_dimensions=None):

        if num_threads is None:
            self.num_threads = multiprocessing.cpu_count()
        else:
            self.num_threads = num_threads

        # Default is RK4
        if a is None and b is None and c is None:
            self.c = np.array([0., 0.5, 0.5, 1.])
            self.b = np.array([1./6, 1./3, 1./3, 1./6])
            self.a = np.zeros((len(self.c), len(self.b)))
            self.a[1, 0] = 0.5
            self.a[2, 1] = 0.5
            self.a[3, 2] = 1.
        else:
            self.a = a
            self.b = b
            self.c = c

        self.ic = None
        self._time = None
        self._pretime = None

        self._recorded_traj = None
        self._recorded_exp = None
        self._recorded_vec = None
        self.n_traj = 0
        self.n_dim = number_of_dimensions
        self.n_records = 0
        self.n_vec = 0
        self.write_steps = 0

        self._adjoint = False
        self._forward = -1
        self._inverse = 1.

        self.func = None
        self.func_jac = None

        self._ics_queue = None
        self._lyap_queue = None

        self._processes_list = list()

    def terminate(self):
        """Stop the workers (threads) and release the resources of the estimator."""

        for process in self._processes_list:

            process.terminate()
            process.join()

    def start(self):
        """Start or restart the workers (threads) of the estimator.

        Warnings
        --------
        If the estimator was not previously terminated, it will be terminated first in the case
        of a restart.
        """

        self.terminate()

        self._processes_list = list()
        self._ics_queue = multiprocessing.JoinableQueue()
        self._lyap_queue = multiprocessing.Queue()

        for i in range(self.num_threads):
            self._processes_list.append(LyapProcess(i, self.func, self.func_jac, self.b, self.c, self.a,
                                                    self._ics_queue, self._lyap_queue))

        for process in self._processes_list:
            process.daemon = True
            process.start()

    def set_bca(self, b=None, c=None, a=None, ic_init=True):
        """Set the coefficients of the `Runge-Kutta method`_ and restart the estimator.

        .. _Runge-Kutta method: https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods

        Parameters
        ----------
        b: None or ~numpy.ndarray, optional
            Vector of coefficients :math:`b_i` of the `Runge-Kutta method`_ .
            If `None`, does not reinitialize these coefficients.
        c: None or ~numpy.ndarray, optional
            Matrix of coefficients :math:`c_{i,j}` of the `Runge-Kutta method`_ .
            If `None`, does not reinitialize these coefficients.
        a: None or ~numpy.ndarray, optional
            Vector of coefficients :math:`a_i` of the `Runge-Kutta method`_ .
            If `None`, does not reinitialize these coefficients.
        ic_init: bool, optional
            Re-initialize or not the initial conditions of the estimator. Default to `True`.
        """

        if a is not None:
            self.a = a
        if b is not None:
            self.b = b
        if c is not None:
            self.c = c
        if ic_init:
            self.ic = None
        self.start()

    def set_func(self, f, fjac):
        """Set the `Numba`_-jitted function :math:`\\boldsymbol{f}` and Jacobian matrix function
        :math:`\\boldsymbol{\\mathrm{J}}` to integrate.

        .. _Numba: https://numba.pydata.org/

        Parameters
        ----------
        f: callable
            The `Numba`_-jitted function :math:`\\boldsymbol{f}`.
            Should have the signature ``f(t, x)`` where ``x`` is the state value and ``t`` is the time.
        fjac: callable
            The `Numba`_-jitted Jacobian matrix function :math:`\\boldsymbol{J}`.
            Should have the signature ``J(t, x)`` where ``x`` is the state value and ``t`` is the time.

        Warnings
        --------
        This function restarts the estimator!
        """

        self.func = f
        self.func_jac = fjac
        self.start()

    def compute_lyapunovs(self, t0, tw, t, dt, mdt, ic=None, write_steps=1, n_vec=None, forward=False, adjoint=False,
                          inverse=False):
        """Estimate the Lyapunov vectors using the Benettin algorithm along a given trajectory, always integrating the said trajectory
        forward in time from `ic` at `t0` to time `t`.
        The result of the estimation can be obtained afterward by calling :meth:`get_lyapunovs`.
        If `forward` is `True`, it yields the Forward Lyapunov Vectors (FLVs) between `t0` and `tw`, otherwise, returns the Backward
        Lyapunov Vectors (BLVs) between `tw` and `t`.

        Parameters
        ----------
        t0: float
            Initial time of the time integration. Corresponds to the initial condition's `ic` time.
        tw: float
            Time at which the algorithm start to store the Lyapunov vectors. Define thus also the transient before the which the Lyapunov
            vectors are considered as having not yet converged. Must be between `t0` and `t`.
        t: float
            Final time of the time integration. Corresponds to the final condition.
        dt: float
            Timestep of the integration.
        mdt: float
            Micro-timestep to integrate the tangent linear equation between the nonlinear system `dt` timesteps. Should be smaller or equal to `dt`.
        ic: None or ~numpy.ndarray(float), optional
            Initial conditions of the system. Can be a 1D or a 2D array:

            * 1D: Provide a single initial condition.
              Should be of shape (`n_dim`,) where `n_dim` = :math:`\\mathrm{dim}(\\boldsymbol{x})`.
            * 2D: Provide an ensemble of initial condition.
              Should be of shape (`n_traj`, `n_dim`) where `n_dim` = :math:`\\mathrm{dim}(\\boldsymbol{x})`,
              and where `n_traj` is the number of initial conditions.

            If `None`, use the initial conditions stored in :attr:`ic`.
            If then :attr:`ic` is `None`, use a zero initial condition.
            Default to `None`.
        forward: bool, optional
            If `True`, yield the `Forward Lyapunov Vectors` (FLVs) between `t0` and `tw`.
            If `False`, yield the `Backward Lyapunov Vectors` (BLVs) between `tw` and `t`.
            Default to `False`, i.e. Backward Lyapunov Vectors estimation.
        adjoint: bool, optional
            If true, integrate the tangent :math:`\\dot{\\boldsymbol{\\delta x}} = \\boldsymbol{\\mathrm{J}}(t, \\boldsymbol{x}) \\cdot \\boldsymbol{\\delta x}` ,
            else, integrate the adjoint linear model :math:`\\dot{\\boldsymbol{\\delta x}} = \\boldsymbol{\\mathrm{J}}^T(t, \\boldsymbol{x}) \\cdot \\boldsymbol{\\delta x}`.
            Integrate the tangent model by default.
        inverse: bool, optional
            Whether or not to invert the Jacobian matrix
            :math:`\\boldsymbol{\\mathrm{J}}(t, \\boldsymbol{x}) \\rightarrow \\boldsymbol{\\mathrm{J}}^{-1}(t, \\boldsymbol{x})`.
            `False` by default.
        write_steps: int, optional
            Save the state of the integration in memory every `write_steps` steps. The other intermediary
            steps are lost. It determines the size of the returned objects. Default is 1.
            Set to 0 to return only the final state.
        n_vec: int, optional
            The number of Lyapunov vectors to compute. Should be smaller or equal to :attr:`n_dim`.
        """

        if self.func is None or self.func_jac is None:
            print('No function to integrate defined!')
            return 0

        if ic is None:
            i = 1
            while True:
                self.ic = np.zeros(i)
                try:
                    x = self.func(0., self.ic)
                except:
                    i += 1
                else:
                    break

            i = len(self.func(0., self.ic))
            self.ic = np.zeros(i)
        else:
            self.ic = ic

        if len(self.ic.shape) == 1:
            self.ic = self.ic.reshape((1, -1))

        self.n_traj = self.ic.shape[0]
        self.n_dim = self.ic.shape[1]
        if n_vec is not None:
            self.n_vec = n_vec
        else:
            self.n_vec = self.n_dim

        self._pretime = np.concatenate((np.arange(t0, tw, dt), np.full((1,), tw)))
        self._time = np.concatenate((np.arange(tw, t, dt), np.full((1,), t)))
        self.write_steps = write_steps

        if forward:
            self._forward = 1
        else:
            self._forward = -1

        self._adjoint = adjoint

        self._inverse = 1.
        if inverse:
            self._inverse *= -1.

        if write_steps == 0:
            self.n_records = 1
        else:
            if not forward:
                tot = self._time[::self.write_steps]
                self.n_records = len(tot)
                if tot[-1] != self._time[-1]:
                    self.n_records += 1
            else:
                tot = self._pretime[::self.write_steps]
                self.n_records = len(tot)
                if tot[-1] != self._pretime[-1]:
                    self.n_records += 1

        self._recorded_traj = np.zeros((self.n_traj, self.n_dim, self.n_records))
        self._recorded_vec = np.zeros((self.n_traj, self.n_dim, self.n_vec, self.n_records))
        self._recorded_exp = np.zeros((self.n_traj, self.n_vec, self.n_records))

        for i in range(self.n_traj):
            self._ics_queue.put((i, self._pretime, self._time, mdt, self.ic[i], self.n_vec, self.write_steps,
                                 self._forward, self._adjoint, self._inverse))

        self._ics_queue.join()

        for i in range(self.n_traj):
            args = self._lyap_queue.get()
            self._recorded_traj[args[0]] = args[1]
            self._recorded_exp[args[0]] = args[2]
            self._recorded_vec[args[0]] = args[3]

    def get_lyapunovs(self):
        """Returns the result of the previous Lyapunov vectors estimation.

        Returns
        -------
        time, traj, exponents, vectors: ~numpy.ndarray
            The result of the estimation:

            * **time:** Time at which the state of the system was saved. Array of shape (:attr:`n_records`,).
            * **traj:** Saved dynamical system states. 3D array of shape (:attr:`n_traj`, :attr:`n_dim`, :attr:`n_records`).
              If :attr:`n_traj` = 1, a 2D array of shape (:attr:`n_dim`, :attr:`n_records`) is returned instead.
            * **exponents:** Saved estimates of the local Lyapunov exponents along the trajectory. 3D array of shape (:attr:`n_traj`, :attr:`n_vec`, :attr:`n_records`).
              If :attr:`n_traj` = 1, a 2D array of shape (:attr:`n_vec`, :attr:`n_records`) is returned instead.
            * **vectors:** Saved estimates of the local Lyapunov vectors along the trajectory.
              Depending on the input initial conditions, it is maximum a 4D array of shape
              (:attr:`n_traj`, :attr:`n_dim`, :attr:`n_vec`, :attr:`n_records`).
              If one of the dimension is 1, it is squeezed.
        """

        if self._forward == -1:
            tt = self._time
        else:
            tt = self._pretime

        if self.write_steps > 0:
            if tt[::self.write_steps][-1] == tt[-1]:
                return tt[::self.write_steps], np.squeeze(self._recorded_traj), np.squeeze(self._recorded_exp), \
                       np.squeeze(self._recorded_vec)
            else:
                return np.concatenate((tt[::self.write_steps], np.full((1,), tt[-1]))), \
                       np.squeeze(self._recorded_traj), np.squeeze(self._recorded_exp), np.squeeze(self._recorded_vec)
        else:
            return tt[-1], np.squeeze(self._recorded_traj), np.squeeze(self._recorded_exp), \
                   np.squeeze(self._recorded_vec)


class LyapProcess(multiprocessing.Process):
    """:class:`LyapunovsEstimator`'s workers class. Allows to multi-thread Lyapunov vectors estimation.

    Parameters
    ----------
    processID: int
        Number identifying the worker.
    func: callable
        `Numba`_-jitted function to integrate assigned to the worker.
    func_jac: callable
        `Numba`_-jitted Jacobian matrix function to integrate assigned to the worker.
    b: ~numpy.ndarray, optional
        Vector of coefficients :math:`b_i` of the `Runge-Kutta method`_ .
    c: ~numpy.ndarray, optional
        Matrix of coefficients :math:`c_{i,j}` of the `Runge-Kutta method`_ .
    a: ~numpy.ndarray, optional
        Vector of coefficients :math:`a_i` of the `Runge-Kutta method`_ .
    ics_queue: multiprocessing.JoinableQueue
        Queue to which the worker ask for initial conditions and parameters input.
    lyap_queue: multiprocessing.Queue
        Queue to which the worker returns the estimation results.

    Attributes
    ----------
    processID: int
        Number identifying the worker.
    func: callable
        `Numba`_-jitted function to integrate assigned to the worker.
    func_jac: callable
        `Numba`_-jitted Jacobian matrix function to integrate assigned to the worker.
    b: ~numpy.ndarray
        Vector of coefficients :math:`b_i` of the `Runge-Kutta method`_ .
    c: ~numpy.ndarray
        Matrix of coefficients :math:`c_{i,j}` of the `Runge-Kutta method`_ .
    a: ~numpy.ndarray
        Vector of coefficients :math:`a_i` of the `Runge-Kutta method`_ .
    """
    def __init__(self, processID, func, func_jac, b, c, a, ics_queue, lyap_queue):

        super().__init__()
        self.processID = processID
        self._ics_queue = ics_queue
        self._lyap_queue = lyap_queue
        self.func = func
        self.func_jac = func_jac
        self.a = a
        self.b = b
        self.c = c

    def run(self):
        """Main worker computing routine. Perform the estimation with the fetched initial conditions and parameters."""

        while True:

            args = self._ics_queue.get()

            if args[7] == -1:
                recorded_traj, recorded_exp, recorded_vec = _compute_backward_lyap_jit(self.func, self.func_jac,
                                                                                       args[1], args[2], args[3],
                                                                                       args[4][np.newaxis, :], args[5],
                                                                                       args[6], args[8], args[9],
                                                                                       self.b, self.c, self.a)
            else:
                recorded_traj, recorded_exp, recorded_vec = _compute_forward_lyap_jit(self.func, self.func_jac,
                                                                                      args[1], args[2], args[3],
                                                                                      args[4][np.newaxis, :], args[5],
                                                                                      args[6], args[8], args[9],
                                                                                      self.b, self.c, self.a)

            self._lyap_queue.put((args[0], np.squeeze(recorded_traj), np.squeeze(recorded_exp),
                                  np.squeeze(recorded_vec)))

            self._ics_queue.task_done()


@njit
def _compute_forward_lyap_jit(f, fjac, time, posttime, mdt, ic, n_vec, write_steps, adjoint, inverse, b, c, a):

    ttraj = integrate._integrate_runge_kutta_jit(f, np.concatenate((time[:-1], posttime)), ic, 1, 1, b, c, a)
    recorded_traj, recorded_exp, recorded_vec = _compute_forward_lyap_traj_jit(f, fjac, time, posttime, ttraj, mdt,
                                                                               n_vec, write_steps, adjoint, inverse, b, c, a)
    return recorded_traj, recorded_exp, recorded_vec


@njit
def _compute_forward_lyap_traj_jit(f, fjac, time, posttime, ttraj, mdt, n_vec, write_steps, adjoint, inverse, b, c, a):

    traj = ttraj[:, :, :len(time)]
    posttraj = ttraj[:, :, len(time)-1:]

    n_traj = ttraj.shape[0]
    n_dim = ttraj.shape[1]

    Id = np.zeros((1, n_dim, n_dim))
    Id[0] = np.eye(n_dim)

    if write_steps == 0:
        n_records = 1
    else:
        tot = time[::write_steps]
        n_records = len(tot)
        if tot[-1] != time[-1]:
            n_records += 1

    recorded_vec = np.zeros((n_traj, n_dim, n_vec, n_records))
    recorded_traj = np.zeros((n_traj, n_dim, n_records))
    recorded_exp = np.zeros((n_traj, n_vec, n_records))

    rposttime = reverse(posttime)
    rtime = reverse(time)

    for i_traj in range(n_traj):

        y = np.zeros((1, n_dim))
        qr = np.linalg.qr(np.random.random((n_dim, n_vec)))
        q = qr[0]
        m_exp = np.zeros((n_dim))

        for ti, (tt, dt) in enumerate(zip(rposttime[:-1], np.diff(rposttime))):

            y[0] = posttraj[i_traj, :, -1-ti]
            subtime = np.concatenate((np.arange(tt + dt, tt, mdt), np.full((1,), tt)))
            y_new, prop = integrate._integrate_runge_kutta_tgls_jit(f, fjac, subtime, y, Id, -1, 0, b, c, a,
                                                                    adjoint, inverse, integrate._zeros_func)

            q_new = prop[0, :, :, 0] @ q
            qr = np.linalg.qr(q_new)
            q = qr[0]

        r = qr[1]
        iw = -1

        for ti, (tt, dt) in enumerate(zip(rtime[:-1], np.diff(rtime))):

            y[0] = traj[i_traj, :, -1-ti]
            m_exp = np.log(np.abs(np.diag(r)))/dt

            if write_steps > 0 and np.mod(ti, write_steps) == 0:
                recorded_exp[i_traj, :, iw] = m_exp
                recorded_traj[i_traj, :, iw] = y[0]
                recorded_vec[i_traj, :, :, iw] = q
                iw -= 1

            subtime = np.concatenate((np.arange(tt + dt, tt, mdt), np.full((1,), tt)))
            y_new, prop = integrate._integrate_runge_kutta_tgls_jit(f, fjac, subtime, y, Id, -1, 0, b, c, a,
                                                                    adjoint, inverse, integrate._zeros_func)

            q_new = prop[0, :, :, 0] @ q
            qr = np.linalg.qr(q_new)
            q = qr[0]
            r = qr[1]

        recorded_exp[i_traj, :, 0] = m_exp
        recorded_traj[i_traj, :, 0] = y[0]
        recorded_vec[i_traj, :, :, 0] = q

    return recorded_traj, recorded_exp, recorded_vec


@njit
def _compute_backward_lyap_jit(f, fjac, pretime, time, mdt, ic, n_vec, write_steps, adjoint, inverse, b, c, a):

    ttraj = integrate._integrate_runge_kutta_jit(f, np.concatenate((pretime[:-1], time)), ic, 1, 1, b, c, a)
    recorded_traj, recorded_exp, recorded_vec = _compute_backward_lyap_traj_jit(f, fjac, pretime, time, ttraj, mdt,
                                                                                n_vec, write_steps, adjoint, inverse, b, c, a)
    return recorded_traj, recorded_exp, recorded_vec


@njit
def _compute_backward_lyap_traj_jit(f, fjac, pretime, time, ttraj, mdt, n_vec, write_steps, adjoint, inverse, b, c, a):

    pretraj = ttraj[:, :, :len(pretime)]
    traj = ttraj[:, :, (len(pretime)-1):]

    n_traj = ttraj.shape[0]
    n_dim = ttraj.shape[1]

    Id = np.zeros((1, n_dim, n_dim))
    Id[0] = np.eye(n_dim)

    if write_steps == 0:
        n_records = 1
    else:
        tot = time[::write_steps]
        n_records = len(tot)
        if tot[-1] != time[-1]:
            n_records += 1

    recorded_vec = np.zeros((n_traj, n_dim, n_vec, n_records))
    recorded_traj = np.zeros((n_traj, n_dim, n_records))
    recorded_exp = np.zeros((n_traj, n_vec, n_records))

    for i_traj in range(n_traj):

        y = np.zeros((1, n_dim))
        y[0] = pretraj[i_traj, :, 0]
        qr = np.linalg.qr(np.random.random((n_dim, n_vec)))
        q = qr[0]
        m_exp = np.zeros((n_dim))

        for ti, (tt, dt) in enumerate(zip(pretime[:-1], np.diff(pretime))):

            subtime = np.concatenate((np.arange(tt, tt + dt, mdt), np.full((1,), tt + dt)))
            y_new, prop = integrate._integrate_runge_kutta_tgls_jit(f, fjac, subtime, y, Id, 1, 0, b, c, a,
                                                                    adjoint, inverse, integrate._zeros_func)
            y[0] = pretraj[i_traj, :, ti+1]
            q_new = prop[0, :, :, 0] @ q
            qr = np.linalg.qr(q_new)
            q = qr[0]

        r = qr[1]
        iw = 0

        for ti, (tt, dt) in enumerate(zip(time[:-1], np.diff(time))):

            m_exp = np.log(np.abs(np.diag(r)))/dt

            if write_steps > 0 and np.mod(ti, write_steps) == 0:
                recorded_exp[i_traj, :, iw] = m_exp
                recorded_traj[i_traj, :, iw] = y[0]
                recorded_vec[i_traj, :, :, iw] = q
                iw += 1

            subtime = np.concatenate((np.arange(tt, tt + dt, mdt), np.full((1,), tt + dt)))
            y_new, prop = integrate._integrate_runge_kutta_tgls_jit(f, fjac, subtime, y, Id, 1, 0, b, c, a,
                                                                    adjoint, inverse, integrate._zeros_func)
            y[0] = traj[i_traj, :, ti+1]
            q_new = prop[0, :, :, 0] @ q
            qr = np.linalg.qr(q_new)
            q = qr[0]
            r = qr[1]

        recorded_exp[i_traj, :, -1] = m_exp
        recorded_traj[i_traj, :, -1] = y[0]
        recorded_vec[i_traj, :, :, -1] = q

    return recorded_traj, recorded_exp, recorded_vec


class CovariantLyapunovsEstimator(object):
    """Class to compute the Covariant `Lyapunov vectors`_ (CLVs) and `exponents`_ along a trajectory of a dynamical system

    .. math:: \\dot{\\boldsymbol{x}} = \\boldsymbol{f}(t, \\boldsymbol{x})

    with a set of :class:`LyapProcess` and a specified `Runge-Kutta method`_.
    The tangent linear model must also be provided. I.e. one must provide the linearized ODEs

    .. math :: \\dot{\\boldsymbol{\\delta x}} = \\boldsymbol{\\mathrm{J}}(t, \\boldsymbol{x}) \\cdot \\boldsymbol{\\delta x}

    where :math:`\\boldsymbol{\\mathrm{J}} = \\frac{\\partial \\boldsymbol{f}}{\\partial \\boldsymbol{x}}` is the
    Jacobian matrix of :math:`\\boldsymbol{f}`.


    Parameters
    ----------
    num_threads: None or int, optional
        Number of :class:`LyapProcess` workers (threads) to use. If `None`, use the number of machine's
        cores available. Default to `None`.
    b: None or ~numpy.ndarray, optional
        Vector of coefficients :math:`b_i` of the `Runge-Kutta method`_ .
        If `None`, use the classic RK4 method coefficients. Default to `None`.
    c: None or ~numpy.ndarray, optional
        Matrix of coefficients :math:`c_{i,j}` of the `Runge-Kutta method`_ .
        If `None`, use the classic RK4 method coefficients. Default to `None`.
    a: None or ~numpy.ndarray, optional
        Vector of coefficients :math:`a_i` of the `Runge-Kutta method`_ .
        If `None`, use the classic RK4 method coefficients. Default to `None`.
    number_of_dimensions: None or int, optional
        Allow to hardcode the dynamical system dimension. If `None`, evaluate the dimension from the
        callable :attr:`func`. Default to `None`.
    method: int, optional
        Allow to select the method used to compute the CLVs. Presently can be `0` or `1`:

        * `0`: Uses the method of Ginelli et al. :cite:`lyap-GPTCLP2007`. Suitable for a trajectory not too long (depends on the memory available).
        * `1`: Uses the method of the intersection of the subspace spanned by the BLVs and FLVs described in :cite:`lyap-ER1985` and :cite:`lyap-KP2012`
          (see also :cite:`lyap-DPV2021`, Appendix A). Suitable for longer trajectories (uses less memory).

        Default to `0`, i.e. Ginelli et al. algorithm.
    noise_pert: float, optional
        Noise perturbation amplitude parameter of the diagonal of the R matrix in the QR decomposition during the Ginelli step. Mainly done to avoid ill-conditioned matrices
        near tangencies (see :cite:`lyap-KP2012`). Default to 0 (no perturbation).
        Only apply if using the Ginelli et al. algorithm, i.e. if ``method=0``.


    Attributes
    ----------
    num_threads: int
        Number of :class:`LyapProcess` workers (threads) to use.
    b: ~numpy.ndarray
        Vector of coefficients :math:`b_i` of the `Runge-Kutta method`_ .
    c: ~numpy.ndarray
        Matrix of coefficients :math:`c_{i,j}` of the `Runge-Kutta method`_ .
    a: ~numpy.ndarray
        Vector of coefficients :math:`a_i` of the `Runge-Kutta method`_ .
    n_dim: int
        Dynamical system dimension.
    n_vec: int
        The number of Lyapunov vectors to compute.
    n_traj: int
        The number of trajectories (initial conditions) computed at the last estimation
        performed by the estimator.
    n_records: int
        The number of saved states of the last estimation performed by the estimator.
    ic: ~numpy.ndarray
        Store the estimator initial conditions.
    func: callable
        Last function :math:`\\boldsymbol{f}` used by the estimator.
    func_jac: callable
        Last Jacobian matrix function :math:`\\boldsymbol{J}` used by the estimator.
    method: int
        Select the method used to compute the CLVs:

        * `0`: Uses the method of Ginelli et al. :cite:`lyap-GPTCLP2007`. Suitable for a trajectory not too long (depends on the memory available).
        * `1`: Uses the method of the intersection of the subspaces spanned by the BLVs and FLVs described in :cite:`lyap-ER1985` and :cite:`lyap-KP2012`
          (see also :cite:`lyap-DPV2021`, Appendix A). Suitable for longer trajectories (uses less memory).

    noise_pert: float
        Noise perturbation parameter of the diagonal of the matrix resulting from the backpropagation during the Ginelli step.
        Mainly done to avoid ill-conditioned matrices near tangencies (see :cite:`lyap-KP2012`).
        Only apply if using the Ginelli et al. algorithm, i.e. if ``method=0``.
    """

    def __init__(self, num_threads=None, b=None, c=None, a=None, number_of_dimensions=None, noise_pert=0., method=0):

        if num_threads is None:
            self.num_threads = multiprocessing.cpu_count()
        else:
            self.num_threads = num_threads

        # Default is RK4
        if a is None and b is None and c is None:
            self.c = np.array([0., 0.5, 0.5, 1.])
            self.b = np.array([1./6, 1./3, 1./3, 1./6])
            self.a = np.zeros((len(self.c), len(self.b)))
            self.a[1, 0] = 0.5
            self.a[2, 1] = 0.5
            self.a[3, 2] = 1.
        else:
            self.a = a
            self.b = b
            self.c = c

        self.noise_pert = noise_pert

        self.ic = None
        self._time = None
        self._pretime = None
        self._aftertime = None

        self._recorded_traj = None
        self._recorded_exp = None
        self._recorded_vec = None
        self._recorded_bvec = None
        self._recorded_fvec = None
        self.n_traj = 0
        self.n_dim = number_of_dimensions
        self.n_records = 0
        self.n_vec = 0
        self.write_steps = 0
        self.method = method

        self.func = None
        self.func_jac = None

        self._ics_queue = None
        self._clv_queue = None

        self._processes_list = list()

    def terminate(self):
        """Stop the workers (threads) and release the resources of the estimator."""

        for process in self._processes_list:

            process.terminate()
            process.join()

    def set_noise_pert(self, noise_pert):
        """Set the noise perturbation :attr:`noise_pert` parameter.

        Parameters
        ----------
        noise_pert: float, optional
            Noise perturbation amplitude parameter of the diagonal of the R matrix in the QR decomposition during the Ginelli step. Mainly done to avoid ill-conditioned matrices
            near tangencies (see :cite:`lyap-KP2012`).
            Only apply if using the Ginelli et al. algorithm, i.e. if :attr:`method` is 0.
         """
        self.noise_pert = noise_pert
        self.start()

    def set_bca(self, b=None, c=None, a=None, ic_init=True):
        """Set the coefficients of the `Runge-Kutta method`_ and restart the estimator.

        .. _Runge-Kutta method: https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods

        Parameters
        ----------
        b: None or ~numpy.ndarray, optional
            Vector of coefficients :math:`b_i` of the `Runge-Kutta method`_ .
            If `None`, does not reinitialize these coefficients.
        c: None or ~numpy.ndarray, optional
            Matrix of coefficients :math:`c_{i,j}` of the `Runge-Kutta method`_ .
            If `None`, does not reinitialize these coefficients.
        a: None or ~numpy.ndarray, optional
            Vector of coefficients :math:`a_i` of the `Runge-Kutta method`_ .
            If `None`, does not reinitialize these coefficients.
        ic_init: bool, optional
            Re-initialize or not the initial conditions of the estimator. Default to `True`.
        """

        if a is not None:
            self.a = a
        if b is not None:
            self.b = b
        if c is not None:
            self.c = c
        if ic_init:
            self.ic = None
        self.start()

    def start(self):
        """Start or restart the workers (threads) of the estimator.

        Warnings
        --------
        If the estimator was not previously terminated, it will be terminated first in the case
        of a restart.
        """

        self.terminate()

        self._processes_list = list()
        self._ics_queue = multiprocessing.JoinableQueue()
        self._clv_queue = multiprocessing.Queue()

        for i in range(self.num_threads):
            self._processes_list.append(ClvProcess(i, self.func, self.func_jac, self.b, self.c, self.a,
                                                   self._ics_queue, self._clv_queue, self.noise_pert))

        for process in self._processes_list:
            process.daemon = True
            process.start()

    def set_func(self, f, fjac):
        """Set the `Numba`_-jitted function :math:`\\boldsymbol{f}` and Jacobian matrix function
        :math:`\\boldsymbol{\\mathrm{J}}` to integrate.

        .. _Numba: https://numba.pydata.org/

        Parameters
        ----------
        f: callable
            The `Numba`_-jitted function :math:`\\boldsymbol{f}`.
            Should have the signature ``f(t, x)`` where ``x`` is the state value and ``t`` is the time.
        fjac: callable
            The `Numba`_-jitted Jacobian matrix function :math:`\\boldsymbol{J}`.
            Should have the signature ``J(t, x)`` where ``x`` is the state value and ``t`` is the time.

        Warnings
        --------
        This function restarts the estimator!
        """

        self.func = f
        self.func_jac = fjac
        self.start()

    def compute_clvs(self, t0, ta, tb, tc, dt, mdt, ic=None, write_steps=1, n_vec=None, method=None, backward_vectors=False, forward_vectors=False):
        """Estimate the Covariant Lyapunov Vectors (CLVs) along a given trajectory, always integrating the said trajectory
        forward in time from `ic` at `t0` to time `tc`. Return the CLVs between `ta` and `tb`.
        The result of the estimation can be obtained afterward by calling :meth:`get_clvs`.

        Parameters
        ----------
        t0: float
            Initial time of the time integration. Corresponds to the initial condition's `ic` time.
        ta: float
            Define the time span between `t0` and `ta` of the first part of the algorithm, which obtain the convergence to the Backward Lyapunov vectors
            (initialization of the Benettin algorithm).
        tb: float
            Define the time span between `ta` and `tb` where the Covariant Lyapunov Vectors are computed.
        tc: float
            Final time of the time integration algorithm. Define the time span between `tb` and `tc` where, depending on the value of :attr:`method`,
            the convergence to the Forward Lyapunov Vectors or to the Covariant Lyapunov Vectors (thanks to the Ginelli steps) is obtained.
        dt: float
            Timestep of the integration.
        mdt: float
            Micro-timestep to integrate the tangent linear equation between the nonlinear system `dt` timesteps. Should be smaller or equal to `dt`.
        ic: None or ~numpy.ndarray(float), optional
            Initial conditions of the system. Can be a 1D or a 2D array:

            * 1D: Provide a single initial condition.
              Should be of shape (`n_dim`,) where `n_dim` = :math:`\\mathrm{dim}(\\boldsymbol{x})`.
            * 2D: Provide an ensemble of initial condition.
              Should be of shape (`n_traj`, `n_dim`) where `n_dim` = :math:`\\mathrm{dim}(\\boldsymbol{x})`,
              and where `n_traj` is the number of initial conditions.

            If `None`, use the initial conditions stored in :attr:`ic`.
            If then :attr:`ic` is `None`, use a zero initial condition.
            Default to `None`.
        write_steps: int, optional
            Save the state of the integration in memory every `write_steps` steps. The other intermediary
            steps are lost. It determines the size of the returned objects. Default is 1.
            Set to 0 to return only the final state.
        n_vec: int, optional
            The number of Lyapunov vectors to compute. Should be smaller or equal to :attr:`n_dim`.
        method: int, optional
            Allow to select the method used to compute the CLVs. Presently can be `0` or `1`:

            * `0`: Uses the method of Ginelli et al. :cite:`lyap-GPTCLP2007`. Suitable for a trajectory not too long (depends on the memory available).
            * `1`: Uses the method of the intersection of the subspace spanned by the BLVs and FLVs described in :cite:`lyap-ER1985` and :cite:`lyap-KP2012`
              (see also :cite:`lyap-DPV2021`, Appendix A). Suitable for longer trajectories (uses less memory).

            Use the Ginelli et al. algorithm if not provided.
        backward_vectors: bool, optional
            Store also the computed Backward Lyapunov vectors between `ta` and `tb`. Only applies if ``method=1``.
            Does not store the BLVs if not provided.
        forward_vectors: bool, optional
            Store also the computed Forward Lyapunov vectors between `ta` and `tb`. Only applies if ``method=1``.
            Does not store the FLVs if not provided.
        """

        if self.func is None or self.func_jac is None:
            print('No function to integrate defined!')
            return 0

        if ic is None:
            i = 1
            while True:
                self.ic = np.zeros(i)
                try:
                    x = self.func(0., self.ic)
                except:
                    i += 1
                else:
                    break

            i = len(self.func(0., self.ic))
            self.ic = np.zeros(i)
        else:
            self.ic = ic

        if len(self.ic.shape) == 1:
            self.ic = self.ic.reshape((1, -1))

        self.n_traj = self.ic.shape[0]
        self.n_dim = self.ic.shape[1]
        if n_vec is not None:
            self.n_vec = n_vec
        else:
            self.n_vec = self.n_dim
        if method is not None:
            self.method = method

        self._pretime = np.concatenate((np.arange(t0, ta, dt), np.full((1,), ta)))
        self._time = np.concatenate((np.arange(ta, tb, dt), np.full((1,), tb)))
        self._aftertime = np.concatenate((np.arange(tb, tc, dt), np.full((1,), tc)))

        self.write_steps = write_steps

        if write_steps == 0:
            self.n_records = 1
        else:
            tot = self._time[::self.write_steps]
            self.n_records = len(tot)
            if tot[-1] != self._time[-1]:
                self.n_records += 1

        self._recorded_traj = np.zeros((self.n_traj, self.n_dim, self.n_records))
        self._recorded_vec = np.zeros((self.n_traj, self.n_dim, self.n_vec, self.n_records))
        self._recorded_exp = np.zeros((self.n_traj, self.n_vec, self.n_records))
        if self.method == 1:
            if forward_vectors:
                self._recorded_fvec = np.zeros((self.n_traj, self.n_dim, self.n_vec, self.n_records))
            if backward_vectors:
                self._recorded_bvec = np.zeros((self.n_traj, self.n_dim, self.n_vec, self.n_records))

        for i in range(self.n_traj):
            self._ics_queue.put((i, self._pretime, self._time, self._aftertime, mdt, self.ic[i], self.n_vec,
                                 self.write_steps, self.method))

        self._ics_queue.join()

        for i in range(self.n_traj):
            args = self._clv_queue.get()
            self._recorded_traj[args[0]] = args[1]
            self._recorded_exp[args[0]] = args[2]
            self._recorded_vec[args[0]] = args[3]
            if self.method == 1:
                if forward_vectors:
                    self._recorded_fvec[args[0]] = args[5]
                if backward_vectors:
                    self._recorded_bvec[args[0]] = args[4]

    def get_clvs(self):
        """Returns the result of the previous CLVs estimation.

        Returns
        -------
        time, traj, exponents, vectors: ~numpy.ndarray
            The result of the estimation:

            * **time:** Time at which the state of the system was saved. Array of shape (:attr:`n_records`,).
            * **traj:** Saved dynamical system states. 3D array of shape (:attr:`n_traj`, :attr:`n_dim`, :attr:`n_records`).
              If :attr:`n_traj` = 1, a 2D array of shape (:attr:`n_dim`, :attr:`n_records`) is returned instead.
            * **exponents:** Saved estimates of the local Lyapunov exponents along the trajectory. 3D array of shape (:attr:`n_traj`, :attr:`n_vec`, :attr:`n_records`).
              If :attr:`n_traj` = 1, a 2D array of shape (:attr:`n_vec`, :attr:`n_records`) is returned instead.
            * **vectors:** Saved estimates of the local Lyapunov vectors along the trajectory.
              Depending on the input initial conditions, it is maximum a 4D array of shape
              (:attr:`n_traj`, :attr:`n_dim`, :attr:`n_vec`, :attr:`n_records`).
              If one of the dimension is 1, it is squeezed.
        """

        if self.write_steps > 0:
            if self._time[::self.write_steps][-1] == self._time[-1]:
                return self._time[::self.write_steps], np.squeeze(self._recorded_traj), np.squeeze(self._recorded_exp), \
                       np.squeeze(self._recorded_vec)
            else:
                return np.concatenate((self._time[::self.write_steps], np.full((1,), self._time[-1]))), \
                       np.squeeze(self._recorded_traj), np.squeeze(self._recorded_exp), np.squeeze(self._recorded_vec)
        else:
            return self._time[-1], np.squeeze(self._recorded_traj), np.squeeze(self._recorded_exp), \
                   np.squeeze(self._recorded_vec)

    def get_blvs(self):
        """Returns the BLVs obtained during the previous CLVs estimation.

        Returns
        -------
        time, traj, exponents, vectors: ~numpy.ndarray
            The result of the estimation:

            * **time:** Time at which the state of the system was saved. Array of shape (:attr:`n_records`,).
            * **traj:** Saved dynamical system states. 3D array of shape (:attr:`n_traj`, :attr:`n_dim`, :attr:`n_records`).
              If :attr:`n_traj` = 1, a 2D array of shape (:attr:`n_dim`, :attr:`n_records`) is returned instead.
            * **exponents:** Saved estimates of the local Lyapunov exponents along the trajectory. 3D array of shape (:attr:`n_traj`, :attr:`n_vec`, :attr:`n_records`).
              If :attr:`n_traj` = 1, a 2D array of shape (:attr:`n_vec`, :attr:`n_records`) is returned instead.
            * **vectors:** Saved estimates of the local Lyapunov vectors along the trajectory.
              Depending on the input initial conditions, it is maximum a 4D array of shape
              (:attr:`n_traj`, :attr:`n_dim`, :attr:`n_vec`, :attr:`n_records`).
              If one of the dimension is 1, it is squeezed.

        Warnings
        --------
        The BLVs are only available if :attr:`method` is set to 1.
        """

        if self._recorded_bvec is None:
            return None

        if self.write_steps > 0:
            if self._time[::self.write_steps][-1] == self._time[-1]:
                return self._time[::self.write_steps], np.squeeze(self._recorded_traj), np.squeeze(self._recorded_exp), \
                       np.squeeze(self._recorded_bvec)
            else:
                return np.concatenate((self._time[::self.write_steps], np.full((1,), self._time[-1]))), \
                       np.squeeze(self._recorded_traj), np.squeeze(self._recorded_exp), np.squeeze(self._recorded_bvec)
        else:
            return self._time[-1], np.squeeze(self._recorded_traj), np.squeeze(self._recorded_exp), \
                   np.squeeze(self._recorded_bvec)

    def get_flvs(self):
        """Returns the FLVs obtained during the previous CLVs estimation.

        Returns
        -------
        time, traj, exponents, vectors: ~numpy.ndarray
            The result of the estimation:

            * **time:** Time at which the state of the system was saved. Array of shape (:attr:`n_records`,).
            * **traj:** Saved dynamical system states. 3D array of shape (:attr:`n_traj`, :attr:`n_dim`, :attr:`n_records`).
              If :attr:`n_traj` = 1, a 2D array of shape (:attr:`n_dim`, :attr:`n_records`) is returned instead.
            * **exponents:** Saved estimates of the local Lyapunov exponents along the trajectory. 3D array of shape (:attr:`n_traj`, :attr:`n_vec`, :attr:`n_records`).
              If :attr:`n_traj` = 1, a 2D array of shape (:attr:`n_vec`, :attr:`n_records`) is returned instead.
            * **vectors:** Saved estimates of the local Lyapunov vectors along the trajectory.
              Depending on the input initial conditions, it is maximum a 4D array of shape
              (:attr:`n_traj`, :attr:`n_dim`, :attr:`n_vec`, :attr:`n_records`).
              If one of the dimension is 1, it is squeezed.

        Warnings
        --------
        The FLVs are only available if :attr:`method` is set to 1.
        """

        if self._recorded_fvec is None:
            return None

        if self.write_steps > 0:
            if self._time[::self.write_steps][-1] == self._time[-1]:
                return self._time[::self.write_steps], np.squeeze(self._recorded_traj), np.squeeze(self._recorded_exp), \
                       np.squeeze(self._recorded_fvec)
            else:
                return np.concatenate((self._time[::self.write_steps], np.full((1,), self._time[-1]))), \
                       np.squeeze(self._recorded_traj), np.squeeze(self._recorded_exp), np.squeeze(self._recorded_fvec)
        else:
            return self._time[-1], np.squeeze(self._recorded_traj), np.squeeze(self._recorded_exp), \
                   np.squeeze(self._recorded_fvec)


class ClvProcess(multiprocessing.Process):
    """:class:`CovariantLyapunovsEstimator`'s workers class. Allows to multi-thread Lyapunov vectors estimation.

    Parameters
    ----------
    processID: int
        Number identifying the worker.
    func: callable
        `Numba`_-jitted function to integrate assigned to the worker.
    func_jac: callable
        `Numba`_-jitted Jacobian matrix function to integrate assigned to the worker.
    b: ~numpy.ndarray, optional
        Vector of coefficients :math:`b_i` of the `Runge-Kutta method`_ .
    c: ~numpy.ndarray, optional
        Matrix of coefficients :math:`c_{i,j}` of the `Runge-Kutta method`_ .
    a: ~numpy.ndarray, optional
        Vector of coefficients :math:`a_i` of the `Runge-Kutta method`_ .
    ics_queue: multiprocessing.JoinableQueue
        Queue to which the worker ask for initial conditions and parameters input.
    clv_queue: multiprocessing.Queue
        Queue to which the worker returns the estimation results.

    Attributes
    ----------
    processID: int
        Number identifying the worker.
    func: callable
        `Numba`_-jitted function to integrate assigned to the worker.
    func_jac: callable
        `Numba`_-jitted Jacobian matrix function to integrate assigned to the worker.
    b: ~numpy.ndarray
        Vector of coefficients :math:`b_i` of the `Runge-Kutta method`_ .
    c: ~numpy.ndarray
        Matrix of coefficients :math:`c_{i,j}` of the `Runge-Kutta method`_ .
    a: ~numpy.ndarray
        Vector of coefficients :math:`a_i` of the `Runge-Kutta method`_ .
    """

    def __init__(self, processID, func, func_jac, b, c, a, ics_queue, clv_queue, noise_pert):

        super().__init__()
        self.processID = processID
        self._ics_queue = ics_queue
        self._clv_queue = clv_queue
        self.func = func
        self.func_jac = func_jac
        self.a = a
        self.b = b
        self.c = c
        self.noise_pert = noise_pert

    def run(self):
        """Main worker computing routine. Perform the estimation with the fetched initial conditions and parameters."""

        while True:

            args = self._ics_queue.get()

            method = args[8]

            if method == 0:
                recorded_traj, recorded_exp, recorded_vec = _compute_clv_gin_jit(self.func, self.func_jac, args[1], args[2],
                                                                                 args[3], args[4], args[5][np.newaxis, :],
                                                                                 args[6], args[7],
                                                                                 self.b, self.c, self.a, self.noise_pert)
                self._clv_queue.put((args[0], np.squeeze(recorded_traj), np.squeeze(recorded_exp),
                                     np.squeeze(recorded_vec)))
            else:
                recorded_traj, recorded_exp, recorded_vec, backward_vec, forward_vec = _compute_clv_sub_jit(self.func, self.func_jac, args[1], args[2],
                                                                                                            args[3], args[4], args[5][np.newaxis, :],
                                                                                                            args[7], self.b, self.c, self.a)

                self._clv_queue.put((args[0], np.squeeze(recorded_traj), np.squeeze(recorded_exp),
                                     np.squeeze(recorded_vec), np.squeeze(backward_vec), np.squeeze(forward_vec)))

            self._ics_queue.task_done()


# Ginelli et al. method
@njit
def _compute_clv_gin_jit(f, fjac, pretime, time, aftertime, mdt, ic, n_vec, write_steps, b, c, a, noise_pert):

    n_traj = ic.shape[0]
    n_dim = ic.shape[1]

    Id = np.zeros((1, n_dim, n_dim))
    Id[0] = np.eye(n_dim)

    if write_steps == 0:
        n_records = 1
    else:
        tot = time[::write_steps]
        n_records = len(tot)
        if tot[-1] != time[-1]:
            n_records += 1
    recorded_vec = np.zeros((n_traj, n_dim, n_vec, n_records))
    recorded_traj = np.zeros((n_traj, n_dim, n_records))
    recorded_exp = np.zeros((n_traj, n_vec, n_records))

    for i_traj in range(n_traj):

        # first part, making the backward vectors converge (initialization of the Benettin algorithm)

        y = np.zeros((1, n_dim))
        y[0] = ic[i_traj]
        qr = np.linalg.qr(np.random.randn(n_dim, n_vec))
        q = qr[0]

        for tt, dt in zip(pretime[:-1], np.diff(pretime)):

            subtime = np.concatenate((np.arange(tt, tt + dt, mdt), np.full((1,), tt + dt)))
            y_new, prop = integrate._integrate_runge_kutta_tgls_jit(f, fjac, subtime, y, Id, 1, 0, b, c, a,
                                                                    False, 1, integrate._zeros_func)
            y[0] = y_new[0, :, 0]
            q_new = prop[0, :, :, 0] @ q
            qr = np.linalg.qr(q_new)
            q = qr[0]

        # second part, stores the backward vectors and the r matrix (Benettin steps)
        # save the trajectories

        tw = len(time)-1
        tew = len(time)+len(aftertime)-2
        tmp_traj = np.zeros((tw+1, n_dim))
        tmp_vec = np.zeros((tw+1, n_dim, n_vec))
        tmp_R = np.zeros((tew, n_vec, n_vec))

        for ti, (tt, dt) in enumerate(zip(time[:-1], np.diff(time))):

            tmp_vec[ti] = q.copy()
            tmp_traj[ti] = y[0].copy()

            subtime = np.concatenate((np.arange(tt, tt + dt, mdt), np.full((1,), tt + dt)))
            y_new, prop = integrate._integrate_runge_kutta_tgls_jit(f, fjac, subtime, y, Id, 1, 0, b, c, a,
                                                                    False, 1, integrate._zeros_func)
            y[0] = y_new[0, :, 0]
            q_new = prop[0, :, :, 0] @ q
            qr = np.linalg.qr(q_new)
            q = qr[0]
            tmp_R[ti] = qr[1].copy()

        tmp_vec[-1] = q.copy()
        tmp_traj[-1] = y[0].copy()

        # third part, stores the r matrix (Benettin steps)

        for ti, (tt, dt) in enumerate(zip(aftertime[:-1], np.diff(aftertime))):

            subtime = np.concatenate((np.arange(tt, tt + dt, mdt), np.full((1,), tt + dt)))
            y_new, prop = integrate._integrate_runge_kutta_tgls_jit(f, fjac, subtime, y, Id, 1, 0, b, c, a,
                                                                    False, 1, integrate._zeros_func)
            y[0] = y_new[0, :, 0]
            q_new = prop[0, :, :, 0] @ q
            qr = np.linalg.qr(q_new)
            q = qr[0]
            tmp_R[ti+tw] = qr[1].copy()

        # fourth part, going backward until tb (Ginelli steps)

        qr = np.linalg.qr(np.random.randn(n_dim, n_vec))
        am, norm = normalize_matrix_columns(qr[1])

        for ti in range(tew-1, tw, -1):

            am_new = solve_triangular_matrix(tmp_R[ti], am)
            noise = np.random.randn(n_dim)
            for i in range(n_vec):
                am_new[i, i] += noise[i] * noise_pert
            am, norm = normalize_matrix_columns(am_new)

        # fifth and last part, going backward from tb to ta (Ginelli steps)
        # save the data

        dte = np.concatenate((np.diff(time), np.full((1,), aftertime[1] - aftertime[0])))
        iw = 1
        for ti in range(tw, -1, -1):

            am_new = solve_triangular_matrix(tmp_R[ti], am)
            noise = np.random.randn(n_vec)
            for i in range(n_dim):
                am_new[i, i] += noise[i] * noise_pert
            am, mloc_exp = normalize_matrix_columns(am_new)

            if write_steps > 0 and np.mod(tw-ti, write_steps) == 0:
                recorded_traj[i_traj, :, -iw] = tmp_traj[ti]
                recorded_exp[i_traj, :, -iw] = -np.log(np.abs(mloc_exp))/dte[ti]
                recorded_vec[i_traj, :, :, -iw] = tmp_vec[ti] @ am
                iw += 1

        recorded_traj[i_traj, :, 0] = tmp_traj[0]
        recorded_exp[i_traj, :, 0] = -np.log(np.abs(mloc_exp))/dte[0]
        recorded_vec[i_traj, :, :, 0] = tmp_vec[0] @ am

    return recorded_traj, recorded_exp, recorded_vec


# Subspace intersection method
@njit
def _compute_clv_sub_jit(f, fjac, pretime, time, aftertime, mdt, ic, write_steps, b, c, a):

    n_traj = ic.shape[0]
    n_dim = ic.shape[1]

    lp = len(pretime)
    la = len(aftertime)

    ttraj = integrate._integrate_runge_kutta_jit(f, np.concatenate((pretime[:-1], time[:-1], aftertime)), ic, 1, 1, b, c, a)
    traj, exp, fvec = _compute_forward_lyap_traj_jit(f, fjac, time, aftertime, ttraj[:, :, lp-1:], mdt,
                                                     n_dim, write_steps, False, 1, b, c, a)
    traj, exp, bvec = _compute_backward_lyap_traj_jit(f, fjac, pretime, time, ttraj[:, :, :-la+1], mdt,
                                                      n_dim, write_steps, False, 1, b, c, a)

    recorded_traj = traj
    recorded_exp = np.zeros_like(traj)
    n_records = traj.shape[-1]
    recorded_vec = np.zeros((n_traj, n_dim, n_dim, n_records))
    subtime = np.array([0., mdt])
    y = np.zeros((1, n_dim))
    vec = np.zeros((1, n_dim, n_dim))

    for i_traj in range(n_traj):
        for ti in range(n_records):
            for j in range(n_dim):
                u, z, w = np.linalg.svd(bvec[i_traj, :, :j+1, ti].T @ fvec[i_traj, :, :n_dim-j, ti])
                basis = bvec[i_traj, :, :j+1, ti] @ u
                recorded_vec[i_traj, :, j, ti] = basis[:, 0]

            y[0] = recorded_traj[i_traj, :, ti]
            vec[0] = recorded_vec[i_traj, :, :, ti]
            y_new, sol = integrate._integrate_runge_kutta_tgls_jit(f, fjac, subtime, y, vec, 1, 0, b, c, a,
                                                                   False, 1, integrate._zeros_func)
            soln, mloc_exp = normalize_matrix_columns(sol[0, :, :, 0])
            recorded_exp[i_traj, :, ti] = np.log(np.abs(mloc_exp))/mdt

    return recorded_traj, recorded_exp, recorded_vec, bvec, fvec


if __name__ == "__main__":

    a = 0.25
    F = 8.
    G = 1.
    b = 4.

    @njit
    def fL84(t, x):
        xx = -x[1] ** 2 - x[2] ** 2 - a * x[0] + a * F
        yy = x[0] * x[1] - b * x[0] * x[2] - x[1] + G
        zz = b * x[0] * x[1] + x[0] * x[2] - x[2]
        return np.array([xx, yy, zz])

    @njit
    def DfL84(t, x):
        return np.array([[     -a        , -2. * x[1], -2. * x[2]],
                         [x[1] - b * x[2], -1. + x[0], -b * x[0]],
                         [b * x[1] + x[2],  b * x[0], -1. + x[0]]])

    sigma = 10.
    r = 28.
    bb = 8. / 3.

    @njit
    def fL63(t, x):
        xx = sigma * (x[1] - x[0])
        yy = r * x[0] - x[1] - x[0] * x[2]
        zz = x[0] * x[1] - bb * x[2]
        return np.array([xx, yy, zz])

    @njit
    def DfL63(t, x):
        return np.array([[-sigma, sigma, 0.],
                         [r - x[2], -1., - x[0]],
                         [x[1],  x[0], -bb]])

    ic = np.random.random(3)

    # tt, ic_L84 = integrate.integrate_runge_kutta(fL84, 0., 10000., 0.01, ic=ic, write_steps=0)
    tt, ic = integrate.integrate_runge_kutta(fL63, 0., 10000., 0.01, ic=ic, write_steps=0)

    print('Computing Backward Lyapunovs')

    lyapint = LyapunovsEstimator()
    # lyapint.set_func(fL84, DfL84)
    lyapint.set_func(fL63, DfL63)
    lyapint.compute_lyapunovs(0., 10000., 30000., 0.01, 0.01, ic, write_steps=1) #, n_vec=2)
    btl, btraj, bexp, bvec = lyapint.get_lyapunovs()

    print('Computing Forward Lyapunovs')

    # lyapint.set_func(fL84, DfL84)
    lyapint.set_func(fL63, DfL63)
    lyapint.compute_lyapunovs(0., 20000., 30000., 0.01, 0.01, ic, write_steps=1, forward=True, adjoint=False, inverse=False) #, n_vec=2)
    ftl, ftraj, fexp, fvec = lyapint.get_lyapunovs()

    print('Computing Covariant Lyapunovs')

    clvint = CovariantLyapunovsEstimator()
    # clvint.set_func(fL84, DfL84)
    clvint.set_func(fL63, DfL63)
    clvint.compute_clvs(0., 10000., 20000., 30000., 0.01, 0.01, ic, write_steps=1) #, n_vec=2)
    ctl, ctraj, cexp, cvec = clvint.get_clvs()

    clvint.compute_clvs(0., 10000., 20000., 30000., 0.01, 0.01, ic, write_steps=10, method=1, backward_vectors=True) #, n_vec=2)
    ctl2, ctraj2, cexp2, cvec2 = clvint.get_clvs()

    lyapint.terminate()
    clvint.terminate()