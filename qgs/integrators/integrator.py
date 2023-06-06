
"""
    Integrator module
    =================

    Module with the classes of integrators to multi-thread the integration of
    an ordinary differential equations

    .. math:: \dot{\\boldsymbol{x}} = \\boldsymbol{f}(t, \\boldsymbol{x})

    of the model and its linearized version.

    Module classes
    --------------

    * :class:`RungeKuttaIntegrator`
    * :class:`RungeKuttaTglsIntegrator`

"""
import multiprocessing
import numpy as np
from numba import njit
from qgs.integrators.integrate import _integrate_runge_kutta_jit, _integrate_runge_kutta_tgls_jit, _zeros_func
from qgs.functions.util import reverse


class RungeKuttaIntegrator(object):
    """Class to integrate the ordinary differential equations (ODEs)

    .. math:: \dot{\\boldsymbol{x}} = \\boldsymbol{f}(t, \\boldsymbol{x})

    with a set of :class:`TrajectoryProcess` and a specified `Runge-Kutta method`_.

    .. _Runge-Kutta method: https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods

    Parameters
    ----------
    num_threads: None or int, optional
        Number of :class:`TrajectoryProcess` workers (threads) to use. If `None`, use the number of machine's
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
        Number of :class:`TrajectoryProcess` workers (threads) to use.
    b: ~numpy.ndarray
        Vector of coefficients :math:`b_i` of the `Runge-Kutta method`_ .
    c: ~numpy.ndarray
        Matrix of coefficients :math:`c_{i,j}` of the `Runge-Kutta method`_ .
    a: ~numpy.ndarray
        Vector of coefficients :math:`a_i` of the `Runge-Kutta method`_ .
    n_dim: int
        Dynamical system dimension.
    n_traj: int
        The number of trajectories (initial conditions) computed at the last integration
        performed by the integrator.
    n_records: int
        The number of saved states of the last integration performed by the integrator.
    ic: ~numpy.ndarray
        Store the integrator initial conditions.
    func: callable
        Last function :math:`\\boldsymbol{f}` used by the integrator to integrate.
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
        self._recorded_traj = None
        self.n_traj = 0
        self.n_dim = number_of_dimensions
        self.n_records = 0
        self._write_steps = 0
        self._time_direction = 1

        self.func = None

        self._ics_queue = None
        self._traj_queue = None

        self._processes_list = list()

    def terminate(self):
        """Stop the workers (threads) and release the resources of the integrator."""

        for process in self._processes_list:

            process.terminate()
            process.join()

    def start(self):
        """Start or restart the workers (threads) of the integrator.

        Warnings
        --------
        If the integrator was not previously terminated, it will be terminated first in the case
        of a restart.
        """

        self.terminate()

        self._processes_list = list()
        self._ics_queue = multiprocessing.JoinableQueue()
        self._traj_queue = multiprocessing.Queue()

        for i in range(self.num_threads):
            self._processes_list.append(TrajectoryProcess(i, self.func, self.b, self.c, self.a,
                                                          self._ics_queue, self._traj_queue))

        for process in self._processes_list:
            process.daemon = True
            process.start()

    def set_func(self, f, ic_init=True):
        """Set the `Numba`_-jitted function :math:`\\boldsymbol{f}` to integrate.

        .. _Numba: https://numba.pydata.org/

        Parameters
        ----------
        f: callable
            The `Numba`_-jitted function :math:`\\boldsymbol{f}`.
            Should have the signature ``f(t, x)`` where ``x`` is the state value and ``t`` is the time.
        ic_init: bool, optional
            Re-initialize or not the initial conditions of the integrator. Default to `True`.

        Warnings
        --------
        This function restarts the integrator!
        """

        self.func = f
        if ic_init:
            self.ic = None
        self.start()

    def set_bca(self, b=None, c=None, a=None, ic_init=True):
        """Set the coefficients of the `Runge-Kutta method`_ and restart the integrator. s

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
            Re-initialize or not the initial conditions of the integrator. Default to `True`.

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

    def initialize(self, convergence_time, dt, pert_size=0.01, reconvergence_time=None, forward=True,
                   number_of_trajectories=1, ic=None, reconverge=False):
        """Initialize the integration on an attractor by running it for a transient time,
        For an ensemble of initial conditions, can do the same transient time for each, or the
        `convergence_time` for the first one, and a smaller `reconvergence_time` for the subsequent ones.
        This results into initial conditions on the attractor, stored in :attr:`ic`.

        Parameters
        ----------
        convergence_time: float
            Transient time needed to converge to the attractor.
        dt: float
            Timestep of the transient integration.
        pert_size:float, optional
            If the reconvergence is activated, size of the perturbation to add to the previous ic to find
            the next one. Default to 0.01 .
        reconvergence_time: None or float, optional
            Transient time for the subsequent trajectories after the first long `transient_time`.
        forward: bool, optional
            Whether to integrate the ODEs forward or backward in time. In case of backward integration, the
            initial condition `ic` becomes a final condition. Default to forward integration.
        number_of_trajectories: int
            Number of initial conditions to find. Default to 1.  Inactive if `ic` is provided.
        ic: None or ~numpy.ndarray(float), optional
            Initial (or final) conditions of the system. Can be a 1D or a 2D array:

            * 1D: Provide a single initial condition.
              Should be of shape (`n_dim`,) where `n_dim` = :math:`\mathrm{dim}(\\boldsymbol{x})`.
            * 2D: Provide an ensemble of initial condition.
              Should be of shape (`n_traj`, `n_dim`) where `n_dim` = :math:`\mathrm{dim}(\\boldsymbol{x})`,
              and where `n_traj` is the number of initial conditions.

            If `None`, use `number_trajectories` random initial conditions. Default to `None`.
            If the `forward` argument is `False`, it specifies final conditions.
        reconverge: bool
            Use or not the smaller transient time reconvergence with a perturbation
            after the first initial conditions have been computed. If activated, only use the :attr:`num_threads`
            first initial conditions of the `ic` arguments. Default to `False`.
        """

        if reconverge is None:
            reconverge = False

        if ic is None:
            if self.n_dim is not None:
                i = self.n_dim
            else:
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

            if number_of_trajectories > self.num_threads:
                reconverge = True
                tmp_ic = np.zeros((number_of_trajectories, i))
                tmp_ic[:self.num_threads] = np.random.randn(self.num_threads, i)
            else:
                tmp_ic = np.random.randn(number_of_trajectories, i)
        else:
            tmp_ic = ic.copy()
            if len(tmp_ic.shape) > 1:
                number_of_trajectories = tmp_ic.shape[0]

        if reconverge and reconvergence_time is not None:

            self.integrate(0., convergence_time, dt, ic=tmp_ic[:self.num_threads], write_steps=0, forward=forward)
            t, x = self.get_trajectories()
            tmp_ic[:self.num_threads] = x
            if number_of_trajectories - self.num_threads > self.num_threads:
                next_len = self.num_threads
            else:
                next_len = number_of_trajectories - self.num_threads

            index = self.num_threads
            while True:
                perturbation = pert_size * np.random.randn(next_len, x.shape[1])
                self.integrate(0., reconvergence_time, dt, ic=x[:next_len]+perturbation, write_steps=0, forward=forward)
                t, x = self.get_trajectories()
                tmp_ic[index:index+next_len] = x
                index += next_len
                if number_of_trajectories - index > self.num_threads:
                    next_len = self.num_threads
                else:
                    next_len = number_of_trajectories - index
                if next_len <= 0:
                    break
            self.ic = tmp_ic
        else:
            self.integrate(0., convergence_time, dt, ic=tmp_ic, write_steps=0, forward=forward)
            t, x = self.get_trajectories()
            self.ic = x

    def integrate(self, t0, t, dt, ic=None, forward=True, write_steps=1):
        """Integrate the ordinary differential equations (ODEs)

        .. math:: \\dot{\\boldsymbol{x}} = \\boldsymbol{f}(t, \\boldsymbol{x})

        with a specified `Runge-Kutta method`_ and workers. The function :math:`\\boldsymbol{f}` is the `Numba`_ jitted
        function stored in :attr:`func`. The result of the integration can be obtained afterward by calling
        :meth:`get_trajectories`.

        .. _Runge-Kutta method: https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
        .. _Numba: https://numba.pydata.org/

        Parameters
        ----------
        t0: float
            Initial time of the time integration. Corresponds to the initial condition's `ic` time.
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

            If `None`, use the initial conditions stored in :attr:`ic`.
            If then :attr:`ic` is `None`, use a zero initial condition.
            Default to `None`.
            If the `forward` argument is `False`, it specifies final conditions.
        forward: bool, optional
            Whether to integrate the ODEs forward or backward in time. In case of backward integration, the
            initial condition `ic` becomes a final condition. Default to forward integration.
        write_steps: int, optional
            Save the state of the integration in memory every `write_steps` steps. The other intermediary
            steps are lost. It determines the size of the returned objects. Default is 1.
            Set to 0 to return only the final state.
        """
        if self.func is None:
            print('No function to integrate defined!')
            return 0

        if ic is None:
            if self.ic is None:
                if self.n_dim is not None:
                    i = self.n_dim
                else:
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
        self._time = np.concatenate((np.arange(t0, t, dt), np.full((1,), t)))
        self._write_steps = write_steps

        if forward:
            self._time_direction = 1
        else:
            self._time_direction = -1

        if write_steps == 0:
            self.n_records = 1
        else:
            tot = self._time[::self._write_steps]
            self.n_records = len(tot)
            if tot[-1] != self._time[-1]:
                self.n_records += 1

        self._recorded_traj = np.zeros((self.n_traj, self.n_dim, self.n_records))

        for i in range(self.n_traj):
            self._ics_queue.put((i, self._time, self.ic[i], self._time_direction, self._write_steps))

        self._ics_queue.join()

        for i in range(self.n_traj):
            args = self._traj_queue.get()
            self._recorded_traj[args[0]] = args[1]

    def get_trajectories(self):
        """Returns the result of the previous integrator integration.

        Returns
        -------
        time, traj: ~numpy.ndarray
            The result of the integration:

            * **time:** Time at which the state of the system was saved. Array of shape (:attr:`n_records`,).
            * **traj:** Saved dynamical system states. 3D array of shape (:attr:`n_traj`, :attr:`n_dim`, :attr:`n_records`).
              If :attr:`n_traj` = 1, a 2D array of shape (:attr:`n_dim`, :attr:`n_records`) is returned instead.
        """
        if self._write_steps > 0:
            if self._time_direction == 1:
                if self._time[::self._write_steps][-1] == self._time[-1]:
                    return self._time[::self._write_steps], np.squeeze(self._recorded_traj)
                else:
                    return np.concatenate((self._time[::self._write_steps], np.full((1,), self._time[-1]))), \
                           np.squeeze(self._recorded_traj)
            else:
                rtime = reverse(self._time[::-self._write_steps])
                if rtime[0] == self._time[0]:
                    return rtime, np.squeeze(self._recorded_traj)
                else:
                    return np.concatenate((np.full((1,), self._time[0]), rtime)), np.squeeze(self._recorded_traj)

        else:
            return self._time[-1], np.squeeze(self._recorded_traj)

    def get_ic(self):
        """Returns the initial conditions stored in the integrator.

        Returns
        -------
        ~numpy.ndarray
            The initial conditions.
        """
        return self.ic

    def set_ic(self, ic):
        """Direct setter for the integrator's initial conditions

        Parameters
        ----------
        ic: ~numpy.ndarray(float)
            Initial condition of the system. Can be a 1D or a 2D array:

            * 1D: Provide a single initial condition.
              Should be of shape (`n_dim`,) where `n_dim` = :math:`\\mathrm{dim}(\\boldsymbol{x})`.
            * 2D: Provide an ensemble of initial condition.
              Should be of shape (`n_traj`, `n_dim`) where `n_dim` = :math:`\\mathrm{dim}(\\boldsymbol{x})`,
              and where `n_traj` is the number of initial conditions.
        """
        self.ic = ic


class TrajectoryProcess(multiprocessing.Process):
    """:class:`RungeKuttaIntegrator`'s workers class. Allows to multi-thread time integration.

    .. _Runge-Kutta method: https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
    .. _Numba: https://numba.pydata.org/

    Parameters
    ----------
    processID: int
        Number identifying the worker.
    func: callable
        `Numba`_-jitted function to integrate assigned to the worker.
    b: ~numpy.ndarray, optional
        Vector of coefficients :math:`b_i` of the `Runge-Kutta method`_ .
    c: ~numpy.ndarray, optional
        Matrix of coefficients :math:`c_{i,j}` of the `Runge-Kutta method`_ .
    a: ~numpy.ndarray, optional
        Vector of coefficients :math:`a_i` of the `Runge-Kutta method`_ .
    ics_queue: multiprocessing.JoinableQueue
        Queue to which the worker ask for initial conditions and parameters input.
    traj_queue: multiprocessing.Queue
        Queue to which the worker returns the integration results.

    Attributes
    ----------
    processID: int
        Number identifying the worker.
    func: callable
        `Numba`_-jitted function to integrate assigned to the worker.
    b: ~numpy.ndarray
        Vector of coefficients :math:`b_i` of the `Runge-Kutta method`_ .
    c: ~numpy.ndarray
        Matrix of coefficients :math:`c_{i,j}` of the `Runge-Kutta method`_ .
    a: ~numpy.ndarray
        Vector of coefficients :math:`a_i` of the `Runge-Kutta method`_ .
    """
    def __init__(self, processID, func, b, c, a, ics_queue, traj_queue):

        super().__init__()
        self.processID = processID
        self._ics_queue = ics_queue
        self._traj_queue = traj_queue
        self.func = func
        self.a = a
        self.b = b
        self.c = c

    def run(self):
        """Main worker computing routine. Perform the time integration with the fetched initial conditions and parameters."""

        while True:

            args = self._ics_queue.get()

            recorded_traj = _integrate_runge_kutta_jit(self.func, args[1], args[2][np.newaxis, :], args[3], args[4],
                                                       self.b, self.c, self.a)

            self._traj_queue.put((args[0], recorded_traj))

            self._ics_queue.task_done()


class RungeKuttaTglsIntegrator(object):
    """Class to integrate simultaneously the ordinary differential equations (ODEs)

    .. math:: \dot{\\boldsymbol{x}} = \\boldsymbol{f}(t, \\boldsymbol{x})

    and its tangent linear model, i.e. the linearized ODEs

    .. math :: \dot{\\boldsymbol{\delta x}} = \\boldsymbol{\mathrm{J}}(t, \\boldsymbol{x}) \cdot \\boldsymbol{\delta x}

    where :math:`\\boldsymbol{\mathrm{J}} = \\frac{\partial \\boldsymbol{f}}{\partial \\boldsymbol{x}}` is the
    Jacobian matrix of :math:`\\boldsymbol{f}`, with a specified `Runge-Kutta method`_.
    To solve this equation, one has to integrate simultaneously both ODEs. This class does so with a set
    of :class:`TglsTrajectoryProcess` workers.

    The function :math:`\\boldsymbol{f}` and :math:`\\boldsymbol{J}` should
    be `Numba`_ jitted functions. These functions must have a signature ``f(t, x)`` and ``J(t, x)`` where ``x`` is
    the state value and ``t`` is the time.

    .. _Runge-Kutta method: https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
    .. _Numba: https://numba.pydata.org/
    .. _fundamental matrix of solutions: https://en.wikipedia.org/wiki/Fundamental_matrix_(linear_differential_equation)

    Parameters
    ----------
    num_threads: None or int, optional
        Number of :class:`TrajectoryProcess` workers (threads) to use. If `None`, use the number of machine's
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
        Number of :class:`TrajectoryProcess` workers (threads) to use.
    b: ~numpy.ndarray
        Vector of coefficients :math:`b_i` of the `Runge-Kutta method`_ .
    c: ~numpy.ndarray
        Matrix of coefficients :math:`c_{i,j}` of the `Runge-Kutta method`_ .
    a: ~numpy.ndarray
        Vector of coefficients :math:`a_i` of the `Runge-Kutta method`_ .
    n_dim: int
        Dynamical system dimension.
    n_traj: int
        The number of trajectories (initial conditions) of the non-linear ODEs computed at the last integration
        performed by the integrator.
    n_tg_traj: int
        The number of trajectories (initial conditions) the linear ODEs computed at the last integration
        performed by the integrator.
    n_records: int
        The number of saved states of the last integration performed by the integrator.
    ic: ~numpy.ndarray
        Store the integrator non-linear ODEs initial conditions.
    tg_ic: ~numpy.ndarray
        Store the integrator linear ODEs initial conditions.
    func: callable
        Last function :math:`\\boldsymbol{f}` used by the integrator to integrate.
    func_jac: callable
        Last Jacobian matrix function :math:`\\boldsymbol{J}` used by the integrator to integrate.
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
        self.tg_ic = None
        self._time = None
        self._recorded_traj = None
        self._recorded_fmatrix = None
        self.n_traj = 0
        self.n_tg_traj = 0
        self.n_dim = number_of_dimensions
        self.n_records = 0
        self._write_steps = 0

        self._time_direction = 1
        self._adjoint = False
        self._boundary = None
        self._inverse = 1.

        self.func = None
        self.func_jac = None

        self._ics_queue = None
        self._traj_queue = None

        self._processes_list = list()

    def terminate(self):
        """Stop the workers (threads) and release the resources of the integrator."""

        for process in self._processes_list:

            process.terminate()
            process.join()

    def start(self):
        """Start or restart the workers (threads) of the integrator.

        Warnings
        --------
        If the integrator was not previously terminated, it will be terminated first in the case
        of a restart.
        """

        self.terminate()

        self._processes_list = list()
        self._ics_queue = multiprocessing.JoinableQueue()
        self._traj_queue = multiprocessing.Queue()

        for i in range(self.num_threads):
            self._processes_list.append(TglsTrajectoryProcess(i, self.func, self.func_jac, self.b, self.c, self.a,
                                                              self._ics_queue, self._traj_queue))

        for process in self._processes_list:
            process.daemon = True
            process.start()

    def set_func(self, f, fjac, ic_init=True):
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
        ic_init: bool, optional
            Re-initialize or not the initial conditions of the integrator. Default to `True`.

        Warnings
        --------
        This function restarts the integrator!
        """

        self.func = f
        self.func_jac = fjac
        if ic_init:
            self.ic = None
        self.start()

    def set_bca(self, b=None, c=None, a=None, ic_init=True):
        """Set the coefficients of the `Runge-Kutta method`_ and restart the integrator. s

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
            Re-initialize or not the initial conditions of the integrator. Default to `True`.

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

    def initialize(self, convergence_time, dt, pert_size=0.01, reconvergence_time=None, forward=True,
                   number_of_trajectories=1, ic=None, reconverge=None):
        """Initialize the integration on an attractor by running it for a transient time,
        For an ensemble of initial conditions, can do the same transient time for each, or the
        `convergence_time` for the first one, and a smaller `reconvergence_time` for the subsequent ones.
        This results into initial conditions on the attractor, stored in :attr:`ic`.

        Parameters
        ----------
        convergence_time: float
            Transient time needed to converge to the attractor.
        dt: float
            Timestep of the transient integration.
        pert_size:float, optional
            If the reconvergence is activated, size of the perturbation to add to the previous ic to find
            the next one. Default to 0.01 .
        reconvergence_time: None or float, optional
            Transient time for the subsequent trajectories after the first long `transient_time`.
        forward: bool, optional
            If true, integrate the ODEs forward in time, else, integrate backward in time. In case of backward integration, the
            initial condition `ic` becomes a final condition. Default to forward integration.
        number_of_trajectories: int
            Number of initial conditions to find. Default to 1.  Inactive if `ic` is provided.
        ic: None or ~numpy.ndarray(float), optional
            Initial condition of the system. Can be a 1D or a 2D array:

            * 1D: Provide a single initial condition.
              Should be of shape (`n_dim`,) where `n_dim` = :math:`\mathrm{dim}(\\boldsymbol{x})`.
            * 2D: Provide an ensemble of initial condition.
              Should be of shape (`n_traj`, `n_dim`) where `n_dim` = :math:`\mathrm{dim}(\\boldsymbol{x})`,
              and where `n_traj` is the number of initial conditions.

            If `None`, use `number_trajectories` random initial conditions. Default to `None`.
        reconverge: bool
            Use or not the smaller transient time reconvergence with a perturbation
            after the first initial conditions have been computed. If activated, only use the :attr:`num_threads`
            first initial conditions of the `ic` arguments. Default to `False`.
        """

        if reconverge is None:
            reconverge = False

        if ic is None:
            if self.n_dim is not None:
                i = self.n_dim
            else:
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

            if number_of_trajectories > self.num_threads:
                reconverge = True
                tmp_ic = np.zeros((number_of_trajectories, i))
                tmp_ic[:self.num_threads] = np.random.randn(self.num_threads, i)
            else:
                tmp_ic = np.random.randn(number_of_trajectories, i)
        else:
            tmp_ic = ic.copy()
            if len(tmp_ic.shape) > 1:
                number_of_trajectories = tmp_ic.shape[0]

        if reconverge and reconvergence_time is not None:

            self.integrate(0., convergence_time, dt, ic=tmp_ic[:self.num_threads], write_steps=0, forward=forward)
            t, x, fm = self.get_trajectories()
            tmp_ic[:self.num_threads] = x
            if number_of_trajectories - self.num_threads > self.num_threads:
                next_len = self.num_threads
            else:
                next_len = number_of_trajectories - self.num_threads

            index = self.num_threads
            while True:
                perturbation = pert_size * np.random.randn(next_len, x.shape[1])
                self.integrate(0., reconvergence_time, dt, ic=x[:next_len]+perturbation, write_steps=0, forward=forward)
                t, x, fm = self.get_trajectories()
                tmp_ic[index:index+next_len] = x
                index += next_len
                if number_of_trajectories - index > self.num_threads:
                    next_len = self.num_threads
                else:
                    next_len = number_of_trajectories - index
                if next_len <= 0:
                    break
            self.ic = tmp_ic
        else:
            self.integrate(0., convergence_time, dt, ic=tmp_ic, write_steps=0, forward=forward)
            t, x, fm = self.get_trajectories()
            self.ic = x

    def integrate(self, t0, t, dt, ic=None, tg_ic=None, forward=True, adjoint=False, inverse=False, boundary=None,
                  write_steps=1):
        """Integrate simultaneously the non-linear and linearized ordinary differential equations (ODEs)

        .. math:: \\dot{\\boldsymbol{x}} = \\boldsymbol{f}(t, \\boldsymbol{x})

        and

        .. math :: \\dot{\\boldsymbol{\\delta x}} = \\boldsymbol{\\mathrm{J}}(t, \\boldsymbol{x}) \cdot \\boldsymbol{\\delta x}

        with a specified `Runge-Kutta method`_ and workers.
        The function :math:`\\boldsymbol{f}` is the `Numba`_ jitted function stored in :attr:`func`.
        The function :math:`\\boldsymbol{J}` is the `Numba`_ jitted function stored in :attr:`func_jac`.
        The result of the integration can be obtained afterward by calling :meth:`get_trajectories`.

        .. _Runge-Kutta method: https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
        .. _Numba: https://numba.pydata.org/
        .. _fundamental matrix of solutions: https://en.wikipedia.org/wiki/Fundamental_matrix_(linear_differential_equation)

        Parameters
        ----------
        t0: float
            Initial time of the time integration. Corresponds to the initial condition.
        t: float
            Final time of the time integration. Corresponds to the final condition.
        dt: float
            Timestep of the integration.
        ic: None or ~numpy.ndarray(float), optional
            Initial (or final) conditions of the system. Can be a 1D or a 2D array:

            * 1D: Provide a single initial condition.
              Should be of shape (`n_dim`,) where `n_dim` = :math:`\\mathrm{dim}(\\boldsymbol{x})`.
            * 2D: Provide an ensemble of initial condition.
              Should be of shape (`n_traj`, `n_dim`) where `n_dim` = :math:`\\mathrm{dim}(\\boldsymbol{x})`,
              and where `n_traj` is the number of initial conditions.

            If `None`, use the initial conditions stored in :attr:`ic`.
            If then :attr:`ic` is `None`, use a zero initial condition.
            Default to `None`.
            If the `forward` argument is `False`, it specifies final conditions.
        tg_ic: None or ~numpy.ndarray(float), optional
            Initial (or final) conditions of the linear ODEs
            :math:`\\dot{\\boldsymbol{\\delta x}} = \\boldsymbol{\\mathrm{J}}(t, \\boldsymbol{x}) \\cdot \\boldsymbol{\\delta x}`. \n
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
            If true, integrate the ODEs forward in time, else, integrate backward in time. In case of backward integration, the
            initial condition `ic` becomes a final condition. Default to forward integration.
        adjoint: bool, optional
            If true, integrate the tangent :math:`\dot{\\boldsymbol{\delta x}} = \\boldsymbol{\mathrm{J}}(t, \\boldsymbol{x}) \cdot \\boldsymbol{\delta x}` ,
            else, integrate the adjoint linear model :math:`\dot{\\boldsymbol{\delta x}} = \\boldsymbol{\mathrm{J}}^T(t, \\boldsymbol{x}) \cdot \\boldsymbol{\delta x}`.
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
        """

        if self.func is None or self.func_jac is None:
            print('No function to integrate defined!')
            return 0

        if ic is None:
            if self.ic is None:
                if self.n_dim is not None:
                    i = self.n_dim
                else:
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
        self._time = np.concatenate((np.arange(t0, t, dt), np.full((1,), t)))
        self._write_steps = write_steps

        if tg_ic is None:
            tg_ic = np.eye(self.ic.shape[1])

        tg_ic_sav = tg_ic.copy()

        if len(tg_ic.shape) == 1:
            tg_ic = tg_ic.reshape((1, -1, 1))
            ict = tg_ic.copy()
            for i in range(self.n_traj-1):
                ict = np.concatenate((ict, tg_ic))
            self.tg_ic = ict
        elif len(tg_ic.shape) == 2:
            if tg_ic.shape[0] == self.n_traj:
                self.tg_ic = tg_ic[..., np.newaxis]
            else:
                tg_ic = tg_ic[np.newaxis, ...]
                tg_ic = np.swapaxes(tg_ic, 1, 2)
                ict = tg_ic.copy()
                for i in range(self.n_traj-1):
                    ict = np.concatenate((ict, tg_ic))
                self.tg_ic = ict
        elif len(tg_ic.shape) == 3:
            if tg_ic.shape[1] != self.n_dim:
                self.tg_ic = np.swapaxes(tg_ic, 1, 2)

        self.n_tg_traj = self.tg_ic.shape[1]

        if forward:
            self._time_direction = 1
        else:
            self._time_direction = -1

        self._adjoint = adjoint

        if boundary is None:
            self._boundary = _zeros_func
        else:
            self._boundary = boundary

        self._inverse = 1.
        if inverse:
            self._inverse *= -1.

        if write_steps == 0:
            self.n_records = 1
        else:
            tot = self._time[::self._write_steps]
            self.n_records = len(tot)
            if tot[-1] != self._time[-1]:
                self.n_records += 1

        self._recorded_traj = np.zeros((self.n_traj, self.n_dim, self.n_records))
        self._recorded_fmatrix = np.zeros((self.n_traj, self.tg_ic.shape[1], self.tg_ic.shape[2], self.n_records))

        for i in range(self.n_traj):
            self._ics_queue.put((i, self._time, self.ic[i], self.tg_ic[i], self._time_direction, self._write_steps,
                                 self._adjoint, self._inverse, self._boundary))

        self._ics_queue.join()

        for i in range(self.n_traj):
            args = self._traj_queue.get()
            self._recorded_traj[args[0]] = args[1]
            self._recorded_fmatrix[args[0]] = args[2]

        if len(tg_ic_sav.shape) == 2:
            if self._recorded_fmatrix.shape[1:3] != tg_ic_sav.shape:
                self._recorded_fmatrix = np.swapaxes(self._recorded_fmatrix, 1, 2)

        elif len(tg_ic_sav.shape) == 3:
            if tg_ic_sav.shape[1] != self.n_dim:
                if self._recorded_fmatrix.shape[:3] != tg_ic_sav.shape:
                    self._recorded_fmatrix = np.swapaxes(self._recorded_fmatrix, 1, 2)

    def get_trajectories(self):
        """Returns the result of the previous integrator integration.

        Returns
        -------
        time, traj, tg_traj: ~numpy.ndarray
            The result of the integration:

            * **time:** time at which the state of the system was saved. Array of shape (:attr:`n_records`,).
            * **traj:** Saved dynamical system states. 3D array of shape (:attr:`n_traj`, :attr:`n_dim`, :attr:`n_records`).
              If :attr:`n_traj` = 1, a 2D array of shape (:attr:`n_dim`, :attr:`n_records`) is returned instead.
            * **tg_traj:** Saved states of the linear ODEs.
              Depending on the input initial conditions of both ODEs,
              it is at maximum a 4D array of shape
              (:attr:`n_traj`, :attr:`n_tg_traj`, :attr:`n_dim`, :attr:`n_records`).
              If one of the dimension is 1, it is squeezed.
        """
        if self._write_steps > 0:
            if self._time_direction == 1:
                if self._time[::self._write_steps][-1] == self._time[-1]:
                    return self._time[::self._write_steps], np.squeeze(self._recorded_traj), \
                           np.squeeze(self._recorded_fmatrix)
                else:
                    return np.concatenate((self._time[::self._write_steps], np.full((1,), self._time[-1]))), \
                           np.squeeze(self._recorded_traj), np.squeeze(self._recorded_fmatrix)
            else:
                rtime = reverse(self._time[::-self._write_steps])
                if rtime[0] == self._time[0]:
                    return rtime, np.squeeze(self._recorded_traj), np.squeeze(self._recorded_fmatrix)
                else:
                    return np.concatenate((np.full((1,), self._time[0]), rtime)), np.squeeze(self._recorded_traj), \
                           np.squeeze(self._recorded_fmatrix)
        else:
            return self._time[-1], np.squeeze(self._recorded_traj), np.squeeze(self._recorded_fmatrix)

    def get_ic(self):
        """Returns the initial conditions of the non-linear ODEs stored in the integrator.

        Returns
        -------
        ~numpy.ndarray
            The initial conditions.
        """
        return self.ic

    def set_ic(self, ic):
        """Direct setter for the integrator's non-linear ODEs initial conditions

        Parameters
        ----------
        ic: ~numpy.ndarray(float)
            Initial condition of the system. Can be a 1D or a 2D array:

            * 1D: Provide a single initial condition.
              Should be of shape (`n_dim`,) where `n_dim` = :math:`\\mathrm{dim}(\\boldsymbol{x})`.
            * 2D: Provide an ensemble of initial condition.
              Should be of shape (`n_traj`, `n_dim`) where `n_dim` = :math:`\\mathrm{dim}(\\boldsymbol{x})`,
              and where `n_traj` is the number of initial conditions.
        """
        self.ic = ic

    def get_tg_ic(self):
        """Returns the initial conditions of the linear ODEs stored in the integrator.

        Returns
        -------
        ~numpy.ndarray
            The initial conditions.
        """
        return self.tg_ic

    def set_tg_ic(self, tg_ic):
        """Direct setter for the integrator's linear ODEs initial conditions

        Parameters
        ----------
        tg_ic: ~numpy.ndarray(float)
            Initial condition of the linear ODEs
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
        """
        self.tg_ic = tg_ic


class TglsTrajectoryProcess(multiprocessing.Process):
    """:class:`RungeKuttaTglsIntegrator`'s workers class. Allows to multi-thread time integration.

    .. _Runge-Kutta method: https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
    .. _Numba: https://numba.pydata.org/

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
        Queue to which the worker ask for initial conditions input.
    traj_queue: multiprocessing.Queue
        Queue to which the worker returns the integration results.

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
    def __init__(self, processID, func, func_jac, b, c, a, ics_queue, traj_queue):

        super().__init__()
        self.processID = processID
        self._ics_queue = ics_queue
        self._traj_queue = traj_queue
        self.func = func
        self.func_jac = func_jac
        self.a = a
        self.b = b
        self.c = c

    def run(self):
        """Main worker computing routine. Perform the time integration with the fetched initial conditions and parameters."""

        while True:

            args = self._ics_queue.get()

            recorded_traj, recorded_fmatrix = _integrate_runge_kutta_tgls_jit(self.func, self.func_jac, args[1], args[2][np.newaxis, ...],
                                                                              args[3][np.newaxis, ...], args[4], args[5],
                                                                              self.b, self.c, self.a,
                                                                              args[6], args[7], args[8])

            self._traj_queue.put((args[0], recorded_traj, recorded_fmatrix))

            self._ics_queue.task_done()


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from scipy.integrate import odeint

    @njit
    def f(t, x):
        return - np.array([1., 2., 3.]) * x

    def fr(x, t):
        return f(t, x)

    ic = np.random.randn(6).reshape(2, 3)

    integrator = RungeKuttaIntegrator()

    integrator.set_func(f)
    integrator.integrate(0., 10., 0.01, ic=ic, write_steps=1)

    time, r = integrator.get_trajectories()

    t = np.arange(0., 10., 0.1)
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
    plt.title('Forward')

    integrator.integrate(0., 10., 0.01, ic=ic, forward=False, write_steps=1)
    timet, rt = integrator.get_trajectories()
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
    plt.title('Backward')

    integrator.integrate(0., 10., 0.01, ic=ic, write_steps=0)
    tt, re = integrator.get_trajectories()
    print(tt)
    print(r[0, :, -1], re[0])
    plt.show(block=False)

    a = 0.25
    F = 16.
    G = 3.
    b = 6.


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


    integrator.set_func(fL84)
    integrator.integrate(0., 10000., 0.01, write_steps=0)
    tt, traj = integrator.get_trajectories()
    integrator.integrate(0.,20.,0.01,ic=traj, write_steps=10)
    tt, traj = integrator.get_trajectories()
    integrator.integrate(0.,20.,0.01,ic=traj[:,-1],write_steps=10, forward=False)
    ttb, trajb = integrator.get_trajectories()

    plt.figure()
    plt.plot(tt, traj[0])
    plt.plot(ttb, trajb[0])
    plt.show(block=False)

    plt.title('Lorenz 63 - Forward then backward')

    integrator.integrate(0., 10000., 0.01, write_steps=0)
    tt, traj = integrator.get_trajectories()
    integrator.integrate(0.,20.,0.01,ic=traj,write_steps=10, forward=False)
    tt, trajb = integrator.get_trajectories()
    integrator.integrate(0.,20.,0.01,ic=trajb[:,0],write_steps=10)
    ttb, traj = integrator.get_trajectories()

    plt.figure()
    plt.plot(tt, traj[0])
    plt.plot(ttb, trajb[0])
    plt.show(block=False)

    plt.title('Lorenz 63 - Backward then forward')
    integrator.terminate()

    tgls_integrator = RungeKuttaTglsIntegrator()
    tgls_integrator.set_func(fL84, DfL84)

    @njit
    def tboundary(t, x):
        return np.array([0.,x[1],0.])

    ic = np.random.randn(4, 3)
    tgls_integrator.initialize(10., 0.01, ic=ic)
    tgls_integrator.integrate(0., 20., 0.01, write_steps=10, tg_ic=np.zeros(3), boundary=tboundary)
    # x, fm = _integrate_runge_kutta_tgls_jit(fL84, DfL84, tgls_integrator.time, tgls_integrator.ic[0][np .newaxis,...], np.zeros((1,3,1)), 1, 1, tgls_integrator.b, tgls_integrator.c, tgls_integrator.a, False, 1., tboundary)

    t, x, fm = tgls_integrator.get_trajectories()

    tgls_integrator.terminate()
