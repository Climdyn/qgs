
import multiprocessing
import numpy as np
from numba import njit
from integrators.integrate import integrate_runge_kutta_jit, integrate_runge_kutta_tgls_jit1, zeros_func
from functions.util import reverse

# Integrator classes


class RungeKuttaIntegrator(object):

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
        self.time = None
        self.recorded_traj = None
        self.n_traj = 0
        self.n_dim = number_of_dimensions
        self.n_records = 0
        self.write_steps = 0
        self.time_direction = 1

        self.func = None

        self.ics_queue = None
        self.traj_queue = None

        self.processes_list = list()

    def terminate(self):

        for process in self.processes_list:

            process.terminate()
            process.join()

    def start(self):

        self.terminate()

        self.processes_list = list()
        self.ics_queue = multiprocessing.JoinableQueue()
        self.traj_queue = multiprocessing.Queue()

        for i in range(self.num_threads):
            self.processes_list.append(TrajectoryProcess(i, self.func, self.b, self.c, self.a,
                                                         self.ics_queue, self.traj_queue))

        for process in self.processes_list:
            process.daemon = True
            process.start()

    def set_func(self, f, ic_init=True):

        self.func = f
        if ic_init:
            self.ic = None
        self.start()

    def set_bca(self, b=None, c=None, a=None, ic_init=True):
        self.a = a
        self.b = b
        self.c = c
        if ic_init:
            self.ic = None
        self.start()

    def initialize(self, convergence_time, dt, pert_size=0.01, reconvergence_time=None, forward=True,
                   number_of_trajectories=1, ic=None, reconverge=None):

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
        self.time = np.concatenate((np.arange(t0, t, dt), np.full((1,), t)))
        self.write_steps = write_steps

        if forward:
            self.time_direction = 1
        else:
            self.time_direction = -1

        if write_steps == 0:
            self.n_records = 1
        else:
            tot = self.time[::self.write_steps]
            self.n_records = len(tot)
            if tot[-1] != self.time[-1]:
                self.n_records += 1

        self.recorded_traj = np.zeros((self.n_traj, self.n_dim, self.n_records))

        for i in range(self.n_traj):
            self.ics_queue.put((i, self.time, self.ic[i], self.time_direction, self.write_steps))

        self.ics_queue.join()

        for i in range(self.n_traj):
            args = self.traj_queue.get()
            self.recorded_traj[args[0]] = args[1]

    def get_trajectories(self):

        if self.write_steps > 0:
            if self.time_direction == 1:
                if self.time[::self.write_steps][-1] == self.time[-1]:
                    return self.time[::self.write_steps], np.squeeze(self.recorded_traj)
                else:
                    return np.concatenate((self.time[::self.write_steps], np.full((1,), self.time[-1]))), \
                           np.squeeze(self.recorded_traj)
            else:
                rtime = reverse(self.time[::-self.write_steps])
                if rtime[0] == time[0]:
                    return rtime, np.squeeze(self.recorded_traj)
                else:
                    return np.concatenate((np.full((1,), self.time[0]), rtime)), np.squeeze(self.recorded_traj)

        else:
            return self.time[-1], np.squeeze(self.recorded_traj)

    def get_ic(self):
        return self.ic

    def set_ic(self, ic):
        self.ic = ic


class TrajectoryProcess(multiprocessing.Process):

    def __init__(self, processID, func, b, c, a, ics_queue, traj_queue):

        super().__init__()
        self.processID = processID
        self.ics_queue = ics_queue
        self.traj_queue = traj_queue
        self.func = func
        self.a = a
        self.b = b
        self.c = c

    def run(self):

        while True:

            args = self.ics_queue.get()

            recorded_traj = integrate_runge_kutta_jit(self.func, args[1], args[2][np.newaxis, :], args[3], args[4],
                                                      self.b, self.c, self.a)

            self.traj_queue.put((args[0], recorded_traj))

            self.ics_queue.task_done()


class RungeKuttaTglsIntegrator(object):

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
        self.time = None
        self.recorded_traj = None
        self.recorded_fmatrix = None
        self.n_traj = 0
        self.n_tgtraj = 0
        self.n_dim = number_of_dimensions
        self.n_records = 0
        self.write_steps = 0

        self.time_direction = 1
        self.adjoint = False
        self.boundary = None
        self.inverse = 1.

        self.func = None
        self.func_jac = None

        self.ics_queue = None
        self.traj_queue = None

        self.processes_list = list()

    def terminate(self):

        for process in self.processes_list:

            process.terminate()
            process.join()

    def start(self):

        self.terminate()

        self.processes_list = list()
        self.ics_queue = multiprocessing.JoinableQueue()
        self.traj_queue = multiprocessing.Queue()

        for i in range(self.num_threads):
            self.processes_list.append(TglsTrajectoryProcess(i, self.func, self.func_jac, self.b, self.c, self.a,
                                                             self.ics_queue, self.traj_queue))

        for process in self.processes_list:
            process.daemon = True
            process.start()

    def set_func(self, f, fjac, ic_init=True):

        self.func = f
        self.func_jac = fjac
        if ic_init:
            self.ic = None
        self.start()

    def set_bca(self, b=None, c=None, a=None, ic_init=True):
        self.a = a
        self.b = b
        self.c = c
        if ic_init:
            self.ic = None
        self.start()

    def initialize(self, convergence_time, dt, pert_size=0.01, reconvergence_time=None, forward=True,
                   number_of_trajectories=1, ic=None, reconverge=None):

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
        self.time = np.concatenate((np.arange(t0, t, dt), np.full((1,), t)))
        self.write_steps = write_steps

        if tg_ic is None:
            tg_ic = np.eye(self.ic.shape[1])

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
                ict = tg_ic.copy()
                for i in range(self.n_traj-1):
                    ict = np.concatenate((ict, tg_ic))
                self.tg_ic = ict

        if forward:
            self.time_direction = 1
        else:
            self.time_direction = -1

        self.adjoint = adjoint

        if boundary is None:
            self.boundary = zeros_func
        else:
            self.boundary = boundary

        self.inverse = 1.
        if inverse:
            self.inverse *= -1.

        if write_steps == 0:
            self.n_records = 1
        else:
            tot = self.time[::self.write_steps]
            self.n_records = len(tot)
            if tot[-1] != self.time[-1]:
                self.n_records += 1

        self.recorded_traj = np.zeros((self.n_traj, self.n_dim, self.n_records))
        self.recorded_fmatrix = np.zeros((self.n_traj, self.tg_ic.shape[1], self.tg_ic.shape[2], self.n_records))

        for i in range(self.n_traj):
            self.ics_queue.put((i, self.time, self.ic[i], self.tg_ic[i], self.time_direction, self.write_steps,
                                self.adjoint, self.inverse, self.boundary))

        self.ics_queue.join()

        for i in range(self.n_traj):
            args = self.traj_queue.get()
            self.recorded_traj[args[0]] = args[1]
            self.recorded_fmatrix[args[0]] = args[2]

    def get_trajectories(self):

        if self.write_steps > 0:
            if self.time_direction == 1:
                if self.time[::self.write_steps][-1] == self.time[-1]:
                    return self.time[::self.write_steps], np.squeeze(self.recorded_traj),\
                           np.squeeze(self.recorded_fmatrix)
                else:
                    return np.concatenate((self.time[::self.write_steps], np.full((1,), self.time[-1]))), \
                           np.squeeze(self.recorded_traj), np.squeeze(self.recorded_fmatrix)
            else:
                rtime = reverse(self.time[::-self.write_steps])
                if rtime[0] == time[0]:
                    return rtime, np.squeeze(self.recorded_traj), np.squeeze(self.recorded_fmatrix)
                else:
                    return np.concatenate((np.full((1,), self.time[0]), rtime)), np.squeeze(self.recorded_traj),\
                           np.squeeze(self.recorded_fmatrix)
        else:
            return self.time[-1], np.squeeze(self.recorded_traj), np.squeeze(self.recorded_fmatrix)

    def get_ic(self):
        return self.ic

    def set_ic(self, ic):
        self.ic = ic

    def get_tg_ic(self):
        return self.tg_ic

    def set_tg_ic(self, tg_ic):
        self.tg_ic = tg_ic


class TglsTrajectoryProcess(multiprocessing.Process):

    def __init__(self, processID, func, func_jac, b, c, a, ics_queue, traj_queue):

        super().__init__()
        self.processID = processID
        self.ics_queue = ics_queue
        self.traj_queue = traj_queue
        self.func = func
        self.func_jac = func_jac
        self.a = a
        self.b = b
        self.c = c

    def run(self):

        while True:

            args = self.ics_queue.get()

            recorded_traj, recorded_fmatrix = integrate_runge_kutta_tgls_jit1(self.func, self.func_jac, args[1], args[2][np.newaxis, ...],
                                                            args[3][np.newaxis, ...], args[4], args[5],
                                                            self.b, self.c, self.a,
                                                            args[6], args[7], args[8])

            self.traj_queue.put((args[0], recorded_traj, recorded_fmatrix))

            self.ics_queue.task_done()


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
    # x, fm = integrate_runge_kutta_tgls_jit1(fL84, DfL84, tgls_integrator.time, tgls_integrator.ic[0][np .newaxis,...], np.zeros((1,3,1)), 1, 1, tgls_integrator.b, tgls_integrator.c, tgls_integrator.a, False, 1., tboundary)

    t, x, fm = tgls_integrator.get_trajectories()

    tgls_integrator.terminate()
