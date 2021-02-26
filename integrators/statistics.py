
# TODO : document this module !

import numpy as np


class TrajectoriesStatistics(object):

    def __init__(self):

        self.ic = None
        self.integrator = None
        self.func_list = list()
        self.mean_func = list()

    def initialize(self, convergence_time, dt, pert_size=0.01, reconvergence_time=None,
                   number_of_trajectories=1, ic=None):

        self.integrator.initialize(convergence_time, dt, pert_size=pert_size, reconvergence_time=reconvergence_time,
                                   number_of_trajectories=number_of_trajectories, ic=ic)
        self.ic = self.integrator.get_ic()

    def set_func_list(self, func_list):
        self.func_list = func_list

    def set_integrator(self, integrator):
        self.integrator = integrator

    def set_ic(self, ic):

        self.ic = ic

    def compute_stats(self, t0, t, dt, ic=None, forward=True, write_steps=1, num=1):

        if ic is not None:
            self.set_ic(ic)

        number_of_trajectories = self.ic.shape[0]
        sub_num_traj = number_of_trajectories // num

        self.integrator.integrate(t0, t, dt, ic=self.ic[:sub_num_traj],
                                  forward=forward, write_steps=write_steps)

        time, traj = self.integrator.get_trajectories()

        mean_func_realization = np.zeros((len(self.func_list), num, traj.shape[1], traj.shape[2]))

        for j, f in enumerate(self.func_list):
            mean_func_realization[j, 0] = np.mean(f(traj), axis=0)

        for i in range(1, num-1):

            self.integrator.integrate(t0, t, dt, ic=self.ic[i*sub_num_traj:(i+1)*sub_num_traj],
                                      forward=forward, write_steps=write_steps)
            time, traj = self.integrator.get_trajectories()

            for j, f in enumerate(self.func_list):
                mean_func_realization[j, i] = np.mean(f(traj), axis=0)

        self.integrator.integrate(t0, t, dt, ic=self.ic[(num-1)*sub_num_traj:], forward=forward, write_steps=write_steps)
        time, traj = self.integrator.get_trajectories()

        for j, f in enumerate(self.func_list):
            mean_func_realization[j, num-1] = np.mean(f(traj), axis=0)

        self.mean_func = np.mean(mean_func_realization, axis=1)

    def get_ic(self):
        return self.ic

    def get_stats(self):
        return self.mean_func





