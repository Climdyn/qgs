import numpy as np


class QgParams(object):

    def __init__(self, dic=None):

        # -----------------------------------------------------------
        # Scale parameters for the ocean and the atmosphere
        # -----------------------------------------------------------

        self.scale = 5.e6  # the characteristic space scale, L*pi
        self.f0 = 1.032e-4  # Coriolis parameter at 45 degrees latitude
        self.n = 1.3e0  # aspect ratio (n = 2Ly/Lx ; Lx = 2*pi*L/n; Ly = pi*L)
        self.rra = 6370.e3  # earth radius
        self.phi0_npi = 0.25e0  # latitude exprimed in fraction of pi

        # Parameters for the atmosphere
        self.k = 0.05  # atmosphere bottom friction coefficient
        self.kp = 0.01  # atmosphere internal friction coefficient
        self.sig0 = 0.1e0  # static stability of the atmosphere
        self.hpp = 0.045  # Newtonian cooling coefficient

        # -----------------------------------------------------------
        # Some general parameters (Domain, beta, friction, orography)
        # -----------------------------------------------------------

        pi = np.pi
        self.L = self.scale / pi
        self.phi0 = self.phi0_npi * pi
        self.betp = self.L / self.rra * np.cos(self.phi0) / np.sin(self.phi0)
        self.kd = self.k * 2
        self.kdp = self.kp

        self._number_of_dimensions = 0
        self._number_of_modes = 0
        self._ams = None

        self.hk = None  # orography coefficients
        self.tethas = None  # Newtonian cooling coefficients

        if dic is not None:
            self.set_params(dic)

        self.var_string = list()
        self.time_unit = 'days'

    def set_params(self, dic):

        for key, val in zip(dic.keys(), dic.values()):
            self.__dict__[key] = val

    @property
    def ndim(self):
        return self._number_of_dimensions

    @property
    def nmod(self):
        return self._number_of_modes

    @property
    def ablocks(self):
        return self._ams

    @ablocks.setter
    def ablocks(self, value):
        self._ams = value
        nmod = 0
        for i in range(0, self.ablocks.shape[0]):
            if self.ablocks[i, 0] == 1:
                nmod = nmod + 3
            else:
                nmod = nmod + 2

        self._number_of_modes = nmod
        self._number_of_dimensions = 2 * nmod
        self.hk = np.zeros(nmod)
        self.thetas = np.zeros(nmod)

        self.hk[1] = 0.1
        self.thetas[0] = 0.1

    def set_max_modes(self, nxmax, nymax):
        res = np.zeros((nxmax * nymax, 2))
        i = 0
        for nx in range(1, nxmax + 1):
            for ny in range(1, nymax+1):
                res[i, 0] = nx
                res[i, 1] = ny
                i += 1

        self.ablocks = res

        self.var_string = list()
        for i in range(self.nmod):
            self.var_string.append('psi_' + str(i + 1))
        for i in range(self.nmod):
            self.var_string.append('theta_' + str(i + 1))

    @property
    def latex_var_string(self):
        l = list()
        for var in self.var_string:
            l.append(r'{\ '[0:-1] + var + r'}')

        return l

    @property
    def dimensional_time(self):
        c = 24 * 3600
        if self.time_unit == 'days':
            c = 24 * 3600
        return 1 / (self.f0 * c)


