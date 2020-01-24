
# TODO : - ...
#        - load and save function

import numpy as np


class Params(object):

    _name = ""

    def __init__(self, dic=None):

        self.set_params(dic)

    def set_params(self, dic):
        if dic is not None:
            for key, val in zip(dic.keys(), dic.values()):
                if key in self.__dict__.keys():
                    self.__dict__[key] = val

    def __str__(self):
        s = ""
        for key, val in zip(self.__dict__.keys(), self.__dict__.values()):
            if 'params' not in key and key[0] != '_':
                s += "'"+key+"': "+str(val)+",\n"
        return s

    def _list_params(self):
        return self._name+" Parameters:\n"+self.__str__()

    def print_params(self):
        print(self._list_params())

    def __repr__(self):
        s = super(Params, self).__repr__()+"\n"+self._list_params()
        return s


class ScaleParams(Params):

    _name = "Scale"

    def __init__(self, dic=None):

        Params.__init__(self, dic)

        # -----------------------------------------------------------
        # Scale parameters for the ocean and the atmosphere
        # -----------------------------------------------------------

        self.scale = 5.e6  # the characteristic space scale, L*pi
        self.f0 = 1.032e-4  # Coriolis parameter at 45 degrees latitude
        self.n = 1.3e0  # aspect ratio (n = 2Ly/Lx ; Lx = 2*pi*L/n; Ly = pi*L)
        self.rra = 6370.e3  # earth radius
        self.phi0_npi = 0.25e0  # latitude exprimed in fraction of pi

        self.hk = None  # orography coefficients

        self.set_params(dic)

    # -----------------------------------------------------------
    # Some general parameters (Domain, beta, friction, orography)
    # -----------------------------------------------------------

    @property
    def L(self):
        return self.scale / np.pi

    @property
    def phi0(self):
        return self.phi0_npi * np.pi

    @property
    def betp(self):
        return self.L / self.rra * np.cos(self.phi0) / np.sin(self.phi0)


class AtmosphericParams(Params):

    _name = "Atmospheric"

    def __init__(self, dic=None):

        Params.__init__(self, dic)

        # Parameters for the atmosphere
        self.k = 0.05  # atmosphere bottom friction coefficient
        self.kp = 0.01  # atmosphere internal friction coefficient
        self.sig0 = 0.1e0  # static stability of the atmosphere

        self.set_params(dic)

    @property
    def kd(self):
        return self.k * 2

    @property
    def kdp(self):
        return self.kp


class AtmosphericTemperatureParams(Params):

    _name = "Atmospheric Temperature"

    def __init__(self, dic=None):

        Params.__init__(self, dic)

        self.hpp = 0.045  # Newtonian cooling coefficient
        self.thetas = None  # Newtonian cooling coefficients

        self.G = 1.e7  # Specific heat capacity of the atmosphere
        self.C = 100.e0  # Constant short-wave radiation of the atmosphere
        self.eps = 0.76e0  # Emissivity coefficient for the grey-body atmosphere
        self.T0 = 270.0  # Stationary solution for the 0-th order atmospheric temperature

        self.sc = 1.  # Ratio of surface to atmosphere temperature
        self.hlambda = 20.00  # Sensible+turbulent heat exchange between oc and atm

        self.set_params(dic)


class OceanicParams(Params):

    _name = "Oceanic"

    def __init__(self, dic=None):

        Params.__init__(self, dic)

        self.gp = 3.1e-2  # reduced gravity
        self.r = 1.e-8  # frictional coefficient at the bottom of the ocean
        self.h = 5.e2  # depth of the water layer of the ocean
        self.d = 1.e-8  # the coupling parameter (should be divided by f0 to be adim)

        self.set_params(dic)


class OceanicTemperatureParams(Params):

    _name = "Oceanic Temperature"

    def __init__(self, dic=None):

        Params.__init__(self, dic)

        self.G = 2.e8  # Specific heat capacity of the ocean (50m layer)
        self.C = 350  # Constant short-wave radiation of the ocean
        self.T0 = 285.0  # Stationary solution for the 0-th order ocean temperature

        self.set_params(dic)


class QgParams(Params):

    _name = "General"

    def __init__(self, dic=None, scale_params=None,
                 atmospheric_params=True, atemperature_params=True,
                 oceanic_params=None, otemperature_params=None):

        Params.__init__(self, dic)

        # General scale parameters object (Mandatory param block)
        if scale_params is None:
            self.scale_params = ScaleParams(dic)
        else:
            self.scale_params = scale_params

        # Atmospheric parameters object
        if atmospheric_params is True:
            self.atmospheric_params = AtmosphericParams(dic)
        else:
            self.atmospheric_params = atmospheric_params

        # Atmospheric temperature parameters object
        if atmospheric_params is True:
            self.atemperature_params = AtmosphericTemperatureParams(dic)
        else:
            self.atemperature_params = atemperature_params

        if oceanic_params is True:
            self.oceanic_params = OceanicParams(dic)
        else:
            self.oceanic_params = oceanic_params

        if otemperature_params is True:
            self.otemperature_params = OceanicTemperatureParams(dic)
        else:
            self.otemperature_params = otemperature_params

        self._number_of_dimensions = 0
        self._number_of_atmospheric_modes = 0
        self._number_of_oceanic_modes = 0
        self._ams = None
        self._oms = None

        self._atmospheric_var_string = list()
        self._oceanic_var_string = list()
        self.time_unit = 'days'

        # Physical constants

        self.rr = 287.058e0  # Gas constant of dry air
        self.sb = 5.67e-8  # Stefan-Boltzmann constant

        self.set_params(dic)

    # -----------------------------------------------------------
    # Derived Quantities (Parameters)
    # -----------------------------------------------------------

    @property
    def LR(self):
        op = self.oceanic_params
        scp = self.scale_params
        if op is not None:
            return np.sqrt(op.gp * op.h) / scp.f0
        else:
            return None

    @property
    def G(self):
        scp = self.scale_params
        if self.LR is not None:
            return -scp.L**2 / self.LR**2
        else:
            return None

    @property
    def rp(self):
        op = self.oceanic_params
        scp = self.scale_params
        if op is not None:
            return op.r / scp.f0
        else:
            return None

    @property
    def dp(self):
        op = self.oceanic_params
        scp = self.scale_params
        if op is not None:
            return op.d / scp.f0
        else:
            return None

    @property
    def Cpo(self):
        otp = self.otemperature_params
        scp = self.scale_params
        if otp is not None:
            return otp.C / (otp.G * scp.f0) * self.rr / (scp.f0 ** 2 * scp.L ** 2)
        else:
            return None

    @property
    def Lpo(self):
        atp = self.atemperature_params
        otp = self.otemperature_params
        scp = self.scale_params
        if atp is not None and otp is not None:
            return atp.hlambda / (otp.G * scp.f0)
        else:
            return None

    # Cpa acts on psi1-psi3, not on theta :
    @property
    def Cpa(self):
        atp = self.atemperature_params
        scp = self.scale_params
        if atp is not None and atp.hpp == 0:
            return atp.C / (atp.G * scp.f0) * self.rr / (scp.f0 ** 2 * scp.L ** 2) / 2
        else:
            return None

    @property
    def Lpa(self):
        atp = self.atemperature_params
        scp = self.scale_params
        if atp is not None and atp.hpp == 0:
            return atp.hlambda / (atp.G * scp.f0)
        else:
            return None

    # long wave radiation lost by ocean to atmosphere space :
    @property
    def sbpo(self):
        otp = self.otemperature_params
        scp = self.scale_params
        if otp is not None:
            return 4 * self.sb * otp.T0 ** 3 / (otp.G * scp.f0)
        else:
            return None

    # long wave radiation from atmosphere absorbed by ocean
    @property
    def sbpa(self):
        atp = self.atemperature_params
        otp = self.otemperature_params
        scp = self.scale_params
        if otp is not None and atp is not None:
            return 8 * atp.eps * self.sb * atp.T0 ** 3 / (otp.G * scp.f0)
        else:
            return None

    # long wave radiation from ocean absorbed by atmosphere
    @property
    def LSBpo(self):
        atp = self.atemperature_params
        otp = self.otemperature_params
        scp = self.scale_params
        if atp is not None and otp is not None:
            return 2 * atp.eps * self.sb * otp.T0 ** 3 / (atp.G * scp.f0)
        else:
            return None

    # long wave radiation lost by atmosphere to space & ocean
    @property
    def LSBpa(self):
        atp = self.atemperature_params
        scp = self.scale_params
        if atp is not None and atp.hpp == 0:
            return 8 * atp.eps * self.sb * atp.T0 ** 3 / (atp.G * scp.f0)
        else:
            return None

    def set_params(self, dic):

        if dic is not None:
            for key, val in zip(dic.keys(), dic.values()):
                if key in self.__dict__.keys():
                    self.__dict__[key] = val

            if 'scale_params' in self.__dict__.keys():
                self.scale_params.set_params(dic)
            if 'atmospheric_params' in self.__dict__.keys():
                if self.atmospheric_params is not None:
                    self.atmospheric_params.set_params(dic)

            if 'atemperature_params' in self.__dict__.keys():
                if self.atemperature_params is not None:
                    self.atemperature_params.set_params(dic)

            if 'oceanic_params' in self.__dict__.keys():
                if self.oceanic_params is not None:
                    self.oceanic_params.set_params(dic)

            if 'otemperature_params' in self.__dict__.keys():
                if self.otemperature_params is not None:
                    self.otemperature_params.set_params(dic)

    def print_params(self):
        s = self._list_params()+"\n"
        if 'scale_params' in self.__dict__.keys():
            s += self.scale_params._list_params()+"\n"
        if 'atmospheric_params' in self.__dict__.keys():
            if self.atmospheric_params is not None:
                s += self.atmospheric_params._list_params()+"\n"

        if 'atemperature_params' in self.__dict__.keys():
            if self.atemperature_params is not None:
                s += self.atemperature_params._list_params()+"\n"

        if 'oceanic_params' in self.__dict__.keys():
            if self.oceanic_params is not None:
                s += self.oceanic_params._list_params()+"\n"

        if 'otemperature_params' in self.__dict__.keys():
            if self.otemperature_params is not None:
                s += self.otemperature_params._list_params()+"\n"

        print("Qgs parameters summary")
        print("======================\n")
        print(s)

    @property
    def ndim(self):
        return self._number_of_dimensions

    @property
    def nmod(self):
        return [self._number_of_atmospheric_modes, self._number_of_oceanic_modes]

    @property
    def ablocks(self):
        return self._ams

    @ablocks.setter
    def ablocks(self, value):
        self._ams = value

        namod = 0
        for i in range(self.ablocks.shape[0]):
            if self.ablocks[i, 0] == 1:
                namod += 3
            else:
                namod += 2

        self._number_of_atmospheric_modes = namod
        self._number_of_dimensions = 2 * (namod + self._number_of_oceanic_modes)

        self.scale_params.hk = np.zeros(namod)
        self.scale_params.hk[1] = 0.1
        if self.atemperature_params is not None:
            self.atemperature_params.thetas = np.zeros(namod)
            self.atemperature_params.thetas[0] = 0.1

    @property
    def oblocks(self):
        return self._oms

    @oblocks.setter
    def oblocks(self, value):
        self._oms = value

        self._number_of_oceanic_modes = self.oblocks.shape[0]
        self._number_of_dimensions = 2 * (self._number_of_oceanic_modes + self._number_of_atmospheric_modes)

        # if setting an ocean, then disable the orography and the Newtonian cooling
        self.scale_params.hk = np.zeros(self.nmod[0])
        if self.atemperature_params is not None:
            self.atemperature_params.thetas = np.zeros(self.nmod[0])
            self.atemperature_params.hpp = 0

    def set_max_atmospheric_modes(self, nxmax, nymax, auto=False):
        res = np.zeros((nxmax * nymax, 2), dtype=np.int)
        i = 0
        for nx in range(1, nxmax + 1):
            for ny in range(1, nymax+1):
                res[i, 0] = nx
                res[i, 1] = ny
                i += 1

        self.ablocks = res

        self._atmospheric_var_string = list()
        for i in range(self.nmod[0]):
            self._atmospheric_var_string.append(r'psi_{\rm a,' + str(i + 1) + "}")
        for i in range(self.nmod[0]):
            self._atmospheric_var_string.append(r'theta_{\rm a,' + str(i + 1) + "}")

        if auto:
            if self.atemperature_params is None:
                self.atemperature_params = AtmosphericTemperatureParams()
            if self.atmospheric_params is None:
                self.atmospheric_params = AtmosphericParams()

    def set_max_oceanic_modes(self, nxmax, nymax, auto=True):
        res = np.zeros((nxmax * nymax, 2), dtype=np.int)
        i = 0
        for nx in range(1, nxmax + 1):
            for ny in range(1, nymax+1):
                res[i, 0] = nx
                res[i, 1] = ny
                i += 1

        self.oblocks = res

        self._oceanic_var_string = list()
        for i in range(self.nmod[1]):
            self._oceanic_var_string.append(r'psi_{\rm o,' + str(i + 1) + "}")
        for i in range(self.nmod[1]):
            self._oceanic_var_string.append(r'theta_{\rm o,' + str(i + 1) + "}")

        if auto:
            if self.otemperature_params is None:
                self.otemperature_params = OceanicTemperatureParams()
            if self.oceanic_params is None:
                self.oceanic_params = OceanicParams()

    @property
    def var_string(self):
        l = list()
        for var in self._atmospheric_var_string:
            l.append(var)
        for var in self._oceanic_var_string:
            l.append(var)

        return l

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
        return 1 / (self.scale_params.f0 * c)


