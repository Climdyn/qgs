"""
    The model's parameters module
    =============================

    This module defines the main classes containing the model configuration parameters.
    The parameters are typically specified as :class:`~params.parameter.Parameter` objects.

    There are five types of parameters arranged in classes:

    * :class:`ScaleParams` contains the model scale parameters. These parameters are used to scale and
      `nondimentionalize`_ the :class:`~params.parameter.Parameter` of the other parameters classes according to
      their :attr:`~params.parameter.Parameter.units` attribute.
    * :class:`AtmosphericParams` contains the atmospheric dynamical parameters.
    * :class:`AtmosphericTemperatureParams` containing the atmosphere's temperature and heat-exchange parameters.
    * :class:`OceanicParams` contains the oceanic dynamical parameters.
    * :class:`OceanicTemperatureParams` contains the ocean's temperature and heat-exchange parameters.

    These parameters classes are regrouped into a global structure :class:`QgParams` which also contains

    * spectral modes definition of the model
    * physical constants
    * parameters derived from the ones provided by the user
    * helper functions to initialize and parameterize the model

    This global parameters structure is used by the other modules to construct the model's ordinary differential
    equations.

    Description of the classes
    --------------------------

    .. _nondimentionalize: https://en.wikipedia.org/wiki/Nondimensionalization
"""

# TODO : - load and save function (pickle)
#        -

import numpy as np
from params.parameter import Parameter


class Params(object):
    """Base class for a model's parameters container.

    Parameters
    ----------
    dic: dict(float or Parameter), optional
        A dictionary with the parameters names and values to be assigned.
    """

    _name = ""

    def __init__(self, dic=None):

        self.set_params(dic)

    def set_params(self, dic):
        """Set the specified parameters values.

        Parameters
        ----------
        dic: dict(float or Parameter)
            A dictionary with the parameters names and values to be assigned.
        """
        if dic is not None:
            for key, val in zip(dic.keys(), dic.values()):
                if key in self.__dict__.keys():
                    if isinstance(self.__dict__[key], Parameter):
                        if isinstance(val, Parameter):
                            self.__dict__[key] = val
                        else:
                            d = self.__dict__[key].__dict__
                            self.__dict__[key] = Parameter(val, input_dimensional=d['_input_dimensional'],
                                                           units=d['_units'],
                                                           description=d['_description'],
                                                           scale_object=d['_scale_object'],
                                                           return_dimensional=d['_return_dimensional'])
                    else:
                        self.__dict__[key] = val

    def __str__(self):
        s = ""
        for key, val in zip(self.__dict__.keys(), self.__dict__.values()):
            if 'params' not in key and key[0] != '_':
                if isinstance(val, Parameter):
                    if val.input_dimensional:
                        units = val.units
                        efval = val.dimensional_value
                    else:
                        efval = val.nondimensional_value
                        if val.nondimensional_value == val.dimensional_value:
                            units = ""
                        else:
                            units = "[nondim]"
                    s += "'" + key + "': " + str(efval) + "  " + units + "  (" + val.description + "),\n"
                else:
                    s += "'"+key+"': "+str(val)+",\n"
        return s

    def _list_params(self):
        return self._name+" Parameters:\n"+self.__str__()

    def print_params(self):
        """Print the parameters contained in the container."""
        print(self._list_params())

    def __repr__(self):
        s = super(Params, self).__repr__()+"\n"+self._list_params()
        return s


class ScaleParams(Params):
    """Class containing the model scales parameters.

    Parameters
    ----------
    dic: dict(float or Parameter), optional
        A dictionary with the parameters names and values to be assigned.

    Attributes
    ----------
    scale: Parameter
        The characteristic meridional space scale, :math:`L_y = \pi \, L`, in meters [:math:`m`].
    f0: Parameter
        Coriolis parameter, in [:math:`s^{-1}`].
    n: Parameter
        Model domain aspect ratio, :math:`n = 2 L_y/L_x` .
    rra: Parameter
        Earth radius, in meters [:math:`m`].
    phi0_npi: Parameter
        Latitude exprimed in fraction of :math:`\pi` .
    hk: ~numpy.ndarray(float)
        Orography coefficients (non-dimensional), an array of shape (:attr:`~QgParams.nmod` [0],).
    deltap: Parameter
        Difference of pressure between the center of the two atmospheric layers, in [:math:`Pa`].
    """
    _name = "Scale"

    def __init__(self, dic=None):

        Params.__init__(self, dic)

        # -----------------------------------------------------------
        # Scale parameters for the ocean and the atmosphere
        # -----------------------------------------------------------

        self.scale = Parameter(5.e6, units='[m]', description="characteristic space scale (L*pi)",
                               return_dimensional=True)
        self.f0 = Parameter(1.032e-4, units='[s^-1]', description="Coriolis parameter at the middle of the domain",
                            return_dimensional=True)
        self.n = Parameter(1.3e0, input_dimensional=False, description="aspect ratio (n = 2 L_y / L_x)")
        self.rra = Parameter(6370.e3, units='[m]', description="earth radius", return_dimensional=True)
        self.phi0_npi = Parameter(0.25e0, input_dimensional=False, description="latitude exprimed in fraction of pi")
        self.deltap = Parameter(5.e4, units='[Pa]', description='pressure difference between the two atmospheric layers',
                                return_dimensional=True)

        self.hk = None  # orography coefficients

        self.set_params(dic)

    # ----------------------------------------
    # Some derived parameters (Domain, beta)
    # ----------------------------------------

    @property
    def L(self):
        """Parameter: Typical length scale :math:`L`  of the model, in meters [:math:`m`]."""
        return Parameter(self.scale / np.pi, units=self.scale.units, description='Typical length scale L',
                         return_dimensional=True)

    @property
    def L_y(self):
        """Parameter: The meridional extent :math:`L_y = \pi \, L` of the model's domain, in meters [:math:`m`]."""
        return Parameter(self.scale, units=self.scale.units, description='The meridional extent of the model domain',
                         return_dimensional=True)

    @property
    def L_x(self):
        """Parameter: The zonal extent :math:`L_x = 2 \pi \, L / n` of the model's domain, in meters [:math:`m`]."""
        return Parameter(2 * self.scale / self.n, units=self.scale.units,
                         description='The zonal extent of the model domain',
                         return_dimensional=True)

    @property
    def phi0(self):
        """Parameter: The reference latitude :math:`\phi_0` at the center of the domain, expressed in radians [:math:`rad`]."""
        return Parameter(self.phi0_npi * np.pi, units='[rad]',
                         description="The reference latitude of the center of the domain",
                         return_dimensional=True)

    @property
    def beta(self):
        """Parameter: The meridional gradient of the Coriolis parameter at :math:`\phi_0`, expressed in [:math:`m^{-1} s^{-1}`]. """
        return Parameter(self.L / self.rra * np.cos(self.phi0) / np.sin(self.phi0), input_dimensional=False,
                         units='[m^-1][s^-1]', scale_object=self,
                         description="Meridional gradient of the Coriolis parameter at phi_0")


class AtmosphericParams(Params):
    """Class containing the atmospheric parameters.

    Parameters
    ----------
    scale_params: ScaleParams
        The scale parameters object of the model.
    dic: dict(float or Parameter), optional
        A dictionary with the parameters names and values to be assigned.

    Attributes
    ----------
    k: Parameter
        Atmosphere bottom friction coefficient.
    kp: Parameter
        Atmosphere internal friction coefficient.
    sigma: Parameter
        Static stability of the atmosphere.
    """

    _name = "Atmospheric"

    def __init__(self, scale_params, dic=None):

        Params.__init__(self, dic)

        self._scale_params = scale_params

        # Parameters for the atmosphere
        self.kd = Parameter(0.1, input_dimensional=False, scale_object=scale_params, units='[s^-1]',
                            description="atmosphere bottom friction coefficient")
        self.kdp = Parameter(0.01, input_dimensional=False, scale_object=scale_params, units='[s^-1]',
                            description="atmosphere internal friction coefficient")
        self.sigma = Parameter(0.2e0, input_dimensional=False, scale_object=scale_params, units='[m^2][s^-2][Pa^-2]',
                               description="static stability of the atmosphere")

        self.set_params(dic)

    @property
    def sig0(self):
        return Parameter(self.sigma / 2, input_dimensional=False, scale_object=self._scale_params, units='[m^2][s^-2][Pa^-2]',
                         description="0.5 * static stability of the atmosphere")


class AtmosphericTemperatureParams(Params):
    """Class containing the atmospheric temperature parameters.

    Parameters
    ----------
    scale_params: ScaleParams
        The scale parameters object of the model.
    dic: dict(float or Parameter), optional
        A dictionary with the parameters names and values to be assigned.

    Attributes
    ----------
    hd: Parameter
        Newtonian cooling coefficient.
        Used if an orography is provided.
    thetas: ~numpy.ndarray(float)
        Spatial Newtonian cooling coefficients (non-dimensional).
    gamma: Parameter
        Specific heat capacity of the atmosphere
    C: Parameter
        Constant short-wave radiation of the atmosphere
    eps: Parameter
        Emissivity coefficient for the grey-body atmosphere
    T0: Parameter
        Stationary solution for the 0-th order atmospheric temperature [:math:`K`].
    sc: Parameter
        Ratio of surface to atmosphere temperature
    hlambda: Parameter
        Sensible + turbulent heat exchange between ocean and atmosphere
    """
    _name = "Atmospheric Temperature"

    def __init__(self, scale_params, dic=None):

        Params.__init__(self, dic)

        self._scale_params = scale_params

        self.hd = Parameter(0.045, input_dimensional=False, units='[s]', scale_object=scale_params,
                            description="Newtonian cooling coefficient")
        self.thetas = None  # Radiative equilibrium mean temperature decomposition on the model's modes

        self.gamma = Parameter(1.e7, units='[J][m^-2][K^-1]', scale_object=scale_params,
                               description='specific heat capacity of the atmosphere', return_dimensional=True)
        self.C = Parameter(100.e0, units='[W][m^-2]', scale_object=scale_params,
                           description="constant short-wave radiation of the atmosphere", return_dimensional=True)
        self.eps = Parameter(0.76e0, input_dimensional=False,
                             description="emissivity coefficient for the grey-body atmosphere")
        self.T0 = Parameter(270.0, units='[K]', scale_object=scale_params, return_dimensional=True,
                            description="stationary solution for the 0-th order atmospheric temperature")

        self.sc = Parameter(1., input_dimensional=False, description="ratio of surface to atmosphere temperature")
        self.hlambda = Parameter(20.00, units='[W][m^-2][K^-1]', scale_object=scale_params, return_dimensional=True,
                                 description="sensible+turbulent heat exchange between ocean and atmosphere")

        self.set_params(dic)


class OceanicParams(Params):
    """Class containing the oceanic parameters

    Parameters
    ----------
    scale_params: ScaleParams
        The scale parameters object of the model.
    dic: dict(float or Parameter), optional
        A dictionary with the parameters names and values to be assigned.

    Attributes
    ----------
    gp: Parameter
        Reduced gravity in [:math:`m \, s^{-2}`].
    r: Parameter
        Friction coefficient at the bottom of the ocean in [:math:`s^{-1}`].
    h: Parameter
        Depth of the water layer of the ocean, in meters [:math:`m`].
    d: Parameter
        The strength of the ocean-atmosphere mechanical coupling in [:math:`s^{-1}`].
    """
    _name = "Oceanic"

    def __init__(self, scale_params, dic=None):

        Params.__init__(self, dic)

        self._scale_params = scale_params

        self.gp = Parameter(3.1e-2, units='[m][s^-2]', return_dimensional=True, scale_object=scale_params,
                            description='reduced gravity')
        self.r = Parameter(1.e-8, units='[s^-1]', scale_object=scale_params,
                           description="frictional coefficient at the bottom of the ocean")
        self.h = Parameter(5.e2, units='[m]', return_dimensional=True, scale_object=scale_params,
                           description="depth of the water layer of the ocean")
        self.d = Parameter(1.e-8, units='[s^-1]', scale_object=scale_params,
                           description="strength of the ocean-atmosphere mechanical coupling")

        self.set_params(dic)


class OceanicTemperatureParams(Params):
    """Class containing the oceanic temperature parameters

    Parameters
    ----------
    scale_params: ScaleParams
        The scale parameters object of the model.
    dic: dict(float or Parameter), optional
        A dictionary with the parameters names and values to be assigned.

    Attributes
    ----------
    gamma: Parameter
        Specific heat capacity of the ocean
    C: Parameter
        Constant short-wave radiation of the ocean
    T0: Parameter
        Stationary solution for the 0-th order oceanic temperature [:math:`K`].
    """

    _name = "Oceanic Temperature"

    def __init__(self, scale_params, dic=None):

        Params.__init__(self, dic)

        self._scale_params = scale_params

        self.gamma = Parameter(2.e8, units='[J][m^-2][K^-1]', scale_object=scale_params, return_dimensional=True,
                               description='specific heat capacity of the ocean')
        self.C = Parameter(350.e0, units='[W][m^-2]', scale_object=scale_params, return_dimensional=True,
                           description="constant short-wave radiation of the atmosphere")
        self.T0 = Parameter(285.0, units='[K]', scale_object=scale_params, return_dimensional=True,
                            description="stationary solution for the 0-th order atmospheric temperature")

        self.set_params(dic)


class QgParams(Params):
    """General qgs parameters container

    Parameters
    ----------
    dic: dict(float or Parameter), optional
        A dictionary with the parameters names and values to be assigned.
    scale_params: ScaleParams
        Scale parameters instance.
    atmospheric_params: AtmosphericParams
        Atmospheric parameters instance.
    atemperature_params: AtmosphericTemperatureParams
        Atmospheric temperature parameters instance.
    oceanic_params: OceanicParams
         Oceanic parameters instance.
    otemperature_params: OceanicTemperatureParams
         Oceanic temperature parameters instance.

    Attributes
    ----------
    scale_params: ScaleParams
        Scale parameters instance.
    atmospheric_params: AtmosphericParams
        Atmospheric parameters instance.
    atemperature_params: AtmosphericTemperatureParams
        Atmospheric temperature parameters instance.
    oceanic_params: OceanicParams
         Oceanic parameters instance.
    otemperature_params: OceanicTemperatureParams
         Oceanic temperature parameters instance.
    time_unit: float
        Dimensional unit of time to be used to represent the data.
    rr: Parameter
        `Gas constant`_ of `dry air`_ in [:math:`J \, kg^{-1} \, K^{-1}`].
    sb: float
        `Stefan-Boltzmann constant`_ in [:math:`J \, m^{-2} \, s^{-1} \, K^{-4}`]


    .. _Gas constant: https://en.wikipedia.org/wiki/Gas_constant
    .. _dry air: https://en.wikipedia.org/wiki/Gas_constant#Specific_gas_constant
    .. _Stefan-Boltzmann constant: https://en.wikipedia.org/wiki/Stefan%E2%80%93Boltzmann_constant
    """
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
            self.atmospheric_params = AtmosphericParams(self.scale_params, dic=dic)
        else:
            self.atmospheric_params = atmospheric_params

        # Atmospheric temperature parameters object
        if atmospheric_params is True:
            self.atemperature_params = AtmosphericTemperatureParams(self.scale_params, dic=dic)
        else:
            self.atemperature_params = atemperature_params

        if oceanic_params is True:
            self.oceanic_params = OceanicParams(self.scale_params, dic)
        else:
            self.oceanic_params = oceanic_params

        if otemperature_params is True:
            self.otemperature_params = OceanicTemperatureParams(self.scale_params, dic)
        else:
            self.otemperature_params = otemperature_params

        self._number_of_dimensions = 0
        self._number_of_atmospheric_modes = 0
        self._number_of_oceanic_modes = 0
        self._ams = None
        self._oms = None

        self._atmospheric_latex_var_string = list()
        self._atmospheric_var_string = list()
        self._oceanic_latex_var_string = list()
        self._oceanic_var_string = list()
        self.time_unit = 'days'

        # Physical constants

        self.rr = Parameter(287.058e0, return_dimensional=True, units='[J][kg^-1][K^-1]',
                            scale_object=self.scale_params, description="gas constant of dry air")
        self.sb = Parameter(5.67e-8, return_dimensional=True, units='[J][m^-2][s^-1][K^-4]',
                            scale_object=self.scale_params, description="Stefan-Boltzmann constant")

        self.set_params(dic)

    # -----------------------------------------------------------
    # Derived Quantities (Parameters)
    # -----------------------------------------------------------

    @property
    def LR(self):
        """float: Reduced Rossby deformation radius :math:`L_\\text{R} = \\sqrt{g' \\, h } / f_0` ."""
        op = self.oceanic_params
        scp = self.scale_params
        if op is not None:
            return np.sqrt(op.gp * op.h) / scp.f0
        else:
            return None

    @property
    def G(self):
        """float: The :math:`G = - L^2/L_R^2` parameter."""
        scp = self.scale_params
        if self.LR is not None:
            return -scp.L**2 / self.LR**2
        else:
            return None

    @property
    def Cpo(self):
        """float: The :math:`C\'_{{\\rm o},i} = R C_{{\\rm o},i} / (\\gamma_{\\rm o} L^2 f_0^3)` parameter."""
        otp = self.otemperature_params
        scp = self.scale_params
        if otp is not None:
            return otp.C / (otp.gamma * scp.f0) * self.rr / (scp.f0 ** 2 * scp.L ** 2)
        else:
            return None

    @property
    def Lpo(self):
        """float: The :math:`\\lambda\'_{{\\rm o}} = \\lambda/(\\gamma_{\\rm o} f_0)` parameter."""
        atp = self.atemperature_params
        otp = self.otemperature_params
        scp = self.scale_params
        if atp is not None and otp is not None:
            return atp.hlambda / (otp.gamma * scp.f0)
        else:
            return None

    @property
    def Cpa(self):
        """float: The :math:`C\'_{{\\rm a},i} = R C_{{\\rm a},i} / (2 \\gamma_{\\rm a} L^2 f_0^3)` parameter."""
        atp = self.atemperature_params
        scp = self.scale_params
        if atp is not None and atp.hd == 0:
            return atp.C / (atp.gamma * scp.f0) * self.rr / (scp.f0 ** 2 * scp.L ** 2) / 2
        else:
            return None

    @property
    def Lpa(self):
        """float: The :math:`\\lambda\'_{\\rm a} = \\lambda / (\\gamma_{\\rm a} f_0)` parameter."""
        atp = self.atemperature_params
        scp = self.scale_params
        if atp is not None and atp.hd == 0:
            return atp.hlambda / (atp.gamma * scp.f0)
        else:
            return None

    @property
    def sbpo(self):
        """float: Long wave radiation lost by ocean to the atmosphere :math:`s_{B,{\\rm o}} = 4\\,\\sigma_B \\, T_{{\\rm a},0}^3 / (\\gamma_{\\rm o} f_0)`."""
        otp = self.otemperature_params
        scp = self.scale_params
        if otp is not None:
            return 4 * self.sb * otp.T0 ** 3 / (otp.gamma * scp.f0)
        else:
            return None

    @property
    def sbpa(self):
        """float: Long wave radiation from atmosphere absorbed by ocean :math:`s_{B,{\\rm a}} = 4\\,\\epsilon_{\\rm a}\\, \\sigma_B \\, T_{{\\rm a},0}^3 / (\\gamma_{\\rm o} f_0)`."""
        atp = self.atemperature_params
        otp = self.otemperature_params
        scp = self.scale_params
        if otp is not None and atp is not None:
            return 8 * atp.eps * self.sb * atp.T0 ** 3 / (otp.gamma * scp.f0)
        else:
            return None

    @property
    def LSBpo(self):
        """float: Long wave radiation from ocean absorbed by atmosphere :math:`S_{B,{\\rm o}} = 2\\,\\epsilon_{\\rm a}\\, \\sigma_B \\, T_{{\\rm a},0}^3 / (\\gamma_{\\rm a} f_0)`."""
        atp = self.atemperature_params
        otp = self.otemperature_params
        scp = self.scale_params
        if atp is not None and otp is not None:
            return 2 * atp.eps * self.sb * otp.T0 ** 3 / (atp.gamma * scp.f0)
        else:
            return None

    @property
    def LSBpa(self):
        """float: Long wave radiation lost by atmosphere to space & ocean :math:`S_{B,{\\rm a}} = 8\\,\\epsilon_{\\rm a}\\, \\sigma_B \\, T_{{\\rm a},0}^3 / (\\gamma_{\\rm a} f_0)`."""
        atp = self.atemperature_params
        scp = self.scale_params
        if atp is not None and atp.hd == 0:
            return 8 * atp.eps * self.sb * atp.T0 ** 3 / (atp.gamma * scp.f0)
        else:
            return None

    def set_params(self, dic):
        """Set the specified parameters values.

        Parameters
        ----------
        dic: dict(float or Parameter)
            A dictionary with the parameters names and values to be assigned.
        """
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
        """Print all the parameters in the container."""
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
        """int: Total number of variables of the model."""
        return self._number_of_dimensions

    @property
    def nmod(self):
        """(int, int): Atmospheric and oceanic number of modes."""
        return [self._number_of_atmospheric_modes, self._number_of_oceanic_modes]

    @property
    def ablocks(self):
        """~numpy.ndarray(int): Spectral blocks detailing the model's atmospheric modes x- and y-wavenumber.
         Array of shape (:attr:`~QgParams.nmod` [0], 2)."""
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
        """~numpy.ndarray(int): Spectral blocks detailing the model's oceanic modes x- and y-wavenumber.
         Array of shape (:attr:`~QgParams.nmod` [1], 2)."""
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
            self.atemperature_params.hd = Parameter(0.0, input_dimensional=False)

    def set_max_atmospheric_modes(self, nxmax, nymax, auto=False):
        """Function to automatically or not configure spectrally contiguous blocks of atmospheric modes.

        Parameters
        ----------
        nxmax: int
            Maximum x-wavenumber to fill the spectral block up to.
        nymax: int
            Maximum y-wavenumber to fill the spectral block up to.
        auto: bool
            Automatically instantiate the parameters container needed to describe the atmospheric models parameters.
            Default is False.

        Examples
        --------

        >>> from params.params import QgParams
        >>> q = QgParams()
        >>> q.set_max_atmospheric_modes(2,2)
        >>> q.ablocks
        array([[1, 1],
               [1, 2],
               [2, 1],
               [2, 2]])
        """
        res = np.zeros((nxmax * nymax, 2), dtype=np.int)
        i = 0
        for nx in range(1, nxmax + 1):
            for ny in range(1, nymax+1):
                res[i, 0] = nx
                res[i, 1] = ny
                i += 1

        self.ablocks = res

        self._atmospheric_latex_var_string = list()
        self._atmospheric_var_string = list()
        for i in range(self.nmod[0]):
            self._atmospheric_latex_var_string.append(r'psi_{\rm a,' + str(i + 1) + "}")
            self._atmospheric_var_string.append(r'psi_a_' + str(i + 1))
        for i in range(self.nmod[0]):
            self._atmospheric_latex_var_string.append(r'theta_{\rm a,' + str(i + 1) + "}")
            self._atmospheric_var_string.append(r'theta_a_' + str(i + 1))

        if auto:
            if self.atemperature_params is None:
                self.atemperature_params = AtmosphericTemperatureParams(self.scale_params)
            if self.atmospheric_params is None:
                self.atmospheric_params = AtmosphericParams(self.scale_params)

    def set_max_oceanic_modes(self, nxmax, nymax, auto=True):
        """Function to automatically configure spectrally contiguous blocks of oceanic modes.

        Parameters
        ----------
        nxmax: int
            Maximum x-wavenumber to fill the spectral block up to.
        nymax: int
            Maximum y-wavenumber to fill the spectral block up to.
        auto: bool
            Automatically instantiate or not the parameters container needed to describe the oceanic models parameters.
            Default is True.

        Examples
        --------

        >>> from params.params import QgParams
        >>> q = QgParams()
        >>> q.set_max_oceanic_modes(2,4)
        >>> q.oblocks
        array([[1, 1],
               [1, 2],
               [1, 3],
               [1, 4],
               [2, 1],
               [2, 2],
               [2, 3],
               [2, 4]])
        """
        res = np.zeros((nxmax * nymax, 2), dtype=np.int)
        i = 0
        for nx in range(1, nxmax + 1):
            for ny in range(1, nymax+1):
                res[i, 0] = nx
                res[i, 1] = ny
                i += 1

        self.oblocks = res

        self._oceanic_latex_var_string = list()
        self._oceanic_var_string = list()
        for i in range(self.nmod[1]):
            self._oceanic_latex_var_string.append(r'psi_{\rm o,' + str(i + 1) + "}")
            self._oceanic_var_string.append(r'psi_o_' + str(i + 1))
        for i in range(self.nmod[1]):
            self._oceanic_latex_var_string.append(r'theta_{\rm o,' + str(i + 1) + "}")
            self._oceanic_var_string.append(r'theta_o_' + str(i + 1))

        if auto:
            if self.otemperature_params is None:
                self.otemperature_params = OceanicTemperatureParams(self.scale_params)
            if self.oceanic_params is None:
                self.oceanic_params = OceanicParams(self.scale_params)

    @property
    def var_string(self):
        """list(str): List of model's variable names."""
        l = list()
        for var in self._atmospheric_var_string:
            l.append(var)
        for var in self._oceanic_var_string:
            l.append(var)

        return l

    @property
    def latex_var_string(self):
        """list(str): List of model's variable names, ready for use in latex."""
        l = list()
        for var in self._atmospheric_latex_var_string+self._oceanic_latex_var_string:
            l.append(r'{\ '[0:-1] + var + r'}')

        return l

    @property
    def dimensional_time(self):
        """float: Return the conversion factor between the non-dimensional time and the dimensional time unit specified
        in :attr:`.time_unit`"""
        c = 24 * 3600
        if self.time_unit == 'days':
            c = 24 * 3600
        if self.time_unit == 'years':
            c = 24 * 3600 * 365
        return 1 / (self.scale_params.f0 * c)


