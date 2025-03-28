"""
    The model's parameters module
    =============================

    This module defines the main classes containing the model configuration parameters.
    The parameters are typically specified as :class:`~.params.parameter.Parameter` objects.

    There are seven types of parameters arranged in classes:

    * :class:`ScaleParams` contains the model scale parameters. These parameters are used to scale and
      `nondimentionalize`_ the :class:`~.params.parameter.Parameter` of the other parameters classes according to
      their :attr:`~.params.parameter.Parameter.units` attribute.
    * :class:`AtmosphericParams` contains the atmospheric dynamical parameters.
    * :class:`AtmosphericTemperatureParams` containing the atmosphere's temperature and heat-exchange parameters.
    * :class:`OceanicParams` contains the oceanic dynamical parameters.
    * :class:`OceanicTemperatureParams` contains the ocean's temperature and heat-exchange parameters.
    * :class:`GroundParams` contains the ground dynamical parameters (e.g. orography).
    * :class:`GroundTemperatureParams` contains the ground's temperature and heat-exchange parameters.

    These parameters classes are regrouped into a global structure :class:`QgParams` which also contains

    * spectral modes definition of the model
    * physical constants
    * parameters derived from the ones provided by the user
    * helper functions to initialize and parameterize the model

    This global parameters structure is used by the other modules to construct the model's ordinary differential
    equations.

    Warning
    -------

    If a model's parameter is set to `None`, it is assumed to be disabled.


    ---------------------

    Description of the classes
    --------------------------

    .. _nondimentionalize: https://en.wikipedia.org/wiki/Nondimensionalization
"""

import numpy as np
import pickle
import warnings
from abc import ABC

from qgs.params.parameter import Parameter, ScalingParameter, ParametersArray
from qgs.basis.fourier import contiguous_channel_basis, contiguous_basin_basis
from qgs.basis.fourier import ChannelFourierBasis, BasinFourierBasis

from sympy import simplify, Symbol


# TODO: - store model version in a variable somewhere
#       - force or warn the user to define the aspect ratio n at parameter object instantiation


class Params(ABC):
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
                            self.__dict__[key] = Parameter(val,
                                                           input_dimensional=d['_input_dimensional'],
                                                           units=d['_units'],
                                                           description=d['_description'],
                                                           scale_object=d['_scale_object'],
                                                           symbol=d['_symbol'],
                                                           return_dimensional=d['_return_dimensional'])
                    elif isinstance(self.__dict__[key], ScalingParameter):
                        if isinstance(val, ScalingParameter):
                            self.__dict__[key] = val
                        else:
                            d = self.__dict__[key].__dict__
                            self.__dict__[key] = ScalingParameter(val,
                                                                  units=d['_units'],
                                                                  description=d['_description'],
                                                                  symbol=d['_symbol'],
                                                                  dimensional=d['_dimensional'])
                    else:
                        self.__dict__[key] = val

    def __str__(self):
        s = ""
        for key, val in zip(self.__dict__.keys(), self.__dict__.values()):
            if 'params' not in key and key[0] != '_':
                if val is None:
                    pass
                elif isinstance(val, Parameter):
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
                elif isinstance(val, ScalingParameter):
                    if val.dimensional:
                        units = val.units
                    else:
                        units = "[nondim]"
                    s += "'" + key + "': " + str(val) + "  " + units + "  (" + val.description + "),\n"
                elif isinstance(val, (np.ndarray, list, tuple)) and isinstance(val[0], Parameter):
                    for i, v in enumerate(val):
                        if v.input_dimensional:
                            units = v.units
                            efval = v.dimensional_value
                        else:
                            efval = v.nondimensional_value
                            if v.nondimensional_value == v.dimensional_value:
                                units = ""
                            else:
                                units = "[nondim]"
                        s += "'" + key + "["+str(i+1)+"]': " + str(efval) + "  " + units + "  (" + v.description + "),\n"
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

    def load_from_file(self, filename, **kwargs):
        """Function to load previously saved Params object with the method :meth:`save_to_file`.

        Parameters
        ----------
        filename: str
            The file name where the Params object was saved.
        kwargs: dict
            Keyword arguments to pass to the :mod:`pickle` module method.
        """
        f = open(filename, 'rb')
        tmp_dict = pickle.load(f, **kwargs)
        f.close()

        self.__dict__.clear()
        self.__dict__.update(tmp_dict)

    def save_to_file(self, filename, **kwargs):
        """Function to save the Params object to a file with the :mod:`pickle` module.

        Parameters
        ----------
        filename: str
            The file name where to save the Params object.
        kwargs: dict
            Keyword arguments to pass to the :mod:`pickle` module method.
        """
        f = open(filename, 'wb')
        pickle.dump(self.__dict__, f, **kwargs)
        f.close()


class ScaleParams(Params):
    """Class containing the model scales parameters.

    Parameters
    ----------
    dic: dict(float or Parameter), optional
        A dictionary with the parameters names and values to be assigned.

    Attributes
    ----------
    scale: Parameter
        The characteristic meridional space scale, :math:`L_y = \\pi \\, L`, in meters [:math:`m`].
    f0: Parameter
        Coriolis parameter, in [:math:`s^{-1}`].
    n: Parameter
        Model domain aspect ratio, :math:`n = 2 L_y/L_x` .
    rra: Parameter
        Earth radius, in meters [:math:`m`].
    phi0_npi: Parameter
        Latitude expressed in fraction of :math:`\\pi` .
    deltap: Parameter
        Difference of pressure between the center of the two atmospheric layers, in [:math:`Pa`].
    """
    _name = "Scale"

    def __init__(self, dic=None):

        Params.__init__(self, dic)

        # -----------------------------------------------------------
        # Scale parameters for the ocean and the atmosphere
        # -----------------------------------------------------------

        self.scale = ScalingParameter(5.e6, units='[m]', description="characteristic space scale (L*pi)", dimensional=True)
        self.f0 = ScalingParameter(1.032e-4, units='[s^-1]', description="Coriolis parameter at the middle of the domain",
                                   dimensional=True, symbol=Symbol('f0'))
        self.n = ScalingParameter(1.3e0, dimensional=False, description="aspect ratio (n = 2 L_y / L_x)", symbol=Symbol('n', positive=True))
        self.rra = ScalingParameter(6370.e3, units='[m]', description="earth radius", dimensional=True)
        self.phi0_npi = ScalingParameter(0.25e0, dimensional=False, description="latitude expressed in fraction of pi")
        self.deltap = ScalingParameter(5.e4, units='[Pa]', description='pressure difference between the two atmospheric layers',
                                       dimensional=True)
        self.Ha = ScalingParameter(8500., units='[m]', description="Average height of the 500 hPa pressure level at midlatitude",
                                   dimensional=True, symbol=Symbol('H_a'))
        self.set_params(dic)

    # ----------------------------------------
    # Some derived parameters (Domain, beta)
    # ----------------------------------------

    @property
    def L(self):
        """Parameter: Typical length scale :math:`L`  of the model, in meters [:math:`m`]."""
        return ScalingParameter(self.scale / np.pi, units=self.scale.units, description='Typical length scale L',
                                symbol=Symbol('L'), dimensional=True)

    @property
    def L_y(self):
        """Parameter: The meridional extent :math:`L_y = \\pi \\, L` of the model's domain, in meters [:math:`m`]."""
        return ScalingParameter(self.scale, units=self.scale.units, description='The meridional extent of the model domain',
                                dimensional=True)

    @property
    def L_x(self):
        """Parameter: The zonal extent :math:`L_x = 2 \\pi \\, L / n` of the model's domain, in meters [:math:`m`]."""
        return ScalingParameter(2 * self.scale / self.n, units=self.scale.units,
                                description='The zonal extent of the model domain',
                                dimensional=True)

    @property
    def phi0(self):
        """Parameter: The reference latitude :math:`\\phi_0` at the center of the domain, expressed in radians [:math:`rad`]."""
        return ScalingParameter(self.phi0_npi * np.pi, units='[rad]',
                                description="The reference latitude of the center of the domain",
                                dimensional=True, symbol=Symbol('phi0'))

    @property
    def beta(self):
        """Parameter: The meridional gradient of the Coriolis parameter at :math:`\\phi_0`, expressed in [:math:`m^{-1} s^{-1}`]. """
        return Parameter(self.L / self.rra * np.cos(self.phi0) / np.sin(self.phi0), input_dimensional=False,
                         units='[m^-1][s^-1]', scale_object=self,
                         description="Meridional gradient of the Coriolis parameter at phi_0", symbol=Symbol('beta'))


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
    kd: Parameter
        Atmosphere bottom friction coefficient [:math:`s^{-1}`].
    kdp: Parameter
        Atmosphere internal friction coefficient [:math:`s^{-1}`].
    sigma: Parameter
        Static stability of the atmosphere [:math:`[m^2 s^{-2} Pa^{-2}`].
    """

    _name = "Atmospheric"

    def __init__(self, scale_params, dic=None):

        Params.__init__(self, dic)

        self._scale_params = scale_params

        # Parameters for the atmosphere
        self.kd = Parameter(0.1, input_dimensional=False, scale_object=scale_params, units='[s^-1]',
                            description="atmosphere bottom friction coefficient", symbol=Symbol('k_d'))
        self.kdp = Parameter(0.01, input_dimensional=False, scale_object=scale_params, units='[s^-1]',
                             description="atmosphere internal friction coefficient", symbol=Symbol('k_p'))
        self.sigma = Parameter(0.2e0, input_dimensional=False, scale_object=scale_params, units='[m^2][s^-2][Pa^-2]',
                               description="static stability of the atmosphere", symbol=Symbol('sigma'))

        self.set_params(dic)

    @property
    def sig0(self):
        """Parameter: Static stability of the atmosphere divided by 2."""
        return Parameter(self.sigma / 2, input_dimensional=False, scale_object=self._scale_params, units='[m^2][s^-2][Pa^-2]',
                         description="0.5 * static stability of the atmosphere", symbol=self.sigma.symbol / 2)


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
    hd: None or Parameter
        Newtonian cooling coefficient.
        Newtonian cooling is disabled if `None`.
    thetas: None or ~numpy.ndarray(float)
        Coefficients of the Newtonian cooling spectral decomposition (non-dimensional).
        Newtonian cooling is disabled if `None`.
    gamma: None or Parameter
        Specific heat capacity of the atmosphere [:math:`J m^{-2} K^{-1}`].
        Heat exchange scheme is disabled if `None`.
    C: None or ~numpy.ndarray(Parameter)
        Spectral decomposition of the constant short-wave radiation of the atmosphere [:math:`W m^{-2}`].
        Heat exchange scheme is disabled if `None`.
    eps: None or Parameter
        Emissivity coefficient for the grey-body atmosphere
        Heat exchange scheme is disabled if `None`.
    T0: None or Parameter
        Stationary solution for the 0-th order atmospheric temperature [:math:`K`].
        Heat exchange scheme is disabled if `None`.
    sc: None or Parameter
        Ratio of surface to atmosphere temperature
        Heat exchange scheme is disabled if `None`.
    hlambda: None or Parameter
        Sensible + turbulent heat exchange between ocean/ground and atmosphere [:math:`W m^{-2} K^{-1}`].
        Heat exchange scheme is disabled if `None`.
    """
    _name = "Atmospheric Temperature"

    def __init__(self, scale_params, dic=None):

        Params.__init__(self, dic)

        self._scale_params = scale_params

        self.hd = Parameter(0.045, input_dimensional=False, units='[s]', scale_object=scale_params,
                            description="Newtonian cooling coefficient", symbol=Symbol('hd'))
        self.thetas = None  # Radiative equilibrium mean temperature decomposition on the model's modes

        self.gamma = None
        self.C = None
        self.eps = None
        self.T0 = None
        self.sc = None
        self.hlambda = None
        self.dynamic_T = None

        self.set_params(dic)

    def set_insolation(self, value, pos=None, dynamic_T=False):
        """Function to define the spectral decomposition of the constant short-wave radiation of the atmosphere (insolation)
        :math:`C_{{\\rm a}, i}` (:attr:`~.AtmosphericTemperatureParams.C`).

        Parameters
        ----------
        value: float, int or iterable
            Value to set. If a scalar is given, the `pos` parameter should be provided to indicate which component to set.
            If an iterable is provided, create a vector of spectral decomposition parameters corresponding to it.
        pos: int, optional
            Indicate in which component to set the `value`.
        dynamic_T: bool, optional
            Whether or not the dynamic temperature scheme is activated.
        """

        # TODO: - check for the dimensionality of the arguments

        if isinstance(value, (float, int)) and pos is not None and self.C is not None:
            offset = 1
            if self.dynamic_T or dynamic_T:
                offset = 0
            self.C[pos] = Parameter(value, units='[W][m^-2]', scale_object=self._scale_params,
                                    description="spectral component "+str(pos+offset)+" of the short-wave radiation of the atmosphere",
                                    return_dimensional=True, symbol=Symbol('C_a'+str(pos+offset)))
        elif hasattr(value, "__iter__"):
            self._create_insolation(value, dynamic_T)
        else:
            warnings.warn('A scalar value was provided, but without the `pos` argument indicating in which ' +
                          'component of the spectral decomposition to put it: Spectral decomposition unchanged !' +
                          'Please specify it or give a vector as `value`.')

    def _create_insolation(self, values, dynamic_T=False):

        if hasattr(values, "__iter__"):
            dim = len(values)
            values = list(values)
        else:
            dim = values
            values = dim * [0.]

        offset = 1
        if dynamic_T:
            offset = 0
            self.dynamic_T = True
        d = ["spectral component "+str(pos+offset)+" of the short-wave radiation of the atmosphere" for pos in range(dim)]
        sy = [Symbol('C_a'+str(pos+offset)) for pos in range(dim)]

        self.C = ParametersArray(values, units='[W][m^-2]', scale_object=self._scale_params,
                                 description=d, return_dimensional=True, symbols=sy)

    def set_thetas(self, value, pos=None):
        """Function to define the spectral decomposition of the Newtonian cooling
        :math:`\\theta^\\star` (:attr:`~.AtmosphericTemperatureParams.thetas`).

        Parameters
        ----------
        value: float, int or iterable
            Value to set. If a scalar is given, the `pos` parameter should be provided to indicate which component to set.
            If an iterable is provided, create a vector of spectral decomposition parameters corresponding to it.
        pos: int, optional
            Indicate in which component to set the `value`.
        """

        # TODO: - check for the dimensionality of the arguments

        if isinstance(value, (float, int)) and pos is not None and self.thetas is not None:
            self.thetas[pos] = Parameter(value, scale_object=self._scale_params,
                                         description="spectral components "+str(pos+1)+" of the temperature profile",
                                         return_dimensional=False, input_dimensional=False, symbol=Symbol('thetas_'+str(pos+1)))
        elif hasattr(value, "__iter__"):
            self._create_thetas(value)
        else:
            warnings.warn('A scalar value was provided, but without the `pos` argument indicating in which ' +
                          'component of the spectral decomposition to put it: Spectral decomposition unchanged !' +
                          'Please specify it or give a vector as `value`.')

    def _create_thetas(self, values):

        if hasattr(values, "__iter__"):
            dim = len(values)
            values = list(values)
        else:
            dim = values
            values = dim * [0.]

        d = ["spectral component "+str(pos+1)+" of the temperature profile" for pos in range(dim)]
        sy = [Symbol('thetas_'+str(pos+1)) for pos in range(dim)]

        self.thetas = ParametersArray(values, scale_object=self._scale_params,
                                      description=d, return_dimensional=False, input_dimensional=False, symbols=sy)


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
        Reduced gravity in [:math:`m \\, s^{-2}`].
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
                            description='reduced gravity', symbol=Symbol('g_p'))
        self.r = Parameter(1.e-8, units='[s^-1]', scale_object=scale_params,
                           description="frictional coefficient at the bottom of the ocean", symbol=Symbol('r'))
        self.h = Parameter(5.e2, units='[m]', return_dimensional=True, scale_object=scale_params,
                           description="depth of the water layer of the ocean", symbol=Symbol('h'))
        self.d = Parameter(1.e-8, units='[s^-1]', scale_object=scale_params,
                           description="strength of the ocean-atmosphere mechanical coupling", symbol=Symbol('d'))

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
    gamma: None or Parameter
        Specific heat capacity of the ocean [:math:`J m^{-2} K^{-1}`].
        Heat exchange scheme is disabled if `None`.
    C: None or ~numpy.ndarray(Parameter)
        Spectral Decomposition of the constant short-wave radiation of the ocean [:math:`W m^{-2}`].
        Heat exchange scheme is disabled if `None`.
    T0: None or Parameter
        Stationary solution for the 0-th order oceanic temperature [:math:`K`].
        Heat exchange scheme is disabled if `None`.
    """

    _name = "Oceanic Temperature"

    def __init__(self, scale_params, dic=None):

        Params.__init__(self, dic)

        self._scale_params = scale_params

        self.gamma = Parameter(2.e8, units='[J][m^-2][K^-1]', scale_object=scale_params, return_dimensional=True,
                               description='specific heat capacity of the ocean', symbol=Symbol('gamma_o'))
        self.C = None

        self.T0 = None
        self.dynamic_T = None

        self.set_params(dic)

    def set_insolation(self, value, pos=None, dynamic_T=False):
        """Function to define the spectral decomposition of the constant short-wave radiation of the ocean (insolation)
        :math:`C_{{\\rm o}, i}` (:attr:`~.OceanicTemperatureParams.C`).

        Parameters
        ----------
        value: float, int or iterable
            Value to set. If a scalar is given, the `pos` parameter should be provided to indicate which component to set.
            If an iterable is provided, create a vector of spectral decomposition parameters corresponding to it.
        pos: int, optional
            Indicate in which component to set the `value`.
        dynamic_T: bool, optional
            Whether or not the dynamic temperature scheme is activated.
        """

        if isinstance(value, (float, int)) and pos is not None and self.C is not None:
            offset = 1
            if self.dynamic_T or dynamic_T:
                offset = 0
            self.C[pos] = Parameter(value, units='[W][m^-2]', scale_object=self._scale_params,
                                    description="spectral component "+str(pos+offset)+" of the short-wave radiation of the ocean",
                                    return_dimensional=True, symbol=Symbol('C_go'+str(pos+offset)))
        elif hasattr(value, "__iter__"):
            self._create_insolation(value, dynamic_T)
        else:
            warnings.warn('A scalar value was provided, but without the `pos` argument indicating in which ' +
                          'component of the spectral decomposition to put it: Spectral decomposition unchanged !' +
                          'Please specify it or give a vector as `value`.')

    def _create_insolation(self, values, dynamic_T=False):

        if hasattr(values, "__iter__"):
            dim = len(values)
            values = list(values)
        else:
            dim = values
            values = dim * [0.]

        offset = 1
        if dynamic_T:
            offset = 0
            self.dynamic_T = True
        d = ["spectral component "+str(pos+offset)+" of the short-wave radiation of the ocean" for pos in range(dim)]
        sy = [Symbol('C_go'+str(pos+offset)) for pos in range(dim)]

        self.C = ParametersArray(values, units='[W][m^-2]', scale_object=self._scale_params,
                                 description=d, return_dimensional=True, symbols=sy)


class GroundParams(Params):
    """Class containing the ground parameters

    Parameters
    ----------
    scale_params: ScaleParams
        The scale parameters object of the model.
    dic: dict(float or Parameter), optional
        A dictionary with the parameters names and values to be assigned.

    Attributes
    ----------
    hk: None or ~numpy.ndarray(float)
        Orography spectral decomposition coefficients (non-dimensional), an array of shape (:attr:`~QgParams.nmod` [0],).
        Orography is disabled (flat) if `None`.
    orographic_basis: str
        String to select which component basis modes to use to develop the orography in series.
        Can be either 'atmospheric' or 'ground'. Default to 'atmospheric'.
    """
    _name = "Ground"

    def __init__(self, scale_params, dic=None):

        Params.__init__(self, dic)

        self._scale_params = scale_params

        self.hk = None  # spectral orography coefficients

        self.orographic_basis = "atmospheric"

        self.set_params(dic)

    def set_orography(self, value, pos=None, basis="atmospheric"):
        """Function to define the spectral decomposition of the orography profile
        :math:`h_k` (:attr:`~.GroundParams.hk`).

        Parameters
        ----------
        value: float, int or iterable
            Value to set. If a scalar is given, the `pos` parameter should be provided to indicate which component to set.
            If an iterable is provided, create a vector of spectral decomposition parameters corresponding to it.
        pos: int, optional
            Indicate in which component to set the `value`.
        basis: str, optional
            Indicate which basis should be used to decompose the orography. Can be either `atmospheric`, `oceanic` or `ground`.
            Default to `atmospheric`.
        """

        # TODO: - check for the dimensionality of the arguments
        #       - check that inner products are symbolic if basis is not 'atmospheric'

        self.orographic_basis = basis

        if isinstance(value, (float, int)) and pos is not None and self.hk is not None:
            self.hk[pos] = Parameter(value, scale_object=self._scale_params,
                                     description="spectral components "+str(pos+1)+" of the orography",
                                     return_dimensional=False, input_dimensional=False, symbol=Symbol('hk_'+str(pos+1)))
        elif hasattr(value, "__iter__"):
            self._create_orography(value)
        else:
            warnings.warn('A scalar value was provided, but without the `pos` argument indicating in which ' +
                          'component of the spectral decomposition to put it: Spectral decomposition unchanged !' +
                          'Please specify it or give a vector as `value`.')

    def _create_orography(self, values):

        if hasattr(values, "__iter__"):
            dim = len(values)
            values = list(values)
        else:
            dim = values
            values = dim * [0.]

        d = ["spectral component "+str(pos+1)+" of the orography" for pos in range(dim)]
        sy = [Symbol('hk_'+str(pos+1)) for pos in range(dim)]

        self.hk = ParametersArray(values, scale_object=self._scale_params,
                                  description=d, return_dimensional=False, input_dimensional=False, symbols=sy)


class GroundTemperatureParams(Params):
    """Class containing the ground temperature parameters

    Parameters
    ----------
    scale_params: ScaleParams
        The scale parameters object of the model.
    dic: dict(float or Parameter), optional
        A dictionary with the parameters names and values to be assigned.

    Attributes
    ----------
    gamma: None or Parameter
        Specific heat capacity of the ground [:math:`J m^{-2} K^{-1}`].
        Heat exchange scheme is disabled if `None`.
    C: None or ~numpy.ndarray(Parameter)
        Spectral decomposition of the constant short-wave radiation of the ground [:math:`W m^{-2}`].
        Heat exchange scheme is disabled if `None`.
    T0: None or Parameter
        Stationary solution for the 0-th order ground temperature [:math:`K`].
        Heat exchange scheme is disabled if `None`.
    """

    _name = "Ground Temperature"

    def __init__(self, scale_params, dic=None):

        Params.__init__(self, dic)

        self._scale_params = scale_params

        self.gamma = Parameter(2.e8, units='[J][m^-2][K^-1]', scale_object=scale_params, return_dimensional=True,
                               description='specific heat capacity of the ground', symbol=Symbol('gamma_g'))
        self.C = None

        self.T0 = None
        self.dynamic_T = None

        self.set_params(dic)

    def set_insolation(self, value, pos=None, dynamic_T=False):
        """Function to define the decomposition of the constant short-wave radiation of the ground (insolation)
        :math:`C_{{\\rm g}, i}` (:attr:`~.GroundTemperatureParams.C`).

        Parameters
        ----------
        value: float, int or iterable
            Value to set. If a scalar is given, the `pos` parameter should be provided to indicate which component to set.
            If an iterable is provided, create a vector of spectral decomposition parameters corresponding to it.
        pos: int, optional
            Indicate in which component to set the `value`.
        dynamic_T: bool, optional
            Whether or not the dynamic temperature scheme is activated.
        """

        # TODO: - check for the dimensionality of the arguments

        if isinstance(value, (float, int)) and pos is not None and self.C is not None:
            offset = 1
            if self.dynamic_T or dynamic_T:
                offset = 0
            self.C[pos] = Parameter(value, units='[W][m^-2]', scale_object=self._scale_params,
                                    description="spectral component "+str(pos+offset)+" of the short-wave radiation of the ground",
                                    return_dimensional=True, symbol=Symbol('C_go'+str(pos+offset)))
        elif hasattr(value, "__iter__"):
            self._create_insolation(value, dynamic_T)
        else:
            warnings.warn('A scalar value was provided, but without the `pos` argument indicating in which ' +
                          'component of the spectral decomposition to put it: Spectral decomposition unchanged !' +
                          'Please specify it or give a vector as `value`.')

    def _create_insolation(self, values, dynamic_T=False):

        if hasattr(values, "__iter__"):
            dim = len(values)
            values = list(values)
        else:
            dim = values
            values = dim * [0.]

        offset = 1
        if dynamic_T:
            offset = 0
            self.dynamic_T = True
        d = ["spectral component "+str(pos+offset)+" of the short-wave radiation of the ground" for pos in range(dim)]
        sy = [Symbol('C_go'+str(pos+offset)) for pos in range(dim)]

        self.C = ParametersArray(values, units='[W][m^-2]', scale_object=self._scale_params,
                                 description=d, return_dimensional=True, symbols=sy)


class QgParams(Params):
    """General qgs parameters container.

    Parameters
    ----------
    dic: dict(float or Parameter), optional
        A dictionary with the parameters names and values to be assigned.
    scale_params: None or ScaleParams, optional
        Scale parameters instance.
        If `None`, create a new ScaleParams instance. Default to None.
        Default to `None`.
    atmospheric_params: bool, None or AtmosphericParams, optional
        Atmospheric parameters instance.
        If 'True`, create a new AtmosphericParams instance.
        If `None`, atmospheric parameters are disabled.
        Default to `True`.
    atemperature_params: bool, None or AtmosphericTemperatureParams, optional
        Atmospheric temperature parameters instance.
        If 'True`, create a new AtmosphericTemperatureParams instance.
        If `None`, atmospheric temperature parameters are disabled.
        Default to `True`.
    oceanic_params: bool, None or OceanicParams, optional
        Oceanic parameters instance.
        If 'True`, create a new OceanicParams instance.
        If `None`, oceanic parameters are disabled.
        Default to `None`.
    otemperature_params: bool, None or OceanicTemperatureParams, optional
        Oceanic temperature parameters instance.
        If 'True`, create a new OceanicTemperatureParams instance.
        If `None`, oceanic temperature parameters are disabled.
        Default to `None`.
    ground_params: bool, None or GroundParams, optional
        Ground parameters instance.
        If 'True`, create a new GroundParams instance.
        If `None`, ground parameters are disabled.
        Default to `True`.
    gtemperature_params: bool, None or GroundTemperatureParams, optional
        Ground temperature parameters instance.
        If 'True`, create a new GroundTemperatureParams instance.
        If `None`, ground temperature parameters are disabled.
        Default to `None`.
    dynamic_T: bool, optional
        Whether to use a fixed or a dynamical reference temperature if the heat exchange scheme is activated.
        Default to `False`.
    T4: bool, optional
        Use or not the :math:`T^4` forcing for the evolution of the temperature field if the heat exchange is activated.
        Activate also the dynamical 0-th temperature.
        Default to `False`.

    Attributes
    ----------
    scale_params: ScaleParams
        Scale parameters instance.
    atmospheric_params: None or AtmosphericParams
        Atmospheric parameters instance.
        If `None`, atmospheric parameters are disabled.
    atemperature_params: None or AtmosphericTemperatureParams
        Atmospheric temperature parameters instance.
        If `None`, atmospheric temperature parameters are disabled.
    oceanic_params: None or OceanicParams
        Oceanic parameters instance.
        If `None`, oceanic parameters are disabled.
    ground_params: None or GroundParams
        Ground parameters instance
        If `None`, ground parameters are disabled.
    gotemperature_params: None, OceanicTemperatureParams or GroundTemperatureParams
        Ground or Oceanic temperature parameters instance.
        If `None`, ground and oceanic temperature parameters are disabled.
    time_unit: float
        Dimensional unit of time to be used to represent the data.
    rr: Parameter
        `Gas constant`_ of `dry air`_ in [:math:`J \\, kg^{-1} \\, K^{-1}`].
    sb: float
        `Stefan-Boltzmann constant`_ in [:math:`J \\, m^{-2} \\, s^{-1} \\, K^{-4}`].
    dynamic_T: bool
        Whether to use a fixed or a dynamical reference temperature if the heat exchange scheme is activated.
        The atmospheric and possibly oceanic (or ground) basis must be reset if this parameter is changed.
    T4: bool
        Use or not the :math:`T^4` forcing for the evolution of the temperature field if the heat exchange is activated.


    .. _Gas constant: https://en.wikipedia.org/wiki/Gas_constant
    .. _dry air: https://en.wikipedia.org/wiki/Gas_constant#Specific_gas_constant
    .. _Stefan-Boltzmann constant: https://en.wikipedia.org/wiki/Stefan%E2%80%93Boltzmann_constant

    """
    _name = "General"

    def __init__(self, dic=None, scale_params=None,
                 atmospheric_params=True, atemperature_params=True,
                 oceanic_params=None, otemperature_params=None,
                 ground_params=True, gtemperature_params=None,
                 dynamic_T=False, T4=False):

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

        if ground_params is True:
            self.ground_params = GroundParams(self.scale_params, dic)
        else:
            self.ground_params = ground_params

        if otemperature_params is True:
            self.gotemperature_params = OceanicTemperatureParams(self.scale_params, dic)
        else:
            self.gotemperature_params = otemperature_params

        if gtemperature_params is True:
            self.gotemperature_params = GroundTemperatureParams(self.scale_params, dic)
        else:
            self.gotemperature_params = gtemperature_params

        self._atmospheric_basis = None
        self._oceanic_basis = None
        self._ground_basis = None

        self._number_of_atmospheric_modes = 0
        self._number_of_oceanic_modes = 0
        self._number_of_ground_modes = 0
        self._ams = None
        self._oms = None
        self._gms = None

        self.dynamic_T = dynamic_T
        self.T4 = T4
        # Force dynamic temperatures if T4 tendencies are activated
        if T4:
            self.dynamic_T = T4

        self._atmospheric_latex_var_string = list()
        self._atmospheric_var_string = list()
        self._oceanic_latex_var_string = list()
        self._oceanic_var_string = list()
        self._ground_latex_var_string = list()
        self._ground_var_string = list()
        self._components_units = [r'm$^2$s$^{-1}$', r'K', r'm$^2$s$^{-1}$', r'K']
        self.time_unit = 'days'

        # Physical constants
        self.rr = Parameter(287.058e0, return_dimensional=True, units='[J][kg^-1][K^-1]',
                            scale_object=self.scale_params, description="gas constant of dry air", symbol=Symbol('R'))
        self.sb = Parameter(5.67e-8, return_dimensional=True, units='[J][m^-2][s^-1][K^-4]',
                            scale_object=self.scale_params, description="Stefan-Boltzmann constant", symbol=Symbol('sigma_b'))

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
            try:
                return (op.gp * op.h) ** 0.5 / scp.f0
            except:
                return None
        else:
            return None

    @property
    def G(self):
        """float: The :math:`G = - L^2/L_R^2` parameter."""
        scp = self.scale_params
        if self.LR is not None:
            try:
                return -scp.L**2 / self.LR**2
            except:
                return None
        else:
            return None

    @property
    def Cpgo(self):
        """float: The :math:`C\'_{{\\rm g/\\rm o},i} = R C_{{\\rm g/\\rm o},i} / (\\gamma_{\\rm g/\\rm o} L^2 f_0^3)` parameter."""
        gotp = self.gotemperature_params
        scp = self.scale_params
        if gotp is not None:
            try:
                return gotp.C / (gotp.gamma * scp.f0) * self.rr / (scp.f0 ** 2 * scp.L ** 2)
            except:
                return None
        else:
            return None

    @property
    def Lpgo(self):
        """float: The :math:`\\lambda\'_{{\\rm g/\\rm o}} = \\lambda/(\\gamma_{\\rm g/\\rm o} f_0)` parameter."""
        atp = self.atemperature_params
        gotp = self.gotemperature_params
        scp = self.scale_params
        if atp is not None and gotp is not None:
            try:
                return atp.hlambda / (gotp.gamma * scp.f0)
            except:
                return None
        else:
            return None

    @property
    def Cpa(self):
        """float: The :math:`C\'_{{\\rm a},i} = R C_{{\\rm a},i} / (2 \\gamma_{\\rm a} L^2 f_0^3)` parameter."""
        atp = self.atemperature_params
        scp = self.scale_params
        if atp is not None:
            try:
                return atp.C / (atp.gamma * scp.f0) * self.rr / (scp.f0 ** 2 * scp.L ** 2) / 2
            except:
                return None
        else:
            return None

    @property
    def Lpa(self):
        """float: The :math:`\\lambda\'_{\\rm a} = \\lambda / (\\gamma_{\\rm a} f_0)` parameter."""
        atp = self.atemperature_params
        scp = self.scale_params
        if atp is not None:
            try:
                return atp.hlambda / (atp.gamma * scp.f0)
            except:
                return None
        else:
            return None

    @property
    def sbpgo(self):
        """float: Long wave radiation lost by ground/ocean to the atmosphere :math:`s_{B,{\\rm g/\\rm o}} = 4\\,\\sigma_B \\, T_{{\\rm a},0}^3 / (\\gamma_{\\rm g/\\rm o} f_0)` in the linearized temperature model equations."""
        gotp = self.gotemperature_params
        scp = self.scale_params
        if gotp is not None and not self.dynamic_T:
            try:
                return 4 * self.sb * gotp.T0 ** 3 / (gotp.gamma * scp.f0)
            except:
                return None
        else:
            return None

    @property
    def sbpa(self):
        """float: Long wave radiation from atmosphere absorbed by ground/ocean :math:`s_{B,{\\rm a}} = 4\\,\\epsilon_{\\rm a}\\, \\sigma_B \\, T_{{\\rm a},0}^3 / (\\gamma_{\\rm g/\\rm o} f_0)` in the linearized temperature model equations."""
        atp = self.atemperature_params
        gotp = self.gotemperature_params
        scp = self.scale_params
        if gotp is not None and atp is not None and not self.dynamic_T:
            try:
                return 8 * atp.eps * self.sb * atp.T0 ** 3 / (gotp.gamma * scp.f0)
            except:
                return None
        else:
            return None

    @property
    def LSBpgo(self):
        """float: Long wave radiation from ground/ocean absorbed by atmosphere :math:`S_{B,{\\rm g/\\rm o}} = 2\\,\\epsilon_{\\rm a}\\, \\sigma_B \\, T_{{\\rm a},0}^3 / (\\gamma_{\\rm a} f_0)` in the linearized temperature model equations."""
        atp = self.atemperature_params
        gotp = self.gotemperature_params
        scp = self.scale_params
        if atp is not None and gotp is not None and not self.dynamic_T:
            try:
                return 2 * atp.eps * self.sb * gotp.T0 ** 3 / (atp.gamma * scp.f0)
            except:
                return None
        else:
            return None

    @property
    def LSBpa(self):
        """float: Long wave radiation lost by atmosphere to space & ground/ocean :math:`S_{B,{\\rm a}} = 8\\,\\epsilon_{\\rm a}\\, \\sigma_B \\, T_{{\\rm a},0}^3 / (\\gamma_{\\rm a} f_0)` in the linearized temperature model equations."""
        atp = self.atemperature_params
        scp = self.scale_params
        if atp is not None and not self.dynamic_T:
            try:
                return 8 * atp.eps * self.sb * atp.T0 ** 3 / (atp.gamma * scp.f0)
            except:
                return None
        else:
            return None

    @property
    def T4sbpgo(self):
        """float: Long wave radiation lost by ground/ocean to the atmosphere :math:`s_{B,{\\rm g/\\rm o}} = \\sigma_B \\, L^6 \\, f_0^5 / (\\gamma_{\\rm g/\\rm o} R^3)` in the :math:`T^4` model equations."""
        gotp = self.gotemperature_params
        scp = self.scale_params
        if gotp is not None:
            try:
                return self.sb * scp.L ** 6 * scp.f0 ** 5 / (gotp.gamma * self.rr ** 3)
            except:
                return None
        else:
            return None

    @property
    def T4sbpa(self):
        """float: Long wave radiation from atmosphere absorbed by ground/ocean :math:`s_{B,{\\rm a}} = 16 \\,\\epsilon_{\\rm a}\\, \\sigma_B \\, L^6 \\, f_0^5 / (\\gamma_{\\rm g/\\rm o} R^3)` in the :math:`T^4` model equations."""
        atp = self.atemperature_params
        gotp = self.gotemperature_params
        scp = self.scale_params
        if gotp is not None and atp is not None:
            try:
                return 16 * atp.eps * self.sb * scp.L ** 6 * scp.f0 ** 5 / (gotp.gamma * self.rr ** 3)
            except:
                return None
        else:
            return None

    @property
    def T4LSBpgo(self):
        """float: Long wave radiation from ground/ocean absorbed by atmosphere :math:`S_{B,{\\rm g/\\rm o}} = \\frac{1}{2} \\, \\epsilon_{\\rm a}\\, \\sigma_B \\, L^6 \\, f_0^5 / (\\gamma_{\\rm a} R^3)` in the :math:`T^4` model equations."""
        atp = self.atemperature_params
        scp = self.scale_params
        if atp is not None:
            try:
                return 0.5 * atp.eps * self.sb * scp.L ** 6 * scp.f0 ** 5 / (atp.gamma * self.rr ** 3)
            except:
                return None
        else:
            return None

    @property
    def T4LSBpa(self):
        """float: Long wave radiation lost by atmosphere to space & ground/ocean :math:`S_{B,{\\rm a}} = 16 \\,\\epsilon_{\\rm a}\\, \\sigma_B \\,  L^6 \\, f_0^5 / (\\gamma_{\\rm a} R^3)` in the :math:`T^4` model equations."""
        atp = self.atemperature_params
        scp = self.scale_params
        if atp is not None:
            try:
                return 16 * atp.eps * self.sb * scp.L ** 6 * scp.f0 ** 5 / (atp.gamma * self.rr ** 3)
            except:
                return None
        else:
            return None

    # The following properties might be refactored if the unit system of the model get more widespread across modules.
    @property
    def streamfunction_scaling(self):
        """float: Dimensional scaling of the streamfunction fields."""
        return self.scale_params.L**2 * self.scale_params.f0

    @property
    def temperature_scaling(self):
        """float: Dimensional scaling of the temperature fields."""
        return self.streamfunction_scaling * self.scale_params.f0 / self.rr

    @property
    def geopotential_scaling(self):
        """float: Dimensional scaling of the geopotential height."""
        return self.scale_params.f0 / 9.81

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
                            self.__dict__[key] = Parameter(val,
                                                           input_dimensional=d['_input_dimensional'],
                                                           units=d['_units'],
                                                           description=d['_description'],
                                                           scale_object=d['_scale_object'],
                                                           symbol=d['_symbol'],
                                                           return_dimensional=d['_return_dimensional'])
                    else:
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

            if 'ground_params' in self.__dict__.keys():
                if self.ground_params is not None:
                    self.ground_params.set_params(dic)

            if 'otemperature_params' in self.__dict__.keys():
                if self.gotemperature_params is not None:
                    self.gotemperature_params.set_params(dic)

            if 'gtemperature_params' in self.__dict__.keys():
                if self.gotemperature_params is not None:
                    self.gotemperature_params.set_params(dic)

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

        if 'ground_params' in self.__dict__.keys():
            if self.ground_params is not None:
                s += self.ground_params._list_params()+"\n"

        if 'gotemperature_params' in self.__dict__.keys():
            if self.gotemperature_params is not None:
                s += self.gotemperature_params._list_params() + "\n"

        print("Qgs v1.0.0 parameters summary")
        print("=============================\n")
        print(s)

    @property
    def ndim(self):
        """int: Total number of variables of the model."""
        return self.variables_range[-1]

    @property
    def nmod(self):
        """(int, int): Atmospheric and ground/oceanic number of modes."""
        if self._number_of_oceanic_modes != 0:
            return [self._number_of_atmospheric_modes, self._number_of_oceanic_modes]
        else:
            return [self._number_of_atmospheric_modes, self._number_of_ground_modes]

    @property
    def var_string(self):
        """list(str): List of model's variable names."""
        ls = list()
        for var in self._atmospheric_var_string+self._oceanic_var_string+self._ground_var_string:
            ls.append(var)

        return ls

    @property
    def latex_var_string(self):
        """list(str): List of model's variable names, ready for use in latex."""
        ls = list()
        for var in self._atmospheric_latex_var_string+self._oceanic_latex_var_string+self._ground_latex_var_string:
            ls.append(r'{\ '[0:-1] + var + r'}')

        return ls

    @property
    def variables_range(self):
        """list(int): List of the variables indices upper bound per component."""
        natm = self.nmod[0]
        ngoc = self.nmod[1]
        vr = list()
        vr.append(natm)
        vr.append(vr[-1] + natm)
        if self.dynamic_T:
            vr[-1] += 1
        if ngoc > 0:
            vr.append(vr[-1] + ngoc)
            if self._oceanic_basis is not None:
                vr.append(vr[-1] + ngoc)
            if self.dynamic_T:
                vr[-1] += 1
        return vr

    @property
    def number_of_variables(self):
        """list(int): Number of variables per component."""
        vr = self.variables_range
        return [vr[0]] + [vr[i] - vr[i-1] for i in range(1, len(vr))]

    @property
    def latex_components_units(self):
        """list(str): The units of every model's components variables, as a list of latex strings."""
        return self._components_units

    def get_variable_units(self, i):
        """Return the units of a model's variable as a string containing latex symbols.

        Parameters
        ----------
        i: int
            The number of the variable.

        Returns
        -------
        str:
            The string with the units of the variable.
        """
        if i >= self.ndim:
            warnings.warn("Variable " + str(i) + " doesn't exist, cannot return its units.")
            return None
        else:
            if i < self.variables_range[0]:
                return self._components_units[0]
            if self.variables_range[0] <= i < self.variables_range[1]:
                return self._components_units[1]
            if self.oceanic_basis is not None:
                if self.variables_range[1] <= i < self.variables_range[2]:
                    return self._components_units[2]
                if self.variables_range[2] <= i < self.variables_range[3]:
                    return self._components_units[3]
            if self.ground_basis is not None:
                if self.variables_range[1] <= i < self.variables_range[2]:
                    return self._components_units[3]

    @property
    def dimensional_time(self):
        """float: Return the conversion factor between the non-dimensional time and the dimensional time unit specified
        in :attr:`.time_unit`"""
        c = 24 * 3600
        if self.time_unit == 'hours':
            c = 3600
        if self.time_unit == 'days':
            c = 24 * 3600
        if self.time_unit == 'years':
            c = 24 * 3600 * 365
        return 1 / (self.scale_params.f0 * c)

    def _parameter_values(self, obj=None):
        """Function produces a list of the values in the parameters class, or any class passed"""

        subs = list()
        iter_vals = obj.__dict__.values() if obj is not None else self.__dict__.values()
        for val in iter_vals:
            if isinstance(val, Parameter):
                subs.append(val)

            if isinstance(val, ScalingParameter):
                subs.append(val)

            if isinstance(val, ParametersArray):
                for v in val:
                    if v.symbol != 0:
                        subs.append(v)
        return subs

    @property
    def _all_items(self):
        """
        Function to return a list of all values in the parameter class,
        along with their current numerical values.

        Returns
        -------
        list
        """

        subs = self._parameter_values()

        for _, obj in self.__dict__.items():
            if issubclass(obj.__class__, Params):
                for v in self._parameter_values(obj):
                    subs.append(v)

        # Manually add properties from scaling class
        subs.append(self.scale_params.L)
        subs.append(self.scale_params.beta)

        return subs

    # -------------------------------------------------------------------
    # Config setters to be used with symbolic inner products
    # -------------------------------------------------------------------

    @property
    def atmospheric_basis(self):
        """Basis: The atmospheric basis of functions used to project the PDEs onto."""
        return self._atmospheric_basis

    @atmospheric_basis.setter
    def atmospheric_basis(self, basis):
        self._ams = None
        self._oms = None
        self._gms = None

        self._atmospheric_basis = basis
        self._number_of_atmospheric_modes = len(basis.functions)
        if self.dynamic_T:
            self._atmospheric_basis.functions.insert(0, simplify("1"))

        if self.ground_params is not None and self.ground_params.orographic_basis == "atmospheric":
            self.ground_params.set_orography(self._number_of_atmospheric_modes * [0.e0])

        if self.atemperature_params is not None:
            self.atemperature_params.set_thetas(self._number_of_atmospheric_modes * [0.e0])

    @property
    def oceanic_basis(self):
        """Basis: The oceanic basis of functions used to project the PDEs onto."""
        return self._oceanic_basis

    @oceanic_basis.setter
    def oceanic_basis(self, basis):
        self._ams = None
        self._oms = None
        self._gms = None

        self._oceanic_basis = basis
        self._number_of_ground_modes = 0
        self._number_of_oceanic_modes = len(basis)
        if self.dynamic_T:
            self._oceanic_basis.functions.insert(0, simplify("1"))

        if self.atemperature_params is not None:
            # disable the Newtonian cooling
            self.atemperature_params.thetas = None
            self.atemperature_params.hd = None

            self.atemperature_params.gamma = Parameter(1.e7, units='[J][m^-2][K^-1]', scale_object=self.scale_params,
                                                       description='specific heat capacity of the atmosphere',
                                                       return_dimensional=True, symbol=Symbol('gamma_a'))
            if self.dynamic_T:
                self.atemperature_params.set_insolation((self.nmod[0] + 1) * [0.e0], None, True)
                self.atemperature_params.set_insolation(100.0, 0, True)
                self.atemperature_params.set_insolation(100.0, 1, True)
            else:
                self.atemperature_params.set_insolation(self.nmod[0] * [0.e0])
                self.atemperature_params.set_insolation(100.0, 0)
                self.atemperature_params.T0 = Parameter(270.0, units='[K]', scale_object=self.scale_params,
                                                        return_dimensional=True,
                                                        description="stationary solution for the 0-th order atmospheric temperature",
                                                        symbol=Symbol('T_a0'))
            self.atemperature_params.eps = Parameter(0.76e0, input_dimensional=False,
                                                     description="emissivity coefficient for the grey-body atmosphere",
                                                     symbol=Symbol('epsilon'))
            self.atemperature_params.sc = Parameter(1., input_dimensional=False,
                                                    description="ratio of surface to atmosphere temperature",
                                                    symbol=Symbol('sc'))
            self.atemperature_params.hlambda = Parameter(20.00, units='[W][m^-2][K^-1]', scale_object=self.scale_params,
                                                         return_dimensional=True,
                                                         description="sensible+turbulent heat exchange between ocean/ground and atmosphere",
                                                         symbol=Symbol('lambda'))

        if self.gotemperature_params is not None:
            if self.dynamic_T:
                self.gotemperature_params.set_insolation((self.nmod[0] + 1) * [0.e0], None, True)
                self.gotemperature_params.set_insolation(350.0, 0, True)
                self.gotemperature_params.set_insolation(350.0, 1, True)
            else:
                self.gotemperature_params.set_insolation(self.nmod[0] * [0.e0])
                self.gotemperature_params.set_insolation(350.0, 0)
                self.gotemperature_params.T0 = Parameter(285.0, units='[K]', scale_object=self.scale_params, return_dimensional=True,
                                                         description="stationary solution for the 0-th order oceanic temperature",
                                                         symbol=Symbol('T_go0'))
            # if setting an ocean, then disable the orography
            if self.ground_params is not None:
                self.ground_params.hk = None

    @property
    def ground_basis(self):
        """Basis: The ground basis of functions used to project the PDEs onto."""
        return self._ground_basis

    @ground_basis.setter
    def ground_basis(self, basis):
        self._ams = None
        self._oms = None
        self._gms = None

        if basis[0] == 1 or basis[0] == Symbol("1"):
            del basis[0]
        self._ground_basis = basis
        self._number_of_ground_modes = len(basis)
        self._number_of_oceanic_modes = 0

        if self.dynamic_T:
            self._ground_basis.functions.insert(0, simplify("1"))
        if self.atemperature_params is not None:
            # disable the Newtonian cooling
            self.atemperature_params.thetas = None
            self.atemperature_params.hd = None

            self.atemperature_params.gamma = Parameter(1.e7, units='[J][m^-2][K^-1]', scale_object=self.scale_params,
                                                       description='specific heat capacity of the atmosphere',
                                                       return_dimensional=True, symbol=Symbol('gamma_a'))
            if self.dynamic_T:
                self.atemperature_params.set_insolation((self.nmod[0] + 1) * [0.e0], None, True)
                self.atemperature_params.set_insolation(100.0, 0, True)
                self.atemperature_params.set_insolation(100.0, 1, True)
            else:
                self.atemperature_params.set_insolation(self.nmod[0] * [0.e0])
                self.atemperature_params.set_insolation(100.0, 0)
                self.atemperature_params.T0 = Parameter(270.0, units='[K]', scale_object=self.scale_params,
                                                        return_dimensional=True,
                                                        description="stationary solution for the 0-th order atmospheric temperature",
                                                        symbol=Symbol('T_a0'))
            self.atemperature_params.eps = Parameter(0.76e0, input_dimensional=False,
                                                     description="emissivity coefficient for the grey-body atmosphere",
                                                     symbol=Symbol('epsilon'))
            self.atemperature_params.sc = Parameter(1., input_dimensional=False,
                                                    description="ratio of surface to atmosphere temperature",
                                                    symbol=Symbol('sc'))
            self.atemperature_params.hlambda = Parameter(20.00, units='[W][m^-2][K^-1]', scale_object=self.scale_params,
                                                         return_dimensional=True,
                                                         description="sensible+turbulent heat exchange between ocean/ground and atmosphere",
                                                         symbol=Symbol('lambda'))

        if self.gotemperature_params is not None:
            # if orography is disabled, enable it!
            if self.ground_params is not None:
                if self.ground_params.hk is None:
                    if self.ground_params.orographic_basis == 'atmospheric':
                        self.ground_params.set_orography(self._number_of_atmospheric_modes * [0.e0])
                    else:
                        self.ground_params.set_orography(self._number_of_ground_modes * [0.e0])
                    self.ground_params.set_orography(0.1, 1)
            if self.dynamic_T:
                self.gotemperature_params.set_insolation((self.nmod[0] + 1) * [0.e0], None, True)
                self.gotemperature_params.set_insolation(350.0, 0, True)
                self.gotemperature_params.set_insolation(350.0, 1, True)
            else:
                self.gotemperature_params.set_insolation(self.nmod[0] * [0.e0])
                self.gotemperature_params.set_insolation(350.0, 0)
                self.gotemperature_params.T0 = Parameter(285.0, units='[K]', scale_object=self.scale_params, return_dimensional=True,
                                                         description="stationary solution for the 0-th order oceanic temperature",
                                                         symbol=Symbol('T_go0'))

    def set_atmospheric_modes(self, basis, auto=False):
        """Function to configure the atmospheric modes (basis functions) used to project the PDEs onto.

        Parameters
        ----------
        basis: Basis
            Basis object containing the definition of the atmospheric modes.
        auto: bool, optional
            Automatically instantiate the parameters container needed to describe the atmospheric models parameters.
            Default is False.

        Examples
        --------

        >>> from qgs.params.params import QgParams
        >>> from qgs.basis.fourier import contiguous_channel_basis
        >>> q = QgParams()
        >>> atm_basis = contiguous_channel_basis(2, 2, 1.5)
        >>> q.set_atmospheric_modes(atm_basis)
        >>> q.atmospheric_basis
        [sqrt(2)*cos(y), 2*sin(y)*cos(n*x), 2*sin(y)*sin(n*x), sqrt(2)*cos(2*y), 2*sin(2*y)*cos(n*x), 2*sin(2*y)*sin(n*x), 2*sin(y)*cos(2*n*x), 2*sin(y)*sin(2*n*x), 2*sin(2*y)*cos(2*n*x), 2*sin(2*y)*sin(2*n*x)]

        """

        if auto:
            if self.atemperature_params is None:
                self.atemperature_params = AtmosphericTemperatureParams(self.scale_params)
            if self.atmospheric_params is None:
                self.atmospheric_params = AtmosphericParams(self.scale_params)

        self.atmospheric_basis = basis

        self._atmospheric_latex_var_string = list()
        self._atmospheric_var_string = list()
        for i in range(1, self.nmod[0] + 1):
            self._atmospheric_latex_var_string.append(r'psi_{{\rm a},' + str(i) + "}")
            self._atmospheric_var_string.append(r'psi_a_' + str(i))
        if self.dynamic_T:
            self._atmospheric_latex_var_string.append(r', T_{{\rm a},0}')
            self._atmospheric_var_string.append(r'T_a_0')
        for i in range(1, self.nmod[0] + 1):
            self._atmospheric_latex_var_string.append(r'theta_{{\rm a},' + str(i) + "}")
            self._atmospheric_var_string.append(r'theta_a_' + str(i))

    def set_oceanic_modes(self, basis, auto=True):
        """Function to configure the oceanic modes (basis functions) used to project the PDEs onto.

        Parameters
        ----------
        basis: Basis
            Basis object containing the definition of the oceanic modes.
        auto: bool, optional
            Automatically instantiate or not the parameters container needed to describe the oceanic models parameters.
            Default is True.

        Examples
        --------

        >>> from qgs.params.params import QgParams
        >>> from qgs.basis.fourier import contiguous_channel_basis, contiguous_basin_basis
        >>> q = QgParams()
        >>> atm_basis = contiguous_channel_basis(2, 2, 1.5)
        >>> oc_basis = contiguous_basin_basis(2, 4, 1.5)
        >>> q.set_atmospheric_modes(atm_basis)
        >>> q.set_oceanic_modes(oc_basis)
        >>> q.oceanic_basis
        [2*sin(y)*sin(0.5*n*x), 2*sin(2*y)*sin(0.5*n*x), 2*sin(3*y)*sin(0.5*n*x), 2*sin(4*y)*sin(0.5*n*x), 2*sin(y)*sin(1.0*n*x), 2*sin(2*y)*sin(1.0*n*x), 2*sin(3*y)*sin(1.0*n*x), 2*sin(4*y)*sin(1.0*n*x)]

        """
        if self._atmospheric_basis is None:  # Presently, the ocean can not yet be set independently of an atmosphere.
            print('Atmosphere modes not set up. Add an atmosphere before adding an ocean!')
            print('Oceanic setup aborted.')
            return

        if auto:
            if self.gotemperature_params is None or isinstance(self.gotemperature_params, GroundTemperatureParams):
                self.gotemperature_params = OceanicTemperatureParams(self.scale_params)
            if self.oceanic_params is None:
                self.oceanic_params = OceanicParams(self.scale_params)

            self.ground_params = None
            self._ground_basis = None

        self.oceanic_basis = basis

        self._oceanic_latex_var_string = list()
        self._oceanic_var_string = list()
        self._ground_latex_var_string = list()
        self._ground_var_string = list()
        for i in range(1, self.nmod[1] + 1):
            self._oceanic_latex_var_string.append(r'psi_{\rm o,' + str(i) + "}")
            self._oceanic_var_string.append(r'psi_o_' + str(i))
        if self.dynamic_T:
            self._oceanic_latex_var_string.append(r', T_{{\rm o},0}')
            self._oceanic_var_string.append(r'T_o_0')
        for i in range(1, self.nmod[1] + 1):
            self._oceanic_latex_var_string.append(r'delta T_{{\rm o},' + str(i) + "}")
            self._oceanic_var_string.append(r'delta_T_o_' + str(i))

    def set_ground_modes(self, basis=None, auto=True):
        """Function to configure the ground modes (basis functions) used to project the PDEs onto.

        Parameters
        ----------
        basis: None or Basis, optional
            Basis object containing the definition of the ground modes. If `None`, use the basis of the atmosphere.
            Default to `None`.
        auto: bool, optional
            Automatically instantiate or not the parameters container needed to describe the ground models parameters.
            Default is True.

        Examples
        --------

        >>> from qgs.params.params import QgParams
        >>> from qgs.basis.fourier import contiguous_channel_basis, contiguous_basin_basis
        >>> q = QgParams()
        >>> atm_basis = contiguous_channel_basis(2, 2, 1.5)
        >>> q.set_atmospheric_modes(atm_basis)
        >>> q.set_ground_modes()
        >>> q.ground_basis
        [sqrt(2)*cos(y), 2*sin(y)*cos(n*x), 2*sin(y)*sin(n*x), sqrt(2)*cos(2*y), 2*sin(2*y)*cos(n*x), 2*sin(2*y)*sin(n*x), 2*sin(y)*cos(2*n*x), 2*sin(y)*sin(2*n*x), 2*sin(2*y)*cos(2*n*x), 2*sin(2*y)*sin(2*n*x)]
        """
        if self._atmospheric_basis is None:  # Presently, the ground can not yet be set independently of an atmosphere.
            print('Atmosphere modes not set up. Add an atmosphere before adding the ground!')
            print('Ground setup aborted.')
            return

        if auto:
            if self.gotemperature_params is None or isinstance(self.gotemperature_params, OceanicTemperatureParams):
                self.gotemperature_params = GroundTemperatureParams(self.scale_params)
            if self.ground_params is None:
                self.ground_params = GroundParams(self.scale_params)

            self.oceanic_params = None
            self._oceanic_basis = None

        if basis is not None:
            self.ground_basis = basis
        else:
            self.ground_basis = self._atmospheric_basis

        self._oceanic_var_string = list()
        self._oceanic_latex_var_string = list()
        self._ground_latex_var_string = list()
        self._ground_var_string = list()
        if self.dynamic_T:
            self._oceanic_latex_var_string.append(r', T_{{\rm g},0}')
            self._oceanic_var_string.append(r'T_g_0')
        for i in range(1, self.nmod[1] + 1):
            self._ground_latex_var_string.append(r'delta T_{\rm g,' + str(i) + "}")
            self._ground_var_string.append(r'delta_T_g_' + str(i))

    # -------------------------------------------------------------------
    # Specific basis setters
    # -------------------------------------------------------------------

    def set_atmospheric_channel_fourier_modes(self, nxmax, nymax, auto=False, mode='analytic'):
        """Function to configure and set the basis for contiguous spectral blocks of atmospheric modes on a channel.

        Parameters
        ----------
        nxmax: int
            Maximum x-wavenumber to fill the spectral block up to.
        nymax: int
            Maximum :math:`y`-wavenumber to fill the spectral block up to.
        auto: bool, optional
            Automatically instantiate the parameters container needed to describe the atmospheric models parameters.
            Default is `False`.
        mode: str, optional
            Mode to set the inner products: Either `analytic` or `symbolic`:
            `analytic` for inner products computed with formula or `symbolic` using `Sympy`_.
            Default to `analytic`.

        Examples
        --------

        >>> from qgs.params.params import QgParams
        >>> q = QgParams()
        >>> q.set_atmospheric_channel_fourier_modes(2, 2)
        >>> q.ablocks
        array([[1, 1],
               [1, 2],
               [2, 1],
               [2, 2]])

        .. _Sympy: https://www.sympy.org/

        """

        if mode == 'symbolic':
            basis = contiguous_channel_basis(nxmax, nymax, self.scale_params.n)
            self.set_atmospheric_modes(basis, auto)
        else:
            self._set_atmospheric_analytic_fourier_modes(nxmax, nymax, auto)

    def set_oceanic_basin_fourier_modes(self, nxmax, nymax, auto=True, mode='analytic'):
        """Function to configure and set the basis for contiguous spectral blocks of oceanic modes on a closed basin.

        Parameters
        ----------
        nxmax: int
            Maximum x-wavenumber to fill the spectral block up to.
        nymax: int
            Maximum :math:`y`-wavenumber to fill the spectral block up to.
        auto: bool, optional
            Automatically instantiate the parameters container needed to describe the atmospheric models parameters.
            Default is `True`.
        mode: str, optional
            Mode to set the inner products: Either `analytic` or `symbolic`.
            `analytic` for inner products computed with formula or `symbolic` using `Sympy`_.
            Default to `analytic`.

        Examples
        --------

        >>> from qgs.params.params import QgParams
        >>> q = QgParams()
        >>> q.set_atmospheric_channel_fourier_modes(2, 2)
        >>> q.set_oceanic_basin_fourier_modes(2, 4)
        >>> q.oblocks
        array([[1, 1],
               [1, 2],
               [1, 3],
               [1, 4],
               [2, 1],
               [2, 2],
               [2, 3],
               [2, 4]])

        .. _Sympy: https://www.sympy.org/
        """

        if mode == 'symbolic':
            basis = contiguous_basin_basis(nxmax, nymax, self.scale_params.n)
            self.set_oceanic_modes(basis, auto)
        else:
            self._set_oceanic_analytic_fourier_modes(nxmax, nymax, auto)

    def set_ground_channel_fourier_modes(self, nxmax=None, nymax=None, auto=True, mode='analytic'):
        """Function to configure and set the basis for contiguous spectral blocks of ground modes on a channel.

        Parameters
        ----------
        nxmax: int, optional
            Maximum x-wavenumber to fill the spectral block up to. Default to `None`.
        nymax: int, optional
            Maximum :math:`y`-wavenumber to fill the spectral block up to. Default to `None`.
        auto: bool, optional
            Automatically instantiate the parameters container needed to describe the atmospheric models parameters.
            Default is `True`.
        mode: str, optional
            Mode to set the inner products: Either `analytic` or `symbolic`.
            `analytic` for inner products computed with formula or `symbolic` using `Sympy`_.
            Default to `analytic`.

        Notes
        -----
        If both `nxmax` and `nymax` are `None`, default to the atmospheric basis configuration if available.

        Examples
        --------

        >>> from qgs.params.params import QgParams
        >>> q = QgParams()
        >>> q.set_atmospheric_channel_fourier_modes(2,4)
        >>> q.set_ground_channel_fourier_modes()
        >>> q.gblocks
        array([[1, 1],
               [1, 2],
               [1, 3],
               [1, 4],
               [2, 1],
               [2, 2],
               [2, 3],
               [2, 4]])

        .. _Sympy: https://www.sympy.org/
        """

        if mode == "symbolic":
            if nxmax is not None and nymax is not None:
                basis = contiguous_channel_basis(nxmax, nymax, self.scale_params.n)
            else:
                basis = None
            self.set_ground_modes(basis, auto)
        else:
            self._set_ground_analytic_fourier_modes(nxmax, nymax, auto)

    # -------------------------------------------------------------------
    # Model configs setter to be used with analytic inner products
    # -------------------------------------------------------------------

    @property
    def ablocks(self):
        """~numpy.ndarray(int): Spectral blocks detailing the model's atmospheric modes :math:`x`- and :math:`y`-wavenumber.
         Array of shape (:attr:`~QgParams.nmod` [0], 2)."""
        return self._ams

    @ablocks.setter
    def ablocks(self, value):
        self._ams = value
        basis = ChannelFourierBasis(self._ams, self.scale_params.n)
        self._atmospheric_basis = basis

        namod = 0
        for i in range(self.ablocks.shape[0]):
            if self.ablocks[i, 0] == 1:
                namod += 3
            else:
                namod += 2

        self._number_of_atmospheric_modes = namod
        if self.ground_params is not None:
            self.ground_params.orographic_basis = 'atmospheric'
            self.ground_params.set_orography(namod * [0.e0])
            self.ground_params.set_orography(0.1, 1)
        if self.atemperature_params is not None:
            self.atemperature_params.set_thetas(namod * [0.e0])
            self.atemperature_params.set_thetas(0.1, 0)

    @property
    def oblocks(self):
        """~numpy.ndarray(int): Spectral blocks detailing the model's oceanic modes :math:`x`-and :math:`y`-wavenumber.
         Array of shape (:attr:`~QgParams.nmod` [1], 2)."""
        return self._oms

    @oblocks.setter
    def oblocks(self, value):
        self._oms = value
        self._gms = None
        basis = BasinFourierBasis(self._oms, self.scale_params.n)
        self._oceanic_basis = basis
        self._ground_basis = None

        if self.atemperature_params is not None:
            # disable the Newtonian cooling
            self.atemperature_params.thetas = None  # np.zeros(self.nmod[0])
            self.atemperature_params.hd = None  # Parameter(0.0, input_dimensional=False)

            self.atemperature_params.gamma = Parameter(1.e7, units='[J][m^-2][K^-1]', scale_object=self.scale_params,
                                                       description='specific heat capacity of the atmosphere',
                                                       return_dimensional=True,
                                                       symbol=Symbol('gamma_a'))
            self.atemperature_params.set_insolation(self.nmod[0] * [0.e0])
            self.atemperature_params.set_insolation(100.0, 0)
            self.atemperature_params.eps = Parameter(0.76e0, input_dimensional=False,
                                                     description="emissivity coefficient for the grey-body atmosphere",
                                                     symbol=Symbol('epsilon'))
            self.atemperature_params.T0 = Parameter(270.0, units='[K]', scale_object=self.scale_params,
                                                    return_dimensional=True,
                                                    description="stationary solution for the 0-th order atmospheric temperature",
                                                    symbol=Symbol('T_a0'))
            self.atemperature_params.sc = Parameter(1., input_dimensional=False,
                                                    description="ratio of surface to atmosphere temperature",
                                                    symbol=Symbol('sc'))
            self.atemperature_params.hlambda = Parameter(20.00, units='[W][m^-2][K^-1]', scale_object=self.scale_params,
                                                         return_dimensional=True,
                                                         description="sensible+turbulent heat exchange between ocean/ground and atmosphere",
                                                         symbol=Symbol('lambda'))

        if self.gotemperature_params is not None:
            self._number_of_ground_modes = 0
            self._number_of_oceanic_modes = self.oblocks.shape[0]
            self.gotemperature_params.set_insolation(self.nmod[0] * [0.e0])
            self.gotemperature_params.set_insolation(350.0, 0)
            self.gotemperature_params.T0 = Parameter(285.0, units='[K]', scale_object=self.scale_params, return_dimensional=True,
                                                     description="stationary solution for the 0-th order oceanic temperature",
                                                     symbol=Symbol('T_go0'))
            # if setting an ocean, then disable the orography
            if self.ground_params is not None:
                self.ground_params.hk = None

    @property
    def gblocks(self):
        """~numpy.ndarray(int): Spectral blocks detailing the model's ground modes :math:`x`-and :math:`y`-wavenumber.
         Array of shape (:attr:`~QgParams.nmod` [1], 2)."""
        return self._gms

    @gblocks.setter
    def gblocks(self, value):
        self._oms = None
        self._gms = value
        basis = ChannelFourierBasis(self._gms, self.scale_params.n)
        self._oceanic_basis = None
        self._ground_basis = basis

        if self.atemperature_params is not None:
            # disable the Newtonian cooling
            self.atemperature_params.thetas = None  # np.zeros(self.nmod[0])
            self.atemperature_params.hd = None  # Parameter(0.0, input_dimensional=False)

            self.atemperature_params.gamma = Parameter(1.e7, units='[J][m^-2][K^-1]',
                                                       scale_object=self.scale_params,
                                                       description='specific heat capacity of the atmosphere',
                                                       return_dimensional=True,
                                                       symbol=Symbol('gamma_a'))
            self.atemperature_params.set_insolation(self.nmod[0] * [0.e0])
            self.atemperature_params.set_insolation(100.0, 0)
            self.atemperature_params.eps = Parameter(0.76e0, input_dimensional=False,
                                                     description="emissivity coefficient for the grey-body atmosphere",
                                                     symbol=Symbol('epsilon'))
            self.atemperature_params.T0 = Parameter(270.0, units='[K]', scale_object=self.scale_params,
                                                    return_dimensional=True,
                                                    description="stationary solution for the 0-th order atmospheric temperature",
                                                    symbol=Symbol('T_a0'))
            self.atemperature_params.sc = Parameter(1., input_dimensional=False,
                                                    description="ratio of surface to atmosphere temperature",
                                                    symbol=Symbol('sc'))
            self.atemperature_params.hlambda = Parameter(20.00, units='[W][m^-2][K^-1]',
                                                         scale_object=self.scale_params,
                                                         return_dimensional=True,
                                                         description="sensible+turbulent heat exchange between ocean/ground and atmosphere",
                                                         symbol=Symbol('lambda'))

        if self.gotemperature_params is not None:
            gmod = 0
            for i in range(self.gblocks.shape[0]):
                if self.ablocks[i, 0] == 1:
                    gmod += 3
                else:
                    gmod += 2
            self._number_of_ground_modes = gmod
            self._number_of_oceanic_modes = 0
            # if orography is disabled, enable it!
            if self.ground_params is not None:
                self.ground_params.orographic_basis = 'atmospheric'
                if self.ground_params.hk is None:
                    self.ground_params.set_orography(self.nmod[0] * [0.e0])
                    self.ground_params.set_orography(0.1, 1)
            self.gotemperature_params.set_insolation(self.nmod[0] * [0.e0])
            self.gotemperature_params.set_insolation(350.0, 0)
            self.gotemperature_params.T0 = Parameter(285.0, units='[K]', scale_object=self.scale_params, return_dimensional=True,
                                                     description="stationary solution for the 0-th order oceanic temperature",
                                                     symbol=Symbol('T_go0'))

    def _set_atmospheric_analytic_fourier_modes(self, nxmax, nymax, auto=False):

        res = np.zeros((nxmax * nymax, 2), dtype=int)
        i = 0
        for nx in range(1, nxmax + 1):
            for ny in range(1, nymax+1):
                res[i, 0] = nx
                res[i, 1] = ny
                i += 1

        if auto:
            if self.atemperature_params is None:
                self.atemperature_params = AtmosphericTemperatureParams(self.scale_params)
            if self.atmospheric_params is None:
                self.atmospheric_params = AtmosphericParams(self.scale_params)

        self.ablocks = res

        self._atmospheric_latex_var_string = list()
        self._atmospheric_var_string = list()
        for i in range(self.nmod[0]):
            self._atmospheric_latex_var_string.append(r'psi_{\rm a,' + str(i + 1) + "}")
            self._atmospheric_var_string.append(r'psi_a_' + str(i + 1))
        for i in range(self.nmod[0]):
            self._atmospheric_latex_var_string.append(r'theta_{\rm a,' + str(i + 1) + "}")
            self._atmospheric_var_string.append(r'theta_a_' + str(i + 1))

    def _set_oceanic_analytic_fourier_modes(self, nxmax, nymax, auto=True):

        if self._ams is None:
            print('Atmosphere modes not set up. Add an atmosphere before adding an ocean!')
            print('Oceanic setup aborted.')
            return
        res = np.zeros((nxmax * nymax, 2), dtype=int)
        i = 0
        for nx in range(1, nxmax + 1):
            for ny in range(1, nymax+1):
                res[i, 0] = nx
                res[i, 1] = ny
                i += 1

        if auto:
            if self.gotemperature_params is None or isinstance(self.gotemperature_params, GroundTemperatureParams):
                self.gotemperature_params = OceanicTemperatureParams(self.scale_params)
            if self.oceanic_params is None:
                self.oceanic_params = OceanicParams(self.scale_params)

            self.ground_params = None

        self.oblocks = res

        self._oceanic_latex_var_string = list()
        self._oceanic_var_string = list()
        self._ground_latex_var_string = list()
        self._ground_var_string = list()
        for i in range(self.nmod[1]):
            self._oceanic_latex_var_string.append(r'psi_{\rm o,' + str(i + 1) + "}")
            self._oceanic_var_string.append(r'psi_o_' + str(i + 1))
        for i in range(self.nmod[1]):
            self._oceanic_latex_var_string.append(r'delta T_{\rm o,' + str(i + 1) + "}")
            self._oceanic_var_string.append(r'delta_T_o_' + str(i + 1))

    def _set_ground_analytic_fourier_modes(self, nxmax=None, nymax=None, auto=True):

        if self._ams is None:
            print('Atmosphere modes not set up. Add an atmosphere before adding the ground!')
            print('Ground setup aborted.')
            return

        if nxmax is None or nymax is None:
            res = self._ams.copy()
        else:
            res = np.zeros((nxmax * nymax, 2), dtype=int)
            i = 0
            for nx in range(1, nxmax + 1):
                for ny in range(1, nymax+1):
                    res[i, 0] = nx
                    res[i, 1] = ny
                    i += 1

        if auto:
            if self.gotemperature_params is None or isinstance(self.gotemperature_params, OceanicTemperatureParams):
                self.gotemperature_params = GroundTemperatureParams(self.scale_params)
            if self.ground_params is None:
                self.ground_params = GroundParams(self.scale_params)

            self.oceanic_params = None

        self.gblocks = res

        self._oceanic_var_string = list()
        self._oceanic_latex_var_string = list()
        self._ground_latex_var_string = list()
        self._ground_var_string = list()
        for i in range(self.nmod[1]):
            self._ground_latex_var_string.append(r'delta T_{\rm g,' + str(i + 1) + "}")
            self._ground_var_string.append(r'delta_T_g_' + str(i + 1))
