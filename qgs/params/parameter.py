"""
    Parameter module
    ================

    This module contains the basic parameter class to hold model's parameters values.
    It allows to manipulate dimensional and nondimensional parameter easily.

    Examples
    --------

    >>> from qgs.params.params import ScaleParams
    >>> from qgs.params.parameter import Parameter, ArrayParameters
    >>> import numpy as np
    >>> # defining a scale object to help Parameter compute the nondimensionalization
    >>> sc = ScaleParams()
    >>> # creating a parameter initialized with a nondimensional value but returning a
    >>> # dimensional one when called
    >>> sigma = Parameter(0.2e0, input_dimensional=False, scale_object=sc,
    ...                   units='[m^2][s^-2][Pa^-2]',
    ...                   description="static stability of the atmosphere",
    ...                   return_dimensional=True)
    >>> sigma
    2.1581898457499433e-06
    >>> sigma.nondimensional_value
    0.2
    >>> sigma.return_dimensional
    True
    >>> # creating a parameter initialized with a dimensional value but returning a
    >>> # nondimensional one when called
    >>> sigma = Parameter(2.1581898457499433e-06, input_dimensional=True, scale_object=sc,
    ...                   units='[m^2][s^-2][Pa^-2]',
    ...                   description="static stability of the atmosphere",
    ...                   return_dimensional=False)
    >>> sigma
    0.2
    >>> sigma.dimensional_value
    2.1581898457499433e-06
    >>> sigma.return_dimensional
    False
    >>> # creating a parameters array initialized with a nondimensional values and returning
    >>> # nondimensional ones when called
    >>> s = ArrayParameters(np.array([[0.1,0.2],[0.3,0.4]]), input_dimensional=False, scale_object=sc, units='[s^-1]',
    ...                     description="atmosphere bottom friction coefficient")
    >>> s
    ArrayParameters([[0.1, 0.2],
                     [0.3, 0.4]], dtype=object)
    >>> # dimensional values can also be retrieved with
    >>> s.dimensional_values
    array([[1.0320000000000001e-05, 2.0640000000000002e-05],
           [3.096e-05, 4.1280000000000005e-05]], dtype=object)

    Main class
    ----------
"""

import warnings
import numpy as np


class ScalingParameter(float):
    """Class of model's dimension parameter.

    Parameters
    ----------
    value: float
        Value of the parameter.
    units: str, optional
        The units of the provided value. Used to compute the conversion between dimensional and nondimensional
        value. Should be specified by joining atoms like `'[unit^power]'`, e.g '`[m^2][s^-2][Pa^-2]'`.
        Empty by default.
    description: str, optional
        String describing the parameter.
    symbol: ~sympy.core.symbol.Symbol, optional
        A `Sympy`_ symbol to represent the parameter in symbolic expressions.
    dimensional: bool, optional
        Indicate if the value of the parameter is dimensional or not. Default to `True`.

    Notes
    -----
    Parameter is immutable. Once instantiated, it cannot be altered. To create a new parameter, one must
    re-instantiate it.

    .. _Sympy: https://www.sympy.org/
    """

    def __new__(cls, value, units="", description="", symbol=None, dimensional=False):

        f = float.__new__(cls, value)
        f._dimensional = dimensional
        f._units = units
        f._description = description
        f._symbol = symbol

        return f

    @property
    def symbol(self):
        """~sympy.core.symbol.Symbol: Returns the symbol of the parameter."""
        return self._symbol

    @property
    def dimensional(self):
        """bool: Indicate if the returned value is dimensional or not."""
        return self._dimensional

    @property
    def units(self):
        """str: The units of the dimensional value."""
        return self._units

    @property
    def description(self):
        """str: Description of the parameter."""
        return self._description


class Parameter(float):
    """Base class of model's parameter.

    Parameters
    ----------
    value: float
        Value of the parameter.
    input_dimensional: bool, optional
        Specify whether the value provided is dimensional or not. Default to `True`.
    units: str, optional
        The units of the provided value. Used to compute the conversion between dimensional and nondimensional
        value. Should be specified by joining atoms like `'[unit^power]'`, e.g '`[m^2][s^-2][Pa^-2]'`.
        Empty by default.
    scale_object: ScaleParams, optional
        A scale parameters object to compute the conversion between dimensional and nondimensional value.
        `None` by default. If `None`, cannot transform between dimensional and nondimentional value.
    description: str, optional
        String describing the parameter.
    symbol: ~sympy.core.symbol.Symbol, optional
        A `Sympy`_ symbol to represent the parameter in symbolic expressions.
    return_dimensional: bool, optional
        Defined if the value returned by the parameter is dimensional or not. Default to `False`.

    Notes
    -----
    Parameter is immutable. Once instantiated, it cannot be altered. To create a new parameter, one must
    re-instantiate it.

    Warnings
    --------
    If no scale_object argument is provided, cannot transform between the dimensional and nondimentional value !

    .. _Sympy: https://www.sympy.org/
    """

    def __new__(cls, value, input_dimensional=True, units="", scale_object=None, description="",
                symbol=None, return_dimensional=False):

        no_scale = False

        if return_dimensional:
            if input_dimensional:
                evalue = value
            else:
                if scale_object is None:
                    return_dimensional = False
                    evalue = value
                    no_scale = True
                else:
                    evalue = value / cls._conversion_factor(units, scale_object)
        else:
            if input_dimensional:
                if scale_object is None:
                    return_dimensional = True
                    evalue = value
                    no_scale = True
                else:
                    evalue = value * cls._conversion_factor(units, scale_object)
            else:
                evalue = value

        if no_scale:
            warnings.warn("Parameter configured to perform dimensional conversion " +
                          "but without specifying a ScaleParams object: Conversion disabled!")

        f = float.__new__(cls, evalue)
        f._input_dimensional = input_dimensional
        f._return_dimensional = return_dimensional
        f._units = units
        f._scale_object = scale_object
        f._description = description
        f._symbol = symbol

        return f

    @property
    def dimensional_value(self):
        """float: Returns the dimensional value."""
        if self._return_dimensional:
            return self
        else:
            return self / self._nondimensionalization

    @property
    def nondimensional_value(self):
        """float: Returns the nondimensional value."""
        if self._return_dimensional:
            return self * self._nondimensionalization
        else:
            return self

    @property
    def symbol(self):
        """~sympy.core.symbol.Symbol: Returns the symbol of the parameter."""
        return self._symbol

    @property
    def input_dimensional(self):
        """bool: Indicate if the provided value is dimensional or not."""
        return self._input_dimensional

    @property
    def return_dimensional(self):
        """bool: Indicate if the returned value is dimensional or not."""
        return self._return_dimensional

    @classmethod
    def _conversion_factor(cls, units, scale_object):
        factor = 1.

        ul = units.split('][')
        ul[0] = ul[0][1:]
        ul[-1] = ul[-1][:-1]

        for us in ul:
            up = us.split('^')
            if len(up) == 1:
                up.append("1")

            if up[0] == 'm':
                factor *= scale_object.L ** (-int(up[1]))
            elif up[0] == 's':
                factor *= scale_object.f0 ** (int(up[1]))
            elif up[0] == 'Pa':
                factor *= scale_object.deltap ** (-int(up[1]))

        return factor

    @property
    def units(self):
        """str: The units of the dimensional value."""
        return self._units

    @property
    def description(self):
        """str: Description of the parameter."""
        return self._description

    @property
    def _nondimensionalization(self):
        if self._scale_object is None:
            return 1.
        else:
            return self._conversion_factor(self._units, self._scale_object)


class ArrayParameters(np.ndarray):
    """Base class of model's array of parameters.

    Parameters
    ----------
    values: list(float) or ~numpy.ndarray(float) or list(Parameter) or ~numpy.ndarray(Parameter)
        Values of the parameter array.
    input_dimensional: bool, optional
        Specify whether the value provided is dimensional or not. Default to `True`.
    units: str, optional
        The units of the provided value. Used to compute the conversion between dimensional and nondimensional
        value. Should be specified by joining atoms like `'[unit^power]'`, e.g '`[m^2][s^-2][Pa^-2]'`.
        Empty by default.
    scale_object: ScaleParams, optional
        A scale parameters object to compute the conversion between dimensional and nondimensional value.
        `None` by default. If `None`, cannot transform between dimensional and nondimentional value.
    description: str or list(str) or array(str), optional
        String or an iterable of string, describing the parameters.
        If an iterable, should have the same length or shape as `values`.
    symbol: ~sympy.core.symbol.Symbol or list(~sympy.core.symbol.Symbol) or ~numpy.ndarray(~sympy.core.symbol.Symbol), optional
        A `Sympy`_ symbol or an iterable of, to represent the parameters in symbolic expressions.
        If an iterable, should have the same length or shape as `values`.
    return_dimensional: bool, optional
        Defined if the value returned by the parameter is dimensional or not. Default to `False`.

    Warnings
    --------
    If no scale_object argument is provided, cannot transform between the dimensional and nondimensional value !

    .. _Sympy: https://www.sympy.org/
    """

    def __new__(cls, values, input_dimensional=True, units="", scale_object=None, description="",
                symbol=None, return_dimensional=False):

        if isinstance(values, (tuple, list)):
            new_arr = np.empty(len(values), dtype=object)
            for i, val in enumerate(values):
                if isinstance(description, (tuple, list, np.ndarray)):
                    descr = description[i]
                else:
                    descr = description
                if isinstance(symbol, (tuple, list, np.ndarray)):
                    sy = symbol[i]
                else:
                    sy = symbol
                new_arr[i] = Parameter(val, input_dimensional=input_dimensional, units=units, scale_object=scale_object, description=descr,
                                       return_dimensional=return_dimensional, symbol=sy)
        else:
            new_arr = np.empty_like(values, dtype=object)
            for idx in np.ndindex(values.shape):
                if isinstance(description, np.ndarray):
                    descr = description[idx]
                else:
                    descr = description
                if isinstance(symbol, np.ndarray):
                    sy = symbol[idx]
                else:
                    sy = symbol
                new_arr[idx] = Parameter(values[idx], input_dimensional=input_dimensional, units=units, scale_object=scale_object, description=descr,
                                         return_dimensional=return_dimensional, symbol=sy)
        arr = np.asarray(new_arr).view(cls)
        arr._input_dimensional = input_dimensional
        arr._return_dimensional = return_dimensional
        arr._units = units
        arr._scale_object = scale_object

        return arr

    def __array_finalize__(self, arr):

        if arr is None:
            return

        self._input_dimensional = getattr(arr, '_input_dimensional', True)
        self._units = getattr(arr, '_units', "")
        self._return_dimensional = getattr(arr, '_return_dimensional', False)
        self._scale_object = getattr(arr, '_scale_object', None)

    @property
    def dimensional_values(self):
        """float: Returns the dimensional value."""
        if self._return_dimensional:
            return self
        else:
            return np.array(self / self._nondimensionalization)

    @property
    def nondimensional_values(self):
        """float: Returns the nondimensional value."""
        if self._return_dimensional:
            return np.array(self * self._nondimensionalization)
        else:
            return self

    @property
    def symbols(self):
        """~numpy.ndarray(~sympy.core.symbol.Symbol): Returns the symbol of the parameters in the array."""
        symbols = np.empty(self.shape, dtype=object)
        for idx in np.ndindex(self.shape):
            symbols[idx] = self[idx].symbol
        return symbols

    @property
    def input_dimensional(self):
        """bool: Indicate if the provided value is dimensional or not."""
        return self._input_dimensional

    @property
    def return_dimensional(self):
        """bool: Indicate if the returned value is dimensional or not."""
        return self._return_dimensional

    @classmethod
    def _conversion_factor(cls, units, scale_object):
        factor = 1.

        ul = units.split('][')
        ul[0] = ul[0][1:]
        ul[-1] = ul[-1][:-1]

        for us in ul:
            up = us.split('^')
            if len(up) == 1:
                up.append("1")

            if up[0] == 'm':
                factor *= scale_object.L ** (-int(up[1]))
            elif up[0] == 's':
                factor *= scale_object.f0 ** (int(up[1]))
            elif up[0] == 'Pa':
                factor *= scale_object.deltap ** (-int(up[1]))

        return factor

    @property
    def units(self):
        """str: The units of the dimensional value."""
        return self._units

    @property
    def descriptions(self):
        """~numpy.ndarray(str): Description of the parameters in the array."""
        descr = np.empty(self.shape, dtype=object)
        for idx in np.ndindex(self.shape):
            descr[idx] = self[idx].description
        return descr

    @property
    def _nondimensionalization(self):
        if self._scale_object is None:
            return 1.
        else:
            return self._conversion_factor(self._units, self._scale_object)
