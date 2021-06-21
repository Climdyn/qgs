"""
    Parameter module
    ================

    This module contains the basic parameter class to hold model's parameters values.
    It allows to manipulate dimensional and nondimensional parameter easily.

    Examples
    --------

    >>> from qgs.params.params import ScaleParams
    >>> from qgs.params.parameter import Parameter
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

    Main class
    ----------
"""

import warnings


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
    return_dimensional: bool, optional
        Defined if the value returned by the parameter is dimensional or not. Default to `False`.

    Notes
    -----
    Parameter is immutable. Once instantiated, it cannot be altered. To create a new parameter, one must
    re-instantiate it.

    Warnings
    --------
    If no scale_object argument is provided, cannot transform between the dimensional and nondimentional value !

    """

    def __new__(cls, value, input_dimensional=True, units="", scale_object=None, description="",
                return_dimensional=False):

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


