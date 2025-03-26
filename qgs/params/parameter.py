"""
    Parameter module
    ================

    This module contains the basic parameter class to hold model's parameters values.
    It allows to manipulate dimensional and nondimensional parameter easily.

    Examples
    --------

    >>> from qgs.params.params import ScaleParams
    >>> from qgs.params.parameter import Parameter, ParametersArray
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
    >>> s = ParametersArray(np.array([[0.1,0.2],[0.3,0.4]]), input_dimensional=False, scale_object=sc, units='[s^-1]',
    ...                     description="atmosphere bottom friction coefficient")
    >>> s
    ArrayParameters([[0.1, 0.2],
                     [0.3, 0.4]], dtype=object)
    >>> # dimensional values can also be retrieved with
    >>> s.dimensional_values
    array([[1.0320000000000001e-05, 2.0640000000000002e-05],
           [3.096e-05, 4.1280000000000005e-05]], dtype=object)
    >>> # you can also ask for the dimensional value of one particular value of the array
    >>> s[0,0]
    0.1
    >>> s[0,0].dimensional_value
    1.0320000000000001e-05

    Main class
    ----------
"""

import warnings
import numpy as np
from fractions import Fraction


# TODO: Automatize warnings and errors

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
    dimensional: bool, optional
        Indicate if the value of the parameter is dimensional or not. Default to `True`.
    symbol: ~sympy.core.symbol.Symbol, optional
        A `Sympy`_ symbol to represent the parameter in symbolic expressions.
    symbolic_expression: ~sympy.core.expr.Expr, optional
        A `Sympy`_ expression to represent a relationship to other parameters.

    Notes
    -----
    Parameter is immutable. Once instantiated, it cannot be altered. To create a new parameter, one must
    re-instantiate it.

    .. _Sympy: https://www.sympy.org/
    """

    def __new__(cls, value, units="", description="", dimensional=False, symbol=None, symbolic_expression=None):

        f = float.__new__(cls, value)
        f._dimensional = dimensional
        f._units = units
        f._description = description
        f._symbol = symbol
        f._symbolic_expression = symbolic_expression

        return f

    @property
    def symbol(self):
        """~sympy.core.symbol.Symbol: Returns the symbol of the parameter."""
        return self._symbol

    @property
    def symbolic_expression(self):
        """~sympy.core.expr.Expr: Returns the symbolic expression of the parameter."""
        if self._symbolic_expression is None and self._symbol is not None:
            return self._symbol
        else:
            return self._symbolic_expression

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

    def __add__(self, other):

        res = float(self) + other
        if isinstance(other, (Parameter, ScalingParameter)):
            if self.units != other.units:
                raise ArithmeticError("ScalingParameter class: Impossible to add two parameters with different units.")
            if self.symbolic_expression is None:
                if other.symbolic_expression is None:
                    if self.symbol is not None and other.symbol is not None:
                        expr = self.symbol + other.symbol
                    else:
                        expr = None
                    descr = self.description + " + " + other.description
                else:
                    if self.symbol is not None:
                        expr = self.symbol + (other.symbolic_expression)
                        descr = self.description + " + (" + other.description + ")"
                    else:
                        expr = None
                        descr = self.description + " + " + other.description
            else:
                if other.symbolic_expression is None:
                    if other.symbol is not None:
                        expr = (self.symbolic_expression) + other.symbol
                        descr = "(" + self.description + ") + " + other.description
                    else:
                        expr = None
                        descr = self.description + " + " + other.description
                else:
                    expr = (self.symbolic_expression) + (other.symbolic_expression)
                    descr = "(" + self.description + ") + (" + other.description + ")"

            if isinstance(other, Parameter):
                return Parameter(res, input_dimensional=other.input_dimensional,
                                 return_dimensional=other.return_dimensional, scale_object=other._scale_object,
                                 description=descr, units=self.units, symbol=None, symbolic_expression=expr)

            else:
                return ScalingParameter(res, description=descr, units=self.units, symbol=None, symbolic_expression=expr)
        else:
            try:
                if self.symbol is not None:
                    expr = self.symbol + other
                    descr = self.description + " + " + str(other)
                elif self.symbolic_expression is not None:
                    expr = (self.symbolic_expression) + other
                    descr = "(" + self.description + ") + " + str(other)
                else:
                    expr = None
                    descr = self.description + " + " + str(other)
                return ScalingParameter(res, description=descr, units=self.units, symbol=None, symbolic_expression=expr)
            except:
                return res

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):

        res = float(self) - other
        if isinstance(other, (Parameter, ScalingParameter)):
            if self.units != other.units:
                raise ArithmeticError("ScalingParameter class: Impossible to subtract two parameters with different units.")
            if self.symbolic_expression is None:
                if other.symbolic_expression is None:
                    if self.symbol is not None and other.symbol is not None:
                        expr = self.symbol - other.symbol
                    else:
                        expr = None
                    descr = self.description + " - " + other.description
                else:
                    if self.symbol is not None:
                        expr = self.symbol - (other.symbolic_expression)
                        descr = self.description + " - (" + other.description + ")"
                    else:
                        expr = None
                        descr = self.description + " - " + other.description
            else:
                if other.symbolic_expression is None:
                    if other.symbol is not None:
                        expr = (self.symbolic_expression) - other.symbol
                        descr = "(" + self.description + ") - " + other.description
                    else:
                        expr = None
                        descr = self.description + " - " + other.description
                else:
                    expr = (self.symbolic_expression) - (other.symbolic_expression)
                    descr = "(" + self.description + ") - (" + other.description + ")"

            if isinstance(other, Parameter):
                return Parameter(res, input_dimensional=other.input_dimensional,
                                 return_dimensional=other.return_dimensional, scale_object=other._scale_object,
                                 description=descr, units=self.units, symbol=None, symbolic_expression=expr)

            else:
                return ScalingParameter(res, description=descr, units=self.units, symbol=None, symbolic_expression=expr)
        else:
            try:
                if self.symbol is not None:
                    expr = self.symbol - other
                    descr = self.description + " - " + str(other)
                elif self.symbolic_expression is not None:
                    expr = (self.symbolic_expression) - other
                    descr = "(" + self.description + ") - " + str(other)
                else:
                    expr = None
                    descr = self.description + " - " + str(other)
                return ScalingParameter(res, description=descr, units=self.units, symbol=None, symbolic_expression=expr)
            except:
                return res

    def __rsub__(self, other):
        try:
            res = other - float(self)
            if self.symbol is not None:
                expr = other - self.symbol
                descr = str(other) + " - " + self.description
            elif self.symbolic_expression is not None:
                expr = other - (self.symbolic_expression)
                descr = str(other) + " - (" + self.description + ")"
            else:
                expr = None
                descr = str(other) + " - " + self.description
            return ScalingParameter(res, description=descr, units=self.units, symbol=None, symbolic_expression=expr)
        except:
            return res

    def __mul__(self, other):

        res = float(self) * other
        if isinstance(other, (Parameter, ScalingParameter)):
            units = _combine_units(self.units, other.units, '+')

            if self.symbolic_expression is None:
                if other.symbolic_expression is None:
                    if self.symbol is not None and other.symbol is not None:
                        expr = self.symbol * other.symbol
                    else:
                        expr = None
                    descr = self.description + " * " + other.description
                else:
                    if self.symbol is not None:
                        expr = self.symbol * (other.symbolic_expression)
                        descr = self.description + " * (" + other.description + ")"
                    else:
                        expr = None
                        descr = self.description + " * " + other.description
            else:
                if other.symbolic_expression is None:
                    if other.symbol is not None:
                        expr = (self.symbolic_expression) * other.symbol
                        descr = "(" + self.description + ") * " + other.description
                    else:
                        expr = None
                        descr = self.description + " * " + other.description
                else:
                    expr = (self.symbolic_expression) * (other.symbolic_expression)
                    descr = "(" + self.description + ") * (" + other.description + ")"

            if isinstance(other, Parameter):
                return Parameter(res, input_dimensional=other.input_dimensional, return_dimensional=other.return_dimensional,
                                 scale_object=other._scale_object, description=descr,
                                 units=units, symbol=None, symbolic_expression=expr)

            else:
                return ScalingParameter(res, description=descr, units=units, symbol=None, symbolic_expression=expr)
        else:
            try:
                if self.symbol is not None:
                    expr = self.symbol * other
                    descr = self.description + " * " + str(other)
                elif self.symbolic_expression is not None:
                    expr = (self.symbolic_expression) * other
                    descr = "(" + self.description + ") * " + str(other)
                else:
                    expr = None
                    descr = self.description + " * " + str(other)
                return ScalingParameter(res, description=descr, units=self.units, symbol=None, symbolic_expression=expr)
            except:
                return res

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):

        res = float(self) / other
        if isinstance(other, (ScalingParameter, Parameter)):
            units = _combine_units(self.units, other.units, '-')
            if self.symbolic_expression is None:
                if other.symbolic_expression is None:
                    if self.symbol is not None and other.symbol is not None:
                        expr = self.symbol / other.symbol
                    else:
                        expr = None
                    descr = self.description + " / " + other.description
                else:
                    if self.symbol is not None:
                        expr = self.symbol / (other.symbolic_expression)
                        descr = self.description + " / (" + other.description + ")"
                    else:
                        expr = None
                        descr = self.description + " / " + other.description
            else:
                if other.symbolic_expression is None:
                    if other.symbol is not None:
                        expr = (self.symbolic_expression) / other.symbol
                        descr = "(" + self.description + ") / " + other.description
                    else:
                        expr = None
                        descr = self.description + " / " + other.description
                else:
                    expr = (self.symbolic_expression) / (other.symbolic_expression)
                    descr = "(" + self.description + ") / (" + other.description + ")"

            if isinstance(other, Parameter):
                return Parameter(res, input_dimensional=other.input_dimensional, return_dimensional=other.return_dimensional,
                                 scale_object=other._scale_object, description=descr,
                                 units=units, symbol=None, symbolic_expression=expr)
            else:
                return ScalingParameter(res, description=descr, units=units, symbol=None, symbolic_expression=expr)
        else:
            try:
                if self.symbol is not None:
                    expr = self.symbol / other
                    descr = self.description + " / " + str(other)
                elif self.symbolic_expression is not None:
                    expr = (self.symbolic_expression) / other
                    descr = "(" + self.description + ") / " + str(other)
                else:
                    expr = None
                    descr = self.description + " / " + str(other)
                return ScalingParameter(res, description=descr, units=self.units, symbol=None, symbolic_expression=expr)
            except:
                return res

    def __rtruediv__(self, other):
        res = other / float(self)
        try:
            if self.symbol is not None:
                expr = other / self.symbol
                descr = str(other) + " / " + self.description
            elif self.symbolic_expression is not None:
                expr = other / (self.symbolic_expression)
                descr = str(other) + " / (" + self.description + ")"
            else:
                expr = None
                descr = str(other) + " / " + self.description
            return ScalingParameter(res, description=descr, units=self.units, symbol=None, symbolic_expression=expr)
        except:
            return res

    def __pow__(self, power, modulo=None):

        if modulo is not None:
            raise NotImplemented('ScalingParameter class: Modular exponentiation not implemented')

        res = float(self) ** power
        if int(power) == power:

            ul = self.units.split('][')
            ul[0] = ul[0][1:]
            ul[-1] = ul[-1][:-1]

            usl = list()
            for us in ul:
                up = us.split('^')
                if len(up) == 1:
                    up.append("1")

                usl.append(tuple(up))

            units_elements = list()
            for us in usl:
                units_elements.append(list((us[0], str(int(us[1]) * power))))

            units = list()
            for us in units_elements:
                if us is not None:
                    if int(us[1]) != 1:
                        units.append("[" + us[0] + "^" + us[1] + "]")
                    else:
                        units.append("[" + us[0] + "]")
            units = "".join(units)

            if self.symbolic_expression is not None:
                expr = (self.symbolic_expression) ** power
                descr = "(" + self.description + ") to the power "+str(power)
            elif self.symbol is not None:
                expr = self.symbol ** power
                descr = self.description + " to the power "+str(power)
            else:
                expr = None
                descr = self.description + " to the power "+str(power)

        else:
            power_fraction = Fraction(power)
            ul = self.units.split('][')
            ul[0] = ul[0][1:]
            ul[-1] = ul[-1][:-1]

            usl = list()
            for us in ul:
                up = us.split('^')
                if len(up) == 1:
                    up.append("1")

                usl.append(tuple(up))

            units_elements = list()
            for us in usl:
                new_power = int(us[1]) * power_fraction.numerator / power_fraction.denominator
                if int(new_power) == new_power:
                    units_elements.append(list((us[0], str(new_power))))
                else:
                    raise ArithmeticError("ScalingParameter class: Only support integer exponent in units")

            units = list()
            for us in units_elements:
                if us is not None:
                    if int(us[1]) != 1:
                        units.append("[" + us[0] + "^" + us[1] + "]")
                    else:
                        units.append("[" + us[0] + "]")
            units = "".join(units)
            if self.symbolic_expression is not None:
                expr = (self.symbolic_expression) ** power
                descr = "(" + self.description + ") to the power " + str(power)
            elif self.symbol is not None:
                expr = self.symbol ** power
                descr = self.description + " to the power " + str(power)
            else:
                expr = None
                descr = self.description + " to the power " + str(power)

        return ScalingParameter(res, description=descr, units=units, symbol=None, symbolic_expression=expr)


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
    symbolic_expression: ~sympy.core.expr.Expr, optional
        A `Sympy`_ expression to represent a relationship to other parameters.
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
                symbol=None, return_dimensional=False, symbolic_expression=None):

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
                    try:
                        evalue = value * cls._conversion_factor(units, scale_object)
                    except:
                        print(description)
                        print(symbol)
                        print(units)
                        print(cls._conversion_factor(units, scale_object))
                        print(scale_object)
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
        f._symbolic_expression = symbolic_expression

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
    def symbolic_expression(self):
        """~sympy.core.expr.Expr: Returns the symbolic expression of the parameter."""
        if self._symbolic_expression is None and self._symbol is not None:
            return self._symbol
        else:
            return self._symbolic_expression
    
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

    def __add__(self, other):

        res = float(self) + other
        if isinstance(other, (Parameter, ScalingParameter)):
            if isinstance(other, Parameter) and self.return_dimensional != other.return_dimensional:
                raise ArithmeticError("Parameter class: Impossible to subtract a dimensional parameter with a non-dimensional one.")
            if self.units != other.units:
                raise ArithmeticError("Parameter class: Impossible to add two parameters with different units.")

            if self.symbolic_expression is None:
                if other.symbolic_expression is None:
                    if self.symbol is not None and other.symbol is not None:
                        expr = self.symbol + other.symbol
                    else:
                        expr = None
                    descr = self.description + " + " + other.description
                else:
                    if self.symbol is not None:
                        expr = self.symbol + (other.symbolic_expression)
                        descr = self.description + " + (" + other.description + ")"
                    else:
                        expr = None
                        descr = self.description + " + " + other.description
            else:
                if other.symbolic_expression is None:
                    if other.symbol is not None:
                        expr = (self.symbolic_expression) + other.symbol
                        descr = "(" + self.description + ") + " + other.description
                    else:
                        expr = None
                        descr = self.description + " + " + other.description
                else:
                    expr = (self.symbolic_expression) + (other.symbolic_expression)
                    descr = "(" + self.description + ") + (" + other.description + ")"

            return Parameter(res, input_dimensional=self.return_dimensional,
                             return_dimensional=self.return_dimensional, scale_object=self._scale_object,
                             description=descr, units=self.units, symbol=None, symbolic_expression=expr)
        else:
            try:
                if self.symbol is not None:
                    expr = self.symbol + other
                    descr = self.description + " + " + str(other)
                elif self.symbolic_expression is not None:
                    expr = (self.symbolic_expression) + other
                    descr = "(" + self.description + ") + " + str(other)
                else:
                    expr = None
                    descr = self.description + " + " + str(other)
                return Parameter(res, input_dimensional=self.return_dimensional,
                                 return_dimensional=self.return_dimensional, scale_object=self._scale_object,
                                 description=descr, units=self.units, symbol=None, symbolic_expression=expr)
            except:
                return res

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):

        res = float(self) - other
        if isinstance(other, (Parameter, ScalingParameter)):
            if isinstance(other, Parameter) and self.return_dimensional != other.return_dimensional:
                raise ArithmeticError("Parameter class: Impossible to subtract a dimensional parameter with a non-dimensional one.")
            if self.units != other.units:
                raise ArithmeticError("Parameter class: Impossible to subtract two parameters with different units.")
            if self.symbolic_expression is None:
                if other.symbolic_expression is None:
                    if self.symbol is not None and other.symbol is not None:
                        expr = self.symbol - other.symbol
                    else:
                        expr = None
                    descr = self.description + " - " + other.description
                else:
                    if self.symbol is not None:
                        expr = self.symbol - (other.symbolic_expression)
                        descr = self.description + " - (" + other.description + ")"
                    else:
                        expr = None
                        descr = self.description + " - " + other.description
            else:
                if other.symbolic_expression is None:
                    if other.symbol is not None:
                        expr = (self.symbolic_expression) - other.symbol
                        descr = "(" + self.description + ") - " + other.description
                    else:
                        expr = None
                        descr = self.description + " - " + other.description
                else:
                    expr = (self.symbolic_expression) - (other.symbolic_expression)
                    descr = "(" + self.description + ") - (" + other.description + ")"

            return Parameter(res, input_dimensional=self.return_dimensional,
                             return_dimensional=self.return_dimensional, scale_object=self._scale_object,
                             description=descr, units=self.units, symbol=None, symbolic_expression=expr)
        else:
            try:
                if self.symbol is not None:
                    expr = self.symbol - other
                    descr = self.description + " - " + str(other)
                elif self.symbolic_expression is not None:
                    expr = (self.symbolic_expression) - other
                    descr = "(" + self.description + ") - " + str(other)
                else:
                    expr = None
                    descr = self.description + " - " + str(other)
                return Parameter(res, input_dimensional=self.return_dimensional,
                                 return_dimensional=self.return_dimensional, scale_object=self._scale_object,
                                 description=descr, units=self.units, symbol=None, symbolic_expression=expr)
            except:
                return res

    def __rsub__(self, other):
        res = other - float(self)
        try:
            if self.symbol is not None:
                expr = other - self.symbol
                descr = str(other) + " - " + self.description
            elif self.symbolic_expression is not None:
                expr = other - (self.symbolic_expression)
                descr = str(other) + " - (" + self.description + ")"
            else:
                expr = None
                descr = str(other) + " - " + self.description
            return Parameter(res, input_dimensional=self.return_dimensional,
                             return_dimensional=self.return_dimensional, scale_object=self._scale_object,
                             description=descr, units=self.units, symbol=None, symbolic_expression=expr)
        except:
            return res

    def __mul__(self, other):

        res = float(self) * other
        if isinstance(other, (Parameter, ScalingParameter)):
            if hasattr(other, "units"):
                units = _combine_units(self.units, other.units, '+')
            else:
                units = ""

            if self.symbolic_expression is None:
                if other.symbolic_expression is None:
                    if self.symbol is not None and other.symbol is not None:
                        expr = self.symbol * other.symbol
                    else:
                        expr = None
                    descr = self.description + " * " + other.description
                else:
                    if self.symbol is not None:
                        expr = self.symbol * (other.symbolic_expression)
                        descr = self.description + " * (" + other.description + ")"
                    else:
                        expr = None
                        descr = self.description + " * " + other.description
            else:
                if other.symbolic_expression is None:
                    if other.symbol is not None:
                        expr = (self.symbolic_expression) * other.symbol
                        descr = "(" + self.description + ") * " + other.description
                    else:
                        expr = None
                        descr = self.description + " * " + other.description
                else:
                    expr = (self.symbolic_expression) * (other.symbolic_expression)
                    descr = "(" + self.description + ") * (" + other.description + ")"

            return Parameter(res, input_dimensional=self.return_dimensional, return_dimensional=self.return_dimensional,
                             scale_object=self._scale_object, description=descr, units=units, symbol=None,
                             symbolic_expression=expr)
        else:
            try:
                if self.symbol is not None:
                    expr = self.symbol * other
                    descr = self.description + " * " + str(other)
                elif self.symbolic_expression is not None:
                    expr = (self.symbolic_expression) * other
                    descr = "(" + self.description + ") * " + str(other)
                else:
                    expr = None
                    descr = self.description + " * " + str(other)
                return Parameter(res, input_dimensional=self.return_dimensional, return_dimensional=self.return_dimensional,
                                 scale_object=self._scale_object, description=descr, units=self.units, symbol=None,
                                 symbolic_expression=expr)
            except:
                return res

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):

        res = float(self) / other
        if isinstance(other, (ScalingParameter, Parameter)):
            units = _combine_units(self.units, other.units, '-')
            if self.symbolic_expression is None:
                if other.symbolic_expression is None:
                    if self.symbol is not None and other.symbol is not None:
                        expr = self.symbol / other.symbol
                    else:
                        expr = None
                    descr = self.description + " / " + other.description
                else:
                    if self.symbol is not None:
                        expr = self.symbol / (other.symbolic_expression)
                        descr = self.description + " / (" + other.description + ")"
                    else:
                        expr = None
                        descr = self.description + " / " + other.description
            else:
                if other.symbolic_expression is None:
                    if other.symbol is not None:
                        expr = (self.symbolic_expression) / other.symbol
                        descr = "(" + self.description + ") / " + other.description
                    else:
                        expr = None
                        descr = self.description + " / " + other.description
                else:
                    expr = (self.symbolic_expression) / (other.symbolic_expression)
                    descr = "(" + self.description + ") / (" + other.description + ")"

            return Parameter(res, input_dimensional=self.return_dimensional, return_dimensional=self.return_dimensional,
                             scale_object=self._scale_object, description=descr, units=units, symbol=None,
                             symbolic_expression=expr)
        else:
            try:
                if self.symbol is not None:
                    expr = self.symbol / other
                    descr = self.description + " / " + str(other)
                elif self.symbolic_expression is not None:
                    expr = (self.symbolic_expression) / other
                    descr = "(" + self.description + ") / " + str(other)
                else:
                    expr = None
                    descr = self.description + " / " + str(other)
                return Parameter(res, input_dimensional=self.return_dimensional, return_dimensional=self.return_dimensional,
                                 scale_object=self._scale_object, description=descr, units=self.units, symbol=None,
                                 symbolic_expression=expr)
            except:
                return res

    def __rtruediv__(self, other):
        res = other / float(self)
        try:
            if self.symbol is not None:
                expr = other / self.symbol
                descr = str(other) + " / " + self.description
            elif self.symbolic_expression is not None:
                expr = other / (self.symbolic_expression)
                descr = str(other) + " / (" + self.description + ")"
            else:
                expr = None
                descr = str(other) + " / " + self.description
            return Parameter(res, input_dimensional=self.return_dimensional,
                             return_dimensional=self.return_dimensional, scale_object=self._scale_object,
                             description=descr, units=self.units, symbol=None, symbolic_expression=expr)
        except:
            return res

    def __pow__(self, power, modulo=None):

        if modulo is not None:
            raise NotImplemented('Parameter class: Modular exponentiation not implemented')

        res = float(self) ** power
        if int(power) == power:

            ul = self.units.split('][')
            ul[0] = ul[0][1:]
            ul[-1] = ul[-1][:-1]

            usl = list()
            for us in ul:
                up = us.split('^')
                if len(up) == 1:
                    up.append("1")

                usl.append(tuple(up))

            units_elements = list()
            for us in usl:
                units_elements.append(list((us[0], str(int(us[1]) * power))))

            units = list()
            for us in units_elements:
                if us is not None:
                    if int(us[1]) != 1:
                        units.append("[" + us[0] + "^" + us[1] + "]")
                    else:
                        units.append("[" + us[0] + "]")
            units = "".join(units)

            if self.symbolic_expression is not None:
                expr = (self.symbolic_expression) ** power
                descr = "(" + self.description + ") to the power "+str(power)
            elif self.symbol is not None:
                expr = self.symbol ** power
                descr = self.description + " to the power "+str(power)
            else:
                expr = None
                descr = self.description + " to the power "+str(power)

        else:
            power_fraction = Fraction(power)
            ul = self.units.split('][')
            ul[0] = ul[0][1:]
            ul[-1] = ul[-1][:-1]

            usl = list()
            for us in ul:
                up = us.split('^')
                if len(up) == 1:
                    up.append("1")

                usl.append(tuple(up))

            units_elements = list()
            for us in usl:
                new_power = int(us[1]) * power_fraction.numerator / power_fraction.denominator
                if int(new_power) == new_power:
                    units_elements.append(list((us[0], str(int(new_power)))))
                else:
                    raise ArithmeticError("Parameter class: Only support integer exponent in units")

            units = list()
            for us in units_elements:
                if us is not None:
                    if int(us[1]) != 1:
                        units.append("[" + us[0] + "^" + us[1] + "]")
                    else:
                        units.append("[" + us[0] + "]")
            units = "".join(units)
            if self.symbolic_expression is not None:
                expr = (self.symbolic_expression) ** power
                descr = "(" + self.description + ") to the power "+str(power)
            elif self.symbol is not None:
                expr = self.symbol ** power
                descr = self.description + " to the power "+str(power)
            else:
                expr = None
                descr = self.description + " to the power "+str(power)

        return Parameter(res, input_dimensional=self.return_dimensional, return_dimensional=self.return_dimensional,
                         description=descr, units=units, scale_object=self._scale_object, symbol=None,
                         symbolic_expression=expr)


class ParametersArray(np.ndarray):
    """Base class of model's array of parameters.

    Parameters
    ----------
    values: list(float) or ~numpy.ndarray(float) or list(Parameter) or ~numpy.ndarray(Parameter) or list(ScalingParameter) or ~numpy.ndarray(ScalingParameter)
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
        String or an iterable of strings, describing the parameters.
        If an iterable, should have the same length or shape as `values`.
    symbols: ~sympy.core.symbol.Symbol or list(~sympy.core.symbol.Symbol) or ~numpy.ndarray(~sympy.core.symbol.Symbol), optional
        A `Sympy`_ symbol or an iterable of symbols, to represent the parameters in symbolic expressions.
        If an iterable, should have the same length or shape as `values`.
    symbolic_expressions: ~sympy.core.expr.Expr or list(~sympy.core.expr.Expr) or ~numpy.ndarray(~sympy.core.expr.Expr), optional
        A `Sympy`_ expression or an iterable of expressions, to represent a relationship to other parameters.
        If an iterable, should have the same length or shape as `values`.
    return_dimensional: bool, optional
        Defined if the value returned by the parameter is dimensional or not. Default to `False`.

    Warnings
    --------
    If no scale_object argument is provided, cannot transform between the dimensional and nondimensional value !

    .. _Sympy: https://www.sympy.org/
    """

    def __new__(cls, values, input_dimensional=True, units="", scale_object=None, description="",
                symbols=None, symbolic_expressions=None, return_dimensional=False):

        if isinstance(values, (tuple, list)):
            new_arr = np.empty(len(values), dtype=object)
            for i, val in enumerate(values):
                if isinstance(description, (tuple, list, np.ndarray)):
                    descr = description[i]
                else:
                    descr = description
                if isinstance(symbols, (tuple, list, np.ndarray)):
                    sy = symbols[i]
                else:
                    sy = symbols
                if isinstance(symbolic_expressions, (tuple, list, np.ndarray)):
                    expr = symbolic_expressions[i]
                else:
                    expr = symbolic_expressions
                new_arr[i] = Parameter(val, input_dimensional=input_dimensional, units=units, scale_object=scale_object, description=descr,
                                       return_dimensional=return_dimensional, symbol=sy, symbolic_expression=expr)
        else:
            if isinstance(values.flatten()[0], (Parameter, ScalingParameter)):
                new_arr = values.copy()
            else:
                new_arr = np.empty_like(values, dtype=object)
                for idx in np.ndindex(values.shape):
                    if isinstance(description, np.ndarray):
                        descr = description[idx]
                    else:
                        descr = description
                    if isinstance(symbols, np.ndarray):
                        sy = symbols[idx]
                    else:
                        sy = symbols
                    if isinstance(symbolic_expressions, np.ndarray):
                        expr = symbolic_expressions[idx]
                    else:
                        expr = symbolic_expressions
                    new_arr[idx] = Parameter(values[idx], input_dimensional=input_dimensional, units=units, scale_object=scale_object, description=descr,
                                             return_dimensional=return_dimensional, symbol=sy, symbolic_expression=expr)
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
    def symbolic_expressions(self):
        """~numpy.ndarray(~sympy.core.expr.Expr): Returns the symbolic expressions of the parameters in the array."""
        symbolic_expressions = np.empty(self.shape, dtype=object)
        for idx in np.ndindex(self.shape):
            symbolic_expressions[idx] = self[idx].symbolic_expression
        return symbolic_expressions

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

    def __add__(self, other):
        if isinstance(other, (Parameter, ScalingParameter, float, int)):
            res = np.empty(self.shape, dtype=object)
            for idx in np.ndindex(self.shape):
                res[idx] = self[idx] + other
            item = res[idx]
            return ParametersArray(res, input_dimensional=item.return_dimensional, return_dimensional=item.return_dimensional,
                                   units=item.units, scale_object=self._scale_object)
        elif isinstance(other, ParametersArray):
            if other.shape == self.shape:  # Does not do broadcast
                res = np.empty(self.shape, dtype=object)
                for idx in np.ndindex(self.shape):
                    res[idx] = self[idx] + other[idx]
                item = res[idx]
                return ParametersArray(res, input_dimensional=item.return_dimensional, return_dimensional=item.return_dimensional,
                                       units=item.units, scale_object=self._scale_object)
            else:
                return self + other
        else:
            return self + other

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, (Parameter, ScalingParameter, float, int)):
            res = np.empty(self.shape, dtype=object)
            for idx in np.ndindex(self.shape):
                res[idx] = self[idx] - other
            item = res[idx]
            return ParametersArray(res, input_dimensional=item.return_dimensional, return_dimensional=item.return_dimensional,
                                   units=item.units, scale_object=self._scale_object)
        elif isinstance(other, ParametersArray):
            if other.shape == self.shape:  # Does not do broadcast
                res = np.empty(self.shape, dtype=object)
                for idx in np.ndindex(self.shape):
                    res[idx] = self[idx] - other[idx]
                item = res[idx]
                return ParametersArray(res, input_dimensional=item.return_dimensional, return_dimensional=item.return_dimensional,
                                       units=item.units, scale_object=self._scale_object)
            else:
                return self - other
        else:
            return self - other

    def __rsub__(self, other):
        if isinstance(other, (Parameter, ScalingParameter, float, int)):
            res = np.empty(self.shape, dtype=object)
            for idx in np.ndindex(self.shape):
                res[idx] = other - self[idx]
            item = res[idx]
            return ParametersArray(res, input_dimensional=item.return_dimensional, return_dimensional=item.return_dimensional,
                                   units=item.units, scale_object=self._scale_object)
        elif isinstance(other, ParametersArray):
            if other.shape == self.shape:  # Does not do broadcast
                res = np.empty(self.shape, dtype=object)
                for idx in np.ndindex(self.shape):
                    res[idx] = other - self[idx]
                item = res[idx]
                return ParametersArray(res, input_dimensional=item.return_dimensional, return_dimensional=item.return_dimensional,
                                       units=item.units, scale_object=self._scale_object)
            else:
                return other - self
        else:
            return other - self

    def __mul__(self, other):
        if isinstance(other, (Parameter, ScalingParameter, float, int)):
            res = np.empty(self.shape, dtype=object)
            for idx in np.ndindex(self.shape):
                res[idx] = self[idx] * other
            item = res[idx]
            return ParametersArray(res, input_dimensional=item.return_dimensional, return_dimensional=item.return_dimensional,
                                   units=item.units, scale_object=self._scale_object)
        elif isinstance(other, ParametersArray):
            if other.shape == self.shape:  # Does not do broadcast
                res = np.empty(self.shape, dtype=object)
                for idx in np.ndindex(self.shape):
                    res[idx] = self[idx] * other[idx]
                item = res[idx]
                return ParametersArray(res, input_dimensional=item.return_dimensional, return_dimensional=item.return_dimensional,
                                       units=item.units, scale_object=self._scale_object)
            else:
                return self * other
        else:
            return self * other

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, (Parameter, ScalingParameter, float, int)):
            res = np.empty(self.shape, dtype=object)
            for idx in np.ndindex(self.shape):
                res[idx] = self[idx] / other
            item = res[idx]
            return ParametersArray(res, input_dimensional=item.return_dimensional, return_dimensional=item.return_dimensional,
                                   units=item.units, scale_object=self._scale_object)
        elif isinstance(other, ParametersArray):
            if other.shape == self.shape:  # Does not do broadcast
                res = np.empty(self.shape, dtype=object)
                for idx in np.ndindex(self.shape):
                    res[idx] = self[idx] / other[idx]
                item = res[idx]
                return ParametersArray(res, input_dimensional=item.return_dimensional, return_dimensional=item.return_dimensional,
                                       units=item.units, scale_object=self._scale_object)
            else:
                return self / other
        else:
            return self / other

    def __rtruediv__(self, other):
        if isinstance(other, (Parameter, ScalingParameter, float, int)):
            res = np.empty(self.shape, dtype=object)
            for idx in np.ndindex(self.shape):
                res[idx] = other / self[idx]
            item = res[idx]
            return ParametersArray(res, input_dimensional=item.return_dimensional, return_dimensional=item.return_dimensional,
                                   units=item.units, scale_object=self._scale_object)
        elif isinstance(other, ParametersArray):
            if other.shape == self.shape:  # Does not do broadcast
                res = np.empty(self.shape, dtype=object)
                for idx in np.ndindex(self.shape):
                    res[idx] = other / self[idx]
                item = res[idx]
                return ParametersArray(res, input_dimensional=item.return_dimensional, return_dimensional=item.return_dimensional,
                                       units=item.units, scale_object=self._scale_object)
            else:
                return other / self
        else:
            return other / self


def _combine_units(units1, units2, operation):
    ul = units1.split('][')
    ul[0] = ul[0][1:]
    ul[-1] = ul[-1][:-1]
    ol = units2.split('][')
    ol[0] = ol[0][1:]
    ol[-1] = ol[-1][:-1]

    usl = list()
    for us in ul:
        up = us.split('^')
        if len(up) == 1:
            up.append("1")

        if up[0]:
            usl.append(tuple(up))

    osl = list()
    for os in ol:
        op = os.split('^')
        if len(op) == 1:
            op.append("1")

        if op[0]:
            osl.append(tuple(op))

    units_elements = list()
    for us in usl:
        new_us = [us[0]]
        i = 0
        for os in osl:
            if os[0] == us[0]:
                if operation == '-':
                    power = int(os[1]) - int(us[1])
                else:
                    power = int(os[1]) + int(us[1])
                del osl[i]
                break
            i += 1
        else:
            power = int(us[1])

        if power != 0:
            new_us.append(str(power))
            units_elements.append(new_us)

    if len(osl) != 0:
        units_elements += osl

    units = list()
    for us in units_elements:
        if us is not None:
            if int(us[1]) != 1:
                units.append("[" + us[0] + "^" + us[1] + "]")
            else:
                units.append("[" + us[0] + "]")
    return "".join(units)
