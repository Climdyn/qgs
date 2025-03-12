
"""
    Inner products definition module
    ================================

    Module containing classes to define the `inner products`_ used by the model.

    .. _inner products: https://en.wikipedia.org/wiki/Inner_product_space
    
"""

from abc import ABC, abstractmethod

# from sympy.simplify import trigsimp
from sympy.simplify.fu import TR8, TR10
from sympy import Symbol, diff, integrate, symbols, pi, Integral

_n = Symbol('n', positive=True)
_x, _y = symbols('x y')


class InnerProductDefinition(ABC):
    """Base class to define the model's basis inner products.

    Parameters
    ----------
    optimizer: None or callable, optional
        A function to optimize the computation of the integrals or the integrand.
        If `None`, does not optimize.

    Attributes
    ----------
    optimizer: None or callable
        A function to optimize the computation of the integrals or the integrand.
        If `None`, does not optimize the computation.
    """

    def __init__(self, optimizer=None):

        self.optimizer = None

        if optimizer is not None:
            self.set_optimizer(optimizer)
        else:
            self.set_optimizer(self._no_optimizer)

    def set_optimizer(self, optimizer):
        """Function to set the optimizer.

        Parameters
        ----------
        optimizer: callable
            A function to optimize the computation of the integrals or the integrand.
        """
        self.optimizer = optimizer

    @staticmethod
    def _no_optimizer(expr):
        return expr

    @staticmethod
    @abstractmethod
    def jacobian(S, G):
        """Jacobian present in the advection terms:

        Parameters
        ----------
        S:
            1st argument the Jacobian.
        G:
            2nd argument the Jacobian.
        """
        pass

    @staticmethod
    @abstractmethod
    def laplacian(S):
        """Laplacian :math:`\\nabla^2 S` of a function :math:`S`."""
        pass

    @abstractmethod
    def ip_lap(self, S, G, symbolic_expr=False):
        """Function to compute the inner product :math:`(S, \\nabla^2 G)`.

        Parameters
        ----------
        S:
            Left-hand side function of the product.
        G:
            Right-hand side function of the product.
        symbolic_expr: bool, optional
            If `True`, return the integral as a symbolic expression object. Else, return the integral performed symbolically.
        """
        pass

    @abstractmethod
    def ip_diff_x(self, S, G, symbolic_expr=False):
        """Function to compute the inner product :math:`(S, \\partial_x G)`.

        Parameters
        ----------
        S:
            Left-hand side function of the product.
        G:
            Right-hand side function of the product.
        symbolic_expr: bool, optional
            If `True`, return the integral as a symbolic expression object. Else, return the integral performed symbolically.
        """
        pass

    @abstractmethod
    def ip_jac(self, S, G, H, symbolic_expr=False):
        """Function to compute the inner product :math:`(S, J(G, H))`, where
        :math:`J` is the :meth:`jacobian`.

        Parameters
        ----------
        S:
            Left-hand side function of the product.
        G:
            1st argument of the right-hand side function of the product.
        H:
            2nd argument of the right-hand side function of the product.
        symbolic_expr: bool, optional
            If `True`, return the integral as a symbolic expression object. Else, return the integral performed symbolically.
        """
        pass

    @abstractmethod
    def ip_jac_lap(self, S, G, H, symbolic_expr=False):
        """Function to compute the inner product :math:`(S, J(G, \\nabla^2 H))`, where
        :math:`J` is the :meth:`jacobian`.

        Parameters
        ----------
        S:
            Left-hand side function of the product.
        G:
            1st argument of the right-hand side function of the product.
        H:
            2nd argument of the right-hand side function of the product.
        symbolic_expr: bool, optional
            If `True`, return the integral as a symbolic expression object. Else, return the integral performed symbolically.
        """
        pass


class SymbolicInnerProductDefinition(InnerProductDefinition):
    """Base class to define symbolic inner products using `Sympy`_.

    Parameters
    ----------
    optimizer: None or callable, optional
        A function to simplify the integrand to optimize the computation of the integrals.
        Should return a `Sympy`_ expression.
        If `None`, does not optimize the computation.

    Attributes
    ----------
    optimizer: None or callable
        A function to simplify the integrand to optimize the computation of the integrals.
        Should return a `Sympy`_ expression.
        If `None`, does not optimize the computation.


    .. _Sympy: https://www.sympy.org/

    """

    def __init__(self, optimizer=None):

        InnerProductDefinition.__init__(self, optimizer)

    @abstractmethod
    def symbolic_inner_product(self, S, G, symbolic_expr=False, integrand=False):
        """Symbolic definition of the inner product :math:`(S, G)`.

        Parameters
        ----------
        S: Sympy expression
            Left-hand side function of the product.
        G: Sympy expression
            Right-hand side function of the product.
        symbolic_expr: bool, optional
            If `True`, return the integral as a symbolic expression object. Else, return the integral performed symbolically.
        integrand: bool, optional
            If `True`, return the integrand of the integral and its integration limits as a list of symbolic expression object. Else, return the integral performed symbolically.

        Returns
        -------
        Sympy expression
            The symbolic result of the inner product.
        """
        pass

    def ip_lap(self, S, G, symbolic_expr=False, integrand=False):
        """Function to compute the inner product :math:`(S, \\nabla^2 G)`.

        Parameters
        ----------
        S: Sympy expression
            Left-hand side function of the product.
        G: Sympy expression
            Right-hand side function of the product.
        symbolic_expr: bool, optional
            If `True`, return the integral as a symbolic expression object. Else, return the integral performed symbolically.
        integrand: bool, optional
            If `True`, return the integrand of the integral and its integration limits as a list of symbolic expression object. Else, return the integral performed symbolically.

        Returns
        -------
        Sympy expression
            The symbolic result of the inner product.
        """
        return self.symbolic_inner_product(S, self.laplacian(G), symbolic_expr=symbolic_expr, integrand=integrand)

    def ip_diff_x(self, S, G, symbolic_expr=False, integrand=False):
        """Function to compute the inner product :math:`(S, \\partial_x G)`.

        Parameters
        ----------
        S: Sympy expression
            Left-hand side function of the product.
        G: Sympy expression
            Right-hand side function of the product.
        symbolic_expr: bool, optional
            If `True`, return the integral as a symbolic expression object. Else, return the integral performed symbolically.
        integrand: bool, optional
            If `True`, return the integrand of the integral and its integration limits as a list of symbolic expression object. Else, return the integral performed symbolically.

        Returns
        -------
        Sympy expression
            The symbolic result of the inner product.
        """
        return self.symbolic_inner_product(S, diff(G, _x), symbolic_expr=symbolic_expr, integrand=integrand)

    def ip_jac(self, S, G, H, symbolic_expr=False, integrand=False):
        """Function to compute the inner product :math:`(S, J(G, H))`, where
        :math:`J` is the :meth:`jacobian`.

        Parameters
        ----------
        S:  Sympy expression
            Left-hand side function of the product.
        G:  Sympy expression
            1st argument of the right-hand side function of the product.
        H:  Sympy expression
            2nd argument of the right-hand side function of the product.
        symbolic_expr: bool, optional
            If `True`, return the integral as a symbolic expression object. Else, return the integral performed symbolically.
        integrand: bool, optional
            If `True`, return the integrand of the integral and its integration limits as a list of symbolic expression object. Else, return the integral performed symbolically.

        Returns
        -------
        Sympy expression
            The symbolic result of the inner product.
        """
        return self.symbolic_inner_product(S, self.jacobian(G, H), symbolic_expr=symbolic_expr, integrand=integrand)

    def ip_jac_lap(self, S, G, H, symbolic_expr=False, integrand=False):
        """Function to compute the inner product :math:`(S, J(G, \\nabla^2 H))`, where
        :math:`J` is the :meth:`jacobian`.

        Parameters
        ----------
        S:  Sympy expression
            Left-hand side function of the product.
        G:  Sympy expression
            1st argument of the right-hand side function of the product.
        H:  Sympy expression
            2nd argument of the right-hand side function of the product.
        symbolic_expr: bool, optional
            If `True`, return the integral as a symbolic expression object. Else, return the integral performed symbolically.
        integrand: bool, optional
            If `True`, return the integrand of the integral and its integration limits as a list of symbolic expression object. Else, return the integral performed symbolically.

        Returns
        -------
        Sympy expression
            The symbolic result of the inner product.
        """
        return self.symbolic_inner_product(S, self.jacobian(G, self.laplacian(H)), symbolic_expr=symbolic_expr, integrand=integrand)


class StandardSymbolicInnerProductDefinition(SymbolicInnerProductDefinition):
    """Standard qgs class to define symbolic inner products using `Sympy`_.

    Parameters
    ----------
    optimizer: None or callable, optional
        A function to simplify the integrand to optimize the computation of the integrals.
        Should return a `Sympy`_ expression.
        If `None`, does not optimize the computation.

    Attributes
    ----------
    optimizer: None or callable
        A function to simplify the integrand to optimize the computation of the integrals.
        Should return a `Sympy`_ expression.
        If `None`, does not optimize the computation.


    .. _Sympy: https://www.sympy.org/

    """

    def __init__(self, optimizer=None):

        if optimizer is None:
            SymbolicInnerProductDefinition.__init__(self, self._trig_optimizer)
        else:
            SymbolicInnerProductDefinition.__init__(self, optimizer)

    @staticmethod
    def _trig_optimizer(expr):
        return TR10(TR8(expr))

    @staticmethod
    def jacobian(S, G):
        """Jacobian present in the advection terms:

        .. math:

            J(S, G) = \\partial_x S\\, \\partial_y G - \\partial_y S\\, \\partial_x G

        Parameters
        ----------
        S: Sympy expression
            1st argument the Jacobian.
        G: Sympy expression
            2nd argument the Jacobian.

        Returns
        -------
        Sympy expression
            The Jacobian.
        """
        return diff(S, _x) * diff(G, _y) - diff(G, _x) * diff(S, _y)

    @staticmethod
    def laplacian(S):
        """Laplacian :math:`\\nabla^2 S` of a given function :math:`S` in 2D cartesian coordinates:
        :math:`\\nabla^2 S = \\partial^2_x S + \\partial^2_y S`.

        Parameters
        ----------
        S: Sympy expression
            Functions to take the Laplacian of.

        Returns
        -------
        Sympy expression
            The Laplacian.
        """
        return diff(S, _x, 2) + diff(S, _y, 2)

    @staticmethod
    def integrate_over_domain(expr, symbolic_expr=False):
        """Definition of the normalized integrals over the spatial domain used by the inner products:
        :math:`\\frac{n}{2\\pi^2}\\int_0^\\pi\\int_0^{2\\pi/n} \\, \\mathrm{expr}(x, y) \\, \\mathrm{d} x \\, \\mathrm{d} y`.

        Parameters
        ----------
        expr: Sympy expression
            The expression to integrate.
        symbolic_expr: bool, optional
            If `True`, return the integral as a symbolic expression object. Else, return the integral performed symbolically.

        Returns
        -------
        Sympy expression
            The result of the symbolic integration.
        """
        if symbolic_expr:
            return Integral(expr, (_x, 0, 2 * pi / _n), (_y, 0, pi))
        else:
            return integrate(expr, (_x, 0, 2 * pi / _n), (_y, 0, pi))

    def symbolic_inner_product(self, S, G, symbolic_expr=False, integrand=False):
        """Function defining the inner product to be computed symbolically:
        :math:`(S, G) = \\frac{n}{2\\pi^2}\\int_0^\\pi\\int_0^{2\\pi/n} S(x,y)\\, G(x,y)\\, \\mathrm{d} x \\, \\mathrm{d} y`.

        Parameters
        ----------
        S: Sympy expression
            Left-hand side function of the product.
        G: Sympy expression
            Right-hand side function of the product.
        symbolic_expr: bool, optional
            If `True`, return the integral as a symbolic expression object. Else, return the integral performed symbolically.
        integrand: bool, optional
            If `True`, return the integrand of the integral and its integration limits as a list of symbolic expression object. Else, return the integral performed symbolically.

        Returns
        -------
        Sympy expression
            The result of the symbolic integration
        """
        expr = (_n / (2 * pi ** 2)) * S * G
        if integrand:
            return expr, (_x, 0, 2 * pi / _n), (_y, 0, pi)
        else:
            return self.integrate_over_domain(self.optimizer(expr), symbolic_expr=symbolic_expr)
