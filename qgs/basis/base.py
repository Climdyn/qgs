"""
    Basis definition module (base class)
    ====================================

    Abstract base classes defining the functions (modes) of the basis of the model and used to configure it.
    (see :ref:`files/model/oro_model:Projecting the equations on a set of basis functions`).

    Description of the classes
    --------------------------

    * :class:`Basis`: General base class.
    * :class:`SymbolicBasis`: Base class for symbolic functions basis.

    Warnings
    --------

    These are `abstract base class`_, they must be subclassed to create new basis!

    .. _abstract base class: https://docs.python.org/3/glossary.html#term-abstract-base-class

"""

# TODO: define setters and init arguments
# TODO: define NumericBasis class

import sys

from abc import ABC
from sympy import symbols, lambdify


class Basis(ABC):
    """General base class for a basis of functions.

    Attributes
    ----------
    functions: list
        List of functions of the basis.
    """

    def __init__(self):

        self.functions = list()

    def __getitem__(self, index):
        return self.functions[index]

    def __repr__(self):
        return self.functions.__repr__()

    def __str__(self):
        return self.functions.__str__()

    def __len__(self):
        return self.functions.__len__()


class SymbolicBasis(Basis):
    """General base class for a basis of symbolic functions.

    Attributes
    ----------
    substitutions: list(tuple)
        List of 2-tuples containing the substitutions to be made with the functions. The 2-tuples contain first
        a `Sympy`_  expression and then the value to substitute.

    .. _Sympy: https://www.sympy.org/

    """

    def __init__(self):

        Basis.__init__(self)

        self.substitutions = list()

    def subs_functions(self, extra_subs=None):
        """Return the basis functions with the substitutions stored in the object being applied.

        Parameters
        ----------
        extra_subs: list(tuple), optional
            List of 2-tuples containing extra substitutions to be made with the functions. The 2-tuples contain first
            a `Sympy`_  expression and then the value to substitute.

        Returns
        -------
        list
            List of the substituted basis functions
        """

        sf = list()

        for f in self.functions:
            ff = f.subs(self.substitutions)
            if extra_subs is not None:
                ff = ff.subs(extra_subs)
            sf.append(ff)

        return sf

    def num_functions(self, extra_subs=None):
        """Return the basis functions with as python callable.

        Parameters
        ----------
        extra_subs: list(tuple), optional
            List of 2-tuples containing extra substitutions to be made with the functions before transforming them into
            python callable. The 2-tuples contain first a `Sympy`_  expression and then the value to substitute.

        Returns
        -------
        list(callable)
            List of callable basis functions
        """

        x, y = symbols('x y')

        nf = list()
        sf = self.subs_functions(extra_subs=extra_subs)

        for f in sf:
            try:
                nf.append(lambdify([x, y], f))
            except:
                tb = sys.exc_info()[2]
                raise Exception.with_traceback(tb)

        return nf


