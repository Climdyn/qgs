"""
    Differential diagnostic base class
    ==================================

    Abstract base classes defining diagnostics on differnentiated grids.

    Description of the classes
    --------------------------

    * :class:`DifferentialFieldDiagnostic`: Base class for diagnostics returning model's fields based on differential grids.

    Warnings
    --------

    These are `abstract base class`_, they must be subclassed to create new diagnostics!

    .. _abstract base class: https://docs.python.org/3/glossary.html#term-abstract-base-class

"""

from qgs.diagnostics.util import create_grid_basis

from qgs.diagnostics.base import FieldDiagnostic


class DifferentialFieldDiagnostic(FieldDiagnostic):
    """General base class for differential fields diagnostic.
    This is an `abstract base class`_, it must be subclassed to create new diagnostics!

    .. _abstract base class: https://docs.python.org/3/glossary.html#term-abstract-base-class

    Parameters
    ----------

    model_params: QgParams
        An instance of the model parameters.
    dimensional: bool
        Indicate if the output diagnostic must be dimensionalized or not.

    Attributes
    ----------

    dimensional: bool
        Indicate if the output diagnostic must be dimensionalized or not.
    """

    def __init__(self, model_params, dimensional):

        FieldDiagnostic.__init__(self, model_params, dimensional)

    def _configure_differential_grid(self, basis, derivative, order, delta_x=None, delta_y=None):

        self._compute_grid(delta_x, delta_y)

        if derivative == "dx":
            dx_basis = basis.x_derivative(order)
            self._grid_basis = create_grid_basis(dx_basis, self._X, self._Y, self._subs)

        elif derivative == "dy":
            dy_basis = basis.y_derivative(order)
            self._grid_basis = create_grid_basis(dy_basis, self._X, self._Y, self._subs)