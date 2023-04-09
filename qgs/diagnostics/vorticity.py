"""
    Diagnostic vorticity classes
    ============================

    Classes defining vorticity fields diagnostics.

    Description of the classes
    --------------------------

    * :class:`AtmosphericVorticityDiagnostic`: General base class for atmospheric vorticity diagnostic.
    * :class:`LowerLayerAtmosphericVorticityDiagnostic`: Diagnostic giving the lower layer atmospheric vorticity fields :math:`\\nabla^2 \\psi^3_{\\rm a}`.
    * :class:`MiddleAtmosphericVorticityDiagnostic`: Diagnostic giving the middle atmospheric vorticity fields :math:`\\nabla^2 \\psi_{\\rm a}`.
    * :class:`UpperLayerAtmosphericVorticityDiagnostic`: Diagnostic giving the upper layer atmospheric vorticity fields :math:`\\nabla^2 \\psi^1_{\\rm a}`.

"""

import warnings

from qgs.diagnostics.differential import LaplacianFieldDiagnostic
from qgs.diagnostics.util import create_grid_basis

import numpy as np
import matplotlib.pyplot as plt


class AtmosphericVorticityDiagnostic(LaplacianFieldDiagnostic):
    """General base class for atmospheric vorticity fields diagnostic.
    Provide a spatial gridded representation of the fields.
    This is an `abstract base class`_, it must be subclassed to create new diagnostics!

    .. _abstract base class: https://docs.python.org/3/glossary.html#term-abstract-base-class

    Parameters
    ----------

    model_params: QgParams
        An instance of the model parameters.
    delta_x: float, optional
        Spatial step in the zonal direction `x` for the gridded representation of the field.
        If not provided, take an optimal guess based on the provided model's parameters.
    delta_y: float, optional
        Spatial step in the meridional direction `y` for the gridded representation of the field.
        If not provided, take an optimal guess based on the provided model's parameters.
    dimensional: bool
        Indicate if the output diagnostic must be dimensionalized or not.

    Attributes
    ----------

    dimensional: bool
        Indicate if the output diagnostic must be dimensionalized or not.
    """

    def __init__(self, model_params, delta_x=None, delta_y=None, dimensional=True):

        LaplacianFieldDiagnostic.__init__(self, model_params, dimensional)
        self._configure(delta_x=delta_x, delta_y=delta_y)

        self._plot_units = r" (in " + r's$^{-1}$' + r")"
        self._default_plot_kwargs['cmap'] = plt.get_cmap('hsv_r')
        self._color_bar_format = False

    def _compute_grid(self, delta_x=None, delta_y=None):

        if delta_x is None:
            ams = self._model_params.ablocks
            if ams is None:
                warnings.warn("AtmosphericVorticityDiagnostic: Unable to configure the grid automatically. Atmospheric wavenumbers information not " +
                              "present in the model's parameters ! Please call the compute_grid method with the delta_x and delta_y parameters.")
                return 1
            xwn = [ams[i][0] for i in range(len(ams))]
            mxwn = max(xwn)
            n_point_x = 4 * mxwn + 2
        else:
            n_point_x = int(np.ceil((2 * np.pi / self._model_params.scale_params.n) / delta_x) + 1)

        if delta_y is None:
            ams = self._model_params.ablocks
            if ams is None:
                warnings.warn("AtmosphericVorticityDiagnostic: Unable to configure the grid automatically. Atmospheric wavenumbers information not " +
                              "present in the model's parameters ! Please call the compute_grid method with the delta_x and delta_y parameters.")
                return 1
            ywn = [ams[i][1] for i in range(len(ams))]
            mywn = max(ywn)
            n_point_y = 4 * mywn + 2
        else:
            n_point_y = int(np.ceil(np.pi / delta_y) + 1)

        x = np.linspace(0., 2 * np.pi / self._model_params.scale_params.n, n_point_x)
        y = np.linspace(0., np.pi, n_point_y)
        self._X, self._Y = np.meshgrid(x, y)

    def _configure(self, delta_x=None, delta_y=None):

        basis = self._model_params.atmospheric_basis
        self._configure_laplacian_grid(basis, delta_x, delta_y)

        if self._orography and self._X is not None and self._Y is not None:
            if self._model_params.ground_params.orographic_basis == "atmospheric":
                self._oro_basis = create_grid_basis(basis, self._X, self._Y, self._subs)
            else:
                self._oro_basis = create_grid_basis(self._model_params.ground_basis, self._X, self._Y, self._subs)
        else:
            self._oro_basis = None


class LowerLayerAtmosphericVorticityDiagnostic(AtmosphericVorticityDiagnostic):
    """Diagnostic giving the lower layer atmospheric vorticity fields :math:`\\nabla^2 \\psi^3_{\\rm a}`.
    Computed as :math:`\\nabla^2 \\psi^3_{\\rm a} = \\nabla^2 \\psi_{\\rm a} - \\nabla^2 \\theta_{\\rm a}` where :math:`\\psi_{\\rm a}` and :math:`\\theta_{\\rm a}` are respectively the barotropic and baroclinic streamfunctions.
    See also the :ref:`files/model/atmosphere:Atmospheric component` and :ref:`files/model/oro_model:Mid-layer equations and the thermal wind relation` sections.

    Parameters
    ----------

    model_params: QgParams
        An instance of the model parameters.
    delta_x: float, optional
        Spatial step in the zonal direction `x` for the gridded representation of the field.
        If not provided, take an optimal guess based on the provided model's parameters.
    delta_y: float, optional
        Spatial step in the meridional direction `y` for the gridded representation of the field.
        If not provided, take an optimal guess based on the provided model's parameters.
    dimensional: bool, optional
        Indicate if the output diagnostic must be dimensionalized or not.
        Default to `True`.

    Attributes
    ----------

    dimensional: bool
        Indicate if the output diagnostic must be dimensionalized or not.

    """

    def __init__(self, model_params, delta_x=None, delta_y=None, dimensional=True):

        AtmosphericVorticityDiagnostic.__init__(self, model_params, delta_x, delta_y, dimensional)

        self._plot_title = r'Atmospheric vorticity in the lower layer'

    def _get_diagnostic(self, dimensional):

        if self._model_params.dynamic_T:
            offset = 1
        else:
            offset = 0

        vr = self._model_params.variables_range
        psi = np.swapaxes(self._data[:vr[0], ...].T @ np.swapaxes(self._grid_basis[offset:], 0, 1), 0, 1)
        theta = np.swapaxes(self._data[vr[0]+offset:vr[1], ...].T @ np.swapaxes(self._grid_basis[offset:], 0, 1), 0, 1)

        psi3 = psi - theta

        if dimensional:
            self._diagnostic_data = psi3 * self._model_params.streamfunction_scaling / (self._model_params.scale_params.L ** 2)
            self._diagnostic_data_dimensional = True
        else:
            self._diagnostic_data = psi3
            self._diagnostic_data_dimensional = False
        return self._diagnostic_data
