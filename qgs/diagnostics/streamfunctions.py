"""
    Diagnostic streamfunction classes
    =================================

    Classes defining streamfunction fields diagnostics.

    Description of the classes
    --------------------------

    * :class:`AtmosphericStreamfunctionDiagnostic`: General base class for atmospheric streamfunction fields diagnostic.
    * :class:`LowerLayerAtmosphericStreamfunctionDiagnostic`: Diagnostic giving the lower layer atmospheric streamfunction fields :math:`\\psi^3_{\\rm a}`.
    * :class:`UpperLayerAtmosphericStreamfunctionDiagnostic`: Diagnostic giving the upper layer atmospheric streamfunction fields :math:`\\psi^1_{\\rm a}`.
    * :class:`MiddleAtmosphericStreamfunctionDiagnostic`: Diagnostic giving the middle atmospheric streamfunction fields :math:`\\psi_{\\rm a}`.
    * :class:`OceanicLayerStreamfunctionDiagnostic`: Diagnostic giving the oceanic layer streamfunction fields :math:`\\psi_{\\rm o}`.

"""

import warnings

import numpy as np
from scipy.integrate import dblquad
import matplotlib.pyplot as plt
from qgs.diagnostics.util import create_grid_basis

from qgs.diagnostics.base import FieldDiagnostic

# TODO: - convert the matmul and swapaxes into tensordot (cleaner)


class AtmosphericStreamfunctionDiagnostic(FieldDiagnostic):
    """General base class for atmospheric streamfunction fields diagnostic.
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

        FieldDiagnostic.__init__(self, model_params, dimensional)
        self._configure(delta_x=delta_x, delta_y=delta_y)

        self._default_plot_kwargs['cmap'] = plt.get_cmap('jet')

    def _compute_grid(self, delta_x=None, delta_y=None):

        if delta_x is None:
            ams = self._model_params.ablocks
            if ams is None:
                warnings.warn("AtmosphericStreamfunctionDiagnostic: Unable to configure the grid automatically. Atmospheric wavenumbers information not " +
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
                warnings.warn("AtmosphericStreamfunctionDiagnostic: Unable to configure the grid automatically. Atmospheric wavenumbers information not " +
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

        self._compute_grid(delta_x, delta_y)

        self._grid_basis = create_grid_basis(self._model_params.atmospheric_basis, self._X, self._Y, self._subs)
        if self._orography:
            if self._model_params.ground_params.orographic_basis == "atmospheric":
                self._oro_basis = self._grid_basis
            else:
                self._oro_basis = create_grid_basis(self._model_params.ground_basis, self._X, self._Y, self._subs)
        else:
            self._oro_basis = None


class LowerLayerAtmosphericStreamfunctionDiagnostic(AtmosphericStreamfunctionDiagnostic):
    """Diagnostic giving the lower layer atmospheric streamfunction fields :math:`\\psi^3_{\\rm a}`.
    Computed as :math:`\\psi^3_{\\rm a} = \\psi_{\\rm a} - \\theta_{\\rm a}` where :math:`\\psi_{\\rm a}` and :math:`\\theta_{\\rm a}` are respectively the barotropic and baroclinic streamfunctions.
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

        AtmosphericStreamfunctionDiagnostic.__init__(self, model_params, delta_x, delta_y, dimensional)

        self._plot_title = r'Atmospheric $\psi_{\rm a}^3$ streamfunction'
        self._plot_units = r" (in " + self._model_params.get_variable_units(0) + r")"

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
            self._diagnostic_data = psi3 * self._model_params.streamfunction_scaling
            self._diagnostic_data_dimensional = True
        else:
            self._diagnostic_data = psi3
            self._diagnostic_data_dimensional = False
        return self._diagnostic_data


class UpperLayerAtmosphericStreamfunctionDiagnostic(AtmosphericStreamfunctionDiagnostic):
    """Diagnostic giving the upper layer atmospheric streamfunction fields :math:`\\psi^1_{\\rm a}`.
    Computed as :math:`\\psi^1_{\\rm a} = \\psi_{\\rm a} + \\theta_{\\rm a}` where :math:`\\psi_{\\rm a}` and :math:`\\theta_{\\rm a}` are
    respectively the barotropic and baroclinic streamfunctions.
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

        AtmosphericStreamfunctionDiagnostic.__init__(self, model_params, delta_x, delta_y, dimensional)

        self._plot_title = r'Atmospheric $\psi_{\rm a}^1$ streamfunction'
        self._plot_units = r" (in " + self._model_params.get_variable_units(0) + r")"

    def _get_diagnostic(self, dimensional):

        if self._model_params.dynamic_T:
            offset = 1
        else:
            offset = 0

        vr = self._model_params.variables_range
        psi = np.swapaxes(self._data[:vr[0], ...].T @ np.swapaxes(self._grid_basis[offset:], 0, 1), 0, 1)
        theta = np.swapaxes(self._data[vr[0]+offset:vr[1], ...].T @ np.swapaxes(self._grid_basis[offset:], 0, 1), 0, 1)

        psi1 = psi + theta

        if dimensional:
            self._diagnostic_data = psi1 * self._model_params.streamfunction_scaling
            self._diagnostic_data_dimensional = True
        else:
            self._diagnostic_data = psi1
            self._diagnostic_data_dimensional = False
        return self._diagnostic_data


class MiddleAtmosphericStreamfunctionDiagnostic(AtmosphericStreamfunctionDiagnostic):
    """Diagnostic giving the middle atmospheric streamfunction fields :math:`\\psi_{\\rm a}` at 500hPa, i.e. the barotropic streamfunction of the system.
    See also :ref:`files/model/oro_model:Mid-layer equations and the thermal wind relation` sections.

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
    geopotential: bool, optional
        Dimensionalize the field in geopotential height (in meter).
        Default to `False`.

    Attributes
    ----------

    dimensional: bool
        Indicate if the output diagnostic must be dimensionalized or not.
    """

    def __init__(self, model_params, delta_x=None, delta_y=None, dimensional=True, geopotential=False):

        AtmosphericStreamfunctionDiagnostic.__init__(self, model_params, delta_x, delta_y, dimensional)

        if geopotential:
            self._plot_title = r'Atmospheric 500hPa geopotential height'
            self._plot_units = r" (in m)"
        else:
            self._plot_title = r'Atmospheric 500hPa $\psi_{\rm a}$ streamfunction'
            self._plot_units = r" (in " + self._model_params.get_variable_units(0) + r")"
        self._default_plot_kwargs['cmap'] = plt.get_cmap('gist_rainbow_r')
        self.geopotential = geopotential
        if geopotential:
            self._color_bar_format = False

    def _get_diagnostic(self, dimensional):

        if self._model_params.dynamic_T:
            offset = 1
        else:
            offset = 0

        vr = self._model_params.variables_range
        psi = np.swapaxes(self._data[:vr[0], ...].T @ np.swapaxes(self._grid_basis[offset:], 0, 1), 0, 1)

        factor = 1.

        if dimensional:
            factor *= self._model_params.streamfunction_scaling
            self._diagnostic_data_dimensional = True
            if self.geopotential:
                factor *= self._model_params.geopotential_scaling
        else:
            self._diagnostic_data_dimensional = False
        self._diagnostic_data = psi * factor
        return self._diagnostic_data


class OceanicStreamfunctionDiagnostic(FieldDiagnostic):
    """General base class for oceanic streamfunction fields diagnostic.
    Provide a spatial gridded representation of the fields.

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

        FieldDiagnostic.__init__(self, model_params, dimensional)
        self._configure(delta_x=delta_x, delta_y=delta_y)

        self._default_plot_kwargs['cmap'] = plt.get_cmap('jet')

    def _compute_grid(self, delta_x=None, delta_y=None):

        if delta_x is None:
            oms = self._model_params.oblocks
            if oms is None:
                warnings.warn("OceanicStreamfunctionDiagnostic: Unable to configure the grid automatically. Oceanic wavenumbers information not " +
                              "present in the model's parameters ! Please call the compute_grid method with the delta_x and delta_y parameters.")
                return 1
            xwn = [oms[i][0] for i in range(len(oms))]
            mxwn = max(xwn)
            n_point_x = 4 * mxwn + 2
        else:
            n_point_x = int(np.ceil((2 * np.pi / self._model_params.scale_params.n) / delta_x) + 1)

        if delta_y is None:
            oms = self._model_params.oblocks
            if oms is None:
                warnings.warn("OceanicStreamfunctionDiagnostic: Unable to configure the grid automatically. Oceanic wavenumbers information not " +
                              "present in the model's parameters ! Please call the compute_grid method with the delta_x and delta_y parameters.")
                return 1
            ywn = [oms[i][1] for i in range(len(oms))]
            mywn = max(ywn)
            n_point_y = 4 * mywn + 2
        else:
            n_point_y = int(np.ceil(np.pi / delta_y) + 1)

        x = np.linspace(0., 2 * np.pi / self._model_params.scale_params.n, n_point_x)
        y = np.linspace(0., np.pi, n_point_y)
        self._X, self._Y = np.meshgrid(x, y)

    def _configure(self, model_params=None, delta_x=None, delta_y=None):

        if not self._ocean:
            warnings.warn("OceanicStreamfunctionDiagnostic: No ocean configuration found in the provided parameters. This model version does not have an ocean. " +
                          "Please check your configuration.")
            return 1

        self._compute_grid(delta_x, delta_y)
        basis = self._model_params.oceanic_basis

        self._grid_basis = create_grid_basis(basis, self._X, self._Y, self._subs)


class OceanicLayerStreamfunctionDiagnostic(OceanicStreamfunctionDiagnostic):
    """Diagnostic giving the oceanic layer streamfunction fields :math:`\\psi_{\\rm o}`.

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
    conserved: bool, optional
        Whether to plot the conserved oceanic fields or not. Default to `True`.

    Attributes
    ----------

    dimensional: bool
        Indicate if the output diagnostic must be dimensionalized or not.
    """

    def __init__(self, model_params, delta_x=None, delta_y=None, dimensional=True, conserved=True):

        OceanicStreamfunctionDiagnostic.__init__(self, model_params, delta_x, delta_y, dimensional)

        vr = self._model_params.variables_range
        self._plot_title = r'Oceanic streamfunction'
        self._plot_units = r" (in " + self._model_params.get_variable_units(vr[1]) + r")"

        self._conserved = conserved

        self._fields_average = list()
        basis = self._model_params.oceanic_basis

        for func in basis.num_functions(self._subs):
            average = dblquad(func, 0, np.pi, 0, 2*np.pi/model_params.scale_params.n)[0] / (np.pi * 2*np.pi/model_params.scale_params.n)
            self._fields_average.append(average)

        self._fields_average = np.array(self._fields_average)

    def _get_diagnostic(self, dimensional):

        if self._model_params.dynamic_T:
            offset = 1
        else:
            offset = 0

        vr = self._model_params.variables_range
        if self._conserved:
            cgrid = np.swapaxes(np.swapaxes(self._grid_basis, 0, -1) - self._fields_average, 0, -1)
        else:
            cgrid = self._grid_basis
        psi = np.swapaxes(self._data[vr[1]:vr[2], ...].T @ np.swapaxes(cgrid[offset:], 0, 1), 0, 1)

        if dimensional:
            self._diagnostic_data = psi * self._model_params.streamfunction_scaling
            self._diagnostic_data_dimensional = True
        else:
            self._diagnostic_data = psi
            self._diagnostic_data_dimensional = False
        return self._diagnostic_data


if __name__ == '__main__':
    from qgs.params.params import QgParams
    from qgs.params.params import QgParams
    from qgs.integrators.integrator import RungeKuttaIntegrator
    from qgs.functions.tendencies import create_tendencies

    pars = QgParams()
    pars.set_atmospheric_channel_fourier_modes(2, 2)
    f, Df = create_tendencies(pars)
    integrator = RungeKuttaIntegrator()
    integrator.set_func(f)
    ic = np.random.rand(pars.ndim) * 0.1
    integrator.integrate(0., 200000., 0.1, ic=ic, write_steps=5)
    time, traj = integrator.get_trajectories()
    integrator.terminate()

    psi3 = LowerLayerAtmosphericStreamfunctionDiagnostic(pars)
    psi3(time, traj)

    psi1 = UpperLayerAtmosphericStreamfunctionDiagnostic(pars)
    psi1(time, traj)

    psi = MiddleAtmosphericStreamfunctionDiagnostic(pars)
    psi(time, traj)
