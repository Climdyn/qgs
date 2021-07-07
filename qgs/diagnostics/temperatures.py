"""
    Diagnostic temperature classes
    =================================

    Classes defining temperature fields diagnostics.

    Description of the classes
    --------------------------

    * :class:`AtmosphericTemperatureDiagnostic`: General base class for atmospheric temperature fields diagnostic.
    * :class:`MiddleAtmosphericTemperatureDiagnostic`: Diagnostic giving the middle atmospheric temperature  anomaly fields :math:`\\delta T_{\\rm a}`.
    * :class:`OceanicTemperatureDiagnostic`: General base class for oceanic temperature fields diagnostic.
    * :class:`OceanicLayerTemperatureDiagnostic`: Diagnostic giving the oceanic layer temperature anomaly fields :math:`\\delta T_{\\rm o}`.
    * :class:`GroundTemperatureDiagnostic`: Diagnostic giving the ground layer temperature anomaly fields :math:`\\delta T_{\\rm g}`.

"""
import warnings

import matplotlib.pyplot as plt
import numpy as np

from qgs.diagnostics.base import FieldDiagnostic


class AtmosphericTemperatureDiagnostic(FieldDiagnostic):
    """General base class for atmospheric temperature fields diagnostic.
    Provide a spatial gridded representation of the fields.
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

    def __init__(self, model_params, delta_x=None, delta_y=None, dimensional=True):

        FieldDiagnostic.__init__(self, model_params, dimensional)
        self._configure(delta_x=delta_x, delta_y=delta_y)
        self._default_plot_kwargs['cmap'] = plt.get_cmap('coolwarm')

    def _compute_grid(self, delta_x=None, delta_y=None):

        if delta_x is None:
            ams = self._model_params.ablocks
            if ams is None:
                warnings.warn("AtmosphericTemperatureDiagnostic: Unable to configure the grid automatically. Atmospheric wavenumbers information not " +
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
                warnings.warn("AtmosphericTemperatureDiagnostic: Unable to configure the grid automatically. Atmospheric wavenumbers information not " +
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

    def _configure(self, model_params=None, delta_x=None, delta_y=None):

        self._compute_grid(delta_x, delta_y)
        basis = self._model_params.atmospheric_basis

        self._grid_basis = list()

        for func in basis.num_functions():
            self._grid_basis.append(func(self._X, self._Y))

        self._grid_basis = np.array(self._grid_basis)


class MiddleAtmosphericTemperatureDiagnostic(AtmosphericTemperatureDiagnostic):
    """Diagnostic giving the middle atmospheric temperature  anomaly fields :math:`\\delta T_{\\rm a}` at 500hPa.
    It is identified with the baroclinic streamfunction :math:`\\theta_{\\rm a}` of the system.
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

    Attributes
    ----------

    dimensional: bool
        Indicate if the output diagnostic must be dimensionalized or not.
    """

    def __init__(self, model_params, delta_x=None, delta_y=None, dimensional=True):

        AtmosphericTemperatureDiagnostic.__init__(self, model_params, delta_x, delta_y, dimensional)

        natm = self._model_params.nmod[0]
        self._plot_title = r'Atmospheric 500hPa temperature anomaly'
        self._plot_units = r" (in " + self._model_params.get_variable_units(natm) + r")"

        self._color_bar_format = False

    def _get_diagnostic(self, dimensional):

        natm = self._model_params.nmod[0]
        theta = np.swapaxes(self._data[natm:2*natm, ...].T @ np.swapaxes(self._grid_basis, 0, 1), 0, 1)

        if dimensional:
            self._diagnostic_data = theta * self._model_params.temperature_scaling * 2
            self._diagnostic_data_dimensional = True
        else:
            self._diagnostic_data = theta
            self._diagnostic_data_dimensional = False
        return self._diagnostic_data


class OceanicTemperatureDiagnostic(FieldDiagnostic):
    """General base class for atmospheric temperature fields diagnostic.
    Provide a spatial gridded representation of the fields.
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

    def __init__(self, model_params, delta_x=None, delta_y=None, dimensional=True):

        FieldDiagnostic.__init__(self, model_params, dimensional)
        self._configure(delta_x=delta_x, delta_y=delta_y)

    def _compute_grid(self, delta_x=None, delta_y=None):

        if delta_x is None:
            oms = self._model_params.oblocks
            if oms is None:
                warnings.warn("OceanicTemperatureDiagnostic: Unable to configure the grid automatically. Oceanic wavenumbers information not " +
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
                warnings.warn("OceanicTemperatureDiagnostic: Unable to configure the grid automatically. Oceanic wavenumbers information not " +
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
            warnings.warn("OceanicTemperatureDiagnostic: No ocean configuration found in the provided parameters. This model version does not have an ocean. " +
                          "Please check your configuration.")
            return 1

        self._compute_grid(delta_x, delta_y)
        basis = self._model_params.oceanic_basis

        self._grid_basis = list()

        for func in basis.num_functions():
            self._grid_basis.append(func(self._X, self._Y))

        self._grid_basis = np.array(self._grid_basis)


class OceanicLayerTemperatureDiagnostic(OceanicTemperatureDiagnostic):
    """Diagnostic giving the oceanic layer temperature anomaly fields :math:`\\delta T_{\\rm o}`.

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

        OceanicTemperatureDiagnostic.__init__(self, model_params, delta_x, delta_y, dimensional)

        natm = self._model_params.nmod[0]
        noc = self._model_params.nmod[1]
        self._plot_title = r'Oceanic temperature anomaly'
        self._plot_units = r" (in " + self._model_params.get_variable_units(2 * natm + noc) + r")"

        self._color_bar_format = False

    def _get_diagnostic(self, dimensional):

        natm = self._model_params.nmod[0]
        noc = self._model_params.nmod[1]
        theta = np.swapaxes(self._data[2*natm+noc:, ...].T @ np.swapaxes(self._grid_basis, 0, 1), 0, 1)

        if dimensional:
            self._diagnostic_data = theta * self._model_params.temperature_scaling
            self._diagnostic_data_dimensional = True
        else:
            self._diagnostic_data = theta
            self._diagnostic_data_dimensional = False
        return self._diagnostic_data


class GroundTemperatureDiagnostic(FieldDiagnostic):
    """Diagnostic giving the ground temperature anomaly fields :math:`\\delta T_{\\rm g}`.

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

        FieldDiagnostic.__init__(self, model_params, dimensional)
        self._configure(delta_x=delta_x, delta_y=delta_y)

        natm = self._model_params.nmod[0]
        self._plot_title = r'Ground temperature anomaly'
        self._plot_units = r" (in " + self._model_params.get_variable_units(2 * natm) + r")"

        self._color_bar_format = False

    def _compute_grid(self, delta_x=None, delta_y=None):

        if delta_x is None:
            gms = self._model_params.gblocks
            if gms is None:
                warnings.warn("GroundTemperatureDiagnostic: Unable to configure the grid automatically. Ground wavenumbers information not " +
                              "present in the model's parameters ! Please call the compute_grid method with the delta_x and delta_y parameters.")
                return 1
            xwn = [gms[i][0] for i in range(len(gms))]
            mxwn = max(xwn)
            n_point_x = 4 * mxwn + 2
        else:
            n_point_x = int(np.ceil((2 * np.pi / self._model_params.scale_params.n) / delta_x) + 1)

        if delta_y is None:
            gms = self._model_params.gblocks
            if gms is None:
                warnings.warn("GroundTemperatureDiagnostic: Unable to configure the grid automatically. Ground wavenumbers information not " +
                              "present in the model's parameters ! Please call the compute_grid method with the delta_x and delta_y parameters.")
                return 1
            ywn = [gms[i][1] for i in range(len(gms))]
            mywn = max(ywn)
            n_point_y = 4 * mywn + 2
        else:
            n_point_y = int(np.ceil(np.pi / delta_y) + 1)

        x = np.linspace(0., 2 * np.pi / self._model_params.scale_params.n, n_point_x)
        y = np.linspace(0., np.pi, n_point_y)
        self._X, self._Y = np.meshgrid(x, y)

    def _configure(self, model_params=None, delta_x=None, delta_y=None):

        if not self._ground:
            warnings.warn("GroundTemperatureDiagnostic: No ground configuration found in the provided parameters. This model version does not have a ground. " +
                          "Please check your configuration.")
            return 1

        self._compute_grid(delta_x, delta_y)
        basis = self._model_params.ground_basis

        self._grid_basis = list()

        for func in basis.num_functions():
            self._grid_basis.append(func(self._X, self._Y))

        self._grid_basis = np.array(self._grid_basis)

    def _get_diagnostic(self, dimensional):

        natm = self._model_params.nmod[0]
        theta = np.swapaxes(self._data[2*natm:, ...].T @ np.swapaxes(self._grid_basis, 0, 1), 0, 1)

        if dimensional:
            self._diagnostic_data = theta * self._model_params.temperature_scaling
            self._diagnostic_data_dimensional = True
        else:
            self._diagnostic_data = theta
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

    theta = MiddleAtmosphericTemperatureDiagnostic(pars)
    theta(time, traj)
