"""
    Diagnostic temperature classes
    =================================

    Classes defining temperature fields diagnostics.

    Description of the classes
    --------------------------

    * :class:`AtmosphericTemperatureDiagnostic`: General base class for atmospheric temperature fields diagnostics.
    * :class:`MiddleAtmosphericTemperatureDiagnostic`: Diagnostic giving the middle atmospheric anomaly fields :math:`T_{\\rm a}`.
    * :class:`MiddleAtmosphericTemperatureAnomalyDiagnostic`: Diagnostic giving the middle atmospheric temperature  anomaly fields :math:`\\delta T_{\\rm a}`.
    * :class:`OceanicTemperatureDiagnostic`: General base class for oceanic temperature fields diagnostic.
    * :class:`OceanicLayerTemperatureDiagnostic`: Diagnostic giving the oceanic layer temperature fields :math:`T_{\\rm o}`.
    * :class:`OceanicLayerTemperatureAnomalyDiagnostic`: Diagnostic giving the oceanic layer temperature anomaly fields :math:`\\delta T_{\\rm o}`.
    * :class:`GroundTemperatureDiagnostic`: Diagnostic giving the ground layer temperature fields :math:`T_{\\rm g}`.
    * :class:`GroundTemperatureAnomalyDiagnostic`: Diagnostic giving the ground layer temperature anomaly fields :math:`\\delta T_{\\rm g}`.
    * :class:`AtmosphericTemperatureMeridionalGradientDiagnostic`: General base class for meridional gradient of atmospheric temperature fields diagnostics.
    * :class:`MiddleAtmosphericTemperatureMeridionalGradientDiagnostic`: Diagnostic giving the meridional gradient of the middle atmospheric temperature fields :math:`\\partial_y T_{\\rm a}`.

"""
import warnings

import numpy as np
import matplotlib.pyplot as plt
from qgs.diagnostics.util import create_grid_basis

from qgs.diagnostics.base import FieldDiagnostic
from qgs.diagnostics.differential import DifferentialFieldDiagnostic


class AtmosphericTemperatureDiagnostic(FieldDiagnostic):
    """General base class for atmospheric temperature fields diagnostic.
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

        self._grid_basis = create_grid_basis(self._model_params.atmospheric_basis, self._X, self._Y, self._subs)
        if self._orography:
            if self._model_params.ground_params.orographic_basis == "atmospheric":
                self._oro_basis = self._grid_basis
            else:
                self._oro_basis = create_grid_basis(self._model_params.ground_basis, self._X, self._Y, self._subs)
        else:
            self._oro_basis = None


class MiddleAtmosphericTemperatureAnomalyDiagnostic(AtmosphericTemperatureDiagnostic):
    """Diagnostic giving the middle atmospheric temperature anomaly fields :math:`\\delta T_{\\rm a}` at 500hPa.
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

        vr = self._model_params.variables_range
        self._plot_title = r'Atmospheric 500hPa temperature anomaly'
        self._plot_units = r" (in " + self._model_params.get_variable_units(vr[0]) + r")"

        self._color_bar_format = False

    def _get_diagnostic(self, dimensional):

        if self._model_params.dynamic_T:
            offset = 1
        else:
            offset = 0

        vr = self._model_params.variables_range
        theta = np.swapaxes(self._data[vr[0]+offset:vr[1], ...].T @ np.swapaxes(self._grid_basis[offset:], 0, 1), 0, 1)

        if dimensional:
            self._diagnostic_data = theta * self._model_params.temperature_scaling * 2
            self._diagnostic_data_dimensional = True
        else:
            self._diagnostic_data = theta
            self._diagnostic_data_dimensional = False
        return self._diagnostic_data


class MiddleAtmosphericTemperatureDiagnostic(AtmosphericTemperatureDiagnostic):
    """Diagnostic giving the middle atmospheric temperature fields :math:`T_{\\rm a} = T_{{\\rm a}, 0} + \\delta T_{\\rm a}` at 500hPa,
    where :math:`T_{{\\rm a}, 0}` is the reference temperature :attr:`~.params.params.AtmosphericTemperatureParams.T0` or the 0-th order dynamic temperature.

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

    Notes
    -----

    Only works if the heat exchange scheme is activated, i.e. does not work with the Newton cooling scheme.

    """

    def __init__(self, model_params, delta_x=None, delta_y=None, dimensional=True):

        AtmosphericTemperatureDiagnostic.__init__(self, model_params, delta_x, delta_y, dimensional)

        vr = self._model_params.variables_range
        self._plot_title = r'Atmospheric 500hPa temperature'
        self._plot_units = r" (in " + self._model_params.get_variable_units(vr[0]) + r")"

        self._color_bar_format = False

        if not self._heat_exchange:
            warnings.warn('Heat exchange scheme is not activated, this diagnostic will not work properly !')

    def _get_diagnostic(self, dimensional):

        vr = self._model_params.variables_range
        T = np.swapaxes(self._data[vr[0]:vr[1], ...].T @ np.swapaxes(self._grid_basis, 0, 1), 0, 1)

        if dimensional:
            self._diagnostic_data = T * self._model_params.temperature_scaling * 2
            if not self._model_params.dynamic_T:
                self._diagnostic_data += self._model_params.atemperature_params.T0
            self._diagnostic_data_dimensional = True
        else:
            self._diagnostic_data = T
            if not self._model_params.dynamic_T:
                self._diagnostic_data += self._model_params.atemperature_params.T0 / (self._model_params.temperature_scaling * 2)
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
        self._default_plot_kwargs['cmap'] = plt.get_cmap('coolwarm')

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

        self._grid_basis = create_grid_basis(basis, self._X, self._Y, self._subs)


class OceanicLayerTemperatureAnomalyDiagnostic(OceanicTemperatureDiagnostic):
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

        vr = self._model_params.variables_range
        self._plot_title = r'Oceanic temperature anomaly'
        self._plot_units = r" (in " + self._model_params.get_variable_units(vr[2]) + r")"

        self._color_bar_format = False

    def _get_diagnostic(self, dimensional):

        if self._model_params.dynamic_T:
            offset = 1
        else:
            offset = 0

        vr = self._model_params.variables_range
        delta_T = np.swapaxes(self._data[vr[2]+offset:, ...].T @ np.swapaxes(self._grid_basis[offset:], 0, 1), 0, 1)

        if dimensional:
            self._diagnostic_data = delta_T * self._model_params.temperature_scaling
            self._diagnostic_data_dimensional = True
        else:
            self._diagnostic_data = delta_T
            self._diagnostic_data_dimensional = False
        return self._diagnostic_data


class OceanicLayerTemperatureDiagnostic(OceanicTemperatureDiagnostic):
    """Diagnostic giving the oceanic layer temperature fields :math:`T_{\\rm o} = T_{{\\rm o}, 0} + \\delta T_{\\rm o}`,
    where :math:`T_{{\\rm o}, 0}` is the reference temperature :attr:`~.params.params.OceanicTemperatureParams.T0` or the 0-th order dynamic temperature.

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

        vr = self._model_params.variables_range
        self._plot_title = r'Oceanic temperature'
        self._plot_units = r" (in " + self._model_params.get_variable_units(vr[2]) + r")"

        self._color_bar_format = False

    def _get_diagnostic(self, dimensional):

        vr = self._model_params.variables_range
        T = np.swapaxes(self._data[vr[2]:, ...].T @ np.swapaxes(self._grid_basis, 0, 1), 0, 1)

        if dimensional:
            self._diagnostic_data = T * self._model_params.temperature_scaling
            if not self._model_params.dynamic_T:
                self._diagnostic_data += self._model_params.gotemperature_params.T0
            self._diagnostic_data_dimensional = True
        else:
            self._diagnostic_data = T
            if not self._model_params.dynamic_T:
                self._diagnostic_data += self._model_params.gotemperature_params.T0 / self._model_params.temperature_scaling
            self._diagnostic_data_dimensional = False
        return self._diagnostic_data


class GroundTemperatureAnomalyDiagnostic(FieldDiagnostic):
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
        self._default_plot_kwargs['cmap'] = plt.get_cmap('coolwarm')

        vr = self._model_params.variables_range
        self._plot_title = r'Ground temperature anomaly'
        self._plot_units = r" (in " + self._model_params.get_variable_units(vr[1]) + r")"

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

        self._grid_basis = create_grid_basis(basis, self._X, self._Y, self._subs)

    def _get_diagnostic(self, dimensional):

        if self._model_params.dynamic_T:
            offset = 1
        else:
            offset = 0

        vr = self._model_params.variables_range
        delta_T = np.swapaxes(self._data[vr[1]+offset:, ...].T @ np.swapaxes(self._grid_basis[offset:], 0, 1), 0, 1)

        if dimensional:
            self._diagnostic_data = delta_T * self._model_params.temperature_scaling
            self._diagnostic_data_dimensional = True
        else:
            self._diagnostic_data = delta_T
            self._diagnostic_data_dimensional = False
        return self._diagnostic_data


class GroundTemperatureDiagnostic(GroundTemperatureAnomalyDiagnostic):
    """Diagnostic giving the ground temperature fields :math:`T_{\\rm g} = T_{{\\rm g}, 0} + \\delta T_{\\rm g}`,
    where :math:`T_{{\\rm g}, 0}` is the reference temperature :attr:`~.params.params.GroundTemperatureParams.T0` or the 0-th order dynamic temperature.

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

        GroundTemperatureAnomalyDiagnostic.__init__(self, model_params, delta_x, delta_y, dimensional)
        self._plot_title = r'Ground temperature'

    def _get_diagnostic(self, dimensional):

        vr = self._model_params.variables_range
        T = np.swapaxes(self._data[vr[1]:, ...].T @ np.swapaxes(self._grid_basis, 0, 1), 0, 1)

        if dimensional:
            self._diagnostic_data = T * self._model_params.temperature_scaling
            if not self._model_params.dynamic_T:
                self._diagnostic_data += self._model_params.gotemperature_params.T0
            self._diagnostic_data_dimensional = True
        else:
            self._diagnostic_data = T
            if not self._model_params.dynamic_T:
                self._diagnostic_data += self._model_params.gotemperature_params.T0 / self._model_params.temperature_scaling
            self._diagnostic_data_dimensional = False
        return self._diagnostic_data


class AtmosphericTemperatureMeridionalGradientDiagnostic(DifferentialFieldDiagnostic):
    """General base class for atmospheric temperature fields meridional gradient diagnostic.
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

        DifferentialFieldDiagnostic.__init__(self, model_params, dimensional)

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

    def _configure(self, delta_x=None, delta_y=None):

        basis = self._model_params.atmospheric_basis

        self._configure_differential_grid(basis, "dy", 1, delta_x, delta_y)

        if self._orography and self._X is not None and self._Y is not None:
            if self._model_params.ground_params.orographic_basis == "atmospheric":
                self._oro_basis = create_grid_basis(basis, self._X, self._Y, self._subs)
            else:
                self._oro_basis = create_grid_basis(self._model_params.ground_basis, self._X, self._Y, self._subs)
        else:
            self._oro_basis = None


class MiddleAtmosphericTemperatureMeridionalGradientDiagnostic(AtmosphericTemperatureMeridionalGradientDiagnostic):
    """Diagnostic giving the meridional gradient of the middle atmospheric temperature fields :math:`\\partial_y T_{\\rm a}` at 500hPa.
    It is identified with the meridional gradient of the baroclinic streamfunction :math:`\\partial_y \\theta_{\\rm a}` of the system.
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

        AtmosphericTemperatureMeridionalGradientDiagnostic.__init__(self, model_params, delta_x, delta_y, dimensional)
        vr = self._model_params.variables_range
        self._plot_title = r'Atmospheric 500hPa Temperature Meridional Gradient'
        self._plot_units = r" (in " + self._model_params.get_variable_units(vr[0]) + r")"

        self._color_bar_format = False

    def _get_diagnostic(self, dimensional):

        natm = self._model_params.nmod[0]
        theta = np.swapaxes(self._data[natm:2 * natm, ...].T @ np.swapaxes(self._grid_basis, 0, 1), 0, 1)

        if dimensional:
            self._diagnostic_data = theta * self._model_params.temperature_scaling * 2
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

    theta = MiddleAtmosphericTemperatureAnomalyDiagnostic(pars)
    theta(time, traj)

    dytheta = MiddleAtmosphericTemperatureMeridionalGradientDiagnostic(pars)
    dytheta(time, traj)
