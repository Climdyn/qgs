"""
    Diagnostic eddy classes
    =======================

    Classes defining eddy related fields diagnostics.

    Description of the classes
    --------------------------

    * :class:`MiddleLayerAtmosphericEddyHeatFluxDiagnostic`: Diagnostic giving the middle layer atmospheric eddy heat flux field.
    * :class:`MiddleLayerAtmosphericEddyHeatFluxProfileDiagnostic`: Diagnostic giving the middle layer atmospheric eddy heat flux zonally averaged profile.

"""

import warnings

import numpy as np
from scipy.integrate import simpson
import matplotlib.pyplot as plt

from qgs.diagnostics.base import FieldDiagnostic, ProfileDiagnostic
from qgs.diagnostics.temperatures import MiddleAtmosphericTemperatureDiagnostic
from qgs.diagnostics.wind import MiddleLayerAtmosphericVWindDiagnostic


class MiddleLayerAtmosphericEddyHeatFluxDiagnostic(FieldDiagnostic):
    """Diagnostic giving the middle layer atmospheric eddy heat flux field.
    Computed as :math:`\\Phi_{\\rm e} = \\bar v'_{\\rm a} \\, \\bar T'_{\\rm a}` and scaled with the
    atmospheric specific heat capicity if available (through the `heat_capacity` argument or the
    :attr:`~.AtmosphericTemperatureParams.gamma` parameter).

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
    temp_mean_state: MiddleLayerAtmosphericEddyHeatFluxDiagnostic, optional
        A temperature diagnostic with a long trajectory as data to compute the mean temperature field.
        If not provided, compute the mean with the data stored in the object.
    vwind_mean_state: MiddleLayerAtmosphericVWindDiagnostic, optional
        A :math:`v` wind diagnostic with a long trajectory as data to compute the mean wind field.
        If not provided, compute the mean with the data stored in the object.
    heat_capacity: float, optional
        The air specific heat capacity. If not provided, uses the one of :attr:`~.AtmosphericTemperatureParams.gamma` if
        available or or let the heat flux in K m s^{-1}.

    Attributes
    ----------

    dimensional: bool
        Indicate if the output diagnostic must be dimensionalized or not.

    """

    def __init__(self, model_params, delta_x=None, delta_y=None, dimensional=True, temp_mean_state=None, vwind_mean_state=None, heat_capacity=None):

        FieldDiagnostic.__init__(self, model_params, dimensional)

        self._plot_title = r'Atmospheric eddy heat flux in the middle layer'
        if heat_capacity is not None or model_params.atemperature_params.gamma is not None:
            self._plot_title += r" $\gamma_{\rm a} \bar v'_{\rm a} \, \bar T'_{\rm a}"
            self._plot_units = r" (in " + r'W m$^{-1}$' + r")"
        else:
            self._plot_title += r" $\bar v'_{\rm a} \, \bar T'_{\rm a}"
            self._plot_units = r" (in " + r'K m s$^{-1}$' + r")"
        self._default_plot_kwargs['cmap'] = plt.get_cmap('hsv_r')
        self._color_bar_format = False

        self._tdiag = MiddleAtmosphericTemperatureDiagnostic(model_params, delta_x, delta_y, dimensional)
        self._vdiag = MiddleLayerAtmosphericVWindDiagnostic(model_params, delta_x, delta_y, dimensional)

        self._X = self._tdiag._X
        self._Y = self._tdiag._Y

        self._temp_mean_state = temp_mean_state
        self._vwind_mean_state = vwind_mean_state

        self._heat_capacity = heat_capacity

    def _compute_grid(self, delta_x=None, delta_y=None):
        pass

    def _configure(self, delta_x=None, delta_y=None):
        pass

    def _get_diagnostic(self, dimensional):

        self._tdiag.set_data(self._time, self._data)
        self._vdiag.set_data(self._time, self._data)

        T = self._tdiag._get_diagnostic(dimensional)
        V = self._vdiag._get_diagnostic(dimensional)

        if self._temp_mean_state is not None:
            Tmean = self._temp_mean_state._get_diagnostic(dimensional).mean(axis=0)
        else:
            Tmean = np.mean(T, axis=0)
        if self._vwind_mean_state is not None:
            Vmean = self._vwind_mean_state._get_diagnostic(dimensional).mean(axis=0)
        else:
            Vmean = np.mean(V, axis=0)

        self._diagnostic_data = (T - Tmean) * (V - Vmean)
        if dimensional:
            if self._model_params.atemperature_params.gamma is not None:
                self._diagnostic_data = self._diagnostic_data * self._model_params.atemperature_params.gamma
            elif self._heat_capacity is not None:
                self._diagnostic_data = self._diagnostic_data * self._heat_capacity
            self._diagnostic_data_dimensional = True
        else:
            self._diagnostic_data_dimensional = False
        return self._diagnostic_data


class MiddleLayerAtmosphericEddyHeatFluxProfileDiagnostic(ProfileDiagnostic):
    """Diagnostic giving the middle layer atmospheric eddy heat flux zonally averaged profile.
    Computed as :math:`\\frac{n}{2\\pi} \\, \\int_0^{2\\pi/n} \\Phi_{\\rm e} \\, \\mathrm{d} x` where
    :math:`\\Phi_{\\rm e} = \\bar v'_{\\rm a} \\, \\bar T'_{\\rm a}` is the eddy heat flux scaled with the
    atmospheric specific heat capicity if available (through the `heat_capacity` argument or the
    :attr:`~.AtmosphericTemperatureParams.gamma` parameter).

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
    temp_mean_state: MiddleLayerAtmosphericEddyHeatFluxDiagnostic, optional
        A temperature diagnostic with a long trajectory as data to compute the mean temperature field.
        If not provided, compute the mean with the data stored in the object.
    vwind_mean_state: MiddleLayerAtmosphericVWindDiagnostic, optional
        A :math:`v` wind diagnostic with a long trajectory as data to compute the mean wind field.
        If not provided, compute the mean with the data stored in the object.
    heat_capacity: float, optional
        The air specific heat capacity. If not provided, uses the one of :attr:`~.AtmosphericTemperatureParams.gamma` if
        available or or let the heat flux in K m s^{-1}.

    Attributes
    ----------

    dimensional: bool
        Indicate if the output diagnostic must be dimensionalized or not.

    """

    def __init__(self, model_params, delta_x=None, delta_y=None, dimensional=True, temp_mean_state=None, vwind_mean_state=None, heat_capacity=None):

        ProfileDiagnostic.__init__(self, model_params, dimensional)

        self._flux = MiddleLayerAtmosphericEddyHeatFluxDiagnostic(model_params, delta_x, delta_y, dimensional, temp_mean_state, vwind_mean_state, heat_capacity)
        self._plot_title = r'Zonally averaged profile'
        self._plot_label = r'Atmospheric mid-layer eddy heat flux'
        if heat_capacity is not None or model_params.atemperature_params.gamma is not None:
            self._plot_label += r" $\gamma_{\rm a} \bar v'_{\rm a} \, \bar T'_{\rm a}"
            self._plot_units = r'W m$^{-1}$'
        else:
            self._plot_label += r" $\bar v'_{\rm a} \, \bar T'_{\rm a}"
            self._plot_units = r'K m s$^{-1}$'
        self._axis_label = r'$y$'
        self._configure()

    def _configure(self):
        self._points_coordinates = self._flux._Y[:, 0]

    def _get_diagnostic(self, dimensional):

        self._flux.set_data(self._time, self._data)

        flux = self._flux._get_diagnostic(dimensional)
        dX = self._flux._X[0, 1] - self._flux._X[0, 0]

        iflux = simpson(flux, dx=dX, axis=2) / (2*np.pi / self._model_params.scale_params.n)

        self._diagnostic_data = iflux
        if dimensional:
            self._diagnostic_data_dimensional = True
        else:
            self._diagnostic_data_dimensional = False
        return self._diagnostic_data


if __name__ == '__main__':
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

    flux = MiddleLayerAtmosphericEddyHeatFluxDiagnostic(pars)
    flux.set_data(time, traj)

    iflux = MiddleLayerAtmosphericEddyHeatFluxProfileDiagnostic(pars, delta_x=0.25, delta_y=0.15)
    iflux.set_data(time, traj)
