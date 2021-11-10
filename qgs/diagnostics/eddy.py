"""
    Diagnostic eddy classes
    =======================

    Classes defining eddy related fields diagnostics.

    Description of the classes
    --------------------------

    * :class:``: .

"""

import warnings

import numpy as np
import matplotlib.pyplot as plt

from qgs.diagnostics.base import FieldDiagnostic
from qgs.diagnostics.temperatures import MiddleAtmosphericTemperatureDiagnostic
from qgs.diagnostics.wind import MiddleLayerAtmosphericVWindDiagnostic


class MiddleLayerAtmosphericEddyFluxDiagnostic(FieldDiagnostic):

    def __init__(self, model_params, delta_x=None, delta_y=None, dimensional=True, temp_mean_state=None, vwind_mean_state=None):

        FieldDiagnostic.__init__(self, model_params, dimensional)

        self._plot_title = r'Atmospheric eddy flux in the middle layer'
        self._plot_units = r" (in " + r'K m s$^{-1}$' + r")"
        self._default_plot_kwargs['cmap'] = plt.get_cmap('hsv_r')
        self._color_bar_format = False

        self._tdiag = MiddleAtmosphericTemperatureDiagnostic(model_params, delta_x, delta_y, dimensional)
        self._vdiag = MiddleLayerAtmosphericVWindDiagnostic(model_params, delta_x, delta_y, dimensional)

        self._X = self._tdiag._X
        self._Y = self._vdiag._X

        self._temp_mean_state = temp_mean_state
        self._vwind_mean_state = vwind_mean_state

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
        return (T - Tmean) * (V - Vmean)


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

    flux = MiddleLayerAtmosphericEddyFluxDiagnostic(pars)
    flux.set_data(time, traj)
