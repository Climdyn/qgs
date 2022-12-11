"""
    Diagnostic energy classes
    =========================

    Classes defining energy diagnostics.

    Description of the classes
    --------------------------

    * :class:`KineticEnergyDensityDiagnostic`: Diagnostic giving the kinetic energy density field :math:`(\\nabla \\psi^1_{\\rm a})^2 + (\\nabla \\psi^3_{\\rm a})^2`.

"""

import matplotlib.pyplot as plt

from qgs.diagnostics.base import FieldDiagnostic
from qgs.diagnostics.variables import VariablesDiagnostic
from qgs.diagnostics.wind import LowerLayerAtmosphericWindIntensityDiagnostic, UpperLayerAtmosphericWindIntensityDiagnostic


class KineticEnergyDensityDiagnostic(FieldDiagnostic):
    """Diagnostic giving the kinetic energy density field :math:`(\\nabla \\psi^1_{\\rm a})^2 + (\\nabla \\psi^3_{\\rm a})^2`.
    Computed using the :ref:`files/technical/diagnostics:Diagnostic wind classes`.

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

    def __init__(self, model_params, delta_x=None, delta_y=None, dimensional=False):
        FieldDiagnostic.__init__(self, model_params, dimensional)

        self._plot_title = r'Kinetic energy density'
        self._plot_units = r" (in " + r'kg s$^{-2}$' + r")"
        self._default_plot_kwargs['cmap'] = plt.get_cmap('magma')
        self._color_bar_format = False

        self._psi1_KE = UpperLayerAtmosphericWindIntensityDiagnostic(model_params, delta_x, delta_y, dimensional)
        self._psi3_KE = LowerLayerAtmosphericWindIntensityDiagnostic(model_params, delta_x, delta_y, dimensional)

        self._X = self._psi1_KE._X
        self._Y = self._psi1_KE._Y

    def _compute_grid(self, delta_x=None, delta_y=None):
        pass

    def _configure(self, delta_x=None, delta_y=None):
        pass

    def _get_diagnostic(self, dimensional):

        self._psi1_KE.set_data(self._time, self._data)
        self._psi3_KE.set_data(self._time, self._data)

        KE1 = self._psi1_KE._get_diagnostic(dimensional)**2
        KE3 = self._psi3_KE._get_diagnostic(dimensional)**2

        if dimensional:
            self._diagnostic_data = (KE1 * 0.394 + KE3 * 0.96) * self._model_params.scale_params.Ha
        else:
            self._diagnostic_data = KE1 + KE3

        self._diagnostic_data_dimensional = dimensional

        return self._diagnostic_data


class KineticEnergyDiagnostic(VariablesDiagnostic):

    def __init__(self, model_params, delta_x=None, delta_y=None, dimensional=False):

        VariablesDiagnostic.__init__(self, [0], model_params, dimensional)

        self._plt = 'KineticEnergy'
        self._plot_units = ' (in $s^{-2}$)'
        if self.dimensional:
            self._plot_title = self._plt + self._plot_units
        else:
            self._plot_title = self._plt

        self._variable_labels = ['']
        self._variable_units = ['']

        self._density = KineticEnergyDensityDiagnostic(model_params, delta_x, delta_y, dimensional)

    def _get_diagnostic(self, dimensional):

        self._density.set_data(self._time, self._data)

        kinetic_energy_density = self._density._get_diagnostic(dimensional)

        # integrate here


if __name__ == '__main__':
    from qgs.params.params import QgParams
    from qgs.integrators.integrator import RungeKuttaIntegrator
    from qgs.functions.tendencies import create_tendencies

    import numpy as np

    pars = QgParams()
    pars.set_atmospheric_channel_fourier_modes(2, 2)
    f, Df = create_tendencies(pars)
    integrator = RungeKuttaIntegrator()
    integrator.set_func(f)
    ic = np.random.rand(pars.ndim) * 0.1
    integrator.integrate(0., 200000., 0.1, ic=ic, write_steps=5)
    time, traj = integrator.get_trajectories()
    integrator.terminate()

    KE = KineticEnergyDensityDiagnostic(pars, dimensional=True)
    KE.set_data(time, traj)
