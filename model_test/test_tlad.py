
from qgs.params.params import QgParams
from qgs.integrators.integrator import RungeKuttaIntegrator, RungeKuttaTglsIntegrator
from qgs.functions.tendencies import create_tendencies

import unittest
import numpy as np

real_eps = 1.e-3


class TestTlAd(unittest.TestCase):

    # Model parameters instantiation with some non-default specs
    model_parameters = QgParams({'phi0_npi': np.deg2rad(50.) / np.pi, 'hd': 0.3})
    # Mode truncation at the wavenumber 2 in both x and y spatial coordinate
    model_parameters.set_atmospheric_channel_fourier_modes(2, 2)

    # Changing (increasing) the orography depth and the meridional temperature gradient
    model_parameters.ground_params.set_orography(0.4, 1)
    model_parameters.atemperature_params.set_thetas(0.2, 0)

    f, Df = create_tendencies(model_parameters)

    integrator = RungeKuttaIntegrator()
    integrator.set_func(f)

    ic = np.random.rand(model_parameters.ndim) * 0.01
    integrator.integrate(0., 200000., 0.1, ic=ic, write_steps=0)
    _, ic = integrator.get_trajectories()

    tgls_integrator = RungeKuttaTglsIntegrator()
    tgls_integrator.set_func(f, Df)

    def test_taylor(self):

        for n in range(0, 7):

            y0 = self.ic
            dy = np.full_like(y0, 2. ** (-n)/np.sqrt(float(self.model_parameters.ndim)))
            y0prime = y0 + dy
            self.integrator.integrate(0., 0.1, 0.1, ic=y0, write_steps=0)
            _, y1 = self.integrator.get_trajectories()
            self.integrator.integrate(0., 0.1, 0.1, ic=y0prime, write_steps=0)
            _, y1prime = self.integrator.get_trajectories()

            dy1 = y1prime - y1

            self.tgls_integrator.integrate(0., 0.1, dt=0.1, write_steps=0, ic=y0, tg_ic=dy)
            _, _, dy1_tl = self.tgls_integrator.get_trajectories()

            print("Resulting difference in trajectory: (epsilon ~ 2^-", n, "= ", dy[0], ")")
            print("diff:    ", np.dot(dy1, dy1))
            print("tl:      ", np.dot(dy1_tl, dy1_tl))
            print("ratio:   ", np.dot(dy1, dy1)/np.dot(dy1_tl, dy1_tl))
            self.assertTrue(self.close_match(np.dot(dy1, dy1)/np.dot(dy1_tl, dy1_tl), 1., dy[0]/10))

    def test_adjoint_identity(self):

        y0 = self.ic
        for i in range(100):
            dy = np.random.randn(self.model_parameters.ndim)
            dy_bis = np.random.randn(self.model_parameters.ndim)

            # Calculate M(TL).x in dy1_tl
            self.tgls_integrator.integrate(0., 0.1, dt=0.1, write_steps=0, ic=y0, tg_ic=dy)
            _, _, dy1_tl = self.tgls_integrator.get_trajectories()

            # Calculate M(AD).x in dy1_ad
            self.tgls_integrator.integrate(0., 0.1, dt=0.1, write_steps=0, ic=y0, tg_ic=dy, adjoint=True)
            _, _, dy1_ad = self.tgls_integrator.get_trajectories()

            # Calculate M(AD).x in dy1_ad
            self.tgls_integrator.integrate(0., 0.1, dt=0.1, write_steps=0, ic=y0, tg_ic=dy_bis)
            _, _, dy1_bis_tl = self.tgls_integrator.get_trajectories()

            # Calculate M(AD).y in dy1_bis_ad
            self.tgls_integrator.integrate(0., 0.1, dt=0.1, write_steps=0, ic=y0, tg_ic=dy_bis, adjoint=True)
            _, _, dy1_bis_ad = self.tgls_integrator.get_trajectories()

            # Calculate norm < M(TL).x, y >
            norm1 = np.dot(dy1_tl, dy_bis)
            # Calculate norm < x, M(AD).y >
            norm2 = np.dot(dy, dy1_bis_ad)

            print("<M(TL).x,y> = ", norm1)
            print("<x,M(AD).y> = ", norm2)
            print("Ratio       = ", norm1 / norm2)
            self.assertTrue(self.close_match(norm1, norm2))

            # Calculate norm <M(TL).y,x>
            norm1 = np.dot(dy1_bis_tl,dy)
            # Calculate norm <y,M(AD).x>
            norm2 = np.dot(dy_bis,dy1_ad)

            print("<M(TL).y,x> = ", norm1)
            print("<y,M(AD).x> = ", norm2)
            print("Ratio       = ", norm1 / norm2)
            self.assertTrue(self.close_match(norm1, norm2))

    @staticmethod
    def close_match(v1, v2, eps=real_eps):
        return abs(v1 - v2) < eps
