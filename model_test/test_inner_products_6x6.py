
import sys
import os

path = os.path.abspath('./')
base = os.path.basename(path)
if base == 'model_test':
    sys.path.extend([os.path.abspath('../')])
else:
    sys.path.extend([path])

import unittest
import numpy as np

from qgs.params.params import QgParams
from qgs.inner_products import analytic, symbolic

from model_test.test_base import TestBase

real_eps = np.finfo(np.float64).eps


class TestAnalyticInnerProducts6x6(TestBase):

    filename = 'test_inprod_analytic_6x6.ref'

    def test_inner_products(self, file=None):
        self.check_lists()
        if file is not None:
            self.write_reference_to_file(self.folder+file+'.ref')
            self.write_values_to_file(self.folder+file+'.val')

    def outputs(self, output_func=None):
        """This function print the coefficients computed in the inprod_analytic module"""

        if output_func is None:
            self.values.clear()
            tfunc = self.save_ip
        else:
            tfunc = output_func

        pars = QgParams()
        pars.set_atmospheric_channel_fourier_modes(6, 6)
        pars.set_oceanic_basin_fourier_modes(6, 6)

        # Setting MAOOAM default parameters
        pars.set_params({'kd': 0.04, 'kdp': 0.04, 'n': 1.5})

        aip = analytic.AtmosphericAnalyticInnerProducts(pars)
        oip = analytic.OceanicAnalyticInnerProducts(pars)
        aip.connect_to_ocean(oip)

        natm = pars.nmod[0]
        noc = pars.nmod[1]

        for i in range(natm):
            for j in range(natm):
                _ip_string_format(tfunc, "a", [i, j], aip.a(i, j))
                _ip_string_format(tfunc, "c", [i, j], aip.c(i, j))
                for k in range(natm):
                    _ip_string_format(tfunc, "b", [i, j, k], aip.b(i, j, k))
                    _ip_string_format(tfunc, "g", [i, j, k], aip.g(i, j, k))

            for j in range(noc):
                _ip_string_format(tfunc, "d", [i, j], aip.d(i, j))
                _ip_string_format(tfunc, "s", [i, j], aip.s(i, j))

        for i in range(noc):
            for j in range(noc):
                _ip_string_format(tfunc, "M", [i, j], oip.M(i, j))
                _ip_string_format(tfunc, "N", [i, j], oip.N(i, j))
                for k in range(noc):
                    _ip_string_format(tfunc, "O", [i, j, k], oip.O(i, j, k))
                    _ip_string_format(tfunc, "C", [i, j, k], oip.C(i, j, k))

            for j in range(natm):
                _ip_string_format(tfunc, "K", [i, j], oip.K(i, j))
                _ip_string_format(tfunc, "W", [i, j], oip.W(i, j))


# class TestSymbolicInnerProducts(TestBase):
#
#     filename = 'test_inprod_analytic_6x6.ref'
#
#     def test_inner_products(self, file=None):
#         self.check_lists()
#         if file is not None:
#             self.write_reference_to_file(self.folder+file+'.ref')
#             self.write_values_to_file(self.folder+file+'.val')
#
#     def outputs(self, output_func=None):
#         """This function print the coefficients computed in the inprod_analytic module"""
#
#         if output_func is None:
#             self.values.clear()
#             tfunc = self.save_ip
#         else:
#             tfunc = output_func
#
#         pars = QgParams()
#         pars.set_atmospheric_channel_fourier_modes(6, 6, mode='symbolic')
#         pars.set_oceanic_basin_fourier_modes(6, 6, mode='symbolic')
#
#         # Setting MAOOAM default parameters
#         pars.set_params({'kd': 0.04, 'kdp': 0.04, 'n': 1.5})
#
#         aip = symbolic.AtmosphericSymbolicInnerProducts(pars, quadrature=True)
#         oip = symbolic.OceanicSymbolicInnerProducts(pars, quadrature=True)
#
#         natm = pars.nmod[0]
#         noc = pars.nmod[1]
#
#         for i in range(natm):
#             for j in range(natm):
#                 _ip_string_format(tfunc, "a", [i, j], aip.a(i, j))
#                 _ip_string_format(tfunc, "c", [i, j], aip.c(i, j))
#                 for k in range(natm):
#                     _ip_string_format(tfunc, "b", [i, j, k], aip.b(i, j, k))
#                     _ip_string_format(tfunc, "g", [i, j, k], aip.g(i, j, k))
#
#             for j in range(noc):
#                 _ip_string_format(tfunc, "d", [i, j], aip.d(i, j))
#                 _ip_string_format(tfunc, "s", [i, j], aip.s(i, j))
#
#         for i in range(noc):
#             for j in range(noc):
#                 _ip_string_format(tfunc, "M", [i, j], oip.M(i, j))
#                 _ip_string_format(tfunc, "N", [i, j], oip.N(i, j))
#                 for k in range(noc):
#                     _ip_string_format(tfunc, "O", [i, j, k], oip.O(i, j, k))
#                     _ip_string_format(tfunc, "C", [i, j, k], oip.C(i, j, k))
#
#             for j in range(natm):
#                 _ip_string_format(tfunc, "K", [i, j], oip.K(i, j))
#                 _ip_string_format(tfunc, "W", [i, j], oip.W(i, j))


def _ip_string_format(func, symbol, indices, value):
    if abs(value) >= real_eps:
        s = symbol
        for i in indices:
            s += "[" + str(i + 1) + "]"
        s += " = % .5E" % value
        func(s)


if __name__ == "__main__":
    unittest.main()
