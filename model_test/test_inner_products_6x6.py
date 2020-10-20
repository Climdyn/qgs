
# TODO: - Should be rewrited with a string composition function

import unittest
import numpy as np

from params.params import QgParams
from inner_products import analytic

from model_test.test_base import TestBase

real_eps = np.finfo(np.float64).eps


class TestAnalyticInnerProducts6x6(TestBase):

    reference = list()
    values = list()
    folder = ""  # 'model_test/'
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
        pars.set_atmospheric_modes(6, 6)
        pars.set_oceanic_modes(6, 6)

        # Setting MAOOAM default parameters
        pars.set_params({'k': 0.02, 'kp': 0.04, 'n': 1.5})

        aip = analytic.AtmosphericInnerProducts(pars)
        oip = analytic.OceanicInnerProducts(pars)
        aip.connect_to_ocean(oip)

        natm = pars.nmod[0]
        noc = pars.nmod[1]

        for i in range(natm):
            for j in range(natm):
                if abs(aip.a[i, j]) >= real_eps:
                    tfunc("a["+str(i+1)+"]"+"["+str(j+1)+"] = % .5E" % aip.a[i, j])
                if abs(aip.c[i, j]) >= real_eps:
                    tfunc("c["+str(i+1)+"]"+"["+str(j+1)+"] = % .5E" % aip.c[i, j])
                for k in range(0, natm):
                    if abs(aip.b[i, j, k]) >= real_eps:
                        tfunc(
                            "b["+str(i+1)+"]["+str(j+1)+"]["+str(k+1)+"] = % .5E"
                            % aip.b[i, j, k])
                    if abs(aip.g[i, j, k]) >= real_eps:

                        tfunc(
                            "g["+str(i+1)+"]["+str(j+1)+"]["+str(k+1)+"] = % .5E"
                            % aip.g[i, j, k])

        for i in range(natm):
            for j in range(0, noc):
                if abs(aip.d[i, j]) >= real_eps:
                    tfunc("d["+str(i+1)+"]"+"["+str(j+1)+"] = % .5E" % aip.d[i, j])
                if abs(aip.s[i, j]) >= real_eps:
                    tfunc("s["+str(i+1)+"]"+"["+str(j+1)+"] = % .5E" % aip.s[i, j])

        for i in range(noc):
            for j in range(noc):
                if abs(oip.M[i, j]) >= real_eps:
                    tfunc("M["+str(i+1)+"]"+"["+str(j+1)+"] = % .5E" % oip.M[i, j])
                if abs(oip.N[i, j]) >= real_eps:
                    tfunc("N["+str(i+1)+"]"+"["+str(j+1)+"] = % .5E" % oip.N[i, j])
                for k in range(noc):
                    if abs(oip.O[i, j, k]) >= real_eps:
                        tfunc(
                            "O["+str(i+1)+"]["+str(j+1)+"]["+str(k+1)+"] = % .5E"
                            % oip.O[i, j, k])
                    if abs(oip.C[i, j, k]) >= real_eps:
                        tfunc(
                            "C["+str(i+1)+"]["+str(j+1)+"]["+str(k+1)+"] = % .5E"
                            % oip.C[i, j, k])
            for j in range(natm):
                if abs(oip.K[i, j]) >= real_eps:
                    tfunc(
                        "K["+str(i+1)+"]"+"["+str(j+1)+"] = % .5E"
                        % oip.K[i, j])
                if abs(oip.W[i, j]) >= real_eps:
                    tfunc(
                        "W["+str(i+1)+"]" + "["+str(j+1)+"] = % .5E"
                        % oip.W[i, j])


if __name__ == "__main__":
    unittest.main()
