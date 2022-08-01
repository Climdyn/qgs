
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
from qgs.inner_products import analytic
from qgs.tensors.qgtensor import QgsTensor

from model_test.test_base import TestBase


real_eps = np.finfo(np.float64).eps


class TestAnalyticAoTensorJacobian(TestBase):

    filename = 'test_aotensor_jacobian.ref'

    def test_aotensor_jacobian(self, file=None):
        self.check_lists()
        if file is not None:
            self.write_reference_to_file(self.folder+file+'.ref')
            self.write_values_to_file(self.folder+file+'.val')

    def outputs(self, output_func=None):
        """This function print the coefficients computed in the aotensor module"""

        if output_func is None:
            self.values.clear()
            tfunc = self.save_ip
        else:
            tfunc = output_func

        pars = QgParams({'rr': 287.e0, 'sb': 5.6e-8})
        pars.set_atmospheric_channel_fourier_modes(2, 2)
        pars.set_oceanic_basin_fourier_modes(2, 4)

        # Setting MAOOAM default parameters
        pars.set_params({'kd': 0.04, 'kdp': 0.04, 'n': 1.5})

        aip = analytic.AtmosphericAnalyticInnerProducts(pars)  # , stored=False)
        oip = analytic.OceanicAnalyticInnerProducts(pars)  # , stored=False)
        aip.connect_to_ocean(oip)

        aotensor = QgsTensor(pars, aip, oip)

        for coo, val in zip(aotensor.jacobian_tensor.coords.T, aotensor.jacobian_tensor.data):
            _ip_string_format(tfunc, 'jac_aotensor', coo, val)


def _ip_string_format(func, symbol, indices, value):
    if abs(value) >= real_eps:
        s = symbol
        for i in indices:
            s += "["+str(i)+"]"
        s += " = % .5E" % value
        func(s)


if __name__ == "__main__":
    unittest.main()
