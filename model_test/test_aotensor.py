
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

from params.params import QgParams
from inner_products import analytic
from tensors.qgtensor import QgsTensor

from model_test.test_base import TestBase


real_eps = np.finfo(np.float64).eps


class TestAoTensor(TestBase):

    filename = 'test_aotensor.ref'

    def test_aotensor(self, file=None):
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
        pars.set_atmospheric_modes(2, 2)
        pars.set_oceanic_modes(2, 4)

        # Setting MAOOAM default parameters
        pars.set_params({'kd': 0.04, 'kdp': 0.04, 'n': 1.5})

        aip = analytic.AtmosphericInnerProducts(pars)
        oip = analytic.OceanicInnerProducts(pars)
        aip.connect_to_ocean(oip)

        aotensor = QgsTensor(aip, oip)

        for coo, val in zip(aotensor.tensor.coords.T, aotensor.tensor.data):
            tfunc("aotensor[" + str(coo[0]) + "]" + "[" + str(coo[1]) + "]" + "[" + str(coo[2]) + "]" + " = % .5E" % val)


if __name__ == "__main__":
    unittest.main()
