
import unittest
import numpy as np

from params.params import QgParams
from inner_products import analytic
from tensors.qgtensor import QgsTensor
from tensors.cootensor import from_csr_mat_list

from model_test.test_base import TestBase

real_eps = np.finfo(np.float64).eps


class TestAoTensor(TestBase):

    reference = list()
    values = list()
    folder = ""  # 'model_test/'
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
        pars.set_max_atmospheric_modes(2, 2)
        pars.set_max_oceanic_modes(2, 4)

        # Setting MAOOAM default parameters
        pars.set_params({'kd': 0.04, 'kdp': 0.04, 'n': 1.5})

        aip = analytic.AtmosphericInnerProducts(pars)
        oip = analytic.OceanicInnerProducts(pars)
        aip.connect_to_ocean(oip)

        aotensor = QgsTensor(aip, oip)

        coo_tensor = from_csr_mat_list(aotensor.tensor)

        for x in coo_tensor:
            tfunc("aotensor[" + str(x[0][0]) + "]" + "[" + str(x[0][1]) + "]" + "[" + str(x[0][2]) + "]" + " = % .5E" % x[1])


if __name__ == "__main__":
    unittest.main()
