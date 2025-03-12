
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
from qgs.inner_products import symbolic
from qgs.tensors.qgtensor import QgsTensor
from qgs.tensors.symbolic_qgtensor import SymbolicQgsTensor

from model_test.test_base_symbolic import TestBaseSymbolic

real_eps = np.finfo(np.float64).eps


class TestSymbolicGroundTensor(TestBaseSymbolic):
    '''
        Test class for the Linear Symbolic Tensor
        The tensor is tested against the reference file, and then the numerical tensor calculated in qgs.
    '''

    filename = 'test_aotensor.ref'

    # def test_sym_against_ref(self):
    #     self.check_lists_flt()
    
    def test_sym_against_num(self):
        self.check_numerical_lists_flt()

    def symbolic_outputs(self, output_func=None):

        if output_func is None:
            self.symbolic_values.clear()
            tfunc = self.save_ip_symbolic
        else:
            tfunc = output_func

        params = QgParams({'rr': 287.e0, 'sb': 5.6e-8})
        params.set_atmospheric_channel_fourier_modes(2, 2, mode="symbolic")
        params.set_ground_channel_fourier_modes(2, 2, mode="symbolic")

        params.ground_params.set_orography(0.2, 1)
        params.gotemperature_params.set_params({'gamma': 1.6e7, 'T0': 300})
        params.atemperature_params.set_params({'hlambda': 10, 'T0': 290})
        params.atmospheric_params.set_params({'sigma': 0.2, 'kd': 0.085, 'kdp': 0.02})
        C_g = 300
        params.atemperature_params.set_insolation(0.4 * C_g, 0)
        params.gotemperature_params.set_insolation(C_g, 0)

        aip = symbolic.AtmosphericSymbolicInnerProducts(params, return_symbolic=True, make_substitution=True)
        gip = symbolic.GroundSymbolicInnerProducts(params, return_symbolic=True, make_substitution=True)

        aip.connect_to_ground(gip, None)

        sym_aotensor = SymbolicQgsTensor(params=params, atmospheric_inner_products=aip, ground_inner_products=gip)
        
        subbed_tensor = sym_aotensor.sub_tensor()

        for coo, val in zip(subbed_tensor.keys(), subbed_tensor.values()):
            _ip_string_format(tfunc, 'sym_aotensor', coo, val)

    def numerical_outputs(self, output_func=None):

        if output_func is None:
            self.numerical_values.clear()
            tfunc = self.save_ip_numeric
        else:
            tfunc = output_func

        params = QgParams({'rr': 287.e0, 'sb': 5.6e-8})
        params.set_atmospheric_channel_fourier_modes(2, 2, mode="symbolic")
        params.set_ground_channel_fourier_modes(2, 2, mode="symbolic")

        params.ground_params.set_orography(0.2, 1)
        params.gotemperature_params.set_params({'gamma': 1.6e7, 'T0': 300})
        params.atemperature_params.set_params({'hlambda': 10, 'T0': 290})
        params.atmospheric_params.set_params({'sigma': 0.2, 'kd': 0.085, 'kdp': 0.02})
        C_g = 300
        params.atemperature_params.set_insolation(0.4 * C_g, 0)
        params.gotemperature_params.set_insolation(C_g, 0)

        aip = symbolic.AtmosphericSymbolicInnerProducts(params, return_symbolic=False)
        gip = symbolic.GroundSymbolicInnerProducts(params, return_symbolic=False)

        aip.connect_to_ground(gip, None)

        num_aotensor = QgsTensor(params=params, atmospheric_inner_products=aip, ground_inner_products=gip)

        for coo, val in zip(num_aotensor.tensor.coords.T, num_aotensor.tensor.data):
            _ip_string_format(tfunc, 'num_aotensor', coo, val)


def _ip_string_format(func, symbol, indices, value):
    if abs(value) >= real_eps:
        s = symbol
        for i in indices:
            s += "["+str(i)+"]"
        s += " = % .5E" % value
        func(s)


if __name__ == "__main__":
    unittest.main()
