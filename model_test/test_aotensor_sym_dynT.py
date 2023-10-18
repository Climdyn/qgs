
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
from qgs.tensors.qgtensor import QgsTensorDynamicT
from qgs.tensors.symbolic_qgtensor import SymbolicQgsTensorDynamicT

from model_test.test_base_symbolic import TestBaseSymbolic

real_eps = np.finfo(np.float64).eps

class TestSymbolicAOTensorDynT(TestBaseSymbolic):
    '''
        Test class for the Dynamic T Symbolic Tensor
        The tensor is tested against the reference file, and then the numerical tensor calculated in qgs.
    '''
    
    def test_sym_against_num(self):
        self.check_numerical_lists_flt()

    def symbolic_outputs(self, output_func=None):

        if output_func is None:
            self.symbolic_values.clear()
            tfunc = self.save_ip_symbolic
        else:
            tfunc = output_func

        params = QgParams({'rr': 287.e0, 'sb': 5.6e-8}, dynamic_T=True)
        params.set_atmospheric_channel_fourier_modes(2, 2, mode="symbolic")
        params.set_oceanic_basin_fourier_modes(2, 4, mode="symbolic")

        # Setting MAOOAM default parameters
        params.set_params({'kd': 0.04, 'kdp': 0.04, 'n': 1.5})

        aip = symbolic.AtmosphericSymbolicInnerProducts(params, return_symbolic=True, make_substitution=True)
        oip = symbolic.OceanicSymbolicInnerProducts(params, return_symbolic=True, make_substitution=True)

        aip.connect_to_ocean(oip)

        sym_aotensor = SymbolicQgsTensorDynamicT(params=params, atmospheric_inner_products=aip, oceanic_inner_products=oip)
        
        subbed_tensor = sym_aotensor.sub_tensor()

        for coo, val in zip(subbed_tensor.keys(), subbed_tensor.values()):
            _ip_string_format(tfunc, 'sym_aotensor', coo, val)

    def numerical_outputs(self, output_func=None):

        if output_func is None:
            self.numerical_values.clear()
            tfunc = self.save_ip_numeric
        else:
            tfunc = output_func

        params = QgParams({'rr': 287.e0, 'sb': 5.6e-8}, dynamic_T=True)
        params.set_atmospheric_channel_fourier_modes(2, 2, mode="symbolic")
        params.set_oceanic_basin_fourier_modes(2, 4, mode="symbolic")

        # Setting MAOOAM default parameters
        params.set_params({'kd': 0.04, 'kdp': 0.04, 'n': 1.5})

        aip = symbolic.AtmosphericSymbolicInnerProducts(params, return_symbolic=False)
        oip = symbolic.OceanicSymbolicInnerProducts(params, return_symbolic=False)

        aip.connect_to_ocean(oip)

        num_aotensor = QgsTensorDynamicT(params=params, atmospheric_inner_products=aip, oceanic_inner_products=oip)

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
