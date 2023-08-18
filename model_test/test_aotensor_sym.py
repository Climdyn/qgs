
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
from qgs.tensors.symbolic_qgtensor import SymbolicTensorLinear

from model_test.test_base_symbolic import TestBaseSymbolic

real_eps = np.finfo(np.float64).eps

class TestSymbolicAOTensor(TestBaseSymbolic):
    '''
        Test class for the Linear Symbolic Tensor
        The tensor is tested against the reference file, and then the numerical tensor calculated in qgs.
    '''

    filename = 'test_aotensor.ref'

    def test_sym_against_ref(self):
        self.check_lists()
    
    def test_sym_against_num(self):
        self.check_numerical_lists()

    def symbolic_outputs(self, output_func=None):

        if output_func is None:
            self.symbolic_values.clear()
            tfunc = self.save_ip_symbolic
        else:
            tfunc = output_func

        params = QgParams({'rr': 287.e0, 'sb': 5.6e-8})
        params.set_atmospheric_channel_fourier_modes(2, 2, mode="symbolic")
        params.set_oceanic_basin_fourier_modes(2, 4, mode="symbolic")

        # Setting MAOOAM default parameters
        params.set_params({'kd': 0.04, 'kdp': 0.04, 'n': 1.5})

        aip = symbolic.AtmosphericSymbolicInnerProducts(params, return_symbolic=True, make_substitution=True)
        oip = symbolic.OceanicSymbolicInnerProducts(params, return_symbolic=True, make_substitution=True)

        aip.connect_to_ocean(oip)

        sym_aotensor = SymbolicTensorLinear(params=params, atmospheric_inner_products=aip, oceanic_inner_products=oip)
        
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
        params.set_oceanic_basin_fourier_modes(2, 4, mode="symbolic")

        # Setting MAOOAM default parameters
        params.set_params({'kd': 0.04, 'kdp': 0.04, 'n': 1.5})

        aip = symbolic.AtmosphericSymbolicInnerProducts(params, return_symbolic=False)
        oip = symbolic.OceanicSymbolicInnerProducts(params, return_symbolic=False)

        aip.connect_to_ocean(oip)

        num_aotensor = QgsTensor(params=params, atmospheric_inner_products=aip, oceanic_inner_products=oip)

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

# def test_tensor_numerically(self, tensor=None, dict_opp=True, tol=1e-10):
#             """
#             Uses sympy substitution to convert the symbolic tensor, or symbolic dictionary, to a numerical one.
#             This is then compared to the tensor calculated by the qgs.tensor.symbolic module.
            
#             """
#             ndim = self.params.ndim

#             if self.params.dynamic_T:
#                 if self.params.T4:

#                     raise ValueError("Symbolic tensor output not configured for T4 version, use Dynamic T version")
#                 else:
#                     dims = (ndim + 1, ndim + 1, ndim + 1, ndim + 1, ndim + 1)
#             else:
#                 dims = (ndim + 1, ndim + 1, ndim + 1)

#             _, _, numerical_tensor = create_tendencies(self.params, return_qgtensor=True)

#             if tensor is None:
#                 if dict_opp:
#                     tensor = self.tensor_dic
#                 else:
#                     tensor = self.tensor

#             subbed_ten = self.subs_tensor(tensor)
#             if isinstance(subbed_ten, dict):
#                 coords = np.array([list(k) for k in subbed_ten.keys()]).T
#                 data = np.array(list(subbed_ten.values()), dtype=float)
#                 subbed_tensor_sp = sp.COO(coords, data, shape=dims)
#             else:
#                 subbed_ten = np.array(subbed_ten)
#                 subbed_tensor_np = np.array(subbed_ten).astype(np.float64)
#                 subbed_tensor_sp = sp.COO.from_numpy(subbed_tensor_np)

#             diff_arr = subbed_tensor_sp.todense() - numerical_tensor.tensor.todense()


#             total_error = np.sum(np.abs(diff_arr))
#             max_error = np.max(np.abs(diff_arr))

#             if max_error > tol:
#                 self.print_tensor(diff_arr, tol=tol)
                
#                 raise ValueError("Symbolic tensor and numerical tensor do not match at the above coordinates, with a total error of: " + str(total_error))
            
#             elif total_error > tol:
#                 warnings.warn("Symbolic tensor and numerical tensor have a combined error of: " + str(total_error))
#             else:
#                 print("Tensor passes numerical test with a combined error of less than: " + str(tol))