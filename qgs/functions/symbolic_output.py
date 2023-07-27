import numpy as np
import sympy as sy

from qgs.functions.symbolic_mul import _symbolic_tensordot, symbolic_sparse_mult2, symbolic_sparse_mult3, symbolic_sparse_mult4, symbolic_sparse_mult5

from qgs.inner_products.symbolic import AtmosphericSymbolicInnerProducts, OceanicSymbolicInnerProducts, GroundSymbolicInnerProducts

from qgs.tensors.symbolic_qgtensor import SymbolicTensorLinear, SymbolicTensorDynamicT

def create_symbolic_equations(params, return_inner_products=False, return_symbolic_dict=False, return_symbolic_qgtensor=False):
    """
    Function to output the raw symbolic functions of the qgs model.
    """
    if params.atmospheric_basis is not None:
        aip = AtmosphericSymbolicInnerProducts(params, return_symbolic=True)
    else:
        aip = None

    if params.oceanic_basis is not None:
        oip = OceanicSymbolicInnerProducts(params, return_symbolic=True)
    else:
        oip = None

    if params.ground_basis is not None:
        gip = GroundSymbolicInnerProducts(params, return_symbolic=True)
    else:
        gip = None

    if aip is not None and oip is not None:
        if not aip.connected_to_ocean:
            aip.connect_to_ocean(oip)
    elif aip is not None and gip is not None:
        if not aip.connected_to_ground:
            aip.connect_to_ground(gip)

    if params.T4:
        raise ValueError("Symbolic tensor output not configured for T4 version, use Dynamic T version")
    elif params.dynamic_T:
        agotensor = SymbolicTensorDynamicT(params, aip, oip, gip)
    else:
        agotensor = SymbolicTensorLinear(params, aip, oip, gip)

    xx = list()
    xx.append(sy.Symbol('1'))

    for i in range(1, params.ndim+1):
        xx.append(sy.Symbol('U_'+str(i)))

    # Dont need to convert this to a sympy arr I think?
    # xx = sy.tensor.array.ImmutableSparseNDimArray(xx, shape=(len(xx)))

    if params.dynamic_T:
        func = symbolic_sparse_mult5(agotensor.tensor_dic, xx, xx, xx, xx)
        Dfunc = symbolic_sparse_mult4(agotensor.jac_dic, xx, xx, xx)

    else:
        func = symbolic_sparse_mult3(agotensor.tensor_dic, xx, xx)
        Dfunc = symbolic_sparse_mult2(agotensor.jac_dic, xx)
    
    ret = list()
    ret.append(func)
    ret.append(Dfunc)
    if return_inner_products:
        ret.append((aip, oip, gip))
    if return_symbolic_dict:
        ret.append(agotensor.tensor_dic)
    if return_symbolic_qgtensor:
        ret.append(agotensor.tensor)
    return ret


