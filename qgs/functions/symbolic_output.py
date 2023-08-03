import numpy as np
import sympy as sy

from qgs.functions.symbolic_mul import _symbolic_tensordot, symbolic_sparse_mult2, symbolic_sparse_mult3, symbolic_sparse_mult4, symbolic_sparse_mult5

from qgs.inner_products.symbolic import AtmosphericSymbolicInnerProducts, OceanicSymbolicInnerProducts, GroundSymbolicInnerProducts

from qgs.tensors.symbolic_qgtensor import SymbolicTensorLinear, SymbolicTensorDynamicT

python_lang_translation = {
    'sqrt': 'math.sqrt',
    'pi': 'math.pi'
}

fortran_lang_translation = {
    '**': '^'
}

julia_lang_translation = {
    '**': '^',
    'pi': 'pi()'
}

mathematica_lang_translation = {
    '**': '^'
}

def create_symbolic_equations(params, atm_ip=None, ocn_ip=None, gnd_ip=None, return_inner_products=False, return_jacobian=False, return_symbolic_dict=False, return_symbolic_qgtensor=False):
    """
    Function to output the raw symbolic functions of the qgs model.
    """
    if params.atmospheric_basis is not None:
        if atm_ip is None:
            aip = AtmosphericSymbolicInnerProducts(params, return_symbolic=True)
        else: aip = atm_ip
    else:
        aip = None

    if params.oceanic_basis is not None:
        if ocn_ip is None:
            oip = OceanicSymbolicInnerProducts(params, return_symbolic=True)
        else:
            oip = ocn_ip
    else:
        oip = None

    if params.ground_basis is not None:
        if gnd_ip is None:
            gip = GroundSymbolicInnerProducts(params, return_symbolic=True)
        else:
            gip = gnd_ip
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
    # xx.append(sy.Symbol('1'))
    xx.append(1)

    for i in range(1, params.ndim+1):
        xx.append(sy.Symbol('U_'+str(i)))

    if params.dynamic_T:
        eq = symbolic_sparse_mult5(agotensor.tensor_dic, xx, xx, xx, xx)
        if return_jacobian:
            Deq = symbolic_sparse_mult4(agotensor.jac_dic, xx, xx, xx)

    else:
        eq = symbolic_sparse_mult3(agotensor.tensor_dic, xx, xx)
        if return_jacobian:
            Deq = symbolic_sparse_mult2(agotensor.jac_dic, xx)

    eq_simplified = dict()
    Deq_simplified = dict()
    
    #//TODO: This section using .simplify() is super slow.
    for i in range(1, params.ndim+1):
        eq_simplified[i] = eq[i].simplify()
        if return_jacobian:
            for j in range(1, params.ndim+1):
                if (i, j) in Deq:
                    Deq_simplified[(i, j)] = Deq[(i, j)].simplify()
    
    ret = list()
    ret.append(eq_simplified)
    if return_jacobian:
        ret.append(Deq_simplified)
    if return_inner_products:
        ret.append((aip, oip, gip))
    if return_symbolic_dict:
        ret.append(agotensor.tensor_dic)
    if return_symbolic_qgtensor:
        ret.append(agotensor.tensor)
    return ret

def translate_equations(equations, params, language='python'):
    '''
        Function to output the model equations as a string in the specified language. The following languages that this is set up for is:
        - Python
        - Fortran
        - Julia
        - Mathematica
    '''

    #//TODO: Finish this function
    _base_file_path = ''
    if language == 'python':
        file_ext = '.py'
        base_file = _base_file_path + file_ext

        # translate mathematical operations
        str_eq = dict()
        for i in range(1, params.ndim):
            temp_str = str(equations[i])
            #//TODO: This only works for single array, not jacobian
            for k in python_lang_translation.keys():
                temp_str = temp_str.replace(k, python_lang_translation[k])
            str_eq[i] = temp_str
    return str_eq