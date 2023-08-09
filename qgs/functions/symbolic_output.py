import numpy as np
import sympy as sy

from qgs.functions.symbolic_mul import _symbolic_tensordot, symbolic_sparse_mult2, symbolic_sparse_mult3, symbolic_sparse_mult4, symbolic_sparse_mult5

from qgs.inner_products.symbolic import AtmosphericSymbolicInnerProducts, OceanicSymbolicInnerProducts, GroundSymbolicInnerProducts

from qgs.tensors.symbolic_qgtensor import SymbolicTensorLinear, SymbolicTensorDynamicT, SymbolicTensorT4

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

def create_symbolic_equations(params, atm_ip=None, ocn_ip=None, gnd_ip=None, simplify=False, return_inner_products=False, return_jacobian=False, return_symbolic_dict=False, return_symbolic_qgtensor=False, numerically_test_tensor=True):
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
        agotensor = SymbolicTensorT4(params, aip, oip, gip, numerically_test_tensor)
    elif params.dynamic_T:
        agotensor = SymbolicTensorDynamicT(params, aip, oip, gip, numerically_test_tensor)
    else:
        agotensor = SymbolicTensorLinear(params, aip, oip, gip, numerically_test_tensor)

    xx = list()
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
    
    if simplify:
        # Simplifying at this step is slow
        # This only needs to be used if no substitutions are being made
        for i in range(1, params.ndim+1):
            eq_simplified[i] = eq[i].simplify()
            if return_jacobian:
                for j in range(1, params.ndim+1):
                    if (i, j) in Deq:
                        Deq_simplified[(i, j)] = Deq[(i, j)].simplify()
    else:
        eq_simplified = eq
        if return_jacobian:
            Deq_simplified = Deq
    
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

def translate_equations(equations, language='python'):
    '''
        Function to output the model equations as a string in the specified language. The languages that are availible to the user are:
        - Python
        - Fortran
        - Julia
        - Mathematica
    '''

    if language == 'python':
        translator = python_lang_translation

    if language == 'julia':
        translator = julia_lang_translation

    # translate mathematical operations
    if isinstance(equations, dict):
        str_eq = dict()
        for key in equations.keys():
            temp_str = str(equations[key])
            #//TODO: This only works for single array, not jacobian
            for k in translator.keys():
                temp_str = temp_str.replace(k, translator[k])
            str_eq[key] = temp_str
    else:
        temp_str = str(equations)
        for k in translator.keys():
            temp_str = temp_str.replace(k, translator[k])
        str_eq = temp_str

    return str_eq

def print_equations(equations, params, save_loc=None, language='python', variables=True, remain_variables=dict(), return_equations=False):
    '''
        Function prints the equations as strings, in the programming language specified, and saves the equations to the specified location.
        The variables in the equation are substituted if the model variable is input.

        Parameters
        ----------
        equations: dictionary of symbolic expressions

        params: qgs model params

        save_loc: String, location to save the outputs as a .txt file

        language: String, programming language to output the strings as

        variables: Set or list of Strings, dict of sympy.Symbol with corrisponding values, or Bool
            
            If a set or list of strings is input, the corrisponding value in the parameters is found

            if a dict of sympy.Symbols is input, this is used to substitute the tensor

            if True is passed, the parameters are used to substitute all variables
        
        remain_variables: Set or list of strings
            A list or set of variables not to substitute. Only is used when variables is set to True.
        

    '''
    if equations is not None:
        equation_dict = dict()
        if variables is not None:
            # make a dictionary of variables to substitute from parameters
            sub_vals = dict()
            if variables == True:
                for key in params.symbol_to_value.keys():
                    if key not in remain_variables:
                        sub_vals[params.symbol_to_value[key][0]] = params.symbol_to_value[key][1]
            else:
                if isinstance(variables, (set, list, dict)):
                    for s in variables:
                        if isinstance(s, str):
                            temp_sym, val = params.symbol_to_value[s]
                        elif isinstance(s, sy.Symbol):
                            temp_sym = s
                            val = params.symbol_to_value[temp_sym]
                        else:
                            raise ValueError("Incorrect type for substitution, needs to be string or sympy.Symbol, not: " + str(type(s)))
                        sub_vals[temp_sym] = val
                else:
                    raise ValueError("Incorrect type for substitution, needs to be list, set, or dictionary, not: " + str(type(variables)))
            

        for k in equations.keys():
            eq = equations[k]
            if sub_vals is not None:
                eq = eq.subs(sub_vals)
                eq = eq.evalf()

            if (language is not None) and not(return_equations):
                eq = translate_equations(eq, language)
            
            if save_loc is None:
                equation_dict[k] = eq
            else:
                equation_dict[k] = str(eq)
        
        if return_equations:
            return equation_dict
        else:
            if save_loc is None:
                for eq in equation_dict.values():
                    if save_loc is None:
                        print(eq)
                else:
                    with open(save_loc, 'w') as f:
                        for eq in equation_dict.values():
                            f.write("%s\n" % eq)
                    print("Equations written")

def equation_as_function(equations, params, string_output=False, language='python', variables=True, remain_variables=dict()):
    
    vector_subs = dict()
    if language == 'python':
        for i in range(1, params.ndim+1):
            vector_subs['U_'+str(i)] = sy.Symbol('U['+str(i-1)+']')
    
    if language == 'fortran':
        for i in range(1, params.ndim+1):
            vector_subs['U_'+str(i)] = sy.Symbol('U('+str(i)+')')

    if language == 'julia':
        for i in range(1, params.ndim+1):
            vector_subs['U_'+str(i)] = sy.Symbol('U['+str(i)+']')

    if language == 'mathematica':
        for i in range(1, params.ndim+1):
            vector_subs['U_'+str(i)] = sy.Symbol('U('+str(i)+')')
        

    subed_eq = dict()
    for k in equations.keys():
        subed_eq[k] = equations[k].subs(vector_subs)

    eq_list = print_equations(subed_eq, params, language=language, variables=variables, remain_variables=remain_variables, return_equations=True)
    
    # Calculate the free variables
    free_vars = set()
    for func in eq_list.values():
        for vars in func.free_symbols:
            if vars not in vector_subs.values():
                free_vars.add(vars)

    if language == 'python':
        if string_output:

            f_output = list()
            f_output.append('def f(t, U, **kwargs):')
            f_output.append('\t#Tendency function of the qgs model')
            f_output.append('\tU_out = np.empty_like(U)')
            for v in free_vars:
                f_output.append('\t' + str(v) + " = kwargs['" + str(v) + "']")

            for n, eq in enumerate(eq_list.values()):
                f_output.append('\tU_out['+str(n)+'] = ' + str(eq))
            
            f_output.append('\treturn U_out')
        else:
            # Return a lamdafied function
            vec = [sy.Symbol('U['+str(i-1)+']') for i in range(1, params.ndim+1)]
            f_output = sy.lambdify([vec], eq_list)

    if language == 'julia':
        eq_list = translate_equations(eq_list, language='julia')

        f_output = list()
        f_output.append('function f(t, U, kwargs...)')
        f_output.append('\t#Tendency function of the qgs model')
        f_output.append('\tU_out = similar(U)')
        for n, eq in enumerate(eq_list.values()):
            f_output.append('\tU_out['+str(n+1)+'] = ' + str(eq))
        
        f_output.append('\treturn U_out')
        f_output.append('end')

    #//TODO: Add statement for Fortran
    #//TODO: Add statement for mathematica

    return f_output