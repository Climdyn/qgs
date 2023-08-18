import numpy as np
import sympy as sy

import warnings
from qgs.functions.symbolic_mul import symbolic_sparse_mult2, symbolic_sparse_mult3, symbolic_sparse_mult4, symbolic_sparse_mult5

from qgs.inner_products.symbolic import AtmosphericSymbolicInnerProducts, OceanicSymbolicInnerProducts, GroundSymbolicInnerProducts

from qgs.tensors.symbolic_qgtensor import SymbolicTensorLinear, SymbolicTensorDynamicT, SymbolicTensorT4

python_lang_translation = {
    'sqrt': 'math.sqrt',
    'pi': 'math.pi'
}

fortran_lang_translation = {
    '**': '^'
    #TODO: Is there a reason that sqrt is replaced with sq2 in auto? For computational speedup?
}

julia_lang_translation = {
    '**': '^',
    'pi': 'pi()'
}

mathematica_lang_translation = {
    '**': '^'
}

def create_symbolic_equations(params, atm_ip=None, ocn_ip=None, gnd_ip=None, continuation_variables={}, language='python', return_inner_products=False, return_jacobian=False, return_symbolic_eqs=False, return_symbolic_qgtensor=False):
    """
    Function to output the raw symbolic functions of the qgs model.
    """
    if 'n' in continuation_variables:
        make_ip_subs = False
        warnings.warn("Calculating innerproducts symbolically, as the variable 'n' has been specified as a variable, this takes several minutes.")
    else:
        make_ip_subs = True

    if params.atmospheric_basis is not None:
        if atm_ip is None:
            aip = AtmosphericSymbolicInnerProducts(params, return_symbolic=True, make_substitution=make_ip_subs)
        else: aip = atm_ip
    else:
        aip = None

    if params.oceanic_basis is not None:
        if ocn_ip is None:
            oip = OceanicSymbolicInnerProducts(params, return_symbolic=True, make_substitution=make_ip_subs)
        else:
            oip = ocn_ip
    else:
        oip = None

    if params.ground_basis is not None:
        if gnd_ip is None:
            gip = GroundSymbolicInnerProducts(params, return_symbolic=True, make_substitution=make_ip_subs)
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
        agotensor = SymbolicTensorT4(params, aip, oip, gip)
    elif params.dynamic_T:
        agotensor = SymbolicTensorDynamicT(params, aip, oip, gip)
    else:
        agotensor = SymbolicTensorLinear(params, aip, oip, gip)

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
    
    if continuation_variables is None:
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

    funcs = equation_as_function(equations=eq_simplified, params=params, language=language, string_output=True, remain_variables=continuation_variables)

    ret = list()
    ret.append('\n'.join(funcs))
    if return_jacobian:
        ret.append(Deq_simplified)
    if return_inner_products:
        ret.append((aip, oip, gip))
    if return_symbolic_eqs:
        ret.append(eq_simplified)
    if return_symbolic_qgtensor:
        ret.append(agotensor.tensor_dic)
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

    if language == 'fortran':
        translator = fortran_lang_translation

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

def format_equations(equations, params, save_loc=None, language='python', remain_variables={}, print_equations=False):
    '''
        Function formats the equations, in the programming language specified, and saves the equations to the specified location.
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
    # Substitute variables
    equation_dict = dict()
    if isinstance(remain_variables, (set, list, dict)):
        # make a dictionary of variables to substitute from parameters
        sub_vals = dict()
        for key in params.symbol_to_value.keys():
            if len(remain_variables) == 0:
                sub_vals[params.symbol_to_value[key][0]] = params.symbol_to_value[key][1]
            else:
                if key not in remain_variables:
                    sub_vals[params.symbol_to_value[key][0]] = params.symbol_to_value[key][1]

    elif remain_variables is None:
        sub_vals = None

    else:
        raise ValueError("Incorrect type for substitution, needs to be set, list, or dict of strings, not: " + str(type(remain_variables)))
    
    # Substitute variable symbols
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

    free_vars = set()
    for k in equations.keys():
        eq = equations[k].subs(vector_subs)

        #substitute syntax
        if sub_vals is not None:
            eq = eq.subs(sub_vals)
            eq = eq.evalf()
        else:
            eq = eq.simplify()
        for vars in eq.free_symbols:
            if vars not in vector_subs.values():
                free_vars.add(vars)

        if (language is not None) and print_equations:
            eq = translate_equations(eq, language)

        equation_dict[k] = eq
        
    if print_equations:
        if save_loc is None:
            for eq in equation_dict.values():
                if save_loc is None:
                    print(eq)
            else:
                with open(save_loc, 'w') as f:
                    for eq in equation_dict.values():
                        f.write("%s\n" % eq)
                print("Equations written")
    else:
        return equation_dict, free_vars

def equation_as_function(equations, params, string_output=False, language='python', remain_variables={}):

    eq_dict, free_vars = format_equations(equations, params, language=language, remain_variables=remain_variables)

    if language == 'python':
        if string_output:

            f_output = list()
            f_output.append('def f(t, U, **kwargs):')
            f_output.append('\t#Tendency function of the qgs model')
            f_output.append('\tF = np.empty_like(U)')
            for v in free_vars:
                f_output.append('\t' + str(v) + " = kwargs['" + str(v) + "']")

            for n, eq in enumerate(eq_dict.values()):
                f_output.append('\tF['+str(n)+'] = ' + str(eq))
            
            f_output.append('\treturn F')
        else:
            # Return a lamdafied function
            vec = [sy.Symbol('U['+str(i-1)+']') for i in range(1, params.ndim+1)]
            array_eqs = np.array(list(eq_dict.values()))
            inputs = ['t', vec]

            for v in free_vars:
                inputs.append(v)

            f_output = sy.lambdify(inputs, array_eqs)

    if language == 'julia':
        eq_dict = translate_equations(eq_dict, language='julia')

        f_output = list()
        f_output.append('function f(t, U, kwargs...)')
        f_output.append('\t#Tendency function of the qgs model')
        f_output.append('\tU_out = similar(U)')
        for n, eq in enumerate(eq_dict.values()):
            f_output.append('\tF['+str(n+1)+'] = ' + str(eq))
        
        f_output.append('\treturn F')
        f_output.append('end')

    #//TODO: Add statement for Fortran
    #//TODO: Add statement for mathematica

    return f_output

def equation_to_auto(equations, params, remain_variables=dict()):
    # User passes the equations, with the variables to leave as variables.
    # The existing model parameters are used to populate the auto file
    # The variables given as `remain_variables` remain in the equations.
    # There is a limit of 1-10 remian variables

    if (len(remain_variables) < 1) or (len(remain_variables) > 10):
        ValueError("Too many variables for auto file")

    str_equations, free_variables = format_equations(equations=equations, params=params, language='fortran', variables=True)

    natm, nog = params.nmod
    dim = params.ndim
    offset = 1 if params.dynamic_T else 0

    # make list of free variables
    var_list = list()
    for i, fv in enumerate(free_variables):
        temp_str = "PAR(" + str(i) + ") = " + str(fv)
        var_list.append(temp_str)

    
    