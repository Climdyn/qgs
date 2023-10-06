import numpy as np
import sympy as sy

import warnings
from qgs.functions.symbolic_mul import symbolic_sparse_mult2, symbolic_sparse_mult3, symbolic_sparse_mult4, symbolic_sparse_mult5
from qgs.inner_products.symbolic import AtmosphericSymbolicInnerProducts, OceanicSymbolicInnerProducts, GroundSymbolicInnerProducts
from qgs.tensors.symbolic_qgtensor import SymbolicTensorLinear, SymbolicTensorDynamicT, SymbolicTensorT4

import os

python_lang_translation = {
    'sqrt': 'math.sqrt',
    'pi': 'math.pi'
}

fortran_lang_translation = {
    '**': '^'
    #TODO: may need to add variable for pi
}

julia_lang_translation = {
    '**': '^',
    'pi': 'pi()',
    'conjugate': 'conj'
}

mathematica_lang_translation = {
    '**': '^'
}

def create_symbolic_equations(params, atm_ip=None, ocn_ip=None, gnd_ip=None, continuation_variables=list(), language='python', return_inner_products=False, return_jacobian=False, return_symbolic_eqs=False, return_symbolic_qgtensor=False):
    """
    Function to output the raw symbolic functions of the qgs model.

    Parameters
    ----------
    params: QGParams
        The parameters fully specifying the model configuration.
    atm_ip: SymbolicAtmosphericInnerProducts, optional
        Allows for stored inner products to be input
    ocn_ip: SymbolicOceanInnerProducts, optional
        Allows for stored inner products to be input
    gnd_ip: SymbolicGroundInnerProducts, optional
        Allows for stored inner products to be input
    continuation_variables: Iterable(Parameter, ScalingParameter, ParametersArray)
        The variables to not substitute and to leave in the equations, if None no variables are substituted
    language: String
        Options for the output language syntax: 'python', 'julia', 'fortran', 'auto', 'mathematica'
    return_inner_products: bool
        If True, return the inner products of the model. Default to False.
    return_jacobian: bool
        If True, return the Jacobian of the model. Default to False.
    return_symbolic_eqs: bool
        If True, return the substituted symbolic equations
    return_symbolic_qgtensor: bool
        If True, return the symbolic tendencies tensor of the model. Default to False.

    Returns
    -------
    funcs: string
        The substituted functions in the language syntax specified, as a string
    Deq_simplified: symbolic equations
        Dict of the substituted Jacobian matrix
    inner_products: (SymbolicAtmosphericInnerProducts, SymbolicOceanicInnerProducts, SymbolicGroundInnerProducts)
        If `return_inner_products` is True, the inner products of the system.
    eq_simplified: Symbolic equations
        If `return_symbolic_eqs` is True, Dict of the model tendencies symbolic functions
    agotensor: SymbolicQgsTensor
        If `return_symbolic_qgtensor` is True, the symbolic tendencies tensor of the system.

    """
    make_ip_subs = True

    if continuation_variables is None:
        make_ip_subs = False
    else:
        for cv in continuation_variables:
            try:
                if params.scale_params.n  == cv:
                    make_ip_subs = False
            except:
                pass


    if not(make_ip_subs):
        warnings.warn("Calculating innerproducts symbolically, as the variable 'n' has been specified as a variable, this takes several minutes.")

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
        eq = symbolic_sparse_mult5(agotensor.sub_tensor(continuation_variables=continuation_variables), xx, xx, xx, xx)
        if return_jacobian:
            Deq = symbolic_sparse_mult4(agotensor.sub_tensor(agotensor.jac_dic, continuation_variables=continuation_variables), xx, xx, xx)

    else:
        eq = symbolic_sparse_mult3(agotensor.sub_tensor(continuation_variables=continuation_variables), xx, xx)
        if return_jacobian:
            Deq = symbolic_sparse_mult2(agotensor.sub_tensor(agotensor.jac_dic, continuation_variables=continuation_variables), xx)

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

    func = equation_as_function(equations=eq_simplified, params=params, language=language, string_output=True, continuation_variables=continuation_variables)

    ret = list()
    ret.append('\n'.join(func))
    if return_jacobian:
        ret.append(Deq_simplified)
    if return_inner_products:
        ret.append((aip, oip, gip))
    if return_symbolic_eqs:
        ret.append(eq_simplified)
    if return_symbolic_qgtensor:
        ret.append(agotensor)
    return ret

def translate_equations(equations, language='python'):
    '''
    Function to output the model equations as a string in the specified language syntax.

    Parameters
    ----------
    equations: dict
        Dictinary of the symbolic model equations
    language: string
        Language syntax that the equations are returned in. Options are:
        - `python`
        - `fortran`
        - `julia`
        - `auto`
        - `mathematica`

    Returns
    -------
    str_eq: dict
        dict of strings of the model equations
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

def format_equations(equations, params, save_loc=None, language='python', print_equations=False):
    '''
    Function formats the equations, in the programming language specified, and saves the equations to the specified location.
    The variables in the equation are substituted if the model variable is input.

    Parameters
    ----------
    equations: Dict
        Dictionary of symbolic model equations
    params: QGParams
        qgs model params
    save_loc: String
        location to save the outputs as a .txt file
    language: string
        Language syntax that the equations are returned in. Options are:
        - `python`
        - `fortran`
        - `julia`
        - `auto`
        - `mathematica`
    free_variables: Set or list or None
        The variables to not substitute and to leave in the equations, if None no variables are substituted
    print_equations: bool
        If True, equations are printed by the function, if False, equation string is returned by the function. Defaults to False

    Returns
    -------
    equation_dict: Dict
        Dictionary of symbolic model equations, that have been substituted with numerical values
    
    free_vars: Set
        Set of strings of model variables that have not been substitued in this function, and remain as variabes in the equaitons.

    '''
    equation_dict = dict()

    # Substitute variable symbols
    vector_subs = dict()
    if language == 'python':
        for i in range(1, params.ndim+1):
            vector_subs['U_'+str(i)] = sy.Symbol('U['+str(i-1)+']')
    
    if language == 'fortran' or language == 'auto':
        for i in range(1, params.ndim+1):
            vector_subs['U_'+str(i)] = sy.Symbol('U('+str(i)+')')

    if language == 'julia':
        for i in range(1, params.ndim+1):
            vector_subs['U_'+str(i)] = sy.Symbol('U['+str(i)+']')

    if language == 'mathematica':
        for i in range(1, params.ndim+1):
            vector_subs['U_'+str(i)] = sy.Symbol('U('+str(i)+')')

    for k in equations.keys():
        eq = equations[k].subs(vector_subs)
        eq = eq.evalf()

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
        return equation_dict

def equation_as_function(equations, params, string_output=True, language='python', continuation_variables=list()):
    '''
    Converts the symbolic equations to a function in string format in the language syntax specified, or a lambdified python function 
    
    Parameters
    ----------
    equations: Dict
        Dictionary of the substituted symbolic model equations
    params: QGParams
        The parameters fully specifying the model configuration.
    string_output: bool
        If True, returns a lambdified python function, if False returns a string function, defaults to True
    free_variables: Set or List or None
        Variables that are not substituted with numerical values. If None, no symbols are substituted


    Returns
    -------
    f_output: lambdified python function, or String
        If string_output is True, output is a funciton in the specified language syntax, if False the output is a lambdified python function
    
    '''

    eq_dict = format_equations(equations, params, language=language)

    f_output = list()
    if language == 'python':
        if string_output:

            f_output.append('def f(t, U, **kwargs):')
            f_output.append('\t#Tendency function of the qgs model')
            f_output.append('\tF = np.empty_like(U)')

            for v in continuation_variables:
                f_output.append('\t' + str(v) + " = kwargs['" + str(v.symbol) + "']")

            for n, eq in enumerate(eq_dict.values()):
                f_output.append('\tF['+str(n)+'] = ' + str(eq))
            
            f_output.append('\treturn F')
        else:
            # Return a lamdafied function
            vec = [sy.Symbol('U['+str(i-1)+']') for i in range(1, params.ndim+1)]
            array_eqs = np.array(list(eq_dict.values()))
            inputs = ['t', vec]

            for v in continuation_variables:
                inputs.append(v.symbol)

            f_output = sy.lambdify(inputs, array_eqs)

    if language == 'julia':
        eq_dict = translate_equations(eq_dict, language='julia')

        f_output.append('function f!(du, U, p, t)')
        f_output.append('\t#Tendency function of the qgs model')

        for v in continuation_variables:
            f_output.append('\t' + str(v) + " = kwargs['" + str(v.symbol) + "']")

        for n, eq in enumerate(eq_dict.values()):
            f_output.append('\tdu['+str(n+1)+'] = ' + str(eq))
        
        f_output.append('end')

    if language == 'fortran':
        eq_dict = translate_equations(eq_dict, language='fortran')

        f_var = ''
        if len(continuation_variables) > 0:
            for fv in continuation_variables:
                f_var += str(fv.symbol) + ', '
            f_output.append('SUBROUTINE FUNC(NDIM, t, U, F, ' + f_var[:-2] + ')')
        else:
            f_output.append('SUBROUTINE FUNC(NDIM, t, U, F)')

        f_output.append('\t!Tendency function of the qgs model')
        f_output.append('\tINTEGER, INTENT(IN) :: NDIM')
        f_output.append('\tDOUBLE PRECISION, INTENT(IN) :: U(NDIM), PAR(*)')
        f_output.append('\tDOUBLE PRECISION, INTENT(OUT) :: F(NDIM)')

        for v in continuation_variables:
            f_output.append('\tDOUBLE PRECISION, INTENT(IN) :: ' + str(v.symbol))

        f_output.append('')

        f_output = _split_equations(eq_dict, f_output)
        
        f_output.append('END SUBROUTINE')

    if language == 'auto':
        eq_dict = translate_equations(eq_dict, language='fortran')

        eq_dict = _split_equations(eq_dict, f_output)
        create_auto_file(eq_dict, params, continuation_variables)
        
    if language == 'mathematica':
        #TODO: This function needs testing before release
        eq_dict = translate_equations(eq_dict, language='mathematica')

        f_output.append('F = Array[' + str(len(eq_dict)) + ']')

        for n, eq in enumerate(eq_dict.values()):
            f_output.append('F['+str(n+1)+'] = ' + str(eq))

        #TODO !!!! Killing output as I have not tested the above code !!!!
        f_output = None

    return f_output

def create_auto_file(equations, params, continuation_variables):
    '''
    Creates the auto configuration file and the model file.
    Saves files to specified folder.

    Parameters
    ----------
    equations: Dict
        Dictionary of the substituted symbolic model equations
    params: QGParams
        The parameters fully specifying the model configuration.
    continuation_variables: Iterable(Parameter, ScalingParameter, ParametersArray)
        Variables that are not substituted with numerical values. If None, no symbols are substituted

    '''

    #TODO: Find out best way to save these files
    #TODO: There is some weird double tab spacings in the output, and I am not sure why

    # User passes the equations, with the variables to leave as variables.
    # The existing model parameters are used to populate the auto file
    # The variables given as `continuation_variables` remain in the equations.
    # There is a limit of 1-10 remian variables
    base_path = os.path.dirname(__file__)
    base_file = '.modelproto'
    base_config = '.cproto'

    if (len(continuation_variables) < 1) or (len(continuation_variables) > 10):
        ValueError("Too many variables for auto file")

    # Declare variables
    declare_var = list()    
    for v in continuation_variables:
        declare_var.append('DOUBLE PRECISION ' + str(v.symbol))

    # make list of parameters
    var_list = list()
    var_ini = list()

    for i, v in enumerate(continuation_variables):

        temp_str = "PAR(" + str(i) + ") = " + str(v.symbol)

        initial_value = "PAR(" + str(i) + ") = " + str(v) + "   Variable: " + str(v.symbol)

        var_list.append(temp_str)
        var_ini.append(initial_value)

    ###### Writing model file ################

    # Open base file and input strings
    f_base = open(base_path + '/' + base_file, 'r')
    lines = f_base.readlines()
    f_base.close()

    auto_file = list()
    #TODO: Tabs not working here correctly
    for ln in lines:
        if 'PARAMETER DECLERATION' in ln:
            for dv in declare_var:
                auto_file.append('\t' + dv)
        elif 'CONTINUATION PARAMETERS' in ln:
            for v in var_list:
                auto_file.append('\t' + v)
        elif 'EVOLUTION EQUATIONS' in ln:
            for e in equations:
                auto_file.append('\t' + e)
        elif 'INITIALISE PARAMETERS' in ln:
            for iv in var_ini:
                auto_file.append('\t' + iv)
        else:
            auto_file.append(ln.replace('\n', ''))

    print('\n'.join(auto_file))
    
    ###### Writing config file ################

    c_base = open(base_path + '/' + base_config, 'r')
    lines = c_base.readlines()
    c_base.close()

    auto_config = list()
    for ln in lines:
        if '! PARAMETERS' in ln:
            auto_config.append('parnames = ' + str({i+1: v.symbol for i, v in enumerate(continuation_variables)}))

        elif '! VARIABLES' in ln:
            auto_config.append('unames = ' + str(_variable_names(params)))

        elif '! DIMENSION' in ln:
            auto_config.append('NDIM = ' + str(params.ndim))

        elif '! CONTINUATION ORDER' in ln:
            auto_config.append('ICP = ' + str([v.symbol for v in continuation_variables]))
        
        elif '! SOLUTION SAVE' in ln:
            auto_config.append("# ! User to input save locations")
            auto_config.append('UZR = ' + str({v.symbol: [] for v in continuation_variables}))

        elif '! STOP CONDITIONS' in ln:
            auto_config.append("# ! User to input variable bounds")
            auto_config.append('UZSTOP = ' + str({v.symbol: [] for v in continuation_variables}))
        
        else:
            auto_config.append(ln.replace('\n', ''))

    print('\n'.join(auto_config))

    return equations
    
def _split_equations(eq_dict, f_output, line_len=80):
    '''
        Function to split FORTRAN equaitons to a set length when producing functions
    '''
    for n, eq in enumerate(eq_dict.values()):
        # split equaitons to be a maximum of `line_len`
        
        #split remainder of equation into chunkcs of length `line_length`
        eq_chunks = [eq[x: x + line_len] for x in range(0, len(eq), line_len)]
        f_output.append('\tF('+str(n+1)+') =\t ' + eq_chunks[0] + "&")
        for ln in eq_chunks[1:-1]:
            f_output.append("\t\t&" + ln + "&")
        
        f_output.append("\t\t&" + eq_chunks[-1])
        f_output.append('')
    return f_output

def _variable_names(params):
    # Function to make the variable names for auto
    num_v = params.number_of_variables
    offset = 1 if params.number_of_variables else 0

    var_list = list()
    if params.atmospheric_basis is not None:
        for i in range(num_v[0]):
            var_list.append('psi' + str(i))
        
        for i in range(offset, num_v[1]+offset):
            var_list.append('theta' + str(i))
    
    if params.ground_basis is not None:
        for i in range(offset, num_v[2] + offset):
            var_list.append('gT' + str(i))
    
    if params.oceanic_basis is not None:
        for i in range(num_v[2]):
            var_list.append('A' + str(i))

        for i in range(offset, num_v[3] + offset):
            var_list.append('T' + str(i))
    
    output = dict()
    for i, v in enumerate(var_list):
        output[i+1] = v
    
    return output