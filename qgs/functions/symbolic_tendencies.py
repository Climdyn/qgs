"""
    Symbolic tendencies module
    ==========================

    This module provides functions to create a symbolic representation of the tendencies functions of the model
    in various languages and for various external software.

"""

from sympy import Symbol

import warnings
from qgs.functions.symbolic_mul import symbolic_sparse_mult2, symbolic_sparse_mult3, symbolic_sparse_mult4, \
    symbolic_sparse_mult5
from qgs.inner_products.symbolic import AtmosphericSymbolicInnerProducts, OceanicSymbolicInnerProducts, \
    GroundSymbolicInnerProducts
from qgs.tensors.symbolic_qgtensor import SymbolicQgsTensor, SymbolicQgsTensorDynamicT, SymbolicQgsTensorT4

python_lang_translation = {
    'sqrt': 'math.sqrt',
    'lambda': 'lmda'  # Remove conflict for lambda function in python
}

fortran_lang_translation = {
    'conjugate': 'CONJG',
    'epsilon': 'eps'  # Remove conflict for EPSILON function in fortran
}

julia_lang_translation = {
    '**': '^',
    'conjugate': 'conj'
}

mathematica_lang_translation = {
    '**': '^'
}


def create_symbolic_tendencies(params, continuation_variables, atm_ip=None, ocn_ip=None, gnd_ip=None,
                               language='python', return_inner_products=False, return_jacobian=False,
                               return_symbolic_eqs=False, return_symbolic_qgtensor=False):
    """Function to output the raw symbolic tendencies of the qgs model.

    Parameters
    ----------
    params: QgParams
        The parameters fully specifying the model configuration.
    continuation_variables: list(Parameter, ScalingParameter or ParametersArray)  or None
        The variables to not substitute by their numerical value and to leave in the equations.
        If `None`, no variables are substituted.
        If an empty list is provided, then all variables are substituted, providing fully numerical tendencies.
    atm_ip: AtmosphericSymbolicInnerProducts, optional
        Allows for stored inner products to be input.
    ocn_ip: OceanSymbolicInnerProducts, optional
        Allows for stored inner products to be input.
    gnd_ip: GroundSymbolicInnerProducts, optional
        Allows for stored inner products to be input.
    language: str, optional
        Options for the output language syntax: 'python', 'julia', 'fortran', 'auto', 'mathematica'.
        Default to 'python'.
    return_inner_products: bool, optional
        If `True`, return the inner products of the model. Default to `False`.
    return_jacobian: bool, optional
        If `True`, return the Jacobian of the model. Default to `False`.
    return_symbolic_eqs: bool, optional
        If `True`, return the substituted symbolic equations.
    return_symbolic_qgtensor: bool, optional
        If `True`, return the symbolic tendencies tensor of the model. Default to `False`.

    Returns
    -------
    funcs: str
        The substituted functions in the language syntax specified, as a string.
    dict_eq_simplified: dict(~sympy.core.expr.Expr)
        Dictionary of the substituted Jacobian matrix.
    inner_products: tuple(AtmosphericSymbolicInnerProducts, OceanicSymbolicInnerProducts, GroundSymbolicInnerProducts)
        If `return_inner_products` is `True`, the inner products of the system.
    eq_simplified: dict(~sympy.core.expr.Expr)
        If `return_symbolic_eqs` is `True`, dictionary of the model tendencies symbolic functions.
    agotensor: SymbolicQgsTensor
        If `return_symbolic_qgtensor` is `True`, the symbolic tendencies tensor of the system.

    """
    make_ip_subs = True

    if continuation_variables is None:
        make_ip_subs = False

        # Generates list of all available parameters
        continuation_variables = params._all_items

    else:
        for cv in continuation_variables:
            try:
                if params.scale_params.n == cv:
                    make_ip_subs = False
            except:
                pass

    if not make_ip_subs:
        warnings.warn("Calculating inner products symbolically, as the variable 'n' has been specified as a variable, "
                      "this may take a while.")

    if params.atmospheric_basis is not None:
        if atm_ip is None:
            aip = AtmosphericSymbolicInnerProducts(params, return_symbolic=True, make_substitution=make_ip_subs)
        else:
            aip = atm_ip
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
        agotensor = SymbolicQgsTensorT4(params, aip, oip, gip)
    elif params.dynamic_T:
        agotensor = SymbolicQgsTensorDynamicT(params, aip, oip, gip)
    else:
        agotensor = SymbolicQgsTensor(params, aip, oip, gip)

    xx = list()
    xx.append(1)

    for i in range(1, params.ndim+1):
        xx.append(Symbol('U_'+str(i)))

    if params.dynamic_T:
        eq = symbolic_sparse_mult5(agotensor.sub_tensor(continuation_variables=continuation_variables), xx, xx, xx, xx)
        if return_jacobian:
            dict_eq = symbolic_sparse_mult4(agotensor.sub_tensor(
                agotensor.jac_dic, continuation_variables=continuation_variables), xx, xx, xx)

    else:
        eq = symbolic_sparse_mult3(agotensor.sub_tensor(continuation_variables=continuation_variables), xx, xx)
        if return_jacobian:
            dict_eq = symbolic_sparse_mult2(agotensor.sub_tensor(
                agotensor.jac_dic, continuation_variables=continuation_variables), xx)

    eq_simplified = dict()
    dict_eq_simplified = dict()
    
    if continuation_variables is None:
        # Simplifying at this step is slow
        # This only needs to be used if no substitutions are being made
        for i in range(1, params.ndim+1):
            eq_simplified[i] = eq[i].simplify()
            if return_jacobian:
                for j in range(1, params.ndim+1):
                    if (i, j) in dict_eq:
                        dict_eq_simplified[(i, j)] = dict_eq[(i, j)].simplify()

    else:
        eq_simplified = eq
        if return_jacobian:
            dict_eq_simplified = dict_eq

    func = equation_as_function(equations=eq_simplified, params=params, language=language,
                                continuation_variables=continuation_variables)

    if return_jacobian:
        func_jac = jacobian_as_function(equations=dict_eq_simplified, params=params, language=language,
                                        continuation_variables=continuation_variables)

    ret = list()
    ret.append(func)
    if return_jacobian:
        ret.append(func_jac)
    if return_inner_products:
        ret.append((aip, oip, gip))
    if return_symbolic_eqs:
        ret.append(eq_simplified)
    if return_symbolic_qgtensor:
        ret.append(agotensor)
    return ret


def translate_equations(equations, language='python'):
    """Function to output the model equations as a string in the specified language syntax.

    Parameters
    ----------
    equations: dict(str), list, str
        Dictionary, list, or string of the symbolic model equations.
    language: str, optional
        Language syntax that the equations are returned in. Options are:

        - `python`
        - `fortran`
        - `julia`
        - `auto`
        - `mathematica`

        Default to `python`.

    Returns
    -------
    str_eq: dict(str)
        Dictionary of strings of the model equations.
    """

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
            temp_str = equations[key]
            for k in translator.keys():
                temp_str = temp_str.replace(k, translator[k])
            str_eq[key] = temp_str
    elif isinstance(equations, list):
        str_eq = list()
        for eq in equations:
            for k in translator.keys():
                eq = eq.replace(k, translator[k])
            str_eq.append(eq)
    elif isinstance(equations, str):
        str_eq = equations
        for k in translator.keys():
            str_eq = str_eq.replace(k, translator[k])
    else:
        raise ValueError("Expected a dict, list, or string input")

    return str_eq


def format_equations(equations, params, save_loc=None, language='python', print_equations=False):
    """Function formats the equations, in the programming language specified, and saves the equations to the specified
    location. The variables in the equation are substituted if the model variable is input.

    Parameters
    ----------
    equations: dict(~sympy.core.expr.Expr)
        Dictionary of symbolic model equations.
    params: QgParams
        The parameters fully specifying the model configuration.
    save_loc: str, optional
        Location to save the outputs as a .txt file.
    language: str, optional
        Language syntax that the equations are returned in. Options are:

        - `python`
        - `fortran`
        - `julia`
        - `auto`
        - `mathematica`

        Default to `python`.
    print_equations: bool, optional
        If `True`, equations are printed by the function, if `False`, equation strings are returned by the function.
        Defaults to `False`

    Returns
    -------
    equation_dict: dict(~sympy.core.expr.Expr)
        Dictionary of symbolic model equations, that have been substituted with numerical values.

    """
    equation_dict = dict()

    # Substitute variable symbols
    vector_subs = dict()
    if language == 'python':
        for i in range(1, params.ndim+1):
            vector_subs['U_'+str(i)] = Symbol('U['+str(i-1)+']')
    
    if language == 'fortran' or language == 'auto':
        for i in range(1, params.ndim+1):
            vector_subs['U_'+str(i)] = Symbol('U('+str(i)+')')

    if language == 'julia':
        for i in range(1, params.ndim+1):
            vector_subs['U_'+str(i)] = Symbol('U['+str(i)+']')

    if language == 'mathematica':
        for i in range(1, params.ndim+1):
            vector_subs['U_'+str(i)] = Symbol('U('+str(i)+')')

    for k in equations.keys():
        if isinstance(equations[k], float):
            eq = equations[k]
        else:
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


def equations_to_string(equations):
    """
    Converts the symbolic equations, held in a dict format, to a dict of strings.

    Parameters
    ----------
    equations: dict(~sympy.core.expr.Expr)
        Dictionary of the substituted symbolic model equations.

    Returns
    -------
    dict(str)
        Dictionary of the substituted symbolic model equations.
    """

    str_eq = dict()
    for key in equations.keys():
        str_eq[key] = str(equations[key])
    return str_eq


def equation_as_function(equations, params, continuation_variables, language='python'):
    """Converts the symbolic equations to a function in string format in the language syntax specified,
    or a lambdified python function.
    
    Parameters
    ----------
    equations: dict(~sympy.core.expr.Expr)
        Dictionary of the substituted symbolic model equations.
    params: QgParams
        The parameters fully specifying the model configuration.
    continuation_variables: list(Parameter, ScalingParameter, ParametersArray) or None
        The variables to not substitute by their numerical value and to leave in the equations.
        If `None`, no variables are substituted.
        If an empty list is provided, then all variables are substituted, providing fully numerical tendencies.
    language: str, optional
        Language syntax that the equations are returned in. Options are:

        - `python`
        - `fortran`
        - `julia`
        - `auto`
        - `mathematica`

        Default to `python`.

    Returns
    -------
    f_output: str
        Output is a function as a string in the specified language syntax.
    
    """

    if continuation_variables is None:
        continuation_variables = list()

    eq_dict = format_equations(equations, params, language=language)
    eq_dict = equations_to_string(eq_dict)

    f_output = list()
    if language == 'python':
        f_output.append('@njit')
        func_def_str = 'def f(t, U'
        for v in continuation_variables:
            func_def_str += ', ' + str(v.symbol)

        f_output.append(func_def_str + '):' )

        f_output.append('\t# Tendency function of the qgs model')
        for v in continuation_variables:
            f_output.append('\t# ' + str(v.symbol) + ":\t" + str(v.description))

        f_output.append('')
        f_output.append('\tF = np.empty_like(U)')

        for n, eq in eq_dict.items():
            f_output.append('\tF['+str(n-1)+'] = ' + eq)

        f_output.append('\treturn F')
        f_output = translate_equations(f_output, language='python')
        f_output = '\n'.join(f_output)

    if language == 'julia':
        f_output.append('function f!(du, U, p, t)')
        f_output.append('\t# Tendency function of the qgs model')

        for i, v in enumerate(continuation_variables):
            f_output.append('\t' + str(v.symbol) + " = p[" + str(i+1) + "] " + "\t# " + str(v.description))

        f_output.append('')
        for n, eq in eq_dict.items():
            f_output.append('\tdu['+str(n)+'] = ' + eq)
        
        f_output.append('end')
        f_output = translate_equations(f_output, language='julia')
        f_output = '\n'.join(f_output)

    if language == 'fortran':
        f_var = ''
        for fv in continuation_variables:
            f_var += ', ' + str(fv.symbol)
        f_output.append('SUBROUTINE FUNC(NDIM, t, U, F' + f_var + ')')

        f_output.append('\t! Tendency function of the qgs model')
        f_output.append('\tINTEGER, INTENT(IN) :: NDIM')
        f_output.append('\tDOUBLE PRECISION, INTENT(IN) :: U(NDIM), PAR(*)')
        f_output.append('\tDOUBLE PRECISION, INTENT(OUT) :: F(NDIM)')

        for v in continuation_variables:
            f_output.append('\tDOUBLE PRECISION, INTENT(IN) :: ' + str(v.symbol) + "\t! " + str(v.description))

        f_output.append('')

        f_output = _split_equations(eq_dict, f_output)
        
        f_output.append('END SUBROUTINE')
        f_output = translate_equations(f_output, language='fortran')
        f_output = '\n'.join(f_output)

    if language == 'auto':
        eq_dict = _split_equations(eq_dict, f_output)
        auto_file, auto_config = create_auto_file(eq_dict, params, continuation_variables)
        auto_file, auto_config = (
            translate_equations(auto_file, language='fortran'), translate_equations(auto_config, language='fortran'))
        f_output = ['\n'.join(auto_file), '\n'.join(auto_config)]

    if language == 'mathematica':
        raise NotImplemented("Mathematica code output is not yet available.")
        # TODO: This function needs testing before release
        f_output.append('F = Array[' + str(len(eq_dict)) + ']')

        for n, eq in eq_dict.items():
            f_output.append('F['+str(n)+'] = ' + str(eq))

        # TODO !!!! Killing output as I have not tested the above code !!!!
        eq_dict = translate_equations(eq_dict, language='mathematica')
        f_output = '\n'.join(f_output)
        f_output = None

    return f_output


def jacobian_as_function(equations, params, continuation_variables, language='python'):
    """Converts the symbolic equations of the jacobain to a function in string format in the language syntax specified,
    or a lambdified python function.

    Parameters
    ----------
    equations: dict(~sympy.core.expr.Expr)
        Dictionary of the substituted symbolic model equations.
    params: QgParams
        The parameters fully specifying the model configuration.
    continuation_variables: list(Parameter, ScalingParameter, ParametersArray) or None
        The variables to not substitute by their numerical value and to leave in the equations.
        If `None`, no variables are substituted.
        If an empty list is provided, then all variables are substituted, providing fully numerical tendencies.
    language: str, optional
        Language syntax that the equations are returned in. Options are:

        - `python`
        - `fortran`
        - `julia`
        - `auto`
        - `mathematica`

        Default to `python`.

    Returns
    -------
    f_output: str
        Output is a function as a string in the specified language syntax

    """

    if continuation_variables is None:
        continuation_variables = list()

    eq_dict = format_equations(equations, params, language=language)
    eq_dict = equations_to_string(eq_dict)

    f_output = list()
    if language == 'python':
        f_output.append('@njit')
        func_def_str = 'def jac(t, U'
        for v in continuation_variables:
            func_def_str += ', ' + str(v.symbol)

        f_output.append(func_def_str + '):')

        f_output.append('\t# Jacobian function of the qgs model')

        for v in continuation_variables:
            f_output.append('\t# ' + str(v.symbol) + ":\t" + str(v.description))

        f_output.append('')
        f_output.append('\tJ = np.zeros((len(U), len(U)))')
        for n, eq in eq_dict.items():
            f_output.append('\tJ[' + str(n[0] - 1) + ', ' + str(n[1] - 1) + '] = ' + str(eq))

        f_output.append('\treturn J')
        f_output = '\n'.join(f_output)

    if language == 'julia':
        f_output.append('function jac!(du, U, p, t)')
        f_output.append('\t# Jacobian function of the qgs model')

        for i, v in enumerate(continuation_variables):
            f_output.append('\t' + str(v.symbol) + " = p[" + str(i+1) + "]")

        f_output.append('')
        for n, eq in eq_dict.items():
            f_output.append('\tdu[' + str(n[0]) + ', ' + str(n[1]) + '] = ' + str(eq))

        f_output.append('end')
        eq_dict = translate_equations(eq_dict, language='julia')
        f_output = '\n'.join(f_output)

    if language == 'fortran':
        f_var = ''

        for fv in continuation_variables:
            f_var += ', ' + str(fv.symbol)
        f_output.append('SUBROUTINE FUNC(NDIM, t, U, JAC' + f_var + ')')

        f_output.append('\t! Jacobian function of the qgs model')
        f_output.append('\tINTEGER, INTENT(IN) :: NDIM')
        f_output.append('\tDOUBLE PRECISION, INTENT(IN) :: U(NDIM), PAR(*)')
        f_output.append('\tDOUBLE PRECISION, INTENT(OUT) :: JAC(NDIM, NDIM)')

        for v in continuation_variables:
            f_output.append('\tDOUBLE PRECISION, INTENT(IN) :: ' + str(v.symbol) + "\t! " + str(v.description))

        f_output.append('')

        f_output = _split_equations(eq_dict, f_output, two_dim=True)

        f_output.append('END SUBROUTINE')
        eq_dict = translate_equations(eq_dict, language='fortran')
        f_output = '\n'.join(f_output)

    if language == 'auto':
        eq_dict = _split_equations(eq_dict, f_output, two_dim=True)
        auto_file, auto_config = create_auto_file(eq_dict, params, continuation_variables)
        auto_file, auto_config = (
            translate_equations(auto_file, language='fortran'), translate_equations(auto_config, language='fortran'))
        f_output = ['\n'.join(auto_file), '\n'.join(auto_config)]

    if language == 'mathematica':
        # TODO: This function needs testing before release
        raise NotImplemented("Mathematica code output is not yet available.")

        f_output.append('jac = Array[' + str(len(eq_dict)) + ']')

        for n, eq in eq_dict.items():
            f_output.append('jac[' + str(n[0]) + ', ' + str(n[1]) + '] = ' + eq)

        # TODO !!!! Killing output as I have not tested the above code !!!!
        eq_dict = translate_equations(eq_dict, language='mathematica')
        f_output = '\n'.join(f_output)
        f_output = None

    return f_output


def create_auto_file(equations, params, continuation_variables, auto_main_template=None, auto_c_template=None,
                     initialize_params=False, initialize_solution=False):
    """Creates the AUTO configuration file and the model file.
    Saves files to specified folder.

    Parameters
    ----------
    equations: dict
        Dictionary of the substituted symbolic model equations
    params: QgParams
        The parameters fully specifying the model configuration.
    continuation_variables: list(Parameter, ScalingParameter, ParametersArray)
        The variables to not substitute by their numerical value and to leave in the equations.
        There must be at least one variable in this list and, due to AUTO constraints, its maximum length is 10.
    auto_main_template: str, optional
        The template to be used to generate the main AUTO file.
        If not provided, use the default template.
    auto_c_template: str, optional
        The template to be used to generate the AUTO config file.
        If not provided, use the default template.
    initialize_params: bool, optional
        Add lines in the AUTO STPNT function to initialize the parameters. Default to `False`.
    initialize_solution: bool, optional
        Add lines in the AUTO STPNT function to initialize the solution. Default to `False`.

    Returns
    -------
    auto_file: str
        The auto model file as a string

    auto_config: str
        Auto configuration file as a string
    """

    if (len(continuation_variables) < 1) or (len(continuation_variables) > 10):
        raise ValueError("Too many variables for auto file")

    # Declare variables
    declare_var = list()    
    for v in continuation_variables:
        declare_var.append('DOUBLE PRECISION ' + str(v.symbol))

    # make list of parameters
    var_list = list()
    var_ini = list()
    sol_ini = list()

    for i, v in enumerate(continuation_variables):
        temp_str = str(v.symbol) + " = PAR(" + str(i+1) + ")"
        initial_value = "PAR(" + str(i+1) + ") = " + str(v) + "  ! Variable: " + str(v.symbol)

        var_list.append(temp_str)
        var_ini.append(initial_value)

    for i in range(1, params.ndim+1):
        initial_sol = "U(" + str(i) + ") = 0.0d0"

        sol_ini.append(initial_sol)

    # Writing model file ################

    if auto_main_template is not None:
        lines = auto_main_template.split('\n')
    else:
        lines = default_auto_main_template.split('\n')

    auto_file = list()
    for ln in lines:
        if 'PARAMETER DECLARATION' in ln:
            for dv in declare_var:
                auto_file.append('\t' + dv)
        elif 'CONTINUATION PARAMETERS' in ln:
            for v in var_list:
                auto_file.append('\t' + v)
        elif 'EVOLUTION EQUATIONS' in ln:
            for e in equations:
                auto_file.append(e)
        elif 'INITIALISE PARAMETERS' in ln and initialize_params:
            for iv in var_ini:
                auto_file.append('\t' + iv)
        elif 'INITIALISE SOLUTION' in ln and initialize_solution:
            for iv in sol_ini:
                auto_file.append('\t' + iv)
        else:
            auto_file.append(ln.replace('\n', ''))
    
    # Writing config file ################

    if auto_c_template is not None:
        lines = auto_c_template.split('\n')
    else:
        lines = default_auto_c_template.split('\n')

    auto_config = list()
    for ln in lines:
        if '! PARAMETERS' in ln:
            params_dic = {i+1: str(v.symbol) for i, v in enumerate(continuation_variables)}
            params_dic.update({11: 'T', 12: 'theta', 14: 't', 25: 'T_r'})
            auto_config.append('parnames = ' + str(params_dic))

        elif '! VARIABLES' in ln:
            auto_config.append('unames = ' + str({i+1: params.var_string[i] for i in range(params.ndim)}))

        elif '! DIMENSION' in ln:
            auto_config.append('NDIM = ' + str(params.ndim))

        elif '! CONTINUATION ORDER' in ln:
            auto_config.append('ICP = ' + str([str(v.symbol) for v in continuation_variables]))
        
        elif '! SOLUTION SAVE' in ln:
            auto_config.append("# ! User to input save locations")
            auto_config.append('UZR = ' + str({str(v.symbol): [] for v in continuation_variables}))

        elif '! STOP CONDITIONS' in ln:
            auto_config.append("# ! User to input variable bounds")
            auto_config.append('UZSTOP = ' + str({str(v.symbol): [] for v in continuation_variables}))
        
        else:
            auto_config.append(ln.replace('\n', ''))

    return auto_file, auto_config


def _split_equations(eq_dict, f_output, line_len=80, two_dim=False):
    """Function to split FORTRAN equations to a set length when producing functions"""

    for n, eq in eq_dict.items():
        # split equations to be a maximum of `line_len`
        # split remainder of equation into chunks of length `line_length`

        # First translate the equation to ensure variable names are not split across rows
        eq_translated = translate_equations(eq, language='fortran')
        eq_chunks = [eq_translated[x: x + line_len] for x in range(0, len(eq_translated), line_len)]
        if len(eq_chunks) > 1:
            if two_dim:
                f_output.append('\tJAC(' + str(n[0]) + ', ' + str(n[1]) + ') =\t ' + eq_chunks[0] + "&")
            else:
                f_output.append('\tF(' + str(n) + ') =\t ' + eq_chunks[0] + "&")
            for ln in eq_chunks[1:-1]:
                f_output.append("\t\t&" + ln + "&")

            f_output.append("\t\t&" + eq_chunks[-1])
        else:
            if two_dim:
                f_output.append('\tJAC(' + str(n[0]) + ', ' + str(n[1]) + ') =\t ' + eq_chunks[0])
            else:
                f_output.append('\tF(' + str(n) + ') =\t ' + eq_chunks[0])
        f_output.append('')
    return f_output


# ------------- Default AUTO files templates ----------------

default_auto_main_template = """!----------------------------------------------------------------------
!----------------------------------------------------------------------
!   AUTO file for qgs model
!----------------------------------------------------------------------
!----------------------------------------------------------------------

SUBROUTINE FUNC(NDIM,U,ICP,PAR,IJAC,F,DFDU,DFDP)
\t!--------- ----

\t! Evaluates the algebraic equations or ODE right hand side

\t! Input arguments :
\t!      NDIM   :   Dimension of the algebraic or ODE system 
\t!      U      :   State variables
\t!      ICP    :   Array indicating the free parameter(s)
\t!      PAR    :   Equation parameters

\t! Values to be returned :
\t!      F      :   Equation or ODE right hand side values
  
\t! Normally unused Jacobian arguments : IJAC, DFDU, DFDP (see manual)

\tIMPLICIT NONE
\tINTEGER, INTENT(IN) :: NDIM, IJAC, ICP(*)
\tDOUBLE PRECISION, INTENT(IN) :: U(NDIM), PAR(*)
\tDOUBLE PRECISION, INTENT(OUT) :: F(NDIM)
\tDOUBLE PRECISION, INTENT(INOUT) :: DFDU(NDIM,NDIM),DFDP(NDIM,*)

! PARAMETER DECLARATION

! CONTINUATION PARAMETERS


! EVOLUTION EQUATIONS

END SUBROUTINE FUNC

!-----------------------------------------------------------------------
!-----------------------------------------------------------------------

SUBROUTINE STPNT(NDIM,U,PAR,T)
\t!--------- -----
  
\t! Input arguments :
\t!      NDIM   :   Dimension of the algebraic or ODE system 

\t! Values to be returned :
\t!      U      :   A starting solution vector
\t!      PAR    :   The corresponding equation-parameter values

\t! Note : For time- or space-dependent solutions this subroutine has
\t!        the scalar input parameter T contains the varying time or space
\t!        variable value.

\tIMPLICIT NONE
\tINTEGER, INTENT(IN) :: NDIM
\tDOUBLE PRECISION, INTENT(INOUT) :: U(NDIM),PAR(*)
\tDOUBLE PRECISION, INTENT(IN) :: T
\tDOUBLE PRECISION :: X(NDIM+1)
\tINTEGER :: i,is

\t! Initialize the equation parameters

! INITIALISE PARAMETERS

\t! Initialize the solution

! INITIALISE SOLUTION

\t! Initialization from a solution file (selection with PAR36)
\t! open(unit=15,file='',status='old')
\t! is=int(PAR(36))
\t! if (is.gt.0) print*, 'Loading from solution :',is
\t! DO i=1,is
\t!    read(15,*) X
\t! ENDDO
\t! close(15)
\t! U=X(2:NDIM+1)

END SUBROUTINE STPNT

!----------------------------------------------------------------------
!----------------------------------------------------------------------

SUBROUTINE BCND(NDIM,PAR,ICP,NBC,U0,U1,FB,IJAC,DBC)
\t!--------- ----

\t! Boundary Conditions

\t! Input arguments :
\t!      NDIM   :   Dimension of the ODE system 
\t!      PAR    :   Equation parameters
\t!      ICP    :   Array indicating the free parameter(s)
\t!      NBC    :   Number of boundary conditions
\t!      U0     :   State variable values at the left boundary
\t!      U1     :   State variable values at the right boundary
    
\t! Values to be returned :
\t!      FB     :   The values of the boundary condition functions 

\t! Normally unused Jacobian arguments : IJAC, DBC (see manual)

\tIMPLICIT NONE
\tINTEGER, INTENT(IN) :: NDIM, ICP(*), NBC, IJAC
\tDOUBLE PRECISION, INTENT(IN) :: PAR(*), U0(NDIM), U1(NDIM)
\tDOUBLE PRECISION, INTENT(OUT) :: FB(NBC)
\tDOUBLE PRECISION, INTENT(INOUT) :: DBC(NBC,*)

\t!X FB(1)=
\t!X FB(2)=

END SUBROUTINE BCND

!----------------------------------------------------------------------
!----------------------------------------------------------------------

SUBROUTINE ICND(NDIM,PAR,ICP,NINT,U,UOLD,UDOT,UPOLD,FI,IJAC,DINT)

\t! Integral Conditions

\t! Input arguments :
\t!      NDIM   :   Dimension of the ODE system 
\t!      PAR    :   Equation parameters
\t!      ICP    :   Array indicating the free parameter(s)
\t!      NINT   :   Number of integral conditions
\t!      U      :   Value of the vector function U at `time' t

\t! The following input arguments, which are normally not needed,
\t! correspond to the preceding point on the solution branch
\t!      UOLD   :   The state vector at 'time' t
\t!      UDOT   :   Derivative of UOLD with respect to arclength
\t!      UPOLD  :   Derivative of UOLD with respect to `time'

\t! Normally unused Jacobian arguments : IJAC, DINT

\t! Values to be returned :
\t!      FI     :   The value of the vector integrand 

\tIMPLICIT NONE
\tINTEGER, INTENT(IN) :: NDIM, ICP(*), NINT, IJAC
\tDOUBLE PRECISION, INTENT(IN) :: PAR(*)
\tDOUBLE PRECISION, INTENT(IN) :: U(NDIM), UOLD(NDIM), UDOT(NDIM), UPOLD(NDIM)
\tDOUBLE PRECISION, INTENT(OUT) :: FI(NINT)
\tDOUBLE PRECISION, INTENT(INOUT) :: DINT(NINT,*)

END SUBROUTINE ICND

!----------------------------------------------------------------------
!----------------------------------------------------------------------


SUBROUTINE FOPT(NDIM,U,ICP,PAR,IJAC,FS,DFDU,DFDP)
\t!--------- ----
\t!
\t! Defines the objective function for algebraic optimization problems
\t!
\t! Supplied variables :
\t!      NDIM   :   Dimension of the state equation
\t!      U      :   The state vector
\t!      ICP    :   Indices of the control parameters
\t!      PAR    :   The vector of control parameters
\t!
\t! Values to be returned :
\t!      FS      :   The value of the objective function
\t!
\t! Normally unused Jacobian argument : IJAC, DFDP

\tIMPLICIT NONE
\tINTEGER, INTENT(IN) :: NDIM, ICP(*), IJAC
\tDOUBLE PRECISION, INTENT(IN) :: U(NDIM), PAR(*)
\tDOUBLE PRECISION, INTENT(OUT) :: FS
\tDOUBLE PRECISION, INTENT(INOUT) :: DFDU(NDIM),DFDP(*)

END SUBROUTINE FOPT

!----------------------------------------------------------------------
!----------------------------------------------------------------------

SUBROUTINE PVLS(NDIM,U,PAR)
\t!--------- ----

\tIMPLICIT NONE
\tINTEGER, INTENT(IN) :: NDIM
\tDOUBLE PRECISION, INTENT(INOUT) :: U(NDIM)
\tDOUBLE PRECISION, INTENT(INOUT) :: PAR(*)
\tDOUBLE PRECISION :: GETP,pi,realfm,imagfm,imagfm1
\tDOUBLE PRECISION :: lw,lw1
\tLOGICAL, SAVE :: first = .TRUE.
\tDOUBLE PRECISION :: T
\tINTEGER :: i

\t!IF (first) THEN
\t\t!CALL STPNT(NDIM,U,PAR,T)
\t\t!first = .FALSE.
\t!ENDIF

\tPAR(25)=0.
\tpi = 4*ATAN(1d0)
\ti=1
\tlw=100.
\tlw1=101.
\tDO WHILE(i < NDIM)
\t\trealfm = GETP('EIG',I*2-1,U)
\t\tIF (ABS(realfm) < lw) THEN
\t\t\tlw = ABS(realfm)
\t\t\tlw1 = ABS(GETP('EIG',(I+1)*2-1,U))
\t\t\timagfm1 = ABS(GETP('EIG',(I+1)*2,U))
\t\t\timagfm = ABS(GETP('EIG',I*2,U))
\t\tEND IF
\t\ti=i+1
\tEND DO
\tIF ((lw==lw1).AND.(imagfm1==imagfm).AND.(imagfm/=0.D0)) THEN
\tPAR(25) = 2*pi/imagfm
\tENDIF
\t!---------------------------------------------------------------------- 
\t! NOTE : 
\t! Parameters set in this subroutine should be considered as ``solution 
\t! measures'' and be used for output purposes only.
\t! 
\t! They should never be used as `true'' continuation parameters. 
\t!
\t! They may, however, be added as ``over-specified parameters'' in the 
\t! parameter list associated with the AUTO-Constant NICP, in order to 
\t! print their values on the screen and in the ``p.xxx file.
\t!
\t! They may also appear in the list associated with AUTO-Constant NUZR.
\t!
\t!---------------------------------------------------------------------- 
\t! For algebraic problems the argument U is, as usual, the state vector.
\t! For differential equations the argument U represents the approximate 
\t! solution on the entire interval [0,1]. In this case its values must 
\t! be accessed indirectly by calls to GETP, as illustrated below.
\t!---------------------------------------------------------------------- 
\t!
\t! Set PAR(2) equal to the L2-norm of U(1)
\t!X PAR(2)=GETP('NRM',1,U)
\t!
\t! Set PAR(3) equal to the minimum of U(2)
\t!X PAR(3)=GETP('MIN',2,U)
\t!
\t! Set PAR(4) equal to the value of U(2) at the left boundary.
\t!X PAR(4)=GETP('BV0',2,U)
\t!
\t! Set PAR(5) equal to the pseudo-arclength step size used.
\t!X PAR(5)=GETP('STP',1,U)
\t!
\t!---------------------------------------------------------------------- 
\t! The first argument of GETP may be one of the following:
\t!        'NRM' (L2-norm),     'MAX' (maximum),
\t!        'INT' (integral),    'BV0 (left boundary value),
\t!        'MIN' (minimum),     'BV1' (right boundary value).
\t!
\t! Also available are
\t!   'STP' (Pseudo-arclength step size used).
\t!   'FLD' (`Fold function', which vanishes at folds).
\t!   'BIF' (`Bifurcation function', which vanishes at singular points).
\t!   'HBF' (`Hopf function'; which vanishes at Hopf points).
\t!   'SPB' ( Function which vanishes at secondary periodic bifurcations).
\t!---------------------------------------------------------------------- 
END SUBROUTINE PVLS
"""

default_auto_c_template = """#Configuration files

#Parameters name
# ! PARAMETERS
#Variables name
# ! VARIABLES
#Dimension of the system
# ! DIMENSION
#Problem type (1 for FP, 2 for PO, -2 for time integration)
IPS =   1
#Start solution label
IRS =   0
#Continuation parameters (in order of use)
# ! CONTINUATION ORDER
#Number of mesh intervals
NTST=   100
#Print and restart every NPR steps (0 to disable)
NPR=   0
#Number of bifurcating branches to compute (negative number means continue only in one direction)
MXBF=0
#Detection of Special Points
ISP=2
#Maximum number of iteration in the Newton-Chord method
ITNW=7
#Arc-length continuation parameters
DS  =  0.00001, DSMIN= 1e-15, DSMAX=   1.0
#Precision parameters (Typiq. EPSS = EPSL * 10^-2)
EPSL=1e-07, EPSU=1e-07, EPSS=1e-05
#Number of parameter (don't change it)
NPAR = 36
#User defined value where to save the solution
# ! SOLUTION SAVE
#Stop conditions
# ! STOP CONDITIONS
"""
