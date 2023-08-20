import sympy as sy

def _add_to_dict(dic, loc, value):
    '''
    Adds `value` to dictionary `dic`, with the dictionary key of `loc`.
    If the dictionary did not have a key of `loc` before, a new key is made.
    '''
    
    try:
        dic[loc] += value
    except:
        dic[loc] = value
    return dic

def _symbolic_tensordot(a, b, axes=2):
    """
    Compute tensor dot product along specified axes of two sympy symbolic arrays

    This is based on numpy.tensordot

    Parameters
    ----------
    a, b: sympy arrays
        Tensors to "dot"

    axes: int
        * integer_like
          If an int N, sum over the last N axes of `a` and the first N axes
          of `b` in order. The sizes of the corresponding axes must match.

    Returns
    -------
    output: sympy tensor
        The tensor dot product of the input

    """
    as_ = a.shape
    nda = len(as_)
    
    a_com = [nda+i for i in range(-axes, 0)]
    b_com = [nda+i for i in range(axes)]
    sum_cols = tuple(a_com + b_com)
    
    prod = sy.tensorproduct(a, b)
    
    return sy.tensorcontraction(prod, sum_cols)

def symbolic_sparse_mult2(dic, vec_a):
    """
    Symbolic multiplication of a tensor with one vector:
    :math:`A_{i,j} = {\displaystyle \sum_{k=0}^{\mathrm{ndim}}} \, \mathcal{T}_{i,j,k} \, a_k`

    Parameters
    ----------
    dic: Dict(Sympy.Symbol)
        A dictionary whose keys are the coordinates of the tensor, and the dictionary values are the values of the tensor.
    vec_a: List(Sympy.Symbol)
        The list :math:`a_k` to contract the tensor with entries of Sympy Symbols. Must be of shape (:attr:`~.params.QgParams.ndim` + 1,).

    Returns
    -------
    res: dict
        The matrix :math:`A_{i,j}`, of shape (:attr:`~.params.QgParams.ndim` + 1, :attr:`~.params.QgParams.ndim` + 1), contained in a dictionary, where the keys are the tensor coordinates, and the values are the tensor values.
    """
    res = dict()

    for key in dic.keys():
        coo1, coo2, coo3 = key
        val = vec_a[coo3] * dic[key]
        res = _add_to_dict(res, (coo1, coo2), val)

    return res

def symbolic_sparse_mult3(dic, vec_a, vec_b):
    """
    Symbolic multiplication of a tensor with two vectors:
    :math:`v_i = {\displaystyle \sum_{j,k=0}^{\mathrm{ndim}}} \, \mathcal{T}_{i,j,k} \, a_j \, b_k`

    Parameters
    ----------
    dic: Dict(Sympy.Symbol)
        A dictionary whose keys are the coordinates of the tensor, and the dictionary values are the values of the tensor.
    vec_a: List(Sympy.Symbol)
        The list :math:`a_j` to contract the tensor with entries of Sympy Symbols. Must be of shape (:attr:`~.params.QgParams.ndim` + 1,).
    vec_b: List(Sympy.Symbol)
        The list :math:`b_k` to contract the tensor with entries of Sympy Symbols. Must be of shape (:attr:`~.params.QgParams.ndim` + 1,).


    Returns
    -------
    res: List(Sympy.Symbol)
        The vector :math:`v_i`, of shape (:attr:`~.params.QgParams.ndim` + 1,), contained in a dictionary, where the keys are the tensor coordinates, and the values are the tensor values.

    """
    res = dict()

    for key in dic.keys():
        coo1, coo2, coo3 = key
        val = vec_a[coo2] * vec_b[coo3] * dic[key]
        res = _add_to_dict(res, coo1, val)

    return res

def symbolic_sparse_mult4(dic, vec_a, vec_b, vec_c):
    """
    Symbolic multiplication of a rank-5 tensor with three vectors:
    :math:`A_{i, j} = {\displaystyle \sum_{k,l,m=0}^{\mathrm{ndim}}} \, \mathcal{T}_{i,j,k,l, m} \, a_k \, b_l \, c_m`


    Parameters
    ----------
    dic: Dict(Sympy.Symbol)
        A dictionary where they keys are a tuple of 5 elements which are the coordinates of the tensor values, which are contained in the dictionary values.
    value: ~numpy.ndarray(float)
        A 1D array of shape (n_elems,), a list of value in the tensor
    vec_a: List(Sympy.Symbol)
        The list :math:`a_j` to contract the tensor with entries of Sympy Symbols. Must be of shape (:attr:`~.params.QgParams.ndim` + 1,).
    vec_b: List(Sympy.Symbol)
        The list :math:`b_k` to contract the tensor with entries of Sympy Symbols. Must be of shape (:attr:`~.params.QgParams.ndim` + 1,).
    vec_c: List(Sympy.Symbol)
        The list :math:`c_l` to contract the tensor with entries of Sympy Symbols. Must be of shape (:attr:`~.params.QgParams.ndim` + 1,).

    Returns
    -------
    res: List(Sympy.Symbol)
        The matrix :math:`A_{i, j}`, of shape (:attr:`~.params.QgParams.ndim` + 1, :attr:`~.params.QgParams.ndim` + 1), contained in a dictionary, where the keys are the tensor coordinates, and the values are the tensor values.
    """
    res = dict()

    for key in dic.keys():
        coo1, coo2, coo3, coo4, coo5 = key
        val = vec_a[coo3] * vec_b[coo4] * vec_c[coo5] * dic[key]
        res = _add_to_dict(res, (coo1, coo2), val)

    return res

def symbolic_sparse_mult5(dic, vec_a, vec_b, vec_c, vec_d):
    """
    Symbolic multiplication of a rank-5 tensor with four vectors:
    :math:`v_i = {\displaystyle \sum_{j,k,l,m=0}^{\mathrm{ndim}}} \, \mathcal{T}_{i,j,k,l,m} \, a_j \, b_k \, c_l \, d_m`

    Parameters
    ----------
    dic: Dict(Sympy.Symbol)
        A dictionary whose keys are the coordinates of the tensor, and the dictionary values are the values of the tensor.
    vec_a: List(Sympy.Symbol)
        The list :math:`a_j` to contract the tensor with entries of Sympy Symbols. Must be of shape (:attr:`~.params.QgParams.ndim` + 1,).
    vec_b: List(Sympy.Symbol)
        The list :math:`b_k` to contract the tensor with entries of Sympy Symbols. Must be of shape (:attr:`~.params.QgParams.ndim` + 1,).
    vec_c: List(Sympy.Symbol)
        The list :math:`c_l` to contract the tensor with entries of Sympy Symbols. Must be of shape (:attr:`~.params.QgParams.ndim` + 1,).
    vec_d: List(Sympy.Symbol)
        The list :math:`d_m` to contract the tensor with entries of Sympy Symbols. Must be of shape (:attr:`~.params.QgParams.ndim` + 1,).

    Returns
    -------
    res: List(Sympy.Symbol)
        The vector :math:`v_i`, of shape (:attr:`~.params.QgParams.ndim` + 1,), contained in a dictionary, where the keys are the tensor coordinates, and the values are the tensor values.
    """
    res = dict()

    for key in dic.keys():
        coo1, coo2, coo3, coo4, coo5 = key
        val = vec_a[coo2] * vec_b[coo3] * vec_c[coo4] * vec_d[coo5] * dic[key]
        res = _add_to_dict(res, coo1, val)

    return res