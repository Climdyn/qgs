"""
    Symbolic matrix operation module
    ================================

    This module supports operations and functions on the symbolic sparse tensors defined in
    the :class:`~.tensors.symbolic_qgtensor.SymbolicQgsTensor` objects.

"""

from sympy import tensorproduct, tensorcontraction
from qgs.functions.util import add_to_dict


def symbolic_tensordot(a, b, axes=2):
    """Compute tensor dot product along specified axes of two sympy symbolic arrays

    This is based on `Numpy`_ :meth:`~numpy.tensordot` .

    .. _Numpy: https://numpy.org/

    Parameters
    ----------
    a, b: ~sympy.tensor.array.DenseNDimArray or ~sympy.tensor.array.SparseNDimArray
        Arrays to take the dot product of.

    axes: int
        Sum over the last `axes` axes of `a` and the first `axes` axes
        of `b` in order. The sizes of the corresponding axes must match.

    Returns
    -------
    output: Sympy tensor
        The tensor dot product of the input.

    """
    as_ = a.shape
    nda = len(as_)
    
    a_com = [nda+i for i in range(-axes, 0)]
    b_com = [nda+i for i in range(axes)]
    sum_cols = tuple(a_com + b_com)
    
    prod = tensorproduct(a, b)
    
    return tensorcontraction(prod, sum_cols)


def symbolic_sparse_mult2(dic, vec_a):
    """Symbolic multiplication of a tensor with one vector:
    :math:`A_{i,j} = {\\displaystyle \\sum_{k=0}^{\\mathrm{ndim}}} \\, \\mathcal{T}_{i,j,k} \\, a_k`

    Parameters
    ----------
    dic: dict(~sympy.core.symbol.Symbol)
        A dictionary whose keys are the coordinates of the tensor, and the dictionary values are the values of the
        tensor.
    vec_a: list(~sympy.core.symbol.Symbol)
        The list :math:`a_k` to contract the tensor with entries of Sympy Symbols.
        Must be of shape (:attr:`~.params.QgParams.ndim` + 1,).

    Returns
    -------
    res: dict(~sympy.core.symbol.Symbol)
        The matrix :math:`A_{i,j}`, of shape (:attr:`~.params.QgParams.ndim` + 1, :attr:`~.params.QgParams.ndim` + 1),
        contained in a dictionary, where the keys are the tensor coordinates, and the values are the tensor values.
    """
    res = dict()

    for key in dic.keys():
        coo1, coo2, coo3 = key
        if coo1 > 0 and coo2 > 0:
            val = vec_a[coo3] * dic[key]
            res = add_to_dict(res, (coo1, coo2), val)

    return res


def symbolic_sparse_mult3(dic, vec_a, vec_b):
    """Symbolic multiplication of a tensor with two vectors:
    :math:`v_i = {\\displaystyle \\sum_{j,k=0}^{\\mathrm{ndim}}} \\, \\mathcal{T}_{i,j,k} \\, a_j \\, b_k`

    Parameters
    ----------
    dic: dict(~sympy.core.symbol.Symbol)
        A dictionary whose keys are the coordinates of the tensor, and the dictionary values are the values of the
        tensor.
    vec_a: list(~sympy.core.symbol.Symbol)
        The list :math:`a_j` to contract the tensor with entries of Sympy Symbols.
        Must be of shape (:attr:`~.params.QgParams.ndim` + 1,).
    vec_b: list(~sympy.core.symbol.Symbol)
        The list :math:`b_k` to contract the tensor with entries of Sympy Symbols.
        Must be of shape (:attr:`~.params.QgParams.ndim` + 1,).


    Returns
    -------
    res: dict(~sympy.core.symbol.Symbol)
        The vector :math:`v_i`, of shape (:attr:`~.params.QgParams.ndim` + 1,), contained in a dictionary, where
        the keys are the tensor coordinates, and the values are the tensor values.

    """
    res = dict()

    for key in dic.keys():
        coo1, coo2, coo3 = key
        if coo1 > 0:
            val = vec_a[coo2] * vec_b[coo3] * dic[key]
            res = add_to_dict(res, coo1, val)

    return res


def symbolic_sparse_mult4(dic, vec_a, vec_b, vec_c):
    """Symbolic multiplication of a rank-5 tensor with three vectors:
    :math:`A_{i, j} = {\\displaystyle \\sum_{k,l,m=0}^{\\mathrm{ndim}}} \\, \\mathcal{T}_{i,j,k,l, m} \\, a_k \\, b_l \\, c_m`


    Parameters
    ----------
    dic: dict(~sympy.core.symbol.Symbol)
        A dictionary where they keys are a tuple of 5 elements which are the coordinates of the tensor values,
        which are contained in the dictionary values.
    vec_a: list(~sympy.core.symbol.Symbol)
        The list :math:`a_j` to contract the tensor with entries of Sympy Symbols.
        Must be of shape (:attr:`~.params.QgParams.ndim` + 1,).
    vec_b: list(~sympy.core.symbol.Symbol)
        The list :math:`b_k` to contract the tensor with entries of Sympy Symbols.
        Must be of shape (:attr:`~.params.QgParams.ndim` + 1,).
    vec_c: list(~sympy.core.symbol.Symbol)
        The list :math:`c_l` to contract the tensor with entries of Sympy Symbols.
        Must be of shape (:attr:`~.params.QgParams.ndim` + 1,).

    Returns
    -------
    res: dict(~sympy.core.symbol.Symbol)
        The matrix :math:`A_{i, j}`, of shape (:attr:`~.params.QgParams.ndim` + 1, :attr:`~.params.QgParams.ndim` + 1),
        contained in a dictionary, where the keys are the tensor coordinates, and the values are the tensor values.
    """
    res = dict()

    for key in dic.keys():
        coo1, coo2, coo3, coo4, coo5 = key
        if coo1 > 0 and coo2 > 0:
            val = vec_a[coo3] * vec_b[coo4] * vec_c[coo5] * dic[key]
            res = add_to_dict(res, (coo1, coo2), val)

    return res


def symbolic_sparse_mult5(dic, vec_a, vec_b, vec_c, vec_d):
    """Symbolic multiplication of a rank-5 tensor with four vectors:
    :math:`v_i = {\\displaystyle \\sum_{j,k,l,m=0}^{\\mathrm{ndim}}} \\, \\mathcal{T}_{i,j,k,l,m} \\, a_j \\, b_k \\, c_l \\, d_m`

    Parameters
    ----------
    dic: dict(~sympy.core.symbol.Symbol)
        A dictionary whose keys are the coordinates of the tensor, and the dictionary values are the values of the
        tensor.
    vec_a: list(~sympy.core.symbol.Symbol)
        The list :math:`a_j` to contract the tensor with entries of Sympy Symbols.
        Must be of shape (:attr:`~.params.QgParams.ndim` + 1,).
    vec_b: list(~sympy.core.symbol.Symbol)
        The list :math:`b_k` to contract the tensor with entries of Sympy Symbols.
        Must be of shape (:attr:`~.params.QgParams.ndim` + 1,).
    vec_c: list(~sympy.core.symbol.Symbol)
        The list :math:`c_l` to contract the tensor with entries of Sympy Symbols.
        Must be of shape (:attr:`~.params.QgParams.ndim` + 1,).
    vec_d: list(~sympy.core.symbol.Symbol)
        The list :math:`d_m` to contract the tensor with entries of Sympy Symbols.
        Must be of shape (:attr:`~.params.QgParams.ndim` + 1,).

    Returns
    -------
    res: dict(~sympy.core.symbol.Symbol)
        The vector :math:`v_i`, of shape (:attr:`~.params.QgParams.ndim` + 1,), contained in a dictionary,
        where the keys are the tensor coordinates, and the values are the tensor values.
    """
    res = dict()

    for key in dic.keys():
        coo1, coo2, coo3, coo4, coo5 = key
        if coo1 > 0:
            val = vec_a[coo2] * vec_b[coo3] * vec_c[coo4] * vec_d[coo5] * dic[key]
            res = add_to_dict(res, coo1, val)

    return res
