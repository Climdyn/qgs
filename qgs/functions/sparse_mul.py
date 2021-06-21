"""
    Sparse matrix operation module
    ==============================

    This module supports operations and functions on the sparse tensors defined in
    the :class:`~.tensors.qgtensor.QgsTensor` objects.

"""
import numpy as np
from numba import njit


@njit
def sparse_mul3(coo, value, vec_a, vec_b):
    """Sparse multiplication of a tensor with two vectors:
    :math:`v_i = {\displaystyle \sum_{j,k=0}^{\mathrm{ndim}}} \, \mathcal{T}_{i,j,k} \, a_j \, b_k`

    Warnings
    --------
    It is a Numba-jitted function, so it cannot take a :class:`sparse.COO` sparse tensor directly.
    The tensor coordinates list and values must be provided separately by the user.

    In principle, this will be solved later in `sparse`, see https://github.com/pydata/sparse/issues/378.

    Parameters
    ----------
    coo: ~numpy.ndarray(int)
        A 2D array of shape (n_elems, 3), a list of n_elems tensor coordinates corresponding to each value provided.
    value: ~numpy.ndarray(float)
        A 1D array of shape (n_elems,), a list of value in the tensor
    vec_a: ~numpy.ndarray(float)
        The vector :math:`a_j` to contract the tensor with. Must be of shape (:attr:`~.params.QgParams.ndim` + 1,).
    vec_b: ~numpy.ndarray(float)
        The vector :math:`b_k` to contract the tensor with. Must be of shape (:attr:`~.params.QgParams.ndim` + 1,).

    Returns
    -------
    ~numpy.ndarray(float)
        The vector :math:`v_i`, of shape (:attr:`~.params.QgParams.ndim` + 1,).
    """
    res = np.zeros_like(vec_a)
    n_elems = coo.shape[0]
    for n in range(n_elems):
        res[coo[n, 0]] += vec_a[coo[n, 1]] * vec_b[coo[n, 2]] * value[n]
    res[0] = 1.
    return res


@njit
def sparse_mul2(coo, value, vec):
    """Sparse multiplication of a tensor with one vector:
    :math:`A_{i,j} = {\displaystyle \sum_{k=0}^{\mathrm{ndim}}} \, \mathcal{T}_{i,j,k} \, a_k`

    Warnings
    --------
    It is a Numba-jitted function, so it cannot take a :class:`sparse.COO` sparse tensor directly.
    The tensor coordinates list and values must be provided separately by the user.

    In principle, this will be solved later in `sparse`, see https://github.com/pydata/sparse/issues/378.

    Parameters
    ----------
    coo: ~numpy.ndarray(int)
        A 2D array of shape (n_elems, 3), a list of n_elems tensor coordinates corresponding to each value provided.
    value: ~numpy.ndarray(float)
        A 1D array of shape (n_elems,), a list of value in the tensor
    vec: ~numpy.ndarray(float)
        The vector :math:`a_k` to contract the tensor with. Must be of shape (:attr:`~.params.QgParams.ndim` + 1,).

    Returns
    -------
    ~numpy.ndarray(float)
        The matrix :math:`A_{i,j}`, of shape (:attr:`~.params.QgParams.ndim` + 1, :attr:`~.params.QgParams.ndim` + 1).
    """

    res = np.zeros((len(vec), len(vec)))

    for n in range(coo.shape[0]):
        res[coo[n, 0], coo[n, 1]] += vec[coo[n, 2]] * value[n]

    return res

