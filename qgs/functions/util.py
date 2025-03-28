"""
    Utility functions module
    ========================

    This module has some useful functions for the model.

"""


import numpy as np
from numba import njit


def add_to_dict(dic, loc, value):
    """Adds `value` to dictionary `dic`, with the dictionary key of `loc`.
    If the dictionary did not have a key of `loc` before, a new key is made.

    Parameters
    ----------
    dic: dict
        Dictionary to add the value to.
    loc:
        Item of the dictionary to add the value to.
    value:
        Value to add.
    """
    try:
        dic[loc] += value
    except:
        dic[loc] = value
    return dic


@njit
def reverse(a):
    """Numba-jitted function to reverse a 1D array.

    Parameters
    ----------
    a: ~numpy.ndarray
        The 1D array to reverse.

    Returns
    -------
    ~numpy.ndarray
        The reversed array.
    """
    out = np.zeros_like(a)
    ii = 0
    for i in range(len(a)-1,-1,-1):
        out[ii] = a[i]
        ii +=1
    return out


@njit
def normalize_matrix_columns(a):
    """Numba-jitted function to normalize the columns of a 2D array to one.

    Parameters
    ----------
    a: ~numpy.ndarray
        The 2D array to column-normalize.

    Returns
    -------
    ~numpy.ndarray
        The normalized array.
    """
    an = np.zeros_like(a)
    norm = np.zeros(a.shape[0])
    for i in range(a.shape[1]):
        norm[i] = np.linalg.norm(a[:, i], 2)
        an[:, i] = a[:, i] / norm[i]
    return an, norm


@njit
def solve_triangular_matrix(a, b):
    """Solve a triangular linear matrix equation :math:`ax = b`.
    
    Parameters
    ----------
    a: ~numpy.ndarray
        The 2D array :math:`a`.
    b: ~numpy.ndarray
        The 2D array :math:`b`.

    Returns
    -------
    ~numpy.ndarray
        The 2D array of solution :math:`x`.
    """
    x = np.zeros_like(a)
    for i in range(2, a.shape[0]+1):
        x[:i, i - 1] = np.linalg.solve(a[:i, :i], b[:i, i - 1])
    x[0, 0] = b[0, 0] / a[0, 0]
    return x
