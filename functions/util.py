"""
    Utility functions module
    ========================

    This module has some useful functions for the model.

"""


import numpy as np
from numba import njit


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

