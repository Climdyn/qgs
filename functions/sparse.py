
import numpy as np
from numba import njit


@njit
def sparse_mul3(coo, value, vec):

    res = np.zeros_like(vec)
    for n in range(coo.shape[0]):
        res[coo[n, 0]] += vec[coo[n, 1]] * vec[coo[n, 2]] * value[n]
    res[0] = 1.
    return res
