
import numpy as np
from numba import njit

from inner_products.analytic import AtmosphericInnerProducts
from tensors.atensor import AtmosphericTensor
from tensors.cootensor import from_csr_mat_list
from functions.sparse import sparse_mul3, sparse_mul_jac


def create_tendencies(params):

    aip= AtmosphericInnerProducts(params)
    atensor = AtmosphericTensor(aip)
    coo_atensor = from_csr_mat_list(atensor.tensor)

    coo = coo_atensor.coo
    val = coo_atensor.value

    @njit
    def f(t, x):
        xx = np.concatenate((np.full((1,), 1.), x))
        xr = sparse_mul3(coo, val, xx)

        return xr[1:]

    return f


def create_linearized_tendencies(params):

    aip= AtmosphericInnerProducts(params)
    atensor = AtmosphericTensor(aip)
    jcoo_atensor = from_csr_mat_list(atensor.jacobian_tensor)

    jcoo = jcoo_atensor.coo
    jval = jcoo_atensor.value

    @njit
    def Df(t, x):
        xx = np.concatenate((np.full((1,), 1.), x))
        mul_jac = sparse_mul_jac(jcoo, jval, xx)
        return mul_jac[1:, 1:]

    return Df

