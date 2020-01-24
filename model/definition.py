
import numpy as np
from numba import njit

from inner_products.analytic import AtmosphericInnerProducts, OceanicInnerProducts
from tensors.qgtensor import QgsTensor
from tensors.cootensor import from_csr_mat_list
from functions.sparse import sparse_mul3, sparse_mul_jac


def create_tendencies(params):

    if params.ablocks is not None:
        aip = AtmosphericInnerProducts(params)
    else:
        aip = None

    if params.oblocks is not None:
        oip = OceanicInnerProducts(params)
    else:
        oip = None

    if aip is not None and oip is not None:
        aip.connect_to_ocean(oip)

    tensor = QgsTensor(aip, oip)
    coo_tensor = from_csr_mat_list(tensor.tensor)

    coo = coo_tensor.coo
    val = coo_tensor.value

    @njit
    def f(t, x):
        xx = np.concatenate((np.full((1,), 1.), x))
        xr = sparse_mul3(coo, val, xx)

        return xr[1:]

    return f


def create_linearized_tendencies(params):

    if params.ablocks is not None:
        aip = AtmosphericInnerProducts(params)
    else:
        aip = None

    if params.oblocks is not None:
        oip = OceanicInnerProducts(params)
    else:
        oip = None

    if aip is not None and oip is not None:
        aip.connect_to_ocean(oip)

    tensor = QgsTensor(aip, oip)
    jcoo_tensor = from_csr_mat_list(tensor.jacobian_tensor)

    jcoo = jcoo_tensor.coo
    jval = jcoo_tensor.value

    @njit
    def Df(t, x):
        xx = np.concatenate((np.full((1,), 1.), x))
        mul_jac = sparse_mul_jac(jcoo, jval, xx)
        return mul_jac[1:, 1:]

    return Df

