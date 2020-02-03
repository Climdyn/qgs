
"""
    Tendencies definition module
    ============================

    This module provides functions to create the tendencies functions of the model, based on
    its parameters.

"""
import numpy as np
from numba import njit

from inner_products.analytic import AtmosphericInnerProducts, OceanicInnerProducts
from tensors.qgtensor import QgsTensor
from tensors.cootensor import from_csr_mat_list
from functions.sparse import sparse_mul3, sparse_mul2


def create_tendencies(params):
    """Function to handle the inner products and tendencies tensors construction.
    Returns the tendencies function :math:`\\boldsymbol{f}` determining the model's ordinary differential
    equations:

    .. math:: \dot{\\boldsymbol{x}} = \\boldsymbol{f}(\\boldsymbol{x})

    which is for the model's integration.

    It returns also the linearized tendencies
    :math:`\\boldsymbol{\mathrm{J}} \equiv \\boldsymbol{\mathrm{D}f} = \\frac{\partial \\boldsymbol{f}}{\partial \\boldsymbol{x}}`
    (Jacobian matrix) which are used by the tangent linear model:

    .. math :: \dot{\\boldsymbol{\delta x}} = \\boldsymbol{\mathrm{J}}(\\boldsymbol{x}) \cdot \\boldsymbol{\delta x}

    Parameters
    ----------
    params: ~params.params.QgParams
        The parameters fully specifying the model configuration.

    Returns
    -------
    f, Df: callable
        The numba-jitted tendencies and linearized tendencies functions.
    """

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
        xr = sparse_mul3(coo, val, xx, xx)

        return xr[1:]

    jcoo_tensor = from_csr_mat_list(tensor.jacobian_tensor)

    jcoo = jcoo_tensor.coo
    jval = jcoo_tensor.value

    @njit
    def Df(t, x):
        xx = np.concatenate((np.full((1,), 1.), x))
        mul_jac = sparse_mul2(jcoo, jval, xx)
        return mul_jac[1:, 1:]

    return f, Df


