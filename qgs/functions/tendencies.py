
"""
    Tendencies definition module
    ============================

    This module provides functions to create the tendencies functions of the model, based on
    its parameters.

"""
import numpy as np
from numba import njit

from qgs.inner_products.analytic import AtmosphericAnalyticInnerProducts, OceanicAnalyticInnerProducts, GroundAnalyticInnerProducts
from qgs.inner_products.symbolic import AtmosphericSymbolicInnerProducts, OceanicSymbolicInnerProducts, GroundSymbolicInnerProducts
from qgs.tensors.qgtensor import QgsTensor, QgsTensorDynamicT, QgsTensorT4
from qgs.tensors.atmo_thermo_tensor import AtmoThermoTensor, AtmoThermoTensorDynamicT, AtmoThermoTensorT4
from qgs.functions.sparse_mul import sparse_mul5, sparse_mul4, sparse_mul3, sparse_mul2


def create_tendencies(params, return_inner_products=False, return_qgtensor=False):
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
    params: QgParams
        The parameters fully specifying the model configuration.
    return_inner_products: bool
        If True, return the inner products of the model. Default to False.
    return_qgtensor: bool
        If True, return the tendencies tensor of the model. Default to False.


    Returns
    -------
    f: callable
        The numba-jitted tendencies function.
    Df: callable
        The numba-jitted linearized tendencies function.
    inner_products: (AtmosphericInnerProducts, OceanicInnerProducts)
        If `return_inner_products` is True, the inner products of the system.
    qgtensor: QgsTensor
        If `return_qgtensor` is True, the tendencies tensor of the system.
    """

    if params.ablocks is not None:
        aip = AtmosphericAnalyticInnerProducts(params)
    elif params.atmospheric_basis is not None:
        aip = AtmosphericSymbolicInnerProducts(params)
    else:
        aip = None

    if params.oblocks is not None:
        oip = OceanicAnalyticInnerProducts(params)
    elif params.oceanic_basis is not None:
        oip = OceanicSymbolicInnerProducts(params)
    else:
        oip = None

    if params.gblocks is not None:
        gip = GroundAnalyticInnerProducts(params)
    elif params.ground_basis is not None:
        gip = GroundSymbolicInnerProducts(params)
    else:
        gip = None

    if aip is not None and oip is not None:
        if not aip.connected_to_ocean:
            aip.connect_to_ocean(oip)
    elif aip is not None and gip is not None:
        if not aip.connected_to_ground:
            aip.connect_to_ground(gip)

    if params.T4:
        agotensor = QgsTensorT4(params, aip, oip, gip)
    elif params.dynamic_T:
        agotensor = QgsTensorDynamicT(params, aip, oip, gip)
    else:
        agotensor = QgsTensor(params, aip, oip, gip)

    coo = agotensor.tensor.coords.T
    val = agotensor.tensor.data

    jcoo = agotensor.jacobian_tensor.coords.T
    jval = agotensor.jacobian_tensor.data

    if params.T4 or params.dynamic_T:
        @njit
        def f(t, x):
            xx = np.concatenate((np.full((1,), 1.), x))
            xr = sparse_mul5(coo, val, xx, xx, xx, xx)
            return xr[1:]

        @njit
        def Df(t, x):
            xx = np.concatenate((np.full((1,), 1.), x))
            mul_jac = sparse_mul4(jcoo, jval, xx, xx, xx)
            return mul_jac[1:, 1:]
    else:
        @njit
        def f(t, x):
            xx = np.concatenate((np.full((1,), 1.), x))
            xr = sparse_mul3(coo, val, xx, xx)
            return xr[1:]

        @njit
        def Df(t, x):
            xx = np.concatenate((np.full((1,), 1.), x))
            mul_jac = sparse_mul2(jcoo, jval, xx)
            return mul_jac[1:, 1:]

    ret = list()
    ret.append(f)
    ret.append(Df)
    if return_inner_products:
        ret.append((aip, oip, gip))
    if return_qgtensor:
        ret.append(agotensor)
    return ret


def create_atmo_thermo_tendencies(params, return_atmo_thermo_tensor=False):
    """Returns a function which return a part of the atmospheric thermodynamic tendencies useful for computing the vertical wind.

    Parameters
    ----------
    params: QgParams
        The parameters fully specifying the model configuration.
    return_atmo_thermo_tensor: bool
        If True, return the tendencies tensor of these tencencies. Default to False.


    Returns
    -------
    f: callable
        The numba-jitted tendencies function.
    atmo_thermo_tensor: AtmoThermoTensor
        If `return_atmo_thermo_tensor` is True, the tendencies tensor of the system.
    """

    if params.ablocks is not None:
        aip = AtmosphericAnalyticInnerProducts(params)
    elif params.atmospheric_basis is not None:
        aip = AtmosphericSymbolicInnerProducts(params)
    else:
        aip = None

    if params.oblocks is not None:
        oip = OceanicAnalyticInnerProducts(params)
    elif params.oceanic_basis is not None:
        oip = OceanicSymbolicInnerProducts(params)
    else:
        oip = None

    if params.gblocks is not None:
        gip = GroundAnalyticInnerProducts(params)
    elif params.ground_basis is not None:
        gip = GroundSymbolicInnerProducts(params)
    else:
        gip = None

    if aip is not None and oip is not None:
        if not aip.connected_to_ocean:
            aip.connect_to_ocean(oip)
    elif aip is not None and gip is not None:
        if not aip.connected_to_ground:
            aip.connect_to_ground(gip)

    if params.T4:
        agotensor = AtmoThermoTensorT4(params, aip, oip, gip)
    elif params.dynamic_T:
        agotensor = AtmoThermoTensorDynamicT(params, aip, oip, gip)
    else:
        agotensor = AtmoThermoTensor(params, aip, oip, gip)

    coo = agotensor.tensor.coords.T
    val = agotensor.tensor.data

    if params.T4 or params.dynamic_T:
        @njit
        def f(t, x):
            xx = np.concatenate((np.full((1,), 1.), x))
            xr = sparse_mul5(coo, val, xx, xx, xx, xx)
            return xr[1:]

    else:
        @njit
        def f(t, x):
            xx = np.concatenate((np.full((1,), 1.), x))
            xr = sparse_mul3(coo, val, xx, xx)
            return xr[1:]

    if return_atmo_thermo_tensor:
        ret = list()
        ret.append(f)
        ret.append(agotensor)
    else:
        ret = f

    return ret


if __name__ == '__main__':
    from qgs.params.params import QgParams

    params = QgParams()
    params.set_atmospheric_channel_fourier_modes(2, 2, mode='symbolic')
    params.set_oceanic_basin_fourier_modes(2, 4, mode='symbolic')
    f, Df = create_tendencies(params)
