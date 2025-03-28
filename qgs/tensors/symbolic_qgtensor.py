"""
    symbolic qgs tensor module
    ==========================

    This module computes and holds the symbolic representation of the tensors representing the tendencies of the model's
    equations.

"""
from qgs.functions.symbolic_mul import symbolic_tensordot
from qgs.functions.util import add_to_dict
from qgs.params.params import Parameter, ScalingParameter, ParametersArray, Params

import numpy as np
from sympy import simplify
import pickle

from sympy.matrices.immutable import ImmutableSparseMatrix
from sympy.tensor.array import ImmutableSparseNDimArray

# TODO: Check non stored IP version of this


class SymbolicQgsTensor(object):
    """Symbolic qgs tendencies tensor class.

    Parameters
    ----------
    params: None or QgParams, optional
        The models parameters to configure the tensor. `None` to initialize an empty tensor. Default to `None`.
    atmospheric_inner_products: None or AtmosphericInnerProducts, optional
        The inner products of the atmospheric basis functions on which the model's PDE atmospheric equations are
        projected.
        If `None`, disable the atmospheric tendencies. Default to `None`.
        The inner product is returned in symbolic or numeric form.
    oceanic_inner_products: None or OceanicInnerProducts, optional
        The inner products of the oceanic basis functions on which the model's PDE oceanic equations are projected.
        If `None`, disable the oceanic tendencies. Default to `None`.
        The inner product is returned in symbolic or numeric form.
    ground_inner_products: None or GroundInnerProducts, optional
        The inner products of the ground basis functions on which the model's PDE ground equations are projected.
        If `None`, disable the ground tendencies. Default to `None`.
        The inner product is returned in symbolic or numeric form.

    Attributes
    ----------
    params: None or QgParams
        The models parameters used to configure the tensor. `None` for an empty tensor.
    atmospheric_inner_products: None or AtmosphericInnerProducts
        The inner products of the atmospheric basis functions on which the model's PDE atmospheric equations are
        projected.
        If `None`, disable the atmospheric tendencies. Default to `None`.
    oceanic_inner_products: None or OceanicInnerProducts
        The inner products of the oceanic basis functions on which the model's PDE oceanic equations are projected.
        If `None`, disable the oceanic tendencies. Default to `None`.
    ground_inner_products: None or GroundInnerProducts
        The inner products of the ground basis functions on which the model's PDE ground equations are projected.
        If `None`, disable the ground tendencies. Default to `None`.
    tensor: ~sympy.tensor.array.ImmutableSparseNDimArray
        The tensor :math:`\\mathcal{T}_{i,j,k}` :math:`i`-th components.
    jacobian_tensor: ~sympy.tensor.array.ImmutableSparseNDimArray
        The jacobian tensor :math:`\\mathcal{T}_{i,j,k} + \\mathcal{T}_{i,k,j}` :math:`i`-th components.
    """

    def __init__(self, params=None, atmospheric_inner_products=None, oceanic_inner_products=None,
                 ground_inner_products=None):

        self.atmospheric_inner_products = atmospheric_inner_products
        self.oceanic_inner_products = oceanic_inner_products
        self.ground_inner_products = ground_inner_products
        self.params = params

        self.tensor = None
        self.jacobian_tensor = None

        if not self.params.dynamic_T:
            self.compute_tensor()

    def _psi_a(self, i):
        """Transform the :math:`\\psi_{\\mathrm a}` :math:`i`-th coefficient into the effective model's variable.

        Parameters
        ----------
        i: int
            The :math:`i`-th coefficients of :math:`\\psi_{\\mathrm a}`

        Returns
        -------
        int
            The effective model's variable.
        """

        return i + 1

    def _theta_a(self, i):
        """Transform the :math:`\\theta_{\\mathrm a}` :math:`i`-th coefficient into the effective model's variable.

        Parameters
        ----------
        i: int
            The :math:`i`-th coefficients of :math:`\\theta_{\\mathrm a}`

        Returns
        -------
        int
            The effective model's variable.
        """
        return i + self.params.variables_range[0] + 1

    def _psi_o(self, i):
        """Transform the :math:`\\psi_{\\mathrm o}` :math:`i`-th coefficient into the effective model's variable.

        Parameters
        ----------
        i: int
            The :math:`i`-th coefficients of :math:`\\psi_{\\mathrm o}`

        Returns
        -------
        int
            The effective model's variable.
        """
        return i + self.params.variables_range[1] + 1

    def _deltaT_o(self, i):
        """Transform the :math:`\\delta T_{\\mathrm o}` :math:`i`-th coefficient into the effective model's variable.

        Parameters
        ----------
        i: int
            The :math:`i`-th coefficients of :math:`\\delta T_{\\mathrm o}`

        Returns
        -------
        int
            The effective model's variable.
        """
        return i + self.params.variables_range[2] + 1

    def _deltaT_g(self, i):
        """Transform the :math:`\\delta T_{\\mathrm o}` :math:`i`-th coefficient into the effective model's variable.

        Parameters
        ----------
        i: int
            The :math:`i`-th coefficients of :math:`\\delta T_{\\mathrm o}`

        Returns
        -------
        int
            The effective model's variable.
        """
        return i + self.params.variables_range[1] + 1
    
    def _compute_tensor_dicts(self):
        if self.params is None:
            return None

        if self.atmospheric_inner_products is None and self.oceanic_inner_products is None \
                and self.ground_inner_products is None:
            return None

        aips = self.atmospheric_inner_products

        par = self.params
        atp = par.atemperature_params
        gp = par.ground_params
        ap = par.atmospheric_params

        nvar = par.number_of_variables

        bips = None
        if self.oceanic_inner_products is not None:
            bips = self.oceanic_inner_products
            ocean = True
        else:
            ocean = False

        if self.ground_inner_products is not None:
            bips = self.ground_inner_products
            ground_temp = True
        else:
            ground_temp = False

        if self.params.dynamic_T:
            offset = 1
        else:
            offset = 0

        # constructing some derived matrices
        if aips is not None:
            a_inv = dict()
            for i in range(offset, nvar[1]):
                for j in range(offset, nvar[1]):
                    a_inv[(i - offset, j - offset)] = aips.a(i, j)

            a_inv = ImmutableSparseMatrix(nvar[0], nvar[0], a_inv)
            a_inv = a_inv.inverse()
            a_inv = a_inv.simplify()

            a_theta = dict()
            for i in range(nvar[1]):
                for j in range(nvar[1]):
                    a_theta[(i, j)] = ap.sig0.symbol * aips.a(i, j) - aips.u(i, j)
            
            a_theta = ImmutableSparseMatrix(nvar[1], nvar[1], a_theta)
            a_theta = a_theta.inverse()
            a_theta = a_theta.simplify()

        if bips is not None:
            if ocean:
                U_inv = dict()
                for i in range(nvar[3]):
                    for j in range(nvar[3]):
                        U_inv[(i, j)] = bips.U(i, j)
                U_inv = ImmutableSparseMatrix(nvar[3], nvar[3], U_inv)
                U_inv = U_inv.inverse()
                U_inv = U_inv.simplify()

                M_psio = dict()
                for i in range(offset, nvar[3]):
                    for j in range(offset, nvar[3]):
                        M_psio[(i - offset, j - offset)] = bips.M(i, j) + self.params.G.symbolic_expression * bips.U(i, j)

                M_psio = ImmutableSparseMatrix(nvar[2], nvar[2], M_psio)
                M_psio = M_psio.inverse()
                M_psio = M_psio.simplify()

            else:
                U_inv = dict()
                for i in range(nvar[2]):
                    for j in range(nvar[2]):
                        U_inv[(i, j)] = bips.U(i, j)
                U_inv = ImmutableSparseMatrix(nvar[2], nvar[2], U_inv)
                U_inv = U_inv.inverse()
                U_inv = U_inv.simplify()

        ################

        if bips is not None:
            go = bips.stored
        else:
            go = True

        sy_arr_dic = dict()

        if aips.stored and go:
            # psi_a part
            a_inv_mult_c = a_inv @ aips._c[offset:, offset:]

            if gp is not None:
                hk_sym_arr = ImmutableSparseMatrix(gp.hk.symbols)

                if gp.orographic_basis == "atmospheric":
                    a_inv_mult_g = symbolic_tensordot(a_inv, aips._g[offset:, offset:, offset:], axes=1)
                    oro = symbolic_tensordot(a_inv_mult_g, hk_sym_arr, axes=1)[:, :, 0]
                else:
                    a_inv_mult_gh = symbolic_tensordot(a_inv, aips._gh[offset:, offset:, offset:], axes=1)
                    oro = symbolic_tensordot(a_inv_mult_gh, hk_sym_arr, axes=1)[:, :, 0]

            a_inv_mult_b = symbolic_tensordot(a_inv, aips._b[offset:, offset:, offset:], axes=1)
            
            if ocean:
                a_inv_mult_d = a_inv @ aips._d[offset:, offset:]

            for i in range(nvar[0]):
                for j in range(nvar[0]):
                    jo = j + offset  # skipping the theta 0 variable if it exists
                    sy_arr_dic = add_to_dict(sy_arr_dic, (self._psi_a(i), self._psi_a(j), 0), -a_inv_mult_c[i, j] * par.scale_params.beta.symbol)
                    sy_arr_dic = add_to_dict(sy_arr_dic, (self._psi_a(i), self._psi_a(j), 0), -(ap.kd.symbol * _kronecker_delta(i, j)) / 2)
                    sy_arr_dic = add_to_dict(sy_arr_dic, (self._psi_a(i), self._theta_a(jo), 0), (ap.kd.symbol * _kronecker_delta(i, j)) / 2)

                    if gp is not None:
                        # convert 
                        if gp.hk is not None:
                            sy_arr_dic = add_to_dict(sy_arr_dic, (self._psi_a(i), self._psi_a(j), 0), -oro[i, j] / 2)
                            sy_arr_dic = add_to_dict(sy_arr_dic, (self._psi_a(i), self._theta_a(jo), 0), oro[i, j] / 2)

                    for k in range(nvar[0]):
                        ko = k + offset  # skipping the theta 0 variable if it exists
                        sy_arr_dic = add_to_dict(sy_arr_dic, (self._psi_a(i), self._psi_a(j), self._psi_a(k)), -a_inv_mult_b[i, j, k])
                        sy_arr_dic = add_to_dict(sy_arr_dic, (self._psi_a(i), self._theta_a(jo), self._theta_a(ko)), -a_inv_mult_b[i, j, k])

                if ocean:
                    for j in range(nvar[2]):
                        sy_arr_dic = add_to_dict(sy_arr_dic, (self._psi_a(i), self._psi_o(j), 0), a_inv_mult_d[i, j] * ap.kd.symbol / 2)
            
            # theta_a part
            a_theta_mult_u = a_theta @ aips._u
            if self.params.Cpa is not None:
                val_Cpa = a_theta_mult_u @ ImmutableSparseMatrix(self.params.Cpa.symbolic_expressions)

            if atp.hd is not None and atp.thetas is not None:
                thetas_sym_arr = ImmutableSparseMatrix(atp.thetas.symbols)
                val_thetas = a_theta_mult_u @ thetas_sym_arr

            a_theta_mult_a = a_theta @ aips._a[:, offset:]
            a_theta_mult_c = a_theta @ aips._c[:, offset:]

            a_theta_mult_g = symbolic_tensordot(a_theta, aips._g[:, offset:, offset:], axes=1)

            if gp is not None:
                if gp.orographic_basis == "atmospheric":
                    oro = symbolic_tensordot(a_theta_mult_g, hk_sym_arr, axes=1)[:, :, 0]
                else:
                    a_theta_mult_gh = symbolic_tensordot(a_theta, aips._gh[:, offset:, offset:], axes=1)
                    oro = symbolic_tensordot(a_theta_mult_gh, hk_sym_arr, axes=1)[:, :, 0]

            a_theta_mult_b = symbolic_tensordot(a_theta, aips._b[:, offset:, offset:], axes=1)

            if ocean:
                a_theta_mult_d = a_theta @ aips._d[:, offset:]
                a_theta_mult_s = a_theta @ aips._s

            if ground_temp:
                a_theta_mult_s = a_theta @ aips._s
                
            for i in range(nvar[1]):
                if self.params.Cpa is not None:
                    sy_arr_dic = add_to_dict(sy_arr_dic, (self._theta_a(i), 0, 0), -val_Cpa[i])

                if atp.hd is not None and atp.thetas is not None:
                    sy_arr_dic = add_to_dict(sy_arr_dic, (self._theta_a(i), 0, 0), -val_thetas[i] * atp.hd.symbol)

                for j in range(nvar[0]):

                    jo = j + offset  # skipping the theta 0 variable if it exists

                    val_2 = a_theta_mult_a[i, j] * ap.kd.symbol * ap.sig0.symbol / 2
                    val_3 = a_theta_mult_a[i, j] * (ap.kd.symbol / 2 + 2 * ap.kdp.symbol) * ap.sig0.symbol
                    sy_arr_dic = add_to_dict(sy_arr_dic, (self._theta_a(i), self._psi_a(j), 0), val_2)
                    sy_arr_dic = add_to_dict(sy_arr_dic, (self._theta_a(i), self._theta_a(jo), 0), -val_3)

                    sy_arr_dic = add_to_dict(sy_arr_dic, (self._theta_a(i), self._theta_a(jo), 0), -a_theta_mult_c[i, j] * par.scale_params.beta.symbol * ap.sig0.symbol)

                    if gp is not None:
                        if gp.hk is not None:
                            sy_arr_dic = add_to_dict(sy_arr_dic, (self._theta_a(i), self._theta_a(jo), 0), -ap.sig0.symbol * oro[i, j] / 2)
                            sy_arr_dic = add_to_dict(sy_arr_dic, (self._theta_a(i), self._psi_a(j), 0), ap.sig0.symbol * oro[i, j] / 2)

                    for k in range(nvar[0]):
                        ko = k + offset  # skipping the theta 0 variable if it exists
                        sy_arr_dic = add_to_dict(sy_arr_dic, (self._theta_a(i), self._psi_a(j), self._theta_a(ko)), - a_theta_mult_b[i, j, k] * ap.sig0.symbol)
                        sy_arr_dic = add_to_dict(sy_arr_dic, (self._theta_a(i), self._theta_a(jo), self._psi_a(k)), - a_theta_mult_b[i, j, k] * ap.sig0.symbol)

                        sy_arr_dic = add_to_dict(sy_arr_dic, (self._theta_a(i), self._psi_a(j), self._theta_a(ko)), a_theta_mult_g[i, j, k])

                for j in range(nvar[1]):
                    if self.params.Lpa is not None:
                        sy_arr_dic = add_to_dict(sy_arr_dic, (self._theta_a(i), self._theta_a(j), 0), a_theta_mult_u[i, j] * atp.sc.symbol * self.params.Lpa.symbolic_expression)
                    if self.params.LSBpa is not None:
                        sy_arr_dic = add_to_dict(sy_arr_dic, (self._theta_a(i), self._theta_a(j), 0), a_theta_mult_u[i, j] * self.params.LSBpa.symbolic_expression)

                    if atp.hd is not None:
                        sy_arr_dic = add_to_dict(sy_arr_dic, (self._theta_a(i), self._theta_a(j), 0), a_theta_mult_u[i, j] * atp.hd)

                if ocean:
                    for j in range(nvar[2]):
                        sy_arr_dic = add_to_dict(sy_arr_dic, (self._theta_a(i), self._psi_o(j), 0), -a_theta_mult_d[i, j] * ap.sig0.symbol * ap.kd.symbol / 2)

                    if self.params.Lpa is not None:
                        for j in range(nvar[3]):
                            sy_arr_dic = add_to_dict(sy_arr_dic, (self._theta_a(i), self._deltaT_o(j), 0), -a_theta_mult_s[i, j] * self.params.Lpa.symbolic_expression / 2)
                            if self.params.LSBpgo is not None:
                                sy_arr_dic = add_to_dict(sy_arr_dic, (self._theta_a(i), self._deltaT_o(j), 0), -a_theta_mult_s[i, j] * self.params.LSBpgo.symbolic_expression)

                if ground_temp:
                    if self.params.Lpa is not None:
                        for j in range(nvar[2]):
                            sy_arr_dic = add_to_dict(sy_arr_dic, (self._theta_a(i), self._deltaT_g(j), 0), -a_theta_mult_s[i, j] * self.params.Lpa.symbolic_expression / 2)
                            if self.params.LSBpgo is not None:
                                sy_arr_dic = add_to_dict(sy_arr_dic, (self._theta_a(i), self._deltaT_g(j), 0), -a_theta_mult_s[i, j] * self.params.LSBpgo.symbolic_expression)

            if ocean:
                # psi_o part
                M_psio_mult_K = M_psio @ bips._K[offset:, offset:]
                M_psio_mult_N = M_psio @ bips._N[offset:, offset:]
                M_psio_mult_M = M_psio @ bips._M[offset:, offset:]
                M_psio_mult_C = symbolic_tensordot(M_psio, bips._C[offset:, offset:, offset:], axes=1)

                for i in range(nvar[2]):
                    for j in range(nvar[0]):
                        jo = j + offset  # skipping the theta 0 variable if it exists

                        sy_arr_dic = add_to_dict(sy_arr_dic, (self._psi_o(i), self._psi_a(j), 0), M_psio_mult_K[i, j] * par.oceanic_params.d.symbol)
                        sy_arr_dic = add_to_dict(sy_arr_dic, (self._psi_o(i), self._theta_a(jo), 0), -M_psio_mult_K[i, j] * par.oceanic_params.d.symbol)

                    for j in range(nvar[2]):
                        sy_arr_dic = add_to_dict(sy_arr_dic, (self._psi_o(i), self._psi_o(j), 0), -M_psio_mult_N[i, j] * par.scale_params.beta.symbol)

                        sy_arr_dic = add_to_dict(sy_arr_dic, (self._psi_o(i), self._psi_o(j), 0), -M_psio_mult_M[i, j] * (par.oceanic_params.r.symbol + par.oceanic_params.d.symbol))

                        for k in range(nvar[2]):
                            sy_arr_dic = add_to_dict(sy_arr_dic, (self._psi_o(i), self._psi_o(j), self._psi_o(k)), - M_psio_mult_C[i, j, k])

                # deltaT_o part
                U_inv_mult_W = U_inv @ bips._W
                Cpgo_sym_arr = ImmutableSparseMatrix(self.params.Cpgo.symbolic_expressions)
                U_inv_mult_W_Cpgo = U_inv_mult_W @ Cpgo_sym_arr

                U_inv_mult_O = symbolic_tensordot(U_inv, bips._O[:, offset:, offset:], axes=1)
                
                for i in range(nvar[3]):
                    sy_arr_dic = add_to_dict(sy_arr_dic, (self._deltaT_o(i), 0, 0), U_inv_mult_W_Cpgo[i])

                    for j in range(nvar[1]):
                        sy_arr_dic = add_to_dict(sy_arr_dic, (self._deltaT_o(i), self._theta_a(j), 0), U_inv_mult_W[i, j] * 2 * atp.sc.symbol * self.params.Lpgo.symbolic_expression)
                        if self.params.sbpa is not None:
                            sy_arr_dic = add_to_dict(sy_arr_dic, (self._deltaT_o(i), self._theta_a(j), 0), U_inv_mult_W[i, j] * self.params.sbpa.symbolic_expression)

                    for j in range(nvar[3]):
                        sy_arr_dic = add_to_dict(sy_arr_dic, (self._deltaT_o(i), self._deltaT_o(j), 0), - self.params.Lpgo.symbolic_expression * _kronecker_delta(i, j))
                        if self.params.sbpgo is not None:
                            sy_arr_dic = add_to_dict(sy_arr_dic, (self._deltaT_o(i), self._deltaT_o(j), 0), - self.params.sbpgo.symbolic_expression * _kronecker_delta(i, j))

                    for j in range(nvar[2]):
                        for k in range(nvar[2]):
                            ko = k + offset  # skipping the T 0 variable if it exists
                            sy_arr_dic = add_to_dict(sy_arr_dic, (self._deltaT_o(i), self._psi_o(j), self._deltaT_o(ko)), -U_inv_mult_O[i, j, k])

            # deltaT_g part
            if ground_temp:
                U_inv_mult_W = U_inv @ bips._W
                Cpgo_sym_arr = ImmutableSparseMatrix(self.params.Cpgo.symbolic_expressions)
                U_inv_mult_W_Cpgo = U_inv_mult_W @ Cpgo_sym_arr
                for i in range(nvar[2]):
                    sy_arr_dic = add_to_dict(sy_arr_dic, (self._deltaT_g(i), 0, 0), U_inv_mult_W_Cpgo[i])

                    for j in range(nvar[1]):
                        sy_arr_dic = add_to_dict(sy_arr_dic, (self._deltaT_g(i), self._theta_a(j), 0), U_inv_mult_W[i, j] * 2 * atp.sc.symbol * self.params.Lpgo.symbolic_expression)
                        if self.params.sbpa is not None:
                            sy_arr_dic = add_to_dict(sy_arr_dic, (self._deltaT_g(i), self._theta_a(j), 0), U_inv_mult_W[i, j] * self.params.sbpa.symbolic_expression)

                    for j in range(nvar[2]):
                        sy_arr_dic = add_to_dict(sy_arr_dic, (self._deltaT_g(i), self._deltaT_g(j), 0), - self.params.Lpgo.symbolic_expression * _kronecker_delta(i, j))
                        if self.params.sbpgo is not None:
                            sy_arr_dic = add_to_dict(sy_arr_dic, (self._deltaT_g(i), self._deltaT_g(j), 0), - self.params.sbpgo.symbolic_expression * _kronecker_delta(i, j))

        else:
            # psi_a part
            for i in range(nvar[0]):
                for j in range(nvar[0]):

                    jo = j + offset

                    val = 0
                    for jj in range(nvar[0]):
                        val += a_inv[i, jj] * aips.c(offset + jj, jo)
                    sy_arr_dic = add_to_dict(sy_arr_dic, (self._psi_a(i), self._psi_a(j), 0), - val * par.scale_params.beta.symbol)
                    sy_arr_dic = add_to_dict(sy_arr_dic, (self._psi_a(i), self._psi_a(j), 0), - (ap.kd.symbol * _kronecker_delta(i, j)) / 2)
                    sy_arr_dic = add_to_dict(sy_arr_dic, (self._psi_a(i), self._theta_a(jo), 0), (ap.kd.symbol * _kronecker_delta(i, j)) / 2)

                    if gp is not None:
                        hk_sym_arr = ImmutableSparseNDimArray(gp.hk.symbols)
                        if gp.hk is not None:
                            oro = 0
                            if gp.orographic_basis == "atmospheric":
                                for jj in range(nvar[0]):
                                    for kk in range(nvar[0]):
                                        oro += a_inv[i, jj] * aips.g(offset + jj, j, offset + kk) * hk_sym_arr[kk]
                            else:
                                for jj in range(nvar[0]):
                                    for kk in range(nvar[0]):
                                        oro += a_inv[i, jj] * aips.gh(offset + jj, j, offset + kk) * hk_sym_arr[kk]
                            sy_arr_dic = add_to_dict(sy_arr_dic, (self._psi_a(i), self._psi_a(j), 0), - oro / 2)
                            sy_arr_dic = add_to_dict(sy_arr_dic, (self._psi_a(i), self._theta_a(jo), 0), oro / 2)

                    for k in range(nvar[0]):
                        ko = k + offset  # skipping the theta 0 variable if it exists
                        val = 0
                        for jj in range(nvar[0]):
                            val += a_inv[i, jj] * aips.b(offset + jj, jo, ko)
                        sy_arr_dic = add_to_dict(sy_arr_dic, (self._psi_a(i), self._psi_a(j), self._psi_a(k)), - val)
                        sy_arr_dic = add_to_dict(sy_arr_dic, (self._psi_a(i), self._theta_a(jo), self._theta_a(ko)), - val)
                if ocean:
                    for j in range(nvar[2]):
                        jo = j + offset
                        val = 0
                        for jj in range(nvar[0]):
                            val += a_inv[i, jj] * aips.d(offset + jj, jo)
                        sy_arr_dic = add_to_dict(sy_arr_dic, (self._psi_a(i), self._psi_o(j), 0), val * ap.kd.symbol / 2)

            # theta_a part
            for i in range(nvar[1]):
                if self.params.Cpa is not None:
                    val = 0
                    for jj in range(nvar[1]):
                        for kk in range(nvar[1]):
                            val -= a_theta[i, jj] * aips.u(jj, kk) * self.params.Cpa.symbolic_expressions[kk]

                    sy_arr_dic = add_to_dict(sy_arr_dic, (self._theta_a(i), 0, 0), val)

                if atp.hd is not None and atp.thetas is not None:
                    thetas_sym_arr = ImmutableSparseNDimArray(atp.thetas.symbols)
                    val = 0
                    for jj in range(nvar[1]):
                        for kk in range(nvar[1]):
                            val -= a_theta[i, jj] * aips.u(jj, kk) * thetas_sym_arr[kk]

                    sy_arr_dic = add_to_dict(sy_arr_dic, (self._theta_a(i), 0, 0), val * atp.hd.symbol)

                for j in range(nvar[0]):

                    jo = j + offset  # skipping the theta 0 variable if it exists

                    val = 0
                    for jj in range(nvar[1]):
                        val += a_theta[i, jj] * aips.a(jj, jo)

                    sy_arr_dic = add_to_dict(sy_arr_dic, (self._theta_a(i), self._psi_a(j), 0), val * ap.kd.symbol * ap.sig0.symbol / 2)
                    sy_arr_dic = add_to_dict(sy_arr_dic, (self._theta_a(i), self._theta_a(jo), 0), - val * (ap.kd.symbol / 2 - 2 * ap.kdp.symbol) * ap.sig0.symbol)

                    val = 0
                    for jj in range(nvar[1]):
                        val -= a_theta[i, jj] * aips.c(jj, jo)

                    sy_arr_dic = add_to_dict(sy_arr_dic, (self._theta_a(i), self._theta_a(jo), 0), val * par.scale_params.beta.symbol * ap.sig0.symbol)

                    if gp is not None:
                        if gp.hk is not None:
                            oro = 0
                            if gp.orographic_basis == "atmospheric":
                                for jj in range(nvar[1]):
                                    for kk in range(nvar[0]):
                                        oro += a_theta[i, jj] * aips.g(jj, jo, offset + kk) * gp.hk[kk].symbol
                            else:
                                for jj in range(nvar[1]):
                                    for kk in range(nvar[0]):
                                        oro += a_theta[i, jj] * aips.gh(jj, jo, offset + kk) * gp.hk[kk].symbol
                            sy_arr_dic = add_to_dict(sy_arr_dic, (self._theta_a(i), self._theta_a(jo), 0), - ap.sig0.symbol * oro / 2)
                            sy_arr_dic = add_to_dict(sy_arr_dic, (self._theta_a(i), self._psi_a(j), 0), ap.sig0.symbol * oro / 2)

                    for k in range(nvar[0]):
                        ko = k + offset  # skipping the theta 0 variable if it exists
                        val = 0
                        for jj in range(nvar[1]):
                            val += a_theta[i, jj] * aips.b(jj, jo, ko)

                        sy_arr_dic = add_to_dict(sy_arr_dic, (self._theta_a(i), self._psi_a(j), self._theta_a(ko)), - val * ap.sig0.symbol)
                        sy_arr_dic = add_to_dict(sy_arr_dic, (self._theta_a(i), self._theta_a(jo), self._psi_a(k)), - val * ap.sig0.symbol)

                        val = 0
                        for jj in range(nvar[1]):
                            val += a_theta[i, jj] * aips.g(jj, jo, ko)

                        sy_arr_dic = add_to_dict(sy_arr_dic, (self._theta_a(i), self._psi_a(j), self._theta_a(ko)), val)

                for j in range(nvar[1]):
                    val = 0
                    for jj in range(nvar[1]):
                        val += a_theta[i, jj] * aips.u(jj, j)

                    if self.params.Lpa is not None:
                        sy_arr_dic = add_to_dict(sy_arr_dic, (self._theta_a(i), self._theta_a(j), 0), val * atp.sc.symbol * self.params.Lpa.symbolic_expression)
                    
                    if self.params.LSBpa is not None:
                        sy_arr_dic = add_to_dict(sy_arr_dic, (self._theta_a(i), self._theta_a(j), 0), val * self.params.LSBpa.symbolic_expression)

                    if atp.hd is not None:
                        sy_arr_dic = add_to_dict(sy_arr_dic, (self._theta_a(i), self._theta_a(j), 0), val * atp.hd.symbol)

                if ocean:
                    for j in range(nvar[2]):
                        jo = j + offset  # skipping the theta 0 variable if it exists
                        val = 0
                        for jj in range(nvar[1]):
                            val -= a_theta[i, jj] * aips.d(jj, jo)
                        
                        sy_arr_dic = add_to_dict(sy_arr_dic, (self._theta_a(i), self._psi_o(j), 0), val * ap.sig0.symbol * ap.kd.symbol / 2)

                    if self.params.Lpa is not None:
                        for j in range(nvar[3]):
                            val = 0
                            for jj in range(nvar[1]):
                                val -= a_theta[i, jj] * aips.s(jj, j)
                            sy_arr_dic = add_to_dict(sy_arr_dic, (self._theta_a(i), self._deltaT_o(j), 0), val * self.params.Lpa.symbolic_expression / 2)
                            if self.params.LSBpgo is not None:
                                sy_arr_dic = add_to_dict(sy_arr_dic, (self._theta_a(i), self._deltaT_o(j), 0), val * self.params.LSBpgo.symbolic_expression)

                if ground_temp:
                    if self.params.Lpa is not None:
                        for j in range(nvar[2]):
                            val = 0
                            for jj in range(nvar[1]):
                                val -= a_theta[i, jj] * aips.s(jj, j)
                            sy_arr_dic = add_to_dict(sy_arr_dic, (self._theta_a(i), self._deltaT_g(j), 0), val * self.params.Lpa.symbolic_expression / 2)
                            if self.params.LSBpgo is not None:
                                sy_arr_dic = add_to_dict(sy_arr_dic, (self._theta_a(i), self._deltaT_g(j), 0), val * self.params.LSBpgo.symbolic_expression)

            if ocean:
                # psi_o part
                for i in range(nvar[2]):
                    for j in range(nvar[0]):
                        jo = j + offset  # skipping the theta 0 variable if it exists

                        for jj in range(nvar[2]):
                            val = M_psio[i, jj] * bips.K(offset + jj, jo) * par.oceanic_params.d.symbol
                        sy_arr_dic = add_to_dict(sy_arr_dic, (self._psi_o(i), self._psi_a(j), 0), val)
                        sy_arr_dic = add_to_dict(sy_arr_dic, (self._psi_o(i), self._theta_a(jo), 0), - val)

                    for j in range(nvar[2]):
                        jo = j + offset  # skipping the T 0 variable if it exists

                        val = 0
                        for jj in range(nvar[2]):
                            val -= M_psio[i, jj] * bips.N(offset + jj, jo)
                        sy_arr_dic = add_to_dict(sy_arr_dic, (self._psi_o(i), self._psi_o(j), 0), val * par.scale_params.beta.symbol)

                        val = 0
                        for jj in range(nvar[2]):
                            val -= M_psio[i, jj] * bips.M(offset + jj, jo)
                        sy_arr_dic = add_to_dict(sy_arr_dic, (self._psi_o(i), self._psi_o(j), 0), val * (par.oceanic_params.r.symbol + par.oceanic_params.d.symbol))

                        for k in range(nvar[2]):
                            ko = k + offset  # skipping the T 0 variable if it exists
                            val = 0
                            for jj in range(nvar[2]):
                                val -= M_psio[i, jj] * bips.C(offset + jj, jo, ko)
                            sy_arr_dic = add_to_dict(sy_arr_dic, (self._psi_o(i), self._psi_o(j), self._psi_o(k)), val)

                # deltaT_o part
                for i in range(nvar[3]):
                    val = 0
                    for jj in range(nvar[1]):
                        for kk in range(nvar[3]):
                            val += U_inv[i, kk] * bips.W(kk, jj) * self.params.Cpgo.symbolic_expressions[jj]
                    
                    sy_arr_dic = add_to_dict(sy_arr_dic, (self._deltaT_o(i), 0, 0), val)

                    for j in range(nvar[1]):
                        val = 0
                        for jj in range(nvar[3]):
                            val += U_inv[i, jj] * bips.W(jj, j)
                        sy_arr_dic = add_to_dict(sy_arr_dic, (self._deltaT_o(i), self._theta_a(j), 0), val * 2 * atp.sc.symbol * self.params.Lpgo.symbolic_expression)
                        if self.params.sbpa is not None:
                            sy_arr_dic = add_to_dict(sy_arr_dic, (self._deltaT_o(i), self._theta_a(j), 0), val * self.params.sbpa.symbolic_expression)

                    for j in range(nvar[3]):
                        sy_arr_dic = add_to_dict(sy_arr_dic, (self._deltaT_o(i), self._deltaT_o(j), 0), - self.params.Lpgo.symbolic_expression * _kronecker_delta(i, j))
                        if self.params.sbpgo is not None:
                            sy_arr_dic = add_to_dict(sy_arr_dic, (self._deltaT_o(i), self._deltaT_o(j), 0), - self.params.sbpgo.symbolic_expression * _kronecker_delta(i, j))

                    for j in range(nvar[2]):
                        for k in range(offset, nvar[3]):
                            jo = j + offset  # skipping the T 0 variable if it exists

                            val = 0
                            for jj in range(nvar[3]):
                                val -= U_inv[i, jj] * bips.O(jj, jo, k)
                            sy_arr_dic = add_to_dict(sy_arr_dic, (self._deltaT_o(i), self._psi_o(j), self._deltaT_o(k)), val)

            # deltaT_g part
            if ground_temp:
                for i in range(nvar[2]):
                    val = 0
                    for jj in range(nvar[1]):
                        for kk in range(nvar[2]):
                            val += U_inv[i, kk] * bips.W(kk, jj) * self.params.Cpgo.symbolic_expressions[jj]
                    
                    sy_arr_dic = add_to_dict(sy_arr_dic, (self._deltaT_g(i), 0, 0), val)

                    for j in range(nvar[1]):
                        val = 0
                        for jj in range(nvar[2]):
                            val += U_inv[i, jj] * bips.W(jj, j)
                        sy_arr_dic = add_to_dict(sy_arr_dic, (self._deltaT_g(i), self._theta_a(j), 0), val * 2 * atp.sc.symbol * self.params.Lpgo.symbolic_expression)
                        
                        if self.params.sbpa is not None:
                            sy_arr_dic = add_to_dict(sy_arr_dic, (self._deltaT_g(i), self._theta_a(j), 0), val * self.params.sbpa.symbolic_expression)

                    for j in range(nvar[2]):
                        sy_arr_dic = add_to_dict(sy_arr_dic, (self._deltaT_g(i), self._deltaT_g(j), 0), - self.params.Lpgo.symbolic_expression * _kronecker_delta(i, j))
                        if self.params.sbpgo is not None:
                            sy_arr_dic = add_to_dict(sy_arr_dic, (self._deltaT_g(i), self._deltaT_g(j), 0), - self.params.sbpgo.symbolic_expression * _kronecker_delta(i, j))

        return sy_arr_dic

    def compute_tensor(self):
        """Routine to compute the tensor."""
        sy_arr_dic = self._compute_tensor_dicts()
        sy_arr_dic = self.remove_dic_zeros(sy_arr_dic)

        if sy_arr_dic is not None:
            self._set_tensor(sy_arr_dic)

    def _set_tensor(self, dic, set_symbolic=False):
        self.jac_dic = self.remove_dic_zeros(self.jacobian_from_dict(dic))
        self.tensor_dic = self.remove_dic_zeros(self.simplify_dict(dic))
        
        if set_symbolic:
            self._set_symbolic_tensor()
            
    def _set_symbolic_tensor(self):
        ndim = self.params.ndim

        if self.params.dynamic_T:
            if self.params.T4:
                raise ValueError("Symbolic tensor output not configured for T4 version, use Dynamic T version")
            else:
                dims = (ndim + 1, ndim + 1, ndim + 1, ndim + 1, ndim + 1)
        else:
            dims = (ndim + 1, ndim + 1, ndim + 1)

        jacobian_tensor = ImmutableSparseNDimArray(self.jac_dic.copy(), dims)
        tensor = ImmutableSparseNDimArray(self.tensor_dic.copy(), dims)

        self.jacobian_tensor = jacobian_tensor.applyfunc(simplify)
        self.tensor = tensor.applyfunc(simplify)

    @staticmethod
    def remove_dic_zeros(dic):
        """Removes zero values from dictionary

        Parameters
        ----------
        tensor: dict
            dictionary which could include 0 in values
        Returns
        -------
        ten_out: dict
            dictionary with same keys and values as input, but keys with value of 0 are removed
        """

        non_zero_dic = dict()
        for key in dic.keys():
            if dic[key] != 0:
                non_zero_dic[key] = dic[key]

        return non_zero_dic
    
    @staticmethod
    def jacobian_from_dict(dic):
        """Calculates the Jacobian from the qgs tensor

        Parameters
        ----------
        dic: dict
            dictionary of tendencies of the model
        Returns
        -------
        dic_jac: dict
            Jacobian tensor stored in a dictionary
        """

        rank = max([len(i) for i in dic.keys()])
        n_perm = rank - 2
        
        orig_order = [i for i in range(rank)]
        
        keys = dic.keys()
        dic_jac = dic.copy()
        
        for i in range(1, n_perm+1):
            new_pos = orig_order.copy()
            new_pos[1] = orig_order[i+1]
            new_pos[i+1] = orig_order[1]
            for key in keys:
                dic_jac = add_to_dict(dic_jac, tuple(key[i] for i in new_pos), dic[key])
        
        return dic_jac
    
    @staticmethod
    def simplify_dict(dic):
        """calculates the upper triangular tensor of a given tensor, stored in dictionary

        Parameters
        ----------
        dic: dict
            dictionary of tendencies of the model

        Returns
        -------
        dic_upp: dict
            Upper triangular tensor, stored as a tensor where the keys are the coordinates of the corresponding value.
        """

        keys = dic.keys()
        dic_upp = dict()

        for key in keys:
            new_key = tuple([key[0]] + sorted(key[1:]))
            dic_upp = add_to_dict(dic_upp, new_key, dic[key])

        return dic_upp
    
    def save_to_file(self, filename, **kwargs):
        """Function to save the tensor object to a file with the :mod:`pickle` module.

        Parameters
        ----------
        filename: str
            The file name where to save the tensor object.
        kwargs: dict
            Keyword arguments to pass to the :mod:`pickle` module method.
        """
        f = open(filename, 'wb')
        pickle.dump(self.__dict__, f, **kwargs)
        f.close()

    def sub_tensor(self, tensor=None, continuation_variables=None):
        """Uses sympy substitution to convert the symbolic tensor or a symbolic dictionary to a numerical one.

        Parameters
        ----------
        tensor: dict(~sympy.core.expr.Expr or float) or ~sympy.tensor.array.ImmutableSparseNDimArray
            Tensor of model tendencies, either as

            - a dictionary with keys of non-zero coordinates, and values of Sympy expressions or floats
            - or a sparse Sympy tensor

        continuation_variables: list(Parameter, ScalingParameter or ParametersArray) or `None`
            Variables which remain symbolic, all other variables are substituted with numerical values.
            If `None` all variables are substituted.

        Returns
        -------
        ten_out: dict(float)
            Dictionary of the substituted tensor of the model tendencies, with coordinates and numerical values
        """

        if continuation_variables is None:
            continuation_variables = list()

        param_subs = _parameter_substitutions(self.params, continuation_variables)
        
        if tensor is None:
            ten = self.tensor_dic
        else:
            ten = tensor

        if isinstance(ten, dict):
            ten_out = dict()
            for key in ten.keys():
                val = ten[key].subs(param_subs)
                try:
                    ten_out[key] = float(val)
                except:
                    ten_out[key] = val

        else:
            # Assuming the tensor is a sympy tensor
            ten_out = ten.subs(param_subs)

        return ten_out
            
    def print_tensor(self, tensor=None, dict_opp=True, tol=1e-10):
        """Print the non-zero coordinates of values of the tensor of the model tendencies

        Parameters
        ----------
        tensor: dict(~sympy.core.expr.Expr or float) or ~sympy.tensor.array.ImmutableSparseNDimArray or `None`
            Tensor of model tendencies, either as

            - a dictionary with keys of non-zero coordinates, and values of Sympy expressions or floats
            - or a sparse Sympy tensor

            If `None`, defaults to the stored tensor.
            Defaults to `None`.

        dict_opp: bool
            If `True`, returns the unsimplified symbolic expressions, if `False` the simplified expressions are returned.
        tol: float
            The tolerance to allow for numerical errors when finding non-zero values.
        """

        if tensor is None:
            if dict_opp:
                temp_ten = self.tensor_dic
            else:
                temp_ten = self.tensor
        else:
            temp_ten = tensor

        if isinstance(temp_ten, dict):
            val_list = [(key, temp_ten[key]) for key in temp_ten.keys()] 
        else:
            val_list = np.ndenumerate(temp_ten)

        for ix, v in val_list:
            if isinstance(v, float):
                bool_test = (abs(v) > tol)
            else:
                bool_test = (v != 0)

            if bool_test:
                try:
                    output_val = float(v)
                except:
                    try:
                        output_val = v.simplify().evalf()
                    except:
                        output_val = v
                print(str(ix) + ": " + str(output_val))


class SymbolicQgsTensorDynamicT(SymbolicQgsTensor):
    """qgs dynamical temperature first order (linear) symbolic tendencies tensor class.

    Parameters
    ----------
    params: None or QgParams, optional
        The models parameters to configure the tensor. `None` to initialize an empty tensor. Default to `None`.
    atmospheric_inner_products: None or AtmosphericInnerProducts, optional
        The inner products of the atmospheric basis functions on which the model's PDE atmospheric equations are projected.
        If `None`, disable the atmospheric tendencies. Default to `None`.
    oceanic_inner_products: None or OceanicInnerProducts, optional
        The inner products of the oceanic basis functions on which the model's PDE oceanic equations are projected.
        If `None`, disable the oceanic tendencies. Default to `None`.
    ground_inner_products: None or GroundInnerProducts, optional
        The inner products of the ground basis functions on which the model's PDE ground equations are projected.
        If `None`, disable the ground tendencies. Default to `None`.

    Attributes
    ----------
    params: None or QgParams
        The models parameters used to configure the tensor. `None` for an empty tensor.
    atmospheric_inner_products: None or AtmosphericInnerProducts
        The inner products of the atmospheric basis functions on which the model's PDE atmospheric equations are projected.
        If `None`, disable the atmospheric tendencies. Default to `None`.
    oceanic_inner_products: None or OceanicInnerProducts
        The inner products of the oceanic basis functions on which the model's PDE oceanic equations are projected.
        If `None`, disable the oceanic tendencies. Default to `None`.
    ground_inner_products: None or GroundInnerProducts
        The inner products of the ground basis functions on which the model's PDE ground equations are projected.
        If `None`, disable the ground tendencies. Default to `None`.
    tensor: ~sympy.tensor.array.ImmutableSparseNDimArray
        The tensor :math:`\\mathcal{T}_{i,j,k}` :math:`i`-th components.
    jacobian_tensor: ~sympy.tensor.array.ImmutableSparseNDimArray
        The jacobian tensor :math:`\\mathcal{T}_{i,j,k} + \\mathcal{T}_{i,k,j}` :math:`i`-th components.
    """

    def __init__(self, params=None, atmospheric_inner_products=None, oceanic_inner_products=None, ground_inner_products=None):

        SymbolicQgsTensor.__init__(self, params, atmospheric_inner_products, oceanic_inner_products, ground_inner_products)
        
        if params.dynamic_T:
            self.compute_tensor()

    def _compute_tensor_dicts(self):

        if self.params is None:
            return None

        if self.atmospheric_inner_products is None and self.oceanic_inner_products is None \
                and self.ground_inner_products is None:
            return None

        aips = self.atmospheric_inner_products

        bips = None
        if self.oceanic_inner_products is not None:
            bips = self.oceanic_inner_products

        elif self.ground_inner_products is not None:
            bips = self.ground_inner_products

        if bips is not None:
            go = bips.stored
        else:
            go = True

        if aips.stored and go:
            symbolic_array_full_dict = self._compute_stored_full_dict()

        else:
            symbolic_array_full_dict = self._compute_non_stored_full_dict()

        return symbolic_array_full_dict

    def _compute_stored_full_dict(self):
        par = self.params
        nvar = par.number_of_variables
        aips = self.atmospheric_inner_products

        bips = None
        if self.oceanic_inner_products is not None:
            bips = self.oceanic_inner_products
            ocean = True
        else:
            ocean = False

        if self.ground_inner_products is not None:
            bips = self.ground_inner_products
            ground_temp = True
        else:
            ground_temp = False

        # constructing some derived matrices
        if aips is not None:

            a_theta = dict()
            for i in range(nvar[1]):
                for j in range(nvar[1]):
                    a_theta[(i, j)] = par.atmospheric_params.sig0.symbol * aips.a(i, j) - aips.u(i, j)
            a_theta = ImmutableSparseMatrix(nvar[1], nvar[1], a_theta)
            a_theta = a_theta.inverse()

        if bips is not None:
            U_inv = dict()
            if ocean:
                for i in range(nvar[3]):
                    for j in range(nvar[3]):
                        U_inv[(i, j)] = bips.U(i, j)
                U_inv = ImmutableSparseMatrix(nvar[3], nvar[3], U_inv)
                U_inv = U_inv.inverse()
            else:
                for i in range(nvar[2]):
                    for j in range(nvar[2]):
                        U_inv[i, j] = bips.U(i, j)
                U_inv = ImmutableSparseMatrix(nvar[2], nvar[2], U_inv)
                U_inv = U_inv.inverse()

        #################

        sy_arr_dic = dict()
        # theta_a part
        for i in range(nvar[1]):
            if self.params.T4LSBpa is not None:
                j = k = ell = 0
                for m in range(nvar[1]):
                    val = 0
                    for jj in range(nvar[1]):
                        val += a_theta[i, jj] * aips._z[jj, j, k, ell, m]
                    if m == 0:
                        sy_arr_dic = add_to_dict(sy_arr_dic, (self._theta_a(i), self._theta_a(j), self._theta_a(k), self._theta_a(ell), self._theta_a(m)), self.params.T4LSBpa.symbolic_expression * val)
                    else:
                        sy_arr_dic = add_to_dict(sy_arr_dic, (self._theta_a(i), self._theta_a(j), self._theta_a(k), self._theta_a(ell), self._theta_a(m)), 4 * self.params.T4LSBpa.symbolic_expression * val)
                        
            if ocean:
                if self.params.T4LSBpgo is not None:
                    j = k = ell = 0
                    for m in range(nvar[3]):
                        val = 0
                        for jj in range(nvar[1]):
                            val += a_theta[i, jj] * aips._v[jj, j, k, ell, m]
                        if m == 0:
                            sy_arr_dic = add_to_dict(sy_arr_dic, (self._theta_a(i), self._deltaT_o(j), self._deltaT_o(k), self._deltaT_o(ell), self._deltaT_o(m)), -self.params.T4LSBpgo.symbolic_expression * val)
                        else:
                            sy_arr_dic = add_to_dict(sy_arr_dic, (self._theta_a(i), self._deltaT_o(j), self._deltaT_o(k), self._deltaT_o(ell), self._deltaT_o(m)), -4 * self.params.T4LSBpgo.symbolic_expression * val)

            if ground_temp:
                if self.params.T4LSBpgo is not None:
                    j = k = ell = 0
                    for m in range(nvar[2]):
                        val = 0
                        for jj in range(nvar[1]):
                            val += a_theta[i, jj] * aips._v[jj, j, k, ell, m]
                        if m == 0:
                            sy_arr_dic = add_to_dict(sy_arr_dic, (self._theta_a(i), self._deltaT_g(j), self._deltaT_g(k), self._deltaT_g(ell), self._deltaT_g(m)), -self.params.T4LSBpgo.symbolic_expression * val)
                        else:
                            sy_arr_dic = add_to_dict(sy_arr_dic, (self._theta_a(i), self._deltaT_g(j), self._deltaT_g(k), self._deltaT_g(ell), self._deltaT_g(m)), -4 * self.params.T4LSBpgo.symbolic_expression * val)

        if ocean:
            # delta_T part
            for i in range(nvar[3]):
                j = k = ell = 0
                for m in range(nvar[1]):
                    val = 0
                    for jj in range(nvar[3]):
                        val += U_inv[i, jj] * bips._Z[jj, j, k, ell, m]
                    if m == 0:
                        sy_arr_dic = add_to_dict(sy_arr_dic, (self._deltaT_o(i), self._theta_a(j), self._theta_a(k), self._theta_a(ell), self._theta_a(m)), self.params.T4sbpa.symbolic_expression * val)
                    else:
                        sy_arr_dic = add_to_dict(sy_arr_dic, (self._deltaT_o(i), self._theta_a(j), self._theta_a(k), self._theta_a(ell), self._theta_a(m)), 4 * self.params.T4sbpa.symbolic_expression * val)
                    
                for m in range(nvar[3]):
                    val = 0
                    for jj in range(nvar[3]):
                        val += U_inv[i, jj] * bips._V[jj, j, k, ell, m]
                    if m == 0:
                        sy_arr_dic = add_to_dict(sy_arr_dic, (self._deltaT_o(i), self._deltaT_o(j), self._deltaT_o(k), self._deltaT_o(ell), self._deltaT_o(m)), - self.params.T4sbpgo.symbolic_expression * val)
                    else:
                        sy_arr_dic = add_to_dict(sy_arr_dic, (self._deltaT_o(i), self._deltaT_o(j), self._deltaT_o(k), self._deltaT_o(ell), self._deltaT_o(m)), -4 * self.params.T4sbpgo.symbolic_expression * val)

        if ground_temp:
            # deltaT_g part
            for i in range(nvar[2]):
                j = k = ell = 0
                for m in range(nvar[1]):
                    val = 0
                    for jj in range(nvar[2]):
                        val += U_inv[i, jj] * bips._Z[jj, j, k, ell, m]
                    if m == 0:
                        sy_arr_dic = add_to_dict(sy_arr_dic, (self._deltaT_g(i), self._theta_a(j), self._theta_a(k), self._theta_a(ell), self._theta_a(m)), self.params.T4sbpa.symbolic_expression * val)
                    else:
                        sy_arr_dic = add_to_dict(sy_arr_dic, (self._deltaT_g(i), self._theta_a(j), self._theta_a(k), self._theta_a(ell), self._theta_a(m)), 4 * self.params.T4sbpa.symbolic_expression * val)

                for m in range(nvar[2]):
                    val = 0 
                    for jj in range(nvar[2]):
                        val += U_inv[i, jj] * bips._V[jj, j, k, ell, m]
                    if m == 0:
                        sy_arr_dic = add_to_dict(sy_arr_dic, (self._deltaT_g(i), self._deltaT_g(j), self._deltaT_g(k), self._deltaT_g(ell), self._deltaT_g(m)), -self.params.T4sbpgo.symbolic_expression * val)
                    else:
                        sy_arr_dic = add_to_dict(sy_arr_dic, (self._deltaT_g(i), self._deltaT_g(j), self._deltaT_g(k), self._deltaT_g(ell), self._deltaT_g(m)), -4 * self.params.T4sbpgo.symbolic_expression * val)
                    
        return sy_arr_dic

    def _compute_non_stored_full_dict(self):
        par = self.params
        nvar = par.number_of_variables
        aips = self.atmospheric_inner_products

        bips = None
        if self.oceanic_inner_products is not None:
            bips = self.oceanic_inner_products
            ocean = True
        else:
            ocean = False

        if self.ground_inner_products is not None:
            bips = self.ground_inner_products
            ground_temp = True
        else:
            ground_temp = False

        # constructing some derived matrices
        if aips is not None:
            a_theta = dict()
            for i in range(nvar[1]):
                for j in range(nvar[1]):
                    a_theta[(i, j)] = par.atmospheric_params.sig0.symbol * aips.a(i, j) - aips.u(i, j)

            a_theta = ImmutableSparseMatrix(nvar[1], nvar[1], a_theta)
            a_theta = a_theta.inverse()

        if bips is not None:
            if ocean:
                U_inv = dict()
                for i in range(nvar[3]):
                    for j in range(nvar[3]):
                        U_inv[i, j] = bips.U(i, j)
                U_inv = ImmutableSparseMatrix(nvar[3], nvar[3], U_inv)

            else:
                U_inv = dict()
                for i in range(nvar[2]):
                    for j in range(nvar[2]):
                        U_inv[(i, j)] = bips.U(i, j)
                U_inv = ImmutableSparseMatrix(nvar[2], nvar[2], U_inv)
            
            U_inv = U_inv.inverse()
                
        #################

        sy_arr_dic = dict()
        # theta_a part
        for i in range(nvar[1]):

            if self.params.T4LSBpa is not None:
                j = k = ell = 0
                for m in range(nvar[1]):
                    val = 0
                    for jj in range(nvar[1]):
                        val += a_theta[i, jj] * aips.z(jj, j, k, ell, m)
                    if m == 0:
                        sy_arr_dic[(self._theta_a(i), self._theta_a(j), self._theta_a(k), self._theta_a(ell), self._theta_a(m))] = self.params.T4LSBpa.symbolic_expression * val
                    else:
                        sy_arr_dic[(self._theta_a(i), self._theta_a(j), self._theta_a(k), self._theta_a(ell), self._theta_a(m))] = 4 * self.params.T4LSBpa.symbolic_expression * val

            if ocean:
                if self.params.T4LSBpgo is not None:
                    j = k = ell = 0
                    for m in range(nvar[3]):
                        val = 0
                        for jj in range(nvar[1]):
                            val -= a_theta[i, jj] * aips.v(jj, j, k, ell, m)
                        if m == 0:
                            sy_arr_dic[(self._theta_a(i), self._deltaT_o(j), self._deltaT_o(k), self._deltaT_o(ell), self._deltaT_o(m))] = self.params.T4LSBpgo.symbolic_expression * val
                        else:
                            sy_arr_dic[(self._theta_a(i), self._deltaT_o(j), self._deltaT_o(k), self._deltaT_o(ell), self._deltaT_o(m))] = 4 * self.params.T4LSBpgo.symbolic_expression * val

            if ground_temp:
                if self.params.T4LSBpgo is not None:
                    j = k = ell = 0
                    for m in range(nvar[2]):
                        val = 0
                        for jj in range(nvar[1]):
                            val -= a_theta[i, jj] * aips.v(jj, j, k, ell, m)
                        if m == 0:
                            sy_arr_dic[(self._theta_a(i), self._deltaT_g(j), self._deltaT_g(k), self._deltaT_g(ell), self._deltaT_g(m))] = self.params.T4LSBpgo.symbolic_expression * val
                        else:
                            sy_arr_dic[(self._theta_a(i), self._deltaT_g(j), self._deltaT_g(k), self._deltaT_g(ell), self._deltaT_g(m))] = 4 * self.params.T4LSBpgo.symbolic_expression * val

        if ocean:

            # deltaT_o part
            for i in range(nvar[3]):

                j = k = ell = 0
                for m in range(nvar[1]):
                    val = 0
                    for jj in range(nvar[3]):
                        val += U_inv[i, jj] * bips.Z(jj, j, k, ell, m)
                    if m == 0:
                        sy_arr_dic[(self._deltaT_o(i), self._theta_a(j), self._theta_a(k), self._theta_a(ell), self._theta_a(m))] = self.params.T4sbpa.symbolic_expression * val
                    else:
                        sy_arr_dic[(self._deltaT_o(i), self._theta_a(j), self._theta_a(k), self._theta_a(ell), self._theta_a(m))] = 4 * self.params.T4sbpa.symbolic_expression * val

                for m in range(nvar[3]):
                    val = 0
                    for jj in range(nvar[3]):
                        val -= U_inv[i, jj] * bips.V(jj, j, k, ell, m)
                    if m == 0:
                        sy_arr_dic[(self._deltaT_o(i), self._deltaT_o(j), self._deltaT_o(k), self._deltaT_o(ell), self._deltaT_o(m))] = self.params.T4sbpgo.symbolic_expression * val
                    else:
                        sy_arr_dic[(self._deltaT_o(i), self._deltaT_o(j), self._deltaT_o(k), self._deltaT_o(ell), self._deltaT_o(m))] = 4 * self.params.T4sbpgo.symbolic_expression * val

        # deltaT_g part
        if ground_temp:
            for i in range(nvar[2]):

                j = k = ell = 0
                for m in range(nvar[1]):
                    val = 0
                    for jj in range(nvar[2]):
                        val += U_inv[i, jj] * bips._Z[jj, j, k, ell, m]
                    if m == 0:
                        sy_arr_dic[(self._deltaT_g(i), self._theta_a(j), self._theta_a(k), self._theta_a(ell), self._theta_a(m))] = self.params.T4sbpa.symbolic_expression * val
                    else:
                        sy_arr_dic[(self._deltaT_g(i), self._theta_a(j), self._theta_a(k), self._theta_a(ell), self._theta_a(m))] = 4 * self.params.T4sbpa.symbolic_expression * val

                for m in range(nvar[2]):
                    val = 0
                    for jj in range(nvar[2]):
                        val -= U_inv[i, jj] * bips._V[jj, j, k, ell, m]
                    if m == 0:
                        sy_arr_dic[(self._deltaT_g(i), self._deltaT_g(j), self._deltaT_g(k), self._deltaT_g(ell), self._deltaT_g(m))] = self.params.T4sbpgo.symbolic_expression * val
                    else:
                        sy_arr_dic[(self._deltaT_g(i), self._deltaT_g(j), self._deltaT_g(k), self._deltaT_g(ell), self._deltaT_g(m))] = 4 * self.params.T4sbpgo.symbolic_expression * val

        return sy_arr_dic

    def compute_tensor(self):
        """Routine to compute the tensor."""
        if self.params.T4:
            # TODO: Make a proper error message for here
            raise ValueError("Parameters are set for T4 version, set dynamic_T=True")

        symbolic_dict_linear = SymbolicQgsTensor._compute_tensor_dicts(self)
        symbolic_dict_linear = _shift_dict_keys(symbolic_dict_linear, (0, 0))

        symbolic_dict_dynT = self._compute_tensor_dicts()

        if symbolic_dict_linear is not None:
            symbolic_dict_dynT = {**symbolic_dict_linear, **symbolic_dict_dynT}

        if symbolic_dict_dynT is not None:
            self._set_tensor(symbolic_dict_dynT)


class SymbolicQgsTensorT4(SymbolicQgsTensor):
    """qgs dynamical temperature first order (linear) symbolic tendencies tensor class.

    Parameters
    ----------
    params: None or QgParams, optional
        The models parameters to configure the tensor. `None` to initialize an empty tensor. Default to `None`.
    atmospheric_inner_products: None or AtmosphericInnerProducts, optional
        The inner products of the atmospheric basis functions on which the model's PDE atmospheric equations are projected.
        If `None`, disable the atmospheric tendencies. Default to `None`.
    oceanic_inner_products: None or OceanicInnerProducts, optional
        The inner products of the oceanic basis functions on which the model's PDE oceanic equations are projected.
        If `None`, disable the oceanic tendencies. Default to `None`.
    ground_inner_products: None or GroundInnerProducts, optional
        The inner products of the ground basis functions on which the model's PDE ground equations are projected.
        If `None`, disable the ground tendencies. Default to `None`.

    Attributes
    ----------
    params: None or QgParams
        The models parameters used to configure the tensor. `None` for an empty tensor.
    atmospheric_inner_products: None or AtmosphericInnerProducts
        The inner products of the atmospheric basis functions on which the model's PDE atmospheric equations are projected.
        If `None`, disable the atmospheric tendencies. Default to `None`.
    oceanic_inner_products: None or OceanicInnerProducts
        The inner products of the oceanic basis functions on which the model's PDE oceanic equations are projected.
        If `None`, disable the oceanic tendencies. Default to `None`.
    ground_inner_products: None or GroundInnerProducts
        The inner products of the ground basis functions on which the model's PDE ground equations are projected.
        If `None`, disable the ground tendencies. Default to `None`.
    tensor: ~sympy.tensor.array.ImmutableSparseNDimArray
        The tensor :math:`\\mathcal{T}_{i,j,k}` :math:`i`-th components.
    jacobian_tensor: ~sympy.tensor.array.ImmutableSparseNDimArray
        The jacobian tensor :math:`\\mathcal{T}_{i,j,k} + \\mathcal{T}_{i,k,j}` :math:`i`-th components.
    """

    def __init__(self, params=None, atmospheric_inner_products=None, oceanic_inner_products=None, ground_inner_products=None):

        SymbolicQgsTensor.__init__(self, params, atmospheric_inner_products, oceanic_inner_products, ground_inner_products)
        
        if params.T4:
            self.compute_tensor()

    def _compute_non_stored_full_dict(self):
        par = self.params
        nvar = par.number_of_variables
        aips = self.atmospheric_inner_products

        bips = None
        if self.oceanic_inner_products is not None:
            bips = self.oceanic_inner_products
            ocean = True
        else:
            ocean = False

        if self.ground_inner_products is not None:
            bips = self.ground_inner_products
            ground_temp = True
        else:
            ground_temp = False

        # constructing some derived matrices
        if aips is not None:
            a_theta = dict()
            for i in range(nvar[1]):
                for j in range(nvar[1]):
                    a_theta[(i, j)] = par.atmospheric_params.sig0.symbol * aips.a(i, j) - aips.u(i, j)

            a_theta = ImmutableSparseMatrix(nvar[1], nvar[1], a_theta)
            a_theta = a_theta.inverse()

        if bips is not None:
            if ocean:
                U_inv = dict()
                for i in range(nvar[3]):
                    for j in range(nvar[3]):
                        U_inv[i, j] = bips.U(i, j)
                U_inv = ImmutableSparseMatrix(nvar[3], nvar[3], U_inv)

            else:
                U_inv = dict()
                for i in range(nvar[2]):
                    for j in range(nvar[2]):
                        U_inv[(i, j)] = bips.U(i, j)
                U_inv = ImmutableSparseMatrix(nvar[2], nvar[2], U_inv)
            
            U_inv = U_inv.inverse()
                
        #################

        sy_arr_dic = dict()
        # theta_a part
        for i in range(nvar[1]):

            if self.params.T4LSBpa is not None:
                for j in range(nvar[1]):
                    for k in range(nvar[1]):
                        for ell in range(nvar[1]):
                            for m in range(nvar[1]):
                                val = 0
                                for jj in range(nvar[1]):
                                    val += a_theta[i, jj] * aips.z(jj, j, k, ell, m)
                                
                                sy_arr_dic[(self._theta_a(i), self._theta_a(j), self._theta_a(k), self._theta_a(ell), self._theta_a(m))] = self.params.T4LSBpa.symbolic_expression * val
                    
            if ocean:
                if self.params.T4LSBpgo is not None:
                    for j in range(nvar[3]):
                        for k in range(nvar[3]):
                            for ell in range(nvar[3]):
                                for m in range(nvar[3]):
                                    val = 0
                                    for jj in range(nvar[1]):
                                        val -= a_theta[i, jj] * aips.v(jj, j, k, ell, m)
                                    
                                    sy_arr_dic[(self._theta_a(i), self._deltaT_o(j), self._deltaT_o(k), self._deltaT_o(ell), self._deltaT_o(m))] = self.params.T4LSBpgo.symbolic_expression * val

            if ground_temp:
                if self.params.T4LSBpgo is not None:
                    for j in range(nvar[2]):
                        for k in range(nvar[2]):
                            for ell in range(nvar[2]):
                                for m in range(nvar[2]):
                                    val = 0
                                    for jj in range(nvar[1]):
                                        val -= a_theta[i, jj] * aips.v(jj, j, k, ell, m)

                                    sy_arr_dic[(self._theta_a(i), self._deltaT_g(j), self._deltaT_g(k), self._deltaT_g(ell), self._deltaT_g(m))] = self.params.T4LSBpgo.symbolic_expression * val
                        
        if ocean:

            # deltaT_o part
            for i in range(nvar[3]):
                for j in range(nvar[1]):
                    for k in range(nvar[1]):
                        for ell in range(nvar[1]):
                            for m in range(nvar[1]):
                                val = 0
                                for jj in range(nvar[3]):
                                    val += U_inv[i, jj] * bips.Z(jj, j, k, ell, m)
                                
                                sy_arr_dic[(self._deltaT_o(i), self._theta_a(j), self._theta_a(k), self._theta_a(ell), self._theta_a(m))] = self.params.T4LSBpa.symbolic_expression * val
                
                for j in range(nvar[3]):
                    for k in range(nvar[3]):
                        for ell in range(nvar[3]):
                            for m in range(nvar[3]):
                                val = 0
                                for jj in range(nvar[3]):
                                    val -= U_inv[i, jj] * bips.V(jj, j, k, ell, m)

                                sy_arr_dic[(self._deltaT_o(i), self._deltaT_o(j), self._deltaT_o(k), self._deltaT_o(ell), self._deltaT_o(m))] = self.params.T4LSBpgo.symbolic_expression * val

        # deltaT_g part
        if ground_temp:
            for i in range(nvar[2]):
                for j in range(nvar[1]):
                    for k in range(nvar[1]):
                        for ell in range(nvar[1]):
                            for m in range(nvar[1]):
                                val = 0
                                for jj in range(nvar[2]):
                                    val += U_inv[i, jj] * bips._Z[jj, j, k, ell, m]
                                
                                sy_arr_dic[(self._deltaT_g(i), self._theta_a(j), self._theta_a(k), self._theta_a(ell), self._theta_a(m))] = self.params.T4LSBpa.symbolic_expression * val
                
                for j in range(nvar[2]):
                    for k in range(nvar[2]):
                        for ell in range(nvar[2]):    
                            for m in range(nvar[2]):
                                val = 0
                                for jj in range(nvar[2]):
                                    val -= U_inv[i, jj] * bips._V[jj, j, k, ell, m]

                                sy_arr_dic[(self._deltaT_g(i), self._deltaT_g(j), self._deltaT_g(k), self._deltaT_g(ell), self._deltaT_g(m))] = self.params.T4LSBpgo.symbolic_expression * val
                   
        return sy_arr_dic

    def compute_tensor(self):
        """Routine to compute the tensor."""
        if not self.params.T4:
            raise ValueError("Parameters are not set for T4 version")

        symbolic_dict_linear = SymbolicQgsTensor._compute_tensor_dicts(self)
        symbolic_dict_linear = _shift_dict_keys(symbolic_dict_linear, (0, 0))

        symbolic_dict_T4 = self._compute_non_stored_full_dict()

        if symbolic_dict_linear is not None:
            symbolic_dict_T4 = {**symbolic_dict_linear, **symbolic_dict_T4}
        
        if symbolic_dict_T4 is not None:
            self._set_tensor(symbolic_dict_T4)


def _kronecker_delta(i, j):
    if i == j:
        return 1
    else:
        return 0


def _shift_dict_keys(dic, shift):
    """
    Keys of given dictionary are altered to add values in the given indicies

    Parameters
    ----------
    dic: dict
        Dictionary representing a tensor.

    shift: tuple
        A tuple that represents the shift in the tensor indices.
    """

    shifted_dic = dict()
    for key in dic.keys():
        new_key = key + shift
        shifted_dic[new_key] = dic[key]
    
    return shifted_dic


def _parameter_substitutions(params, continuation_variables):
    """
    Returns a dict of parameters values that are to be substituted,
    removing the parameters given in `continuation_variables`.
    """

    subs = params._all_items

    if continuation_variables is None:
        continuation_variables = list()

    # Remove variables in continuation variables
    for cv in continuation_variables:
        if isinstance(cv, ParametersArray):
            for cv_i in cv:
                subs.remove(cv_i)
        elif isinstance(cv, Parameter):
            subs.remove(cv)
        else:  # Try ... who knows...
            subs.remove(cv)

    # make the remaining items into a dict to pass to sympy subs function
    sub_dic = {}
    for p in subs:
        if p.symbol is not None:
            sub_dic[p.symbol] = float(p)
    return sub_dic


if __name__ == "__main__":
    dic = dict()
    dic = add_to_dict(dic, (0, 0), 1)
    dic = add_to_dict(dic, (0, 0), 2)
    print(dic)

    from qgs.params.params import QgParams
    from qgs.inner_products import symbolic

    params = QgParams({'rr': 287.e0, 'sb': 5.6e-8})
    params.set_atmospheric_channel_fourier_modes(6, 6, mode="symbolic")
    params.atmospheric_params.set_params({'sigma': 0.2, 'kd': 0.1, 'kdp': 0.01})

    params.ground_params.set_orography(0.2, 1)
    params.atemperature_params.set_thetas(0.1, 0)

    aip = symbolic.AtmosphericSymbolicInnerProducts(params, return_symbolic=True, make_substitution=True)

    # sym_aotensor = SymbolicQgsTensor(params=params, atmospheric_inner_products=aip)
