"""
    symbolic qgs tensor module
    =================

    This module computes and holds the symbolic representation of the tensors representing the tendencies of the model's equations.

"""
from qgs.functions.symbolic_mul import _add_to_dict, _symbolic_tensordot
from qgs.params.params import Parameter, ScalingParameter, ParametersArray

import numpy as np
import sympy as sy
import pickle

from sympy.matrices.immutable import ImmutableSparseMatrix
from sympy.tensor.array import ImmutableSparseNDimArray

#//TODO: Check non stored IP version of this

class SymbolicTensorLinear(object):
    """Symbolic qgs tendencies tensor class.

    Parameters
    ----------
    params: None or QgParams, optional
        The models parameters to configure the tensor. `None` to initialize an empty tensor. Default to `None`.
    atmospheric_inner_products: None or AtmosphericInnerProducts, optional
        The inner products of the atmospheric basis functions on which the model's PDE atmospheric equations are projected.
        If None, disable the atmospheric tendencies. Default to `None`.
        The inner product is returned in symbolic or numeric form.
    oceanic_inner_products: None or OceanicInnerProducts, optional
        The inner products of the oceanic basis functions on which the model's PDE oceanic equations are projected.
        If None, disable the oceanic tendencies. Default to `None`.
        The inner product is returned in symbolic or numeric form.
    ground_inner_products: None or GroundInnerProducts, optional
        The inner products of the ground basis functions on which the model's PDE ground equations are projected.
        If None, disable the ground tendencies. Default to `None`.
        The inner product is returned in symbolic or numeric form.

    Attributes
    ----------
    params: None or QgParams
        The models parameters used to configure the tensor. `None` for an empty tensor.
    atmospheric_inner_products: None or AtmosphericInnerProducts
        The inner products of the atmospheric basis functions on which the model's PDE atmospheric equations are projected.
        If None, disable the atmospheric tendencies. Default to `None`.
    oceanic_inner_products: None or OceanicInnerProducts
        The inner products of the oceanic basis functions on which the model's PDE oceanic equations are projected.
        If None, disable the oceanic tendencies. Default to `None`.
    ground_inner_products: None or GroundInnerProducts
        The inner products of the ground basis functions on which the model's PDE ground equations are projected.
        If None, disable the ground tendencies. Default to `None`.
    tensor: sparse.COO(float)
        The tensor :math:`\\mathcal{T}_{i,j,k}` :math:`i`-th components.
    jacobian_tensor: sparse.COO(float)
        The jacobian tensor :math:`\\mathcal{T}_{i,j,k} + \\mathcal{T}_{i,k,j}` :math:`i`-th components.
    """

    def __init__(self, params=None, atmospheric_inner_products=None, oceanic_inner_products=None, ground_inner_products=None):

        self.atmospheric_inner_products = atmospheric_inner_products
        self.oceanic_inner_products = oceanic_inner_products
        self.ground_inner_products = ground_inner_products
        self.params = params

        self.tensor = None
        self.jacobian_tensor = None

        if not(self.params.dynamic_T):
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
    
    #//TODO: Im not happy with having these set of properties in two places, one for numerics and one for symbolic. This should be combined, or at least put in the parameter section somewhere.

    @property
    def LR(self):
        if self.params.oceanic_params.gp is None or self.params.oceanic_params.h is None:
            return None
        else:
            return sy.sqrt(self.params.oceanic_params.gp.symbol * self.params.oceanic_params.h.symbol) / self.params.scale_params.f0.symbol

    @property
    def G(self):
        return -self.params.scale_params.L.symbol ** 2 / self.LR ** 2
    
    @property
    def Cpgo(self):
        if self.params.gotemperature_params.C is None:
            return None
        else:
            C = ImmutableSparseNDimArray(self.params.gotemperature_params.C.symbols)
            return C / (self.params.gotemperature_params.gamma.symbol * self.params.scale_params.f0.symbol) * self.params.rr.symbol / (self.params.scale_params.f0.symbol ** 2 * self.params.scale_params.L.symbol ** 2)
    
    @property
    def Lpgo(self):
        if self.params.atemperature_params.hlambda is None or self.params.gotemperature_params.gamma is None:
            return None
        else:
            return self.params.atemperature_params.hlambda.symbol / (self.params.gotemperature_params.gamma.symbol * self.params.scale_params.f0.symbol)
    
    @property
    def Cpa(self):
        if self.params.atemperature_params.C is None:
            return None
        else:
            C = ImmutableSparseNDimArray(self.params.atemperature_params.C.symbols)
            return C / (self.params.atemperature_params.gamma.symbol * self.params.scale_params.f0.symbol) * self.params.rr.symbol / (self.params.scale_params.f0.symbol ** 2 * self.params.scale_params.L.symbol ** 2) / 2
    
    @property
    def Lpa(self):
        if self.params.atemperature_params.hlambda is None or self.params.atemperature_params.gamma is None:
            return None
        else:
            return self.params.atemperature_params.hlambda.symbol / (self.params.atemperature_params.gamma.symbol * self.params.scale_params.f0.symbol)
    
    @property
    def sbpgo(self):
        if self.params.gotemperature_params.T0 is None:
            None
        else:
            return 4 * self.params.sb.symbol * self.params.gotemperature_params.T0.symbol ** 3 / (self.params.gotemperature_params.gamma.symbol * self.params.scale_params.f0.symbol)
    
    @property
    def sbpa(self):
        if self.params.atemperature_params.T0 is None:
            return None
        else:
            return 8 * self.params.atemperature_params.eps.symbol * self.params.sb.symbol * self.params.atemperature_params.T0.symbol ** 3 / (self.params.gotemperature_params.gamma.symbol * self.params.scale_params.f0.symbol)
    
    @property
    def LSBpgo(self):
        if self.params.gotemperature_params.T0 is None:
            None
        else:
            return 2 * self.params.atemperature_params.eps.symbol * self.params.sb.symbol * self.params.gotemperature_params.T0.symbol ** 3 / (self.params.atemperature_params.gamma.symbol * self.params.scale_params.f0.symbol)
    
    @property
    def LSBpa(self):
        if self.params.atemperature_params.T0 is None:
            return None
        else:
            return 8 * self.params.atemperature_params.eps.symbol * self.params.sb.symbol * self.params.atemperature_params.T0.symbol ** 3 / (self.params.atemperature_params.gamma.symbol * self.params.scale_params.f0.symbol)
    
    @property
    def T4sbpgo(self):
        return self.params.sb.symbol * self.params.scale_params.L.symbol ** 6 * self.params.scale_params.f0.symbol ** 5 / (self.params.gotemperature_params.gamma.symbol * self.params.rr.symbol ** 3)
    
    @property
    def T4sbpa(self):
        return 16 * self.params.atemperature_params.eps.symbol * self.params.sb.symbol * self.params.scale_params.L.symbol ** 6 * self.params.scale_params.f0.symbol ** 5 / (self.params.gotemperature_params.gamma.symbol * self.params.rr.symbol ** 3)
    
    @property
    def T4LSBpgo(self):
        return 0.5 * self.params.atemperature_params.eps.symbol * self.params.sb.symbol * self.params.scale_params.L.symbol ** 6 * self.params.scale_params.f0.symbol ** 5 / (self.params.atemperature_params.gamma.symbol * self.params.rr.symbol ** 3)
    
    @property
    def T4LSBpa(self):
        return 16 * self.params.atemperature_params.eps.symbol * self.params.sb.symbol * self.params.scale_params.L.symbol ** 6 * self.params.scale_params.f0.symbol ** 5 / (self.params.atemperature_params.gamma.symbol * self.params.rr.symbol ** 3)
    
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
                        M_psio[(i - offset, j - offset)] = bips.M(i, j) + self.G * bips.U(i, j)

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
            a_inv_mult_c = _symbolic_tensordot(a_inv, aips._c[offset:, offset:], axes=1)

            if gp is not None:
                hk_sym_arr = ImmutableSparseNDimArray(gp.hk.symbols)

                if gp.orographic_basis == "atmospheric":
                    a_inv_mult_g = _symbolic_tensordot(a_inv, aips._g[offset:, offset:, offset:], axes=1)
                    oro = _symbolic_tensordot(a_inv_mult_g, hk_sym_arr, axes=1)
                else:
                    a_inv_mult_gh = _symbolic_tensordot(a_inv, aips._gh[offset:, offset:, offset:], axes=1)
                    oro = _symbolic_tensordot(a_inv_mult_gh, hk_sym_arr, axes=1)

            a_inv_mult_b = _symbolic_tensordot(a_inv, aips._b[offset:, offset:, offset:], axes=1)
            
            if ocean:
                a_inv_mult_d = a_inv @ aips._d[offset:, offset:]


            for i in range(nvar[0]):
                for j in range(nvar[0]):
                    jo = j + offset  # skipping the theta 0 variable if it exists
                    sy_arr_dic = _add_to_dict(sy_arr_dic, (self._psi_a(i), self._psi_a(j), 0), -a_inv_mult_c[i, j] * par.scale_params.beta.symbol)
                    sy_arr_dic = _add_to_dict(sy_arr_dic, (self._psi_a(i), self._psi_a(j), 0), -(ap.kd.symbol * _kronecker_delta(i, j)) / 2)
                    sy_arr_dic = _add_to_dict(sy_arr_dic, (self._psi_a(i), self._theta_a(jo), 0), (ap.kd.symbol * _kronecker_delta(i, j)) / 2)

                    if gp is not None:
                        # convert 
                        if gp.hk is not None:
                            sy_arr_dic = _add_to_dict(sy_arr_dic, (self._psi_a(i), self._psi_a(j), 0), -oro[i, j] / 2)
                            sy_arr_dic = _add_to_dict(sy_arr_dic, (self._psi_a(i), self._theta_a(jo), 0), oro[i, j] / 2)

                    for k in range(nvar[0]):
                        ko = k + offset  # skipping the theta 0 variable if it exists
                        sy_arr_dic = _add_to_dict(sy_arr_dic, (self._psi_a(i), self._psi_a(j), self._psi_a(k)), -a_inv_mult_b[i, j, k])
                        sy_arr_dic = _add_to_dict(sy_arr_dic, (self._psi_a(i), self._theta_a(jo), self._theta_a(ko)), -a_inv_mult_b[i, j, k])

                if ocean:
                    for j in range(nvar[2]):
                        sy_arr_dic = _add_to_dict(sy_arr_dic, (self._psi_a(i), self._psi_o(j), 0), a_inv_mult_d[i, j] * ap.kd.symbol / 2)
            
            # theta_a part
            a_theta_mult_u = _symbolic_tensordot(a_theta, aips._u, axes=1)
            if self.Cpa is not None:
                val_Cpa = _symbolic_tensordot(a_theta_mult_u , self.Cpa, axes=1)

            if atp.hd is not None and atp.thetas is not None:
                thetas_sym_arr = ImmutableSparseNDimArray(atp.thetas.symbols)
                val_thetas = _symbolic_tensordot(a_theta_mult_u, thetas_sym_arr, axes=1)  # not perfect

            a_theta_mult_a = _symbolic_tensordot(a_theta, aips._a[:, offset:], axes=1)
            a_theta_mult_c = _symbolic_tensordot(a_theta, aips._c[:, offset:], axes=1)
            
            a_theta_mult_g = _symbolic_tensordot(a_theta, aips._g[:, offset:, offset:], axes=1)

            if gp is not None:
                if gp.orographic_basis == "atmospheric":
                    oro = _symbolic_tensordot(a_theta_mult_g, hk_sym_arr, axes=1)
                else:
                    a_theta_mult_gh = _symbolic_tensordot(a_theta, aips._gh[:, offset:, offset:], axes=1)
                    oro = _symbolic_tensordot(a_theta_mult_gh, hk_sym_arr, axes=1)

            a_theta_mult_b = _symbolic_tensordot(a_theta, aips._b[:, offset:, offset:], axes=1)

            if ocean:
                a_theta_mult_d = _symbolic_tensordot(a_theta, aips._d[:, offset:], axes=1)
                a_theta_mult_s = _symbolic_tensordot(a_theta, aips._s, axes=1)

            if ground_temp:
                a_theta_mult_s = _symbolic_tensordot(a_theta, aips._s, axes=1)
                
            for i in range(nvar[1]):
                if self.Cpa is not None:
                    sy_arr_dic = _add_to_dict(sy_arr_dic, (self._theta_a(i), 0, 0), -val_Cpa[i])

                if atp.hd is not None and atp.thetas is not None:
                    sy_arr_dic = _add_to_dict(sy_arr_dic, (self._theta_a(i), 0, 0), -val_thetas[i] * atp.hd.symbol)

                for j in range(nvar[0]):

                    jo = j + offset  # skipping the theta 0 variable if it exists

                    val_2 = a_theta_mult_a[i, j] * ap.kd.symbol * ap.sig0.symbol / 2
                    val_3 = a_theta_mult_a[i, j] * (ap.kd.symbol / 2 + 2 * ap.kdp.symbol) * ap.sig0.symbol
                    sy_arr_dic = _add_to_dict(sy_arr_dic, (self._theta_a(i), self._psi_a(j), 0), val_2)
                    sy_arr_dic = _add_to_dict(sy_arr_dic, (self._theta_a(i), self._theta_a(jo), 0), -val_3)

                    sy_arr_dic = _add_to_dict(sy_arr_dic, (self._theta_a(i), self._theta_a(jo), 0), -a_theta_mult_c[i, j] * par.scale_params.beta.symbol * ap.sig0.symbol)

                    if gp is not None:
                        if gp.hk is not None:
                            sy_arr_dic = _add_to_dict(sy_arr_dic, (self._theta_a(i), self._theta_a(jo), 0), -ap.sig0.symbol * oro[i, j] / 2)
                            sy_arr_dic = _add_to_dict(sy_arr_dic, (self._theta_a(i), self._psi_a(j), 0), ap.sig0.symbol * oro[i, j] / 2)

                    for k in range(nvar[0]):
                        ko = k + offset  # skipping the theta 0 variable if it exists
                        sy_arr_dic = _add_to_dict(sy_arr_dic, (self._theta_a(i), self._psi_a(j), self._theta_a(ko)), - a_theta_mult_b[i, j, k] * ap.sig0.symbol)
                        sy_arr_dic = _add_to_dict(sy_arr_dic, (self._theta_a(i), self._theta_a(jo), self._psi_a(k)), - a_theta_mult_b[i, j, k] * ap.sig0.symbol)

                        sy_arr_dic = _add_to_dict(sy_arr_dic, (self._theta_a(i), self._psi_a(j), self._theta_a(ko)),  a_theta_mult_g[i, j, k])

                for j in range(nvar[1]):
                    if self.Lpa is not None:
                        sy_arr_dic = _add_to_dict(sy_arr_dic, (self._theta_a(i), self._theta_a(j), 0), a_theta_mult_u[i, j] * atp.sc.symbol * self.Lpa)
                    if self.LSBpa is not None:
                        sy_arr_dic = _add_to_dict(sy_arr_dic, (self._theta_a(i), self._theta_a(j), 0), a_theta_mult_u[i, j] * self.LSBpa)

                    if atp.hd is not None:
                        sy_arr_dic = _add_to_dict(sy_arr_dic, (self._theta_a(i), self._theta_a(j), 0), a_theta_mult_u[i, j] * atp.hd)

                if ocean:
                    for j in range(nvar[2]):
                        sy_arr_dic = _add_to_dict(sy_arr_dic, (self._theta_a(i), self._psi_o(j), 0), -a_theta_mult_d[i, j] * ap.sig0.symbol * ap.kd.symbol / 2)

                    if self.Lpa is not None:
                        for j in range(nvar[3]):
                            sy_arr_dic = _add_to_dict(sy_arr_dic, (self._theta_a(i), self._deltaT_o(j), 0), -a_theta_mult_s[i, j] * self.Lpa / 2)
                            if self.LSBpgo is not None:
                                sy_arr_dic = _add_to_dict(sy_arr_dic, (self._theta_a(i), self._deltaT_o(j), 0), -a_theta_mult_s[i, j] * self.LSBpgo)

                if ground_temp:
                    if self.Lpa is not None:
                        for j in range(nvar[2]):
                            sy_arr_dic = _add_to_dict(sy_arr_dic, (self._theta_a(i), self._deltaT_g(j), 0), -a_theta_mult_s[i, j] * self.Lpa / 2)
                            if self.LSBpgo is not None:
                                sy_arr_dic = _add_to_dict(sy_arr_dic, (self._theta_a(i), self._deltaT_g(j), 0), -a_theta_mult_s[i, j] * self.LSBpgo)

            if ocean:
                # psi_o part
                M_psio_mult_K = _symbolic_tensordot(M_psio, bips._K[offset:, offset:], axes=1)
                M_psio_mult_N = _symbolic_tensordot(M_psio, bips._N[offset:, offset:], axes=1)
                M_psio_mult_M = _symbolic_tensordot(M_psio, bips._M[offset:, offset:], axes=1)
                M_psio_mult_C = _symbolic_tensordot(M_psio, bips._C[offset:, offset:, offset:], axes=1)

                for i in range(nvar[2]):
                    for j in range(nvar[0]):
                        jo = j + offset  # skipping the theta 0 variable if it exists

                        sy_arr_dic = _add_to_dict(sy_arr_dic, (self._psi_o(i), self._psi_a(j), 0), M_psio_mult_K[i, j] * par.oceanic_params.d.symbol)
                        sy_arr_dic = _add_to_dict(sy_arr_dic, (self._psi_o(i), self._theta_a(jo), 0), -M_psio_mult_K[i, j] * par.oceanic_params.d.symbol)

                    for j in range(nvar[2]):
                        sy_arr_dic = _add_to_dict(sy_arr_dic, (self._psi_o(i), self._psi_o(j), 0), -M_psio_mult_N[i, j] * par.scale_params.beta.symbol)

                        sy_arr_dic = _add_to_dict(sy_arr_dic, (self._psi_o(i), self._psi_o(j), 0), -M_psio_mult_M[i, j] * (par.oceanic_params.r.symbol + par.oceanic_params.d.symbol))

                        for k in range(nvar[2]):
                            sy_arr_dic = _add_to_dict(sy_arr_dic, (self._psi_o(i), self._psi_o(j), self._psi_o(k)), - M_psio_mult_C[i, j, k])

                # deltaT_o part
                U_inv_mult_W = _symbolic_tensordot(U_inv, bips._W, axes=1)
                U_inv_mult_W_Cpgo = _symbolic_tensordot(U_inv_mult_W, self.Cpgo, axes=1)

                U_inv_mult_O = _symbolic_tensordot(U_inv, bips._O[:, offset:, offset:], axes=1)
                
                for i in range(nvar[3]):
                    sy_arr_dic = _add_to_dict(sy_arr_dic, (self._deltaT_o(i), 0, 0), U_inv_mult_W_Cpgo[i])

                    for j in range(nvar[1]):
                        sy_arr_dic = _add_to_dict(sy_arr_dic, (self._deltaT_o(i), self._theta_a(j), 0), U_inv_mult_W[i, j] * 2 * atp.sc.symbol * self.Lpgo)
                        if self.sbpa is not None:
                            sy_arr_dic = _add_to_dict(sy_arr_dic, (self._deltaT_o(i), self._theta_a(j), 0), U_inv_mult_W[i, j] * self.sbpa)

                    for j in range(nvar[3]):
                        sy_arr_dic = _add_to_dict(sy_arr_dic, (self._deltaT_o(i), self._deltaT_o(j), 0), - self.Lpgo * _kronecker_delta(i, j))
                        if self.sbpgo is not None:
                            sy_arr_dic = _add_to_dict(sy_arr_dic, (self._deltaT_o(i), self._deltaT_o(j), 0), - self.sbpgo * _kronecker_delta(i, j))

                    for j in range(nvar[2]):
                        for k in range(nvar[2]):
                            ko = k + offset  # skipping the T 0 variable if it exists
                            sy_arr_dic = _add_to_dict(sy_arr_dic, (self._deltaT_o(i), self._psi_o(j), self._deltaT_o(ko)), -U_inv_mult_O[i, j, k])

            # deltaT_g part
            if ground_temp:
                U_inv_mult_W = _symbolic_tensordot(U_inv, bips._W, axes=1)
                U_inv_mult_W_Cpgo = _symbolic_tensordot(U_inv_mult_W, self.Cpgo, axes=1)
                for i in range(nvar[2]):
                    sy_arr_dic = _add_to_dict(sy_arr_dic, (self._deltaT_g(i), 0, 0), U_inv_mult_W_Cpgo[i])

                    for j in range(nvar[1]):
                        sy_arr_dic = _add_to_dict(sy_arr_dic, (self._deltaT_g(i), self._theta_a(j), 0), U_inv_mult_W[i, j] * 2 * atp.sc.symbol * self.Lpgo)
                        if self.sbpa is not None:
                            sy_arr_dic = _add_to_dict(sy_arr_dic, (self._deltaT_g(i), self._theta_a(j), 0), U_inv_mult_W[i, j] * self.sbpa)

                    for j in range(nvar[2]):
                        sy_arr_dic = _add_to_dict(sy_arr_dic, (self._deltaT_g(i), self._deltaT_g(j), 0), - self.Lpgo * _kronecker_delta(i, j))
                        if self.sbpgo is not None:
                            sy_arr_dic = _add_to_dict(sy_arr_dic, (self._deltaT_g(i), self._deltaT_g(j), 0), - self.sbpgo * _kronecker_delta(i, j))

        else:
            # psi_a part
            for i in range(nvar[0]):
                for j in range(nvar[0]):

                    jo = j + offset

                    val = 0
                    for jj in range(nvar[0]):
                        val += a_inv[i, jj] * aips.c(offset + jj, jo)
                    sy_arr_dic = _add_to_dict(sy_arr_dic, (self._psi_a(i), self._psi_a(j), 0), - val * par.scale_params.beta.symbol)
                    sy_arr_dic = _add_to_dict(sy_arr_dic, (self._psi_a(i), self._psi_a(j), 0), - (ap.kd.symbol * _kronecker_delta(i, j)) / 2)
                    sy_arr_dic = _add_to_dict(sy_arr_dic, (self._psi_a(i), self._theta_a(jo), 0), (ap.kd.symbol * _kronecker_delta(i, j)) / 2)

                    if gp is not None:
                        hk_sym_arr = ImmutableSparseNDimArray(gp.hk.symbols)
                        if gp.hk is not None:
                            oro = 0
                            if gp.orographic_basis == "atmospheric":
                                for jj in range(nvar[0]):
                                    for kk in range(nvar[0]):
                                        oro += a_inv[i, jj] * aips.g(offset + jj, j, offset + kk) * ap.kd.symbol
                            else:
                                for jj in range(nvar[0]):
                                    for kk in range(nvar[0]):
                                        oro += a_inv[i, jj] * aips.gh(offset + jj, j, offset + kk) * hk_sym_arr[kk]
                            sy_arr_dic = _add_to_dict(sy_arr_dic, (self._psi_a(i), self._psi_a(j), 0), - oro / 2)
                            sy_arr_dic = _add_to_dict(sy_arr_dic, (self._psi_a(i), self._theta_a(jo), 0), oro / 2)

                    for k in range(nvar[0]):
                        ko = k + offset  # skipping the theta 0 variable if it exists
                        val = 0
                        for jj in range(nvar[0]):
                            val += a_inv[i, jj] * aips.b(offset + jj, jo, ko)
                        sy_arr_dic = _add_to_dict(sy_arr_dic, (self._psi_a(i), self._psi_a(j), self._psi_a(k)), - val)
                        sy_arr_dic = _add_to_dict(sy_arr_dic, (self._psi_a(i), self._theta_a(jo), self._theta_a(ko)), - val)
                if ocean:
                    for j in range(nvar[2]):
                        jo = j + offset
                        val = 0
                        for jj in range(nvar[0]):
                            val += a_inv[i, jj] * aips.d(offset + jj, jo)
                        sy_arr_dic = _add_to_dict(sy_arr_dic, (self._psi_a(i), self._psi_o(j), 0), val * ap.kd.symbol / 2)


            # theta_a part
            for i in range(nvar[1]):
                if self.Cpa is not None:
                    val = 0
                    for jj in range(nvar[1]):
                        for kk in range(nvar[1]):
                            val -= a_theta[i, jj] * aips.u(jj, kk) * self.Cpa[kk]

                    sy_arr_dic = _add_to_dict(sy_arr_dic, (self._theta_a(i), 0, 0), val)

                if atp.hd is not None and atp.thetas is not None:
                    thetas_sym_arr = ImmutableSparseNDimArray(atp.thetas.symbols)
                    val = 0
                    for jj in range(nvar[1]):
                        for kk in range(nvar[1]):
                            val -= a_theta[i, jj] * aips.u(jj, kk) * thetas_sym_arr[kk]

                    sy_arr_dic = _add_to_dict(sy_arr_dic, (self._theta_a(i), 0, 0), val * atp.hd.symbol)

                for j in range(nvar[0]):

                    jo = j + offset  # skipping the theta 0 variable if it exists

                    val = 0
                    for jj in range(nvar[1]):
                        val += a_theta[i, jj] * aips.a(jj, jo)

                    sy_arr_dic = _add_to_dict(sy_arr_dic, (self._theta_a(i), self._psi_a(j), 0), val * ap.kd.symbol * ap.sig0.symbol / 2)
                    sy_arr_dic = _add_to_dict(sy_arr_dic, (self._theta_a(i), self._theta_a(jo), 0), - val * (ap.kd.symbol / 2 - 2 * ap.kdp.symbol) * ap.sig0.symbol)

                    val = 0
                    for jj in range(nvar[1]):
                        val -= a_theta[i, jj] * aips.c(jj, jo)

                    sy_arr_dic = _add_to_dict(sy_arr_dic, (self._theta_a(i), self._theta_a(jo), 0), val * par.scale_params.beta.symbol * ap.sig0.symbol)

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
                            sy_arr_dic = _add_to_dict(sy_arr_dic, (self._theta_a(i), self._theta_a(jo), 0), - ap.sig0.symbol * oro / 2)
                            sy_arr_dic = _add_to_dict(sy_arr_dic, (self._theta_a(i), self._psi_a(j), 0), ap.sig0.symbol * oro / 2)

                    for k in range(nvar[0]):
                        ko = k + offset  # skipping the theta 0 variable if it exists
                        val = 0
                        for jj in range(nvar[1]):
                            val += a_theta[i, jj] * aips.b(jj, jo, ko)

                        sy_arr_dic = _add_to_dict(sy_arr_dic, (self._theta_a(i), self._psi_a(j), self._theta_a(ko)), - val * ap.sig0.symbol)
                        sy_arr_dic = _add_to_dict(sy_arr_dic, (self._theta_a(i), self._theta_a(jo), self._psi_a(k)), - val * ap.sig0.symbol)

                        val = 0
                        for jj in range(nvar[1]):
                            val += a_theta[i, jj] * aips.g(jj, jo, ko)

                        sy_arr_dic = _add_to_dict(sy_arr_dic, (self._theta_a(i), self._psi_a(j), self._theta_a(ko)), val)

                for j in range(nvar[1]):
                    val = 0
                    for jj in range(nvar[1]):
                        val += a_theta[i, jj] * aips.u(jj, j)

                    if self.Lpa is not None:
                        sy_arr_dic = _add_to_dict(sy_arr_dic, (self._theta_a(i), self._theta_a(j), 0), val * atp.sc.symbol * self.Lpa)
                    
                    if self.LSBpa is not None:
                        sy_arr_dic = _add_to_dict(sy_arr_dic, (self._theta_a(i), self._theta_a(j), 0), val * self.LSBpa)

                    if atp.hd is not None:
                        sy_arr_dic = _add_to_dict(sy_arr_dic, (self._theta_a(i), self._theta_a(j), 0), val * atp.hd.symbol)

                if ocean:
                    for j in range(nvar[2]):
                        jo = j + offset  # skipping the theta 0 variable if it exists
                        val = 0
                        for jj in range(nvar[1]):
                            val -= a_theta[i, jj] * aips.d(jj, jo)
                        
                        sy_arr_dic = _add_to_dict(sy_arr_dic, (self._theta_a(i), self._psi_o(j), 0), val * ap.sig0.symbol * ap.kd.symbol / 2)

                    if self.Lpa is not None:
                        for j in range(nvar[3]):
                            val = 0
                            for jj in range(nvar[1]):
                                val -= a_theta[i, jj] * aips.s(jj, j)
                            sy_arr_dic = _add_to_dict(sy_arr_dic, (self._theta_a(i), self._deltaT_o(j), 0), val * self.Lpa / 2)
                            if self.LSBpgo is not None:
                                sy_arr_dic = _add_to_dict(sy_arr_dic, (self._theta_a(i), self._deltaT_o(j), 0), val * self.LSBpgo)

                if ground_temp:
                    if self.Lpa is not None:
                        for j in range(nvar[2]):
                            val = 0
                            for jj in range(nvar[1]):
                                val -= a_theta[i, jj] * aips.s(jj, j)
                            sy_arr_dic = _add_to_dict(sy_arr_dic, (self._theta_a(i), self._deltaT_g(j), 0), val * self.Lpa / 2)
                            if self.LSBpgo is not None:
                                sy_arr_dic = _add_to_dict(sy_arr_dic, (self._theta_a(i), self._deltaT_g(j), 0), val * self.LSBpgo)

            if ocean:
                # psi_o part
                for i in range(nvar[2]):
                    for j in range(nvar[0]):
                        jo = j + offset  # skipping the theta 0 variable if it exists

                        for jj in range(nvar[2]):
                            val = M_psio[i, jj] * bips.K(offset + jj, jo) * par.oceanic_params.d.symbol
                        sy_arr_dic = _add_to_dict(sy_arr_dic, (self._psi_o(i), self._psi_a(j), 0), val)
                        sy_arr_dic = _add_to_dict(sy_arr_dic, (self._psi_o(i), self._theta_a(jo), 0), - val)

                    for j in range(nvar[2]):
                        jo = j + offset  # skipping the T 0 variable if it exists

                        val = 0
                        for jj in range(nvar[2]):
                            val -= M_psio[i, jj] * bips.N(offset + jj, jo)
                        sy_arr_dic = _add_to_dict(sy_arr_dic, (self._psi_o(i), self._psi_o(j), 0), val * par.scale_params.beta.symbol)

                        val = 0
                        for jj in range(nvar[2]):
                            val -= M_psio[i, jj] * bips.M(offset + jj, jo)
                        sy_arr_dic = _add_to_dict(sy_arr_dic, (self._psi_o(i), self._psi_o(j), 0), val * (par.oceanic_params.r.symbol + par.oceanic_params.d.symbol))

                        for k in range(nvar[2]):
                            ko = k + offset  # skipping the T 0 variable if it exists
                            val = 0
                            for jj in range(nvar[2]):
                                val -= M_psio[i, jj] * bips.C(offset + jj, jo, ko)
                            sy_arr_dic = _add_to_dict(sy_arr_dic, (self._psi_o(i), self._psi_o(j), self._psi_o(k)), val)

                # deltaT_o part
                for i in range(nvar[3]):
                    val = 0
                    for jj in range(nvar[1]):
                        for kk in range(nvar[3]):
                            val += U_inv[i, kk] * bips.W(kk, jj) * self.Cpgo[jj]
                    
                    sy_arr_dic = _add_to_dict(sy_arr_dic, (self._deltaT_o(i), 0, 0), val)

                    for j in range(nvar[1]):
                        val = 0
                        for jj in range(nvar[3]):
                            val += U_inv[i, jj] * bips.W(jj, j)
                        sy_arr_dic = _add_to_dict(sy_arr_dic, (self._deltaT_o(i), self._theta_a(j), 0), val * 2 * atp.sc.symbol * self.Lpgo)
                        if self.sbpa is not None:
                            sy_arr_dic = _add_to_dict(sy_arr_dic, (self._deltaT_o(i), self._theta_a(j), 0), val * self.sbpa)

                    for j in range(nvar[3]):
                        sy_arr_dic = _add_to_dict(sy_arr_dic, (self._deltaT_o(i), self._deltaT_o(j), 0), - self.Lpgo * _kronecker_delta(i, j))
                        if self.sbpgo is not None:
                            sy_arr_dic = _add_to_dict(sy_arr_dic, (self._deltaT_o(i), self._deltaT_o(j), 0), - self.sbpgo * _kronecker_delta(i, j))

                    for j in range(nvar[2]):
                        for k in range(offset, nvar[3]):
                            jo = j + offset  # skipping the T 0 variable if it exists

                            val = 0
                            for jj in range(nvar[3]):
                                val -= U_inv[i, jj] * bips.O(jj, jo, k)
                            sy_arr_dic = _add_to_dict(sy_arr_dic, (self._deltaT_o(i), self._psi_o(j), self._deltaT_o(k)), val)

            # deltaT_g part
            if ground_temp:
                for i in range(nvar[2]):
                    val = 0
                    for jj in range(nvar[1]):
                        for kk in range(nvar[2]):
                            val += U_inv[i, kk] * bips.W(kk, jj) * self.Cpgo[jj]
                    
                    sy_arr_dic = _add_to_dict(sy_arr_dic, (self._deltaT_g(i), 0, 0), val)

                    for j in range(nvar[1]):
                        val = 0
                        for jj in range(nvar[2]):
                            val += U_inv[i, jj] * bips.W(jj, j)
                        sy_arr_dic = _add_to_dict(sy_arr_dic, (self._deltaT_g(i), self._theta_a(j), 0), val * 2 * atp.sc.symbol * self.Lpgo)
                        
                        if self.sbpa is not None:
                            sy_arr_dic = _add_to_dict(sy_arr_dic, (self._deltaT_g(i), self._theta_a(j), 0), val * self.sbpa)

                    for j in range(nvar[2]):
                        sy_arr_dic = _add_to_dict(sy_arr_dic, (self._deltaT_g(i), self._deltaT_g(j), 0), - self.Lpgo * _kronecker_delta(i, j))
                        if self.sbpgo is not None:
                            sy_arr_dic = _add_to_dict(sy_arr_dic, (self._deltaT_g(i), self._deltaT_g(j), 0), - self.sbpgo * _kronecker_delta(i, j))

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

        self.jacobian_tensor = jacobian_tensor.applyfunc(sy.simplify)
        self.tensor = tensor.applyfunc(sy.simplify)

    @staticmethod
    def remove_dic_zeros(dic):
        non_zero_dic = dict()
        for key in dic.keys():
            if dic[key] != 0:
                non_zero_dic[key] = dic[key]

        return non_zero_dic
    
    @staticmethod
    def jacobian_from_dict(dic):
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

                dic_jac = _add_to_dict(dic_jac, tuple(key[i] for i in new_pos), dic[key])
        
        return dic_jac
    
    @staticmethod
    def simplify_dict(dic):
        keys = dic.keys()
        dic_upp = dict()

        for key in keys:
            new_key = tuple([key[0]] + sorted(key[1:]))
            dic_upp = _add_to_dict(dic_upp, new_key, dic[key])

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

    def sub_tensor(self, tensor=None, continuation_variables=list()):
        """
        Uses sympy substitution to convert the symbolic tensor or a symbolic dictionary to a numerical one.

        Parameters
        ----------
        tensor: dict, sympy array

        continuation_variables: Iterable(Parameter, ScalingParameter, ParametersArray)
            if None all variables are substituted. This variable is the opposite of 'variables'

        Returns
        -------
        ten_out: dict
            Dictionary of the substituted tensor of the model tendencies, with coordinates and value
        """

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
            #Assuming the tensor is a sympy tensor
            ten_out = ten.subs(param_subs)

        
        return ten_out
            
    def print_tensor(self, tensor=None, dict_opp=True, tol=1e-10):
        '''
        Print the non-zero coordinates of values of the tensor of the model tendencies

        Parameters
        ----------
        tensor: dict or Sympy ImmutableSparseNDimArray
            Tensor of model tendencies, either as a dictionary with keys of non-zero coordinates, and values of Sympy Symbols or floats, or as a ImmutableSparseNDimArray.
        '''
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

class SymbolicTensorDynamicT(SymbolicTensorLinear):

    """qgs dynamical temperature first order (linear) symbolic tendencies tensor class.

    Parameters
    ----------
    params: None or QgParams, optional
        The models parameters to configure the tensor. `None` to initialize an empty tensor. Default to `None`.
    atmospheric_inner_products: None or AtmosphericInnerProducts, optional
        The inner products of the atmospheric basis functions on which the model's PDE atmospheric equations are projected.
        If None, disable the atmospheric tendencies. Default to `None`.
    oceanic_inner_products: None or OceanicInnerProducts, optional
        The inner products of the oceanic basis functions on which the model's PDE oceanic equations are projected.
        If None, disable the oceanic tendencies. Default to `None`.
    ground_inner_products: None or GroundInnerProducts, optional
        The inner products of the ground basis functions on which the model's PDE ground equations are projected.
        If None, disable the ground tendencies. Default to `None`.

    Attributes
    ----------
    params: None or QgParams
        The models parameters used to configure the tensor. `None` for an empty tensor.
    atmospheric_inner_products: None or AtmosphericInnerProducts
        The inner products of the atmospheric basis functions on which the model's PDE atmospheric equations are projected.
        If None, disable the atmospheric tendencies. Default to `None`.
    oceanic_inner_products: None or OceanicInnerProducts
        The inner products of the oceanic basis functions on which the model's PDE oceanic equations are projected.
        If None, disable the oceanic tendencies. Default to `None`.
    ground_inner_products: None or GroundInnerProducts
        The inner products of the ground basis functions on which the model's PDE ground equations are projected.
        If None, disable the ground tendencies. Default to `None`.
    tensor: sparse.COO(float)
        The tensor :math:`\\mathcal{T}_{i,j,k}` :math:`i`-th components.
    jacobian_tensor: sparse.COO(float)
        The jacobian tensor :math:`\\mathcal{T}_{i,j,k} + \\mathcal{T}_{i,k,j}` :math:`i`-th components.
    """

    def __init__(self, params=None, atmospheric_inner_products=None, oceanic_inner_products=None, ground_inner_products=None):

        SymbolicTensorLinear.__init__(self, params, atmospheric_inner_products, oceanic_inner_products, ground_inner_products)
        
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
        val = 0
        for i in range(nvar[1]):
            if self.T4LSBpa is not None:
                j = k = ell = 0
                for m in range(nvar[1]):
                    val = 0
                    for jj in range(nvar[1]):
                        val += a_theta[i, jj] * aips._z[jj, j, k, ell, m]
                    if m == 0:
                        sy_arr_dic = _add_to_dict(sy_arr_dic, (self._theta_a(i), self._theta_a(j), self._theta_a(k), self._theta_a(ell), self._theta_a(m)), self.T4LSBpa * val)
                    else:
                        sy_arr_dic = _add_to_dict(sy_arr_dic, (self._theta_a(i), self._theta_a(j), self._theta_a(k), self._theta_a(ell), self._theta_a(m)), 4 * self.T4LSBpa * val)
                        
            if ocean:
                if self.T4LSBpgo is not None:
                    j = k = ell = 0
                    for m in range(nvar[3]):
                        val = 0
                        for jj in range(nvar[1]):
                            val += a_theta[i, jj] * aips._v[jj, j, k, ell, m]
                        if m == 0:
                            sy_arr_dic = _add_to_dict(sy_arr_dic, (self._theta_a(i), self._deltaT_o(j), self._deltaT_o(k), self._deltaT_o(ell), self._deltaT_o(m)), -self.T4LSBpgo * val)
                        else:
                            sy_arr_dic = _add_to_dict(sy_arr_dic, (self._theta_a(i), self._deltaT_o(j), self._deltaT_o(k), self._deltaT_o(ell), self._deltaT_o(m)), -4 * self.T4LSBpgo * val)

            if ground_temp:
                if self.T4LSBpgo is not None:
                    j = k = ell = 0
                    for m in range(nvar[2]):
                        val = 0
                        for jj in range(nvar[1]):
                            val += a_theta[i, jj] * aips._v[jj, j, k, ell, m]
                        if m == 0:
                            sy_arr_dic = _add_to_dict(sy_arr_dic, (self._theta_a(i), self._deltaT_g(j), self._deltaT_g(k), self._deltaT_g(ell), self._deltaT_g(m)), -self.T4LSBpgo * val)
                        else:
                            sy_arr_dic = _add_to_dict(sy_arr_dic, (self._theta_a(i), self._deltaT_g(j), self._deltaT_g(k), self._deltaT_g(ell), self._deltaT_g(m)), -4 * self.T4LSBpgo * val)

        if ocean:
            #delta_T part
            for i in range(nvar[3]):
                j = k = ell = 0
                for m in range(nvar[1]):
                    val = 0
                    for jj in range(nvar[3]):
                        val += U_inv[i, jj] * bips._Z[jj, j, k, ell, m]
                    if m == 0:
                        sy_arr_dic = _add_to_dict(sy_arr_dic, (self._deltaT_o(i), self._theta_a(j), self._theta_a(k), self._theta_a(ell), self._theta_a(m)), self.T4sbpa * val)
                    else:
                        sy_arr_dic = _add_to_dict(sy_arr_dic, (self._deltaT_o(i), self._theta_a(j), self._theta_a(k), self._theta_a(ell), self._theta_a(m)), 4 * self.T4sbpa * val)
                    
                for m in range(nvar[3]):
                    val = 0
                    for jj in range(nvar[3]):
                        val += U_inv[i, jj] * bips._V[jj, j, k, ell, m]
                    if m == 0:
                        sy_arr_dic = _add_to_dict(sy_arr_dic, (self._deltaT_o(i), self._deltaT_o(j), self._deltaT_o(k), self._deltaT_o(ell), self._deltaT_o(m)), - self.T4sbpgo * val)
                    else:
                        sy_arr_dic = _add_to_dict(sy_arr_dic, (self._deltaT_o(i), self._deltaT_o(j), self._deltaT_o(k), self._deltaT_o(ell), self._deltaT_o(m)), -4 * self.T4sbpgo * val)

        if ground_temp:
            # deltaT_g part
            for i in range(nvar[2]):
                j = k = ell = 0
                for m in range(nvar[1]):
                    val = 0
                    for jj in range(nvar[2]):
                        val += U_inv[i, jj] * bips._Z[jj, j, k, ell, m]
                    if m == 0:
                        sy_arr_dic = _add_to_dict(sy_arr_dic, (self._deltaT_g(i), self._theta_a(j), self._theta_a(k), self._theta_a(ell), self._theta_a(m)), self.T4sbpa * val)
                    else:
                        sy_arr_dic = _add_to_dict(sy_arr_dic, (self._deltaT_g(i), self._theta_a(j), self._theta_a(k), self._theta_a(ell), self._theta_a(m)), 4 * self.T4sbpa * val)

                for m in range(nvar[2]):
                    val = 0 
                    for jj in range(nvar[2]):
                        val += U_inv[i, jj] * bips._V[jj, j, k, ell, m]
                    if m == 0:
                        sy_arr_dic = _add_to_dict(sy_arr_dic, (self._deltaT_g(i), self._deltaT_g(j), self._deltaT_g(k), self._deltaT_g(ell), self._deltaT_g(m)), -self.T4sbpgo * val)    
                    else:
                        sy_arr_dic = _add_to_dict(sy_arr_dic, (self._deltaT_g(i), self._deltaT_g(j), self._deltaT_g(k), self._deltaT_g(ell), self._deltaT_g(m)), -4 * self.T4sbpgo * val)
                    
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

            if self.T4LSBpa is not None:
                j = k = ell = 0
                for m in range(nvar[1]):
                    val = 0
                    for jj in range(nvar[1]):
                        val += a_theta[i, jj] * aips.z(jj, j, k, ell, m)
                    if m == 0:
                        sy_arr_dic[(self._theta_a(i), self._theta_a(j), self._theta_a(k), self._theta_a(ell), self._theta_a(m))] = self.T4LSBpa * val
                    else:
                        sy_arr_dic[(self._theta_a(i), self._theta_a(j), self._theta_a(k), self._theta_a(ell), self._theta_a(m))] = 4 * self.T4LSBpa * val

            if ocean:
                if self.T4LSBpgo is not None:
                    j = k = ell = 0
                    for m in range(nvar[3]):
                        val = 0
                        for jj in range(nvar[1]):
                            val -= a_theta[i, jj] * aips.v(jj, j, k, ell, m)
                        if m == 0:
                            sy_arr_dic[(self._theta_a(i), self._deltaT_o(j), self._deltaT_o(k), self._deltaT_o(ell), self._deltaT_o(m))] = self.T4LSBpgo * val
                        else:
                            sy_arr_dic[(self._theta_a(i), self._deltaT_o(j), self._deltaT_o(k), self._deltaT_o(ell), self._deltaT_o(m))] = 4 * self.T4LSBpgo * val

            if ground_temp:
                if self.T4LSBpgo is not None:
                    j = k = ell = 0
                    for m in range(nvar[2]):
                        val = 0
                        for jj in range(nvar[1]):
                            val -= a_theta[i, jj] * aips.v(jj, j, k, ell, m)
                        if m == 0:
                            sy_arr_dic[(self._theta_a(i), self._deltaT_g(j), self._deltaT_g(k), self._deltaT_g(ell), self._deltaT_g(m))] = self.T4LSBpgo * val
                        else:
                            sy_arr_dic[(self._theta_a(i), self._deltaT_g(j), self._deltaT_g(k), self._deltaT_g(ell), self._deltaT_g(m))] = 4 * self.T4LSBpgo * val

        if ocean:

            # deltaT_o part
            for i in range(nvar[3]):

                j = k = ell = 0
                for m in range(nvar[1]):
                    val = 0
                    for jj in range(nvar[3]):
                        val += U_inv[i, jj] * bips.Z(jj, j, k, ell, m)
                    if m == 0:
                        sy_arr_dic[(self._deltaT_o(i), self._theta_a(j), self._theta_a(k), self._theta_a(ell), self._theta_a(m))] = self.T4sbpa * val
                    else:
                        sy_arr_dic[(self._deltaT_o(i), self._theta_a(j), self._theta_a(k), self._theta_a(ell), self._theta_a(m))] = 4 * self.T4sbpa * val

                for m in range(nvar[3]):
                    val = 0
                    for jj in range(nvar[3]):
                        val -= U_inv[i, jj] * bips.V(jj, j, k, ell, m)
                    if m == 0:
                        sy_arr_dic[(self._deltaT_o(i), self._deltaT_o(j), self._deltaT_o(k), self._deltaT_o(ell), self._deltaT_o(m))] = self.T4sbpgo * val
                    else:
                        sy_arr_dic[(self._deltaT_o(i), self._deltaT_o(j), self._deltaT_o(k), self._deltaT_o(ell), self._deltaT_o(m))] = 4 * self.T4sbpgo * val

        # deltaT_g part
        if ground_temp:
            for i in range(nvar[2]):

                j = k = ell = 0
                for m in range(nvar[1]):
                    val = 0
                    for jj in range(nvar[2]):
                        val += U_inv[i, jj] * bips._Z[jj, j, k, ell, m]
                    if m == 0:
                        sy_arr_dic[(self._deltaT_g(i), self._theta_a(j), self._theta_a(k), self._theta_a(ell), self._theta_a(m))] = self.T4sbpa * val
                    else:
                        sy_arr_dic[(self._deltaT_g(i), self._theta_a(j), self._theta_a(k), self._theta_a(ell), self._theta_a(m))] = 4 * self.T4sbpa * val

                for m in range(nvar[2]):
                    val = 0
                    for jj in range(nvar[2]):
                        val -= U_inv[i, jj] * bips._V[jj, j, k, ell, m]
                    if m == 0:
                        sy_arr_dic[(self._deltaT_g(i), self._deltaT_g(j), self._deltaT_g(k), self._deltaT_g(ell), self._deltaT_g(m))] = self.T4sbpgo * val
                    else:
                        sy_arr_dic[(self._deltaT_g(i), self._deltaT_g(j), self._deltaT_g(k), self._deltaT_g(ell), self._deltaT_g(m))] = 4 * self.T4sbpgo * val

        return sy_arr_dic

    def compute_tensor(self):
        """Routine to compute the tensor."""
        # gathering
        if self.params.T4:
            #//TODO: Make a proper error message for here
            raise ValueError("Parameters are set for T4 version, set dynamic_T=True")

        symbolic_dict_linear = SymbolicTensorLinear._compute_tensor_dicts(self)
        symbolic_dict_linear = _shift_dict_keys(symbolic_dict_linear, (0, 0))

        symbolic_dict_dynT = self._compute_tensor_dicts()

        if symbolic_dict_linear is not None:
            symbolic_dict_dynT = {**symbolic_dict_linear, **symbolic_dict_dynT}

        if symbolic_dict_dynT is not None:
            self._set_tensor(symbolic_dict_dynT)

class SymbolicTensorT4(SymbolicTensorLinear):
    # TODO: this takes a long time (>1hr) to run. I think we need a better way to run the non-stored z, v, Z, V IPs. Maybe do not allow `n` as a continuation parameter for this version?
    """qgs dynamical temperature first order (linear) symbolic tendencies tensor class.

    Parameters
    ----------
    params: None or QgParams, optional
        The models parameters to configure the tensor. `None` to initialize an empty tensor. Default to `None`.
    atmospheric_inner_products: None or AtmosphericInnerProducts, optional
        The inner products of the atmospheric basis functions on which the model's PDE atmospheric equations are projected.
        If None, disable the atmospheric tendencies. Default to `None`.
    oceanic_inner_products: None or OceanicInnerProducts, optional
        The inner products of the oceanic basis functions on which the model's PDE oceanic equations are projected.
        If None, disable the oceanic tendencies. Default to `None`.
    ground_inner_products: None or GroundInnerProducts, optional
        The inner products of the ground basis functions on which the model's PDE ground equations are projected.
        If None, disable the ground tendencies. Default to `None`.

    Attributes
    ----------
    params: None or QgParams
        The models parameters used to configure the tensor. `None` for an empty tensor.
    atmospheric_inner_products: None or AtmosphericInnerProducts
        The inner products of the atmospheric basis functions on which the model's PDE atmospheric equations are projected.
        If None, disable the atmospheric tendencies. Default to `None`.
    oceanic_inner_products: None or OceanicInnerProducts
        The inner products of the oceanic basis functions on which the model's PDE oceanic equations are projected.
        If None, disable the oceanic tendencies. Default to `None`.
    ground_inner_products: None or GroundInnerProducts
        The inner products of the ground basis functions on which the model's PDE ground equations are projected.
        If None, disable the ground tendencies. Default to `None`.
    tensor: sparse.COO(float)
        The tensor :math:`\\mathcal{T}_{i,j,k}` :math:`i`-th components.
    jacobian_tensor: sparse.COO(float)
        The jacobian tensor :math:`\\mathcal{T}_{i,j,k} + \\mathcal{T}_{i,k,j}` :math:`i`-th components.
    """

    def __init__(self, params=None, atmospheric_inner_products=None, oceanic_inner_products=None, ground_inner_products=None):

        SymbolicTensorLinear.__init__(self, params, atmospheric_inner_products, oceanic_inner_products, ground_inner_products)
        
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

            if self.T4LSBpa is not None:
                for j in range(nvar[1]):
                    for k in range(nvar[1]):
                        for ell in range(nvar[1]):
                            for m in range(nvar[1]):
                                val = 0
                                for jj in range(nvar[1]):
                                    val += a_theta[i, jj] * aips.z(jj, j, k, ell, m)
                                
                                sy_arr_dic[(self._theta_a(i), self._theta_a(j), self._theta_a(k), self._theta_a(ell), self._theta_a(m))] = self.T4LSBpa * val
                    
            if ocean:
                if self.T4LSBpgo is not None:
                    for j in range(nvar[3]):
                        for k in range(nvar[3]):
                            for ell in range(nvar[3]):
                                for m in range(nvar[3]):
                                    val = 0
                                    for jj in range(nvar[1]):
                                        val -= a_theta[i, jj] * aips.v(jj, j, k, ell, m)
                                    
                                    sy_arr_dic[(self._theta_a(i), self._deltaT_o(j), self._deltaT_o(k), self._deltaT_o(ell), self._deltaT_o(m))] = self.T4LSBpgo * val

            if ground_temp:
                if self.T4LSBpgo is not None:
                    for j in range(nvar[2]):
                        for k in range(nvar[2]):
                            for ell in range(nvar[2]):
                                for m in range(nvar[2]):
                                    val = 0
                                    for jj in range(nvar[1]):
                                            val -= a_theta[i, jj] * aips.v(jj, j, k, ell, m)

                                    sy_arr_dic[(self._theta_a(i), self._deltaT_g(j), self._deltaT_g(k), self._deltaT_g(ell), self._deltaT_g(m))] = self.T4LSBpgo * val
                        
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
                                
                                sy_arr_dic[(self._deltaT_o(i), self._theta_a(j), self._theta_a(k), self._theta_a(ell), self._theta_a(m))] = self.T4sbpa * val
                
                for j in range(nvar[3]):
                    for k in range(nvar[3]):
                        for ell in range(nvar[3]):
                            for m in range(nvar[3]):
                                val = 0
                                for jj in range(nvar[3]):
                                    val -= U_inv[i, jj] * bips.V(jj, j, k, ell, m)

                                sy_arr_dic[(self._deltaT_o(i), self._deltaT_o(j), self._deltaT_o(k), self._deltaT_o(ell), self._deltaT_o(m))] = self.T4sbpgo * val

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
                                
                                sy_arr_dic[(self._deltaT_g(i), self._theta_a(j), self._theta_a(k), self._theta_a(ell), self._theta_a(m))] = self.T4sbpa * val
                
                for j in range(nvar[2]):
                    for k in range(nvar[2]):
                        for ell in range(nvar[2]):    
                            for m in range(nvar[2]):
                                val = 0
                                for jj in range(nvar[2]):
                                    val -= U_inv[i, jj] * bips._V[jj, j, k, ell, m]

                                sy_arr_dic[(self._deltaT_g(i), self._deltaT_g(j), self._deltaT_g(k), self._deltaT_g(ell), self._deltaT_g(m))] = self.T4sbpgo * val
                   
        return sy_arr_dic

    def compute_tensor(self):
        """Routine to compute the tensor."""
        # gathering
        if not(self.params.T4):
            raise ValueError("Parameters are not set for T4 version")

        symbolic_dict_linear = SymbolicTensorLinear._compute_tensor_dicts(self)
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
    dic: dictionary

    shift: Tuple
    """

    shifted_dic = dict()
    for key in dic.keys():
        new_key = key + shift
        shifted_dic[new_key] = dic[key]
    
    return shifted_dic

def _parameter_substitutions(params, continuation_varaibles):
        
    subs = _parameter_values(params)

    if 'scale_params' in params.__dict__.keys():
        subs.update(_parameter_values(params.scale_params))

        # TODO: Is there a better way to do this which is dynamic, the __dict__ method doesnt include the properties?
        # Manually add properties from class
        subs[params.scale_params.L.symbol] = params.scale_params.L
        subs[params.scale_params.beta.symbol] = params.scale_params.beta

    if 'atmospheric_params' in params.__dict__.keys():
        if params.atmospheric_params is not None:
            subs.update(_parameter_values(params.atmospheric_params))

    if 'atemperature_params' in params.__dict__.keys():
        if params.atemperature_params is not None:
            subs.update(_parameter_values(params.atemperature_params))

    if 'oceanic_params' in params.__dict__.keys():
        if params.oceanic_params is not None:
            subs.update(_parameter_values(params.oceanic_params))

    if 'ground_params' in params.__dict__.keys():
        if params.ground_params is not None:
            subs.update(_parameter_values(params.ground_params))

    if 'gotemperature_params' in params.__dict__.keys():
        if params.gotemperature_params is not None:
            subs.update(_parameter_values(params.gotemperature_params))

    # Remove variables in continuation variables
    for cv in continuation_varaibles:
        if isinstance(cv, ParametersArray):
            for cv_i in cv.symbols:
                subs.pop(cv_i)
        else:
            subs.pop(cv.symbol)

    return subs

def _parameter_values(pars):
    """
    Function takes a parameter class and produces a dictionary of the symbol and the corrisponding numerical value
    """

    subs = dict()
    for val in pars.__dict__.values():
        if isinstance(val, Parameter):
            if val.symbol is not None:
                subs[val.symbol] = val

        if isinstance(val, ScalingParameter):
            if val.symbol is not None:
                subs[val.symbol] = val

        if isinstance(val, ParametersArray):
            for v in val:
                if v.symbol is not None or v.symbol != 0:
                    subs[v.symbol] = v
    return subs
    
if __name__ == "__main__":
    dic = dict()
    dic = _add_to_dict(dic, (0, 0), 1)
    dic = _add_to_dict(dic, (0, 0), 2)
    print(dic)