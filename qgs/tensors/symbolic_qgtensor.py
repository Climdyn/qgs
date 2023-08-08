"""
    symbolic qgs tensor module
    =================

    This module computes and holds the symbolic representation of the tensors representing the tendencies of the model's equations.

"""
from contextlib import redirect_stdout

from qgs.functions.tendencies import create_tendencies
from qgs.functions.symbolic_mul import _add_to_dict, _symbolic_tensordot

import numpy as np
import sparse as sp
import sympy as sy
import warnings
import pickle

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

    def __init__(self, params=None, atmospheric_inner_products=None, oceanic_inner_products=None, ground_inner_products=None, numerically_test_tensor=True):

        self.atmospheric_inner_products = atmospheric_inner_products
        self.oceanic_inner_products = oceanic_inner_products
        self.ground_inner_products = ground_inner_products
        self.params = params

        self.sym_params = self.params.symbolic_params

        self.tensor = None
        self.jacobian_tensor = None

        self.test_tensor = numerically_test_tensor

        self.params.symbolic_insolation_array()
        self.params.symbolic_orography_array()

        self.params._set_symbolic_parameters()

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
    def sig0(self):
        return self.sym_params['sigma'] / 2

    @property
    def LR(self):
        return sy.sqrt(self.sym_params['gp'] * self.sym_params['h']) / self.sym_params['fo']

    @property
    def G(self):
        return -self.sym_params['L'] ** 2 / self.LR ** 2
    
    @property
    def Cpgo(self):
        return self.sym_params['go_C'] / (self.sym_params['gnd_gamma'] * self.sym_params['fo']) * self.params.rr / (self.sym_params['fo'] ** 2 * self.sym_params['L'] ** 2)
    
    @property
    def Lpgo(self):
        return self.sym_params['hlambda'] / (self.sym_params['gnd_gamma'] * self.sym_params['fo'])
    
    @property
    def Cpa(self):
        return self.sym_params['atm_C'] / (self.sym_params['atm_gamma'] * self.sym_params['fo']) * self.params.rr / (self.sym_params['fo'] ** 2 * self.sym_params['L'] ** 2) / 2
    
    @property
    def Lpa(self):
        return self.sym_params['hlambda'] / (self.sym_params['atm_gamma'] * self.sym_params['fo'])
    
    #//TODO: Do we want to keep everthing symbolic? Including the Stefan Bolzmann const?
    
    @property
    def sbpgo(self):
        if self.params.gotemperature_params.T0 is None:
            None
        else:
            return 4 * self.params.sb * self.sym_params['go_T0'] ** 3 / (self.sym_params['gnd_gamma'] * self.sym_params['fo'])
    
    @property
    def sbpa(self):
        if self.params.atemperature_params.T0 is None:
            return None
        else:
            return 8 * self.sym_params['eps'] * self.params.sb * self.sym_params['atm_T0'] ** 3 / (self.sym_params['gnd_gamma'] * self.sym_params['fo'])
    
    @property
    def LSBpgo(self):
        if self.params.gotemperature_params.T0 is None:
            None
        else:
            return 2 * self.sym_params['eps'] * self.params.sb * self.sym_params['go_T0'] ** 3 / (self.sym_params['atm_gamma'] * self.sym_params['fo'])
    
    @property
    def LSBpa(self):
        if self.params.atemperature_params.T0 is None:
            return None
        else:
            return 8 * self.sym_params['eps'] * self.params.sb * self.sym_params['atm_T0'] ** 3 / (self.sym_params['atm_gamma'] * self.sym_params['fo'])
    
    @property
    def T4sbpgo(self):
        return self.params.sb * self.sym_params['L'] ** 6 * self.sym_params['fo'] ** 5 / (self.sym_params['gnd_gamma'] * self.params.rr ** 3)
    
    @property
    def T4sbpa(self):
        return 16 * self.sym_params['eps'] * self.params.sb * self.sym_params['L'] ** 6 * self.sym_params['fo'] ** 5 / (self.sym_params['gnd_gamma'] * self.params.rr ** 3)
    
    @property
    def T4LSBpgo(self):
        return 0.5 * self.sym_params['eps'] * self.params.sb * self.sym_params['L'] ** 6 * self.sym_params['fo'] ** 5 / (self.sym_params['atm_gamma'] * self.params.rr ** 3)
    
    @property
    def T4LSBpa(self):
        return 16 * self.sym_params['eps'] * self.params.sb * self.sym_params['L'] ** 6 * self.sym_params['fo'] ** 5 / (self.sym_params['atm_gamma'] * self.params.rr ** 3)
    
    #//TODO: Do i need the scaling parameters?
    
    def _compute_tensor_dicts(self):
        if self.params is None:
            return None

        if self.atmospheric_inner_products is None and self.oceanic_inner_products is None \
                and self.ground_inner_products is None:
            return None

        aips = self.atmospheric_inner_products
        par = self.params
        symbolic_params = self.sym_params
        atp = par.atemperature_params
        gp = par.ground_params
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

            a_inv = sy.matrices.immutable.ImmutableSparseMatrix(nvar[0], nvar[0], a_inv)
            a_inv = a_inv.inverse()
            a_inv = a_inv.simplify()

            a_theta = dict()
            for i in range(nvar[1]):
                for j in range(nvar[1]):
                    a_theta[(i, j)] = self.sig0 * aips.a(i, j) - aips.u(i, j)
            
            a_theta = sy.matrices.immutable.ImmutableSparseMatrix(nvar[1], nvar[1], a_theta)
            a_theta = a_theta.inverse()
            a_theta = a_theta.simplify()

        if bips is not None:
            if ocean:
                U_inv = dict()
                for i in range(nvar[3]):
                    for j in range(nvar[3]):
                        U_inv[(i, j)] = bips.U(i, j)
                U_inv = sy.matrices.immutable.ImmutableSparseMatrix(nvar[3], nvar[3], U_inv)
                U_inv = U_inv.inverse()
                U_inv = U_inv.simplify()

                M_psio = dict()
                for i in range(offset, nvar[3]):
                    for j in range(offset, nvar[3]):
                        M_psio[(i - offset, j - offset)] = bips.M(i, j) + self.G * bips.U(i, j)

                M_psio = sy.matrices.immutable.ImmutableSparseMatrix(nvar[2], nvar[2], M_psio)
                M_psio = M_psio.inverse()
                M_psio = M_psio.simplify()

            else:
                U_inv = dict()
                for i in range(nvar[2]):
                    for j in range(nvar[2]):
                        U_inv[(i, j)] = bips.U(i, j)
                U_inv = sy.matrices.immutable.ImmutableSparseMatrix(nvar[2], nvar[2], U_inv)
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
            if symbolic_params['hk'] is not None:
                a_inv_mult_g = _symbolic_tensordot(a_inv, aips._g[offset:, offset:, offset:], axes=1)
                oro = _symbolic_tensordot(a_inv_mult_g, symbolic_params['hk'], axes=1)
            else:
                a_inv_mult_gh = _symbolic_tensordot(a_inv, aips._gh[offset:, offset:, offset:], axes=1)
                oro = _symbolic_tensordot(a_inv_mult_gh, symbolic_params['hk'], axes=1)

            a_inv_mult_b = _symbolic_tensordot(a_inv, aips._b[offset:, offset:, offset:], axes=1)
            a_inv_mult_d = a_inv @ aips._d[offset:, offset:]


            for i in range(nvar[0]):
                for j in range(nvar[0]):

                    jo = j + offset  # skipping the theta 0 variable if it exists
                    sy_arr_dic = _add_to_dict(sy_arr_dic, (self._psi_a(i), self._psi_a(j), 0), -a_inv_mult_c[i, j] * symbolic_params['beta'])
                    sy_arr_dic = _add_to_dict(sy_arr_dic, (self._psi_a(i), self._psi_a(j), 0), -(symbolic_params['kd'] * _kronecker_delta(i, j)) / 2)
                    sy_arr_dic = _add_to_dict(sy_arr_dic, (self._psi_a(i), self._theta_a(jo), 0), (symbolic_params['kd'] * _kronecker_delta(i, j)) / 2)

                    if gp is not None:
                        # convert 
                        if symbolic_params['hk'] is not None:

                            sy_arr_dic = _add_to_dict(sy_arr_dic, (self._psi_a(i), self._psi_a(j), 0), oro[i, j][0] / 2)
                            sy_arr_dic = _add_to_dict(sy_arr_dic, (self._psi_a(i), self._theta_a(jo), 0), oro[i, j][0] / 2)

                    for k in range(nvar[0]):
                        ko = k + offset  # skipping the theta 0 variable if it exists
                        sy_arr_dic = _add_to_dict(sy_arr_dic, (self._psi_a(i), self._psi_a(j), self._psi_a(k)), -a_inv_mult_b[i, j, k])
                        sy_arr_dic = _add_to_dict(sy_arr_dic, (self._psi_a(i), self._theta_a(jo), self._theta_a(ko)), -a_inv_mult_b[i, j, k])

                if ocean:
                    for j in range(nvar[2]):
                        sy_arr_dic = _add_to_dict(sy_arr_dic, (self._psi_a(i), self._psi_o(j), 0), a_inv_mult_d[i, j] * symbolic_params['kd'] / 2)
            # theta_a part
            a_theta_mult_u = _symbolic_tensordot(a_theta, aips._u, axes=1)
            if self.Cpa is not None:
                val_Cpa = _symbolic_tensordot(a_theta_mult_u , self.Cpa, axes=1)

            if symbolic_params['hd'] is not None and symbolic_params['thetas'] is not None:
                val_thetas = _symbolic_tensordot(a_theta_mult_u, symbolic_params['thetas'], axes=1)  # not perfect

            a_theta_mult_a = _symbolic_tensordot(a_theta, aips._a[:, offset:], axes=1)
            a_theta_mult_c = _symbolic_tensordot(a_theta, aips._c[:, offset:], axes=1)
            
            a_theta_mult_g = _symbolic_tensordot(a_theta, aips._g[:, offset:, offset:], axes=1)

            if gp is not None:
                if gp.orographic_basis == "atmospheric":
                    oro = _symbolic_tensordot(a_theta_mult_g, symbolic_params['hk'], axes=1)
                else:
                    a_theta_mult_gh = _symbolic_tensordot(a_theta, aips._gh[:, offset:, offset:], axes=1)
                    oro = _symbolic_tensordot(a_theta_mult_gh, symbolic_params['hk'], axes=1)

            a_theta_mult_b = _symbolic_tensordot(a_theta, aips._b[:, offset:, offset:], axes=1)

            if ocean or ground_temp:
                a_theta_mult_d = _symbolic_tensordot(a_theta, aips._d[:, offset:], axes=1)
                a_theta_mult_s = _symbolic_tensordot(a_theta, aips._s, axes=1)

            for i in range(nvar[1]):
                if self.Cpa is not None:
                    sy_arr_dic = _add_to_dict(sy_arr_dic, (self._theta_a(i), 0, 0), -val_Cpa[i][0])

                if symbolic_params['hd'] is not None and atp.thetas is not None:
                    sy_arr_dic = _add_to_dict(sy_arr_dic, (self._theta_a(i), 0, 0), -val_thetas[i][0] * symbolic_params['hd'])

                for j in range(nvar[0]):

                    jo = j + offset  # skipping the theta 0 variable if it exists

                    val_2 = a_theta_mult_a[i, j] * symbolic_params['kd'] * self.sig0 / 2
                    val_3 = a_theta_mult_a[i, j] * (symbolic_params['kd'] / 2 + 2 * symbolic_params['kdp']) * self.sig0
                    sy_arr_dic = _add_to_dict(sy_arr_dic, (self._theta_a(i), self._psi_a(j), 0), val_2)
                    sy_arr_dic = _add_to_dict(sy_arr_dic, (self._theta_a(i), self._theta_a(jo), 0), -val_3)

                    sy_arr_dic = _add_to_dict(sy_arr_dic, (self._theta_a(i), self._theta_a(jo), 0), -a_theta_mult_c[i, j] * symbolic_params['beta'] * self.sig0)

                    if gp is not None:
                        if symbolic_params['hk'] is not None:

                            sy_arr_dic = _add_to_dict(sy_arr_dic, (self._theta_a(i), self._theta_a(jo), 0), -self.sig0 * oro[i, j][0] / 2)
                            sy_arr_dic = _add_to_dict(sy_arr_dic, (self._theta_a(i), self._psi_a(j), 0), self.sig0 * oro[i, j][0] / 2)

                    for k in range(nvar[0]):
                        ko = k + offset  # skipping the theta 0 variable if it exists
                        sy_arr_dic = _add_to_dict(sy_arr_dic, (self._theta_a(i), self._psi_a(j), self._theta_a(ko)), - a_theta_mult_b[i, j, k] * self.sig0)
                        sy_arr_dic = _add_to_dict(sy_arr_dic, (self._theta_a(i), self._theta_a(jo), self._psi_a(k)), - a_theta_mult_b[i, j, k] * self.sig0)

                        sy_arr_dic = _add_to_dict(sy_arr_dic, (self._theta_a(i), self._psi_a(j), self._theta_a(ko)),  a_theta_mult_g[i, j, k])

                for j in range(nvar[1]):
                    if self.Lpa is not None:
                        sy_arr_dic = _add_to_dict(sy_arr_dic, (self._theta_a(i), self._theta_a(j), 0), a_theta_mult_u[i, j] * symbolic_params['sc'] * self.Lpa)
                    if self.LSBpa is not None:
                        sy_arr_dic = _add_to_dict(sy_arr_dic, (self._theta_a(i), self._theta_a(j), 0), a_theta_mult_u[i, j] * self.LSBpa)

                    if atp.hd is not None:
                        sy_arr_dic = _add_to_dict(sy_arr_dic, (self._theta_a(i), self._theta_a(j), 0), a_theta_mult_u[i, j] * atp.hd)

                if ocean:
                    for j in range(nvar[2]):
                        sy_arr_dic = _add_to_dict(sy_arr_dic, (self._theta_a(i), self._psi_o(j), 0), -a_theta_mult_d[i, j] * self.sig0 * symbolic_params['kd'] / 2)

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

                        sy_arr_dic = _add_to_dict(sy_arr_dic, (self._psi_o(i), self._psi_a(j), 0), M_psio_mult_K[i, j] * symbolic_params['d'])
                        sy_arr_dic = _add_to_dict(sy_arr_dic, (self._psi_o(i), self._theta_a(jo), 0), -M_psio_mult_K[i, j] * symbolic_params['d'])

                    for j in range(nvar[2]):
                        sy_arr_dic = _add_to_dict(sy_arr_dic, (self._psi_o(i), self._psi_o(j), 0), -M_psio_mult_N[i, j] * symbolic_params['beta'])

                        sy_arr_dic = _add_to_dict(sy_arr_dic, (self._psi_o(i), self._psi_o(j), 0), -M_psio_mult_M[i, j] * (symbolic_params['r'] + symbolic_params['d']))

                        for k in range(nvar[2]):
                            sy_arr_dic = _add_to_dict(sy_arr_dic, (self._psi_o(i), self._psi_o(j), self._psi_o(k)), - M_psio_mult_C[i, j, k])

                # deltaT_o part
                U_inv_mult_W = _symbolic_tensordot(U_inv, bips._W, axes=1)
                U_inv_mult_W_Cpgo = _symbolic_tensordot(U_inv_mult_W, self.Cpgo, axes=1)

                U_inv_mult_O = _symbolic_tensordot(U_inv, bips._O[:, offset:, offset:], axes=1)
                
                for i in range(nvar[3]):
                    sy_arr_dic = _add_to_dict(sy_arr_dic, (self._deltaT_o(i), 0, 0), U_inv_mult_W_Cpgo[i][0])

                    for j in range(nvar[1]):
                        sy_arr_dic = _add_to_dict(sy_arr_dic, (self._deltaT_o(i), self._theta_a(j), 0), U_inv_mult_W[i, j] * 2 * symbolic_params['sc'] * self.Lpgo)
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
                    sy_arr_dic = _add_to_dict(sy_arr_dic, (self._deltaT_g(i), 0, 0), U_inv_mult_W_Cpgo[i])  # not perfect

                    for j in range(nvar[1]):
                        sy_arr_dic = _add_to_dict(sy_arr_dic, (self._deltaT_g(i), self._theta_a(j), 0), U_inv_mult_W[i, j] * 2 * symbolic_params['sc'] * self.Lpgo)
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
                    sy_arr_dic = _add_to_dict(sy_arr_dic, (self._psi_a(i), self._psi_a(j), 0), - val * symbolic_params['beta'])
                    sy_arr_dic = _add_to_dict(sy_arr_dic, (self._psi_a(i), self._psi_a(j), 0), - (symbolic_params['kd'] * _kronecker_delta(i, j)) / 2)
                    sy_arr_dic = _add_to_dict(sy_arr_dic, (self._psi_a(i), self._theta_a(jo), 0), (symbolic_params['kd'] * _kronecker_delta(i, j)) / 2)

                    if gp is not None:
                        if symbolic_params['hk'] is not None:
                            oro = 0
                            if gp.orographic_basis == "atmospheric":
                                for jj in range(nvar[0]):
                                    for kk in range(nvar[0]):
                                        oro += a_inv[i, jj] * aips.g(offset + jj, j, offset + kk) * symbolic_params['kd']
                            else:
                                for jj in range(nvar[0]):
                                    for kk in range(nvar[0]):
                                        oro += a_inv[i, jj] * aips.gh(offset + jj, j, offset + kk) * symbolic_params['hk_val']
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
                        sy_arr_dic = _add_to_dict(sy_arr_dic, (self._psi_a(i), self._psi_o(j), 0), val * symbolic_params['kd'] / 2)


            # theta_a part
            for i in range(nvar[1]):
                if self.Cpa is not None:
                    for jj in range(nvar[1]):
                        for kk in range(nvar[1]):
                            sy_arr_dic = _add_to_dict(sy_arr_dic, (self._theta_a(i), 0, 0), - a_theta[i, jj] * aips.u(jj, kk) * self.Cpa[kk])

                if symbolic_params['hd'] is not None and self.thetas is not None:
                    val = 0
                    for jj in range(nvar[1]):
                        for kk in range(nvar[1]):
                            val -= a_theta[i, jj] * aips.u(jj, kk) * self.thetas[kk]
                    sy_arr_dic = _add_to_dict(sy_arr_dic, (self._theta_a(i), 0, 0), val * symbolic_params['hd'])

                for j in range(nvar[0]):

                    jo = j + offset  # skipping the theta 0 variable if it exists

                    val = 0
                    for jj in range(nvar[1]):
                        val += a_theta[i, jj] * aips.a(jj, jo)
                    sy_arr_dic = _add_to_dict(sy_arr_dic, (self._theta_a(i), self._psi_a(j), 0), val * symbolic_params['kd'] * self.sig0 / 2)
                    sy_arr_dic = _add_to_dict(sy_arr_dic, (self._theta_a(i), self._theta_a(jo), 0), - val * (symbolic_params['kd'] / 2 - 2 * symbolic_params['kdp']) * self.sig0)

                    val = 0
                    for jj in range(nvar[1]):
                        val -= a_theta[i, jj] * aips.c(jj, jo)
                    sy_arr_dic = _add_to_dict(sy_arr_dic, (self._theta_a(i), self._theta_a(jo), 0), val * symbolic_params['beta'] * self.sig0)

                    if gp is not None:
                        if symbolic_params['hk'] is not None:
                            oro = 0
                            if gp.orographic_basis == "atmospheric":
                                for jj in range(nvar[1]):
                                    for kk in range(nvar[0]):
                                        oro += a_theta[i, jj] * aips.g(jj, jo, offset + kk) * symbolic_params['hk'][kk]
                            else:
                                for jj in range(nvar[1]):
                                    for kk in range(nvar[0]):
                                        oro += a_theta[i, jj] * aips.gh(jj, jo, offset + kk) * symbolic_params['hk'][kk]
                            sy_arr_dic = _add_to_dict(sy_arr_dic, (self._theta_a(i), self._theta_a(jo), 0), - self.sig0 * oro / 2)
                            sy_arr_dic = _add_to_dict(sy_arr_dic, (self._theta_a(i), self._psi_a(j), 0), self.sig0 * oro / 2)

                    for k in range(nvar[0]):
                        ko = k + offset  # skipping the theta 0 variable if it exists
                        val = 0
                        for jj in range(nvar[1]):
                            val += a_theta[i, jj] * aips.b(jj, jo, ko)
                        sy_arr_dic = _add_to_dict(sy_arr_dic, (self._theta_a(i), self._psi_a(j), self._theta_a(ko)), - val * self.sig0)
                        sy_arr_dic = _add_to_dict(sy_arr_dic, (self._theta_a(i), self._theta_a(jo), self._psi_a(k)), - val * self.sig0)

                        val = 0
                        for jj in range(nvar[1]):
                            val += a_theta[i, jj] * aips.g(jj, jo, ko)

                        sy_arr_dic = _add_to_dict(sy_arr_dic, (self._theta_a(i), self._psi_a(j), self._theta_a(ko)), val)

                for j in range(nvar[1]):
                    val = 0
                    for jj in range(nvar[1]):
                        val += a_theta[i, jj] * aips.u(jj, j)

                    if self.Lpa is not None:
                        sy_arr_dic = _add_to_dict(sy_arr_dic, (self._theta_a(i), self._theta_a(j), 0), val * symbolic_params['sc'] * self.Lpa)
                    if self.LSBpa is not None:
                        sy_arr_dic = _add_to_dict(sy_arr_dic, (self._theta_a(i), self._theta_a(j), 0), val * self.LSBpa)

                    if symbolic_params['hd'] is not None:
                        sy_arr_dic = _add_to_dict(sy_arr_dic, (self._theta_a(i), self._theta_a(j), 0), val * symbolic_params['hd'])

                if ocean:
                    for j in range(nvar[2]):
                        jo = j + offset  # skipping the theta 0 variable if it exists
                        val = 0
                        for jj in range(nvar[1]):
                            val -= a_theta[i, jj] * aips.d(jj, jo)
                            sy_arr_dic = _add_to_dict(sy_arr_dic, (self._theta_a(i), self._psi_o(j), 0), val * self.sig0 * symbolic_params['kd'] / 2)

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
                            val = M_psio[i, jj] * bips.K(offset + jj, jo) * symbolic_params['d']
                            sy_arr_dic = _add_to_dict(sy_arr_dic, (self._psi_o(i), self._psi_a(j), 0), val)
                            sy_arr_dic = _add_to_dict(sy_arr_dic, (self._psi_o(i), self._theta_a(jo), 0), - val)

                    for j in range(nvar[2]):
                        jo = j + offset  # skipping the T 0 variable if it exists

                        val = 0
                        for jj in range(nvar[2]):
                            val -= M_psio[i, jj] * bips.N(offset + jj, jo)
                        sy_arr_dic = _add_to_dict(sy_arr_dic, (self._psi_o(i), self._psi_o(j), 0), val * symbolic_params['beta'])

                        val = 0
                        for jj in range(nvar[2]):
                            val -= M_psio[i, jj] * bips.M(offset + jj, jo)
                        sy_arr_dic = _add_to_dict(sy_arr_dic, (self._psi_o(i), self._psi_o(j), 0), val * (symbolic_params['r'] + symbolic_params['d']))

                        for k in range(nvar[2]):
                            ko = k + offset  # skipping the T 0 variable if it exists
                            for jj in range(nvar[2]):
                                sy_arr_dic = _add_to_dict(sy_arr_dic, (self._psi_o(i), self._psi_o(j), self._psi_o(k)), - M_psio[i, jj] * bips.C(offset + jj, jo, ko))

                # deltaT_o part
                for i in range(nvar[3]):
                    for jj in range(nvar[1]):
                        for kk in range(nvar[3]):
                            sy_arr_dic = _add_to_dict(sy_arr_dic, (self._deltaT_o(i), 0, 0), U_inv[i, kk] * bips.W(kk, jj) * self.Cpgo[jj])

                    for j in range(nvar[1]):
                        val = 0
                        for jj in range(nvar[3]):
                            val += U_inv[i, jj] * bips.W(jj, j)
                        sy_arr_dic = _add_to_dict(sy_arr_dic, (self._deltaT_o(i), self._theta_a(j), 0), val * 2 * symbolic_params['sc'] * self.Lpgo)
                        if self.sbpa is not None:
                            sy_arr_dic = _add_to_dict(sy_arr_dic, (self._deltaT_o(i), self._theta_a(j), 0), val * self.sbpa)

                    for j in range(nvar[3]):
                        sy_arr_dic = _add_to_dict(sy_arr_dic, (self._deltaT_o(i), self._deltaT_o(j), 0), - self.Lpgo * _kronecker_delta(i, j))
                        if self.sbpgo is not None:
                            sy_arr_dic = _add_to_dict(sy_arr_dic, (self._deltaT_o(i), self._deltaT_o(j), 0), - self.sbpgo * _kronecker_delta(i, j))

                    for j in range(nvar[2]):
                        for k in range(offset, nvar[3]):
                            jo = j + offset  # skipping the T 0 variable if it exists
                            for jj in range(nvar[3]):
                                sy_arr_dic = _add_to_dict(sy_arr_dic, (self._deltaT_o(i), self._psi_o(j), self._deltaT_o(k)), - U_inv[i, jj] * bips.O(jj, jo, k))

            # deltaT_g part
            if ground_temp:
                for i in range(nvar[2]):
                    for jj in range(nvar[1]):
                        for kk in range(nvar[2]):
                            sy_arr_dic = _add_to_dict(sy_arr_dic, (self._deltaT_g(i), 0, 0), U_inv[i, kk] * bips.W(kk, jj) * self.Cpgo[jj])

                    for j in range(nvar[1]):
                        val = 0
                        for jj in range(nvar[2]):
                            val += U_inv[i, jj] * bips.W(jj, j)
                            sy_arr_dic = _add_to_dict(sy_arr_dic, (self._deltaT_g(i), self._theta_a(j), 0), val * 2 * symbolic_params['sc'] * self.Lpgo)
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

        if self.test_tensor:
            print("Testing Tensor Numerically")
            self.test_tensor_numerically(self.tensor_dic)
            
    def _set_symbolic_tensor(self):
        ndim = self.params.ndim

        if self.params.dynamic_T:
            if self.params.T4:
                #//TODO: Make a proper error message for here
                raise ValueError("Symbolic tensor output not configured for T4 version, use Dynamic T version")
            else:
                dims = (ndim + 1, ndim + 1, ndim + 1, ndim + 1, ndim + 1)
        else:
            dims = (ndim + 1, ndim + 1, ndim + 1)

        #//TODO: I needed to make copies here, but I am not sure why. If I tried to input the dict without the copy it messed up the keys
        jacobian_tensor = sy.tensor.array.ImmutableSparseNDimArray(self.jac_dic.copy(), dims)
        tensor = sy.tensor.array.ImmutableSparseNDimArray(self.tensor_dic.copy(), dims)

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
        # this is not correct, I should not permute the tuple, but sort the list of them
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

    def subs_tensor(self, tensor=None, dict_opp=True):
        """
        Uses sympy substitution to convert the symbolic tensor or a symbolic dictionary to a numerical one.
        """

        symbol_to_number_map = list()

        for key in self.sym_params.keys():
            try:
                symbol_to_number_map.append(self.params.symbol_to_value[key])
            except:
                pass
        
        if isinstance(tensor, dict):
            ten_out = dict()
            for key in tensor.keys():
                val = tensor[key].subs(symbol_to_number_map)
                try:
                    ten_out[key] = float(val)
                except:
                    ten_out[key] = val

        elif dict_opp and self.tensor_dic is not None:
            ten_out = dict()
            for key in self.tensor_dic.keys():
                val = self.tensor_dic[key].subs(symbol_to_number_map)
                try:
                    ten_out[key] = float(val)
                except:
                    ten_out[key] = val

        else:
            if tensor is not None:
                ten_out = tensor.subs(symbol_to_number_map)
            else:
                ten_out = self.tensor.subs(symbol_to_number_map)
        
        return ten_out
        
    def test_tensor_numerically(self, tensor=None, dict_opp=True, tol=1e-14):
        """
        Uses sympy substitution to convert the symbolic tensor, or symbolic dictionary, to a numerical one.
        This is then compared to the tensor calculated by the qgs.tensor.symbolic module.
        
        """
        ndim = self.params.ndim

        if self.params.dynamic_T:
            if self.params.T4:
                #//TODO: Make a proper error message for here
                raise ValueError("Symbolic tensor output not configured for T4 version, use Dynamic T version")
            else:
                dims = (ndim + 1, ndim + 1, ndim + 1, ndim + 1, ndim + 1)
        else:
            dims = (ndim + 1, ndim + 1, ndim + 1)

        _, _, numerical_tensor = create_tendencies(self.params, return_qgtensor=True)

        if tensor is None:
            if dict_opp:
                tensor = self.tensor_dic
            else:
                tensor = self.tensor

        subbed_ten = self.subs_tensor(tensor)
        if isinstance(subbed_ten, dict):
            coords = np.array([list(k) for k in subbed_ten.keys()]).T
            data = np.array(list(subbed_ten.values()), dtype=float)
            subbed_tensor_sp = sp.COO(coords, data, shape=dims)
        else:
            subbed_ten = np.array(subbed_ten)
            subbed_tensor_np = np.array(subbed_ten).astype(np.float64)
            subbed_tensor_sp = sp.COO.from_numpy(subbed_tensor_np)

        diff_arr = subbed_tensor_sp.todense() - numerical_tensor.tensor.todense()

        #TODO: Is there a better way to do error/pass messages?
        total_error = np.sum(np.abs(diff_arr))
        max_error = np.max(np.abs(diff_arr))

        if max_error > tol:
            self.print_tensor(diff_arr, tol=tol)
            
            raise ValueError("Symbolic tensor and numerical tensor do not match at the above coordinates, with a total error of: " + str(total_error))
        
        elif total_error > tol:
            warnings.warn("Symbolic tensor and numerical tensor have a combined error of: " + str(total_error))
        else:
            print("Tensor passes numerical test with a combined error of less than: " + str(tol))
            
    def print_tensor(self, tensor=None, dict_opp=True, tol=1e-10):
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

    def __init__(self, params=None, atmospheric_inner_products=None, oceanic_inner_products=None, ground_inner_products=None, numerically_test_tensor=True):

        SymbolicTensorLinear.__init__(self, params, atmospheric_inner_products, oceanic_inner_products, ground_inner_products, numerically_test_tensor)
        
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
                    a_theta[(i, j)] = self.sig0 * aips.a(i, j) - aips.u(i, j)
            a_theta = sy.matrices.immutable.ImmutableSparseMatrix(nvar[1], nvar[1], a_theta)
            a_theta = a_theta.inverse()

        if bips is not None:
            U_inv = dict()
            if ocean:
                for i in range(nvar[3]):
                    for j in range(nvar[3]):
                        U_inv[(i, j)] = bips.U(i, j)
                U_inv = sy.matrices.immutable.ImmutableSparseMatrix(nvar[3], nvar[3], U_inv)
                U_inv = U_inv.inverse()
            else:
                for i in range(nvar[2]):
                    for j in range(nvar[2]):
                        U_inv[i, j] = bips.U(i, j)
                U_inv = sy.matrices.immutable.ImmutableSparseMatrix(nvar[2], nvar[2], U_inv)
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
                    a_theta[(i, j)] = self.sig0 * aips.a(i, j) - aips.u(i, j)

            a_theta = sy.matrices.immutable.ImmutableSparseMatrix(nvar[1], nvar[1], a_theta)
            a_theta = a_theta.inverse()

        if bips is not None:
            if ocean:
                U_inv = dict()
                for i in range(nvar[3]):
                    for j in range(nvar[3]):
                        U_inv[i, j] = bips.U(i, j)
                U_inv = sy.matrices.immutable.ImmutableSparseMatrix(nvar[3], nvar[3], U_inv)

            else:
                U_inv = dict()
                for i in range(nvar[2]):
                    for j in range(nvar[2]):
                        U_inv[(i, j)] = bips.U(i, j)
                U_inv = sy.matrices.immutable.ImmutableSparseMatrix(nvar[2], nvar[2], U_inv)
            
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
    # TODO: this takes a long time (>1hr) to run. I think we need a better way to run the non-stored z, v, Z, V IPs.
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

    def __init__(self, params=None, atmospheric_inner_products=None, oceanic_inner_products=None, ground_inner_products=None, numerically_test_tensor=True):

        SymbolicTensorLinear.__init__(self, params, atmospheric_inner_products, oceanic_inner_products, ground_inner_products, numerically_test_tensor)
        
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
                    a_theta[(i, j)] = self.sig0 * aips.a(i, j) - aips.u(i, j)

            a_theta = sy.matrices.immutable.ImmutableSparseMatrix(nvar[1], nvar[1], a_theta)
            a_theta = a_theta.inverse()

        if bips is not None:
            if ocean:
                U_inv = dict()
                for i in range(nvar[3]):
                    for j in range(nvar[3]):
                        U_inv[i, j] = bips.U(i, j)
                U_inv = sy.matrices.immutable.ImmutableSparseMatrix(nvar[3], nvar[3], U_inv)

            else:
                U_inv = dict()
                for i in range(nvar[2]):
                    for j in range(nvar[2]):
                        U_inv[(i, j)] = bips.U(i, j)
                U_inv = sy.matrices.immutable.ImmutableSparseMatrix(nvar[2], nvar[2], U_inv)
            
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
            #//TODO: Make a proper error message for here
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
    
if __name__ == "__main__":
    dic = dict()
    dic = _add_to_dict(dic, (0, 0), 1)
    dic = _add_to_dict(dic, (0, 0), 2)
    print(dic)