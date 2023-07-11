"""
    symbolic qgs tensor module
    =================

    This module computes and holds the symbolic representation of the tensors representing the tendencies of the model's equations.

"""
from contextlib import redirect_stdout

import numpy as np
import sparse as sp
import sympy as sy
import pickle

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

        self.sym_params = self.params.symbolic_params

        self.tensor = None
        self.jacobian_tensor = None

        self.params.symbolic_insolation_array()

        # self.compute_tensor()

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
        return self.sym_params['L'] ** 2 / self.LR ** 2
    
    @property
    def Cpgo(self):
        return self.sym_params['gnd_C'] / (self.sym_params['gnd_gamma'] * self.sym_params['fo']) * self.params.rr / (self.sym_params['fo'] ** 2 * self.sym_params['L'] ** 2)
    
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
        return 4 * self.params.sb * self.sym_params['gnd_T0'] ** 3 / (self.sym_params['gnd_gamma'] * self.sym_params['fo'])
    
    @property
    def sbpa(self):
        return 8 * self.sym_params['eps'] * self.param.sb * self.sym_params['atm_T0'] ** 3 / (self.sym_params['gnd_gamma'] * self.sym_params['fo'])
    
    @property
    def LSBpgo(self):
        return 2 * self.sym_params['eps'] * self.params.sb * self.sym_params['gnd_T0'] ** 3 / (self.sym_params['atm_gamma'] * self.sym_params['fo'])
    
    @property
    def LSBpa(self):
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
        ap = par.atmospheric_params
        op = par.oceanic_params
        scp = par.scale_params
        gp = par.ground_params
        nvar = par.number_of_variables
        ndim = par.ndim

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
            a_inv = sy.matrices.Inverse(a_inv)

            a_theta = dict()
            for i in range(nvar[1]):
                for j in range(nvar[1]):
                    a_theta[(i, j)] = self.sig0 * aips.a(i, j) - aips.u(i, j)
            
            a_theta = sy.matrices.immutable.ImmutableSparseMatrix(nvar[1], nvar[1], a_theta)
            a_theta = sy.matrices.Inverse(a_theta)

        if bips is not None:
            if ocean:
                U_inv = dict()
                for i in range(nvar[3]):
                    for j in range(nvar[3]):
                        U_inv[(i, j)] = bips.U(i, j)
                U_inv = sy.matrices.immutable.ImmutableSparseMatrix(nvar[3], nvar[3], U_inv)
                U_inv = sy.matrices.Inverse(U_inv)

                M_psio = dict()
                for i in range(offset, nvar[3]):
                    for j in range(offset, nvar[3]):
                        M_psio[(i - offset, j - offset)] = bips.M(i, j) + self.G * bips.U(i, j)

                M_psio = sy.matrices.immutable.ImmutableSparseMatrix(nvar[2], nvar[2], M_psio)
                M_psio = sy.matrices.Inverse(M_psio)

            else:
                U_inv = dict()
                for i in range(nvar[2]):
                    for j in range(nvar[2]):
                        U_inv[(i, j)] = bips.U(i, j)
                U_inv = sy.matrices.immutable.ImmutableSparseMatrix(nvar[2], nvar[2], U_inv)
                U_inv = sy.matrices.immutable.ImmutableSparseMatrix(U_inv)

        ################

        if bips is not None:
            go = bips.stored
        else:
            go = True

        symbolic_array_dic = dict()

        if aips.stored and go:
            # psi_a part
            for i in range(nvar[0]):
                for j in range(nvar[0]):

                    jo = j + offset  # skipping the theta 0 variable if it exists
                    #//TODO: A =- was converted to = here, I need to make sure this doesnt alter the results
                    val = a_inv[i, :] * aips._c[offset:, jo]
                    print(val)
                    symbolic_array_dic[(self._psi_a(i), self._psi_a(j), 0)] = val * symbolic_params['beta']

                    symbolic_array_dic[(self._psi_a(i), self._psi_a(j), 0)] -= (symbolic_params['kd'] * _kronecker_delta(i, j)) / 2
                    symbolic_array_dic[(self._psi_a(i), self._theta_a(jo), 0)] = (symbolic_params['kd'] * _kronecker_delta(i, j)) / 2

                    if gp is not None:
                        # convert 
                        if gp.hk is not None:
                            #//TODO: Need to make this symbolic
                            if gp.orographic_basis == "atmospheric":
                                oro = a_inv[i, :] * aips._g[offset:, jo, offset:] * symbolic_params['hk']  # not perfect
                            else:
                                #//TODO: Need to make this symbolic
                                # TODO: Can only be used with symbolic inner products here - a warning or an error should be raised if this is not the case.
                                oro = a_inv[i, :] * aips._gh[offset:, jo, offset:] * symbolic_params['hk'] # not perfect
                            symbolic_array_dic[(self._psi_a(i), self._psi_a(j), 0)] -= oro / 2
                            symbolic_array_dic[(self._psi_a(i), self._theta_a(jo), 0)] += oro / 2

                    for k in range(nvar[0]):
                        ko = k + offset  # skipping the theta 0 variable if it exists
                        val = a_inv[i, :] * aips._b[offset:, jo, ko]
                        symbolic_array_dic[(self._psi_a(i), self._psi_a(j), self._psi_a(k))] = - val
                        symbolic_array_dic[(self._psi_a(i), self._theta_a(jo), self._theta_a(ko))] = - val
                if ocean:
                    for j in range(nvar[2]):
                        jo = j + offset  # skipping the theta 0 variable if it exists
                        val = a_inv[i, :] * aips._d[offset:, jo]
                        symbolic_array_dic[(self._psi_a(i), self._psi_o(j), 0)] += val * symbolic_params['kd'] / 2

            # theta_a part
            for i in range(nvar[1]):
                if self.Cpa is not None:
                    symbolic_array_dic[(self._theta_a(i), 0, 0)] -= a_theta[i, :] * aips._u * self.Cpa  # not perfect

                if atp.hd is not None and atp.thetas is not None:
                    val = - a_theta[i, :] * aips._u * sp.COO(atp.thetas.astype(float))  # not perfect
                    symbolic_array_dic[(self._theta_a(i), 0, 0)] += val * symbolic_params['hd']

                for j in range(nvar[0]):

                    jo = j + offset  # skipping the theta 0 variable if it exists

                    val = a_theta[i, :] @ aips._a[:, jo]
                    symbolic_array_dic[(self._theta_a(i), self._psi_a(j), 0)] += val * symbolic_params['kd'] * self.sig0 / 2
                    symbolic_array_dic[(self._theta_a(i), self._theta_a(jo), 0)] -= val * (symbolic_params['kd'] / 2 + 2 * symbolic_params['kdp']) * self.sig0

                    val = - a_theta[i, :] @ aips._c[:, jo]
                    symbolic_array_dic[(self._theta_a(i), self._theta_a(jo), 0)] += val * symbolic_params['beta'] * self.sig0

                    if gp is not None:
                        if gp.hk is not None:
                            #//TODO: Need to make this symbolic
                            if gp.orographic_basis == "atmospheric":
                                oro = a_theta[i, :] @ aips._g[:, jo, offset:] @ sp.COO(gp.hk.astype(float))  # not perfect
                            else:
                                oro = a_theta[i, :] @ aips._gh[:, jo, offset:] @ sp.COO(gp.hk.astype(float))  # not perfect
                            symbolic_array_dic[(self._theta_a(i), self._theta_a(jo), 0)] -= self.sig0 * oro / 2
                            symbolic_array_dic[(self._theta_a(i), self._psi_a(j), 0)] += self.sig0 * oro / 2

                    for k in range(nvar[0]):
                        ko = k + offset  # skipping the theta 0 variable if it exists
                        val = a_theta[i, :] @ aips._b[:, jo, ko]
                        symbolic_array_dic[(self._theta_a(i), self._psi_a(j), self._theta_a(ko))] = - val * self.sig0
                        symbolic_array_dic[(self._theta_a(i), self._theta_a(jo), self._psi_a(k))] = - val * self.sig0

                        val = a_theta[i, :] @ aips._g[:, jo, ko]
                        symbolic_array_dic[(self._theta_a(i), self._psi_a(j), self._theta_a(ko))] += val

                for j in range(nvar[1]):
                    val = a_theta[i, :] @ aips._u[:, j]
                    if self.Lpa is not None:
                        symbolic_array_dic[(self._theta_a(i), self._theta_a(j), 0)] += val * symbolic_params['sc'] * self.Lpa
                    if self.LSBpa is not None:
                        symbolic_array_dic[(self._theta_a(i), self._theta_a(j), 0)] += val * self.LSBpa

                    if atp.hd is not None:
                        symbolic_array_dic[(self._theta_a(i), self._theta_a(j), 0)] += val * atp.hd

                if ocean:
                    for j in range(nvar[2]):
                        jo = j + offset  # skipping the theta 0 variable if it exists
                        val = - a_theta[i, :] @ aips._d[:, jo]
                        symbolic_array_dic[(self._theta_a(i), self._psi_o(j), 0)] += val * self.sig0 * symbolic_params['kd'] / 2

                    if self.Lpa is not None:
                        for j in range(nvar[3]):
                            val = - a_theta[i, :] @ aips._s[:, j]
                            symbolic_array_dic[(self._theta_a(i), self._deltaT_o(j), 0)] += val * self.Lpa / 2
                            if self.LSBpgo is not None:
                                symbolic_array_dic[(self._theta_a(i), self._deltaT_o(j), 0)] += val * self.LSBpgo

                if ground_temp:
                    if self.Lpa is not None:
                        for j in range(nvar[2]):
                            val = - a_theta[i, :] @ aips._s[:, j]
                            symbolic_array_dic[(self._theta_a(i), self._deltaT_g(j), 0)] += val * self.Lpa / 2
                            if self.LSBpgo is not None:
                                symbolic_array_dic[(self._theta_a(i), self._deltaT_g(j), 0)] += val * self.LSBpgo

            if ocean:
                # psi_o part
                for i in range(nvar[2]):
                    for j in range(nvar[0]):
                        jo = j + offset  # skipping the theta 0 variable if it exists
                        val = M_psio[i, :] @ bips._K[offset:, jo] * symbolic_params['d']
                        symbolic_array_dic[(self._psi_o(i), self._psi_a(j), 0)] += val
                        symbolic_array_dic[(self._psi_o(i), self._theta_a(jo), 0)] -= val

                    for j in range(nvar[2]):
                        jo = j + offset  # skipping the T 0 variable if it exists
                        val = - M_psio[i, :] @ bips._N[offset:, jo]
                        symbolic_array_dic[(self._psi_o(i), self._psi_o(j), 0)] += val * symbolic_params['beta']

                        val = - M_psio[i, :] @ bips._M[offset:, jo]
                        symbolic_array_dic[(self._psi_o(i), self._psi_o(j), 0)] += val * (symbolic_params['r'] + symbolic_params['d'])

                        for k in range(nvar[2]):
                            ko = k + offset  # skipping the T 0 variable if it exists
                            symbolic_array_dic[(self._psi_o(i), self._psi_o(j), self._psi_o(k))] -= M_psio[i, :] @ bips._C[offset:, jo, ko]

                # deltaT_o part
                for i in range(nvar[3]):

                    symbolic_array_dic[(self._deltaT_o(i), 0, 0)] += U_inv[i, :] @ bips._W @ sp.COO(self.Cpgo)

                    for j in range(nvar[1]):
                        val = U_inv[i, :] @ bips._W[:, j]
                        symbolic_array_dic[(self._deltaT_o(i), self._theta_a(j), 0)] += val * 2 * symbolic_params['sc'] * self.Lpgo
                        if self.sbpa is not None:
                            symbolic_array_dic[(self._deltaT_o(i), self._theta_a(j), 0)] += val * self.sbpa

                    for j in range(nvar[3]):
                        symbolic_array_dic[(self._deltaT_o(i), self._deltaT_o(j), 0)] = - self.Lpgo * _kronecker_delta(i, j)
                        if self.sbpgo is not None:
                            symbolic_array_dic[(self._deltaT_o(i), self._deltaT_o(j), 0)] += - self.sbpgo * _kronecker_delta(i, j)

                    for j in range(nvar[2]):
                        for k in range(nvar[2]):
                            jo = j + offset  # skipping the T 0 variable if it exists
                            ko = k + offset  # skipping the T 0 variable if it exists
                            symbolic_array_dic[(self._deltaT_o(i), self._psi_o(j), self._deltaT_o(ko))] -= U_inv[i, :] @ bips._O[:, jo, ko]

            # deltaT_g part
            if ground_temp:
                for i in range(nvar[2]):
                    symbolic_array_dic[(self._deltaT_g(i), 0, 0)] += U_inv[i, :] @ bips._W @ sp.COO(self.Cpgo)  # not perfect

                    for j in range(nvar[1]):
                        val = U_inv[i, :] @ bips._W[:, j]
                        symbolic_array_dic[(self._deltaT_g(i), self._theta_a(j), 0)] += val * 2 * symbolic_params['sc'] * self.Lpgo
                        if self.sbpa is not None:
                            symbolic_array_dic[(self._deltaT_g(i), self._theta_a(j), 0)] += val * self.sbpa

                    for j in range(nvar[2]):
                        symbolic_array_dic[(self._deltaT_g(i), self._deltaT_g(j), 0)] = - self.Lpgo * _kronecker_delta(i, j)
                        if self.sbpgo is not None:
                            symbolic_array_dic[(self._deltaT_g(i), self._deltaT_g(j), 0)] += - self.sbpgo * _kronecker_delta(i, j)

        else:
            # psi_a part
            for i in range(nvar[0]):
                for j in range(nvar[0]):

                    jo = j + offset

                    val = 0
                    for jj in range(nvar[0]):
                        val += a_inv[i, jj] * aips.c(offset + jj, jo)
                    symbolic_array_dic[(self._psi_a(i), self._psi_a(j), 0)] -= val * symbolic_params['beta']

                    symbolic_array_dic[(self._psi_a(i), self._psi_a(j), 0)] -= (symbolic_params['kd'] * _kronecker_delta(i, j)) / 2
                    symbolic_array_dic[(self._psi_a(i), self._theta_a(jo), 0)] = (ap.kd * _kronecker_delta(i, j)) / 2

                    #//TODO: what is gp.hk parameter???
                    if gp is not None:
                        if symbolic_params['hk'] is not None:
                            oro = 0
                            if gp.orographic_basis == "atmospheric":
                                for jj in range(nvar[0]):
                                    for kk in range(nvar[0]):
                                        oro += a_inv[i, jj] * aips.g(offset + jj, j, offset + kk) * self.hk[kk]
                            else:
                                for jj in range(nvar[0]):
                                    for kk in range(nvar[0]):
                                        oro += a_inv[i, jj] * aips.gh(offset + jj, j, offset + kk) * gp.hk[kk]
                            symbolic_array_dic[(self._psi_a(i), self._psi_a(j), 0)] -= oro / 2
                            symbolic_array_dic[(self._psi_a(i), self._theta_a(jo), 0)] += oro / 2

                    for k in range(nvar[0]):
                        ko = k + offset  # skipping the theta 0 variable if it exists
                        val = 0
                        for jj in range(nvar[0]):
                            val += a_inv[i, jj] * aips.b(offset + jj, jo, ko)
                        symbolic_array_dic[(self._psi_a(i), self._psi_a(j), self._psi_a(k))] = - val
                        symbolic_array_dic[(self._psi_a(i), self._theta_a(jo), self._theta_a(ko))] = - val
                if ocean:
                    for j in range(nvar[2]):
                        jo = j + offset
                        val = 0
                        for jj in range(nvar[0]):
                            val += a_inv[i, jj] * aips.d(offset + jj, jo)
                        symbolic_array_dic[(self._psi_a(i), self._psi_o(j), 0)] += val * ap.kd / 2


            # theta_a part
            for i in range(nvar[1]):
                if self.Cpa is not None:
                    for jj in range(nvar[1]):
                        for kk in range(nvar[1]):
                            symbolic_array_dic[(self._theta_a(i), 0, 0)] -= a_theta[i, jj] * aips.u(jj, kk) * self.Cpa[kk]

                if symbolic_params['hd'] is not None and self.thetas is not None:
                    val = 0
                    for jj in range(nvar[1]):
                        for kk in range(nvar[1]):
                            val -= a_theta[i, jj] * aips.u(jj, kk) * self.thetas[kk]
                    symbolic_array_dic[(self._theta_a(i), 0, 0)] += val * symbolic_params['hd']

                for j in range(nvar[0]):

                    jo = j + offset  # skipping the theta 0 variable if it exists

                    val = 0
                    for jj in range(nvar[1]):
                        val += a_theta[i, jj] * aips.a(jj, jo)
                    symbolic_array_dic[(self._theta_a(i), self._psi_a(j), 0)] += val * symbolic_params['kd'] * self.sig0 / 2
                    symbolic_array_dic[(self._theta_a(i), self._theta_a(jo), 0)] -= val * (symbolic_params['kd'] / 2 + 2 * symbolic_params['kdp']) * self.sig0

                    val = 0
                    for jj in range(nvar[1]):
                        val -= a_theta[i, jj] * aips.c(jj, jo)
                    symbolic_array_dic[(self._theta_a(i), self._theta_a(jo), 0)] += val * symbolic_params['beta'] * self.sig0

                    if gp is not None:
                        if symbolic_params['hk'] is not None:
                            oro = 0
                            if gp.orographic_basis == "atmospheric":
                                for jj in range(nvar[1]):
                                    for kk in range(nvar[0]):
                                        oro += a_theta[i, jj] * aips.g(jj, jo, offset + kk) * self.hk[kk]
                            else:
                                for jj in range(nvar[1]):
                                    for kk in range(nvar[0]):
                                        oro += a_theta[i, jj] * aips.gh(jj, jo, offset + kk) * self.hk[kk]
                            symbolic_array_dic[(self._theta_a(i), self._theta_a(jo), 0)] -= self.sig0 * oro / 2
                            symbolic_array_dic[(self._theta_a(i), self._psi_a(j), 0)] += self.sig0 * oro / 2

                    for k in range(nvar[0]):
                        ko = k + offset  # skipping the theta 0 variable if it exists
                        val = 0
                        for jj in range(nvar[1]):
                            val += a_theta[i, jj] * aips.b(jj, jo, ko)
                        symbolic_array_dic[(self._theta_a(i), self._psi_a(j), self._theta_a(ko))] = - val * self.sig0
                        symbolic_array_dic[(self._theta_a(i), self._theta_a(jo), self._psi_a(k))] = - val * self.sig0

                        val = 0
                        for jj in range(nvar[1]):
                            val += a_theta[i, jj] * aips.g(jj, jo, ko)

                        symbolic_array_dic[(self._theta_a(i), self._psi_a(j), self._theta_a(ko))] += val

                for j in range(nvar[1]):
                    val = 0
                    for jj in range(nvar[1]):
                        val += a_theta[i, jj] * aips.u(jj, j)

                    if self.Lpa is not None:
                        symbolic_array_dic[(self._theta_a(i), self._theta_a(j), 0)] += val * symbolic_params['sc'] * self.Lpa
                    if self.LSBpa is not None:
                        symbolic_array_dic[(self._theta_a(i), self._theta_a(j), 0)] += val * self.LSBpa

                    if symbolic_params['hd'] is not None:
                        symbolic_array_dic[(self._theta_a(i), self._theta_a(j), 0)] += val * symbolic_params['hd']

                if ocean:
                    for j in range(nvar[2]):
                        jo = j + offset  # skipping the theta 0 variable if it exists
                        val = 0
                        for jj in range(nvar[1]):
                            val -= a_theta[i, jj] * aips.d(jj, jo)
                        symbolic_array_dic[(self._theta_a(i), self._psi_o(j), 0)] += val * self.sig0 * symbolic_params['kd'] / 2

                    if self.Lpa is not None:
                        for j in range(nvar[3]):
                            val = 0
                            for jj in range(nvar[1]):
                                val -= a_theta[i, jj] * aips.s(jj, j)
                            symbolic_array_dic[(self._theta_a(i), self._deltaT_o(j), 0)] += val * self.Lpa / 2
                            if self.LSBpgo is not None:
                                symbolic_array_dic[(self._theta_a(i), self._deltaT_o(j), 0)] += val * self.LSBpgo

                if ground_temp:
                    if self.Lpa is not None:
                        for j in range(nvar[2]):
                            val = 0
                            for jj in range(nvar[1]):
                                val -= a_theta[i, jj] * aips.s(jj, j)
                            symbolic_array_dic[(self._theta_a(i), self._deltaT_g(j), 0)] += val * self.Lpa / 2
                            if self.LSBpgo is not None:
                                symbolic_array_dic[(self._theta_a(i), self._deltaT_g(j), 0)] += val * self.LSBpgo

            if ocean:
                # psi_o part
                for i in range(nvar[2]):
                    for j in range(nvar[0]):
                        jo = j + offset  # skipping the theta 0 variable if it exists

                        for jj in range(nvar[2]):
                            val = M_psio[i, jj] * bips.K(offset + jj, jo) * symbolic_params['d']
                            symbolic_array_dic[(self._psi_o(i), self._psi_a(j), 0)] += val
                            symbolic_array_dic[(self._psi_o(i), self._theta_a(jo), 0)] -= val

                    for j in range(nvar[2]):
                        jo = j + offset  # skipping the T 0 variable if it exists

                        val = 0
                        for jj in range(nvar[2]):
                            val -= M_psio[i, jj] * bips.N(offset + jj, jo)
                        symbolic_array_dic[(self._psi_o(i), self._psi_o(j), 0)] += val * symbolic_params['beta']

                        val = 0
                        for jj in range(nvar[2]):
                            val -= M_psio[i, jj] * bips.M(offset + jj, jo)
                        symbolic_array_dic[(self._psi_o(i), self._psi_o(j), 0)] += val * (symbolic_params['r'] + symbolic_params['d'])

                        for k in range(nvar[2]):
                            ko = k + offset  # skipping the T 0 variable if it exists
                            for jj in range(nvar[2]):
                                symbolic_array_dic[(self._psi_o(i), self._psi_o(j), self._psi_o(k))] -= M_psio[i, jj] * bips.C(offset + jj, jo, ko)

                # deltaT_o part
                for i in range(nvar[3]):
                    for jj in range(nvar[1]):
                        for kk in range(nvar[3]):
                            symbolic_array_dic[(self._deltaT_o(i), 0, 0)] += U_inv[i, kk] * bips.W(kk, jj) * self.Cpgo[jj]

                    for j in range(nvar[1]):
                        val = 0
                        for jj in range(nvar[3]):
                            val += U_inv[i, jj] * bips.W(jj, j)
                        symbolic_array_dic[(self._deltaT_o(i), self._theta_a(j), 0)] += val * 2 * symbolic_params['sc'] * self.Lpgo
                        if self.sbpa is not None:
                            symbolic_array_dic[(self._deltaT_o(i), self._theta_a(j), 0)] += val * self.sbpa

                    for j in range(nvar[3]):
                        symbolic_array_dic[(self._deltaT_o(i), self._deltaT_o(j), 0)] = - self.Lpgo * _kronecker_delta(i, j)
                        if self.sbpgo is not None:
                            symbolic_array_dic[(self._deltaT_o(i), self._deltaT_o(j), 0)] += - self.sbpgo * _kronecker_delta(i, j)

                    for j in range(nvar[2]):
                        for k in range(offset, nvar[3]):
                            jo = j + offset  # skipping the T 0 variable if it exists
                            for jj in range(nvar[3]):
                                symbolic_array_dic[(self._deltaT_o(i), self._psi_o(j), self._deltaT_o(k))] -= U_inv[i, jj] * bips.O(jj, jo, k)

            # deltaT_g part
            if ground_temp:
                for i in range(nvar[2]):
                    for jj in range(nvar[1]):
                        for kk in range(nvar[2]):
                            symbolic_array_dic[(self._deltaT_g(i), 0, 0)] += U_inv[i, kk] * bips.W(kk, jj) * self.Cpgo[jj]

                    for j in range(nvar[1]):
                        val = 0
                        for jj in range(nvar[2]):
                            val += U_inv[i, jj] * bips.W(jj, j)
                        symbolic_array_dic[(self._deltaT_g(i), self._theta_a(j), 0)] += val * 2 * symbolic_params['sc'] * self.Lpgo
                        if self.sbpa is not None:
                            symbolic_array_dic[(self._deltaT_g(i), self._theta_a(j), 0)] += val * self.sbpa

                    for j in range(nvar[2]):
                        symbolic_array_dic[(self._deltaT_g(i), self._deltaT_g(j), 0)] = - self.Lpgo * _kronecker_delta(i, j)
                        if self.sbpgo is not None:
                            symbolic_array_dic[(self._deltaT_g(i), self._deltaT_g(j), 0)] += - self.sbpgo * _kronecker_delta(i, j)

        return symbolic_array_dic

    def compute_tensor(self):
        """Routine to compute the tensor."""
        # gathering
        par = self.params
        ndim = par.ndim

        sparse_arrays_dict = self._compute_tensor_dicts()

        tensor = sp.zeros((ndim + 1, ndim + 1, ndim + 1), dtype=np.float64, format='coo')
        if sparse_arrays_dict is not None:
            tensor = self._add_dict_to_tensor(sparse_arrays_dict, tensor)
        self._set_tensor(tensor)

    def _set_tensor(self, tensor):
        if not isinstance(tensor, sp.COO):
            tensor = tensor.to_coo()
        self.jacobian_tensor = self.jacobian_from_tensor(tensor)
        self.tensor = self.simplify_tensor(tensor)

    #//TODO: include other functions for altering dictionary.

class SymbolicTensorDynamicT(SymbolicTensorLinear):
    #//TODO: Need to work out symbolic tensor dot

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
            a_theta = sy.matrices.Inverse(a_theta)

        if bips is not None:
            U_inv = dict()
            if ocean:
                for i in range(nvar[3]):
                    for j in range(nvar[3]):
                        U_inv[(i, j)] = bips.U(i, j)
                U_inv = sy.matrices.immutable.ImmutableSparseMatrix(nvar[3], nvar[3], U_inv)
                U_inv = sy.matrices.Inverse(U_inv)
            else:
                for i in range(nvar[2]):
                    for j in range(nvar[2]):
                        U_inv[i, j] = bips.U(i, j)
                U_inv = sy.matrices.immutable.ImmutableSparseMatrix(nvar[2], nvar[2], U_inv)
                U_inv = sy.matrices.immutable.ImmutableSparseMatrix(U_inv)

        #################

        symbolic_array_full_dict = dict()
        # theta_a part
        for i in range(nvar[1]):
            # symbolic_array_full_dict[self._theta_a(i)] = list()

            if self.T4LSBpa is not None:
                # val = sy.physics.quantum.tensorproduct.TensorProduct((a_theta[i], aips._z))
                # val = sp.tensordot(a_theta[i], aips._z, axes=1)
                if val.nnz > 0:
                    symbolic_array_full_dict[self._theta_a(i)].append(self._shift_tensor_coordinates(self.T4LSBpa * val, self._theta_a(0)))

            if ocean:
                if par.T4LSBpgo is not None:
                    val = sp.tensordot(a_theta[i], aips._v, axes=1)
                    if val.nnz > 0:
                        symbolic_array_full_dict[self._theta_a(i)].append(self._shift_tensor_coordinates(- self.T4LSBpgo * val, self._deltaT_o(0)))

            if ground_temp:

                if par.T4LSBpgo is not None:
                    val = sp.tensordot(a_theta[i], aips._v, axes=1)
                    if val.nnz > 0:
                        symbolic_array_full_dict[self._theta_a(i)].append(self._shift_tensor_coordinates(- self.T4LSBpgo * val, self._deltaT_g(0)))

        if ocean:
            # deltaT_o part
            for i in range(nvar[3]):
                symbolic_array_full_dict[self._deltaT_o(i)] = list()
                val = sp.tensordot(U_inv[i], bips._Z, axes=1)
                if val.nnz > 0:
                    symbolic_array_full_dict[self._deltaT_o(i)].append(self._shift_tensor_coordinates(self.T4sbpa * val, self._theta_a(0)))
                val = sp.tensordot(U_inv[i], bips._V, axes=1)
                if val.nnz > 0:
                    symbolic_array_full_dict[self._deltaT_o(i)].append(self._shift_tensor_coordinates(- self.T4sbpgo * val, self._deltaT_o(0)))

        if ground_temp:
            # deltaT_g part
            for i in range(nvar[2]):
                symbolic_array_full_dict[self._deltaT_g(i)] = list()
                val = sp.tensordot(U_inv[i], bips._Z, axes=1)
                if val.nnz > 0:
                    symbolic_array_full_dict[self._deltaT_g(i)].append(self._shift_tensor_coordinates(self.T4sbpa * val, self._theta_a(0)))
                val = sp.tensordot(U_inv[i], bips._V, axes=1)
                if val.nnz > 0:
                    symbolic_array_full_dict[self._deltaT_g(i)].append(self._shift_tensor_coordinates(- self.T4sbpgo * val, self._deltaT_g(0)))

        return symbolic_array_full_dict

    def _compute_non_stored_full_dict(self):
        par = self.params
        ap = par.atmospheric_params
        nvar = par.number_of_variables
        ndim = par.ndim
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

            a_theta = np.zeros((nvar[1], nvar[1]))
            for i in range(nvar[1]):
                for j in range(nvar[1]):
                    a_theta[i, j] = self.sig0 * aips.a(i, j) - aips.u(i, j)
            a_theta = np.linalg.inv(a_theta)
            a_theta = sp.COO(a_theta)

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
            
            U_inv = sy.matrices.Inverse(U_inv)
                

        #################

        symbolic_array_full_dict = dict()
        # theta_a part
        for i in range(nvar[1]):
            # t_full = sp.zeros((ndim + 1, ndim + 1, ndim + 1, ndim + 1), dtype=np.float64, format='dok')

            if self.T4LSBpa is not None:
                j = k = ell = 0
                for m in range(nvar[1]):
                    val = 0
                    for jj in range(nvar[1]):
                        val += a_theta[i, jj] * aips.z(jj, j, k, ell, m)
                    if m == 0:
                        symbolic_array_full_dict[(self._theta_a(i), self._theta_a(j), self._theta_a(k), self._theta_a(ell), self._theta_a(m))] = self.T4LSBpa * val
                    else:
                        symbolic_array_full_dict[(self._theta_a(i), self._theta_a(j), self._theta_a(k), self._theta_a(ell), self._theta_a(m))] = 4 * self.T4LSBpa * val

            if ocean:
                if self.T4LSBpgo is not None:
                    j = k = ell = 0
                    for m in range(nvar[3]):
                        val = 0
                        for jj in range(nvar[1]):
                            val -= a_theta[i, jj] * aips.v(jj, j, k, ell, m)
                        if m == 0:
                            symbolic_array_full_dict[(self._theta_a(i), self._deltaT_o(j), self._deltaT_o(k), self._deltaT_o(ell), self._deltaT_o(m))] = self.T4LSBpgo * val
                        else:
                            symbolic_array_full_dict[(self._theta_a(i), self._deltaT_o(j), self._deltaT_o(k), self._deltaT_o(ell), self._deltaT_o(m))] = 4 * self.T4LSBpgo * val

            if ground_temp:
                if self.T4LSBpgo is not None:
                    j = k = ell = 0
                    for m in range(nvar[2]):
                        val = 0
                        for jj in range(nvar[1]):
                            val -= a_theta[i, jj] * aips.v(jj, j, k, ell, m)
                        if m == 0:
                            symbolic_array_full_dict[(self._theta_a(i), self._deltaT_g(j), self._deltaT_g(k), self._deltaT_g(ell), self._deltaT_g(m))] = self.T4LSBpgo * val
                        else:
                            symbolic_array_full_dict[(self._theta_a(i), self._deltaT_g(j), self._deltaT_g(k), self._deltaT_g(ell), self._deltaT_g(m))] = 4 * self.T4LSBpgo * val

        if ocean:

            # deltaT_o part
            for i in range(nvar[3]):

                # t_full = sp.zeros((ndim + 1, ndim + 1, ndim + 1, ndim + 1), dtype=np.float64, format='dok')

                j = k = ell = 0
                for m in range(nvar[1]):
                    val = 0
                    for jj in range(nvar[3]):
                        val += U_inv[i, jj] * bips.Z(jj, j, k, ell, m)
                    if m == 0:
                        symbolic_array_full_dict[(self._deltaT_g(i), self._theta_a(j), self._theta_a(k), self._theta_a(ell), self._theta_a(m))] = self.T4sbpa * val
                    else:
                        symbolic_array_full_dict[(self._deltaT_g(i), self._theta_a(j), self._theta_a(k), self._theta_a(ell), self._theta_a(m))] = 4 * self.T4sbpa * val

                for m in range(nvar[3]):
                    val = 0
                    for jj in range(nvar[3]):
                        val -= U_inv[i, jj] * bips.V(jj, j, k, ell, m)
                    if m == 0:
                        symbolic_array_full_dict[(self._deltaT_g(i), self._deltaT_o(j), self._deltaT_o(k), self._deltaT_o(ell), self._deltaT_o(m))] = self.T4sbpgo * val
                    else:
                        symbolic_array_full_dict[(self._deltaT_g(i), self._deltaT_o(j), self._deltaT_o(k), self._deltaT_o(ell), self._deltaT_o(m))] = 4 * self.T4sbpgo * val

        # deltaT_g part
        if ground_temp:
            for i in range(nvar[2]):

                # t_full = sp.zeros((ndim + 1, ndim + 1, ndim + 1, ndim + 1), dtype=np.float64, format='dok')

                j = k = ell = 0
                for m in range(nvar[1]):
                    val = 0
                    for jj in range(nvar[2]):
                        val += U_inv[i, jj] * bips._Z[jj, j, k, ell, m]
                    if m == 0:
                        symbolic_array_full_dict[(self._deltaT_g(i), self._theta_a(j), self._theta_a(k), self._theta_a(ell), self._theta_a(m))] = self.T4sbpa * val
                    else:
                        symbolic_array_full_dict[(self._deltaT_g(i), self._theta_a(j), self._theta_a(k), self._theta_a(ell), self._theta_a(m))] = 4 * self.T4sbpa * val

                for m in range(nvar[2]):
                    val = 0
                    for jj in range(nvar[2]):
                        val -= U_inv[i, jj] * bips._V[jj, j, k, ell, m]
                    if m == 0:
                        symbolic_array_full_dict[(self._deltaT_g(i), self._deltaT_g(j), self._deltaT_g(k), self._deltaT_g(ell), self._deltaT_g(m))] = self.T4sbpgo * val
                    else:
                        symbolic_array_full_dict[(self._deltaT_g(i), self._deltaT_g(j), self._deltaT_g(k), self._deltaT_g(ell), self._deltaT_g(m))] = 4 * self.T4sbpgo * val

        return symbolic_array_full_dict

    def compute_tensor(self):
        """Routine to compute the tensor."""
        # gathering
        par = self.params
        ndim = par.ndim

        sparse_arrays_dict = SymbolicTensorLinear._compute_tensor_dicts(self)
        symbolic_array_full_dict = self._compute_tensor_dicts()

        tensor = sp.zeros((ndim + 1, ndim + 1, ndim + 1, ndim + 1, ndim + 1), dtype=np.float64, format='coo')
        if sparse_arrays_dict is not None:
            tensor = self._add_dict_to_tensor(sparse_arrays_dict, tensor)
        if symbolic_array_full_dict is not None:
            tensor = self._add_dict_to_tensor(symbolic_array_full_dict, tensor)
        self._set_tensor(tensor)


def _kronecker_delta(i, j):

    if i == j:
        return 1

    else:
        return 0