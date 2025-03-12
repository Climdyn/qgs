"""
    qgs tensor module
    =================

    This module computes and holds the tensors representing the tendencies of the model's equations.

    TODO: Add a list of the different tensor available

"""
from contextlib import redirect_stdout

import numpy as np
import sparse as sp
import pickle

real_eps = np.finfo(np.float64).eps


class QgsTensor(object):
    """qgs tendencies tensor class.

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
            a_inv = np.zeros((nvar[0], nvar[0]))
            for i in range(offset, nvar[1]):
                for j in range(offset, nvar[1]):
                    a_inv[i - offset, j - offset] = aips.a(i, j)

            a_inv = np.linalg.inv(a_inv)
            a_inv = sp.COO(a_inv)

            a_theta = np.zeros((nvar[1], nvar[1]))
            for i in range(nvar[1]):
                for j in range(nvar[1]):
                    a_theta[i, j] = ap.sig0 * aips.a(i, j) - aips.u(i, j)
            a_theta = np.linalg.inv(a_theta)
            a_theta = sp.COO(a_theta)

        if bips is not None:
            if ocean:
                U_inv = np.zeros((nvar[3], nvar[3]))
                for i in range(nvar[3]):
                    for j in range(nvar[3]):
                        U_inv[i, j] = bips.U(i, j)
                U_inv = np.linalg.inv(U_inv)
                U_inv = sp.COO(U_inv)

                M_psio = np.zeros((nvar[2], nvar[2]))
                for i in range(offset, nvar[3]):
                    for j in range(offset, nvar[3]):
                        M_psio[i - offset, j - offset] = bips.M(i, j) + par.G * bips.U(i, j)
                M_psio = np.linalg.inv(M_psio)
                M_psio = sp.COO(M_psio)
            else:
                U_inv = np.zeros((nvar[2], nvar[2]))
                for i in range(nvar[2]):
                    for j in range(nvar[2]):
                        U_inv[i, j] = bips.U(i, j)
                U_inv = np.linalg.inv(U_inv)
                U_inv = sp.COO(U_inv)

        #################

        if bips is not None:
            go = bips.stored
        else:
            go = True

        sparse_arrays_dict = dict()

        if aips.stored and go:
            # psi_a part
            for i in range(nvar[0]):
                t = sp.zeros((ndim + 1, ndim + 1), dtype=np.float64, format='dok')

                for j in range(nvar[0]):

                    jo = j + offset  # skipping the theta 0 variable if it exists

                    val = a_inv[i, :] @ aips._c[offset:, jo]
                    t[self._psi_a(j), 0] -= val * scp.beta

                    t[self._psi_a(j), 0] -= (ap.kd * _kronecker_delta(i, j)) / 2
                    t[self._theta_a(jo), 0] = (ap.kd * _kronecker_delta(i, j)) / 2

                    if gp is not None:
                        if gp.hk is not None:
                            if gp.orographic_basis == "atmospheric":
                                oro = a_inv[i, :] @ aips._g[offset:, jo, offset:] @ sp.COO(gp.hk.astype(float))  # not perfect
                            else:
                                # TODO: Can only be used with symbolic inner products here - a warning or an error should be raised if this is not the case.
                                oro = a_inv[i, :] @ aips._gh[offset:, jo, offset:] @ sp.COO(gp.hk.astype(float))  # not perfect
                            t[self._psi_a(j), 0] -= oro / 2
                            t[self._theta_a(jo), 0] += oro / 2

                    for k in range(nvar[0]):
                        ko = k + offset  # skipping the theta 0 variable if it exists
                        val = a_inv[i, :] @ aips._b[offset:, jo, ko]
                        t[self._psi_a(j), self._psi_a(k)] = - val
                        t[self._theta_a(jo), self._theta_a(ko)] = - val
                if ocean:
                    for j in range(nvar[2]):
                        jo = j + offset  # skipping the theta 0 variable if it exists
                        val = a_inv[i, :] @ aips._d[offset:, jo]
                        t[self._psi_o(j), 0] += val * ap.kd / 2

                sparse_arrays_dict[self._psi_a(i)] = t.to_coo()

            # theta_a part
            for i in range(nvar[1]):
                t = sp.zeros((ndim + 1, ndim + 1), dtype=np.float64, format='dok')

                if par.Cpa is not None:
                    t[0, 0] -= a_theta[i, :] @ aips._u @ sp.COO(par.Cpa.astype(float))  # not perfect

                if atp.hd is not None and atp.thetas is not None:
                    val = - a_theta[i, :] @ aips._u @ sp.COO(atp.thetas.astype(float))  # not perfect
                    t[0, 0] += val * atp.hd

                for j in range(nvar[0]):

                    jo = j + offset  # skipping the theta 0 variable if it exists

                    val = a_theta[i, :] @ aips._a[:, jo]
                    t[self._psi_a(j), 0] += val * ap.kd * ap.sig0 / 2
                    t[self._theta_a(jo), 0] -= val * (ap.kd / 2 + 2 * ap.kdp) * ap.sig0

                    val = - a_theta[i, :] @ aips._c[:, jo]
                    t[self._theta_a(jo), 0] += val * scp.beta * ap.sig0

                    if gp is not None:
                        if gp.hk is not None:
                            if gp.orographic_basis == "atmospheric":
                                oro = a_theta[i, :] @ aips._g[:, jo, offset:] @ sp.COO(gp.hk.astype(float))  # not perfect
                            else:
                                oro = a_theta[i, :] @ aips._gh[:, jo, offset:] @ sp.COO(gp.hk.astype(float))  # not perfect
                            t[self._theta_a(jo), 0] -= ap.sig0 * oro / 2
                            t[self._psi_a(j), 0] += ap.sig0 * oro / 2

                    for k in range(nvar[0]):
                        ko = k + offset  # skipping the theta 0 variable if it exists
                        val = a_theta[i, :] @ aips._b[:, jo, ko]
                        t[self._psi_a(j), self._theta_a(ko)] = - val * ap.sig0
                        t[self._theta_a(jo), self._psi_a(k)] = - val * ap.sig0

                        val = a_theta[i, :] @ aips._g[:, jo, ko]
                        t[self._psi_a(j), self._theta_a(ko)] += val

                for j in range(nvar[1]):
                    val = a_theta[i, :] @ aips._u[:, j]
                    if par.Lpa is not None:
                        t[self._theta_a(j), 0] += val * atp.sc * par.Lpa
                    if par.LSBpa is not None:
                        t[self._theta_a(j), 0] += val * par.LSBpa

                    if atp.hd is not None:
                        t[self._theta_a(j), 0] += val * atp.hd

                if ocean:
                    for j in range(nvar[2]):
                        jo = j + offset  # skipping the theta 0 variable if it exists
                        val = - a_theta[i, :] @ aips._d[:, jo]
                        t[self._psi_o(j), 0] += val * ap.sig0 * ap.kd / 2

                    if par.Lpa is not None:
                        for j in range(nvar[3]):
                            val = - a_theta[i, :] @ aips._s[:, j]
                            t[self._deltaT_o(j), 0] += val * par.Lpa / 2
                            if par.LSBpgo is not None:
                                t[self._deltaT_o(j), 0] += val * par.LSBpgo

                if ground_temp:
                    if par.Lpa is not None:
                        for j in range(nvar[2]):
                            val = - a_theta[i, :] @ aips._s[:, j]
                            t[self._deltaT_g(j), 0] += val * par.Lpa / 2
                            if par.LSBpgo is not None:
                                t[self._deltaT_g(j), 0] += val * par.LSBpgo

                sparse_arrays_dict[self._theta_a(i)] = t.to_coo()

            if ocean:
                # psi_o part
                for i in range(nvar[2]):

                    t = sp.zeros((ndim + 1, ndim + 1), dtype=np.float64, format='dok')

                    for j in range(nvar[0]):
                        jo = j + offset  # skipping the theta 0 variable if it exists
                        val = M_psio[i, :] @ bips._K[offset:, jo] * op.d
                        t[self._psi_a(j), 0] += val
                        t[self._theta_a(jo), 0] -= val

                    for j in range(nvar[2]):
                        jo = j + offset  # skipping the T 0 variable if it exists
                        val = - M_psio[i, :] @ bips._N[offset:, jo]
                        t[self._psi_o(j), 0] += val * scp.beta

                        val = - M_psio[i, :] @ bips._M[offset:, jo]
                        t[self._psi_o(j), 0] += val * (op.r + op.d)

                        for k in range(nvar[2]):
                            ko = k + offset  # skipping the T 0 variable if it exists
                            t[self._psi_o(j), self._psi_o(k)] -= M_psio[i, :] @ bips._C[offset:, jo, ko]

                    sparse_arrays_dict[self._psi_o(i)] = t.to_coo()

                # deltaT_o part
                for i in range(nvar[3]):

                    t = sp.zeros((ndim + 1, ndim + 1), dtype=np.float64, format='dok')
                    t[0, 0] += U_inv[i, :] @ bips._W @ sp.COO(par.Cpgo.astype(float))

                    for j in range(nvar[1]):
                        val = U_inv[i, :] @ bips._W[:, j]
                        t[self._theta_a(j), 0] += val * 2 * atp.sc * par.Lpgo
                        if par.sbpa is not None:
                            t[self._theta_a(j), 0] += val * par.sbpa

                    for j in range(nvar[3]):
                        t[self._deltaT_o(j), 0] = - par.Lpgo * _kronecker_delta(i, j)
                        if par.sbpgo is not None:
                            t[self._deltaT_o(j), 0] += - par.sbpgo * _kronecker_delta(i, j)

                    for j in range(nvar[2]):
                        for k in range(nvar[2]):
                            jo = j + offset  # skipping the T 0 variable if it exists
                            ko = k + offset  # skipping the T 0 variable if it exists
                            t[self._psi_o(j), self._deltaT_o(ko)] -= U_inv[i, :] @ bips._O[:, jo, ko]

                    sparse_arrays_dict[self._deltaT_o(i)] = t.to_coo()

            # deltaT_g part
            if ground_temp:
                for i in range(nvar[2]):

                    t = sp.zeros((ndim + 1, ndim + 1), dtype=np.float64, format='dok')
                    t[0, 0] += U_inv[i, :] @ bips._W @ sp.COO(par.Cpgo.astype(float))  # not perfect

                    for j in range(nvar[1]):
                        val = U_inv[i, :] @ bips._W[:, j]
                        t[self._theta_a(j), 0] += val * 2 * atp.sc * par.Lpgo
                        if par.sbpa is not None:
                            t[self._theta_a(j), 0] += val * par.sbpa

                    for j in range(nvar[2]):
                        t[self._deltaT_g(j), 0] = - par.Lpgo * _kronecker_delta(i, j)
                        if par.sbpgo is not None:
                            t[self._deltaT_g(j), 0] += - par.sbpgo * _kronecker_delta(i, j)

                    sparse_arrays_dict[self._deltaT_g(i)] = t.to_coo()

        else:
            # psi_a part
            for i in range(nvar[0]):
                t = sp.zeros((ndim + 1, ndim + 1), dtype=np.float64, format='dok')

                for j in range(nvar[0]):

                    jo = j + offset

                    val = 0
                    for jj in range(nvar[0]):
                        val += a_inv[i, jj] * aips.c(offset + jj, jo)
                    t[self._psi_a(j), 0] -= val * scp.beta

                    t[self._psi_a(j), 0] -= (ap.kd * _kronecker_delta(i, j)) / 2
                    t[self._theta_a(jo), 0] = (ap.kd * _kronecker_delta(i, j)) / 2

                    if gp is not None:
                        if gp.hk is not None:
                            oro = 0
                            if gp.orographic_basis == "atmospheric":
                                for jj in range(nvar[0]):
                                    for kk in range(nvar[0]):
                                        oro += a_inv[i, jj] * aips.g(offset + jj, j, offset + kk) * gp.hk[kk]
                            else:
                                for jj in range(nvar[0]):
                                    for kk in range(nvar[0]):
                                        oro += a_inv[i, jj] * aips.gh(offset + jj, j, offset + kk) * gp.hk[kk]
                            t[self._psi_a(j), 0] -= oro / 2
                            t[self._theta_a(jo), 0] += oro / 2

                    for k in range(nvar[0]):
                        ko = k + offset  # skipping the theta 0 variable if it exists
                        val = 0
                        for jj in range(nvar[0]):
                            val += a_inv[i, jj] * aips.b(offset + jj, jo, ko)
                        t[self._psi_a(j), self._psi_a(k)] = - val
                        t[self._theta_a(jo), self._theta_a(ko)] = - val
                if ocean:
                    for j in range(nvar[2]):
                        jo = j + offset
                        val = 0
                        for jj in range(nvar[0]):
                            val += a_inv[i, jj] * aips.d(offset + jj, jo)
                        t[self._psi_o(j), 0] += val * ap.kd / 2

                sparse_arrays_dict[self._psi_a(i)] = t.to_coo()

            # theta_a part
            for i in range(nvar[1]):
                t = sp.zeros((ndim + 1, ndim + 1), dtype=np.float64, format='dok')

                if par.Cpa is not None:
                    for jj in range(nvar[1]):
                        for kk in range(nvar[1]):
                            t[0, 0] -= a_theta[i, jj] * aips.u(jj, kk) * par.Cpa[kk]

                if atp.hd is not None and atp.thetas is not None:
                    val = 0
                    for jj in range(nvar[1]):
                        for kk in range(nvar[1]):
                            val -= a_theta[i, jj] * aips.u(jj, kk) * atp.thetas[kk]
                    t[0, 0] += val * atp.hd

                for j in range(nvar[0]):

                    jo = j + offset  # skipping the theta 0 variable if it exists

                    val = 0
                    for jj in range(nvar[1]):
                        val += a_theta[i, jj] * aips.a(jj, jo)
                    t[self._psi_a(j), 0] += val * ap.kd * ap.sig0 / 2
                    t[self._theta_a(jo), 0] -= val * (ap.kd / 2 + 2 * ap.kdp) * ap.sig0

                    val = 0
                    for jj in range(nvar[1]):
                        val -= a_theta[i, jj] * aips.c(jj, jo)
                    t[self._theta_a(jo), 0] += val * scp.beta * ap.sig0

                    if gp is not None:
                        if gp.hk is not None:
                            oro = 0
                            if gp.orographic_basis == "atmospheric":
                                for jj in range(nvar[1]):
                                    for kk in range(nvar[0]):
                                        oro += a_theta[i, jj] * aips.g(jj, jo, offset + kk) * gp.hk[kk]
                            else:
                                for jj in range(nvar[1]):
                                    for kk in range(nvar[0]):
                                        oro += a_theta[i, jj] * aips.gh(jj, jo, offset + kk) * gp.hk[kk]
                            t[self._theta_a(jo), 0] -= ap.sig0 * oro / 2
                            t[self._psi_a(j), 0] += ap.sig0 * oro / 2

                    for k in range(nvar[0]):
                        ko = k + offset  # skipping the theta 0 variable if it exists
                        val = 0
                        for jj in range(nvar[1]):
                            val += a_theta[i, jj] * aips.b(jj, jo, ko)
                        t[self._psi_a(j), self._theta_a(ko)] = - val * ap.sig0
                        t[self._theta_a(jo), self._psi_a(k)] = - val * ap.sig0

                        val = 0
                        for jj in range(nvar[1]):
                            val += a_theta[i, jj] * aips.g(jj, jo, ko)

                        t[self._psi_a(j), self._theta_a(ko)] += val

                for j in range(nvar[1]):
                    val = 0
                    for jj in range(nvar[1]):
                        val += a_theta[i, jj] * aips.u(jj, j)

                    if par.Lpa is not None:
                        t[self._theta_a(j), 0] += val * atp.sc * par.Lpa
                    if par.LSBpa is not None:
                        t[self._theta_a(j), 0] += val * par.LSBpa

                    if atp.hd is not None:
                        t[self._theta_a(j), 0] += val * atp.hd

                if ocean:
                    for j in range(nvar[2]):
                        jo = j + offset  # skipping the theta 0 variable if it exists
                        val = 0
                        for jj in range(nvar[1]):
                            val -= a_theta[i, jj] * aips.d(jj, jo)
                        t[self._psi_o(j), 0] += val * ap.sig0 * ap.kd / 2

                    if par.Lpa is not None:
                        for j in range(nvar[3]):
                            val = 0
                            for jj in range(nvar[1]):
                                val -= a_theta[i, jj] * aips.s(jj, j)
                            t[self._deltaT_o(j), 0] += val * par.Lpa / 2
                            if par.LSBpgo is not None:
                                t[self._deltaT_o(j), 0] += val * par.LSBpgo

                if ground_temp:
                    if par.Lpa is not None:
                        for j in range(nvar[2]):
                            val = 0
                            for jj in range(nvar[1]):
                                val -= a_theta[i, jj] * aips.s(jj, j)
                            t[self._deltaT_g(j), 0] += val * par.Lpa / 2
                            if par.LSBpgo is not None:
                                t[self._deltaT_g(j), 0] += val * par.LSBpgo

                sparse_arrays_dict[self._theta_a(i)] = t.to_coo()

            if ocean:
                # psi_o part
                for i in range(nvar[2]):

                    t = sp.zeros((ndim + 1, ndim + 1), dtype=np.float64, format='dok')

                    for j in range(nvar[0]):
                        jo = j + offset  # skipping the theta 0 variable if it exists

                        for jj in range(nvar[2]):
                            val = M_psio[i, jj] * bips.K(offset + jj, jo) * op.d
                            t[self._psi_a(j), 0] += val
                            t[self._theta_a(jo), 0] -= val

                    for j in range(nvar[2]):
                        jo = j + offset  # skipping the T 0 variable if it exists

                        val = 0
                        for jj in range(nvar[2]):
                            val -= M_psio[i, jj] * bips.N(offset + jj, jo)
                        t[self._psi_o(j), 0] += val * scp.beta

                        val = 0
                        for jj in range(nvar[2]):
                            val -= M_psio[i, jj] * bips.M(offset + jj, jo)
                        t[self._psi_o(j), 0] += val * (op.r + op.d)

                        for k in range(nvar[2]):
                            ko = k + offset  # skipping the T 0 variable if it exists
                            for jj in range(nvar[2]):
                                t[self._psi_o(j), self._psi_o(k)] -= M_psio[i, jj] * bips.C(offset + jj, jo, ko)

                    sparse_arrays_dict[self._psi_o(i)] = t.to_coo()

                # deltaT_o part
                for i in range(nvar[3]):

                    t = sp.zeros((ndim + 1, ndim + 1), dtype=np.float64, format='dok')
                    for jj in range(nvar[1]):
                        for kk in range(nvar[3]):
                            t[0, 0] += U_inv[i, kk] * bips.W(kk, jj) * par.Cpgo[jj]

                    for j in range(nvar[1]):
                        val = 0
                        for jj in range(nvar[3]):
                            val += U_inv[i, jj] * bips.W(jj, j)
                        t[self._theta_a(j), 0] += val * 2 * atp.sc * par.Lpgo
                        if par.sbpa is not None:
                            t[self._theta_a(j), 0] += val * par.sbpa

                    for j in range(nvar[3]):
                        t[self._deltaT_o(j), 0] = - par.Lpgo * _kronecker_delta(i, j)
                        if par.sbpgo is not None:
                            t[self._deltaT_o(j), 0] += - par.sbpgo * _kronecker_delta(i, j)

                    for j in range(nvar[2]):
                        for k in range(offset, nvar[3]):
                            jo = j + offset  # skipping the T 0 variable if it exists
                            for jj in range(nvar[3]):
                                t[self._psi_o(j), self._deltaT_o(k)] -= U_inv[i, jj] * bips.O(jj, jo, k)

                    sparse_arrays_dict[self._deltaT_o(i)] = t.to_coo()

            # deltaT_g part
            if ground_temp:
                for i in range(nvar[2]):

                    t = sp.zeros((ndim + 1, ndim + 1), dtype=np.float64, format='dok')
                    for jj in range(nvar[1]):
                        for kk in range(nvar[2]):
                            t[0, 0] += U_inv[i, kk] * bips.W(kk, jj) * par.Cpgo[jj]

                    for j in range(nvar[1]):
                        val = 0
                        for jj in range(nvar[2]):
                            val += U_inv[i, jj] * bips.W(jj, j)
                        t[self._theta_a(j), 0] += val * 2 * atp.sc * par.Lpgo
                        if par.sbpa is not None:
                            t[self._theta_a(j), 0] += val * par.sbpa

                    for j in range(nvar[2]):
                        t[self._deltaT_g(j), 0] = - par.Lpgo * _kronecker_delta(i, j)
                        if par.sbpgo is not None:
                            t[self._deltaT_g(j), 0] += - par.sbpgo * _kronecker_delta(i, j)

                    sparse_arrays_dict[self._deltaT_g(i)] = t.to_coo()

        return sparse_arrays_dict

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

    @staticmethod
    def _add_dict_to_tensor(dic, tensor):
        shape = tensor.shape
        rank = len(shape)
        if tensor.coords.size == 0:
            coords = np.array(rank * [[0]], dtype=int)
            data = np.array([0.])
        else:
            coords = tensor.coords.copy()
            data = tensor.data.copy()
        for i in dic:
            if not isinstance(dic[i], (list, tuple)):
                dl = [dic[i]]
            else:
                dl = dic[i]
            for d in dl:
                t_coords = d.coords
                t_rank = t_coords.shape[0]
                values = d.data
                new_coords = np.concatenate((np.full((1, len(values)), i), t_coords), axis=0)
                for _ in range(t_rank+1, rank):
                    new_coords = np.concatenate((new_coords, np.zeros((1, len(values)), dtype=int)))
                data = np.concatenate((data, values))
                coords = np.concatenate((coords, new_coords), axis=1)

        return sp.COO(coords, data, shape=shape)

    @staticmethod
    def _shift_tensor_coordinates(tensor, shift):
        new_coords = tensor.coords.copy() + shift
        return sp.COO(new_coords.astype(int), tensor.data)

    @staticmethod
    def jacobian_from_tensor(tensor):
        """Function to compute the Jacobian tensor.

        Parameters
        ----------
        tensor: sparse.COO
            The qgs tensor.

        Returns
        -------
        sparse.COO
            The Jacobian tensor.
        """

        n_perm = len(tensor.shape) - 2

        jacobian_tensor = tensor.copy()

        for i in range(1, n_perm+1):
            jacobian_tensor += tensor.swapaxes(1, i+1)

        return jacobian_tensor

    @staticmethod
    def simplify_tensor(tensor):
        """Routine that simplifies the component of a tensor :math:`\\mathcal{T}`.
        For each index :math:`i`, it upper-triangularizes the
        tensor :math:`\\mathcal{T}_{i,\\ldots}` for all the subsequent indices.

        Parameters
        ----------
        tensor: sparse.COO
            The tensor to simplify.

        Returns
        -------
        sparse.COO
            The upper-triangularized tensor.
        """
        coords = tensor.coords.copy()
        sorted_indices = np.sort(coords[1:, :], axis=0)
        coords[1:, :] = sorted_indices

        upp_tensor = sp.COO(coords, tensor.data.copy(), shape=tensor.shape, prune=True)

        return upp_tensor

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

    def load_from_file(self, filename, **kwargs):
        """Function to load previously a saved tensor object with the method :meth:`save_to_file`.

        Parameters
        ----------
        filename: str
            The file name where the tensor object was saved.
        kwargs: dict
            Keyword arguments to pass to the :mod:`pickle` module method.
        """
        f = open(filename, 'rb')
        tmp_dict = pickle.load(f, **kwargs)
        f.close()

        self.__dict__.clear()
        self.__dict__.update(tmp_dict)

    def print_tensor(self, tensor_name=""):
        """Routine to print the tensor.

        Parameters
        ----------
        tensor_name: str, optional
            Specify the name to print beside the values of the tensor. Default to `QgsTensor`.
        """
        if not tensor_name:
            tensor_name = 'QgsTensor'
        for coo, val in zip(self.tensor.coords.T, self.tensor.data):
            self._string_format(print, tensor_name, coo, val)

    def print_tensor_to_file(self, filename, tensor_name=""):
        """Routine to print the tensor to a file.

        Parameters
        ----------
        filename: str
            The filename where to print the tensor.
        tensor_name: str, optional
            Specify the name to print beside the values of the tensor. Default to `QgsTensor`.
        """
        with open(filename, 'w') as f:
            with redirect_stdout(f):
                self.print_tensor(tensor_name)

    def print_jacobian_tensor(self, tensor_name=""):
        """Routine to print the Jacobian tensor.

        Parameters
        ----------
        tensor_name: str, optional
            Specify the name to print beside the values of the tensor. Default to `QgsTensorJacobian`.
        """
        if not tensor_name:
            tensor_name = 'QgsTensorJacobian'
        for coo, val in zip(self.jacobian_tensor.coords.T, self.jacobian_tensor.data):
            self._string_format(print, tensor_name, coo, val)

    def print_jacobian_tensor_to_file(self, filename, tensor_name=""):
        """Routine to print the Jacobian tensor to a file.

        Parameters
        ----------
        filename: str
            The filename where to print the tensor.
        tensor_name: str, optional
            Specify the name to print beside the values of the tensor. Default to `QgsTensorJacobian`.
        """
        with open(filename, 'w') as f:
            with redirect_stdout(f):
                self.print_jacobian_tensor(tensor_name)

    @staticmethod
    def _string_format(func, symbol, indices, value):
        if abs(value) >= real_eps:
            s = symbol
            for i in indices:
                s += "["+str(i)+"]"
            s += " = % .5E" % value
            func(s)


class QgsTensorDynamicT(QgsTensor):
    """qgs dynamical temperature first order (linear) tendencies tensor class.

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
    tensor: sparse.COO(float)
        The tensor :math:`\\mathcal{T}_{i,j,k}` :math:`i`-th components.
    jacobian_tensor: sparse.COO(float)
        The jacobian tensor :math:`\\mathcal{T}_{i,j,k} + \\mathcal{T}_{i,k,j}` :math:`i`-th components.
    """

    def __init__(self, params=None, atmospheric_inner_products=None, oceanic_inner_products=None, ground_inner_products=None):

        QgsTensor.__init__(self, params, atmospheric_inner_products, oceanic_inner_products, ground_inner_products)

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

            sparse_arrays_full_dict = self._compute_stored_full_dict()

        else:

            sparse_arrays_full_dict = self._compute_non_stored_full_dict()

        return sparse_arrays_full_dict

    def _compute_stored_full_dict(self):
        par = self.params
        ap = par.atmospheric_params
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

            a_theta = np.zeros((nvar[1], nvar[1]))
            for i in range(nvar[1]):
                for j in range(nvar[1]):
                    a_theta[i, j] = ap.sig0 * aips.a(i, j) - aips.u(i, j)
            a_theta = np.linalg.inv(a_theta)
            a_theta = sp.COO(a_theta)

        if bips is not None:
            if ocean:
                U_inv = np.zeros((nvar[3], nvar[3]))
                for i in range(nvar[3]):
                    for j in range(nvar[3]):
                        U_inv[i, j] = bips.U(i, j)
                U_inv = np.linalg.inv(U_inv)
                U_inv = sp.COO(U_inv)
            else:
                U_inv = np.zeros((nvar[2], nvar[2]))
                for i in range(nvar[2]):
                    for j in range(nvar[2]):
                        U_inv[i, j] = bips.U(i, j)
                U_inv = np.linalg.inv(U_inv)
                U_inv = sp.COO(U_inv)

        #################

        sparse_arrays_full_dict = dict()
        # theta_a part
        for i in range(nvar[1]):
            sparse_arrays_full_dict[self._theta_a(i)] = list()

            if par.T4LSBpa is not None:
                val = sp.tensordot(a_theta[i], aips._z, axes=1)
                if val.nnz > 0:
                    sparse_arrays_full_dict[self._theta_a(i)].append(self._shift_tensor_coordinates(par.T4LSBpa * val, self._theta_a(0)))

            if ocean:
                if par.T4LSBpgo is not None:
                    val = sp.tensordot(a_theta[i], aips._v, axes=1)
                    if val.nnz > 0:
                        sparse_arrays_full_dict[self._theta_a(i)].append(self._shift_tensor_coordinates(- par.T4LSBpgo * val, self._deltaT_o(0)))

            if ground_temp:
                if par.T4LSBpgo is not None:
                    val = sp.tensordot(a_theta[i], aips._v, axes=1)
                    if val.nnz > 0:
                        sparse_arrays_full_dict[self._theta_a(i)].append(self._shift_tensor_coordinates(- par.T4LSBpgo * val, self._deltaT_g(0)))

        if ocean:
            # deltaT_o part
            for i in range(nvar[3]):
                sparse_arrays_full_dict[self._deltaT_o(i)] = list()
                val = sp.tensordot(U_inv[i], bips._Z, axes=1)
                if val.nnz > 0:
                    sparse_arrays_full_dict[self._deltaT_o(i)].append(self._shift_tensor_coordinates(par.T4sbpa * val, self._theta_a(0)))
                val = sp.tensordot(U_inv[i], bips._V, axes=1)
                if val.nnz > 0:
                    sparse_arrays_full_dict[self._deltaT_o(i)].append(self._shift_tensor_coordinates(- par.T4sbpgo * val, self._deltaT_o(0)))

        if ground_temp:
            # deltaT_g part
            for i in range(nvar[2]):
                sparse_arrays_full_dict[self._deltaT_g(i)] = list()
                val = sp.tensordot(U_inv[i], bips._Z, axes=1)
                if val.nnz > 0:
                    sparse_arrays_full_dict[self._deltaT_g(i)].append(self._shift_tensor_coordinates(par.T4sbpa * val, self._theta_a(0)))
                val = sp.tensordot(U_inv[i], bips._V, axes=1)
                if val.nnz > 0:
                    sparse_arrays_full_dict[self._deltaT_g(i)].append(self._shift_tensor_coordinates(- par.T4sbpgo * val, self._deltaT_g(0)))

        return sparse_arrays_full_dict

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
                    a_theta[i, j] = ap.sig0 * aips.a(i, j) - aips.u(i, j)
            a_theta = np.linalg.inv(a_theta)
            a_theta = sp.COO(a_theta)

        if bips is not None:
            if ocean:
                U_inv = np.zeros((nvar[3], nvar[3]))
                for i in range(nvar[3]):
                    for j in range(nvar[3]):
                        U_inv[i, j] = bips.U(i, j)
                U_inv = np.linalg.inv(U_inv)
                U_inv = sp.COO(U_inv)
            else:
                U_inv = np.zeros((nvar[2], nvar[2]))
                for i in range(nvar[2]):
                    for j in range(nvar[2]):
                        U_inv[i, j] = bips.U(i, j)
                U_inv = np.linalg.inv(U_inv)
                U_inv = sp.COO(U_inv)

        #################

        sparse_arrays_full_dict = dict()
        # theta_a part
        for i in range(nvar[1]):
            t_full = sp.zeros((ndim + 1, ndim + 1, ndim + 1, ndim + 1), dtype=np.float64, format='dok')

            if par.T4LSBpa is not None:
                j = k = ell = 0
                for m in range(nvar[1]):
                    val = 0
                    for jj in range(nvar[1]):
                        val += a_theta[i, jj] * aips.z(jj, j, k, ell, m)
                    if m == 0:
                        t_full[self._theta_a(j), self._theta_a(k), self._theta_a(ell), self._theta_a(m)] = par.T4LSBpa * val
                    else:
                        t_full[self._theta_a(j), self._theta_a(k), self._theta_a(ell), self._theta_a(m)] = 4 * par.T4LSBpa * val

            if ocean:
                if par.T4LSBpgo is not None:
                    j = k = ell = 0
                    for m in range(nvar[3]):
                        val = 0
                        for jj in range(nvar[1]):
                            val -= a_theta[i, jj] * aips.v(jj, j, k, ell, m)
                        if m == 0:
                            t_full[self._deltaT_o(j), self._deltaT_o(k), self._deltaT_o(ell), self._deltaT_o(m)] = par.T4LSBpgo * val
                        else:
                            t_full[self._deltaT_o(j), self._deltaT_o(k), self._deltaT_o(ell), self._deltaT_o(m)] = 4 * par.T4LSBpgo * val

            if ground_temp:
                if par.T4LSBpgo is not None:
                    j = k = ell = 0
                    for m in range(nvar[2]):
                        val = 0
                        for jj in range(nvar[1]):
                            val -= a_theta[i, jj] * aips.v(jj, j, k, ell, m)
                        if m == 0:
                            t_full[self._deltaT_g(j), self._deltaT_g(k), self._deltaT_g(ell), self._deltaT_g(m)] = par.T4LSBpgo * val
                        else:
                            t_full[self._deltaT_g(j), self._deltaT_g(k), self._deltaT_g(ell), self._deltaT_g(m)] = 4 * par.T4LSBpgo * val

            sparse_arrays_full_dict[self._theta_a(i)] = t_full.to_coo()

        if ocean:

            # deltaT_o part
            for i in range(nvar[3]):

                t_full = sp.zeros((ndim + 1, ndim + 1, ndim + 1, ndim + 1), dtype=np.float64, format='dok')

                j = k = ell = 0
                for m in range(nvar[1]):
                    val = 0
                    for jj in range(nvar[3]):
                        val += U_inv[i, jj] * bips.Z(jj, j, k, ell, m)
                    if m == 0:
                        t_full[self._theta_a(j), self._theta_a(k), self._theta_a(ell), self._theta_a(m)] = par.T4sbpa * val
                    else:
                        t_full[self._theta_a(j), self._theta_a(k), self._theta_a(ell), self._theta_a(m)] = 4 * par.T4sbpa * val

                for m in range(nvar[3]):
                    val = 0
                    for jj in range(nvar[3]):
                        val -= U_inv[i, jj] * bips.V(jj, j, k, ell, m)
                    if m == 0:
                        t_full[self._deltaT_o(j), self._deltaT_o(k), self._deltaT_o(ell), self._deltaT_o(m)] = par.T4sbpgo * val
                    else:
                        t_full[self._deltaT_o(j), self._deltaT_o(k), self._deltaT_o(ell), self._deltaT_o(m)] = 4 * par.T4sbpgo * val

                sparse_arrays_full_dict[self._deltaT_o(i)] = t_full.to_coo()

        # deltaT_g part
        if ground_temp:
            for i in range(nvar[2]):

                t_full = sp.zeros((ndim + 1, ndim + 1, ndim + 1, ndim + 1), dtype=np.float64, format='dok')

                j = k = ell = 0
                for m in range(nvar[1]):
                    val = 0
                    for jj in range(nvar[2]):
                        val += U_inv[i, jj] * bips._Z[jj, j, k, ell, m]
                    if m == 0:
                        t_full[self._theta_a(j), self._theta_a(k), self._theta_a(ell), self._theta_a(m)] = par.T4sbpa * val
                    else:
                        t_full[self._theta_a(j), self._theta_a(k), self._theta_a(ell), self._theta_a(m)] = 4 * par.T4sbpa * val

                for m in range(nvar[2]):
                    val = 0
                    for jj in range(nvar[2]):
                        val -= U_inv[i, jj] * bips._V[jj, j, k, ell, m]
                    if m == 0:
                        t_full[self._deltaT_g(j), self._deltaT_g(k), self._deltaT_g(ell), self._deltaT_g(m)] = par.T4sbpgo * val
                    else:
                        t_full[self._deltaT_g(j), self._deltaT_g(k), self._deltaT_g(ell), self._deltaT_g(m)] = 4 * par.T4sbpgo * val

                sparse_arrays_full_dict[self._deltaT_g(i)] = t_full.to_coo()

        return sparse_arrays_full_dict

    def compute_tensor(self):
        """Routine to compute the tensor."""
        # gathering
        par = self.params
        ndim = par.ndim

        sparse_arrays_dict = QgsTensor._compute_tensor_dicts(self)
        sparse_arrays_full_dict = self._compute_tensor_dicts()

        tensor = sp.zeros((ndim + 1, ndim + 1, ndim + 1, ndim + 1, ndim + 1), dtype=np.float64, format='coo')
        if sparse_arrays_dict is not None:
            tensor = self._add_dict_to_tensor(sparse_arrays_dict, tensor)
        if sparse_arrays_full_dict is not None:
            tensor = self._add_dict_to_tensor(sparse_arrays_full_dict, tensor)
        self._set_tensor(tensor)


class QgsTensorT4(QgsTensorDynamicT):
    """qgs :math:`T^4` tendencies tensor class. Implies dynamical zeroth-order temperature.

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
    tensor: sparse.COO(float)
        The tensor :math:`\\mathcal{T}_{i,j,k}` :math:`i`-th components.
    jacobian_tensor: sparse.COO(float)
        The jacobian tensor :math:`\\mathcal{T}_{i,j,k} + \\mathcal{T}_{i,k,j}` :math:`i`-th components.
    """

    def __init__(self, params=None, atmospheric_inner_products=None, oceanic_inner_products=None, ground_inner_products=None):

        QgsTensorDynamicT.__init__(self, params, atmospheric_inner_products, oceanic_inner_products, ground_inner_products)

    def _compute_non_stored_full_dict(self):

        if self.params is None:
            return None

        if self.atmospheric_inner_products is None and self.oceanic_inner_products is None \
                and self.ground_inner_products is None:
            return None

        aips = self.atmospheric_inner_products
        par = self.params
        ap = par.atmospheric_params
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

        # constructing some derived matrices
        if aips is not None:

            a_theta = np.zeros((nvar[1], nvar[1]))
            for i in range(nvar[1]):
                for j in range(nvar[1]):
                    a_theta[i, j] = ap.sig0 * aips.a(i, j) - aips.u(i, j)
            a_theta = np.linalg.inv(a_theta)
            a_theta = sp.COO(a_theta)

        if bips is not None:
            if ocean:
                U_inv = np.zeros((nvar[3], nvar[3]))
                for i in range(nvar[3]):
                    for j in range(nvar[3]):
                        U_inv[i, j] = bips.U(i, j)
                U_inv = np.linalg.inv(U_inv)
                U_inv = sp.COO(U_inv)
            else:
                U_inv = np.zeros((nvar[2], nvar[2]))
                for i in range(nvar[2]):
                    for j in range(nvar[2]):
                        U_inv[i, j] = bips.U(i, j)
                U_inv = np.linalg.inv(U_inv)
                U_inv = sp.COO(U_inv)

        #################

        sparse_arrays_full_dict = dict()

        # theta_a part
        for i in range(nvar[1]):
            t_full = sp.zeros((ndim + 1, ndim + 1, ndim + 1, ndim + 1), dtype=np.float64, format='dok')

            if par.T4LSBpa is not None:
                for j in range(nvar[1]):
                    for k in range(nvar[1]):
                        for ell in range(nvar[1]):
                            for m in range(nvar[1]):
                                val = 0
                                for jj in range(nvar[1]):
                                    val += a_theta[i, jj] * aips.z(jj, j, k, ell, m)
                                t_full[self._theta_a(j), self._theta_a(k), self._theta_a(ell), self._theta_a(m)] = par.T4LSBpa * val

            if ocean:
                if par.T4LSBpgo is not None:
                    for j in range(nvar[3]):
                        for k in range(nvar[3]):
                            for ell in range(nvar[3]):
                                for m in range(nvar[3]):
                                    val = 0
                                    for jj in range(nvar[1]):
                                        val -= a_theta[i, jj] * aips.v(jj, j, k, ell, m)
                                    t_full[self._deltaT_o(j), self._deltaT_o(k), self._deltaT_o(ell), self._deltaT_o(m)] = par.T4LSBpgo * val

            if ground_temp:
                if par.T4LSBpgo is not None:
                    for j in range(nvar[2]):
                        for k in range(nvar[2]):
                            for ell in range(nvar[2]):
                                for m in range(nvar[2]):
                                    val = 0
                                    for jj in range(nvar[1]):
                                        val -= a_theta[i, jj] * aips.v(jj, j, k, ell, m)
                                    t_full[self._deltaT_g(j), self._deltaT_g(k), self._deltaT_g(ell), self._deltaT_g(m)] = par.T4LSBpgo * val

            sparse_arrays_full_dict[self._theta_a(i)] = t_full.to_coo()

        if ocean:

            # deltaT_o part
            for i in range(nvar[3]):

                t_full = sp.zeros((ndim + 1, ndim + 1, ndim + 1, ndim + 1), dtype=np.float64, format='dok')

                for j in range(nvar[1]):
                    for k in range(nvar[1]):
                        for ell in range(nvar[1]):
                            for m in range(nvar[1]):
                                val = 0
                                for jj in range(nvar[3]):
                                    val += U_inv[i, jj] * bips.Z(jj, j, k, ell, m)
                                t_full[self._theta_a(j), self._theta_a(k), self._theta_a(ell), self._theta_a(m)] = par.T4sbpa * val

                for j in range(nvar[3]):
                    for k in range(nvar[3]):
                        for ell in range(nvar[3]):
                            for m in range(nvar[3]):
                                val = 0
                                for jj in range(nvar[3]):
                                    val -= U_inv[i, jj] * bips.V(jj, j, k, ell, m)
                                t_full[self._deltaT_o(j), self._deltaT_o(k), self._deltaT_o(ell), self._deltaT_o(m)] = par.T4sbpgo * val

                sparse_arrays_full_dict[self._deltaT_o(i)] = t_full.to_coo()

        # deltaT_g part
        if ground_temp:
            for i in range(nvar[2]):

                t_full = sp.zeros((ndim + 1, ndim + 1, ndim + 1, ndim + 1), dtype=np.float64, format='dok')

                for j in range(nvar[1]):
                    for k in range(nvar[1]):
                        for ell in range(nvar[1]):
                            for m in range(nvar[1]):
                                val = 0
                                for jj in range(nvar[2]):
                                    val += U_inv[i, jj] * bips._Z[jj, j, k, ell, m]
                                t_full[self._theta_a(j), self._theta_a(k), self._theta_a(ell), self._theta_a(m)] = par.T4sbpa * val

                for j in range(nvar[2]):
                    for k in range(nvar[2]):
                        for ell in range(nvar[2]):
                            for m in range(nvar[2]):
                                val = 0
                                for jj in range(nvar[2]):
                                    val -= U_inv[i, jj] * bips._V[jj, j, k, ell, m]
                                t_full[self._deltaT_g(j), self._deltaT_g(k), self._deltaT_g(ell), self._deltaT_g(m)] = par.T4sbpgo * val

                sparse_arrays_full_dict[self._deltaT_g(i)] = t_full.to_coo()

        return sparse_arrays_full_dict


def _kronecker_delta(i, j):

    if i == j:
        return 1

    else:
        return 0


if __name__ == '__main__':
    from qgs.params.params import QgParams
    from qgs.inner_products.analytic import AtmosphericAnalyticInnerProducts, OceanicAnalyticInnerProducts
    from qgs.inner_products.symbolic import AtmosphericSymbolicInnerProducts, OceanicSymbolicInnerProducts

    # Analytic test

    params = QgParams({'rr': 287.e0, 'sb': 5.6e-8})
    params.set_params({'kd': 0.04, 'kdp': 0.04, 'n': 1.5})
    params.set_atmospheric_channel_fourier_modes(2, 2)
    params.set_oceanic_basin_fourier_modes(2, 4)
    aip = AtmosphericAnalyticInnerProducts(params)
    oip = OceanicAnalyticInnerProducts(params)
    aip.connect_to_ocean(oip)
    agotensor = QgsTensor(params, aip, oip)

    # Symbolic dynamic T test

    params_t = QgParams({'rr': 287.e0, 'sb': 5.6e-8}, dynamic_T=True)
    params_t.set_params({'kd': 0.04, 'kdp': 0.04, 'n': 1.5})
    params_t.set_atmospheric_channel_fourier_modes(2, 2, mode='symbolic')
    params_t.set_oceanic_basin_fourier_modes(2, 4, mode='symbolic')

    aip = AtmosphericSymbolicInnerProducts(params_t, quadrature=True)  # , stored=False)
    oip = OceanicSymbolicInnerProducts(params_t, quadrature=True)  # , stored=False)
    agotensor_t = QgsTensorDynamicT(params_t, aip, oip)

    # Symbolic dynamic T4 test

    params_t4 = QgParams({'rr': 287.e0, 'sb': 5.6e-8}, T4=True)
    params_t4.set_params({'kd': 0.04, 'kdp': 0.04, 'n': 1.5})
    params_t4.set_atmospheric_channel_fourier_modes(2, 2, mode='symbolic')
    params_t4.set_oceanic_basin_fourier_modes(2, 4, mode='symbolic')

    aip = AtmosphericSymbolicInnerProducts(params_t4, quadrature=True)  # , stored=False)
    oip = OceanicSymbolicInnerProducts(params_t4, quadrature=True)  # , stored=False)

    aip.save_to_file("aip.ip")
    oip.save_to_file("oip.ip")

    aip.load_from_file("aip.ip")
    oip.load_from_file("oip.ip")

    agotensor_t4 = QgsTensorT4(params_t4, aip, oip)
