"""
    qgs tensor module
    =================

    This module computes and holds the tensor representing the tendencies of the model's equations.

"""
import numpy as np
import sparse as sp
import pickle


class QgsTensor(object):
    """qgs tendencies tensor class.

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
        The tensor :math:`\mathcal{T}_{i,j,k}` :math:`i`-th components.
    jacobian_tensor: sparse.COO(float)
        The jacobian tensor :math:`\mathcal{T}_{i,j,k} + \mathcal{T}_{i,k,j}` :math:`i`-th components.
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
        """Transform the :math:`\psi_{\mathrm a}` :math:`i`-th coefficient into the effective model's variable.

        Parameters
        ----------
        i: int
            The :math:`i`-th coefficients of :math:`\psi_{\mathrm a}`

        Returns
        -------
        int
            The effective model's variable.
        """
        return i

    def _theta_a(self, i):
        """Transform the :math:`\\theta_{\mathrm a}` :math:`i`-th coefficient into the effective model's variable.

        Parameters
        ----------
        i: int
            The :math:`i`-th coefficients of :math:`\\theta_{\mathrm a}`

        Returns
        -------
        int
            The effective model's variable.
        """
        return i + self.params.nmod[0]

    def _psi_o(self, i):
        """Transform the :math:`\psi_{\mathrm o}` :math:`i`-th coefficient into the effective model's variable.

        Parameters
        ----------
        i: int
            The :math:`i`-th coefficients of :math:`\psi_{\mathrm o}`

        Returns
        -------
        int
            The effective model's variable.
        """
        return i + 2 * self.params.nmod[0]

    def _deltaT_o(self, i):
        """Transform the :math:`\delta T_{\mathrm o}` :math:`i`-th coefficient into the effective model's variable.

        Parameters
        ----------
        i: int
            The :math:`i`-th coefficients of :math:`\delta T_{\mathrm o}`

        Returns
        -------
        int
            The effective model's variable.
        """
        return i + 2 * self.params.nmod[0] + self.params.nmod[1]

    def _deltaT_g(self, i):
        """Transform the :math:`\delta T_{\mathrm o}` :math:`i`-th coefficient into the effective model's variable.

        Parameters
        ----------
        i: int
            The :math:`i`-th coefficients of :math:`\delta T_{\mathrm o}`

        Returns
        -------
        int
            The effective model's variable.
        """
        return i + 2 * self.params.nmod[0]

    def compute_tensor(self):
        """Routine to compute the tensor."""

        if self.params is None:
            return

        if self.atmospheric_inner_products is None and self.oceanic_inner_products is None \
                and self.ground_inner_products is None:
            return

        aips = self.atmospheric_inner_products
        par = self.params
        atp = par.atemperature_params
        ap = par.atmospheric_params
        op = par.oceanic_params
        scp = par.scale_params
        gp = par.ground_params
        namod = par.nmod[0]
        ngomod = par.nmod[1]
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

        # 0-th tensor component is an empty matrix
        tensor = sp.zeros((ndim+1, ndim + 1, ndim + 1), dtype=np.float64, format='dok')
        jacobian_tensor = sp.zeros((ndim+1, ndim + 1, ndim + 1), dtype=np.float64, format='dok')

        # constructing some derived matrices
        if aips is not None:
            a_inv = np.zeros((namod, namod))
            for i in range(namod):
                for j in range(namod):
                    a_inv[i, j] = aips.a(i, j)
            a_inv = np.linalg.inv(a_inv)

            a_theta = np.zeros((namod, namod))
            for i in range(namod):
                for j in range(namod):
                    a_theta[i, j] = ap.sig0 * aips.a(i, j) - aips.u(i, j)
            a_theta = np.linalg.inv(a_theta)

        if bips is not None:
            U_inv = np.zeros((ngomod, ngomod))
            for i in range(ngomod):
                for j in range(ngomod):
                    U_inv[i, j] = bips.U(i, j)
            U_inv = np.linalg.inv(U_inv)

            if ocean:
                M_psio = np.zeros((ngomod, ngomod))
                for i in range(ngomod):
                    for j in range(ngomod):
                        M_psio[i, j] = bips.M(i, j) + par.G * bips.U(i, j)
                M_psio = np.linalg.inv(M_psio)

        #################

        if bips is not None:
            go = bips.stored
        else:
            go = True

        if aips.stored and go:
            # psi_a part
            for i in range(1, namod + 1):
                t = np.zeros((ndim + 1, ndim + 1), dtype=np.float64)

                for j in range(1, namod + 1):

                    val = a_inv[(i - 1), :] @ aips._c[:, (j - 1)]
                    t[self._psi_a(j), 0] -= val * scp.beta

                    t[self._psi_a(j), 0] -= (ap.kd * _kronecker_delta((i - 1), (j - 1))) / 2
                    t[self._theta_a(j), 0] = (ap.kd * _kronecker_delta((i - 1), (j - 1))) / 2

                    if gp is not None:
                        if gp.hk is not None:
                            if gp.orographic_basis == "atmospheric":
                                oro = a_inv[(i - 1), :] @ aips._g[:, (j - 1), :] @ gp.hk
                            else:
                                oro = a_inv[(i - 1), :] @ aips._gh[:, (j - 1), :] @ gp.hk
                            t[self._psi_a(j), 0] -= oro / 2
                            t[self._theta_a(j), 0] += oro / 2

                    for k in range(1, namod + 1):
                        val = a_inv[(i - 1), :] @ aips._b[:, (j - 1), (k - 1)]
                        t[self._psi_a(j), self._psi_a(k)] = - val
                        t[self._theta_a(j), self._theta_a(k)] = - val
                if ocean:
                    for j in range(1, ngomod + 1):
                        val = a_inv[(i - 1), :] @ aips._d[:, (j - 1)]
                        t[self._psi_o(j), 0] += val * ap.kd / 2

                t = self.simplify_matrix(t)
                tensor[self._psi_a(i)] = t
                jacobian_tensor[self._psi_a(i)] = t + t.T

            # theta_a part
            for i in range(1, namod + 1):
                t = np.zeros((ndim + 1, ndim + 1), dtype=np.float64)

                if par.Cpa is not None:
                    t[0, 0] -= a_theta[(i - 1), :] @ aips._u @ par.Cpa

                if atp.hd is not None and atp.thetas is not None:
                    val = - a_theta[(i - 1), :] @ aips._u @ atp.thetas
                    t[0, 0] += val * atp.hd

                for j in range(1, namod + 1):

                    val = a_theta[(i - 1), :] @ aips._a[:, (j - 1)]
                    t[self._psi_a(j), 0] += val * ap.kd * ap.sig0 / 2
                    t[self._theta_a(j), 0] -= val * (ap.kd / 2 + 2 * ap.kdp) * ap.sig0

                    val = - a_theta[(i - 1), :] @ aips._c[:, (j - 1)]
                    t[self._theta_a(j), 0] += val * scp.beta * ap.sig0

                    val = a_theta[(i - 1), :] @ aips._u[:, (j - 1)]
                    if par.LSBpa is not None and par.Lpa is not None:
                        t[self._theta_a(j), 0] += val * (par.LSBpa + atp.sc * par.Lpa)
                    if atp.hd is not None:
                        t[self._theta_a(j), 0] += val * atp.hd

                    if gp is not None:
                        if gp.hk is not None:
                            if gp.orographic_basis == "atmospheric":
                                oro = a_theta[(i - 1), :] @ aips._g[:, (j - 1), :] @ gp.hk
                            else:
                                oro = a_theta[(i - 1), :] @ aips._gh[:, (j - 1), :] @ gp.hk
                            t[self._theta_a(j), 0] -= ap.sig0 * oro / 2
                            t[self._psi_a(j), 0] += ap.sig0 * oro / 2

                    for k in range(1, namod + 1):
                        val = a_theta[(i - 1), :] @ aips._b[:, (j - 1), (k - 1)]
                        t[self._psi_a(j), self._theta_a(k)] = - val * ap.sig0
                        t[self._theta_a(j), self._psi_a(k)] = - val * ap.sig0

                        val = a_theta[(i - 1), :] @ aips._g[:, (j - 1), (k - 1)]
                        t[self._psi_a(j), self._theta_a(k)] += val

                if ocean:
                    for j in range(1, ngomod + 1):
                        val = - a_theta[(i - 1), :] @ aips._d[:, (j - 1)]
                        t[self._psi_o(j), 0] += val * ap.sig0 * ap.kd / 2

                        if par.LSBpgo is not None and par.Lpa is not None:
                            val = - a_theta[(i - 1), :] @ aips._s[:, (j - 1)]
                            t[self._deltaT_o(j), 0] += val * (par.LSBpgo + par.Lpa / 2)

                if ground_temp:
                    for j in range(1, ngomod + 1):
                        if par.LSBpgo is not None and par.Lpa is not None:
                            val = - a_theta[(i - 1), :] @ aips._s[:, (j - 1)]
                            t[self._deltaT_g(j), 0] += val * (par.LSBpgo + par.Lpa / 2)

                t = self.simplify_matrix(t)
                tensor[self._theta_a(i)] = t
                jacobian_tensor[self._theta_a(i)] = t + t.T

            if ocean:
                # psi_o part
                for i in range(1, ngomod + 1):

                    t = np.zeros((ndim + 1, ndim + 1), dtype=np.float64)

                    for j in range(1, namod + 1):
                        val = M_psio[(i - 1), :] @ bips._K[:, (j - 1)] * op.d
                        t[self._psi_a(j), 0] += val
                        t[self._theta_a(j), 0] -= val

                    for j in range(1, ngomod + 1):
                        val = - M_psio[(i - 1), :] @ bips._N[:, (j - 1)]
                        t[self._psi_o(j), 0] += val * scp.beta

                        val = - M_psio[(i - 1), :] @ bips._M[:, (j - 1)]
                        t[self._psi_o(j), 0] += val * (op.r + op.d)

                        for k in range(1, ngomod + 1):
                            t[self._psi_o(j), self._psi_o(k)] -= M_psio[(i - 1), :] @ bips._C[:, (j - 1), (k - 1)]

                    t = self.simplify_matrix(t)
                    tensor[self._psi_o(i)] = t
                    jacobian_tensor[self._psi_o(i)] = t + t.T

                # deltaT_o part
                for i in range(1, ngomod + 1):

                    t = np.zeros((ndim + 1, ndim + 1), dtype=np.float64)
                    t[0, 0] += bips._W[(i - 1), :] @ np.array(par.Cpgo, dtype=np.float64)  # problem with sparse matmul and object dtype

                    for j in range(1, namod + 1):
                        val = U_inv[(i - 1), :] @ bips._W[:, (j - 1)]
                        t[self._theta_a(j), 0] += val * (2 * atp.sc * par.Lpgo + par.sbpa)

                    for j in range(1, ngomod + 1):
                        t[self._deltaT_o(j), 0] = - (par.Lpgo + par.sbpgo) * _kronecker_delta((i - 1), (j - 1))

                        for k in range(1, ngomod + 1):
                            t[self._psi_o(j), self._deltaT_o(k)] -= U_inv[(i - 1), :] @ bips._O[:, (j - 1), (k - 1)]

                    t = self.simplify_matrix(t)
                    tensor[self._deltaT_o(i)] = t
                    jacobian_tensor[self._deltaT_o(i)] = t + t.T

            # deltaT_g part
            if ground_temp:
                for i in range(1, ngomod + 1):

                    t = np.zeros((ndim + 1, ndim + 1), dtype=np.float64)
                    t[0, 0] += bips._W[(i - 1), :] @ np.array(par.Cpgo, dtype=np.float64)

                    for j in range(1, namod + 1):
                        val = U_inv[(i - 1), :] @ bips._W[:, (j - 1)]
                        t[self._theta_a(j), 0] += val * (2 * atp.sc * par.Lpgo + par.sbpa)

                    for j in range(1, ngomod + 1):
                        t[self._deltaT_g(j), 0] = - (par.Lpgo + par.sbpgo) * _kronecker_delta((i - 1), (j - 1))

                    t = self.simplify_matrix(t)
                    tensor[self._deltaT_g(i)] = t
                    jacobian_tensor[self._deltaT_g(i)] = t + t.T

            self.tensor = tensor.to_coo()
            self.jacobian_tensor = jacobian_tensor.to_coo()
        else:
            # psi_a part
            for i in range(1, namod + 1):
                t = np.zeros((ndim + 1, ndim + 1), dtype=np.float64)

                for j in range(1, namod + 1):

                    val = 0
                    for jj in range(1, namod + 1):
                        val -= a_inv[(i - 1), (jj - 1)] * aips.c((jj - 1), (j - 1))
                    t[self._psi_a(j), 0] += val * scp.beta

                    t[self._psi_a(j), 0] -= (ap.kd * _kronecker_delta((i - 1), (j - 1))) / 2
                    t[self._theta_a(j), 0] = (ap.kd * _kronecker_delta((i - 1), (j - 1))) / 2

                    if gp is not None:
                        if gp.hk is not None:
                            oro = 0
                            if gp.orographic_basis == "atmospheric":
                                for jj in range(1, namod + 1):
                                    for k in range(1, namod + 1):
                                        oro += a_inv[(i - 1), (jj - 1)] * aips.g((jj - 1), (j - 1), (k - 1)) * gp.hk[(k - 1)]
                            else:
                                for jj in range(1, namod + 1):
                                    for k in range(1, ngomod + 1):
                                        oro += a_inv[(i - 1), (jj - 1)] * aips.gh((jj - 1), (j - 1), (k - 1)) * gp.hk[(k - 1)]
                            t[self._psi_a(j), 0] -= oro / 2
                            t[self._theta_a(j), 0] += oro / 2

                    for k in range(1, namod + 1):
                        val = 0
                        for jj in range(1, namod + 1):
                            val += a_inv[(i - 1), (jj - 1)] * aips.b((jj - 1), (j - 1), (k - 1))
                        t[self._psi_a(j), self._psi_a(k)] = - val
                        t[self._theta_a(j), self._theta_a(k)] = - val
                if ocean:
                    for j in range(1, ngomod + 1):
                        val = 0
                        for jj in range(1, namod + 1):
                            val += a_inv[(i - 1), (jj - 1)] * aips.d((jj - 1), (j - 1))
                        t[self._psi_o(j), 0] += val * ap.kd / 2

                t = self.simplify_matrix(t)
                tensor[self._psi_a(i)] = t
                jacobian_tensor[self._psi_a(i)] = t + t.T

            # theta_a part
            for i in range(1, namod + 1):
                t = np.zeros((ndim + 1, ndim + 1), dtype=np.float64)

                if par.Cpa is not None:
                    for jj in range(1, namod + 1):
                        for kk in range(1, namod + 1):
                            t[0, 0] -= a_theta[(i - 1), (jj - 1)] * aips.u((jj - 1), (kk - 1)) * par.Cpa[kk - 1]

                if atp.hd is not None and atp.thetas is not None:
                    val = 0
                    for jj in range(1, namod + 1):
                        for kk in range(1, namod + 1):
                            val -= a_theta[(i - 1), (jj - 1)] * aips.u((jj - 1), (kk - 1)) * atp.thetas[(kk - 1)]
                    t[0, 0] += val * atp.hd

                for j in range(1, namod + 1):

                    val = 0
                    for jj in range(1, namod + 1):
                        val += a_theta[(i - 1), (jj - 1)] * aips.a((jj - 1), (j - 1))
                    t[self._psi_a(j), 0] += val * ap.kd * ap.sig0 / 2

                    val = 0
                    for jj in range(1, namod + 1):
                        val -= a_theta[(i - 1), (jj - 1)] * aips.c((jj - 1), (j - 1))
                    t[self._theta_a(j), 0] += val * scp.beta * ap.sig0

                    val = 0
                    for jj in range(1, namod + 1):
                        val -= a_theta[(i - 1), (jj - 1)] * aips.a((jj - 1), (j - 1))
                    t[self._theta_a(j), 0] += val * (ap.kd / 2 + 2 * ap.kdp) * ap.sig0

                    if par.LSBpa is not None and par.Lpa is not None:
                        val = 0
                        for jj in range(1, namod + 1):
                            val += a_theta[(i - 1), (jj - 1)] * aips.u((jj - 1), (j - 1))
                        t[self._theta_a(j), 0] += val * (par.LSBpa + atp.sc * par.Lpa)

                    if atp.hd is not None:
                        val = 0
                        for jj in range(1, namod + 1):
                            val += a_theta[(i - 1), (jj - 1)] * aips.u((jj - 1), (j - 1))
                        t[self._theta_a(j), 0] += val * atp.hd

                    if gp is not None:
                        if gp.hk is not None:
                            oro = 0
                            if gp.orographic_basis == "atmospheric":
                                for jj in range(1, namod + 1):
                                    for k in range(1, namod + 1):
                                        oro += a_theta[(i - 1), (jj - 1)] * aips.g((jj - 1), (j - 1), (k - 1)) * gp.hk[(k - 1)]
                            else:
                                for jj in range(1, namod + 1):
                                    for k in range(1, ngomod + 1):
                                        oro += a_theta[(i - 1), (jj - 1)] * aips.gh((jj - 1), (j - 1), (k - 1)) * gp.hk[(k - 1)]
                            t[self._theta_a(j), 0] -= ap.sig0 * oro / 2
                            t[self._psi_a(j), 0] += ap.sig0 * oro / 2

                    for k in range(1, namod + 1):
                        val = 0
                        for jj in range(1, namod + 1):
                            val += a_theta[(i - 1), (jj - 1)] * aips.b((jj - 1), (j - 1), (k - 1))
                        t[self._psi_a(j), self._theta_a(k)] = - val * ap.sig0
                        t[self._theta_a(j), self._psi_a(k)] = - val * ap.sig0

                        val = 0
                        for jj in range(1, namod + 1):
                            val += a_theta[(i - 1), (jj - 1)] * aips.g((jj - 1), (j - 1), (k - 1))

                        t[self._psi_a(j), self._theta_a(k)] += val

                if ocean:
                    for j in range(1, ngomod + 1):
                        val = 0
                        for jj in range(1, namod + 1):
                            val -= a_theta[(i - 1), (jj - 1)] * aips.d((jj - 1), (j - 1))
                        t[self._psi_o(j), 0] += val * ap.sig0 * ap.kd / 2

                        if par.LSBpgo is not None and par.Lpa is not None:
                            val = 0
                            for jj in range(1, namod + 1):
                                val -= a_theta[(i - 1), (jj - 1)] * aips.s((jj - 1), (j - 1))
                            t[self._deltaT_o(j), 0] += val * (par.LSBpgo + par.Lpa / 2)

                if ground_temp:
                    for j in range(1, ngomod + 1):
                        if par.LSBpgo is not None and par.Lpa is not None:
                            val = 0
                            for jj in range(1, namod + 1):
                                val -= a_theta[(i - 1), (jj - 1)] * aips.s((jj - 1), (j - 1))
                            t[self._deltaT_g(j), 0] += val * (par.LSBpgo + par.Lpa / 2)

                t = self.simplify_matrix(t)
                tensor[self._theta_a(i)] = t
                jacobian_tensor[self._theta_a(i)] = t + t.T

            if ocean:
                # psi_o part
                for i in range(1, ngomod + 1):

                    t = np.zeros((ndim + 1, ndim + 1), dtype=np.float64)

                    for j in range(1, namod + 1):
                        for jj in range(1, ngomod + 1):
                            val = M_psio[(i - 1), (jj - 1)] * bips.K((jj - 1), (j - 1)) * op.d
                            t[self._psi_a(j), 0] += val
                            t[self._theta_a(j), 0] -= val

                    for j in range(1, ngomod + 1):
                        val = 0
                        for jj in range(1, ngomod + 1):
                            val -= M_psio[(i - 1), (jj - 1)] * bips.N((i - 1), (j - 1))
                        t[self._psi_o(j), 0] += val * scp.beta
                        val = 0
                        for jj in range(1, ngomod + 1):
                            val -= M_psio[(i - 1), (jj - 1)] * bips.M((i - 1), (j - 1))
                        t[self._psi_o(j), 0] += val * (op.r + op.d)

                        for k in range(1, ngomod + 1):
                            for jj in range(1, ngomod + 1):
                                t[self._psi_o(j), self._psi_o(k)] -= M_psio[(i - 1), (jj - 1)] * bips.C((jj - 1), (j - 1),
                                                                                                        (k - 1))

                    t = self.simplify_matrix(t)
                    tensor[self._psi_o(i)] = t
                    jacobian_tensor[self._psi_o(i)] = t + t.T

                # deltaT_o part
                for i in range(1, ngomod + 1):

                    t = np.zeros((ndim + 1, ndim + 1), dtype=np.float64)
                    for jj in range(1, namod + 1):
                        t[0, 0] += bips.W((i - 1), (jj - 1)) * par.Cpgo[(jj - 1)]

                    for j in range(1, namod + 1):
                        val = 0
                        for jj in range(1, ngomod + 1):
                            val += U_inv[(i - 1), (jj - 1)] * bips.W((jj - 1), (j - 1))
                        t[self._theta_a(j), 0] += val * (2 * atp.sc * par.Lpgo + par.sbpa)

                    for j in range(1, ngomod + 1):
                        t[self._deltaT_o(j), 0] = - (par.Lpgo + par.sbpgo) * _kronecker_delta((i - 1), (j - 1))

                        for k in range(1, ngomod + 1):
                            for jj in range(1, ngomod + 1):
                                t[self._psi_o(j), self._deltaT_o(k)] -= U_inv[(i - 1), (jj - 1)] * bips.O((jj - 1), (j - 1),
                                                                                                          (k - 1))

                    t = self.simplify_matrix(t)
                    tensor[self._deltaT_o(i)] = t
                    jacobian_tensor[self._deltaT_o(i)] = t + t.T

            # deltaT_g part
            if ground_temp:
                for i in range(1, ngomod + 1):

                    t = np.zeros((ndim + 1, ndim + 1), dtype=np.float64)
                    for jj in range(1, namod + 1):
                        t[0, 0] += bips.W[(i - 1), (jj - 1)] * par.Cpgo[(jj - 1)]

                    for j in range(1, namod + 1):
                        val = 0
                        for jj in range(1, ngomod + 1):
                            val += U_inv[(i - 1), (jj - 1)] * bips.W((jj - 1), (j - 1))
                        t[self._theta_a(j), 0] += val * (2 * atp.sc * par.Lpgo + par.sbpa)

                    for j in range(1, ngomod + 1):
                        t[self._deltaT_g(j), 0] = - (par.Lpgo + par.sbpgo) * _kronecker_delta((i - 1), (j - 1))

                    t = self.simplify_matrix(t)
                    tensor[self._deltaT_g(i)] = t
                    jacobian_tensor[self._deltaT_g(i)] = t + t.T

            self.tensor = tensor.to_coo()
            self.jacobian_tensor = jacobian_tensor.to_coo()

    @staticmethod
    def simplify_matrix(matrix):
        """Routine that simplifies the component of the 3D tensors :math:`\mathcal{T}`.
        For each index :math:`i`, it upper-triangularizes the
        matrix :math:`\mathcal{T}_{i,j,k} \quad 0 \leq j,k \leq \mathrm{ndim}`.

        Parameters
        ----------
        matrix: ~numpy.ndarray
            :math:`i`-th matrix component of the tensor :math:`\mathcal{T}_{i,j,k}` to simplify.

        Returns
        -------
        ~numpy.ndarray
            The upper-triangularized matrix.
        """
        return np.triu(matrix) + np.tril(matrix, -1).T

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


def _kronecker_delta(i, j):

    if i == j:
        return 1

    else:
        return 0


if __name__ == '__main__':
    from qgs.params.params import QgParams
    from qgs.inner_products.analytic import AtmosphericAnalyticInnerProducts, OceanicAnalyticInnerProducts

    params = QgParams()
    params.set_atmospheric_channel_fourier_modes(2, 2)
    params.set_oceanic_basin_fourier_modes(2, 4)
    aip = AtmosphericAnalyticInnerProducts(params)
    oip = OceanicAnalyticInnerProducts(params)
    aip.connect_to_ocean(oip)
    agotensor = QgsTensor(params, aip, oip)
