"""
    qgs tensor module
    =================

    This module computes and holds the tensor representing the tendencies of the model's equations.

    Notes
    -----

    These are computed using the analytical expressions from:

    * De Cruz, L., Demaeyer, J. and Vannitsem, S.: *The Modular Arbitrary-Order Ocean-Atmosphere Model: MAOOAM v1.0*,
      Geosci. Model Dev., **9**, 2793-2808, `doi:10.5194/gmd-9-2793-2016 <http://dx.doi.org/10.5194/gmd-9-2793-2016>`_, 2016.
    * Cehelsky, P., & Tung, K. K. (1987). *Theories of multiple equilibria and weather regimes—A critical reexamination.
      Part II: Baroclinic two-layer models*. Journal of the atmospheric sciences, **44** (21), 3282-3303.
      `link <https://journals.ametsoc.org/doi/abs/10.1175/1520-0469(1987)044%3C3282%3ATOMEAW%3E2.0.CO%3B2>`_

"""
import numpy as np
import sparse as sp


class QgsTensor(object):
    """qgs tendencies tensor class.

    Parameters
    ----------
    atmospheric_innner_product: AtmosphericInnerProducts or None
        The inner products of the atmospheric basis functions on which the model's PDE atmospheric equations are projected.
        If None, disable the atmospheric tendencies.
    oceanic_innner_product: OceanicInnerProducts or None
        The inner products of the atmospheric basis functions on which the model's PDE oceanic equations are projected.
        If None, disable the oceanic tendencies.

    Attributes
    ----------
    atmospheric_innner_product: AtmosphericInnerProducts or None
        The inner products of the atmospheric basis functions on which the model's PDE equations are projected.
        If None, the atmospheric tendencies are disabled.
    oceanic_innner_product: OceanicInnerProducts or None
        The inner products of the atmospheric basis functions on which the model's PDE equations are projected.
        If None, the oceanic tendencies are disabled.
    params: QgParams
        The models parameters.
    tensor: sparse.COO(float)
        The tensor :math:`\mathcal{T}_{i,j,k}` :math:`i`-th components.
    jacobian_tensor: sparse.COO(float)
        The jacobian tensor :math:`\mathcal{T}_{i,j,k} + \mathcal{T}_{i,k,j}` :math:`i`-th components.
    """

    def __init__(self, atmospheric_inner_products=None, oceanic_inner_products=None):

        self.atmospheric_inner_products = atmospheric_inner_products
        self.oceanic_inner_products = oceanic_inner_products
        if self.atmospheric_inner_products is not None:
            self.params = self.atmospheric_inner_products.params
        if self.oceanic_inner_products is not None:
            self.params = self.oceanic_inner_products.params

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
        aips = self.atmospheric_inner_products
        oips = self.oceanic_inner_products
        par = self.params
        atp = par.atemperature_params
        ap = par.atmospheric_params
        op = par.oceanic_params
        scp = par.scale_params
        gp = par.ground_params
        namod = par.nmod[0]
        ngomod = par.nmod[1]
        ndim = par.ndim

        if par.gotemperature_params is not None:
            ocean = par.gotemperature_params._name == "Oceanic Temperature"
            ground_temp = par.gotemperature_params._name == "Ground Temperature"
        else:
            ocean = False
            ground_temp = False

        # 0-th tensor component is an empty matrix
        tensor = sp.zeros((ndim+1, ndim + 1, ndim + 1), dtype=np.float64, format='dok')
        jacobian_tensor = sp.zeros((ndim+1, ndim + 1, ndim + 1), dtype=np.float64, format='dok')

        ## Problem with matmul with object and DOK archi : Temporary fix until a better solution is found
        hk = np.array(gp.hk, dtype=np.float)
        g = aips.g.to_coo()
        #################
        # psi_a part
        for i in range(1, namod + 1):
            t = np.zeros((ndim + 1, ndim + 1), dtype=np.float64)

            for j in range(1, namod + 1):

                t[self._psi_a(j), 0] = -((aips.c[(i - 1), (j - 1)] * scp.beta) / aips.a[(i - 1), (i - 1)]) \
                                       - (ap.kd * _kronecker_delta((i - 1), (j - 1))) / 2

                t[self._theta_a(j), 0] = (ap.kd * _kronecker_delta((i - 1), (j - 1))) / 2

                if gp.hk is not None:
                    oro = (g[(i - 1), (j - 1), :] @ hk) / (2 * aips.a[(i - 1), (i - 1)])
                    t[self._psi_a(j), 0] -= oro
                    t[self._theta_a(j), 0] += oro

                for k in range(1, namod + 1):
                    t[self._psi_a(j), self._psi_a(k)] = - aips.b[(i - 1), (j - 1), (k - 1)] \
                                                        / aips.a[(i - 1), (i - 1)]

                    t[self._theta_a(j), self._theta_a(k)] = - aips.b[(i - 1), (j - 1), (k - 1)] \
                                                            / aips.a[(i - 1), (i - 1)]
            if ocean:
                for j in range(1, ngomod + 1):
                    t[self._psi_o(j), 0] = ap.kd * aips.d[(i - 1), (j - 1)] / \
                                       (2 * aips.a[(i - 1), (i - 1)])

            t = self.simplify_matrix(t)
            tensor[self._psi_a(i)] = t
            jacobian_tensor[self._psi_a(i)] = t + t.T

        # theta_a part
        for i in range(1, namod + 1):
            t = np.zeros((ndim + 1, ndim + 1), dtype=np.float64)

            if par.Cpa is not None:
                t[0, 0] = par.Cpa[i - 1] / (1 - aips.a[0, 0] * ap.sig0)

            if atp.hd is not None and atp.thetas is not None:
                t[0, 0] += atp.hd * atp.thetas[(i - 1)] / (1. - ap.sig0 * aips.a[(i - 1), (i - 1)])

            for j in range(1, namod + 1):

                t[self._psi_a(j), 0] = (aips.a[(i - 1), (j - 1)] * ap.kd * ap.sig0) \
                                       / (-2 + 2 * aips.a[(i - 1), (i - 1)] * ap.sig0)

                if par.LSBpa is not None and par.Lpa is not None:
                    heat = 2. * (par.LSBpa + atp.sc * par.Lpa) * _kronecker_delta((i - 1), (j - 1))
                else:
                    heat = 0

                t[self._theta_a(j), 0] = (-((ap.sig0 * (2. * aips.c[(i - 1), (j - 1)]
                                                        * scp.beta + aips.a[(i - 1), (j - 1)] * (ap.kd + 4. * ap.kdp))))
                                          + heat) / (-2. + 2. * aips.a[(i - 1), (i - 1)] * ap.sig0)

                if atp.hd is not None:
                   t[self._theta_a(j), 0] += (atp.hd * _kronecker_delta((i - 1), (j - 1))) / (ap.sig0 * aips.a[(i - 1), (i - 1)] - 1.)

                if gp.hk is not None:
                    oro = (ap.sig0 * g[(i - 1), (j - 1), :] @ hk) / (2 * aips.a[(i - 1), (i - 1)] * ap.sig0 - 2.)
                    t[self._theta_a(j), 0] -= oro
                    t[self._psi_a(j), 0] += oro

                for k in range(1, namod + 1):
                    t[self._psi_a(j), self._theta_a(k)] = (aips.g[(i - 1), (j - 1), (k - 1)]
                                                           - aips.b[(i - 1), (j - 1), (k - 1)] * ap.sig0) / \
                                                          (-1 + aips.a[(i - 1), (i - 1)] * ap.sig0)

                    t[self._theta_a(j), self._psi_a(k)] = (aips.b[(i - 1), (j - 1), (k - 1)] * ap.sig0) \
                                                          / (1 - aips.a[(i - 1), (i - 1)] * ap.sig0)

            if ocean:
                for j in range(1, ngomod + 1):
                    t[self._psi_o(j), 0] = ap.kd * (aips.d[(i - 1), (j - 1)] * ap.sig0) \
                                       / (2 - 2 * aips.a[(i - 1), (i - 1)] * ap.sig0)

                    if par.LSBpgo is not None and par.Lpa is not None:
                        t[self._deltaT_o(j), 0] = aips.s[(i - 1), (j - 1)] * (2 * par.LSBpgo + par.Lpa) \
                                              / (2 - 2 * aips.a[(i - 1), (i - 1)] * ap.sig0)

            if ground_temp and i <= ngomod:
                t[self._deltaT_g(i), 0] = (2 * par.LSBpgo + par.Lpa) / (2 - 2 * aips.a[(i - 1), (i - 1)] * ap.sig0)

            t = self.simplify_matrix(t)
            tensor[self._theta_a(i)] = t
            jacobian_tensor[self._theta_a(i)] = t + t.T

        if ocean:
            # psi_o part
            for i in range(1, ngomod + 1):

                t = np.zeros((ndim + 1, ndim + 1), dtype=np.float64)

                for j in range(1, namod + 1):
                    t[self._psi_a(j), 0] = oips.K[(i - 1), (j - 1)] * op.d \
                                           / (oips.M[(i - 1), (i - 1)] + par.G)

                    t[self._theta_a(j), 0] = -(oips.K[(i - 1), (j - 1)]) * op.d \
                                             / (oips.M[(i - 1), (i - 1)] + par.G)

                for j in range(1, ngomod + 1):

                    t[self._psi_o(j), 0] = -((oips.N[(i - 1), (j - 1)] * scp.beta +
                                              oips.M[(i - 1), (i - 1)] * (op.r + op.d) *
                                              _kronecker_delta((i - 1), (j - 1)))) / (oips.M[(i - 1), (i - 1)] + par.G)

                    for k in range(1, ngomod + 1):
                        t[self._psi_o(j), self._psi_o(k)] = -(oips.C[(i - 1), (j - 1), (k - 1)]) \
                                                            / (oips.M[(i - 1), (i - 1)] + par.G)

                t = self.simplify_matrix(t)
                tensor[self._psi_o(i)] = t
                jacobian_tensor[self._psi_o(i)] = t + t.T

            # deltaT_o part
            ## Problem with matmul with object and DOK archi : Temporary fix until a better solution is found
            Cpgo = np.array(par.Cpgo, dtype=np.float)
            W = oips.W.to_coo()
            #################
            for i in range(1, ngomod + 1):

                t = np.zeros((ndim + 1, ndim + 1), dtype=np.float64)

                t[0, 0] = W[(i - 1), :] @ Cpgo

                for j in range(1, namod + 1):
                    t[self._theta_a(j), 0] = oips.W[(i - 1), (j - 1)] * (2 * atp.sc * par.Lpgo + par.sbpa)

                for j in range(1, ngomod + 1):

                    t[self._deltaT_o(j), 0] = - (par.Lpgo + par.sbpgo) * _kronecker_delta((i - 1), (j - 1))

                    for k in range(1, ngomod + 1):
                        t[self._psi_o(j), self._deltaT_o(k)] = -(oips.O[(i - 1), (j - 1), (k - 1)])

                t = self.simplify_matrix(t)
                tensor[self._deltaT_o(i)] = t
                jacobian_tensor[self._deltaT_o(i)] = t + t.T

        # deltaT_g part
        if ground_temp:
            for i in range(1, ngomod + 1):

                t = np.zeros((ndim + 1, ndim + 1), dtype=np.float64)

                t[0, 0] = par.Cpgo[(i - 1)]

                t[self._theta_a(i), 0] = 2 * atp.sc * par.Lpgo + par.sbpa

                t[self._deltaT_g(i), 0] = - (par.Lpgo + par.sbpgo)

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


def _kronecker_delta(i, j):

    if i == j:
        return 1

    else:
        return 0


if __name__ == '__main__':
    from params.params import QgParams
    from inner_products.analytic import AtmosphericInnerProducts, OceanicInnerProducts

    params = QgParams()
    params.set_atmospheric_modes(2, 2)
    params.set_oceanic_modes(2, 4)
    aip = AtmosphericInnerProducts(params)
    oip = OceanicInnerProducts(params)
    aip.connect_to_ocean(oip)
    agotensor = QgsTensor(aip, oip)