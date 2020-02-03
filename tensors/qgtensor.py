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
from scipy.sparse import csr_matrix


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
    tensor: list(~scipy.sparse.csr_matrix)
        List of the tensor :math:`\mathcal{T}_{i,j,k}` :math:`i`-th components, as a list of :mod:`scipy.sparse` matrices.
    jacobian_tensor: list(~scipy.sparse.csr_matrix)
        List of the jacobian tensor :math:`\mathcal{T}_{i,j,k} + \mathcal{T}_{i,k,j}` :math:`i`-th components,
        as a list of :mod:`scipy.sparse` matrices.
    """

    def __init__(self, atmospheric_inner_products=None, oceanic_inner_products=None):

        self.atmospheric_inner_products = atmospheric_inner_products
        self.oceanic_inner_products = oceanic_inner_products
        if self.atmospheric_inner_products is not None:
            self.params = self.atmospheric_inner_products.params
        if self.oceanic_inner_products is not None:
            self.params = self.oceanic_inner_products.params

        self.tensor = list()
        self.jacobian_tensor = list()

        self.compute_tensor()

    def psi_a(self, i):
        """Transform the :math:`\psi_{\mathrm a}` :math:`i`-th coefficient into the effective model's coordinate.

        Parameters
        ----------
        i: int
            The :math:`i`-th coefficients of :math:`\psi_{\mathrm a}`

        Returns
        -------
        int
            The effective model's coordinate.
        """
        return i

    def theta(self, i):
        """Transform the :math:`\theta_{\mathrm a}` :math:`i`-th coefficient into the effective model's coordinate.

        Parameters
        ----------
        i: int
            The :math:`i`-th coefficients of :math:`\theta_{\mathrm a}`

        Returns
        -------
        int
            The effective model's coordinate.
        """
        return i + self.params.nmod[0]

    def psi_o(self, i):
        """Transform the :math:`\psi_{\mathrm o}` :math:`i`-th coefficient into the effective model's coordinate.

        Parameters
        ----------
        i: int
            The :math:`i`-th coefficients of :math:`\psi_{\mathrm o}`

        Returns
        -------
        int
            The effective model's coordinate.
        """
        return i + 2 * self.params.nmod[0]

    def deltaT_o(self, i):
        """Transform the :math:`\delta T_{\mathrm o}` :math:`i`-th coefficient into the effective model's coordinate.

        Parameters
        ----------
        i: int
            The :math:`i`-th coefficients of :math:`\delta T_{\mathrm o}`

        Returns
        -------
        int
            The effective model's coordinate.
        """
        return i + 2 * self.params.nmod[0] + self.params.nmod[1]

    def compute_tensor(self):
        """Routine to compute the tensor."""
        aips = self.atmospheric_inner_products
        oips = self.oceanic_inner_products
        par = self.params
        atp = par.atemperature_params
        ap = par.atmospheric_params
        op = par.oceanic_params
        scp = par.scale_params
        namod = par.nmod[0]
        nomod = par.nmod[1]
        ndim = par.ndim

        # 0-th tensor component is an empty csr matrix
        t = np.zeros((ndim + 1, ndim + 1), dtype=np.float64)
        t = self.simplify_matrix(t)
        X = csr_matrix(t)
        self.tensor.append(X)

        X = csr_matrix(t + t.T)
        self.jacobian_tensor.append(X)

        # psi_a part
        for i in range(1, namod + 1):
            t = np.zeros((ndim + 1, ndim + 1), dtype=np.float64)

            for j in range(1, namod + 1):

                t[self.psi_a(j), 0] = -((aips.c[(i - 1), (j - 1)] * scp.beta) / aips.a[(i - 1), (i - 1)]) \
                                      - (ap.kd * _kronecker_delta((i - 1), (j - 1))) / 2

                t[self.theta(j), 0] = (ap.kd * _kronecker_delta((i - 1), (j - 1))) / 2

                if np.any(scp.hk != 0):
                    oro = (aips.g[(i - 1), (j - 1), :] @ scp.hk) / (2 * aips.a[(i - 1), (i - 1)])
                    t[self.psi_a(j), 0] -= oro
                    t[self.theta(j), 0] += oro

                for k in range(1, namod + 1):
                    t[self.psi_a(j), self.psi_a(k)] = - aips.b[(i - 1), (j - 1), (k - 1)] \
                                                      / aips.a[(i - 1), (i - 1)]

                    t[self.theta(j), self.theta(k)] = - aips.b[(i - 1), (j - 1), (k - 1)] \
                                                      / aips.a[(i - 1), (i - 1)]

            for j in range(1, nomod + 1):
                t[self.psi_o(j), 0] = ap.kd * aips.d[(i - 1), (j - 1)] / \
                                      (2 * aips.a[(i - 1), (i - 1)])

            t = self.simplify_matrix(t)
            X = csr_matrix(t)
            self.tensor.append(X)

            X = csr_matrix(t + t.T)
            self.jacobian_tensor.append(X)

        # theta_a part
        for i in range(1, namod + 1):
            t = np.zeros((ndim + 1, ndim + 1), dtype=np.float64)

            if i == 1 and par.Cpa is not None:
                t[0, 0] = par.Cpa / (1 - aips.a[0, 0] * ap.sig0)

            t[0, 0] -= atp.hpp * atp.thetas[(i - 1)] / (ap.sig0 * aips.a[(i - 1), (i - 1)])
            for j in range(1, namod + 1):

                t[self.psi_a(j), 0] = (aips.a[(i - 1), (j - 1)] * ap.kd * ap.sig0) \
                                      / (-2 + 2 * aips.a[(i - 1), (i - 1)] * ap.sig0)

                if par.LSBpa is not None and par.Lpa is not None:
                    heat = 2. * (par.LSBpa + atp.sc * par.Lpa) * _kronecker_delta((i - 1), (j - 1))
                else:
                    heat = 0

                t[self.theta(j), 0] = (-((ap.sig0 * (2. * aips.c[(i - 1), (j - 1)]
                                                     * scp.beta + aips.a[(i - 1), (j - 1)] * (ap.kd + 4. * ap.kdp))))
                                       + heat) / (-2. + 2. * aips.a[(i - 1), (i - 1)] * ap.sig0) \
                                      + (atp.hpp * _kronecker_delta((i - 1), (j - 1))) / (
                                              ap.sig0 * aips.a[(i - 1), (i - 1)])

                if np.any(scp.hk != 0):
                    oro = (aips.g[(i - 1), (j - 1), :] @ scp.hk) / (2 * aips.a[(i - 1), (i - 1)])
                    t[self.theta(j), 0] -= oro
                    t[self.psi_a(j), 0] += oro

                for k in range(1, namod + 1):
                    t[self.psi_a(j), self.theta(k)] = (aips.g[(i - 1), (j - 1), (k - 1)]
                                                       - aips.b[(i - 1), (j - 1), (k - 1)] * ap.sig0) / \
                                                      (-1 + aips.a[(i - 1), (i - 1)] * ap.sig0)

                    t[self.theta(j), self.psi_a(k)] = (aips.b[(i - 1), (j - 1), (k - 1)] * ap.sig0) \
                                                      / (1 - aips.a[(i - 1), (i - 1)] * ap.sig0)

            for j in range(1, nomod + 1):
                t[self.psi_o(j), 0] = ap.kd * (aips.d[(i - 1), (j - 1)] * ap.sig0) \
                                      / (2 - 2 * aips.a[(i - 1), (i - 1)] * ap.sig0)

                t[self.deltaT_o(j), 0] = aips.s[(i - 1), (j - 1)] * (2 * par.LSBpo + par.Lpa) \
                                         / (2 - 2 * aips.a[(i - 1), (i - 1)] * ap.sig0)

            t = self.simplify_matrix(t)
            X = csr_matrix(t)
            self.tensor.append(X)

            X = csr_matrix(t + t.T)
            self.jacobian_tensor.append(X)

        # psi_o part
        for i in range(1, nomod + 1):

            t = np.zeros((ndim + 1, ndim + 1), dtype=np.float64)

            for j in range(1, namod + 1):
                t[self.psi_a(j), 0] = oips.K[(i - 1), (j - 1)] * op.d \
                                      / (oips.M[(i - 1), (i - 1)] + par.G)

                t[self.theta(j), 0] = -(oips.K[(i - 1), (j - 1)]) * op.d \
                                      / (oips.M[(i - 1), (i - 1)] + par.G)

            for j in range(1, nomod + 1):

                t[self.psi_o(j), 0] = -((oips.N[(i - 1), (j - 1)] * scp.beta +
                                         oips.M[(i - 1), (i - 1)] * (op.r + op.d) *
                                         _kronecker_delta((i - 1), (j - 1)))) / (oips.M[(i - 1), (i - 1)] + par.G)

                for k in range(1, nomod + 1):
                    t[self.psi_o(j), self.psi_o(k)] = -(oips.C[(i - 1), (j - 1), (k - 1)]) \
                                                      / (oips.M[(i - 1), (i - 1)] + par.G)

            t = self.simplify_matrix(t)
            X = csr_matrix(t)
            self.tensor.append(X)

            X = csr_matrix(t + t.T)
            self.jacobian_tensor.append(X)

        # deltaT_o part
        for i in range(1, nomod + 1):

            t = np.zeros((ndim + 1, ndim + 1), dtype=np.float64)

            t[0, 0] = par.Cpo * oips.W[(i - 1), 0]

            for j in range(1, namod + 1):
                t[self.theta(j), 0] = oips.W[(i - 1), (j - 1)] * (2 * atp.sc * par.Lpo + par.sbpa)

            for j in range(1, nomod + 1):

                t[self.deltaT_o(j), 0] = - (par.Lpo + par.sbpo) * _kronecker_delta((i - 1), (j - 1))

                for k in range(1, nomod + 1):
                    t[self.psi_o(j), self.deltaT_o(k)] = -(oips.O[(i - 1), (j - 1), (k - 1)])

            t = self.simplify_matrix(t)
            X = csr_matrix(t)
            self.tensor.append(X)

            X = csr_matrix(t + t.T)
            self.jacobian_tensor.append(X)

    @staticmethod
    def simplify_matrix(matrix):
        """Routine that simplifies the component of the 3D tensors :math:`\mathcal{T}`.
        For each index :math:`i`, it upper-triangularizes the
        matrix :math:`\mathcal{T}_{i,j,k} \quad 0 \leq j,k \leq \mathrm{n\_dim}`.

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
    from tensors.cootensor import from_csr_mat_list

    params = QgParams()
    params.set_max_atmospheric_modes(2, 2)
    params.set_max_oceanic_modes(2, 4)
    params.scale_params.hk[1] = 0.2
    params.atemperature_params.thetas[0] = 0.1
    aip = AtmosphericInnerProducts(params)
    oip = OceanicInnerProducts(params)
    aip.connect_to_ocean(oip)
    aotensor = QgsTensor(aip, oip)
    coo_tensor = from_csr_mat_list(aotensor.tensor)