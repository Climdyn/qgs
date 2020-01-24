
import numpy as np
from scipy.sparse import csr_matrix


class QgsTensor(object):

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
        return i

    def theta(self, i):
        return i + self.params.nmod[0]

    def psi_o(self, i):
        return i + 2 * self.params.nmod[0]

    def deltaT_o(self, i):
        return i + 2 * self.params.nmod[0] + self.params.nmod[1]

    def compute_tensor(self):

        aips = self.atmospheric_inner_products
        oips = self.oceanic_inner_products
        par = self.params
        atp = par.atemperature_params
        ap = par.atmospheric_params
        scp = par.scale_params
        namod = par.nmod[0]
        nomod = par.nmod[1]
        ndim = par.ndim
        t = np.zeros((ndim+1, ndim + 1, ndim + 1), dtype=np.float64)  # could be initialized better

        if par.Cpa is not None:
            t[self.theta(1), 0, 0] = par.Cpa / (1 - aips.a[0, 0] * ap.sig0)

        for i in range(1, namod + 1):

            t[self.theta(i), 0, 0] -= atp.hpp * atp.thetas[(i - 1)] / (ap.sig0 * aips.a[(i - 1), (i - 1)])

            for j in range(1, namod + 1):

                t[self.psi_a(i), self.psi_a(j), 0] = -((aips.c[(i - 1), (j - 1)] * scp.betp) / aips.a[(i - 1), (i - 1)]) \
                                                     - (ap.kd * kronecker_delta((i - 1), (j - 1))) / 2

                t[self.theta(i), self.psi_a(j), 0] = (aips.a[(i - 1), (j - 1)] * ap.kd * ap.sig0) \
                                                     / (-2 + 2 * aips.a[(i - 1), (i - 1)] * ap.sig0)

                t[self.psi_a(i), self.theta(j), 0] = (ap.kd * kronecker_delta((i - 1), (j - 1))) / 2

                if par.LSBpa is not None and par.Lpa is not None:
                    heat = 2. * (par.LSBpa + atp.sc * par.Lpa) * kronecker_delta((i - 1), (j - 1))
                else:
                    heat = 0

                t[self.theta(i), self.theta(j), 0] = (-((ap.sig0 * (2. * aips.c[(i - 1), (j - 1)]
                                                        * scp.betp + aips.a[(i - 1), (j - 1)] * (ap.kd + 4. * ap.kdp))))
                                                      + heat) / (-2. + 2. * aips.a[(i - 1), (i - 1)] * ap.sig0)\
                                                      + (atp.hpp * kronecker_delta((i - 1), (j - 1))) / (ap.sig0 * aips.a[(i - 1), (i - 1)])

                if np.any(scp.hk != 0):
                    oro = (aips.g[(i - 1), (j - 1), :] @ scp.hk) / (2 * aips.a[(i - 1), (i - 1)])
                    t[self.psi_a(i), self.psi_a(j), 0] -= oro
                    t[self.theta(i), self.theta(j), 0] -= oro
                    t[self.theta(i), self.psi_a(j), 0] += oro
                    t[self.psi_a(i), self.theta(j), 0] += oro

                for k in range(1, namod + 1):

                    t[self.psi_a(i), self.psi_a(j), self.psi_a(k)] = - aips.b[(i - 1), (j - 1), (k - 1)] \
                                                                     / aips.a[(i - 1), (i - 1)]

                    t[self.psi_a(i), self.theta(j), self.theta(k)] = - aips.b[(i - 1), (j - 1), (k - 1)] \
                                                                     / aips.a[(i - 1), (i - 1)]

                    t[self.theta(i), self.psi_a(j), self.theta(k)] = (aips.g[(i - 1), (j - 1), (k - 1)]
                                                                      - aips.b[(i - 1), (j - 1), (k - 1)] * ap.sig0) / \
                                                                     (-1 + aips.a[(i - 1), (i - 1)] * ap.sig0)

                    t[self.theta(i), self.theta(j), self.psi_a(k)] = (aips.b[(i - 1), (j - 1), (k - 1)] * ap.sig0) \
                                                                     / (1 - aips.a[(i - 1), (i - 1)] * ap.sig0)

            for j in range(1, nomod+1):

                t[self.psi_a(i), self.psi_o(j), 0] = ap.kd * aips.d[(i - 1), (j - 1)] / \
                                                     (2 * aips.a[(i - 1), (i - 1)])

                t[self.theta(i), self.psi_o(j), 0] = ap.kd * (aips.d[(i - 1), (j - 1)] * ap.sig0) \
                                                     / (2 - 2 * aips.a[(i - 1), (i - 1)] * ap.sig0)

                t[self.theta(i), self.deltaT_o(j), 0] = aips.s[(i - 1), (j - 1)] * (2 * par.LSBpo + par.Lpa) \
                                                        / (2 - 2 * aips.a[(i - 1), (i - 1)] * ap.sig0)

        for i in range(1, nomod + 1):
            for j in range(1, namod + 1):
                t[self.psi_o(i), self.psi_a(j), 0] = oips.K[(i - 1), (j - 1)] * par.dp \
                                                     / (oips.M[(i - 1), (i - 1)] + par.G)

                t[self.psi_o(i), self.theta(j), 0] = -(oips.K[(i - 1), (j - 1)]) * par.dp \
                                                     / (oips.M[(i - 1), (i - 1)] + par.G)

            for j in range(1, nomod + 1):

                t[self.psi_o(i), self.psi_o(j), 0] = -((oips.N[(i - 1), (j - 1)] * scp.betp +
                                                        oips.M[(i - 1), (i - 1)] * (par.rp + par.dp) *
                                                        kronecker_delta((i - 1), (j - 1)))) / (oips.M[(i - 1), (i - 1)] + par.G)

                for k in range(1, nomod + 1):
                    t[self.psi_o(i), self.psi_o(j), self.psi_o(k)] = -(oips.C[(i - 1), (j - 1), (k - 1)]) \
                                                                     / (oips.M[(i - 1), (i - 1)] + par.G)

        for i in range(1, nomod + 1):

            t[self.deltaT_o(i), 0, 0] = par.Cpo * oips.W[(i - 1), 0]

            for j in range(1, namod + 1):
                t[self.deltaT_o(i), self.theta(j), 0] = oips.W[(i - 1), (j - 1)] * (2 * atp.sc * par.Lpo + par.sbpa)

            for j in range(1, nomod + 1):

                t[self.deltaT_o(i), self.deltaT_o(j), 0] = - (par.Lpo + par.sbpo) * kronecker_delta((i - 1), (j - 1))

                for k in range(1, nomod + 1):
                    t[self.deltaT_o(i), self.psi_o(j), self.deltaT_o(k)] = -(oips.O[(i - 1), (j - 1), (k - 1)])

        simplify(t)

        for i in range(0, ndim + 1):
            X = csr_matrix(t[i])
            self.tensor.append(X)

        for i in range(0, ndim + 1):
            X = csr_matrix(t[i] + t[i].T)
            self.jacobian_tensor.append(X)


def simplify(t):

    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            for k in range(j):
                if t[i, j, k] != 0:
                    t[i, k, j] += t[i, j, k]
                    t[i, j, k] = 0.


def kronecker_delta(i, j):

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