
import numpy as np
from scipy.sparse import csr_matrix


# real_eps = np.finfo(np.float64).eps


class AtmosphericTensor(object):

    def __init__(self, inner_products):

        self.inner_products = inner_products
        self.params = self.inner_products.params
        self.tensor = list()

        self.compute_tensor()

    def psi(self, i):
        return i

    def theta(self, i):
        return i + self.params.nmod

    def compute_tensor(self):

        ips = self.inner_products
        par = self.params
        nmod = par.nmod
        ndim = par.ndim
        t = np.zeros((ndim+1, ndim + 1, ndim + 1), dtype=np.float64)  # could be initialized better

        for i in range(1, nmod + 1):

            t[self.theta(i), 0, 0] = - par.hpp * par.thetas[(i - 1)] / (par.sig0 * ips.a[(i - 1), (i - 1)])

            for j in range(1, nmod + 1):

                t[self.psi(i), self.psi(j), 0] = -((ips.c[(i - 1), (j - 1)] * par.betp) / ips.a[(i - 1), (i - 1)]) \
                                                 - (par.kd * kronecker_delta((i - 1), (j - 1))) / 2 \
                                                 - (ips.g[(i - 1), (j - 1), :] @ par.hk) / (2 * ips.a[(i - 1), (i - 1)])

                t[self.theta(i), self.psi(j), 0] = (ips.a[(i - 1), (j - 1)] * par.kd * par.sig0)\
                                                   / (-2 + 2 * ips.a[(i - 1), (i - 1)] * par.sig0) \
                                                   + (ips.g[(i - 1), (j - 1), :] @ par.hk) / (2 * ips.a[(i - 1), (i - 1)])

                t[self.psi(i), self.theta(j), 0] = (par.kd * kronecker_delta((i - 1), (j - 1))) / 2 \
                                                   + (ips.g[(i - 1), (j - 1), :] @ par.hk) / (2 * ips.a[(i - 1), (i - 1)])

                t[self.theta(i), self.theta(j), 0] = - (par.sig0*(2*ips.c[(i - 1), (j - 1)] * par.betp
                                                                  + ips.a[(i - 1), (j - 1)] * (par.kd + 4. * par.kdp)))\
                                                     / (-2. + 2. * ips.a[(i - 1), (i - 1)] * par.sig0) \
                                                     - (ips.g[(i - 1), (j - 1), :] @ par.hk) / (2 * ips.a[(i - 1), (i - 1)]) \
                                                     + (par.hpp * kronecker_delta((i - 1), (j - 1))) / (par.sig0 * ips.a[(i - 1), (i - 1)])

                for k in range(1, nmod + 1):

                    t[self.psi(i), self.psi(j), self.psi(k)] = - ips.b[(i - 1), (j - 1), (k - 1)]\
                                                               / ips.a[(i - 1), (i - 1)]

                    t[self.psi(i), self.theta(j), self.theta(k)] = - ips.b[(i - 1), (j - 1), (k - 1)]\
                                                                   / ips.a[(i - 1), (i - 1)]

                    t[self.theta(i), self.psi(j), self.theta(k)] = (ips.g[(i - 1), (j - 1), (k - 1)]
                                                                    - ips.b[(i - 1), (j - 1), (k - 1)] * par.sig0) / \
                                                                   (-1 + ips.a[(i - 1), (i - 1)] * par.sig0)

                    t[self.theta(i), self.theta(j), self.psi(k)] = (ips.b[(i - 1), (j - 1), (k - 1)] * par.sig0)\
                                                                   / (1 - ips.a[(i - 1), (i - 1)] * par.sig0)

        simplify(t)

        for i in range(0, ndim + 1):
            X = csr_matrix(t[i])
            self.tensor.append(X)


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
    from params.params import QGParams
    from inner_products.analytic import AtmosphericInnerProducts
    from tensors.cootensor import from_csr_mat_list

    params = QGParams()
    params.set_max_modes(2, 2)
    params.hk[1] = 0.2
    params.thetas[0] = 0.1
    aip = AtmosphericInnerProducts(params)
    atensor = AtmosphericTensor(aip)
    coo_atensor = from_csr_mat_list(atensor.tensor)