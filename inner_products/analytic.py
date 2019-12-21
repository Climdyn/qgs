
import numpy as np
# from tensors.cootensor import CooTensor


class WaveNumber(object):

    def __init__(self, typ, P, M, H, Nx, Ny):
        self.typ = typ
        self.P = P
        self.M = M
        self.H = H
        self.Nx = Nx
        self.Ny = Ny

    def __repr__(self):
        return "type = {}, P = {}, M= {},H={}, Nx= {}, Ny={}".format(self.typ, self.P, self.M, self.H, self.Nx, self.Ny)


class AtmosphericInnerProducts(object):

    def __init__(self, params):

        self.params = params
        natm = self.params.nmod

        if natm == 0:
            exit("*** Problem with inner products : natm==0!***")

        self.a = np.zeros((natm, natm), dtype=float)
        self.c = np.zeros((natm, natm), dtype=float)
        self.b = np.zeros((natm, natm, natm), dtype=float)  # could be a CooTensor
        self.g = np.zeros((natm, natm, natm), dtype=float)  # could be a CooTensor

        # initialization of the variables
        atmospheric_wavenumbers = np.empty(natm, dtype=object)

        j = -1

        # awavenum definition

        ams = self.params.ablocks

        for i in range(0, ams.shape[0]):

            if ams[i, 0] == 1:

                atmospheric_wavenumbers[j + 1] = WaveNumber(
                    'A', ams[i, 1], 0, 0, 0, ams[i, 1])

                atmospheric_wavenumbers[j + 2] = WaveNumber(
                    'K', ams[i, 1], ams[i, 0], 0, ams[i, 0], ams[i, 1])

                atmospheric_wavenumbers[j + 3] = WaveNumber(
                    'L', ams[i, 1], 0, ams[i, 0], ams[i, 0], ams[i, 1])

                j = j+3

            else:

                atmospheric_wavenumbers[j + 1] = WaveNumber(
                    'K', ams[i, 1], ams[i, 0], 0, ams[i, 0], ams[i, 1])

                atmospheric_wavenumbers[j + 2] = WaveNumber(
                    'L', ams[i, 1], 0, ams[i, 0], ams[i, 0], ams[i, 1])

                j = j+2

        self.atmospheric_wavenumbers = atmospheric_wavenumbers

        self.calculate_a()
        self.calculate_g()
        self.calculate_b()
        self.calculate_c()

    # !--------------------------------------------------------!
    # ! 1. Inner products in the equations for the atmosphere  !
    # !--------------------------------------------------------!

    def calculate_a(self):
        r"""
        .. math::
            a_{i, j} = (F_i, {\nabla}^2 F_j)

        .. note:: Eigenvalues of the Laplacian (atmospheric)
        """
        nmod = self.params.nmod
        n = self.params.n
        for i in range(0, nmod):
            ti = self.atmospheric_wavenumbers[i]
            self.a[i, i] = - (n**2) * ti.Nx**2 - ti.Ny**2

    def calculate_b(self):
        r"""
        .. math::
            b_{i, j, k} = (F_i, J(F_j, \nabla^2 F_k))

        .. note:: Atmospheric g and a tensors must be computed before \
            calling this routine.
        """

        nmod = self.params.nmod
        for i in range(0, nmod):
            for j in range(0, nmod):
                for k in range(0, nmod):
                    val = self.a[k, k]*self.g[i, j, k]
                    self.b[i, j, k] = val

    def calculate_c(self):
        """
        .. math::
            c_{i,j} = (F_i, \partial_x F_j)
        .. note:: Beta term for the atmosphere
        """

        nmod = self.params.nmod
        n = self.params.n
        for i in range(0, nmod):
            for j in range(0, nmod):
                val = 0.
                Ti = self.atmospheric_wavenumbers[i]
                Tj = self.atmospheric_wavenumbers[j]

                if (Ti.typ, Tj.typ) == ('K', 'L'):
                    val = delta(Ti.M - Tj.H) * delta(Ti.P - Tj.P)
                    val = n * Ti.M * val

                if val != 0.:
                    self.c[i, j] = val
                    self.c[j, i] = -val

    def calculate_g(self):
        """
        .. math::
            g_{i,j,k} = (F_i, J(F_j, F_k))
        .. note:: This is a strict function: it only accepts AKL, KKL
            and LLL types.
            For any other combination, it will not calculate anything.
        """

        nmod = self.params.nmod
        sq2 = np.sqrt(2.)
        pi = np.pi
        n = self.params.n

        for i in range(0, nmod):
            for j in range(0, nmod):
                for k in range(0, nmod):
                    Ti = self.atmospheric_wavenumbers[i]
                    Tj = self.atmospheric_wavenumbers[j]
                    Tk = self.atmospheric_wavenumbers[k]
                    val = 0.

                    if (Ti.typ, Tj.typ, Tk.typ) == ('A', 'K', 'L'):
                        vb1 = B1(Ti.P, Tj.P, Tk.P)
                        vb2 = B2(Ti.P, Tj.P, Tk.P)
                        val = -2 * (sq2 / pi) * Tj.M * delta(Tj.M - Tk.H) \
                            * flambda(Ti.P + Tj.P + Tk.P)
                        if val != 0:
                            val = val * (((vb1**2) / (vb1**2 - 1)) - ((vb2**2)
                            / (vb2**2 - 1)))

                    if (Ti.typ, Tj.typ, Tk.typ) == ('K', 'K', 'L'):
                        vs1 = S1(Tj.P, Tk.P, Tj.M, Tk.H)
                        vs2 = S2(Tj.P, Tk.P, Tj.M, Tk.H)
                        val = vs1 * (delta(Ti.M - Tk.H - Tj.M)
                                     * delta(Ti.P - Tk.P + Tj.P)
                                     - delta(Ti.M - Tk.H - Tj.M)
                                     * delta(Ti.P + Tk.P - Tj.P)
                                     + (delta(Tk.H - Tj.M + Ti.M)
                                     + delta(Tk.H - Tj.M - Ti.M))
                                     * delta(Tk.P + Tj.P - Ti.P)) \
                              + vs2 * (delta(Ti.M - Tk.H - Tj.M)
                                       * delta(Ti.P - Tk.P - Tj.P)
                                       + (delta(Tk.H - Tj.M - Ti.M)
                                       + delta(Ti.M + Tk.H - Tj.M))
                                       * (delta(Ti.P - Tk.P + Tj.P)
                                       - delta(Tk.P - Tj.P + Ti.P)))

                    val = val * n

                    if val != 0.:
                        self.g[i, j, k] = val
                        self.g[j, k, i] = val
                        self.g[k, i, j] = val
                        self.g[i, k, j] = -val
                        self.g[j, i, k] = -val
                        self.g[k, j, i] = -val

        for i in range(0, nmod):
            for j in range(i+1, nmod):
                for k in range(j+1, nmod):

                    Ti = self.atmospheric_wavenumbers[i]
                    Tj = self.atmospheric_wavenumbers[j]
                    Tk = self.atmospheric_wavenumbers[k]

                    val = 0.

                    if (Ti.typ, Tj.typ, Tk.typ) == ('L', 'L', 'L'):
                        vs3 = S3(Tj.P, Tk.P, Tj.H, Tk.H)
                        vs4 = S4(Tj.P, Tk.P, Tj.H, Tk.H)
                        val = vs3 * ((delta(Tk.H - Tj.H - Ti.H)
                                      - delta(Tk.H - Tj.H + Ti.H))
                                     * delta(Tk.P + Tj.P - Ti.P)
                                     + delta(Tk.H + Tj.H - Ti.H)
                                     * (delta(Tk.P - Tj.P + Ti.P)
                                     - delta(Tk.P - Tj.P - Ti.P))) \
                              + vs4 * ((delta(Tk.H + Tj.H - Ti.H)
                                        * delta(Tk.P - Tj.P - Ti.P))
                                       + (delta(Tk.H - Tj.H + Ti.H)
                                       - delta(Tk.H - Tj.H - Ti.H))
                                       * (delta(Tk.P - Tj.P - Ti.P)
                                       - delta(Tk.P - Tj.P + Ti.P)))

                    val = val * n

                    if val != 0.:
                        self.g[i, j, k] = val
                        self.g[j, k, i] = val
                        self.g[k, i, j] = val
                        self.g[i, k, j] = -val
                        self.g[j, i, k] = -val
                        self.g[k, j, i] = -val

#  !-----------------------------------------------------!
#  !                                                     !
#  ! Definition of the Helper functions from Cehelsky    !
#  ! \ Tung                                              !
#  !                                                     !
#  !-----------------------------------------------------!


def B1(Pi, Pj, Pk):
    return (Pk+Pj)/float(Pi)


def B2(Pi, Pj, Pk):
    return (Pk-Pj)/float(Pi)


def delta(r):

    if r == 0:
        return 1.

    else:
        return 0.


def flambda(r):

    if r % 2 == 0:
        return 0.

    else:
        return 1.


def S1(Pj, Pk, Mj, Hk):
    return -(Pk * Mj + Pj * Hk) / 2.


def S2(Pj, Pk, Mj, Hk):
    return (Pk * Mj - Pj * Hk) / 2.


def S3(Pj, Pk, Hj, Hk):
    return (Pk * Hj + Pj * Hk) / 2.


def S4(Pj, Pk, Hj, Hk):
    return (Pk * Hj - Pj * Hk) / 2.


if __name__ == '__main__':
    from params.params import QGParams
    params = QGParams()
    params.set_max_modes(2, 2)
    aip = AtmosphericInnerProducts(params)
