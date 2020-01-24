
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

        self.connected_to_ocean = False
        self.params = params
        natm, noc = self.params.nmod

        if natm == 0:
            exit("*** Problem with inner products : natm==0!***")

        self.a = np.zeros((natm, natm), dtype=float)
        self.c = np.zeros((natm, natm), dtype=float)
        self.b = np.zeros((natm, natm, natm), dtype=float)  # could be a CooTensor
        self.g = np.zeros((natm, natm, natm), dtype=float)  # could be a CooTensor
        self.d = None
        self.s = None

        # initialization of the variables
        atmospheric_wavenumbers = np.empty(natm, dtype=object)

        j = -1

        # Atmospheric wavenumbers definition

        ams = self.params.ablocks

        for i in range(ams.shape[0]):  # function type is limited to AKL for the moment: atmosphere is a channel

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

    def connect_to_ocean(self, ocean_inner_products):

        natm, noc = self.params.nmod

        self.d = np.zeros((natm, noc), dtype=float)
        self.s = np.zeros((natm, noc), dtype=float)

        self.calculate_s(ocean_inner_products)

        if not ocean_inner_products.connected_to_atmosphere:
            ocean_inner_products.connect_to_atmosphere(self)

        self.calculate_d(ocean_inner_products)
        self.connected_to_ocean = True

    # !-----------------------------------------------------!
    # ! Inner products in the equations for the atmosphere  !
    # !-----------------------------------------------------!

    def calculate_a(self):
        r"""
        .. math::
            a_{i, j} = (F_i, {\nabla}^2 F_j)

        .. note:: Eigenvalues of the Laplacian (atmospheric)
        """
        nmod = self.params.nmod[0]
        n = self.params.scale_params.n
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

        nmod = self.params.nmod[0]
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

        nmod = self.params.nmod[0]
        n = self.params.scale_params.n
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

        nmod = self.params.nmod[0]
        sq2 = np.sqrt(2.)
        pi = np.pi
        n = self.params.scale_params.n

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

    def calculate_s(self, ocean_inner_products):
        """
        .. math::
            s_{i,j} = (F_i, \eta_j)
        .. note:: Forcing (thermal) of the ocean on the atmosphere.
        """

        sq2 = np.sqrt(2.)
        pi = np.pi
        natm, noc = self.params.nmod

        val = 0

        for i in range(0, natm):
            for j in range(0, noc):

                Ti = self.atmospheric_wavenumbers[i]
                Dj = ocean_inner_products.oceanic_wavenumbers[j]

                val = 0.

                if Ti.typ == 'A':
                    val = flambda(Dj.H) * flambda(Dj.P + Ti.P)
                    if val != 0.:
                        val = val * 8 * sq2 * Dj.P / \
                              (pi ** 2 * (Dj.P ** 2 - Ti.P ** 2) * Dj.H)

                if Ti.typ == 'K':
                    val = flambda(2 * Ti.M + Dj.H) * delta(Dj.P - Ti.P)

                    if val != 0:
                        val = val * 4 * Dj.H / (pi * (-4 * Ti.M ** 2 + Dj.H ** 2))

                if Ti.typ == 'L':
                    val = delta(Dj.P - Ti.P) * delta(2 * Ti.H - Dj.H)

                if val != 0.:
                    self.s[i, j] = val

    def calculate_d(self, ocean_inner_products):
        r"""
        .. math::
            d_{i,j} = (F_i, \nabla^2 \eta_j)
        .. note:: Forcing of the ocean on the atmosphere.
            Atmospheric s tensor and oceanic M tensor must be computed
            before calling this routine !
        """

        natm, noc = self.params.nmod

        for i in range(0, natm):
            for j in range(0, noc):
                self.d[i, j] = self.s[i, j] * ocean_inner_products.M[j, j]


class OceanicInnerProducts(object):

    def __init__(self, params):

        self.init = False
        self.connected_to_atmosphere = False
        self.params = params
        natm, noc = self.params.nmod

        self.M = np.zeros((noc, noc), dtype=float)
        self.N = np.zeros((noc, noc), dtype=float)

        self.O = np.zeros((noc, noc, noc), dtype=float)  # could be a CooTensor
        self.C = np.zeros((noc, noc, noc), dtype=float)  # could be a CooTensor

        self.K = None
        self.W = None

        # initialization of the variables
        oceanic_wavenumbers = np.empty(noc, dtype=object)

        # Oceanic wavenumbers definition
        oms = self.params.oblocks

        for i in range(oms.shape[0]):  # function type is limited to L for the moment: ocean is a closed basin
            oceanic_wavenumbers[i] = WaveNumber('L', oms[i, 1], 0, oms[i, 0], oms[i, 0] / 2., oms[i, 1])

        self.oceanic_wavenumbers = oceanic_wavenumbers

        self.calculate_M()
        self.calculate_N()
        self.calculate_O()
        self.calculate_C()

    def connect_to_atmosphere(self, atmosphere_inner_products):

        natm, noc = self.params.nmod

        self.K = np.zeros((noc, natm), dtype=float)
        self.W = np.zeros((noc, natm), dtype=float)

        if atmosphere_inner_products.s is None:
            atmosphere_inner_products.s = np.zeros((natm, noc), dtype=float)
            atmosphere_inner_products.calculate_s(self)

        self.calculate_W(atmosphere_inner_products)
        self.calculate_K(atmosphere_inner_products)

        self.connected_to_atmosphere = True

    def calculate_K(self, atmosphere_inner_products):
        r"""
        Forcing of the atmosphere on the ocean.

        .. math::
            K_{i,j} = (\eta_i, \nabla^2 F_j)

        .. note::
            atmospheric a and s tensors must be computed before calling
            this function !
        """

        natm, noc = self.params.nmod

        for i in range(0, noc):
            for j in range(0, natm):
                self.K[i, j] = atmosphere_inner_products.s[j, i] * atmosphere_inner_products.a[j, j]


    def calculate_M(self):
        r"""
        Forcing of the ocean fields on the ocean.

        .. math::
            M_{i,j} = (\eta_i, \nabla^2 \eta_j)

        """

        nmod = self.params.nmod[1]
        n = self.params.scale_params.n
        for i in range(nmod):
            Di = self.oceanic_wavenumbers[i]
            self.M[i, i] = - (n**2) * Di.Nx**2 - Di.Ny**2

    def calculate_N(self):
        """
        Beta term for the ocean

        .. math::
            N_{i,j} = (\eta_i, \partial_x \eta_j)

        """

        nmod = self.params.nmod[1]
        n = self.params.scale_params.n
        pi = np.pi

        val = 0.

        for i in range(0, nmod):
            for j in range(0, nmod):
                Di = self.oceanic_wavenumbers[i]
                Dj = self.oceanic_wavenumbers[j]
                val = delta(Di.P - Dj.P) * flambda(Di.H + Dj.H)

                if val != 0.:
                    self.N[i, j] = val * (-2) * Dj.H * Di.H * n / \
                                   ((Dj.H**2 - Di.H**2) * pi)

    def calculate_O(self):
        """
        Temperature advection term (passive scalar)

        .. math::
            O_{i,j,k} = (\eta_i, J(\eta_j, \eta_k))
        """

        nmod = self.params.nmod[1]
        n = self.params.scale_params.n

        val = 0.

        for i in range(0, nmod):
            for j in range(i, nmod):
                for k in range(i, nmod):

                    Di = self.oceanic_wavenumbers[i]
                    Dj = self.oceanic_wavenumbers[j]
                    Dk = self.oceanic_wavenumbers[k]

                    vs3 = S3(Dj.P, Dk.P, Dj.H, Dk.H)

                    vs4 = S4(Dj.P, Dk.P, Dj.H, Dk.H)

                    val = vs3*((delta(Dk.H - Dj.H - Di.H)
                                - delta(Dk.H - Dj.H + Di.H))
                               * delta(Dk.P + Dj.P - Di.P)
                               + delta(Dk.H + Dj.H - Di.H)
                               * (delta(Dk.P - Dj.P + Di.P)
                                  - delta(Dk.P - Dj.P - Di.P))) \
                        + vs4 * ((delta(Dk.H + Dj.H - Di.H)
                                 * delta(Dk.P - Dj.P - Di.P))
                                 + (delta(Dk.H - Dj.H + Di.H)
                                 - delta(Dk.H - Dj.H - Di.H))
                                 * (delta(Dk.P - Dj.P - Di.P)
                                    - delta(Dk.P - Dj.P + Di.P)))

                    val = val * n / 2

                    if val != 0:
                        self.O[i, j, k] = val
                        self.O[j, k, i] = val
                        self.O[k, i, j] = val
                        self.O[i, k, j] = -val
                        self.O[j, i, k] = -val
                        self.O[k, j, i] = -val

    def calculate_C(self):
        r"""
        .. math::
            C_{i,j,k} = (\eta_i, J(\eta_j,\nabla^2 \eta_k))

        .. note :: Requires :math:`O_{i,j,k}` \

        and :math:`M_{i,j}` to be calculated beforehand.

        """

        nmod = self.params.nmod[1]

        val = 0.

        for i in range(0, nmod):
            for j in range(0, nmod):
                for k in range(0, nmod):

                    val = self.M[k, k] * self.O[i, j, k]

                    if val != 0:
                        self.C[i, j, k] = val

    def calculate_W(self, atmosphere_inner_products):
        """
        Short-wave radiative forcing of the ocean.

        .. math::
            W_{i,j} = (\eta_i, F_j)

        .. note ::
            atmospheric s tensor must be computed before calling
            this function !
        """

        natm, noc = self.params.nmod

        for i in range(0, noc):
            for j in range(0, natm):
                self.W[i, j] = atmosphere_inner_products.s[j, i]

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
    from params.params import QgParams
    pars = QgParams()
    pars.set_max_atmospheric_modes(2, 2)
    aip = AtmosphericInnerProducts(pars)
    pars.set_max_oceanic_modes(2, 2)
    oip = OceanicInnerProducts(pars)
    aip.connect_to_ocean(oip)
