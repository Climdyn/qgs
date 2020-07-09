"""
    Inner products module (analytic)
    ================================

    Inner products

    .. math::

        (S, G) = \\frac{n}{2\\pi^2}\\int_0^\\pi\\int_0^{2\\pi/n} S(x,y)\\, G(x,y)\\, \\mathrm{d} x \\, \\mathrm{d} y


    between the truncated set of basis functions :math:`\phi_i` for the ocean streamfunctions and
    :math:`F_i` for the atmosphere streamfunction and temperature fields (see :ref:`files/model/oro_model:Projecting the equations on a set of basis functions`).

    Notes
    -----

    These are computed using the analytical expressions from:

    * De Cruz, L., Demaeyer, J. and Vannitsem, S.: *The Modular Arbitrary-Order Ocean-Atmosphere Model: MAOOAM v1.0*,
      Geosci. Model Dev., **9**, 2793-2808, `doi:10.5194/gmd-9-2793-2016 <http://dx.doi.org/10.5194/gmd-9-2793-2016>`_, 2016.
    * Cehelsky, P., & Tung, K. K. (1987). *Theories of multiple equilibria and weather regimes—A critical reexamination.
      Part II: Baroclinic two-layer models*. Journal of the atmospheric sciences, **44** (21), 3282-3303.
      `doi:10.1175/1520-0469(1987)044%3C3282%3ATOMEAW%3E2.0.CO%3B2 <https://doi.org/10.1175/1520-0469(1987)044%3C3282%3ATOMEAW%3E2.0.CO%3B2>`_

    Description of the classes
    --------------------------

    There is one class to define the wavenumber of the basis function of the model: :class:`WaveNumber`.

    The two classes computing and holding the inner products of the basis functions are:

    * :class:`AtmosphericInnerProducts`
    * :class:`OceanicInnerProducts`

"""

# TODO : inner products should be sparse tensor.

import numpy as np
import sparse as sp


class WaveNumber(object):
    """Class to define model base functions wavenumber. The basis function available are:

    * `'A'` for a function of the form :math:`F^A_{P} (x, y) =  \sqrt{2}\, \cos(P y) = \sqrt{2}\, \cos(n_y\, y)`
    * `'K'` for a function of the form :math:`F^K_{M,P} (x, y) =  2\cos(M nx)\, \sin(P y) = 2\cos(n_x\,  n\, x)\, \sin(n_y\, y)`
    * `'L'` for a function of the form :math:`F^L_{H,P} (x, y) = 2\sin(H nx)\, \sin(P y) = 2\sin(n_x\, n \,x)\, \sin(n_y\, y)`

    where :math:`x` and :math:`y` are the nondimensional model's domain coordinates (see :ref:`files/model/oro_model:Projecting the equations on a set of basis functions`).

    Parameters
    ----------
    function_type: str
        One character string to define the type of basis function. It can be `'A'`, `'K'` or `'L'`.
    P: int
        The :math:`y` wavenumber integer.
    M: int
        The :math:`x` wavenumber integer.
    H: int
        The :math:`x` wavenumber integer.
    nx: float
        The :math:`x` wavenumber.
    ny: float
        The :math:`y` wavenumber.

    Attributes
    ----------
    function_type: str
        One character string to define the type of basis function. It can be `'A'`, `'K'` or `'L'`.
    P: int
        The :math:`y` wavenumber integer.
    M: int
        The :math:`x` wavenumber integer.
    H: int
        The :math:`x` wavenumber integer.
    nx: float
        The :math:`x` wavenumber.
    ny: float
        The :math:`y` wavenumber.

    """

    def __init__(self, function_type, P, M, H, nx, ny):
        self.type = function_type
        self.P = P
        self.M = M
        self.H = H
        self.nx = nx
        self.ny = ny

    def __repr__(self):
        return "type = {}, P = {}, M= {},H={}, nx= {}, ny={}".format(self.type, self.P, self.M, self.H, self.nx, self.ny)


class AtmosphericInnerProducts(object):
    """Class which contains all the atmospheric inner products coefficients needed for the tendencies
    tensor :class:`~tensors.qgtensor.QgsTensor` computation.

    Warnings
    --------

    * Atmospheric :attr:`g` tensor and :attr:`a` matrix must be computed before computing the :attr:`b` tensor.
    * Atmospheric :attr:`s` matrix and oceanic :attr:`OceanInnerProducts.M` matrix must be computed before computing :attr:`d`.

    Parameters
    ----------
    params: ~params.params.QgParams
        An instance of model's parameters object.

    Attributes
    ----------
    connected_to_ocean: bool
        Indicate if the atmosphere is connected to an ocean.
    params: ~params.params.QgParams
        An instance of model's parameters object.
    atmospheric_wavenumbers: ~numpy.ndarray(WaveNumber)
        An array of shape (:attr:`~params.params.QgParams.nmod` [0], ) of the wavenumber object of each mode.
    a: sparse.DOK(float)
        Matrix of the eigenvalues of the Laplacian (atmospheric): :math:`a_{i, j} = (F_i, {\\nabla}^2 F_j)`. \n
        Array of shape (:attr:`~params.params.QgParams.nmod` [0], :attr:`~params.params.QgParams.nmod` [0]).
    c: ~sparse.DOK(float)
        Matrix of beta terms for the atmosphere: :math:`c_{i,j} = (F_i, \partial_x F_j)`. \n
        Array of shape (:attr:`~params.params.QgParams.nmod` [0], :attr:`~params.params.QgParams.nmod` [0]).
    b: ~sparse.DOK(float)
        Tensors holding the Jacobian inner products: :math:`b_{i, j, k} = (F_i, J(F_j, \\nabla^2 F_k))`.
        Array of shape (:attr:`~params.params.QgParams.nmod` [0], :attr:`~params.params.QgParams.nmod` [0],
        :attr:`~params.params.QgParams.nmod` [0]).
    g: ~sparse.DOK(float)
        Tensors holding the Jacobian inner products: :math:`g_{i,j,k} = (F_i, J(F_j, F_k))`. \n
        Array of shape (:attr:`~params.params.QgParams.nmod` [0], :attr:`~params.params.QgParams.nmod` [0],
        :attr:`~params.params.QgParams.nmod` [0]).
    d: None or ~sparse.DOK(float)
        Forcing of the ocean on the atmosphere: :math:`d_{i,j} = (F_i, \\nabla^2 \phi_j)`. \n
        Not defined if no ocean is present. \n
        Array of shape (:attr:`~params.params.QgParams.nmod` [0], :attr:`~params.params.QgParams.nmod` [0]).
    s: None or ~sparse.DOK(float)
        Forcing (thermal) of the ocean on the atmosphere: :math:`s_{i,j} = (F_i, \phi_j)`. \n
        Not defined if no ocean is present. \n
        Array of shape (:attr:`~params.params.QgParams.nmod` [0], :attr:`~params.params.QgParams.nmod` [0]).
    """

    def __init__(self, params):

        self.connected_to_ocean = False
        self.params = params
        natm, noc = self.params.nmod

        if natm == 0:
            exit("*** Problem with inner products : natm==0!***")

        self.a = sp.zeros((natm, natm), dtype=float, format='dok')
        self.c = sp.zeros((natm, natm), dtype=float, format='dok')
        self.b = sp.zeros((natm, natm, natm), dtype=float, format='dok')
        self.g = sp.zeros((natm, natm, natm), dtype=float, format='dok')
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

        self._calculate_a()
        self._calculate_g()
        self._calculate_b()
        self._calculate_c()

    def connect_to_ocean(self, ocean_inner_products):
        """Connect the atmosphere to an ocean.

        Parameters
        ----------
        ocean_inner_products: OceanicInnerProducts
            The inner products of the ocean.
        """

        natm, noc = self.params.nmod

        self.d = sp.zeros((natm, noc), dtype=float, format='dok')
        self.s = sp.zeros((natm, noc), dtype=float, format='dok')

        self._calculate_s(ocean_inner_products)

        # ensure that the ocean is connected as well
        if not ocean_inner_products.connected_to_atmosphere:
            ocean_inner_products.connect_to_atmosphere(self)

        self._calculate_d(ocean_inner_products)
        self.connected_to_ocean = True

    # !-----------------------------------------------------!
    # ! Inner products in the equations for the atmosphere  !
    # !-----------------------------------------------------!

    def _calculate_a(self):
        nmod = self.params.nmod[0]
        n = self.params.scale_params.n
        for i in range(0, nmod):
            Ti = self.atmospheric_wavenumbers[i]
            self.a[i, i] = - (n**2) * Ti.nx**2 - Ti.ny**2

    def _calculate_b(self):
        nmod = self.params.nmod[0]
        for i in range(0, nmod):
            for j in range(0, nmod):
                for k in range(0, nmod):
                    val = self.a[k, k]*self.g[i, j, k]
                    self.b[i, j, k] = val

    def _calculate_c(self):
        nmod = self.params.nmod[0]
        n = self.params.scale_params.n
        for i in range(0, nmod):
            for j in range(0, nmod):
                val = 0.
                Ti = self.atmospheric_wavenumbers[i]
                Tj = self.atmospheric_wavenumbers[j]

                if (Ti.type, Tj.type) == ('K', 'L'):
                    val = delta(Ti.M - Tj.H) * delta(Ti.P - Tj.P)
                    val = n * Ti.M * val

                if val != 0.:
                    self.c[i, j] = val
                    self.c[j, i] = -val

    def _calculate_g(self):
        """

        Warnings
        --------
        This is a strict function: it only accepts AKL, KKL and LLL types.
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

                    if (Ti.type, Tj.type, Tk.type) == ('A', 'K', 'L'):
                        vb1 = B1(Ti.P, Tj.P, Tk.P)
                        vb2 = B2(Ti.P, Tj.P, Tk.P)
                        val = -2 * (sq2 / pi) * Tj.M * delta(Tj.M - Tk.H) \
                            * flambda(Ti.P + Tj.P + Tk.P)
                        if val != 0:
                            val = val * (((vb1**2) / (vb1**2 - 1)) - ((vb2**2)
                            / (vb2**2 - 1)))

                    if (Ti.type, Tj.type, Tk.type) == ('K', 'K', 'L'):
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

                    if (Ti.type, Tj.type, Tk.type) == ('L', 'L', 'L'):
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

    def _calculate_s(self, ocean_inner_products):

        sq2 = np.sqrt(2.)
        pi = np.pi
        natm, noc = self.params.nmod

        val = 0

        for i in range(0, natm):
            for j in range(0, noc):

                Ti = self.atmospheric_wavenumbers[i]
                Dj = ocean_inner_products.oceanic_wavenumbers[j]

                val = 0.

                if Ti.type == 'A':
                    val = flambda(Dj.H) * flambda(Dj.P + Ti.P)
                    if val != 0.:
                        val = val * 8 * sq2 * Dj.P / \
                              (pi ** 2 * (Dj.P ** 2 - Ti.P ** 2) * Dj.H)

                if Ti.type == 'K':
                    val = flambda(2 * Ti.M + Dj.H) * delta(Dj.P - Ti.P)

                    if val != 0:
                        val = val * 4 * Dj.H / (pi * (-4 * Ti.M ** 2 + Dj.H ** 2))

                if Ti.type == 'L':
                    val = delta(Dj.P - Ti.P) * delta(2 * Ti.H - Dj.H)

                if val != 0.:
                    self.s[i, j] = val

    def _calculate_d(self, ocean_inner_products):

        natm, noc = self.params.nmod

        for i in range(0, natm):
            for j in range(0, noc):
                self.d[i, j] = self.s[i, j] * ocean_inner_products.M[j, j]


class OceanicInnerProducts(object):
    """Class which contains all the oceanic inner products coefficients needed for the tendencies
    tensor :class:`~tensors.qgtensor.QgsTensor` computation.

    Warnings
    --------

    * The computation of the tensor :attr:`C` requires that the tensor :attr:`O` and the matrix :attr:`M` be computed beforehand.
    * The computation of the matrix :attr:`W` requires that the matrix :attr:`AtmosphericInnerProducts.s` be computed beforehand.
    * The computation of the matrix :attr:`K` requires that the matrices :attr:`AtmosphericInnerProducts.a`
      and :attr:`AtmosphericInnerProducts.s` be computed beforhand.

    Parameters
    ----------
    params: ~params.params.QgParams
        An instance of model's parameters object.

    Attributes
    ----------
    init: bool
        Indicate if the initialization of the ocean inner products has been done.
    connected_to_atmosphere: bool
        Indicate if the ocean is connected to an atmosphere.
    params: ~params.params.QgParams
        An instance of model's parameters object.
    M: ~sparse.DOK(float)
        Forcing of the ocean fields on the ocean: :math:`M_{i,j} = (\phi_i, \\nabla^2 \phi_j)`. \n
        Array of shape (:attr:`~params.params.QgParams.nmod` [1], :attr:`~params.params.QgParams.nmod` [1]).
    N: ~sparse.DOK(float)
        Beta term for the ocean: :math:`N_{i,j} = (\phi_i, \partial_x \phi_j)`.
        Array of shape (:attr:`~params.params.QgParams.nmod` [1], :attr:`~params.params.QgParams.nmod` [1]).
    O: ~sparse.DOK(float)
        Temperature advection term (passive scalar): :math:`O_{i,j,k} = (\phi_i, J(\phi_j, \phi_k))`.
        Array of shape (:attr:`~params.params.QgParams.nmod` [1], :attr:`~params.params.QgParams.nmod` [1],
        :attr:`~params.params.QgParams.nmod` [1]).
    C: ~sparse.DOK(float)
        Tensors holding the Jacobian inner products: :math:`C_{i,j,k} = (\phi_i, J(\phi_j,\\nabla^2 \phi_k))`. \n
        Array of shape (:attr:`~params.params.QgParams.nmod` [1], :attr:`~params.params.QgParams.nmod` [1],
        :attr:`~params.params.QgParams.nmod` [1]).
    K: None or ~sparse.DOK(float)
        Forcing of the ocean by the atmosphere: :math:`K_{i,j} = (\phi_i, \\nabla^2 F_j)`.
        Not defined if no atmosphere is present. \n
        Array of shape (:attr:`~params.params.QgParams.nmod` [1], :attr:`~params.params.QgParams.nmod` [1]).
    W: None or ~sparse.DOK(float)
        Short-wave radiative forcing of the ocean: :math:`W_{i,j} = (\phi_i, F_j)`. \n
        Not defined if no atmosphere is present. \n
        Array of shape (:attr:`~params.params.QgParams.nmod` [1], :attr:`~params.params.QgParams.nmod` [1]).

    """
    def __init__(self, params):

        self.init = False
        self.connected_to_atmosphere = False
        self.params = params
        natm, noc = self.params.nmod

        self.M = sp.zeros((noc, noc), dtype=float, format='dok')
        self.N = sp.zeros((noc, noc), dtype=float, format='dok')

        self.O = sp.zeros((noc, noc, noc), dtype=float, format='dok')
        self.C = sp.zeros((noc, noc, noc), dtype=float, format='dok')

        self.K = None
        self.W = None

        # initialization of the variables
        oceanic_wavenumbers = np.empty(noc, dtype=object)

        # Oceanic wavenumbers definition
        oms = self.params.goblocks

        for i in range(oms.shape[0]):  # function type is limited to L for the moment: ocean is a closed basin
            oceanic_wavenumbers[i] = WaveNumber('L', oms[i, 1], 0, oms[i, 0], oms[i, 0] / 2., oms[i, 1])

        self.oceanic_wavenumbers = oceanic_wavenumbers

        self._calculate_M()
        self._calculate_N()
        self._calculate_O()
        self._calculate_C()

    def connect_to_atmosphere(self, atmosphere_inner_products):
        """Connect the ocean to an atmosphere.

        Parameters
        ----------
        atmosphere_inner_products: AtmosphericInnerProducts
            The inner products of the atmosphere.
        """

        natm, noc = self.params.nmod

        self.K = sp.zeros((noc, natm), dtype=float, format='dok')
        self.W = sp.zeros((noc, natm), dtype=float, format='dok')

        if atmosphere_inner_products.s is None:
            atmosphere_inner_products.s = sp.zeros((natm, noc), dtype=float, format='dok')
            atmosphere_inner_products._calculate_s(self)

        self._calculate_W(atmosphere_inner_products)
        self._calculate_K(atmosphere_inner_products)

        self.connected_to_atmosphere = True

    def _calculate_K(self, atmosphere_inner_products):

        natm, noc = self.params.nmod

        for i in range(0, noc):
            for j in range(0, natm):
                self.K[i, j] = atmosphere_inner_products.s[j, i] * atmosphere_inner_products.a[j, j]

    def _calculate_M(self):

        nmod = self.params.nmod[1]
        n = self.params.scale_params.n
        for i in range(nmod):
            Di = self.oceanic_wavenumbers[i]
            self.M[i, i] = - (n**2) * Di.nx**2 - Di.ny**2

    def _calculate_N(self):

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

    def _calculate_O(self):

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

    def _calculate_C(self):

        nmod = self.params.nmod[1]

        val = 0.

        for i in range(0, nmod):
            for j in range(0, nmod):
                for k in range(0, nmod):

                    val = self.M[k, k] * self.O[i, j, k]

                    if val != 0:
                        self.C[i, j, k] = val

    def _calculate_W(self, atmosphere_inner_products):

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
    pars.set_atmospheric_modes(2, 2)
    aip = AtmosphericInnerProducts(pars)
    pars.set_oceanic_modes(2, 2)
    oip = OceanicInnerProducts(pars)
    aip.connect_to_ocean(oip)
