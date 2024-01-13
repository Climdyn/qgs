"""
    Analytic Inner products module
    ==============================

    Inner products

    .. math::

        (S, G) = \\frac{n}{2\\pi^2}\\int_0^\\pi\\int_0^{2\\pi/n} S(x,y)\\, G(x,y)\\, \\mathrm{d} x \\, \\mathrm{d} y


    between the truncated set of basis functions :math:`\phi_i` for the ocean and land fields, and
    :math:`F_i` for the atmosphere fields (see :ref:`files/model/oro_model:Projecting the equations on a set of basis functions`).

    Notes
    -----

    These inner products are computed using the analytical expressions from:

    * De Cruz, L., Demaeyer, J. and Vannitsem, S.: *The Modular Arbitrary-Order Ocean-Atmosphere Model: MAOOAM v1.0*,
      Geosci. Model Dev., **9**, 2793-2808, `doi:10.5194/gmd-9-2793-2016 <http://dx.doi.org/10.5194/gmd-9-2793-2016>`_, 2016.
    * Cehelsky, P., & Tung, K. K. (1987). *Theories of multiple equilibria and weather regimes—A critical reexamination.
      Part II: Baroclinic two-layer models*. Journal of the atmospheric sciences, **44** (21), 3282-3303.
      `doi:10.1175/1520-0469(1987)044%3C3282%3ATOMEAW%3E2.0.CO%3B2 <https://doi.org/10.1175/1520-0469(1987)044%3C3282%3ATOMEAW%3E2.0.CO%3B2>`_

    Description of the classes
    --------------------------

    The three classes computing and holding the inner products of the basis functions are:

    * :class:`AtmosphericAnalyticInnerProducts`
    * :class:`OceanicAnalyticInnerProducts`
    * :class:`GroundAnalyticInnerProducts`

"""

import numpy as np
import sparse as sp

from qgs.params.params import QgParams
from qgs.basis.fourier import channel_wavenumbers, basin_wavenumbers
from qgs.inner_products.base import AtmosphericInnerProducts, OceanicInnerProducts, GroundInnerProducts

# TODO: - Add warnings if trying to connect analytic and symbolic inner products together
#       - Allow analytic inner product to be returned as symbolic arrays


class AtmosphericAnalyticInnerProducts(AtmosphericInnerProducts):
    """Class which contains all the atmospheric inner products coefficients needed for the tendencies
    tensor :class:`~.tensors.qgtensor.QgsTensor` computation, computed with analytic formula.

    Parameters
    ----------
    params: None or QgParams or list, optional
        An instance of model's parameters object or a list in the form [aspect_ratio, ablocks, natm].
        If a list is provided, `aspect_ratio` is the aspect ratio of the domain, `ablocks` is a spectral blocks
        detailing the model's atmospheric modes :math:`x`-and :math:`y`-wavenumber as an array of shape (natm, 2), and `natm` is
        the number of modes in the atmosphere.
        If `None`, an empty object is initialized.
    stored: bool, optional
        Indicate if the inner products must be computed and stored at the initialization. Default to `True`.

    Attributes
    ----------
    n: float
        The aspect ratio of the domain.
    ocean_inner_products: None or OceanicAnalyticInnerProducts
            The inner products of the ocean. `None` if no ocean.
    connected_to_ocean: bool
        Indicate if the atmosphere is connected to an ocean.
    ground_inner_products: None or GroundAnalyticInnerProducts
            The inner products of the ground. `None` if no ground.
    connected_to_ground: bool
        Indicate if the atmosphere is connected to the ground.
    stored: bool
        Indicate if the inner products are stored in the object.
    atmospheric_wavenumbers: ~numpy.ndarray(WaveNumber)
        An array of shape (:attr:`~.params.QgParams.nmod` [0], ) of the wavenumber object of each mode.
    """

    def __init__(self, params=None, stored=True):

        AtmosphericInnerProducts.__init__(self)

        if params is not None:
            if isinstance(params, QgParams):
                self.n = params.scale_params.n
                self._natm = params.nmod[0]
                ams = params.ablocks
            else:
                self.n = params[0]
                self._natm = params[2]
                ams = params[1]
        else:
            self.n = None
            stored = False
            ams = None

        self.ocean_inner_products = None
        self.connected_to_ocean = False

        self.ground_inner_products = None
        self.connected_to_ground = False

        # Atmospheric wavenumbers definition
        if ams is not None:
            self.atmospheric_wavenumbers = channel_wavenumbers(ams)
        else:
            self.atmospheric_wavenumbers = None

        self.stored = stored
        if stored:
            self.compute_inner_products()

    def connect_to_ocean(self, ocean_inner_products):
        """Connect the atmosphere to an ocean.

        Parameters
        ----------
        ocean_inner_products: OceanicAnalyticInnerProducts
            The inner products of the ocean.
        """

        self.ground_inner_products = None
        self.connected_to_ground = False
        self.ocean_inner_products = ocean_inner_products
        self.connected_to_ocean = True

        if self.stored:
            noc = ocean_inner_products.noc
            self._s = sp.zeros((self.natm, noc), dtype=float, format='dok')
            self._d = sp.zeros((self.natm, noc), dtype=float, format='dok')

            args_list = [(i, j) for i in range(self.natm) for j in range(noc)]

            for arg in args_list:
                # s inner products
                self._s[arg] = self._s_comp(*arg)
                # d inner products
                self._d[arg] = self._d_comp(*arg)

            self._s = self._s.to_coo()
            self._d = self._d.to_coo()

        # ensure that the ocean is connected as well
        if not ocean_inner_products.connected_to_atmosphere:
            ocean_inner_products.connect_to_atmosphere(self)

        if self.stored:
            self.ocean_inner_products = None

    def connect_to_ground(self, ground_inner_products):
        """Connect the atmosphere to the ground.

        Parameters
        ----------
        ground_inner_products: GroundAnalyticInnerProducts
            The inner products of the ground.
        """

        self.ocean_inner_products = None
        self.connected_to_ocean = False
        self.ground_inner_products = ground_inner_products
        self.connected_to_ground = True

        if self.stored:
            ngr = ground_inner_products.ngr
            self._s = sp.zeros((self.natm, ngr), dtype=float, format='dok')
            args_list = [(i, j) for i in range(self.natm) for j in range(ngr)]

            # s inner products
            for arg in args_list:
                self._s[arg] = self._s_comp(*arg)

            self._s = self._s.to_coo()

        # ensure that the ocean is connected as well
        if not ground_inner_products.connected_to_atmosphere:
            ground_inner_products.connect_to_atmosphere(self)

        if self.stored:
            self.ground_inner_products = None

    def compute_inner_products(self):
        """Function computing and storing all the inner products at once."""

        self._a = sp.zeros((self.natm, self.natm), dtype=float, format='dok')
        self._u = sp.zeros((self.natm, self.natm), dtype=float, format='dok')
        self._c = sp.zeros((self.natm, self.natm), dtype=float, format='dok')
        self._b = sp.zeros((self.natm, self.natm, self.natm), dtype=float, format='dok')
        self._g = sp.zeros((self.natm, self.natm, self.natm), dtype=float, format='dok')

        args_list = [(i, j) for i in range(self.natm) for j in range(self.natm)]

        for arg in args_list:
            # a inner products
            self._a[arg] = self._a_comp(*arg)
            # u inner products
            self._u[arg] = self._u_comp(*arg)
            # c inner products
            self._c[arg] = self._c_comp(*arg)

        args_list = [(i, j, k) for i in range(self.natm) for j in range(self.natm) for k in range(self.natm)]

        for arg in args_list:
            # g inner products
            val = self._g_comp(*arg)
            self._g[arg] = val
            # b inner products
            self._b[arg] = val * self._a[arg[-1], arg[-1]]

        self._a = self._a.to_coo()
        self._u = self._u.to_coo()
        self._c = self._c.to_coo()
        self._g = self._g.to_coo()
        self._b = self._b.to_coo()

    @property
    def natm(self):
        """Number of atmospheric modes."""
        return self._natm

    # !-----------------------------------------------------!
    # ! Inner products in the equations for the atmosphere  !
    # !-----------------------------------------------------!

    def a(self, i, j):
        """Function to compute the matrix of the eigenvalues of the Laplacian (atmospheric): :math:`a_{i, j} = (F_i, {\\nabla}^2 F_j)`."""
        if self.stored and self._a is not None:
            return self._a[i, j]
        else:
            return self._a_comp(i, j)

    def _a_comp(self, i, j):
        if i == j:
            n = self.n
            Ti = self.atmospheric_wavenumbers[i]
            return - (n ** 2) * Ti.nx ** 2 - Ti.ny ** 2
        else:
            return 0

    def u(self, i, j):
        """Function to compute the matrix of inner product: :math:`u_{i, j} = (F_i, F_j)`."""
        if self.stored and self._u is not None:
            return self._u[i, j]
        else:
            return self._u_comp(i, j)

    def _u_comp(self, i, j):
        return _delta(i - j)

    def b(self, i, j, k):
        """Function to compute the tensors holding the Jacobian inner products: :math:`b_{i, j, k} = (F_i, J(F_j, \\nabla^2 F_k))`."""
        if self.stored and self._b is not None:
            return self._b[i, j, k]
        else:
            return self._b_comp(i, j, k)

    def _b_comp(self, i, j, k):
        return self._a_comp(k, k) * self._g_comp(i, j, k)

    def c(self, i, j):
        """Function to compute the matrix of beta terms for the atmosphere: :math:`c_{i,j} = (F_i, \\partial_x F_j)`."""
        if self.stored and self._c is not None:
            return self._c[i, j]
        else:
            return self._c_comp(i, j)

    def _c_comp(self, i, j):
        n = self.n
        Ti = self.atmospheric_wavenumbers[i]
        Tj = self.atmospheric_wavenumbers[j]

        val = 0.

        if (Ti.type, Tj.type) == ('K', 'L'):
            val = _delta(Ti.M - Tj.H) * _delta(Ti.P - Tj.P)
            val = n * Ti.M * val
        elif (Ti.type, Tj.type) == ('L', 'K'):
            val = _delta(Tj.M - Ti.H) * _delta(Tj.P - Ti.P)
            val = - n * Tj.M * val

        return val

    def g(self, i, j, k):
        """Function to compute tensors holding the Jacobian inner products: :math:`g_{i,j,k} = (F_i, J(F_j, F_k))`."""
        if self.stored and self._g is not None:
            return self._g[i, j, k]
        else:
            return self._g_comp(i, j, k)

    def _g_comp(self, i, j, k):

        sq2 = np.sqrt(2.)
        pi = np.pi
        n = self.n

        Ti = self.atmospheric_wavenumbers[i]
        Tj = self.atmospheric_wavenumbers[j]
        Tk = self.atmospheric_wavenumbers[k]

        val = 0.
        par = 1

        s = [Ti.type, Tj.type, Tk.type]
        indices = [i, j, k]

        if s == ['L', 'L', 'L']:

            a, par = _piksort(indices)

            Ti = self.atmospheric_wavenumbers[a[0]]
            Tj = self.atmospheric_wavenumbers[a[1]]
            Tk = self.atmospheric_wavenumbers[a[2]]

            vs3 = _S3(Tj.P, Tk.P, Tj.H, Tk.H)
            vs4 = _S4(Tj.P, Tk.P, Tj.H, Tk.H)
            val = vs3 * ((_delta(Tk.H - Tj.H - Ti.H)
                          - _delta(Tk.H - Tj.H + Ti.H))
                         * _delta(Tk.P + Tj.P - Ti.P)
                         + _delta(Tk.H + Tj.H - Ti.H)
                         * (_delta(Tk.P - Tj.P + Ti.P)
                            - _delta(Tk.P - Tj.P - Ti.P))) \
                  + vs4 * ((_delta(Tk.H + Tj.H - Ti.H)
                            * _delta(Tk.P - Tj.P - Ti.P))
                           + (_delta(Tk.H - Tj.H + Ti.H)
                              - _delta(Tk.H - Tj.H - Ti.H))
                           * (_delta(Tk.P - Tj.P - Ti.P)
                              - _delta(Tk.P - Tj.P + Ti.P)))
        else:
            if 'A' in s and 'K' in s and 'L' in s:

                ii = s.index('A')
                jj = s.index('K')
                kk = s.index('L')

                Ti = self.atmospheric_wavenumbers[indices[ii]]
                Tj = self.atmospheric_wavenumbers[indices[jj]]
                Tk = self.atmospheric_wavenumbers[indices[kk]]

                ss, par = _piksort(s)

                vb1 = _B1(Ti.P, Tj.P, Tk.P)
                vb2 = _B2(Ti.P, Tj.P, Tk.P)
                val = -2 * (sq2 / pi) * Tj.M * _delta(Tj.M - Tk.H) \
                      * _flambda(Ti.P + Tj.P + Tk.P)
                if val != 0:
                    val = val * (((vb1 ** 2) / (vb1 ** 2 - 1)) - ((vb2 ** 2)
                                                                  / (vb2 ** 2 - 1)))

            elif 'A' not in s:
                K_indices = [i for i, x in enumerate(s) if x == "K"]
                if len(K_indices) == 2:
                    ss, par = _piksort(s)
                    perm = np.argsort(s)

                    Ti = self.atmospheric_wavenumbers[indices[perm[0]]]
                    Tj = self.atmospheric_wavenumbers[indices[perm[1]]]
                    Tk = self.atmospheric_wavenumbers[indices[perm[2]]]

                    vs1 = _S1(Tj.P, Tk.P, Tj.M, Tk.H)
                    vs2 = _S2(Tj.P, Tk.P, Tj.M, Tk.H)
                    val = vs1 * (_delta(Ti.M - Tk.H - Tj.M)
                                 * _delta(Ti.P - Tk.P + Tj.P)
                                 - _delta(Ti.M - Tk.H - Tj.M)
                                 * _delta(Ti.P + Tk.P - Tj.P)
                                 + (_delta(Tk.H - Tj.M + Ti.M)
                                    + _delta(Tk.H - Tj.M - Ti.M))
                                 * _delta(Tk.P + Tj.P - Ti.P)) \
                          + vs2 * (_delta(Ti.M - Tk.H - Tj.M)
                                   * _delta(Ti.P - Tk.P - Tj.P)
                                   + (_delta(Tk.H - Tj.M - Ti.M)
                                      + _delta(Ti.M + Tk.H - Tj.M))
                                   * (_delta(Ti.P - Tk.P + Tj.P)
                                      - _delta(Tk.P - Tj.P + Ti.P)))

        return val * n * par

    def s(self, i, j):
        """Function to compute the forcing (thermal) of the ocean on the atmosphere: :math:`s_{i,j} = (F_i, \\phi_j)`."""
        if self.stored and self._s is not None:
            return self._s[i, j]
        else:
            return self._s_comp(i, j)

    def _s_comp(self, i, j):
        if self.connected_to_ocean:
            sq2 = np.sqrt(2.)
            pi = np.pi

            Ti = self.atmospheric_wavenumbers[i]
            Dj = self.ocean_inner_products.oceanic_wavenumbers[j]

            val = 0.

            if Ti.type == 'A':
                val = _flambda(Dj.H) * _flambda(Dj.P + Ti.P)
                if val != 0.:
                    val = val * 8 * sq2 * Dj.P / \
                          (pi ** 2 * (Dj.P ** 2 - Ti.P ** 2) * Dj.H)

            if Ti.type == 'K':
                val = _flambda(2 * Ti.M + Dj.H) * _delta(Dj.P - Ti.P)

                if val != 0:
                    val = val * 4 * Dj.H / (pi * (-4 * Ti.M ** 2 + Dj.H ** 2))

            if Ti.type == 'L':
                val = _delta(Dj.P - Ti.P) * _delta(2 * Ti.H - Dj.H)

        elif self.connected_to_ground:
            val = 0
            if i == j:
                val = 1
        else:
            val = 0
        return val

    def d(self, i, j):
        """Function to compute the forcing of the ocean on the atmosphere: :math:`d_{i,j} = (F_i, \\nabla^2 \\phi_j)`."""
        if self.stored and self._d is not None:
            return self._d[i, j]
        else:
            return self._d_comp(i, j)

    def _d_comp(self, i, j):
        if self.connected_to_ocean:
            return self._s_comp(i, j) * self.ocean_inner_products._M_comp(j, j)
        else:
            return 0

    def z(self, i, j, k, l, m):
        pass

    def v(self, i, j, k, l, m):
        pass


class OceanicAnalyticInnerProducts(OceanicInnerProducts):
    """Class which contains all the oceanic inner products coefficients needed for the tendencies
    tensor :class:`~.tensors.qgtensor.QgsTensor` computation, computed with analytic formula.

    Parameters
    ----------
    params: None or QgParams or list, optional
        An instance of model's parameters object or a list in the form [aspect_ratio, oblocks, noc].
        If a list is provided, `aspect_ratio` is the aspect ratio of the domain, `ablocks` is a spectral blocks
        detailing the model's oceanic modes :math:`x`-and :math:`y`-wavenumber as an array of shape (noc, 2), and `noc` is
        the number of modes in the ocean.
        If `None`, an empty object is initialized.
    stored: bool, optional
        Indicate if the inner products must be computed and stored at the initialization. Default to `True`.

    Attributes
    ----------
    n: float
        The aspect ratio of the domain.
    atmosphere_inner_products: None or AtmosphericInnerProducts
            The inner products of the atmosphere. `None` if no atmosphere.
    connected_to_atmosphere: bool
        Indicate if the ocean is connected to an atmosphere.
    stored: bool
        Indicate if the inner products are stored in the object.
    oceanic_wavenumbers: ~numpy.ndarray(WaveNumber)
        An array of shape (:attr:`~.params.QgParams.nmod` [1], ) of the wavenumber object of each mode.

    """

    def __init__(self, params=None, stored=True):

        OceanicInnerProducts.__init__(self)

        if params is not None:
            if isinstance(params, QgParams):
                self.n = params.scale_params.n
                self._noc = params.nmod[1]
                oms = params.oblocks
            else:
                self.n = params[0]
                self._noc = params[2]
                oms = params[1]
        else:
            self.n = None
            stored = False
            oms = None

        self.connected_to_atmosphere = False
        self.atmosphere_inner_products = None

        # Oceanic wavenumbers definition
        if oms is not None:
            self.oceanic_wavenumbers = basin_wavenumbers(oms)
        else:
            self.oceanic_wavenumbers = None

        self.stored = stored
        if stored:
            self.compute_inner_products()

    def connect_to_atmosphere(self, atmosphere_inner_products):
        """Connect the ocean to an atmosphere.

        Parameters
        ----------
        atmosphere_inner_products: AtmosphericAnalyticInnerProducts
            The inner products of the atmosphere.
        """

        self.atmosphere_inner_products = atmosphere_inner_products
        self.connected_to_atmosphere = True

        if self.stored:
            natm = atmosphere_inner_products.natm
            self._K = sp.zeros((self.noc, natm), dtype=float, format='dok')
            self._W = sp.zeros((self.noc, natm), dtype=float, format='dok')

            args_list = [(i, j) for i in range(self.noc) for j in range(natm)]

            for arg in args_list:
                # K inner products
                self._K[arg] = self._K_comp(*arg)
                # W inner products
                self._W[arg] = self._W_comp(*arg)

            self._K = self._K.to_coo()
            self._W = self._W.to_coo()

            self.atmosphere_inner_products = None

    def compute_inner_products(self):
        """Function computing and storing all the inner products at once."""

        self._M = sp.zeros((self.noc, self.noc), dtype=float, format='dok')
        self._U = sp.zeros((self.noc, self.noc), dtype=float, format='dok')
        self._N = sp.zeros((self.noc, self.noc), dtype=float, format='dok')
        self._O = sp.zeros((self.noc, self.noc, self.noc), dtype=float, format='dok')
        self._C = sp.zeros((self.noc, self.noc, self.noc), dtype=float, format='dok')

        args_list = [(i, j) for i in range(self.noc) for j in range(self.noc)]

        for arg in args_list:
            # M inner products
            self._M[arg] = self._M_comp(*arg)
            # U inner products
            self._U[arg] = self._U_comp(*arg)
            # N inner products
            self._N[arg] = self._N_comp(*arg)

        args_list = [(i, j, k) for i in range(self.noc) for j in range(self.noc) for k in range(self.noc)]

        for arg in args_list:
            # O inner products
            val = self._O_comp(*arg)
            self._O[arg] = val
            # C inner products
            self._C[arg] = val * self._M[arg[-1], arg[-1]]

        self._M = self._M.to_coo()
        self._U = self._U.to_coo()
        self._N = self._N.to_coo()
        self._O = self._O.to_coo()
        self._C = self._C.to_coo()

    @property
    def noc(self):
        """Number of oceanic modes."""
        return self._noc

    # !-----------------------------------------------------!
    # ! Inner products in the equations for the ocean       !
    # !-----------------------------------------------------!

    def K(self, i, j):
        """Forcing of the ocean by the atmosphere: :math:`K_{i,j} = (\\phi_i, \\nabla^2 F_j)`."""
        if self.stored and self._K is not None:
            return self._K[i, j]
        else:
            return self._K_comp(i, j)

    def _K_comp(self, i, j):
        if self.connected_to_atmosphere:
            return self.atmosphere_inner_products._s_comp(j, i) * self.atmosphere_inner_products._a_comp(j, j)
        else:
            return 0

    def M(self, i, j):
        """Forcing of the ocean fields on the ocean: :math:`M_{i,j} = (\\phi_i, \\nabla^2 \\phi_j)`."""
        if self.stored and self._M is not None:
            return self._M[i, j]
        else:
            return self._M_comp(i, j)

    def _M_comp(self, i, j):
        if i == j:
            n = self.n
            Di = self.oceanic_wavenumbers[i]
            return - (n ** 2) * Di.nx ** 2 - Di.ny ** 2
        else:
            return 0

    def U(self, i, j):
        """Function to compute the inner products: :math:`U_{i,j} = (\\phi_i, \\phi_j)`."""
        if self.stored and self._U is not None:
            return self._U[i, j]
        else:
            return self._U_comp(i, j)

    def _U_comp(self, i, j):
        return _delta(i - j)

    def N(self, i, j):
        """Function computing the beta term for the ocean: :math:`N_{i,j} = (\\phi_i, \\partial_x \\phi_j)`."""
        if self.stored and self._N is not None:
            return self._N[i, j]
        else:
            return self._N_comp(i, j)

    def _N_comp(self, i, j):
        n = self.n
        pi = np.pi

        Di = self.oceanic_wavenumbers[i]
        Dj = self.oceanic_wavenumbers[j]
        val = _delta(Di.P - Dj.P) * _flambda(Di.H + Dj.H)

        if val != 0:
            val = val * (-2) * Dj.H * Di.H * n / ((Dj.H ** 2 - Di.H ** 2) * pi)

        return val

    def O(self, i, j, k):
        """Function to compute the temperature advection term (passive scalar): :math:`O_{i,j,k} = (\\phi_i, J(\\phi_j, \\phi_k))`"""
        if self.stored and self._O is not None:
            return self._O[i, j, k]
        else:
            return self._O_comp(i, j, k)

    def _O_comp(self, i, j, k):
        n = self.n

        indices = [i, j, k]
        a, par = _piksort(indices)

        Di = self.oceanic_wavenumbers[a[0]]
        Dj = self.oceanic_wavenumbers[a[1]]
        Dk = self.oceanic_wavenumbers[a[2]]

        vs3 = _S3(Dj.P, Dk.P, Dj.H, Dk.H)

        vs4 = _S4(Dj.P, Dk.P, Dj.H, Dk.H)

        val = vs3 * ((_delta(Dk.H - Dj.H - Di.H)
                      - _delta(Dk.H - Dj.H + Di.H))
                     * _delta(Dk.P + Dj.P - Di.P)
                     + _delta(Dk.H + Dj.H - Di.H)
                     * (_delta(Dk.P - Dj.P + Di.P)
                        - _delta(Dk.P - Dj.P - Di.P))) \
              + vs4 * ((_delta(Dk.H + Dj.H - Di.H)
                        * _delta(Dk.P - Dj.P - Di.P))
                       + (_delta(Dk.H - Dj.H + Di.H)
                          - _delta(Dk.H - Dj.H - Di.H))
                       * (_delta(Dk.P - Dj.P - Di.P)
                          - _delta(Dk.P - Dj.P + Di.P)))

        return par * val * n / 2

    def C(self, i, j, k):
        """Function to compute the tensors holding the Jacobian inner products: :math:`C_{i,j,k} = (\\phi_i, J(\\phi_j,\\nabla^2 \\phi_k))`."""
        if self.stored and self._C is not None:
            return self._C[i, j, k]
        else:
            return self._C_comp(i, j, k)

    def _C_comp(self, i, j, k):
        return self._M_comp(k, k) * self._O_comp(i, j, k)

    def W(self, i, j):
        """Function to compute the short-wave radiative forcing of the ocean: :math:`W_{i,j} = (\\phi_i, F_j)`."""
        if self.stored and self._W is not None:
            return self._W[i, j]
        else:
            return self._W_comp(i, j)

    def _W_comp(self, i, j):
        if self.connected_to_atmosphere:
            return self.atmosphere_inner_products._s_comp(j, i)
        else:
            return 0

    def Z(self, i, j, k, l, m):
        pass

    def V(self, i, j, k, l, m):
        pass


class GroundAnalyticInnerProducts(GroundInnerProducts):
    """Class which contains all the ground inner products coefficients needed for the tendencies
    tensor :class:`~.tensors.qgtensor.QgsTensor` computation, computed with analytic formula.

    Parameters
    ----------
    params: None or QgParams or list, optional
        An instance of model's parameters object or a list in the form [aspect_ratio, gblocks, ngr].
        If a list is provided, `aspect_ratio` is the aspect ratio of the domain, `gblocks` is a spectral blocks
        detailing the model's oceanic modes :math:`x`-and :math:`y`-wavenumber as an array of shape (ngr, 2), and `ngr` is
        the number of modes in the ocean.
        If `None`, an empty object is initialized.
    stored: bool, optional
        Indicate if the inner products must be computed and stored at the initialization. Default to `True`.

    Attributes
    ----------
    n: float
        The aspect ratio of the domain.
    atmosphere_inner_products: None or AtmosphericInnerProducts
            The inner products of the atmosphere. `None` if no atmosphere.
    connected_to_atmosphere: bool
        Indicate if the ocean is connected to an atmosphere.
    stored: bool
        Indicate if the inner products are stored in the object.
    ground_wavenumbers: ~numpy.ndarray(WaveNumber)
        An array of shape (:attr:`~.params.QgParams.nmod` [1], ) of the wavenumber object of each mode.

    """

    def __init__(self, params=None, stored=True):

        GroundInnerProducts.__init__(self)

        if params is not None:
            if isinstance(params, QgParams):
                self.n = params.scale_params.n
                self._ngr = params.nmod[1]
                gms = params.oblocks
            else:
                self.n = params[0]
                self._ngr = params[2]
                gms = params[1]
        else:
            self.n = None
            stored = False
            gms = None

        self.connected_to_atmosphere = False
        self.atmosphere_inner_products = None

        # Ground wavenumbers definition
        if gms is not None:
            self.ground_wavenumbers = channel_wavenumbers(gms)
        else:
            self.ground_wavenumbers = None

        self.stored = stored

        if stored:
            self.compute_inner_products()

    def compute_inner_products(self):
        """Function computing and storing all the inner products at once."""
        self._U = sp.zeros((self.ngr, self.ngr), dtype=float, format='dok')

        args_list = [(i, j) for i in range(self.ngr) for j in range(self.ngr)]

        for arg in args_list:
            # U inner products
            self._U[arg] = self._U_comp(*arg)

        self._U = self._U.to_coo()

    def connect_to_atmosphere(self, atmosphere_inner_products):
        """Connect the ground to an atmosphere.

        Parameters
        ----------
        atmosphere_inner_products: AtmosphericAnalyticInnerProducts
            The inner products of the atmosphere.
        """

        self.atmosphere_inner_products = atmosphere_inner_products
        self.connected_to_atmosphere = True

        if self.stored:
            natm = atmosphere_inner_products.natm
            self._W = sp.zeros((self.ngr, natm), dtype=float, format='dok')

            args_list = [(i, j) for i in range(self.ngr) for j in range(natm)]

            for arg in args_list:
                # W inner products
                self._W[arg] = self._W_comp(*arg)

            self._W = self._W.to_coo()

            self.atmosphere_inner_products = None

    @property
    def ngr(self):
        """Number of ground modes."""
        return self._ngr

    # !-----------------------------------------------------!
    # ! Inner products in the equations for the ground      !
    # !-----------------------------------------------------!

    def K(self, i, j):
        """:math:`K_{i,j} = (\\phi_i, \\nabla^2 F_j)`

        Warnings
        --------
        Not defined and not used."""
        return 0

    def M(self, i, j):
        """:math:`M_{i,j} = (\\phi_i, \\nabla^2 \\phi_j)`

        Warnings
        --------
        Not defined and not used.
        """
        return 0

    def U(self, i, j):
        """Function to compute the inner products: :math:`U_{i,j} = (\\phi_i, \\phi_j)`."""
        if self.stored and self._U is not None:
            return self._U[i, j]
        else:
            return self._U_comp(i, j)

    def _U_comp(self, i, j):
        return _delta(i - j)

    def N(self, i, j):
        """:math:`N_{i,j} = (\\phi_i, \\partial_x \\phi_j)`

        Warnings
        --------
        Not defined and not used.
        """
        return 0

    def O(self, i, j, k):
        """:math:`O_{i,j,k} = (\\phi_i, J(\\phi_j, \\phi_k))`

        Warnings
        --------
        Not defined and not used.
        """
        return 0

    def C(self, i, j, k):
        """:math:`C_{i,j,k} = (\\phi_i, J(\\phi_j,\\nabla^2 \\phi_k))`

        Warnings
        --------
        Not defined and not used.
        """
        return 0

    def W(self, i, j):
        """Function to compute the short-wave radiative forcing of the ground: :math:`W_{i,j} = (\\phi_i, F_j)`."""
        if self.stored and self._W is not None:
            return self._W[i, j]
        else:
            return self._W_comp(i, j)

    def _W_comp(self, i, j):
        if self.connected_to_atmosphere:
            return self.atmosphere_inner_products._s_comp(j, i)
        else:
            return 0

    def V(self, i, j, k, l, m):
        pass

    def Z(self, i, j, k, l, m):
        pass


def _piksort(arr):
    k = len(arr)
    arro = arr.copy()

    par = 1

    for i in range(1, k):
        a = arro[i]
        l = i - 1
        for j in range(i - 1, -1, -1):
            if arro[j] <= a:
                l = j
                break
            arro[j + 1] = arro[j]
            par = -par
            l = j - 1
        arro[l + 1] = a
    return arro, par


#  !-----------------------------------------------------!
#  !                                                     !
#  ! Definition of the Helper functions from Cehelsky    !
#  ! \ Tung                                              !
#  !                                                     !
#  !-----------------------------------------------------!


def _B1(Pi, Pj, Pk):
    return (Pk + Pj) / float(Pi)


def _B2(Pi, Pj, Pk):
    return (Pk - Pj) / float(Pi)


def _delta(r):
    if r == 0:
        return 1.

    else:
        return 0.


def _flambda(r):
    if r % 2 == 0:
        return 0.

    else:
        return 1.


def _S1(Pj, Pk, Mj, Hk):
    return -(Pk * Mj + Pj * Hk) / 2.


def _S2(Pj, Pk, Mj, Hk):
    return (Pk * Mj - Pj * Hk) / 2.


def _S3(Pj, Pk, Hj, Hk):
    return (Pk * Hj + Pj * Hk) / 2.


def _S4(Pj, Pk, Hj, Hk):
    return (Pk * Hj - Pj * Hk) / 2.


if __name__ == '__main__':
    from qgs.params.params import QgParams

    pars = QgParams()
    pars._set_atmospheric_analytic_fourier_modes(2, 2)
    pars._set_oceanic_analytic_fourier_modes(2, 4)
    aip = AtmosphericAnalyticInnerProducts(pars)
    oip = OceanicAnalyticInnerProducts(pars)
    aip.connect_to_ocean(oip)
