"""
    Symbolic Inner products module
    ==============================

    Inner products between the truncated set of basis functions :math:`\phi_i` for the ocean and land fields, and
    :math:`F_i` for the atmosphere fields (see :ref:`files/model/oro_model:Projecting the equations on a set of basis functions`).

    Notes
    -----

    These inner products are computed symbolically using `Sympy`_, and thus support arbitrary (but symbolic) basis functions.

    Description of the classes
    --------------------------

    The three classes computing and holding the inner products of the basis functions are:

    * :class:`AtmosphericSymbolicInnerProducts`
    * :class:`OceanicSymbolicInnerProducts`
    * :class:`GroundSymbolicInnerProducts`

    .. _Sympy: https://www.sympy.org/
"""

import sparse as sp
import multiprocessing

from qgs.params.params import QgParams
from qgs.inner_products.base import AtmosphericInnerProducts, OceanicInnerProducts, GroundInnerProducts
from qgs.inner_products.definition import StandardSymbolicInnerProductDefinition
from sympy import symbols

# TODO: - Add warnings if trying to connect analytic and symbolic inner products together
#       - Switch to numerical integration of the inner products if the symbolic one is to long (to be done when NumericBasis is ready)


class AtmosphericSymbolicInnerProducts(AtmosphericInnerProducts):
    """Class which contains all the atmospheric inner products coefficients needed for the tendencies
    tensor :class:`~tensors.qgtensor.QgsTensor` computation, computed with analytic formula.

    Parameters
    ----------
    params: ~.params.QgParams or list
        An instance of model's parameters object or a list in the form [aspect_ratio, atmospheric_basis, basis, oog, oro_basis].
        If a list is provided, `aspect_ratio` is the aspect ratio of the domain, `atmospheric_basis` is a SymbolicBasis with
        the modes of the atmosphere, and `ocean_basis` is either `None` or a SymbolicBasis object with the modes of
        the ocean or the ground. Finally `oog` indicates if it is an ocean or a ground component that is connected,
        by setting it to `ocean` or to 'ground', and in this latter case, `oro_basis` indicates on which basis the orography is developed.
    stored: bool, optional
        Indicate if the inner product must be stored or computed on the fly. Default to `True`
    inner_product_definition: None or InnerProductDefinition, optional
        The definition of the inner product being used. If `None`, use the canonical StandardInnerProductDefinition object.
        Default to `None`.
    interaction_inner_product_definition: None or InnerProductDefinition, optional
        The definition of the inner product being used for the interaction with the other components, i.e. to compute the inner products with the other component base of funcitons.
        If `None`, use the `inner_product_definition` provided.
        Default to `None`.
    num_threads: int or None, optional
        Number of threads to use to compute the symbolic inner products. If `None` use all the cpus available.
        Default to `None`.

    Attributes
    ----------
    n: float
        The aspect ratio of the domain.
    atmospheric_basis: SymbolicBasis
        Object holding the symbolic modes of the atmosphere.
    oceanic_basis: None or SymbolicBasis
        Object holding the symbolic modes of the ocean (or `None` if there is no ocean).
    connected_to_ocean: bool
        Indicate if the atmosphere is connected to an ocean.
    stored: bool
        Indicate if the inner product must be stored or computed on the fly.
    ip: InnerProductDefinition
        Object defining the inner product.
    iip: InnerProductDefinition
        Object defining the interaction inner product.
    subs: list(tuple)
        List of 2-tuples containing the substitutions to be made with the functions after the inner products
        symbolic computation.
    """

    def __init__(self, params=None, stored=True, inner_product_definition=None, interaction_inner_product_definition=None, num_threads=None):

        AtmosphericInnerProducts.__init__(self)

        if params is not None:
            if isinstance(params, QgParams):
                self.n = params.scale_params.n
                self.atmospheric_basis = params.atmospheric_basis
                if params.oceanic_basis is not None:
                    goc_basis = params.oceanic_basis
                    oog = "ocean"
                elif params.ground_basis is not None:
                    goc_basis = params.ground_basis
                    oog = "ground"
                    oro_basis = params.ground_params.orographic_basis
                else:
                    goc_basis = None
                    oog = ""
                    oro_basis = params.ground_params.orographic_basis
            else:
                self.n = params[0]
                self.atmospheric_basis = params[1]
                goc_basis = params[2]
                oog = params[3]
                oro_basis = params[4]
            self._gh = None
        else:  # initialize an empty inner product object
            self.n = None
            self.atmospheric_basis = None
            goc_basis = None
            oog = ""
            self._gh = None
            stored = False
            oro_basis = ""

        self.oceanic_basis = None
        self.connected_to_ocean = False

        self.ground_basis = None
        self.connected_to_ground = False

        self.subs = [('n', self.n)]

        if inner_product_definition is None:
            self.ip = StandardSymbolicInnerProductDefinition()
        else:
            self.ip = inner_product_definition

        if interaction_inner_product_definition is None:
            self.iip = self.ip
        else:
            self.iip = interaction_inner_product_definition

        self.stored = stored
        if stored:
            self.compute_inner_products(num_threads)

        if goc_basis is not None:
            if oog == 'ocean':
                self.connect_to_ocean(goc_basis)
            else:
                self.connect_to_ground(goc_basis, oro_basis)

    def _F(self, i):
        if self.atmospheric_basis is not None:
            return self.atmospheric_basis.functions[i]

    def _phi(self, i):
        if self.oceanic_basis is not None:
            return self.oceanic_basis.functions[i]

    def connect_to_ocean(self, ocean_basis, num_threads=None):
        """Connect the atmosphere to an ocean.

        Parameters
        ----------
        ocean_basis: SymbolicBasis or OceanicSymbolicInnerProducts
            Basis of function of the ocean or a symbolic oceanic inner products object containing the basis.
        num_threads: int or None, optional
            Number of threads to use to compute the symbolic inner products. If `None` use all the cpus available.
            Default to `None`.
        """
        if isinstance(ocean_basis, OceanicSymbolicInnerProducts):
            ocean_basis = ocean_basis.oceanic_basis
        self.ground_basis = None
        self.connected_to_ground = False

        self.oceanic_basis = ocean_basis
        self.connected_to_ocean = True

        if self.stored:
            if num_threads is None:
                pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
            else:
                pool = multiprocessing.Pool(processes=num_threads)

            noc = len(ocean_basis)
            self._gh = None
            self._d = sp.zeros((self.natm, noc), dtype=float, format='dok')
            self._s = sp.zeros((self.natm, noc), dtype=float, format='dok')

            # d inner products
            args_list = [[(i, j), self.iip.ip_lap, (self._F(i), self._phi(j))] for i in range(self.natm)
                         for j in range(noc)]

            result = pool.map(_apply, args_list)

            for res in result:
                self._d[res[0]] = float(res[1].subs(self.subs).subs(self.atmospheric_basis.substitutions)
                                        .subs(self.oceanic_basis.substitutions))

            # s inner products
            args_list = [[(i, j), self.iip.symbolic_inner_product, (self._F(i), self._phi(j))] for i in range(self.natm)
                         for j in range(noc)]

            result = pool.map(_apply, args_list)

            for res in result:
                self._s[res[0]] = float(res[1].subs(self.subs).subs(self.atmospheric_basis.substitutions)
                                        .subs(self.oceanic_basis.substitutions))

            self._s = self._s.to_coo()
            self._d = self._d.to_coo()

            pool.terminate()

    def connect_to_ground(self, ground_basis, orographic_basis, num_threads=None):
        """Connect the atmosphere to the ground.

        Parameters
        ----------
        ground_basis: SymbolicBasis or GroundSymbolicInnerProducts
            Basis of function of the ground or a symbolic ground inner products object containing the basis.
        orographic_basis: str
            String to select which component basis modes to use to develop the orography in series.
            Can be either 'atmospheric' or 'ground'.
        num_threads: int or None, optional
            Number of threads to use to compute the symbolic inner products. If `None` use all the cpus available.
            Default to `None`.
        """

        if isinstance(ground_basis, GroundSymbolicInnerProducts):
            ground_basis = ground_basis.ground_basis

        self.oceanic_basis = None
        self.connected_to_ocean = False

        self.ground_basis = ground_basis
        self.connected_to_ground = True

        if self.stored:
            if num_threads is None:
                pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
            else:
                pool = multiprocessing.Pool(processes=num_threads)

            ngr = len(ground_basis)
            if orographic_basis == "atmospheric":
                self._gh = None
            else:
                self._gh = sp.zeros((self.natm, self.natm, ngr), dtype=float, format='dok')
            self._d = None
            self._s = sp.zeros((self.natm, ngr), dtype=float, format='dok')

            # s inner products
            args_list = [[(i, j), self.iip.symbolic_inner_product, (self._F(i), self._phi(j))] for i in range(self.natm)
                         for j in range(ngr)]

            result = pool.map(_apply, args_list)

            for res in result:
                self._s[res[0]] = float(res[1].subs(self.subs).subs(self.atmospheric_basis.substitutions)
                                        .subs(self.ground_basis.substitutions))

            # gh inner products
            args_list = [[(i, j, k), self.iip.ip_jac, (self._F(i), self._F(j), self._phi(k))] for i in range(self.natm)
                         for j in range(self.natm) for k in range(ngr)]

            result = pool.map(_apply, args_list)

            for res in result:
                self._gh[res[0]] = float(res[1].subs(self.subs).subs(self.atmospheric_basis.substitutions)
                                         .subs(self.ground_basis.substitutions))

            self._s = self._s.to_coo()
            if self._gh is not None:
                self._gh = self._gh.to_coo()

            pool.terminate()

    def compute_inner_products(self, num_threads=None):
        """Function computing and storing all the inner products at once.

        Parameters
        ----------
        num_threads: int or None, optional
            Number of threads to use to compute the symbolic inner products. If `None` use all the cpus available.
            Default to `None`.
        """

        self._a = sp.zeros((self.natm, self.natm), dtype=float, format='dok')
        self._u = sp.zeros((self.natm, self.natm), dtype=float, format='dok')
        self._c = sp.zeros((self.natm, self.natm), dtype=float, format='dok')
        self._b = sp.zeros((self.natm, self.natm, self.natm), dtype=float, format='dok')
        self._g = sp.zeros((self.natm, self.natm, self.natm), dtype=float, format='dok')

        if num_threads is None:
            pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
        else:
            pool = multiprocessing.Pool(processes=num_threads)

        # a inner products
        args_list = [[(i, j), self.ip.ip_lap, (self._F(i), self._F(j))] for i in range(self.natm)
                     for j in range(self.natm)]
        result = pool.map(_apply, args_list)

        for res in result:
            self._a[res[0]] = float(res[1].subs(self.subs).subs(self.atmospheric_basis.substitutions))

        # u inner products
        args_list = [[(i, j), self.ip.symbolic_inner_product, (self._F(i), self._F(j))] for i in range(self.natm)
                     for j in range(self.natm)]
        result = pool.map(_apply, args_list)

        for res in result:
            self._u[res[0]] = float(res[1].subs(self.subs).subs(self.atmospheric_basis.substitutions))

        # c inner products
        args_list = [[(i, j), self.ip.ip_diff_x, (self._F(i), self._F(j))] for i in range(self.natm)
                     for j in range(self.natm)]
        result = pool.map(_apply, args_list)

        for res in result:
            self._c[res[0]] = float(res[1].subs(self.subs).subs(self.atmospheric_basis.substitutions))

        # b inner products
        args_list = [[(i, j, k), self.ip.ip_jac_lap, (self._F(i), self._F(j), self._F(k))] for i in range(self.natm)
                     for j in range(self.natm) for k in range(self.natm)]

        result = pool.map(_apply, args_list)

        for res in result:
            self._b[res[0]] = float(res[1].subs(self.subs).subs(self.atmospheric_basis.substitutions))

        # g inner products
        args_list = [[(i, j, k), self.ip.ip_jac, (self._F(i), self._F(j), self._F(k))] for i in range(self.natm)
                     for j in range(self.natm) for k in range(self.natm)]

        result = pool.map(_apply, args_list)

        for res in result:
            self._g[res[0]] = float(res[1].subs(self.subs).subs(self.atmospheric_basis.substitutions))

        self._a = self._a.to_coo()
        self._u = self._u.to_coo()
        self._c = self._c.to_coo()
        self._g = self._g.to_coo()
        self._b = self._b.to_coo()

        pool.terminate()

    @property
    def natm(self):
        """Number of atmospheric modes."""
        return len(self.atmospheric_basis.functions)

    # !-----------------------------------------------------!
    # ! Inner products in the equations for the atmosphere  !
    # !-----------------------------------------------------!

    def a(self, i, j):
        """Function to compute the matrix of the eigenvalues of the Laplacian (atmospheric): :math:`a_{i, j} = (F_i, {\\nabla}^2 F_j)`."""
        if not self.stored:
            res = self.ip.ip_lap(self._F(i), self._F(j))
            return float(res.subs(self.subs).subs(self.atmospheric_basis.substitutions))
        else:
            return self._a[i, j]

    def u(self, i, j):
        """Function to compute the matrix of inner product: :math:`u_{i, j} = (F_i, F_j)`."""
        if not self.stored:
            res = self.ip.symbolic_inner_product(self._F(i), self._F(j))
            return float(res.subs(self.subs).subs(self.atmospheric_basis.substitutions))
        else:
            return self._u[i, j]

    def b(self, i, j, k):
        """Function to compute the tensors holding the Jacobian inner products: :math:`b_{i, j, k} = (F_i, J(F_j, \\nabla^2 F_k))`."""
        if not self.stored:
            res = self.ip.ip_jac_lap(self._F(i), self._F(j), self._F(k))
            return float(res.subs(self.subs).subs(self.atmospheric_basis.substitutions))
        else:
            return self._b[i, j, k]

    def c(self, i, j):
        """Function to compute the matrix of beta terms for the atmosphere: :math:`c_{i,j} = (F_i, \partial_x F_j)`."""
        if not self.stored:
            res = self.ip.ip_diff_x(self._F(i), self._F(j))
            return float(res.subs(self.subs).subs(self.atmospheric_basis.substitutions))
        else:
            return self._c[i, j]

    def g(self, i, j, k):
        """Function to compute tensors holding the Jacobian inner products: :math:`g_{i,j,k} = (F_i, J(F_j, F_k))`."""
        if not self.stored:
            res = self.ip.ip_jac(self._F(i), self._F(j), self._F(k))
            return float(res.subs(self.subs).subs(self.atmospheric_basis.substitutions))
        else:
            return self._g[i, j, k]

    def gh(self, i, j, k):
        """Function to compute tensors holding the Jacobian inner products: :math:`g_{i,j,k} = (F_i, J(F_j, phi_k))`."""
        if self.connected_to_ocean:
            extra_subs = self.oceanic_basis.substitutions
        elif self.connected_to_ground:
            extra_subs = self.ground_basis.substitutions
        else:
            extra_subs = None

        if self.oceanic_basis or self.connected_to_ground:
            if self.stored and self._gh is not None:
                return self._gh[i, j, k]
            else:
                res = self.iip.ip_jac(self._F(i), self._F(j), self._phi(k))
                return float(res.subs(self.subs).subs(self.atmospheric_basis.substitutions).subs(extra_subs))

        else:
            return 0

    def s(self, i, j):
        """Function to compute the forcing (thermal) of the ocean on the atmosphere: :math:`s_{i,j} = (F_i, \phi_j)`."""
        if self.connected_to_ocean:
            extra_subs = self.oceanic_basis.substitutions
        elif self.connected_to_ground:
            extra_subs = self.ground_basis.substitutions
        else:
            extra_subs = None

        if self.connected_to_ocean or self.connected_to_ground:
            if self.stored and self._s is not None:
                return self._s[i, j]
            else:
                res = self.iip.symbolic_inner_product(self._F(i), self._phi(j))
                return float(res.subs(self.subs).subs(self.atmospheric_basis.substitutions).subs(extra_subs))
        else:
            return 0

    def d(self, i, j):
        """Function to compute the forcing of the ocean on the atmosphere: :math:`d_{i,j} = (F_i, \\nabla^2 \phi_j)`."""
        if self.connected_to_ocean:
            extra_subs = self.oceanic_basis.substitutions
        elif self.connected_to_ground:
            extra_subs = self.ground_basis.substitutions
        else:
            extra_subs = None

        if self.connected_to_ocean or self.connected_to_ground:
            if self.stored and self._d is not None:
                return self._d[i, j]
            else:
                res = self.iip.ip_lap(self._F(i), self._phi(j))
                return float(res.subs(self.subs).subs(self.atmospheric_basis.substitutions).subs(extra_subs))
        else:
            return 0


class OceanicSymbolicInnerProducts(OceanicInnerProducts):
    """Class which contains all the oceanic inner products coefficients needed for the tendencies
    tensor :class:`~tensors.qgtensor.QgsTensor` computation.

    Parameters
    ----------
    params: ~.params.QgParams or list
        An instance of model's parameters object or a list in the form [aspect_ratio, ocean_basis, atmospheric_basis].
        If a list is provided, `aspect_ratio` is the aspect ratio of the domain, `ocean_basis` is a SymbolicBasis object
        with the modes of the ocean, and `atmospheric_basis` is either a SymbolicBasis with the modes of the atmosphere
        or `None` if there is no atmosphere.
    stored: bool, optional
        Indicate if the inner product must be stored or computed on the fly. Default to `True`
    inner_product_definition: None or InnerProductDefinition, optional
        The definition of the inner product being used. If `None`, use the canonical StandardInnerProductDefinition object.
        Default to `None`.
    interaction_inner_product_definition: None or InnerProductDefinition, optional
        The definition of the inner product being used for the interaction with the other components, i.e. to compute the inner products with the other component base of funcitons.
        If `None`, use the `inner_product_definition` provided.
        Default to `None`.
    num_threads: int or None, optional
        Number of threads to use to compute the symbolic inner products. If `None` use all the cpus available.
        Default to `None`.

    Attributes
    ----------
    n: float
        The aspect ratio of the domain.
    oceanic_basis: SymbolicBasis
        Object holding the symbolic modes of the ocean.
    atmospheric_basis: None or SymbolicBasis
        Object holding the symbolic modes of the atmosphere (or `None` if there is no atmosphere).
    connected_to_atmosphere: bool
        Indicate if the ocean is connected to an atmosphere.
    stored: bool
        Indicate if the inner product must be stored or computed on the fly.
    ip: InnerProductDefinition
        Object defining the inner product.
    iip: InnerProductDefinition
        Object defining the interaction inner product.
    subs: list(tuple)
        List of 2-tuples containing the substitutions to be made with the functions after the inner products
        symbolic computation.
    """
    def __init__(self, params=None, stored=True, inner_product_definition=None, interaction_inner_product_definition=None, num_threads=None):

        OceanicInnerProducts.__init__(self)

        if params is not None:
            if isinstance(params, QgParams):
                self.n = params.scale_params.n
                self.oceanic_basis = params.oceanic_basis
                atm_basis = params.atmospheric_basis
            else:
                self.n = params[0]
                self.oceanic_basis = params[1]
                atm_basis = params[2]
        else:
            self.n = None
            self.oceanic_basis = None
            atm_basis = None
            stored = False

        self.atmospheric_basis = None
        self.connected_to_atmosphere = False

        self.subs = [('n', self.n)]

        if inner_product_definition is None:
            self.ip = StandardSymbolicInnerProductDefinition()
        else:
            self.ip = inner_product_definition

        if interaction_inner_product_definition is None:
            self.iip = self.ip
        else:
            self.iip = interaction_inner_product_definition

        self.stored = stored
        if stored:
            self.compute_inner_products(num_threads)

        if atm_basis is not None:
            self.connect_to_atmosphere(atm_basis)

    def _F(self, i):
        if self.atmospheric_basis is not None:
            return self.atmospheric_basis[i]

    def _phi(self, i):
        if self.oceanic_basis is not None:
            return self.oceanic_basis[i]

    def connect_to_atmosphere(self, atmosphere_basis, num_threads=None):
        """Connect the ocean to an atmosphere.

        Parameters
        ----------
        atmosphere_basis: SymbolicBasis or AtmosphericSymbolicInnerProducts
            Basis of function of the atmosphere or a symbolic atmospheric inner products object containing the basis.
        num_threads: int or None, optional
            Number of threads to use to compute the symbolic inner products. If `None` use all the cpus available.
            Default to `None`.
        """

        if isinstance(atmosphere_basis, AtmosphericSymbolicInnerProducts):
            atmosphere_basis = atmosphere_basis.atmospheric_basis
        self.atmospheric_basis = atmosphere_basis
        self.connected_to_atmosphere = True

        if self.stored:
            if num_threads is None:
                pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
            else:
                pool = multiprocessing.Pool(processes=num_threads)
            natm = len(atmosphere_basis)
            self._K = sp.zeros((self.noc, natm), dtype=float, format='dok')
            self._W = sp.zeros((self.noc, natm), dtype=float, format='dok')

            # K inner products
            l = [[(i, j), self.iip.ip_lap, (self._phi(i), self._F(j))] for i in range(self.noc)
                 for j in range(natm)]

            result = pool.map(_apply, l)

            for res in result:
                self._K[res[0]] = float(res[1].subs(self.subs).subs(self.oceanic_basis.substitutions)
                                        .subs(self.atmospheric_basis.substitutions))

            # W inner products
            l = [[(i, j), self.iip.symbolic_inner_product, (self._phi(i), self._F(j))] for i in range(self.noc)
                 for j in range(natm)]

            result = pool.map(_apply, l)

            for res in result:
                self._W[res[0]] = float(res[1].subs(self.subs).subs(self.oceanic_basis.substitutions)
                                        .subs(self.atmospheric_basis.substitutions))

            self._K = self._K.to_coo()
            self._W = self._W.to_coo()

            pool.terminate()

    def compute_inner_products(self, num_threads=None):
        """Function computing and storing all the inner products at once.

        Parameters
        ----------
        num_threads: int or None, optional
            Number of threads to use to compute the symbolic inner products. If `None` use all the cpus available.
            Default to `None`.
        """

        self._M = sp.zeros((self.noc, self.noc), dtype=float, format='dok')
        self._U = sp.zeros((self.noc, self.noc), dtype=float, format='dok')
        self._N = sp.zeros((self.noc, self.noc), dtype=float, format='dok')
        self._O = sp.zeros((self.noc, self.noc, self.noc), dtype=float, format='dok')
        self._C = sp.zeros((self.noc, self.noc, self.noc), dtype=float, format='dok')

        if num_threads is None:
            pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
        else:
            pool = multiprocessing.Pool(processes=num_threads)

        # N inner products
        l = [[(i, j), self.ip.ip_diff_x, (self._phi(i), self._phi(j))] for i in range(self.noc) for j in range(self.noc)]
        result = pool.map(_apply, l)

        for res in result:
            self._N[res[0]] = float(res[1].subs(self.subs).subs(self.oceanic_basis.substitutions))

        # M inner products
        l = [[(i, j), self.ip.ip_lap, (self._phi(i), self._phi(j))] for i in range(self.noc) for j in range(self.noc)]
        result = pool.map(_apply, l)

        for res in result:
            self._M[res[0]] = float(res[1].subs(self.subs).subs(self.oceanic_basis.substitutions))

        # U inner products
        l = [[(i, j), self.ip.symbolic_inner_product, (self._phi(i), self._phi(j))] for i in range(self.noc) for j in range(self.noc)]
        result = pool.map(_apply, l)

        for res in result:
            self._U[res[0]] = float(res[1].subs(self.subs).subs(self.oceanic_basis.substitutions))

        # O inner products
        l = [[(i, j, k), self.ip.ip_jac, (self._phi(i), self._phi(j), self._phi(k))] for i in range(self.noc)
             for j in range(self.noc) for k in range(self.noc)]

        result = pool.map(_apply, l)

        for res in result:
            self._O[res[0]] = float(res[1].subs(self.subs).subs(self.oceanic_basis.substitutions))

        # C inner products
        l = [[(i, j, k), self.ip.ip_jac_lap, (self._phi(i), self._phi(j), self._phi(k))] for i in range(self.noc)
             for j in range(self.noc) for k in range(self.noc)]

        result = pool.map(_apply, l)

        for res in result:
            self._C[res[0]] = float(res[1].subs(self.subs).subs(self.oceanic_basis.substitutions))

        self._M = self._M.to_coo()
        self._U = self._U.to_coo()
        self._N = self._N.to_coo()
        self._O = self._O.to_coo()
        self._C = self._C.to_coo()

        pool.terminate()

    @property
    def noc(self):
        """Number of oceanic modes."""
        return len(self.oceanic_basis)

    # !-----------------------------------------------------!
    # ! Inner products in the equations for the ocean       !
    # !-----------------------------------------------------!

    def M(self, i, j):
        """Function to compute the forcing of the ocean fields on the ocean: :math:`M_{i,j} = (\phi_i, \\nabla^2 \phi_j)`."""
        if not self.stored:
            res = self.ip.ip_lap(self._phi(i), self._phi(j))
            return float(res.subs(self.subs).subs(self.oceanic_basis.substitutions))
        else:
            return self._M[i, j]

    def U(self, i, j):
        """Function to compute the inner products: :math:`U_{i,j} = (\phi_i, \phi_j)`."""
        if not self.stored:
            res = self.ip.symbolic_inner_product(self._phi(i), self._phi(j))
            return float(res.subs(self.subs).subs(self.oceanic_basis.substitutions))
        else:
            return self._U[i, j]

    def N(self, i, j):
        """Function computing the beta term for the ocean: :math:`N_{i,j} = (\phi_i, \partial_x \phi_j)`."""
        if not self.stored:
            res = self.ip.ip_diff_x(self._phi(i), self._phi(j))
            return float(res.subs(self.subs).subs(self.oceanic_basis.substitutions))
        else:
            return self._N[i, j]

    def O(self, i, j, k):
        """Function to compute the temperature advection term (passive scalar): :math:`O_{i,j,k} = (\phi_i, J(\phi_j, \phi_k))`"""
        if not self.stored:
            res = self.ip.ip_jac(self._phi(i), self._phi(j), self._phi(k))
            return float(res.subs(self.subs).subs(self.oceanic_basis.substitutions))
        else:
            return self._O[i, j, k]

    def C(self, i, j, k):
        """Function to compute the tensors holding the Jacobian inner products: :math:`C_{i,j,k} = (\phi_i, J(\phi_j,\\nabla^2 \phi_k))`."""
        if not self.stored:
            res = self.ip.ip_jac_lap(self._phi(i), self._phi(j), self._phi(k))
            return float(res.subs(self.subs).subs(self.oceanic_basis.substitutions))
        else:
            return self._C[i, j, k]

    def K(self, i, j):
        """Function to commpute the forcing of the ocean by the atmosphere: :math:`K_{i,j} = (\phi_i, \\nabla^2 F_j)`."""
        if self.connected_to_atmosphere:
            if not self.stored:
                res = self.iip.ip_lap(self._phi(i), self._F(j))
                return float(res.subs(self.subs).subs(self.oceanic_basis.substitutions).subs(self.atmospheric_basis.substitutions))
            else:
                return self._K[i, j]
        else:
            return 0

    def W(self, i, j):
        """Function to compute the short-wave radiative forcing of the ocean: :math:`W_{i,j} = (\phi_i, F_j)`."""
        if self.connected_to_atmosphere:
            if not self.stored:
                res = self.iip.symbolic_inner_product(self._phi(i), self._F(j))
                return float(res.subs(self.subs).subs(self.atmospheric_basis.substitutions).subs(self.oceanic_basis.substitutions))
            else:
                return self._W[i, j]
        else:
            return 0


class GroundSymbolicInnerProducts(GroundInnerProducts):
    """Class which contains all the ground inner products coefficients needed for the tendencies
    tensor :class:`~tensors.qgtensor.QgsTensor` computation.

    Parameters
    ----------
    params: ~.params.QgParams or list
        An instance of model's parameters object or a list in the form [aspect_ratio, ground_basis, atmospheric_basis].
        If a list is provided, `aspect_ratio` is the aspect ratio of the domain, `ground_basis` is a SymbolicBasis object
        with the modes of the ground, and `atmospheric_basis` is either a SymbolicBasis with the modes of the atmosphere
        or `None` if there is no atmosphere.
    stored: bool, optional
        Indicate if the inner product must be stored or computed on the fly. Default to `True`
    inner_product_definition: None or InnerProductDefinition, optional
        The definition of the inner product being used. If `None`, use the canonical StandardInnerProductDefinition object.
        Default to `None`.
    interaction_inner_product_definition: None or InnerProductDefinition, optional
        The definition of the inner product being used for the interaction with the other components, i.e. to compute the inner products with the other component base of funcitons.
        If `None`, use the `inner_product_definition` provided.
        Default to `None`.
    num_threads: int or None, optional
        Number of threads to use to compute the symbolic inner products. If `None` use all the cpus available.
        Default to `None`.

    Attributes
    ----------
    n: float
        The aspect ratio of the domain.
    ground_basis: SymbolicBasis
        Object holding the symbolic modes of the ground.
    atmospheric_basis: None or SymbolicBasis
        Object holding the symbolic modes of the atmosphere (or `None` if there is no atmosphere).
    connected_to_atmosphere: bool
        Indicate if the ground is connected to an atmosphere.
    stored: bool
        Indicate if the inner product must be stored or computed on the fly.
    ip: InnerProductDefinition
        Object defining the inner product.
    iip: InnerProductDefinition
        Object defining the interaction inner product.
    subs: list(tuple)
        List of 2-tuples containing the substitutions to be made with the functions after the inner products
        symbolic computation.
    """
    def __init__(self, params=None, stored=True, inner_product_definition=None, interaction_inner_product_definition=None, num_threads=None):

        GroundInnerProducts.__init__(self)

        if params is not None:
            if isinstance(params, QgParams):
                self.n = params.scale_params.n
                self.ground_basis = params.ground_basis
                atm_basis = params.atmospheric_basis
            else:
                self.n = params[0]
                self.ground_basis = params[1]
                atm_basis = params[2]
        else:
            self.n = None
            self.ground_basis = None
            atm_basis = None
            stored = False

        self.atmospheric_basis = None
        self.connected_to_atmosphere = False

        self.subs = [('n', self.n)]

        if inner_product_definition is None:
            self.ip = StandardSymbolicInnerProductDefinition()
        else:
            self.ip = inner_product_definition

        if interaction_inner_product_definition is None:
            self.iip = self.ip
        else:
            self.iip = interaction_inner_product_definition
        self.stored = stored

        if stored:
            self.compute_inner_products(num_threads)

        if atm_basis is not None:
            self.connect_to_atmosphere(atm_basis)

    def _F(self, i):
        if self.atmospheric_basis is not None:
            return self.atmospheric_basis[i]

    def _phi(self, i):
        if self.ground_basis is not None:
            return self.ground_basis[i]

    def connect_to_atmosphere(self, atmosphere_basis, num_threads=None):
        """Connect the ocean to an atmosphere.

        Parameters
        ----------
        atmosphere_basis: SymbolicBasis or AtmosphericSymbolicInnerProducts
            Basis of function of the atmosphere or a symbolic atmospheric inner products object containing the basis.
        num_threads: int or None, optional
            Number of threads to use to compute the symbolic inner products. If `None` use all the cpus available.
            Default to `None`.
        """
        if isinstance(atmosphere_basis, AtmosphericSymbolicInnerProducts):
            atmosphere_basis = atmosphere_basis.atmospheric_basis
        self.atmospheric_basis = atmosphere_basis
        self.connected_to_atmosphere = True

        if self.stored:
            if num_threads is None:
                pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
            else:
                pool = multiprocessing.Pool(processes=num_threads)

            natm = len(atmosphere_basis)
            self._W = sp.zeros((self.ngr, natm), dtype=float, format='dok')

            # W inner products
            l = [[(i, j), self.iip.symbolic_inner_product, (self._phi(i), self._F(j))] for i in range(self.ngr)
                 for j in range(natm)]

            result = pool.map(_apply, l)

            for res in result:
                self._W[res[0]] = float(res[1].subs(self.subs).subs(self.ground_basis.substitutions)
                                        .subs(self.atmospheric_basis.substitutions))

            self._W = self._W.to_coo()

            pool.terminate()

    def compute_inner_products(self, num_threads=None):
        """Function computing and storing all the inner products at once.

        Parameters
        ----------
        num_threads: int or None, optional
            Number of threads to use to compute the symbolic inner products. If `None` use all the cpus available.
            Default to `None`.
        """

        self._U = sp.zeros((self.ngr, self.ngr), dtype=float, format='dok')

        if num_threads is None:
            pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
        else:
            pool = multiprocessing.Pool(processes=num_threads)

        # U inner products
        l = [[(i, j), self.ip.symbolic_inner_product, (self._phi(i), self._phi(j))] for i in range(self.ngr) for j in range(self.ngr)]
        result = pool.map(_apply, l)

        for res in result:
            self._U[res[0]] = float(res[1].subs(self.subs).subs(self.ground_basis.substitutions))

        pool.terminate()

    @property
    def ngr(self):
        """Number of ground modes."""
        return len(self.ground_basis)

    # !-----------------------------------------------------!
    # ! Inner products in the equations for the ocean       !
    # !-----------------------------------------------------!

    def K(self, i, j):
        """Function to commpute the forcing of the ocean by the atmosphere: :math:`K_{i,j} = (\phi_i, \\nabla^2 F_j)`.

        Warnings
        --------
        Not defined and not used."""
        return 0

    def M(self, i, j):
        """Function to compute the forcing of the ocean fields on the ocean: :math:`M_{i,j} = (\phi_i, \\nabla^2 \phi_j)`.

        Warnings
        --------
        Not defined and not used."""
        return 0

    def U(self, i, j):
        """Function to compute the inner products: :math:`U_{i,j} = (\phi_i, \phi_j)`."""
        if not self.stored:
            res = self.ip.symbolic_inner_product(self._phi(i), self._phi(j))
            return float(res.subs(self.subs).subs(self.ground_basis.substitutions))
        else:
            return self._U[i, j]

    def N(self, i, j):
        """Function computing the beta term for the ocean: :math:`N_{i,j} = (\phi_i, \partial_x \phi_j)`.

        Warnings
        --------
        Not defined and not used."""
        return 0

    def O(self, i, j, k):
        """Function to compute the temperature advection term (passive scalar): :math:`O_{i,j,k} = (\phi_i, J(\phi_j, \phi_k))`

        Warnings
        --------
        Not defined and not used."""
        return 0

    def C(self, i, j, k):
        """Function to compute the tensors holding the Jacobian inner products: :math:`C_{i,j,k} = (\phi_i, J(\phi_j,\\nabla^2 \phi_k))`.

        Warnings
        --------
        Not defined and not used."""
        return 0

    def W(self, i, j):
        """Function to compute the short-wave radiative forcing of the ocean: :math:`W_{i,j} = (\phi_i, F_j)`."""
        if self.connected_to_atmosphere:
            if not self.stored:
                res = self.iip.symbolic_inner_product(self._phi(i), self._F(j))
                return float(res.subs(self.subs).subs(self.atmospheric_basis.substitutions).subs(self.ground_basis.substitutions))
            else:
                return self._W[i, j]
        else:
            return 0


def _apply(ls):
    return ls[0], ls[1](*ls[2])


if __name__ == '__main__':
    from qgs.params.params import QgParams
    pars = QgParams()
    pars.set_atmospheric_channel_fourier_modes(2, 2, mode='symbolic')
    pars.set_oceanic_basin_fourier_modes(2, 4, mode='symbolic')
    aip = AtmosphericSymbolicInnerProducts(pars)
    oip = OceanicSymbolicInnerProducts(pars)
