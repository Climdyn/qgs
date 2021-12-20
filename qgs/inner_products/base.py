"""
    Inner products module (base class)
    ==================================

    Abstract base classes of the structure containing the inner products :math:`(\\cdot, \\, \\cdot)`  between the truncated set of basis functions :math:`\phi_i` for the ocean and
    land fields, and :math:`F_i` for the atmosphere fields (see :ref:`files/model/oro_model:Projecting the equations on a set of basis functions`).

    Description of the classes
    --------------------------

    The three classes computing and holding the inner products of the basis functions are:

    * :class:`AtmosphericInnerProducts`
    * :class:`OceanicInnerProducts`
    * :class:`GroundInnerProducts`

    Warnings
    --------

    These are `abstract base class`_, they must be subclassed and the functions to compute the inner products must be defined!

    .. _abstract base class: https://docs.python.org/3/glossary.html#term-abstract-base-class

"""

from abc import ABC, abstractmethod
import pickle


class AtmosphericInnerProducts(ABC):
    """Base class which contains all the atmospheric inner products coefficients needed for the tendencies
    tensor :class:`~.tensors.qgtensor.QgsTensor` computation.
    """

    def __init__(self):
        self._a = None
        self._u = None
        self._c = None
        self._b = None
        self._g = None
        self._d = None
        self._s = None
        self._z = None
        self._v = None
        self.stored = False

    # !-----------------------------------------------------!
    # ! Inner products in the equations for the atmosphere  !
    # !-----------------------------------------------------!
    @property
    @abstractmethod
    def natm(self):
        """Number of atmospheric modes."""
        pass

    @abstractmethod
    def a(self, i, j):
        """Function to compute the matrix of the eigenvalues of the Laplacian (atmospheric): :math:`a_{i, j} = (F_i, {\\nabla}^2 F_j)`."""
        pass

    @abstractmethod
    def u(self, i, j):
        """Function to compute the matrix of inner product: :math:`u_{i, j} = (F_i, F_j)`."""
        pass

    @abstractmethod
    def b(self, i, j, k):
        """Function to compute the tensors holding the Jacobian inner products: :math:`b_{i, j, k} = (F_i, J(F_j, \\nabla^2 F_k))`."""
        pass

    @abstractmethod
    def c(self, i, j):
        """Function to compute the matrix of beta terms for the atmosphere: :math:`c_{i,j} = (F_i, \\partial_x F_j)`."""
        pass

    @abstractmethod
    def g(self, i, j, k):
        """Function to compute tensors holding the Jacobian inner products: :math:`g_{i,j,k} = (F_i, J(F_j, F_k))`."""
        pass

    @abstractmethod
    def s(self, i, j):
        """Function to compute the forcing (thermal) of the ocean on the atmosphere: :math:`s_{i,j} = (F_i, \\phi_j)`."""
        pass

    @abstractmethod
    def d(self, i, j):
        """Function to compute the forcing of the ocean on the atmosphere: :math:`d_{i,j} = (F_i, \\nabla^2 \\phi_j)`."""
        pass

    @abstractmethod
    def z(self, i, j, k, l, m):
        """Function to compute the :math:`T^4` temperature forcing for the radiation lost by atmosphere to space & ground/ocean: :math:`z_{i,j,k,l,m} = (F_i, F_j F_k F_l F_m)`."""
        pass

    @abstractmethod
    def v(self, i, j, k, l, m):
        """Function to compute the :math:`T^4` temperature forcing of the ocean on the atmosphere: :math:`v_{i,j,k,l,m} = (F_i, \\phi_j \\phi_k \\phi_l \\phi_m)`."""
        pass

    def save_to_file(self, filename, **kwargs):
        """Function to save the inner object to a file with the :mod:`pickle` module.

        Parameters
        ----------
        filename: str
            The file name where to save the inner product object.
        kwargs: dict
            Keyword arguments to pass to the :mod:`pickle` module method.
        """
        f = open(filename, 'wb')
        pickle.dump(self.__dict__, f, **kwargs)
        f.close()

    def load_from_file(self, filename, **kwargs):
        """Function to load previously saved inner product object with the method :meth:`save_to_file`.

        Parameters
        ----------
        filename: str
            The file name where the inner product object was saved.
        kwargs: dict
            Keyword arguments to pass to the :mod:`pickle` module method.
        """
        f = open(filename, 'rb')
        tmp_dict = pickle.load(f, **kwargs)
        f.close()

        self.__dict__.clear()
        self.__dict__.update(tmp_dict)


class OceanicInnerProducts(ABC):
    """Base class which contains all the oceanic inner products coefficients needed for the tendencies
    tensor :class:`~.tensors.qgtensor.QgsTensor` computation.
    """

    def __init__(self):
        self._M = None
        self._U = None
        self._N = None
        self._O = None
        self._C = None
        self._K = None
        self._W = None
        self._V = None
        self._Z = None

        self.stored = False

    # !-----------------------------------------------------!
    # ! Inner products in the equations for the ocean       !
    # !-----------------------------------------------------!

    @property
    @abstractmethod
    def noc(self):
        """Number of oceanic modes."""
        pass

    @abstractmethod
    def M(self, i, j):
        """Function to compute the forcing of the ocean fields on the ocean: :math:`M_{i,j} = (\\phi_i, \\nabla^2 \\phi_j)`."""
        pass

    @abstractmethod
    def U(self, i, j):
        """Function to compute the inner products: :math:`U_{i,j} = (\\phi_i, \\phi_j)`."""
        pass

    @abstractmethod
    def N(self, i, j):
        """Function computing the beta term for the ocean: :math:`N_{i,j} = (\\phi_i, \\partial_x \\phi_j)`."""
        pass

    @abstractmethod
    def O(self, i, j, k):
        """Function to compute the temperature advection term (passive scalar): :math:`O_{i,j,k} = (\\phi_i, J(\\phi_j, \\phi_k))`"""
        pass

    @abstractmethod
    def C(self, i, j, k):
        """Function to compute the tensors holding the Jacobian inner products: :math:`C_{i,j,k} = (\\phi_i, J(\\phi_j,\\nabla^2 \\phi_k))`."""
        pass

    @abstractmethod
    def K(self, i, j):
        """Function to compute the forcing of the ocean by the atmosphere: :math:`K_{i,j} = (\\phi_i, \\nabla^2 F_j)`."""
        pass

    @abstractmethod
    def W(self, i, j):
        """Function to compute the short-wave radiative forcing of the ocean: :math:`W_{i,j} = (\\phi_i, F_j)`."""
        pass

    @abstractmethod
    def Z(self, i, j, k, l, m):
        """Function to compute the :math:`T^4` temperature forcing from the atmosphere to the ocean: :math:`Z_{i,j,k,l,m} = (\\phi_i, F_j, F_k, F_l, F_m)`."""
        pass

    @abstractmethod
    def V(self, i, j, k, l, m):
        """Function to compute the :math:`T^4` temperature forcing from the ocean to the atmosphere: :math:`V_{i,j,k,l,m} = (\\phi_i, \\phi_j, \\phi_k, \\phi_l, \\phi_m)`."""
        pass

    def save_to_file(self, filename, **kwargs):
        """Function to save the inner product object to a file with the :mod:`pickle` module.

        Parameters
        ----------
        filename: str
            The file name where to save the inner product object.
        kwargs: dict
            Keyword arguments to pass to the :mod:`pickle` module method.
        """
        f = open(filename, 'wb')
        pickle.dump(self.__dict__, f, **kwargs)
        f.close()

    def load_from_file(self, filename, **kwargs):
        """Function to load previously saved inner product object with the method :meth:`save_to_file`.

        Parameters
        ----------
        filename: str
            The file name where the inner product object was saved.
        kwargs: dict
            Keyword arguments to pass to the :mod:`pickle` module method.
        """
        f = open(filename, 'rb')
        tmp_dict = pickle.load(f, **kwargs)
        f.close()

        self.__dict__.clear()
        self.__dict__.update(tmp_dict)


class GroundInnerProducts(ABC):
    """Base class which contains all the ground inner products coefficients needed for the tendencies
    tensor :class:`~.tensors.qgtensor.QgsTensor` computation.
    """

    def __init__(self):
        self._M = None
        self._U = None
        self._N = None
        self._O = None
        self._C = None
        self._K = None
        self._W = None
        self._Z = None
        self._V = None

        self.stored = False

    # !-----------------------------------------------------!
    # ! Inner products in the equations for the ocean       !
    # !-----------------------------------------------------!

    @property
    @abstractmethod
    def ngr(self):
        """Number of ground modes."""
        pass

    @abstractmethod
    def K(self, i, j):
        """Function to compute the forcing of the ocean by the atmosphere: :math:`K_{i,j} = (\\phi_i, \\nabla^2 F_j)`."""
        pass

    @abstractmethod
    def M(self, i, j):
        """Function to compute the forcing of the ocean fields on the ocean: :math:`M_{i,j} = (\\phi_i, \\nabla^2 \\phi_j)`."""
        pass

    @abstractmethod
    def U(self, i, j):
        """Function to compute the inner products: :math:`U_{i,j} = (\\phi_i, \\phi_j)`."""
        pass

    @abstractmethod
    def N(self, i, j):
        """Function computing the beta term for the ocean: :math:`N_{i,j} = (\\phi_i, \\partial_x \\phi_j)`."""
        pass

    @abstractmethod
    def O(self, i, j, k):
        """Function to compute the temperature advection term (passive scalar): :math:`O_{i,j,k} = (\\phi_i, J(\\phi_j, \\phi_k))`"""
        pass

    @abstractmethod
    def C(self, i, j, k):
        """Function to compute the tensors holding the Jacobian inner products: :math:`C_{i,j,k} = (\\phi_i, J(\\phi_j,\\nabla^2 \\phi_k))`."""
        pass

    @abstractmethod
    def W(self, i, j):
        """Function to compute the short-wave radiative forcing of the ocean: :math:`W_{i,j} = (\\phi_i, F_j)`."""
        pass

    @abstractmethod
    def V(self, i, j, k, l, m):
        pass

    @abstractmethod
    def Z(self, i, j, k, l, m):
        pass

    def save_to_file(self, filename, **kwargs):
        """Function to save the inner object to a file with the :mod:`pickle` module.

        Parameters
        ----------
        filename: str
            The file name where to save the inner product object.
        kwargs: dict
            Keyword arguments to pass to the :mod:`pickle` module method.
        """
        f = open(filename, 'wb')
        pickle.dump(self.__dict__, f, **kwargs)
        f.close()

    def load_from_file(self, filename, **kwargs):
        """Function to load previously saved inner product object with the method :meth:`save_to_file`.

        Parameters
        ----------
        filename: str
            The file name where the inner product object was saved.
        kwargs: dict
            Keyword arguments to pass to the :mod:`pickle` module method.
        """
        f = open(filename, 'rb')
        tmp_dict = pickle.load(f, **kwargs)
        f.close()

        self.__dict__.clear()
        self.__dict__.update(tmp_dict)
