
"""

    Sparse tensor module
    ====================

    This module includes the classes and functions to define the model's multi-dimensional sparse tensor
    in coordinates-values form.

"""

import numpy as np

real_eps = np.finfo(np.float64).eps


# TODO: Should be rewritten as a numba jit class so that it can be passed directly to the sparse_mul's
class CooTensor(object):
    """The base class for a multi-dimensional sparse tensor :math:`\mathcal{T}` in coordinates-values form.

    Parameters
    ----------
    ncoo: int, optional
        The dimension of the tensor (number of coordinates).
        Default is 3, which result in a tensor :math:`\mathcal{T}_{i,j,k}`.

    Attributes
    ----------
    coo: ~numpy.ndarray(int)
        The list of coordinates, as a 2D array of shape (:attr:`n_elems`, :attr:`ncoo`).
    value: ~numpy.ndarray
        The list of values, as 1D array of shape (:attr:`n_elems`,).

    Notes
    -----
    Can be iterated over, returning tuples of coordinates tuple and values.

    """

    def __init__(self, ncoo=3, dtype=None):

        self.coo = np.zeros((1, ncoo), dtype=np.int32)

        if dtype is None:
            dtype = np.float64
        self.value = np.zeros(1, dtype=dtype)

    def add_entries(self, coo, value):
        """Add entries to the tensor.

        Parameters
        ----------
        coo: ~numpy.ndarray(int)
            The list of coordinates, as a 2D array.
        value: ~numpy.ndarray
            The list of values to add, as 1D array.
        """

        self.coo = np.vstack((self.coo, coo))
        self.value = np.concatenate((self.value, value))
        self.clean_entries()

    def delete_entries(self, indices):
        """Delete entries of the tensor.

        Parameters
        ----------
        indices: slice, int, list(int) or ~numpy.ndarray(int)
             Indicate indices of tensors entries to remove.
        """
        self.coo = np.delete(self.coo, indices, axis=0)
        self.value = np.delete(self.value, indices, axis=0)

    def clean_entries(self, exact=False):
        """Remove the null entries of the tensor.

        Parameters
        ----------
        exact: bool
            Remove the value that exactly equal to zero, or smaller than the machine precision.
            Default is machine precision.
        """

        if exact:
            indices_to_delete = np.where(self.value == 0)
        else:
            indices_to_delete = np.where(np.abs(self.value) < real_eps)

        self.delete_entries(indices_to_delete[0])

    def _def_entries(self, coo_list, value_list):

        self.coo = np.array(coo_list)
        self.value = np.array(value_list)

    def __getitem__(self, index):

        return self.coo[index], self.value[index]

    def __len__(self):
        return len(self.value)

    @property
    def ncoo(self):
        """int: The dimension of the tensor (number of coordinates)."""
        return self.coo.shape[1]

    @property
    def n_elems(self):
        """int: The number of non-zero tensor entries (values)."""
        return self.__len__()


def from_coo_mat_list(coo_mat_list):
    """Transform a list of :class:`~scipy.sparse.coo_matrix` into a 3D :class:`CooTensor` tensor object.

    Parameters
    ----------
    coo_mat_list: list(~scipy.sparse.coo_matrix)
        The list of :mod:`scipy.sparse` matrices to convert.

    Returns
    -------
    CooTensor
        The 3D tensor object.
    """

    out = CooTensor()

    for i, mat in enumerate(coo_mat_list):
        row = mat.row
        col = mat.col
        l = len(row)

        value = mat.data

        coo = np.hstack((np.full((l, 1), i, dtype=np.int32), row[..., np.newaxis], col[..., np.newaxis]))

        out.add_entries(coo, value)

    # out.delete_entries(0)

    return out


def from_csr_mat_list(csr_mat_list):
    """Transform a list of :class:`~scipy.sparse.csr_matrix` into a 3D :class:`CooTensor` tensor object.

    Parameters
    ----------
    csr_mat_list: list(~scipy.sparse.csr_matrix)
        The list of :mod:`scipy.sparse` matrices to convert.

    Returns
    -------
    CooTensor
        The 3D tensor object.


    """

    coo_mat_list = list()

    for mat in csr_mat_list:
        coo_mat_list.append(mat.tocoo())

    return from_coo_mat_list(coo_mat_list)
