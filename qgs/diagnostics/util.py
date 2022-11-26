"""
    Utility functions module
    ========================

    This module has some useful functions for the diagnostics.

"""

import numpy as np


def create_grid_basis(basis, X, Y, extra_subs=None):
    """Create an array from a basis with the basis functions evaluated on a grid.


    Parameters
    ----------
    basis: SymbolicBasis
        Symbolic basis to compute gridded representation from.
    X: ~numpy.ndarray
        The zonal x-coordinates of the grid points.
    Y: ~numpy.ndarray
        The meridional y-coordinates of the grid points.
    extra_subs: list(tuple), optional
        List of 2-tuples containing extra substitutions to be made with the functions providing
        the grib basis, before transforming them into python callable.
        The 2-tuples contain first a `Sympy`_  expression and then the value to substitute.

    Returns
    -------
    ~numpy.ndarray
        The gridded representation of the basis functions packed along the 0-th axis.

    """

    grid_basis = list()

    for func in basis.num_functions(extra_subs):
        grid = func(X, Y)
        if isinstance(grid, (int, float)):
            grid = np.ones_like(X) * grid
        grid_basis.append(grid)

    return np.array(grid_basis)
