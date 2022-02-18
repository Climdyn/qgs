"""
    Utility functions module
    ========================

    This module has some useful functions for the diagnostics.

"""

import numpy as np


def create_grid_basis(basis, X, Y):
    """Create an array from a basis with the basis functions evaluated on a grid.


    Parameters
    ----------
    basis: SymbolicBasis
        Symbolic basis to compute gridded representation from.
    X: ~numpy.ndarray
        The zonal x-coordinates of the grid points.
    Y: ~numpy.ndarray
        The meridional y-coordinates of the grid points.

    Returns
    -------
    ~numpy.ndarray
        The gridded representation of the basis functions packed along the 0-th axis.

    """

    grid_basis = list()

    for func in basis.num_functions():
        grid = func(X, Y)
        if isinstance(grid, (int, float)):
            grid = np.ones_like(X) * grid
        grid_basis.append(grid)

    return np.array(grid_basis)
