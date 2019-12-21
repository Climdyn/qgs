
import numpy as np
from numba import njit
from scipy.integrate import cumtrapz


@njit
def _norm(x):
    return np.sqrt(np.sum(x*x))


@njit
def ddiff(f, x, d, h):
    dd = d/_norm(d)
    return (f(x+dd*h) - f(x))/h


@njit
def ddiv(f, x, d, h):
    dd = d/_norm(d)
    return np.sum((f(x+dd*h) - f(x))*dd)/h


@njit
def projected_div(f, x, basis, h):
    m = len(x)
    d = 0.

    for i in range(m):
        d += ddiv(f, x, basis[:, i], h)

    return d


@njit
def normalize_matrix_columns(a):
    an = np.zeros_like(a)
    norm = np.zeros(a.shape[0])
    for i in range(a.shape[1]):
        norm[i] = np.linalg.norm(a[:, i], 2)
        an[:, i] = a[:, i] / norm[i]
    return an, norm


@njit
def solve_triangular_matrix(a, b):
    x = np.zeros_like(a)
    for i in range(2, a.shape[0]+1):
        x[:i, i - 1] = np.linalg.solve(a[:i, :i], b[:i, i - 1])
    x[0, 0] = b[0, 0] / a[0, 0]
    return x


@njit
def gram_schmidt_columns(X):
    Q, R = np.linalg.qr(X)
    return Q


@njit
def reverse(a):
    out = np.zeros_like(a)
    ii = 0
    for i in range(len(a)-1,-1,-1):
        out[ii] = a[i]
        ii +=1
    return out


def time_average(y, x, axis=-1):
    ta = np.trapz(y, x, axis=axis) / (x[-1] - x[0])
    return ta


def cum_time_average(y, x, initial=0., axis=-1):
    cta = cumtrapz(y, x, axis=axis, initial=initial) / x
    return cta