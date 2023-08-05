import numba
import numpy as np
from numpy.linalg import inv as inv


@numba.jit
def k_exp_nb(x1, x2, l=1., var_ii=0., nu=2):
    K = np.empty(shape=(len(x1), len(x2)))
    for i in range(len(x1)):
        for j in range(len(x2)):
            norm = np.linalg.norm(x1[i] - x2[j])
            K[i, j] = np.exp(-norm ** nu / (2 * l ** 2)) * (1 - var_ii)
            if norm == 0:
                K[i, j] += var_ii

    return K


def k_exp(x1, x2, l=1., var_ii=0., nu=2):
    x1 = np.atleast_2d(x1)
    x2 = np.atleast_2d(x2)
    return k_exp_nb(x1, x2, l=l, var_ii=var_ii, nu=nu)


def k_exp_shape(dx, l=1., var_ii=0, nu=2):
    k = np.exp(-dx ** nu / (2 * l ** 2)) * (1 - var_ii)
    k = np.where(dx == 0, k + var_ii, k)
    return k


def arcsine_kernel(xx):
        return np.arcsin(xx) / (np.pi / 2)

def arccosine_kernel(xx):
    theta = np.arccos(xx)
    return np.sin(theta) + (np.pi - theta) * np.cos(theta)

def linear_kernel(xx):
    return xx