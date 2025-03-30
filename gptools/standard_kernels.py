import numba
import numpy as np
import jax.numpy as jnp
from jax.numpy.linalg import inv as inv


@numba.jit
def k_exp_nb(x1, x2, l=1., var_ii=0., nu=2):
    K = jnp.empty(shape=(len(x1), len(x2)))
    for i in range(len(x1)):
        for j in range(len(x2)):
            norm = jnp.linalg.norm(x1[i] - x2[j])
            K[i, j] = jnp.exp(-norm ** nu / (2 * l ** 2)) * (1 - var_ii)
            if norm == 0:
                K[i, j] += var_ii

    return K


def k_exp(x1, x2, l=1., var_ii=0., nu=2):
    x1 = jnp.atleast_2d(x1)
    x2 = jnp.atleast_2d(x2)
    return k_exp_nb(x1, x2, l=l, var_ii=var_ii, nu=nu)


def k_exp_shape(dx, l=1., var_ii=0, nu=2):
    k = jnp.exp(-dx ** nu / (2 * l ** 2)) * (1 - var_ii)
    k = jnp.where(dx == 0, k + var_ii, k)
    return k


def arcsine_kernel(xx, g=1.):
    xx = jnp.clip(xx, -1, 1)
    return jnp.sign(xx)*jnp.abs(jnp.arcsin(xx) / (jnp.pi / 2)) ** g

def arccosine_kernel(xx):
    theta = jnp.arccos(xx)
    return jnp.sin(theta) + (jnp.pi - theta) * jnp.cos(theta)

def linear_kernel(xx):
    return xx

def affine_kernel(xx, a=1., drop_1=None, f_compress=1., eps=1e-2):
    xx = jnp.clip(xx, -1, 1)
    if drop_1 is None:
        drop_1 = (1. - a) * f_compress
    k = jnp.where((xx > 1. - eps) & (xx <= 1.), 1.-drop_1, a*xx,)
    return k

from functools import partial
@partial(jnp.vectorize, excluded=[1, 2, "g", "g_V"])  # important to have working kwargs!
def erf_kernel(xx, g=1., g_V=1.):
    from bayesianchaos.src.network_theory import c_erf
    """
    Result of 

    <T(V * x1 + J * xi1)T(V * x2 + J * xi2)>_J, V

    where V ~ N(0, g_V^2/N) and J ~ N(0, g^2/N).

    The analytical result is 

    k(x1x2; g, g_V) = <T(h1)T(h2)>_h1,h2
    h1h1 = g_V^2 + g^2
    h2h2 = g_V^2 + g^2
    h1h2 = x1x2*g_V^2
    """

    h1h2 = xx*g_V**2
    h1h1 = g_V**2 + g**2
    h2h2 = g_V**2 + g**2

    return c_erf(h1h2, h1h1, h2h2, 0, 0)



