
# Gaussfrom pathlib import Path
from joblib import Memory
from matplotlib import pyplot as plt
import numpy as np
from numpy.polynomial import Polynomial
from scipy import integrate
from scipy.interpolate import approximate_taylor_polynomial
from scipy.optimize import curve_fit
from scipy.special import gamma, factorial, gegenbauer
from scipy.special import gamma, binom
import logging

logger = logging.getLogger(__name__)

# set verbosity to warning
logger.setLevel(logging.WARNING)


def shell_measure(d_emb):
    assert d_emb >= 3
    d = d_emb
    return 2*np.pi**(d/2)/gamma(d/2)

def get_uniform_measure(d):
    # get the xx measure for uniformly distributed vectors on the d-1 sphere
    # https://djalil.chafai.net/blog/2021/05/22/the-funk-hecke-formula/
    Z_inv = np.pi**-0.5 * frac_gamma(x=d/2, da=0, db=-1/2)
    return lambda t: Z_inv*(1-t**2)**((d-3)/2)


def get_Y_from_spectrum(var_lmbds, d, seed):
    from scipy.special import sph_harm
    rng_GP = np.random.default_rng(seed+1)

    Omega = shell_measure(d_emb=d)
    # get the index order right, and use the convention where theta is the polar angle and phi is the azimuthal angle
    Ylm = lambda l, m, theta, phi: sph_harm(m,l, phi, theta) * Omega**0.5

    def Ylm_real(l, m, phi, theta):
        """
        https://en.wikipedia.org/wiki/Table_of_spherical_harmonics#Real_spherical_harmonics
        """
        
        if m == 0:
            return Ylm(l, m, theta, phi).real
        elif m > 0:
            return 2**0.5*(-1)**m*Ylm(l, m, theta, phi).real
        elif m < 0:
            return 2**0.5*(-1)**m*Ylm(l, abs(m), theta, phi).imag
        
    pref = lambda l, m: ((2*l+1) / (4*np.pi) * (factorial(l-m) / factorial(l+m)))**0.5
    
    def Y(PHI, THETA):
        Y_ = Ylm_real(0, 0, PHI, THETA)*0
        
        for l in range(0, len(var_lmbds)):
            sqrtlmbds = rng_GP.normal(0, var_lmbds[l]**0.5, size=(2*l+1,))
            sqrtlmbds_d = {m: sqrtlmbd for sqrtlmbd, m in zip(sqrtlmbds, range(-l, l+1))}

            for im, m in enumerate(range(-l, l+1)):
                # correlated weights enable contructive interference between modes
                sqrtlmbd = sqrtlmbds_d[m]  # whether or not use the same lmbd at a given frequency. This will make the sample smoother. 
                # scipy convention flips theta and phi compared to standard spherical coordinates
                #* frac_fac**-0.5
                Y_ += sqrtlmbd*Ylm_real(l, m, PHI, THETA)

        return Y_
    
    return Y

def frac_gamma(x, da, db):
    """
    Prevents overflow if dimension x=d/2 is very large
    """
    if x < 50:
        return gamma(x + da) / gamma(x + db)
    else:
        # use stirling, see https://math.stackexchange.com/questions/117977/quotient-of-gamma-functions
        return x ** (da - db)

def get_lambda_taylor(bn, d):
    lmd = []
    for l in range(len(bn)):
        # db += 1/2 is correction from Azevedo 2015, also pi ** ((d-1)/2) and 2**(l+1)
        fac = factorial
        f1 = 1 / (np.pi**((d-1)/2)  * 2 ** (l+1))
        f2 = 0

        for s in range(np.floor((len(bn) - l) / 2).astype(int)):

            f2 += bn[2 * s + l] * fac(2 * s + l) / fac(2 * s) * gamma(s + 1 / 2) * frac_gamma(x=d/2, da=0, db=s+l+1/2)
        lmbd_l = f1 * f2
        lmd += [lmbd_l]

    return np.array(lmd)


def get_bn_fn(fn, deg, x0=0):
    bn_poly = approximate_taylor_polynomial(fn, x0, deg, scale=1)
    return bn_poly

code = """
def f(x, {vars}):
    coeff = [{vars}]
    r = 0
    for d, c in enumerate(coeff):
        r += c*(x-{x0})**d

    return r
"""


def get_bn_disc(x,y,deg, x0=0, omit_0=True):
    """
    Fits a function of the form 
    c*(x-{x0})**d
    to the data, where d can be non-integer (i.e. roots).
    """
    # def a function
    vars = ", ".join([f"a{i}" for i in range(deg+1)])
    code_ = code.replace("{vars}", vars).replace("{x0}", str(x0))
    l = dict()
    g = dict()
    exec(code_, g, l)
    f = list(l.values())[0]
    if omit_0:
        x = x[y!=1]
        y = y[y != 1]
    p_opt, p_err = curve_fit(f, x, y, bounds=(0,np.inf))
    assert (p_opt >= 0).all()
    bn_poly = Polynomial(p_opt)
    # coef = np.polynomial.polynomial.polyfit(x, y, deg, full=False)
    # bn_poly = Polynomial(coef)
    return bn_poly


def reconstruct_from_spec(lmbds, d):
    w_d = np.pi**-0.5 * frac_gamma(x=d/2, da=0, db=-1/2)
    alpha = (d-2)/2
    ggb = lambda l: gegenbauer(alpha=alpha, n=l)
    C_alpha_l = ggb(1)
    mu = lambda xx: (1-xx**2)**((d-3)/2)

    def f(xx):
        # because the ggb polynomials are orthonormal wrt (1-t^2)^(alpha-1/2), we can omit this factor in the reconstruction
        return np.sum([lmbds[l]*ggb(l)(xx) for l in range(len(lmbds))], axis=0) / len(lmbds)
    
    return f



# get the dimension of the Hilbert space in dimension d and at mode l
def z_dl(d, l): 
    """
    Multiplicity of the spherical harmonic of degree l in dimension d
    """
    z_canatar = np.round(binom(d+l - 1 , l) - binom(d+l-3, l-2)).astype(int)
    z_canatar_apx = np.round(d**l/factorial(l)).astype(int)

    # incorrect
    z_dutordoir = np.round((2*l+d-2) * (gamma(l+d-2)) / (gamma(l+1)*gamma(d-2))).astype(int)
    z_wiki = np.round(binom(d+l - 1 , d-1) - binom(d+l-3, d-1)).astype(int)

    assert z_canatar == z_wiki
    # assert z_canatar == z_wiki

    if d >= 10:
        logger.warning(f"Large degeneracy at dimension {d}.") 

    return z_canatar


def get_lambda_int(l, d, shape_fn=None, p=None, d_opts=None, repeat_degenerate=False, error_level="warn"):
    """
    see Dutordoir2020, Funk-Hecke
    """


    if p is None:
        logger.info("No measure p given, using uniform measure on the sphere.")
        p = get_uniform_measure(d)


    assert (d_opts is not None) or (shape_fn is not None and p is not None), "Either operate in data or kernel mode. "
    assert not d < 3, "Spherical harmonics undefined for d<3"

    def get_lambda_int_(l, d):
        """
        Note: This follows the prescription at 
        https://djalil.chafai.net/blog/2021/05/22/the-funk-hecke-formula/

        Another working account is Canatar21. 
        """
        alpha = (d-2)/2
        ggb = gegenbauer(alpha=alpha, n=l)
        C_alpha_l = ggb(1)
        if shape_fn:
            integrand = lambda cth: C_alpha_l**-1 * shape_fn(cth) * ggb(cth) * p(cth)
            # custom integration routine
            integral = funk_hecke_integrator(integrand)
        else:
            XX_centers, YY_hist = kernel_hist(d_opts, nbins=100)
            integral = np.sum(ggb(XX_centers) * YY_hist) / len(XX_centers)

        assert not np.isnan(integral).any()
        lmbd_l = integral

        if lmbd_l < 0:
            if error_level:
                assert lmbd_l > -1e-2
            lmbd_l = 0.

        return lmbd_l, integrand

    get_lambda_int_ = np.vectorize(get_lambda_int_, excluded=["d"])
    lmbd_l, f = get_lambda_int_(l, d)

    PLOT = False
    if PLOT:
        l = np.atleast_1d(l)
        fig, axes = plt.subplots(np.ceil(len(l)**0.5).astype(int), np.ceil(len(l)**0.5).astype(int), constrained_layout=True, figsize=(20, 20))
        for iax, ax in enumerate((axes.flat)[:len(l)]):
            cths = np.linspace(-1,1,1000)
            ax.plot(cths, f[iax](cths), label=f"alk={lmbd_l[iax]:.2E}")
            ax.set_title(f"l={l[iax]}")
            ax.legend()
        fig.savefig("test_lambda_int.png", dpi=200)
    
    return lmbd_l

def funk_hecke_integrator(f):
    """
    Alternatively, a quadrature rule could be used, as detailed in 
    Canatar21, Supplementary Material, eq. (121f)
    """
    unit_int = np.geomspace(1e-4, 1, int(1e4-2))
    xx_int = np.concatenate([[-1], -1+unit_int, 1-unit_int[::-1], [1]])
    dx = np.diff(xx_int)
    f_int = f(xx_int)
    # handle divergences at boundaries
    if not np.isfinite(f_int[-1]):
        f_int[0] = f_int[1]
    if not np.isfinite(f_int[-1]):
        f_int[-1] = f_int[-2]

    f_int = (f_int[1:] + f_int[:-1])/2
    overlap = np.sum(f_int*dx)
    return overlap



def kernel_hist(d_opts, nbins=100, ax=None):
    (X, Y, x, y) = d_opts.data
    X = X / np.linalg.norm(X, axis=-1, keepdims=True)
    YY = np.einsum("kp,lp->klp", Y, Y)

    YY = YY.mean(axis=-1).flatten()
    XX = (X @ X.T).flatten()

    YY_hist, bin_edg, show, w = bin_XX_YY_with_std(XX, YY, ax, nbins)

    XX_centers = bin_edg[:-1] + w / 2
    return XX_centers, YY_hist


def bin_XX_YY_with_std(XX, YY, bins=None, separate_1=False, exclude_1=False, interpolate_nan=False, return_empty=False):
    """
    Turns flattened input and output tensors into bins and values at bins, thus effectively discretizing the data.
    """
    # assert (XX <= 1).all() and (XX >= -1).all()

    one_thsd = 1e-6
    if exclude_1:
        YY = YY[np.abs(XX - 1) > one_thsd]
        XX = XX[np.abs(XX - 1) > one_thsd]
        separate_1 = False

    if separate_1:
        _, bins = np.histogram(XX.flatten(), bins=bins)
        bins = list(bins)
        bins = bins[:-1] + [1. - one_thsd, 1., ]

    XX_hist, bin_edg = np.histogram(XX, bins=bins)
    w = bin_edg[1:] - bin_edg[:-1]
    XX_centers = (bin_edg[1:] + bin_edg[:-1]) / 2
    YY_hist_mean = np.empty_like(XX_hist, dtype=float)
    YY_hist_std = np.empty_like(XX_hist, dtype=float)

    for i in range(len(bin_edg) - 1):
        hits = YY[(XX >= bin_edg[i]) & (XX <= bin_edg[i + 1])]
        if len(hits) > 0:
            YY_hist_mean[i] = hits.mean()
            YY_hist_std[i] = hits.std()
        else:
            YY_hist_mean[i] = np.nan
            YY_hist_std[i] = np.nan

    if not return_empty:
        idx_nonempty = ~(np.isnan(YY_hist_mean) | np.isnan(YY_hist_std))
        XX_centers = XX_centers[idx_nonempty]
        YY_hist_mean = YY_hist_mean[idx_nonempty]
        YY_hist_std = YY_hist_std[idx_nonempty]

    # ax.bar(bin_edg[:-1] - 1, YY_hist, width=w, align='edge', color="gray", alpha=0.3)
    YY_hist_mean = (YY_hist_mean)
    YY_hist_std = (YY_hist_std)

    if interpolate_nan:
        nans, x_ = nan_helper(YY_hist_mean)
        YY_hist_mean[nans] = np.interp(x_(nans), x_(~nans), YY_hist_mean[~nans])
        nans, x_ = nan_helper(YY_hist_std)
        YY_hist_std[nans] = np.interp(x_(nans), x_(~nans), YY_hist_std[~nans])
    # ax.bar(-np.flip(bin_edg[1:]) + 1, YY_hist, width=np.flip(w), align='edge', edgecolor="gray", color="none",
    #        alpha=0.3, zorder=1)
    # ax.bar(-np.flip(bin_edg[1:]) + 1, height=2 * YY_hist_std, bottom=YY_hist - YY_hist_std, width=np.flip(w),
    #        align='edge', color="tab:blue", alpha=0.1, zorder=-1)

    return XX_centers, YY_hist_mean, YY_hist_std


def test_funk_hecke_integrator():
    f = lambda t: np.where(t<0, 1-np.tanh((t+1)**2), 1-np.tanh((t-1)**2))
    overlap1 = funk_hecke_integrator(f)
    from scipy.integrate import quad
    overlap2 = quad(f, -1, 1, limit=1000)
    print(overlap1)
    print(overlap2)